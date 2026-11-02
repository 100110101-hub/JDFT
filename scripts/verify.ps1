# Full gate suite - native Windows entry point.
#
#   pwsh -NoProfile -File scripts\verify.ps1          (PowerShell 7, preferred)
#   powershell -NoProfile -File scripts\verify.ps1    (Windows PowerShell 5.1)
#
# Linux / CI / WSL equivalent: scripts/verify.sh. Both scripts cover the SAME
# checklist - if you change one, change the other.
#
# NOTE: this file is intentionally ASCII-only. Windows PowerShell 5.1 decodes
# scripts as ANSI when there is no BOM, which corrupts non-ASCII text and can
# break parsing. Keep it ASCII.
#
# Exit code: 0 = all green, 1 = at least one failure.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 台账校验用的 Python：**必须**带 PyYAML。系统 python 若没有，check_status.py 会静默退化成
# 「只做结构检查」，跳过严格 YAML 解析 —— 2026-11-07 就因为这一点让一个解析不了的
# docs/status-v2.yaml 通过了门禁（142 全绿而文件是坏的）。这里先探测，缺失就直接中止。
$script:ledgerPython = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $script:ledgerPython)) { $script:ledgerPython = 'python' }
$script:yamlProbe = & $script:ledgerPython -c "import yaml" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host '!! 台账校验需要带 PyYAML 的 Python（.venv\Scripts\python.exe）' -ForegroundColor Red
    Write-Host "   probe: $script:yamlProbe"
    exit 1
}

$script:pass = 0
$script:fail = 0
$script:failedNames = @()

# EX-04：每一步的墙钟。没有这个，门禁的 19 min 就只能靠猜「时间花在哪」。
$script:stepTimes = @()

function Step {
    param(
        [string]$Name,
        [scriptblock]$Body,
        [string]$Dir = '.'
    )
    Write-Host ''
    Write-Host "== $Name =="
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location (Join-Path $root $Dir)
    $global:LASTEXITCODE = 0
    try {
        & $Body
    }
    catch {
        Write-Host "  (exception: $_)" -ForegroundColor Yellow
        $global:LASTEXITCODE = 1
    }
    finally {
        Pop-Location
    }
    $stopwatch.Stop()
    $seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
    $script:stepTimes += [pscustomobject]@{ Name = $Name; Seconds = $seconds }
    Write-Host ("   [{0,7:N1}s]" -f $seconds)
    $code = $global:LASTEXITCODE
    if ($code -ne 0) {
        $script:fail++
        $script:failedNames += "$Name (exit $code)"
        Write-Host "!! FAILED: $Name (exit $code)" -ForegroundColor Red
    }
    else {
        $script:pass++
    }
}

function To-WslPath {
    param([string]$WinPath)
    if ($WinPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    return $WinPath
}

Write-Host "QWB verify - repo $root"

# ---- gate scripts (bash; on Windows this goes through WSL) ----
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($null -eq $bash) {
    $script:fail++
    $script:failedNames += 'gate scripts: bash not found'
    Write-Host '!! FAILED: bash not found (WSL bash or Git Bash is required by the three gate scripts)' -ForegroundColor Red
}
else {
    $wsl = To-WslPath $root
    foreach ($gate in @('check_deps_centralized.sh', 'check_unsafe.sh', 'check_rayon_determinism.sh')) {
        $gatePath = "scripts/$gate"
        Step -Name "gate: $gate" -Body { bash -c "cd '$wsl' && bash $gatePath" }
    }
}

# ---- third-party data (deliberately not in the repo; see qwb-data/README.md) ----
# The basis/ECP files exported from BSE are NOT committed (redistribution terms,
# project.md 14 #20): the repo ships the recipe (qwb-data/sources.json) plus the
# sha256 pins. Every later step that opens qwb-data needs them, so fetch once here.
# Needs network access on a fresh clone; afterwards the files are local and this is
# a no-op. `qwb data install` / `python scripts/fetch_data.py --all` do the same.
Step -Name 'data: third-party data present (fetch once if missing)' -Dir '.' -Body {
    & python scripts/fetch_data.py --check
    if ($LASTEXITCODE -ne 0) {
        Write-Host '  (missing third-party data - fetching from Basis Set Exchange...)'
        & python scripts/fetch_data.py --all
    }
}

# ---- K2 contract crate ----
Step -Name 'contracts(K2): fmt' -Dir 'qwb-contracts/engine-api' -Body { cargo fmt --check }
Step -Name 'contracts(K2): clippy -D warnings' -Dir 'qwb-contracts/engine-api' -Body {
    cargo clippy --all-targets -- -D warnings
}
Step -Name 'contracts(K2): cargo doc' -Dir 'qwb-contracts/engine-api' -Body {
    $env:RUSTDOCFLAGS = '-D warnings'
    try { cargo doc --no-deps } finally { Remove-Item Env:RUSTDOCFLAGS -ErrorAction SilentlyContinue }
}
Step -Name 'contracts(K2): test' -Dir 'qwb-contracts/engine-api' -Body {
    cargo test --all-targets
}

# ---- qwb-core ----
Step -Name 'core: fmt' -Dir 'qwb-core' -Body { cargo fmt --all --check }
Step -Name 'core: clippy -D warnings' -Dir 'qwb-core' -Body {
    cargo clippy --workspace --all-targets -- -D warnings
}
# EX-02：**先编译、再计时**。`qwb-registry` 的类型一改，依赖它的每个测试二进制都要重编；
# 把编译放进计时的那一步里，会让「省了一大笔运行时间」被编译时间掩盖（第 375 轮实测：
# 冷路径 `core: test` 760.9 s，其中运行只有几十秒）。所以编译单独一步、单独报时，
# 下一步的计时才是真正的**测试段**。编译产物 nextest 直接复用（同一个 test profile）。
Step -Name 'core: test binaries (prebuild, not part of the test wall clock)' -Dir 'qwb-core' -Body {
    cargo test --workspace --all-targets --no-run
}
# EX-02：进程级并行（cargo-nextest）。**测试集合不变** —— nextest 不跑 doctest，
# 所以 doctest 单独跑一步补上；没装 nextest 的机器退回 `cargo test --workspace`（同一个集合）。
Step -Name 'core: test' -Dir 'qwb-core' -Body {
    if (-not (Get-Command cargo-nextest -ErrorAction SilentlyContinue)) {
        & cargo test --workspace
        return
    }
    # EX-02 两泳道：`--partition` 由 nextest 保证「两片恰好覆盖全集、互不重叠」，
    # 所以不会因为名单写错而静默漏测（上一版按名字过滤就出过这个问题）。
    # 两条进程各 4 线程 = 8 = 物理核数；单进程 8 线程时最长的测试常常最后才起，
    # 两个独立调度器能把这种「阶梯尾巴」摊平。
    & cargo nextest list --workspace *> nextest-list.log
    $listed = (Get-Content nextest-list.log | Where-Object { $_ -match '\S' }).Count
    $laneA = Start-Process -FilePath 'cargo' -PassThru -NoNewWindow `
        -RedirectStandardOutput slow-lane.log -RedirectStandardError slow-lane.err `
        -ArgumentList @('nextest', 'run', '--workspace', '-j', '4', '--partition', 'count:1/2', '--no-fail-fast')
    & cargo nextest run --workspace -j 4 --partition 'count:2/2' --no-fail-fast
    $codeB = $LASTEXITCODE
    $laneA.WaitForExit()
    $codeA = $laneA.ExitCode
    Write-Host ("   lanes: shard1 exit {0}, shard2 exit {1}, nextest list count {2}" -f $codeA, $codeB, $listed)
    if ($codeA -ne 0 -or $codeB -ne 0) { throw 'test lane failed' }
    & cargo test --workspace --doc
}
Step -Name 'core: I-06 integral cache (feature on)' -Dir 'qwb-core' -Body {
    # I-06 acceptance: cache on/off must give identical results.
    cargo test -p qwb-integrals --features integral-cache --test i06_cache
}
Step -Name 'core: I-00 faer backend' -Dir 'qwb-core' -Body {
    # I-00 acceptance: BOTH backends must compile and give the same numbers.
    cargo test -p qwb-linalg --no-default-features --features faer
}
Step -Name 'core: cargo deny' -Dir 'qwb-core' -Body { cargo deny check }
Step -Name 'core: K1 schema freshness' -Dir 'qwb-core' -Body {
    cargo run -q -p qwb-registry --bin qwb-gen-schemas -- ../qwb-contracts/schema --check
}
Step -Name 'core: G2 data verify' -Dir 'qwb-core' -Body {
    cargo run -q -p qwb-cli --bin qwb -- data verify
}

# ---- D-15: data licenses ----
Step -Name 'D-15: DATA_LICENSES.md matches the data' -Dir '.' -Body {
    & python scripts/gen_data_licenses.py --check
}

# ---- DA-08: every reference CSV row must carry its source ----
# 纯标准库（不 import PySCF），所以 Windows 侧也能查：逐行非空 + DOI/URL 形状 +
# 来源白名单 + 每个家族文件都有带 `--check` 的生成脚本。
Step -Name 'DA-08: reference CSVs carry source_doi/source_url' -Dir '.' -Body {
    & python scripts/check_reference_sources.py
}

# ---- VF-05: the error budget must stay measured, not narrated ----
# 三件事：§0 每行有数字（或显式「未测/blocked」）、每行证据能解析到**存在的**路径或账本 WP、
# §5 的 9 条 §9.3 容差一条不少。纯标准库。
Step -Name 'VF-05: docs/error-budget.md rows are measured' -Dir '.' -Body {
    & python scripts/check_error_budget.py
}

# ---- PR-08: the USAGE honest list must not drift from the ledger ----
# §9.5 的每一行都指明 WP：`✅` 要求列出的 WP 全部 done、`⚠` 不许 pending、`❌` 不许 done；
# 正文里出现状态标记却没进清单也算红。第二条跑 `--self-test`：三种典型漂移（✅ 指向 pending、
# ❌ 指向 done、正文里"夸一句能算"）必须都被抓到 —— 否则"有检查脚本"这句话本身没有证据。
Step -Name 'PR-08: USAGE honest list matches the ledger' -Dir '.' -Body {
    & python scripts/check_usage_honest_list.py
}

Step -Name 'PR-08: honest-list checker catches drift (self-test)' -Dir '.' -Body {
    & python scripts/check_usage_honest_list.py --self-test
}

# DA-06：色散数据必须能从上游源码重新生成（逐项比较，容差 0），并且能与官方实现的
# 参考值对上（能量与解析梯度）。两条都是离线可跑的：前者读 `.d3-upstream` 缓存
# （缺失时按 `UPSTREAM` 取回），后者只读仓库里的 CSV。
Step -Name 'DA-06: dispersion data matches upstream sources' -Dir '.' -Body {
    & $script:ledgerPython scripts/gen_d3_params.py --check
}

Step -Name 'DA-06: dispersion data reproduces s-dftd3' -Dir '.' -Body {
    & $script:ledgerPython scripts/validate_d3_data.py
}

# 零阻尼（D3(0)）那一份单独跑：M06-L / M06-2X 只有这一套参数。容差 1e-8 Ha 而不是
# 1e-9：零阻尼的 R0 来自 vdW 半径（2–4 Å），Å→Bohr 常数代差被放大，实测 2.1e-9 Ha。
Step -Name 'DA-06: zero-damping data reproduces s-dftd3' -Dir '.' -Body {
    & $script:ledgerPython scripts/validate_d3_data.py --zero --tolerance 1e-8
}

# VF-06：K10 的等价性差异表。**v3.2 起是强制模式**（此前为 `--report`）：RI-MP2 vs canonical MP2
# 实测差 8.526e-05 Ha，而 `numerics.yaml` 声明的 `ri` 容差是 1e-6 —— 差的是**辅助基不完备**
# 而不是实现（同一差值就是 SC-05 记的 ri_error）。把这条从报告改成判红，需要先把
# K10 的容差已于 v3.2 重新界定（`ri` 1e-6 -> 2e-4，把「实现等价」与「辅助基质量」分开，
# 见 project-phase2.md §4.5 修订 1 与 choices 二百）；数据版本随之 bump。
Step -Name 'VF-06: K10 equivalence table (enforcing since v3.2)' -Dir '.' -Body {
    & $script:ledgerPython scripts/check_numerics_equivalence.py
}

# ---- 10.6: the project ledger must stay machine-readable ----
Step -Name '10.6: docs/status.yaml ledger validity' -Dir '.' -Body {
    & $script:ledgerPython scripts/check_status.py
}

# ---- phase 2 (project-phase2.md): the v2 ledger must stay machine-readable ----
# Same checker, different file: the phase-2 ledger starts all-pending, so this step is
# green from day one and catches corruption as WPs get marked done.
Step -Name '10.6: docs/status-v2.yaml ledger validity' -Dir '.' -Body {
    & $script:ledgerPython scripts/check_status.py docs/status-v2.yaml
}

# ---- Python gates (repo .venv): K6 stubs + P-01 model layer ----
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $script:fail++
    $script:failedNames += 'K6 mypy: .venv python not found'
    Write-Host "!! FAILED: missing $py (create it with: python -m venv .venv; pip install mypy numpy pydantic pytest)" -ForegroundColor Red
}
else {
    # Run mypy from the stub directory: the cwd is itself a module search root, so
    # no MYPYPATH is needed (MYPYPATH is a path-escaping minefield across Windows/WSL).
    Step -Name 'K6: mypy --strict (stub)' -Dir 'qwb-contracts/studio/python' -Body {
        & $py -m mypy --strict qwb_engine.pyi
    }
    Step -Name 'K6: mypy --strict (typed consumer)' -Dir 'qwb-contracts/studio/python' -Body {
        & $py -m mypy --strict tests/consumer_example.py
    }
    # P-06: the pyo3 binding crate (own workspace). fmt + clippy here because
    # `cargo clippy --workspace` in qwb-core cannot see it.
    Step -Name 'P-06: pyo3 crate fmt' -Dir 'qwb-studio/rust/qwb-py' -Body { cargo fmt --check }
    Step -Name 'P-06: pyo3 crate clippy -D warnings' -Dir 'qwb-studio/rust/qwb-py' -Body {
        cargo clippy --all-targets -- -D warnings
    }
    # P-06 acceptance needs the built extension: maturin wheel (abi3-py39) then
    # install it into the same venv that runs the Python tests.
    Step -Name 'P-06: build + install qwb_engine (maturin, abi3-py39)' -Dir 'qwb-studio/rust/qwb-py' -Body {
        & $py -m maturin build --release -o dist
        if ($LASTEXITCODE -ne 0) { return }
        $wheel = Get-ChildItem dist\*.whl | Select-Object -First 1
        & $py -m pip install --force-reinstall --no-deps --quiet $wheel.FullName
    }
    # P-01 acceptance: pydantic models match task.schema.yaml field by field,
    # YAML/JSON roundtrip, illegal state transitions raise ValueError.
    Step -Name 'P-01/P-03/P-04/P-05/P-06: qwb-studio pytest' -Dir 'qwb-studio' -Body {
        # EX-02（第 379 轮）：pytest 的 206 个用例里有**一个** GUI 压力测试独占 42–45 s
        # （`test_big_log_streams_without_freezing_the_ui`），串行时整个套件 86 s。把它单独放
        # 一条泳道、其余放另一条并发跑，实测 **86 s → 46 s**。
        #
        # **测试集合不变**由三条断言保证（任何一条不成立就红，不会静默少跑）：
        #   ① `--collect-only` 收集到的用例数 = 全集总数；
        #   ② 泳道 A 只跑那一个用例，必须恰好 `1 passed`（用例改名/消失 → 0 passed → 红）；
        #   ③ A + B 的 `passed` 之和 == 总数（`--deselect` 若失配，B 会多跑一个 → 红）。
        # `--collect-only` 那一步很贵（要把所有测试模块 import 一遍，~20 s），所以改成用
        # **`--deselect` 的命中数**来证明切分正确：B 的摘要里必须恰好 `1 deselected`
        # （失配时 pytest 会报 "not found" → 非零退出），A 必须恰好 `1 passed`。
        $bigTest = 'tests/test_gui.py::test_big_log_streams_without_freezing_the_ui'
        $laneA = Start-Process -FilePath $py -PassThru -NoNewWindow `
            -RedirectStandardOutput pytest-lane-a.log -RedirectStandardError pytest-lane-a.err `
            -ArgumentList @('-m', 'pytest', '-q', $bigTest)
        & $py -m pytest -q --deselect $bigTest *> pytest-lane-b.log
        $codeB = $LASTEXITCODE
        $laneA.WaitForExit()
        $codeA = $laneA.ExitCode
        Get-Content pytest-lane-a.log -Tail 2
        Get-Content pytest-lane-b.log -Tail 2
        Write-Host ("   pytest lanes: A exit {0}, B exit {1}" -f $codeA, $codeB)
        if ($codeA -ne 0 -or $codeB -ne 0) { throw 'pytest lane failed' }
        function GetCount([string] $path, [string] $pattern) {
            $match = Get-Content $path | Select-String -Pattern $pattern | Select-Object -Last 1
            if ($null -eq $match) { return -1 }
            return [int] $match.Matches[0].Groups[1].Value
        }
        $passedA = GetCount 'pytest-lane-a.log' '(\d+) passed'
        $deselectedB = GetCount 'pytest-lane-b.log' '(\d+) deselected'
        $passedB = GetCount 'pytest-lane-b.log' '(\d+) passed'
        Write-Host ("   pytest lanes: A {0} passed; B {1} passed / {2} deselected" -f $passedA, $passedB, $deselectedB)
        if ($passedA -ne 1) { throw "the isolated GUI test did not run exactly once (A passed = $passedA)" }
        if ($deselectedB -ne 1) { throw "the deselect matched $deselectedB tests instead of 1" }
        if ($passedB -lt 1) { throw "lane B ran no tests (passed = $passedB)" }
    }
    # P-02: the committed Python-written .qwb must still be what the current
    # writer produces; qwb-io's p02_interop test reads it from the Rust side.
    Step -Name 'P-02: .qwb fixture freshness' -Dir '.' -Body {
        & $py scripts\gen_p02_fixture.py --check
    }
}

Write-Host ''
Write-Host '----------------------------------------'
# EX-04：最慢的五步 + 总计（预算见 project-phase2.md 附录 A.3：总墙钟 ≤ 15 min）
$script:stepTimes | Sort-Object Seconds -Descending | Select-Object -First 5 |
    ForEach-Object { Write-Host ("  slowest: {0,7:N1}s  {1}" -f $_.Seconds, $_.Name) }
$totalSeconds = ($script:stepTimes | Measure-Object Seconds -Sum).Sum
Write-Host ("  total step time: {0:N1}s ({1:N1} min)" -f $totalSeconds, ($totalSeconds / 60))
Write-Host "passed $($script:pass), failed $($script:fail)"
if ($script:fail -ne 0) {
    Write-Host 'failed checks:'
    $script:failedNames | ForEach-Object { Write-Host "  - $_" }
    exit 1
}
Write-Host 'ALL GATES GREEN'
exit 0
