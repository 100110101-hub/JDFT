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

# ---- 10.6: the project ledger must stay machine-readable ----
Step -Name '10.6: docs/status.yaml ledger validity' -Dir '.' -Body {
    & python scripts/check_status.py
}

# ---- phase 2 (project-phase2.md): the v2 ledger must stay machine-readable ----
# Same checker, different file: the phase-2 ledger starts all-pending, so this step is
# green from day one and catches corruption as WPs get marked done.
Step -Name '10.6: docs/status-v2.yaml ledger validity' -Dir '.' -Body {
    & python scripts/check_status.py docs/status-v2.yaml
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
        & $py -m pytest -q
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
