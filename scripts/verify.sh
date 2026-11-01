#!/usr/bin/env bash
# 全套门禁（Linux / CI / WSL 用）。
#
# Windows 原生等价入口：scripts/verify.ps1（覆盖同一清单；改一个就同步改另一个）。
#
# 退出码：0 = 全绿；1 = 至少一项失败。
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0
fail=0
names_failed=""

step_times=""
step() { # step <name> <cmd...>
    local name="$1"
    shift
    printf '\n== %s ==\n' "$name"
    # EX-04：与 verify.ps1 同步，逐步报墙钟（预算见附录 A.3：总墙钟 ≤ 15 min）
    local started ended seconds
    started=$(date +%s)
    if "$@"; then
        ended=$(date +%s); seconds=$((ended - started)); step_times="$step_times$seconds $name
"
        printf '   [%7ss]\n' "$seconds"
        pass=$((pass + 1))
    else
        local rc=$?
        fail=$((fail + 1))
        names_failed="$names_failed
  - $name (exit $rc)"
        printf '\n!! FAILED: %s (exit %d)\n' "$name" "$rc"
    fi
}

echo "QWB verify · 仓库 $root"

# ---- 工具定位 ----
# WSL 下 Windows 的 cargo 是 cargo.exe（bash 不会自动补 .exe），所以要显式兜底。
CARGO="${CARGO:-}"
if [ -z "$CARGO" ]; then
    CARGO="$(command -v cargo || command -v cargo.exe || true)"
fi
if [ -z "$CARGO" ]; then
    echo "!! FAILED: 找不到 cargo（Windows 请用 scripts/verify.ps1；Linux/CI 请装 Rust toolchain）"
    exit 1
fi

PY=""
if [ -x "$root/.venv/Scripts/python.exe" ]; then
    PY="$root/.venv/Scripts/python.exe" # Windows venv（WSL 下 cwd 会被翻译，故只用相对路径）
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import mypy' >/dev/null 2>&1; then
    PY="$(command -v python3)"
fi

# P-01 需要 pydantic v2 + pytest（依赖白名单 §2.4 允许 pydantic；pytest 只在 test extra 里）。
PY_STUDIO=""
if [ -x "$root/.venv/Scripts/python.exe" ]; then
    PY_STUDIO="$root/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pydantic, pytest' >/dev/null 2>&1; then
    PY_STUDIO="$(command -v python3)"
fi

# ---- 门禁脚本 ----
step "data: DATA_LICENSES.md 与数据一致（D-15）"
python scripts/gen_data_licenses.py --check
step "data: 参考集逐行带来源（DA-08）" python scripts/check_reference_sources.py
step "data: 误差预算逐行可核（VF-05）" python scripts/check_error_budget.py

# DA-06：色散数据能从上游源码重新生成（容差 0），且能复现官方 s-dftd3 的能量与梯度。
# 前者读 .d3-upstream 缓存（缺失时取回），后者只读仓库里的参考 CSV，都不需要 WSL。
step "data: 色散数据与上游逐项一致（DA-06）" "$PY" scripts/gen_d3_params.py --check
step "data: 色散数据复现官方 s-dftd3（DA-06）" "$PY" scripts/validate_d3_data.py
step "data: 零阻尼数据复现官方 s-dftd3（DA-06）" "$PY" scripts/validate_d3_data.py --zero --tolerance 1e-8
step "data: K10 等价性差异表（VF-06，报告模式）" "$PY" scripts/check_numerics_equivalence.py --report
step "data: status.yaml 账本可解析（§10.6）"
python scripts/check_status.py
step "data: status-v2.yaml 账本可解析（阶段二 §10.6）"
python scripts/check_status.py docs/status-v2.yaml
step "gate: deps centralized" bash scripts/check_deps_centralized.sh
step "gate: no unsafe outside allowlist" bash scripts/check_unsafe.sh
step "gate: rayon determinism" bash scripts/check_rayon_determinism.sh

# ---- 第三方数据（仓库故意不含这些文件，见 qwb-data/README.md）----
# BSE 导出的基组/ECP **不提交**（再分发条款，project.md §14 #20）：仓库带的是配方
# （qwb-data/sources.json）与 sha256 钉。后面每一步打开 qwb-data 都需要这些文件，
# 所以在这里取一次；新 clone 需要网络，取回之后就是纯本地。
# 等价入口：`qwb data install` / `python scripts/fetch_data.py --all`。
step "data: 第三方数据就位（缺则取回）" bash -c "python scripts/fetch_data.py --check || python scripts/fetch_data.py --all"

# ---- K2 契约 crate ----
step "contracts(K2): fmt" bash -c "cd qwb-contracts/engine-api && \"\$0\" fmt --check" "$CARGO"
step "contracts(K2): clippy -D warnings" bash -c "cd qwb-contracts/engine-api && \"\$0\" clippy --all-targets -- -D warnings" "$CARGO"
step "contracts(K2): cargo doc" bash -c "cd qwb-contracts/engine-api && RUSTDOCFLAGS='-D warnings' \"\$0\" doc --no-deps" "$CARGO"
step "contracts(K2): test" bash -c "cd qwb-contracts/engine-api && \"\$0\" test --all-targets" "$CARGO"

# ---- qwb-core ----
step "core: fmt" bash -c "cd qwb-core && \"\$0\" fmt --all --check" "$CARGO"
step "core: clippy -D warnings" bash -c "cd qwb-core && \"\$0\" clippy --workspace --all-targets -- -D warnings" "$CARGO"
# EX-02：与 verify.ps1 同步 —— 有 cargo-nextest 就走进程级并行（doctest 单独跑一步，
# 集合不变），没有就退回 `cargo test --workspace`。
# EX-02：与 verify.ps1 同步 —— 编译单独一步（不计入测试段计时），下一行才是测试段。
step "core: test binaries (prebuild, not part of the test wall clock)" bash -c '
    cd qwb-core || exit 1
    cargo test --workspace --all-targets --no-run
'
step "core: test" bash -c '
    cd qwb-core || exit 1
    NEXTEST="$(command -v cargo-nextest || command -v cargo-nextest.exe || true)"
    if [ -z "$NEXTEST" ]; then
        "$0" test --workspace
        exit $?
    fi
    # EX-02：与 verify.ps1 同步 —— --partition 保证两片恰好覆盖全集（不会漏测）。
    "$NEXTEST" run --workspace -j 4 --partition count:1/2 --no-fail-fast > shard1.log 2>&1 &
    shard1_pid=$!
    "$NEXTEST" run --workspace -j 4 --partition count:2/2 --no-fail-fast
    shard2_code=$?
    wait $shard1_pid
    shard1_code=$?
    tail -3 shard1.log
    if [ $shard1_code -ne 0 ] || [ $shard2_code -ne 0 ]; then exit 1; fi
    "$0" test --workspace --doc
' "$CARGO"
# I-00 验收：两个后端都必须编译，且给出同样的数（tests/backend_consistency.rs 的 golden 值）。
step "core: I-06 integral cache" bash -c "cd qwb-core && \"\$0\" test -p qwb-integrals --features integral-cache --test i06_cache" "$CARGO"
step "core: I-00 faer backend" bash -c "cd qwb-core && \"\$0\" test -p qwb-linalg --no-default-features --features faer" "$CARGO"
step "core: cargo deny" bash -c "cd qwb-core && \"\$0\" deny check" "$CARGO"
step "core: K1 schema freshness" bash -c "cd qwb-core && \"\$0\" run -q -p qwb-registry --bin qwb-gen-schemas -- ../qwb-contracts/schema --check" "$CARGO"
step "core: G2 data verify" bash -c "cd qwb-core && \"\$0\" run -q -p qwb-cli --bin qwb -- data verify" "$CARGO"

# ---- K6 Python 类型门禁 ----
# 在桩文件所在目录里跑 mypy：cwd 本身就是模块搜索根，因此不需要 MYPYPATH
# （MYPYPATH 在 Windows/WSL 之间会变成路径转义的雷区）。
if [ -z "$PY" ]; then
    fail=$((fail + 1))
    names_failed="$names_failed
  - K6 mypy: no python with mypy (Windows: use scripts/verify.ps1)"
    printf '\n!! FAILED: K6 mypy — 找不到带 mypy 的 python\n'
else
    step "K6: mypy --strict (stub)" bash -c "cd qwb-contracts/studio/python && \"\$0\" -m mypy --strict qwb_engine.pyi" "$PY"
    step "K6: mypy --strict (typed consumer)" bash -c "cd qwb-contracts/studio/python && \"\$0\" -m mypy --strict tests/consumer_example.py" "$PY"
    if [ -z "$PY_STUDIO" ]; then
        fail=$((fail + 1))
        names_failed="$names_failed
  - P-01 qwb-studio: no python with pydantic+pytest"
        printf '\n!! FAILED: P-01 — 找不到带 pydantic 与 pytest 的 python（pip install pydantic pytest）\n'
    else
        # P-06：pyo3 绑定是独立 workspace，qwb-core 的 clippy 看不到它，所以单独跑。
        step "P-06: pyo3 crate fmt" bash -c "cd qwb-studio/rust/qwb-py && \"\$0\" fmt --check" "$CARGO"
        step "P-06: pyo3 crate clippy -D warnings" bash -c "cd qwb-studio/rust/qwb-py && \"\$0\" clippy --all-targets -- -D warnings" "$CARGO"
        step "P-06: build + install qwb_engine (maturin, abi3-py39)" bash -c "cd qwb-studio/rust/qwb-py && \"\$0\" -m maturin build --release -o dist && \"\$0\" -m pip install --force-reinstall --no-deps --quiet dist/*.whl" "$PY_STUDIO"
        # P-01 验收：模型与 task.schema.yaml 逐字段一致、YAML/JSON 往返、非法转移抛 ValueError。
        # EX-02（第 379 轮，与 verify.ps1 同步）：一个 GUI 压力测试独占 42–45 s，把它单独放
        # 一条泳道、其余放另一条并发跑（实测 86 s → 46 s）。**测试集合不变**由三条断言保证：
        # ① 收集数 = 全集；② 泳道 A 恰好 1 passed；③ A+B 的 passed 之和 == 收集数。
        step "P-01/P-03/P-04/P-05/P-06: qwb-studio pytest" bash -c '
            cd qwb-studio || exit 1
            big="tests/test_gui.py::test_big_log_streams_without_freezing_the_ui"
            "$0" -m pytest -q "$big" > pytest-lane-a.log 2>&1 &
            laneA=$!
            "$0" -m pytest -q --deselect "$big" > pytest-lane-b.log 2>&1
            codeB=$?
            wait $laneA; codeA=$?
            tail -2 pytest-lane-a.log; tail -2 pytest-lane-b.log
            echo "   pytest lanes: A exit $codeA, B exit $codeB"
            [ "$codeA" -eq 0 ] && [ "$codeB" -eq 0 ] || exit 1
            passedA="$(grep -o "[0-9]\+ passed" pytest-lane-a.log | tail -1 | cut -d" " -f1)"
            passedB="$(grep -o "[0-9]\+ passed" pytest-lane-b.log | tail -1 | cut -d" " -f1)"
            deselectedB="$(grep -o "[0-9]\+ deselected" pytest-lane-b.log | tail -1 | cut -d" " -f1)"
            echo "   pytest lanes: A ${passedA:-0} passed; B ${passedB:-0} passed / ${deselectedB:-0} deselected"
            [ "${passedA:-0}" -eq 1 ] || { echo "the isolated GUI test did not run exactly once"; exit 1; }
            [ "${deselectedB:-0}" -eq 1 ] || { echo "the deselect matched ${deselectedB:-0} tests instead of 1"; exit 1; }
            [ "${passedB:-0}" -ge 1 ] || { echo "lane B ran no tests"; exit 1; }
        ' "$PY_STUDIO"
        # P-02：入库的 Python 写的 .qwb 必须仍是当前写入器的输出（Rust 侧 p02_interop 读它）。
        step "P-02: .qwb fixture freshness" bash -c "\"\$0\" scripts/gen_p02_fixture.py --check" "$PY_STUDIO"
    fi
fi

printf '\n────────────────────────────────────────\n'
printf '\nEX-04 最慢五步：\n'
printf '%s' "$step_times" | sort -rn | head -5 | awk '{printf "  slowest: %7ss  %s\n", $1, substr($0, index($0,$2))}'
total=$(printf '%s' "$step_times" | awk '{sum+=$1} END {print sum+0}')
printf '  total step time: %ss (%s min)\n' "$total" "$(awk "BEGIN{printf \"%.1f\", $total/60}")"
printf '通过 %d 项，失败 %d 项\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
    printf '失败清单：%s\n' "$names_failed"
    exit 1
fi
echo "全部门禁通过 ✅"
exit 0
