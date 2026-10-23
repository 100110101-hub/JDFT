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

step() { # step <name> <cmd...>
    local name="$1"
    shift
    printf '\n== %s ==\n' "$name"
    if "$@"; then
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

# ---- 门禁脚本 ----
step "data: DATA_LICENSES.md 与数据一致（D-15）"
python scripts/gen_data_licenses.py --check
step "data: status.yaml 账本可解析（§10.6）"
python scripts/check_status.py
step "gate: deps centralized" bash scripts/check_deps_centralized.sh
step "gate: no unsafe outside allowlist" bash scripts/check_unsafe.sh
step "gate: rayon determinism" bash scripts/check_rayon_determinism.sh

# ---- K2 契约 crate ----
step "contracts(K2): fmt" bash -c "cd qwb-contracts/engine-api && \"\$0\" fmt --check" "$CARGO"
step "contracts(K2): clippy -D warnings" bash -c "cd qwb-contracts/engine-api && \"\$0\" clippy --all-targets -- -D warnings" "$CARGO"
step "contracts(K2): cargo doc" bash -c "cd qwb-contracts/engine-api && RUSTDOCFLAGS='-D warnings' \"\$0\" doc --no-deps" "$CARGO"
step "contracts(K2): test" bash -c "cd qwb-contracts/engine-api && \"\$0\" test --all-targets" "$CARGO"

# ---- qwb-core ----
step "core: fmt" bash -c "cd qwb-core && \"\$0\" fmt --all --check" "$CARGO"
step "core: clippy -D warnings" bash -c "cd qwb-core && \"\$0\" clippy --workspace --all-targets -- -D warnings" "$CARGO"
step "core: test" bash -c "cd qwb-core && \"\$0\" test --workspace" "$CARGO"
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
fi

printf '\n────────────────────────────────────────\n'
printf '通过 %d 项，失败 %d 项\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
    printf '失败清单：%s\n' "$names_failed"
    exit 1
fi
echo "全部门禁通过 ✅"
exit 0
