#!/usr/bin/env bash
# G7 / R-13：所有 rayon 归约必须确定性（按索引分块，不许顺序依赖 `reduce`）。
#
# 本脚本做两件事：
#   1. 禁止出现 `reduce(` / `.fold(` 后的顺序敏感累加模式（黑名单式 grep）；
#   2. 校验 `DeterministicReduce` 的使用：凡是 `par_iter` 就在附近必须能找到它。
#
# C 轨还没有 rayon 使用，所以现在主要防“未来的第一次误用”。数值并行落地（I-04）后
# 这里要加 “1 线程 vs N 线程 byte-identical” 的实际执行。
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sources="$root/qwb-core/crates"
fail=0

echo "== 1. 禁止非确定性归约 =="
if grep -rnE --include='*.rs' '\.reduce\(' "$sources" 2>/dev/null | grep -v '//' ; then
    echo "FAIL rayon::reduce 出现在产品代码里（G7 禁止顺序依赖归约）"
    fail=1
else
    echo "ok: no .reduce() in product code"
fi

echo "== 2. par_iter 必须配 DeterministicReduce =="
mapfile -t par_iters < <(grep -rln --include='*.rs' 'par_iter' "$sources" 2>/dev/null || true)
for file in "${par_iters[@]:-}"; do
    [ -n "$file" ] || continue
    if ! grep -q 'DeterministicReduce' "$file"; then
        echo "FAIL $(realpath --relative-to="$root" "$file") uses par_iter without DeterministicReduce"
        fail=1
    fi
done
if [ "${#par_iters[@]}" -eq 0 ]; then
    echo "ok: no par_iter yet (F/I 轨会引入)"
fi

exit "$fail"
