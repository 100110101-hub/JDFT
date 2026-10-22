#!/usr/bin/env bash
# G6 / R-16：`#![deny(unsafe_code)]` 是默认，`unsafe` 只允许在
# scripts/unsafe_allowlist.txt 里登记过的文件出现，且每处必须有 `// SAFETY:` 说明。
#
# 例外需要人类批准（§3.7 / §10.4），批准后把文件路径加进白名单并写理由。
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
allowlist="$root/scripts/unsafe_allowlist.txt"
fail=0

echo "== 1. 每个 crate 都必须声明 deny(unsafe_code) =="
while IFS= read -r -d '' manifest; do
    dir="$(dirname "$manifest")"
    if ! grep -rq --include='*.rs' '#!\[deny(unsafe_code)\]' "$dir"; then
        echo "FAIL $(realpath --relative-to="$root" "$dir"): missing #![deny(unsafe_code)]"
        fail=1
    fi
done < <(find "$root/qwb-core/crates" "$root/qwb-contracts/engine-api" -name Cargo.toml -print0)

echo "== 2. 扫描 unsafe 用法 =="
# 只匹配真正的 unsafe 关键字，跳过注释与字符串里出现 "unsafe" 的说明文字。
mapfile -t hits < <(
    grep -rnE --include='*.rs' '\bunsafe\b' "$root/qwb-core/crates" "$root/qwb-contracts/engine-api/src" \
        | grep -vE '^[^:]+:[0-9]+: *(//|///|//!)' \
        | grep -vE '^[^:]+:[0-9]+: *#!?\[(deny|forbid)\(unsafe_code\)\]' \
        || true
)

if [ "${#hits[@]}" -eq 0 ]; then
    echo "ok: no unsafe keyword in product code"
    exit 0
fi

for hit in "${hits[@]}"; do
    file="${hit%%:*}"
    if [ -f "$allowlist" ] && grep -qxF "$(realpath --relative-to="$root" "$file")" "$allowlist"; then
        echo "allowed: $hit"
    else
        echo "FAIL unsafe outside allowlist: $hit"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "unsafe 需要人批准 + // SAFETY: 注释 + 白名单登记（R-16）。"
fi
exit "$fail"
