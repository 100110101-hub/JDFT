#!/usr/bin/env bash
# G6 / §10.4 的可检查形式：三方依赖只能出现在 workspace 的 [workspace.dependencies]。
#
# 规则：
#   1. qwb-core/crates/*/Cargo.toml 里的 [dependencies]/[dev-dependencies] 只允许
#      `name.workspace = true` 形式；
#   2. 版本号只允许出现在 qwb-core/Cargo.toml 的 [workspace.dependencies]。
#
# 违反了就退 1，并打印是哪一行。
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

for manifest in "$root"/qwb-core/crates/*/Cargo.toml; do
    section=""
    lineno=0
    while IFS= read -r line; do
        lineno=$((lineno + 1))
        case "$line" in
            \[dependencies\]* | \[dev-dependencies\]* | \[build-dependencies\]*)
                section="deps"
                continue
                ;;
            \[*)
                section=""
                continue
                ;;
        esac
        [ "$section" = "deps" ] || continue
        case "$line" in
            "" | "#"*) continue ;;
        esac
        # 两种合法写法都必须继承 workspace 的版本：
        #   name.workspace = true
        #   name = { workspace = true, optional = true }
        case "$line" in
            *"workspace = true"*) continue ;;
        esac
        echo "FAIL $(realpath --relative-to="$root" "$manifest"):$lineno: dependency must inherit from [workspace.dependencies] (name.workspace = true): $line"
        fail=1
    done <"$manifest"
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "把三方依赖的版本集中到 qwb-core/Cargo.toml 的 [workspace.dependencies]，"
    echo "并在 PR body 里说明理由（project.md §10.2.6 / §10.4）。"
    exit 1
fi

echo "ok: all workspace crates inherit dependencies from [workspace.dependencies]"
