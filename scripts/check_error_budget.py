"""VF-05 门禁：`docs/error-budget.md` v2 的每一行都必须有数字**和有出处的证据**。

判据原文：「每个误差源一行实测数字 + 来源；与 §9 容差表一致」。散文没法自动验证，
但下面三件事可以，而且正是"误差预算退化成感想"的三种典型方式：

1. **`§0` 的每一行都要有量级**：`当前量级` 列必须含数字（`\\d`），或者是显式的
   `未测` / `blocked` / `记录` 标记。空着、写 "小"、"可接受"、"—" 都不行；
2. **每一行都要有能解析的证据**：`证据` 列里至少出现一个**已存在**的仓库路径
   （测试/源码/账本）或一个**账本里有的** WP id（`docs/status-v2.yaml` 的键）。
   写得再漂亮的数字，找不到出处就是传闻；
3. **`§5` 的映射表不许比 `§9.3` 少行**：那 9 条容差每条都要有对应的实测行 ——
   删行能让 §0 变干净，但这里会红。

只依赖标准库（Windows 门禁也能跑）。
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "error-budget.md")
LEDGER = os.path.join(ROOT, "docs", "status-v2.yaml")

NUMBER = re.compile(r"\d")
#: 允许的"没有数字"标记（显式说明为什么没有数）。
EXPLICIT_MARKERS = ("未测", "blocked", "记录")
PATH_TOKEN = re.compile(r"`([A-Za-z0-9_./-]+\.(?:rs|py|md|yaml|yml|sh|ps1))`")
WP_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,2}-?[A-Z]?[0-9]{1,2}[a-c]?)(?![A-Za-z0-9])")


def table_rows(text: str, heading: str) -> list[list[str]]:
    """取某个 `## heading` 之后第一张表的行（跳过表头与分隔行）。"""
    start = text.index(heading)
    lines = text[start:].splitlines()
    rows = []
    started = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            if not started:
                started = True
                continue  # 表头
            if set(stripped) <= set("|-: "):
                continue  # 分隔行
            rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
        elif started:
            break
    return rows


def ledger_keys() -> set[str]:
    keys = set()
    for line in io.open(LEDGER, encoding="utf-8"):
        match = re.match(r"^([A-Z]{1,2}-?[A-Z]?[0-9]{1,2}[a-c]?):\s*$", line)
        if match:
            keys.add(match.group(1))
    return keys


def main() -> int:
    text = io.open(DOC, encoding="utf-8").read()
    problems = []
    keys = ledger_keys()
    if not keys:
        problems.append("docs/status-v2.yaml: could not parse any WP id")

    rows = table_rows(text, "## 0. 阶段二总表")
    if len(rows) < 10:
        problems.append(f"section-0 has only {len(rows)} rows: the approximation table is incomplete")
    sources = 0
    for row in rows:
        if len(row) != 5:
            problems.append(f"section-0 row has {len(row)} columns (expected 5): {row}")
            continue
        source, magnitude, _tolerance, _status, evidence = row
        if not (NUMBER.search(magnitude) or any(marker in magnitude for marker in EXPLICIT_MARKERS)):
            problems.append(f"section-0 {source}: magnitude has no number and no 未测/blocked marker: {magnitude!r}")
        paths = [token for token in PATH_TOKEN.findall(evidence) if os.path.exists(os.path.join(ROOT, token))]
        wps = [token for token in WP_TOKEN.findall(evidence) if token in keys]
        if not paths and not wps:
            problems.append(f"section-0 {source}: evidence cites no existing path or ledger WP: {evidence!r}")
        sources += len(paths) + len(wps)

    mapping = table_rows(text, "## 5. 与 §9.3 容差表的对应")
    if len(mapping) != 9:
        problems.append(
            f"section-5 has {len(mapping)} rows, but project-phase2.md §9.3 has 9 tolerances"
        )
    for row in mapping:
        if len(row) != 3:
            problems.append(f"section-5 row has {len(row)} columns (expected 3): {row}")

    # §6 的未测项必须点名阻塞它的 WP，否则"未测"就成了万能挡箭牌
    for row in table_rows(text, "## 6. 未测项与它们的阻塞"):
        if len(row) != 3:
            continue
        blocked = [token for token in WP_TOKEN.findall(row[1]) if token in keys]
        if not blocked:
            problems.append(f"section-6 row: no ledger WP named as the blocker: {row[1]!r}")

    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print(
        f"ok: section-0 {len(rows)} approximation rows, {sources} evidence links, "
        f"section-5 {len(mapping)}/9 tolerances mapped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
