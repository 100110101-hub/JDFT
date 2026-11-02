"""PR-08 门禁：`docs/USAGE.md` 的「诚实清单」不许与 `docs/status-v2.yaml` 漂移。

文档里最危险的句子是「能算」。散文没法自动验证，但只要**每句断言都指出它在台账里的位置**，
下面这些东西就能机械核对 —— 而它们正是"文档漂移"的典型形态：

1. `§9.5` 有一张包在 `<!-- honest-list:begin -->` / `<!-- honest-list:end -->` 之间的表，
   每行 `| 条目 | 声称 | WP |`（第四列可写备注，不参与判定）；
2. **`✅` 要求列出的 WP 全部 `done`**：说"能算"就得有一个已验收的工作包撑着；
3. **`⚠` 要求没有任何列出的 WP 是 `pending`**：`pending` 意味着代码还没落地，
   那就不是"能算但有缺口"而是"没有"；
4. **`❌` 不许列出一个 `done` 的 WP**：反过来也不能拿已完成的工作包当"不能算"的挡箭牌；
5. **同一个 WP 不许在一行里 `✅`、在另一行里 `❌`**（自相矛盾）；
6. **文中任何 `✅/⚠/❌` 都必须在清单里**：想加一句"能算"，就得同时写出它对应哪个 WP。
   为了能把规则本身写进 §9.5，该节标题到清单块标记之间的**解释性文字**豁免（只豁免这一段）。

`--self-test` 拿三种典型漂移在内存里改一遍，必须都被抓到 —— 否则"有检查脚本"这句话
本身就没有证据。

只依赖标准库（Windows 与 WSL 两侧门禁都能跑，与 `check_error_budget.py` 同一个口径）。
"""

import io
import os
import re
import sys

# Windows 的控制台默认是 GBK，打不出 ✅ 之类的符号（会 UnicodeEncodeError，而不是报错）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "USAGE.md")
LEDGER = os.path.join(ROOT, "docs", "status-v2.yaml")

BEGIN = "<!-- honest-list:begin -->"
END = "<!-- honest-list:end -->"
GLYPHS = "✅⚠❌"
CLAIM_NAMES = {"✅": "ok", "⚠": "partial", "❌": "missing"}
SECTION = "### 9.5"
KEY_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):\s*$")
STATE_LINE = re.compile(r"^  state:\s*([A-Za-z_]+)\s*$")
WP_SPLIT = re.compile(r"[,，、/\s]+")
ROW_LINE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([" + GLYPHS + r"])\s*\|\s*([^|]*?)\s*\|")


def ledger_states():
    """用正则读顶层键与它下面第一行 `  state:`（不 import PyYAML）。"""
    states = {}
    current = None
    for line in io.open(LEDGER, encoding="utf-8"):
        match = KEY_LINE.match(line)
        if match:
            current = match.group(1)
            states.pop(current, None)
            continue
        if current is not None:
            match = STATE_LINE.match(line.rstrip("\n"))
            if match:
                states[current] = match.group(1)
                current = None
    return states


def check(text, states):
    """返回 (problems, stats)。stats 记行数与三种声称的计数。"""
    lines = text.splitlines()
    problems = []
    stats = {"rows": 0, "✅": 0, "⚠": 0, "❌": 0}

    begins = [index for index, line in enumerate(lines) if line.strip() == BEGIN]
    ends = [index for index, line in enumerate(lines) if line.strip() == END]
    if len(begins) != 1 or len(ends) != 1:
        return [f"honest-list block: {len(begins)} begin marker(s), {len(ends)} end marker(s)"], stats
    begin, end = begins[0], ends[0]
    if end <= begin:
        return [f"honest-list block: end marker (line {end + 1}) precedes begin (line {begin + 1})"], stats

    # ---- 6. 正文里的状态标记必须都在清单块里（§9.5 的解释性文字豁免）----
    exempt_from = None
    for index in range(begin - 1, -1, -1):
        if lines[index].startswith(SECTION):
            exempt_from = index
            break
    for index, line in enumerate(lines):
        if begin < index < end:
            continue
        if exempt_from is not None and exempt_from <= index < begin:
            continue
        found = [glyph for glyph in GLYPHS if glyph in line]
        if found:
            problems.append(
                f"line {index + 1}: status glyph {''.join(found)} outside the honest-list block "
                f"(add it to §9.5 with its WP): {line.strip()[:80]!r}"
            )

    # ---- 1. 表的形状 ----
    rows = []
    for index in range(begin + 1, end):
        stripped = lines[index].strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # 分隔行
        if cells[0] == "条目":
            continue  # 表头
        if len(cells) < 3:
            problems.append(f"line {index + 1}: honest-list row needs 条目 / 声称 / WP: {stripped}")
            continue
        rows.append((index + 1, cells))

    if len(rows) < 10:
        problems.append(f"honest list has only {len(rows)} rows: the §9 claims are not covered")
    stats["rows"] = len(rows)

    if len(states) < 50:
        problems.append(f"docs/status-v2.yaml: parsed only {len(states)} WP states")

    conflicts = {}
    for line_no, cells in rows:
        label, claim, wps = cells[0], cells[1], cells[2]
        if claim not in GLYPHS:
            problems.append(f"line {line_no}: claim must be one of {GLYPHS}, got {claim!r} ({label})")
            continue
        stats[claim] += 1
        ids = [token for token in WP_SPLIT.split(wps) if token]
        if not ids:
            problems.append(f"line {line_no}: claim {claim} names no WP ({label})")
            continue
        for wp in ids:
            if wp not in states:
                problems.append(f"line {line_no}: {wp} is not a WP in docs/status-v2.yaml ({label})")
                continue
            state = states[wp]
            conflicts.setdefault(wp, set()).add(claim)
            if claim == "✅" and state != "done":
                problems.append(f"line {line_no}: claims 能算 but {wp} is {state} ({label})")
            elif claim == "⚠" and state == "pending":
                problems.append(f"line {line_no}: claims 能算但有缺口 but {wp} is pending ({label})")
            elif claim == "❌" and state == "done":
                problems.append(f"line {line_no}: claims 不能算 but {wp} is done ({label})")

    # ---- 5. 同一个 WP 不许既 ✅ 又 ❌ ----
    for wp, seen in sorted(conflicts.items()):
        if "✅" in seen and "❌" in seen:
            problems.append(f"{wp} is claimed both ✅ and ❌ across rows")
    return problems, stats


def mutations(text, states):
    """三种典型漂移，每一种都必须被 check() 抓到。"""
    pending = sorted(key for key, state in states.items() if state == "pending")
    done = sorted(key for key, state in states.items() if state == "done")
    cases = []

    def replace_first_row(claim, new_wps):
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            match = ROW_LINE.match(line)
            if match and match.group(2) == claim:
                lines[index] = f"| {match.group(1)} | {claim} | {new_wps} | self-test |\n"
                return "".join(lines)
        raise SystemExit(f"self-test: no {claim} row to mutate")

    if not pending or not done:
        raise SystemExit("self-test: the ledger has no pending/done WP to build mutations from")
    cases.append(("✅ row pointing at a pending WP", replace_first_row("✅", pending[0])))
    cases.append(("❌ row pointing at a done WP", replace_first_row("❌", done[0])))
    cases.append(("a 能算 claim outside the block", "能算（自测）✅\n" + text))
    return cases


def main(argv):
    states = ledger_states()
    text = io.open(DOC, encoding="utf-8").read()

    if "--self-test" in argv:
        failures = 0
        for label, mutated in mutations(text, states):
            problems, _stats = check(mutated, states)
            if problems:
                print(f"ok: self-test caught it — {label} -> {problems[0]}")
            else:
                print(f"FAIL self-test missed it — {label}")
                failures += 1
        return 1 if failures else 0

    problems, stats = check(text, states)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    counts = " / ".join(f"{stats[glyph]} {CLAIM_NAMES[glyph]}" for glyph in GLYPHS)
    print(
        f"ok: {stats['rows']} honest-list rows checked against {len(states)} ledger WPs "
        f"({counts}); no status glyph outside §9.5"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
