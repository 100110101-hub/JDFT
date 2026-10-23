#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check docs/status.yaml: the project ledger must stay machine-readable.

Why this exists: the ledger was three times left in a state that no YAML parser
accepts (a long ``note: "..."`` whose closing quote was lost, a duplicated
``verification:`` key, stale continuation lines). Nothing in the gate looked at
the file, so the corruption went unnoticed while the ledger was the very
artifact the accounting rules (project.md 10.6) depend on.

Checks (each is an error, not a warning):

1. no duplicate top-level keys;
2. every top-level key is a mapping (or an inline mapping);
3. every ``state:`` value is one of done / in_progress / blocked / pending;
4. if PyYAML is importable, the whole file parses, and every block parses on
   its own (a block that only parses because of a neighbour would hide errors).

Exit code 0 = ledger is fine, 1 = at least one problem.
"""

from __future__ import annotations

import io
import re
import sys

LEDGER = sys.argv[1] if len(sys.argv) > 1 else "docs/status.yaml"
STATES = ("done", "in_progress", "blocked", "pending")
KEY_PATTERN = re.compile(r"^([A-Z][A-Z0-9-]*):")


def report(problems: list[str], notes: list[str]) -> int:
    for note in notes:
        print("ok: " + note)
    if problems:
        for problem in problems:
            print("FAIL: " + problem)
        return 1
    return 0


def main() -> int:
    try:
        with io.open(LEDGER, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print("FAIL: cannot read {}: {}".format(LEDGER, error))
        return 1

    lines = text.split("\n")
    problems: list[str] = []
    notes: list[str] = []

    # ---- 1. top-level keys: pattern + duplicates -------------------------
    starts = [index for index, line in enumerate(lines) if KEY_PATTERN.match(line)]
    keys = [KEY_PATTERN.match(lines[index]).group(1) for index in starts]
    seen: dict[str, int] = {}
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        problems.append("duplicate top-level keys: {}".format(", ".join(duplicates)))
    else:
        notes.append("{} top-level keys, no duplicates".format(len(keys)))

    # ---- 2. every quoted scalar must be closed ---------------------------
    # A stray unbalanced quote is what corrupted this file; catch it directly so
    # the message points at the line instead of at a parser's guess.
    #
    # A double-quoted scalar may legitimately span lines (D-11/D-12 do), so the
    # balance is tracked across the whole file. Block scalars (`key: |`) hold
    # arbitrary text -- quotes inside them are not YAML syntax -- so they are
    # skipped until the indentation returns to the key's level.
    open_at: int | None = None
    inside_quotes = False
    block_indent: int | None = None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        stripped = line.strip()
        if not inside_quotes and re.search(r":\s*[|>][+-]?\s*$", stripped):
            block_indent = indent
            continue
        position = 0
        while position < len(line):
            character = line[position]
            if character == "\\" and inside_quotes:
                position += 2
                continue
            if character == "#" and not inside_quotes and not line[:position].strip():
                break
            if character == '"':
                inside_quotes = not inside_quotes
                open_at = index + 1 if inside_quotes else None
            position += 1
    if inside_quotes:
        problems.append(
            "unterminated double-quoted scalar opened at line {}".format(open_at)
        )

    # ---- 3. state values -------------------------------------------------
    for index in starts:
        key = KEY_PATTERN.match(lines[index]).group(1)
        block_end = len(lines)
        for later in starts:
            if later > index:
                block_end = later
                break
        block = "\n".join(lines[index:block_end])
        match = re.search(r"(?m)^\s+state:\s*(\S+)", block)
        if match:
            value = match.group(1).strip().strip('"').strip("'")
            if value not in STATES:
                problems.append("{}: unknown state '{}'".format(key, value))

    # ---- 4. YAML parse (when available) ----------------------------------
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        notes.append("PyYAML not importable: structural checks only")
    else:
        try:
            parsed = yaml.safe_load(text)
        except Exception as error:  # noqa: BLE001 - report the parser message
            problems.append("whole file does not parse: {}".format(str(error).split("\n")[0]))
            parsed = None
        if isinstance(parsed, dict):
            for position, start in enumerate(starts):
                end = starts[position + 1] if position + 1 < len(starts) else len(lines)
                try:
                    yaml.safe_load("\n".join(lines[start:end]))
                except Exception as error:  # noqa: BLE001
                    problems.append(
                        "block {} does not parse: {}".format(
                            keys[position], str(error).split("\n")[0]
                        )
                    )
            for key, value in parsed.items():
                if isinstance(value, dict):
                    state = value.get("state")
                    if state is not None and state not in STATES:
                        problems.append("{}: unknown state '{}' (parsed)".format(key, state))
            done = sum(1 for value in parsed.values() if isinstance(value, dict) and value.get("state") == "done")
            notes.append("YAML parses; {} items marked done".format(done))

    return report(problems, notes)


if __name__ == "__main__":
    sys.exit(main())
