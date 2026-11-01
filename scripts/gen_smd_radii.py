#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取 **SMD 的参数化半径**（Marenich–Cramer–Truhlar 2009, eq. 16）并落成 staging JSON。

# 为什么这是**另一套**半径

SMD 的静电项用"参数化半径"：`R_i = R_i^SMD`（元素表 + O 依赖溶剂的 Abraham α），
而 CDS 项用的是 Bondi/Mantina 的 vdW 半径（C 1.70 / N 1.55 / O 1.52）。两者对 C/N 分别差
0.15/0.34 Å —— 混用会静默错数，所以它们落在同一个数据文件的两块里、各自标名字，并有测试
断言"两套不许相等"。

# 上游是哪个文件

PySCF 2.14.0 的 `pyscf/data/radii.py`（基础表 `VDW`，单位 Å，逐元素带文献注记）与
`pyscf/solvent/smd.py` 的 `smd_radii(alpha)`（eq. 16 的覆盖项与 O 的 α 规则）。两份原始
文本都复制到 `_smd_radii_staging/raw/` 并用 sha256 钉住 —— 与 `mnsol.F` 同一个做法。

注意 `radii.VDW` 在 PySCF 里是 `1/BOHR * array(...)`（bohr），而 `smd_radii` 先乘回 BOHR
（变成 Å）、改完覆盖项再除以 BOHR（变回 bohr）。本脚本直接取数组里的 **Å** 数值、覆盖项也
按 Å 落盘，避免那个来回换算把单位弄反。

用法：
    python3 scripts/gen_smd_radii.py            # 生成/覆盖 staging JSON
    python3 scripts/gen_smd_radii.py --check    # 与磁盘逐值比较
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(ROOT, "qwb-data", "solvent", "_smd_radii_staging")
RADII_RAW = os.path.join(STAGING, "raw", "radii.py")
RULE_RAW = os.path.join(STAGING, "raw", "smd_radii.py")
OUTPUT = os.path.join(STAGING, "smd_radii.json")

#: PySCF 版本与本项目 oracle 一致（`scripts/reference_sources.py` 的 PYSCF_DOI）。
PYSCF_VERSION = "2.14.0"
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def parse_vdw(source: str) -> list[float]:
    """取 `VDW = 1/BOHR * numpy.array((...))` 里的 Å 数值（下标 0 是 ghost）。"""
    match = re.search(r"^VDW\s*=\s*1/BOHR\s*\*\s*numpy\.array\(\(", source, re.M)
    if match is None:
        raise SystemExit("gen_smd_radii: `VDW = 1/BOHR * numpy.array((` not found")
    body = source[match.end():]
    end = body.index("))")
    values: list[float] = []
    for line in body[:end].splitlines():
        # 先按行去掉 `#` 注释：源码里注释也带数字（`# 2 He [1]`），不去掉会被当成初值。
        line = line.split("#")[0]
        line = re.sub(r"[()\s]+$", " ", line)
        for token in re.split(r"[\s,]+", line):
            token = token.strip()
            if not token:
                continue
            if token == "unknown":
                values.append(1.999999)
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def parse_rule(source: str) -> dict:
    """从 `smd_radii` 的源码里取出覆盖项与 O 的 α 规则。"""
    overrides = {
        int(index): float(value)
        for index, value in re.findall(r"radii_table\[(\d+)\]\s*=\s*([\d.]+)", source)
    }
    threshold = re.search(r"if alpha >= ([\d.]+):", source)
    high = re.search(r"if alpha >= [\d.]+:\s*\n\s*r = ([\d.]+)", source)
    low = re.search(r"r = ([\d.]+) \+ ([\d.]+) \* \(([\d.]+) - alpha\)", source)
    if threshold is None or high is None or low is None:
        raise SystemExit("gen_smd_radii: the oxygen rule in smd_radii is not recognised")
    return {
        "overrides": overrides,
        "oxygen": {
            "threshold_alpha": float(threshold.group(1)),
            "radius_high_angstrom": float(high.group(1)),
            "slope_angstrom": float(low.group(2)),
            "offset_alpha": float(low.group(3)),
        },
    }


def build_rows() -> dict:
    radii_source = io.open(RADII_RAW, encoding="utf-8").read()
    rule_source = io.open(RULE_RAW, encoding="utf-8").read()
    vdw = parse_vdw(radii_source)
    if len(vdw) < 37:
        raise SystemExit(f"gen_smd_radii: VDW has only {len(vdw)} entries")
    rule = parse_rule(rule_source)
    overrides = rule["overrides"]
    # eq. 16 的 `radii_table[8]` 由 α 决定，不落成常数覆盖项。
    overrides.pop(8, None)
    # `vdw[0]` 是 ghost（`unknown`），所以元素 Z 用 `vdw[Z]`、符号表从 Z=1 开始。
    base = {symbol: vdw[number] for number, symbol in enumerate(ELEMENTS, start=1)}
    unknown = int(round(overrides.get(1, 0.0)))  # 占位，避免未使用告警
    del unknown
    rows = {
        "schema_version": 1,
        "description": (
            "SMD parameterised radii (eq. 16): a per-element base table plus the SMD overrides "
            "and the oxygen rule that depends on the solvent Abraham alpha. Distinct from the "
            "CDS vdW radii."
        ),
        "generated_by": "scripts/gen_smd_radii.py",
        "upstream": [
            {
                "file": "pyscf/data/radii.py",
                "url": "https://github.com/pyscf/pyscf/blob/master/pyscf/data/radii.py",
                "pyscf_version": PYSCF_VERSION,
                "sha256": sha256(RADII_RAW),
                "license": "Apache-2.0",
            },
            {
                "file": "pyscf/solvent/smd.py (smd_radii)",
                "url": "https://github.com/pyscf/pyscf/blob/master/pyscf/solvent/smd.py",
                "pyscf_version": PYSCF_VERSION,
                "sha256": sha256(RULE_RAW),
                "license": "Apache-2.0",
            },
        ],
        "base_angstrom": base,
        "overrides_angstrom": {
            str(index): value for index, value in sorted(overrides.items())
        },
        "oxygen": rule["oxygen"],
        "note": (
            "base_angstrom is PySCF's radii.VDW array (values in angstrom); overrides_angstrom "
            "are the eq. 16 element-wise SMD radii, keyed by nuclear charge; oxygen follows "
            "R = radius_high for alpha >= threshold, else R = radius_high + slope*(offset - alpha)"
        ),
    }
    return rows


def compare(want, got, where: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(want, dict):
        for key, value in want.items():
            if key not in got:
                problems.append(f"{where}.{key}: missing")
                continue
            problems += compare(value, got[key], f"{where}.{key}")
    elif isinstance(want, list):
        if not isinstance(got, list) or len(want) != len(got):
            return [f"{where}: list mismatch"]
        for index, (a, b) in enumerate(zip(want, got)):
            problems += compare(a, b, f"{where}[{index}]")
    elif isinstance(want, float):
        if not isinstance(got, (int, float)) or float(want) != float(got):
            problems.append(f"{where}: {want!r} vs {got!r}")
    elif want != got:
        problems.append(f"{where}: {want!r} vs {got!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path in (RADII_RAW, RULE_RAW):
        if not os.path.isfile(path):
            print(f"missing {path}")
            return 2
    rows = build_rows()
    if args.check:
        on_disk = json.load(io.open(OUTPUT, encoding="utf-8"))
        problems = compare(rows, on_disk)
        if problems:
            print(f"{len(problems)} 处不一致")
            for problem in problems[:10]:
                print("  -", problem)
            return 1
        print(f"ok: {os.path.relpath(OUTPUT, ROOT)} matches the upstream sources")
        return 0
    io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}")
    print(f"  SMD overrides: {rows['overrides_angstrom']}")
    print(f"  oxygen rule: {rows['oxygen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
