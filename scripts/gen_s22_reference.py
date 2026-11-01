#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 **S22** 弱相互作用二聚体集（几何 + CCSD(T)/CBS CP 参考相互作用能）落成参考 CSV。

# 为什么用它、为什么不是从 PDF 抄

M2 的第 ④ 条判据要"苯二聚体的相互作用能与文献差 < 1 kcal/mol"，而"文献"必须落到具体
几何 + 具体数值上。几何与数值都在 Jurečka 等的 S22 集里；本脚本从 **ASE** 随包分发的
`ase/data/s22.py`（LGPL-2.1-or-later，文件头保留原文与引用）逐字取出 —— 与从 PDF 手抄相比，
好处是**可程序化重算、可 sha256 钉住**，不必人工核对小数点。

S22 的参考相互作用能在 ASE 里沿用 Takatani 等（J. Chem. Phys. 132, 144104 (2010)，
doi:10.1063/1.3378024）的"基组一致修订值"，**单位是 eV**（`get_interaction_energy_cc`
的 docstring 明写 "in eV"）；原始 Jurečka 值见 doi:10.1039/b600027d。CSV 里同时保留 eV 与
kcal/mol 两个数（4.184 J/cal 精确换算），避免下游各自换算时引入不同的常数。

# 解析方式

只对源码里的 `data = {...}` 与 `s22 = [...]` 两个**字面量**赋值做 `ast.literal_eval` ——
不 `exec`、不 import ASE（本机没装），因此脚本能在任何 Python 3 上跑，且**只认数据**。

运行：
    python3 scripts/gen_s22_reference.py            # 写 CSV + staging JSON
    python3 scripts/gen_s22_reference.py --check    # 与磁盘逐字比对
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_sources  # noqa: E402

ROOT = reference_sources.ROOT
STAGING = os.path.join(ROOT, "qwb-data", "reference", "_s22_staging")
RAW = os.path.join(STAGING, "raw", "s22.py")
PROVENANCE = os.path.join(STAGING, "s22.json")
OUTPUT = os.path.join(
    ROOT, "qwb-core/crates/qwb-engine/tests/reference/s22_dimers.csv"
)
FIELDS = (
    "system",
    "n_atoms",
    "atom_index",
    "symbol",
    "x",
    "y",
    "z",
    "e_int_ev",
    "e_int_kcal_per_mol",
)

#: eV → kcal/mol（1 cal = 4.184 J 精确；1 eV = 1.602176634e-19 J）。
EV_TO_KCAL = 23.060548867

S22_DOI = reference_sources.S22_DOI
S22_URL = reference_sources.S22_URL


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise SystemExit(f"gen_s22_reference: no top-level `{name} = ...` literal")


def build_rows() -> list[dict[str, str]]:
    source = io.open(RAW, encoding="utf-8").read()
    names = _literal_assignment(source, "s22")
    data = _literal_assignment(source, "data")
    if len(names) != 22:
        raise SystemExit(f"gen_s22_reference: expected 22 dimers, got {len(names)}")
    missing = [name for name in names if name not in data]
    if missing:
        raise SystemExit(f"gen_s22_reference: no data for {missing}")

    rows: list[dict[str, str]] = []
    for name in names:
        entry = data[name]
        symbols = entry["symbols"]
        positions = entry["positions"]
        if len(symbols) != len(positions):
            raise SystemExit(f"gen_s22_reference: {name}: symbols/positions length mismatch")
        energy_ev = float(entry["interaction energy CC"])
        if energy_ev > 0.0:
            raise SystemExit(
                f"gen_s22_reference: {name}: interaction energy must be attractive, "
                f"got {energy_ev} eV"
            )
        for index, (symbol, position) in enumerate(zip(symbols, positions)):
            rows.append(
                {
                    "system": name,
                    "n_atoms": str(len(symbols)),
                    "atom_index": str(index),
                    "symbol": symbol,
                    "x": f"{position[0]:.7f}",
                    "y": f"{position[1]:.7f}",
                    "z": f"{position[2]:.7f}",
                    "e_int_ev": f"{energy_ev:.10g}",
                    "e_int_kcal_per_mol": f"{energy_ev * EV_TO_KCAL:.10g}",
                }
            )
    return reference_sources.stamp(rows, S22_DOI, S22_URL)


def build_provenance() -> dict:
    digest = hashlib.sha256(io.open(RAW, "rb").read()).hexdigest()
    return {
        "schema_version": 1,
        "description": (
            "S22 weakly interacting dimers: geometries and CCSD(T)/CBS CP interaction "
            "energies, extracted from ASE's data/s22.py (which in turn cites Jurecka 2006 "
            "for the geometries and Takatani 2010 for the revised energies)."
        ),
        "generated_by": "scripts/gen_s22_reference.py",
        "upstream": {
            "file": "ase/data/s22.py",
            "url": "https://gitlab.com/ase/ase/-/raw/master/ase/data/s22.py",
            "license": "LGPL-2.1-or-later",
            "sha256": digest,
        },
        "citations": [
            {
                "role": "geometries (S22 set)",
                "doi": "10.1039/b600027d",
                "note": "Jurecka, Sponer, Cerny, Hobza, PCCP 8, 1985 (2006)",
            },
            {
                "role": "reference interaction energies (basis-set consistent revision)",
                "doi": "10.1063/1.3378024",
                "note": "Takatani, Hohenstein, Malagoli, Marshall, Sherrill, JCP 132, 144104 (2010); units eV",
            },
        ],
        "dos": (
            "the two DOIs were checked against api.crossref.org on 2026-11-13; the ASE file "
            "was fetched from the URL above and pinned by sha256"
        ),
    }


def main(argv: list[str]) -> int:
    table = build_rows()
    provenance = build_provenance()
    if "--check" in argv:
        problems = []
        on_disk = json.load(io.open(PROVENANCE, encoding="utf-8"))
        if on_disk != provenance:
            problems.append("staging JSON differs from the generator")
        code = reference_sources.check_csv(OUTPUT, FIELDS, table)
        if problems:
            print("\n".join(problems))
            return 1
        return code
    reference_sources.write_csv(OUTPUT, FIELDS, table)
    os.makedirs(STAGING, exist_ok=True)
    io.open(PROVENANCE, "w", encoding="utf-8", newline="\n").write(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    systems = {row["system"] for row in table}
    print(f"wrote {reference_sources.rel(OUTPUT)} ({len(table)} rows, {len(systems)} systems)")
    print(f"wrote {reference_sources.rel(PROVENANCE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
