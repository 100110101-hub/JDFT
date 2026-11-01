#!/usr/bin/env python3
"""用官方 dftd4（Python 绑定，黑盒）生成 **D4** 的参考能量、梯度与中间量（DA-06 的剩余一半）。

**为什么是黑盒**：D4 与 D3 一样，验收判据是「与官方参考实现在同一批二聚体上一致」，所以
参考值必须来自官方实现（`dftd4.interface`），不能是"我们自己的公式再算一遍"—— 后者只能
证明代码与它自己一致。本脚本同时导出 CN、EEQ 电荷与极化率，使后续 Rust 实现可以**逐层**
对照（先 CN，再电荷，最后 C6/能量），而不是只看一个总数。

来源（逐行写进 CSV）：
- `source_doi = 10.1063/1.5090222`：E. Caldeweyher, C. Bannwarth, S. Grimme,
  "Extension of the D3 dispersion coefficient model", J. Chem. Phys. 147, 034112 (2017)
  以及 E. Caldeweyher, J.-M. Mewes, S. Ehlert, S. Grimme,
  "Extension of the D3 dispersion coefficient model" 之后继工作
  "A generally applicable atomic-charge dependent London dispersion correction",
  J. Chem. Phys. 150, 154122 (2019)。**D4 的正式出处是后者**；本文件用 10.1063/1.5090222。
- `source_url = https://github.com/dftd4/dftd4`（本机装的是 pip 包 `dftd4` 4.3.1，
  其 C 库版本写进 `generator_version` 列）。

几何：与 D3 参考**同一份** `d3_dimers.csv`（理想化几何，`geometry = idealized`），
两个校正、两个代码读同一组坐标，比较的是实现而不是几何来源。

覆盖的泛函：`b3lyp`、`pbe`、`pbe0`、`hf`、`m06l`。**M06-2X 不在其中**：dftd4 4.3.1 的
`Functional 'm06-2x' not known`（实测，见 `docs/choices.md`），上游没有它的 D4 参数，
所以我们不编造一组。

电荷依赖：D4 与 D3 最大的不同是参数与 EEQ 电荷有关，所以对 `water_dimer` 额外算
`q = ±1` 的两行（同一几何、不同总电荷 → 不同 EEQ 电荷 → 不同能量），把"电荷相关"这条
路径钉住；其余二聚体只算 `q = 0`。

用法（WSL，需要 `pip install dftd4`；Windows 侧读同一份 CSV 做校验）：
    python3 scripts/dftd4_dispersion_reference.py            # 生成/覆盖两份 CSV
    python3 scripts/dftd4_dispersion_reference.py --check    # 重新生成并与磁盘逐字比较
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_sources import check_csv, stamp, write_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOMETRY = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3_dimers.csv"
)
DISPERSION = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d4_dispersion.csv"
)
PROPERTIES = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d4_properties.csv"
)

D4_DOI = "10.1063/1.5090222"
D4_URL = "https://github.com/dftd4/dftd4"

#: Å→Bohr：**用 QWB 的 K2 常数**（`qwb_engine_api::molecule::BOHR_PER_ANGSTROM`），
#: 这样两边看到的是同一组 Bohr 坐标（与 `dftd3_dispersion_reference.py` 同一个约定）。
BOHR_PER_ANGSTROM = 1.889726089838572

#: dftd4 4.3.1 认的泛函名（实测：`m06-l` 要写成 `m06l`；`m06-2x` 不存在）。
METHODS = ["b3lyp", "pbe", "pbe0", "hf", "m06l"]

#: 电荷依赖的采样（其余二聚体只算 q = 0）。
CHARGED_CASE = "water_dimer"
CHARGED_VALUES = [-1.0, 1.0]

DISPERSION_FIELDS = [
    "dimer",
    "method",
    "charge",
    "n_atoms",
    "geometry",
    "energy_hartree",
    "gradient_max_abs_hartree_per_bohr",
]

PROPERTY_FIELDS = [
    "dimer",
    "atom_index",
    "element",
    "charge",
    "coordination_number",
    "eeq_charge",
    "polarizability",
    "c6_to_first_atom",
]


def read_geometry() -> dict[str, list[dict[str, str]]]:
    """`d3_dimers.csv` → {dimer: [atom rows]}（保持文件里的原子顺序）。"""
    dimers: dict[str, list[dict[str, str]]] = {}
    with open(GEOMETRY, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dimers.setdefault(row["dimer"], []).append(row)
    return dimers


def positions_and_numbers(atoms: list[dict[str, str]]):
    import numpy as np  # noqa: PLC0415

    from pyscf.data.elements import charge as element_charge  # noqa: PLC0415

    numbers = np.array([int(element_charge(atom["element"])) for atom in atoms])
    positions = np.array(
        [
            [
                float(atom["x_angstrom"]) * BOHR_PER_ANGSTROM,
                float(atom["y_angstrom"]) * BOHR_PER_ANGSTROM,
                float(atom["z_angstrom"]) * BOHR_PER_ANGSTROM,
            ]
            for atom in atoms
        ]
    )
    return numbers, positions


def dispersion_rows() -> list[dict[str, str]]:
    import numpy as np  # noqa: PLC0415
    from dftd4.interface import DampingParam, DispersionModel  # noqa: PLC0415

    dimers = read_geometry()
    rows = []
    for name, atoms in dimers.items():
        numbers, positions = positions_and_numbers(atoms)
        charges = [0.0] + (CHARGED_VALUES if name == CHARGED_CASE else [])
        for charge in charges:
            model = DispersionModel(numbers, positions, charge=charge)
            for method in METHODS:
                result = model.get_dispersion(DampingParam(method=method), grad=True)
                gradient = np.asarray(result["gradient"])
                rows.append(
                    {
                        "dimer": name,
                        "method": method,
                        "charge": f"{charge:+.1f}",
                        "n_atoms": len(atoms),
                        "geometry": atoms[0]["geometry"],
                        "energy_hartree": f"{float(result['energy']):.12f}",
                        "gradient_max_abs_hartree_per_bohr": f"{float(np.abs(gradient).max()):.12e}",
                    }
                )
    return stamp(rows, D4_DOI, D4_URL)


def property_rows() -> list[dict[str, str]]:
    import numpy as np  # noqa: PLC0415
    from dftd4.interface import DispersionModel  # noqa: PLC0415

    dimers = read_geometry()
    rows = []
    for name, atoms in dimers.items():
        numbers, positions = positions_and_numbers(atoms)
        model = DispersionModel(numbers, positions, charge=0.0)
        properties = model.get_properties()
        cn = np.asarray(properties["coordination numbers"])
        charges = np.asarray(properties["partial charges"])
        polar = np.asarray(properties["polarizabilities"])
        c6 = np.asarray(properties["c6 coefficients"])
        for index, atom in enumerate(atoms):
            rows.append(
                {
                    "dimer": name,
                    "atom_index": index + 1,
                    "element": atom["element"],
                    "charge": "+0.0",
                    "coordination_number": f"{float(cn[index]):.10f}",
                    "eeq_charge": f"{float(charges[index]):.10f}",
                    "polarizability": f"{float(polar[index]):.10f}",
                    "c6_to_first_atom": f"{float(c6[index, 0]):.10f}",
                }
            )
    return stamp(rows, D4_DOI, D4_URL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    dispersion = dispersion_rows()
    properties = property_rows()
    if args.check:
        status = check_csv(DISPERSION, DISPERSION_FIELDS, dispersion)
        return status | check_csv(PROPERTIES, PROPERTY_FIELDS, properties)

    write_csv(DISPERSION, DISPERSION_FIELDS, dispersion)
    write_csv(PROPERTIES, PROPERTY_FIELDS, properties)
    print(f"wrote {len(dispersion)} dispersion rows and {len(properties)} property rows")
    for row in dispersion[: len(METHODS)]:
        print(
            f"  {row['dimer']:16s} {row['method']:6s} q={row['charge']} "
            f"E = {row['energy_hartree']} Ha"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
