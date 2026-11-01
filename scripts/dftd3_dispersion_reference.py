"""用官方 s-dftd3（Python 绑定，黑盒）生成 D3(BJ) 的参考能量与梯度（FX-04 / DA-06）。

**为什么是黑盒**：D3(BJ) 的验收判据是「与官方参考实现在 10 个二聚体上差 < 1e-3 kcal/mol」，
所以参考值必须来自官方实现本身（`dftd3.interface`），而不是我们自己的公式再算一遍 ——
后者只能证明「代码和它自己一致」。

来源（逐行写进 CSV，DA-08 的要求）：
- `source_doi = 10.1063/1.3382344`：S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
  "A consistent and accurate ab initio parametrization of density functional dispersion
  correction (DFT-D) for the 94 elements H-Pu", J. Chem. Phys. 132, 154104 (2010)。
  **2026-11-09 用 api.crossref.org/works/10.1063/1.3382344 核对过题名、卷号与年份。**
- `source_url = https://github.com/dftd3/simple-dftd3`（`main` 分支；本机装的是 pip 包
  `dftd3`，版本号写进 `generator_version` 列）。

几何：**理想化几何**（不是文献构象）—— 二聚体由「单体理想内坐标 + 固定平移」拼出来，
两个代码读同一份 `d3_dimers.csv`，所以比较的是实现而不是几何来源。这一点写在
`geometry` 列里（值恒为 `idealized`），不假装它是实验/优化构象。

用法（WSL，需要 pip 包 dftd3；Windows 侧读同一份 CSV 做校验）：
    python3 scripts/dftd3_dispersion_reference.py            # 生成/覆盖两份 CSV
    python3 scripts/dftd3_dispersion_reference.py --check    # 重新生成并与磁盘逐字比较
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_sources import check_csv, stamp, write_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOMETRY = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3_dimers.csv"
)
REFERENCE = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "d3_dispersion.csv"
)

SOURCE_DOI = "10.1063/1.3382344"
SOURCE_URL = "https://github.com/dftd3/simple-dftd3"

GEOMETRY_FIELDS = (
    "dimer",
    "atom",
    "element",
    "x_angstrom",
    "y_angstrom",
    "z_angstrom",
    "geometry",
)
REFERENCE_FIELDS = (
    "dimer",
    "method",
    "atom",
    "element",
    "energy_hartree",
    "gradient_x",
    "gradient_y",
    "gradient_z",
    "generator_version",
)

#: Å→Bohr：**用 QWB 的 K2 常数**（`qwb_engine_api::molecule::BOHR_PER_ANGSTROM`），
#: 这样两边看到的是同一组 Bohr 坐标；s-dftd3 内部的 mctc 常数比它旧 3.6e-8（相对），
#: 若各自换算会引入 ~1e-9 Ha 的伪差，掩盖我们真正要看的实现差异。
BOHR_PER_ANGSTROM = 1.889726089838572

#: 方法（QWB 侧同名的泛函；只有官方 BJ 参数表里有的这四个）。
METHODS = ("b3lyp", "pbe", "pbe0", "hf")

_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


def water():
    return [
        ("O", 0.0, 0.0, 0.0),
        ("H", 0.7570, 0.5860, 0.0),
        ("H", -0.7570, 0.5860, 0.0),
    ]


def ammonia():
    return [
        ("N", 0.0, 0.0, 0.0),
        ("H", 0.9390, 0.0, -0.3810),
        ("H", -0.4695, 0.8132, -0.3810),
        ("H", -0.4695, -0.8132, -0.3810),
    ]


def methane():
    d = 0.6276
    return [
        ("C", 0.0, 0.0, 0.0),
        ("H", d, d, d),
        ("H", -d, -d, d),
        ("H", -d, d, -d),
        ("H", d, -d, -d),
    ]


def hydrogen_fluoride():
    return [("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 0.9171)]


def carbon_dioxide():
    return [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.1630),
        ("O", 0.0, 0.0, -1.1630),
    ]


def benzene():
    cc = 1.3900
    ch = 1.0900
    atoms = []
    for index in range(6):
        angle = math.radians(60.0 * index)
        atoms.append(("C", cc * math.cos(angle), cc * math.sin(angle), 0.0))
    for index in range(6):
        angle = math.radians(60.0 * index)
        radius = cc + ch
        atoms.append(("H", radius * math.cos(angle), radius * math.sin(angle), 0.0))
    return atoms


def methanol():
    angle = math.radians(108.5)
    return [
        ("C", 0.0, 0.0, 0.0),
        ("O", 1.4300, 0.0, 0.0),
        ("H", 1.4300 + 0.9600 * math.cos(angle), 0.9600 * math.sin(angle), 0.0),
        ("H", -0.3600, 1.0300, 0.0),
        ("H", -0.3600, -0.5150, 0.8900),
        ("H", -0.3600, -0.5150, -0.8900),
    ]


def ethyne():
    return [
        ("C", 0.0, 0.0, 0.0),
        ("C", 0.0, 0.0, 1.2030),
        ("H", 0.0, 0.0, -1.0610),
        ("H", 0.0, 0.0, 2.2640),
    ]


def shifted(monomer, shift):
    return [
        (symbol, x + shift[0], y + shift[1], z + shift[2])
        for symbol, x, y, z in monomer
    ]


#: 10 个二聚体：名字 → 单体理想内坐标 + 平移（Å）。平移量取常见接触距离，
#: 目的是让色散项落在「既不是零也不发散」的区间。
DIMERS = [
    ("water_dimer", lambda: water() + shifted(water(), (2.980, 0.0, 0.0))),
    ("ammonia_dimer", lambda: ammonia() + shifted(ammonia(), (3.350, 0.0, 0.0))),
    ("methane_dimer", lambda: methane() + shifted(methane(), (3.800, 0.0, 0.0))),
    ("methanol_dimer", lambda: methanol() + shifted(methanol(), (2.850, 0.0, 0.0))),
    ("benzene_dimer_stacked", lambda: benzene() + shifted(benzene(), (0.0, 0.0, 3.500))),
    (
        "benzene_dimer_parallel_displaced",
        lambda: benzene() + shifted(benzene(), (1.400, 0.0, 3.500)),
    ),
    (
        "co2_dimer_slipped",
        lambda: carbon_dioxide() + shifted(carbon_dioxide(), (3.200, 0.0, 0.600)),
    ),
    ("hf_dimer", lambda: hydrogen_fluoride() + shifted(hydrogen_fluoride(), (2.790, 0.0, 0.0))),
    ("ethyne_dimer_stacked", lambda: ethyne() + shifted(ethyne(), (0.0, 3.300, 0.0))),
    ("methane_water_complex", lambda: methane() + shifted(water(), (3.500, 0.0, 0.0))),
]


def geometry_rows():
    rows = []
    for name, build in DIMERS:
        for index, (symbol, x, y, z) in enumerate(build(), start=1):
            rows.append(
                {
                    "dimer": name,
                    "atom": index,
                    "element": symbol,
                    "x_angstrom": f"{x:.6f}",
                    "y_angstrom": f"{y:.6f}",
                    "z_angstrom": f"{z:.6f}",
                    "geometry": "idealized",
                }
            )
    return rows


def dispersion_rows(geometry):
    """调官方 `dftd3` 算能量与梯度（梯度是坐标的解析导数，Hartree/Bohr）。"""
    import numpy as np  # noqa: PLC0415
    import dftd3  # noqa: PLC0415
    from dftd3.interface import DispersionModel, RationalDampingParam  # noqa: PLC0415

    version = dftd3.__version__
    rows = []
    by_dimer = {}
    for row in geometry:
        by_dimer.setdefault(row["dimer"], []).append(row)
    for name, _ in DIMERS:
        atoms = by_dimer[name]
        numbers = np.array([_Z[atom["element"]] for atom in atoms], dtype=int)
        positions = np.array(
            [
                [
                    float(atom["x_angstrom"]) * BOHR_PER_ANGSTROM,
                    float(atom["y_angstrom"]) * BOHR_PER_ANGSTROM,
                    float(atom["z_angstrom"]) * BOHR_PER_ANGSTROM,
                ]
                for atom in atoms
            ],
            dtype=float,
        )
        model = DispersionModel(numbers, positions)
        for method in METHODS:
            result = model.get_dispersion(RationalDampingParam(method=method), grad=True)
            energy = float(result.get("energy"))
            gradient = result.get("gradient")
            for index, atom in enumerate(atoms):
                gx, gy, gz = (float(gradient[index][axis]) for axis in range(3))
                rows.append(
                    {
                        "dimer": name,
                        "method": method,
                        "atom": atom["atom"],
                        "element": atom["element"],
                        "energy_hartree": f"{energy:.12e}",
                        "gradient_x": f"{gx:.12e}",
                        "gradient_y": f"{gy:.12e}",
                        "gradient_z": f"{gz:.12e}",
                        "generator_version": version,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    geometry = stamp(geometry_rows(), SOURCE_DOI, SOURCE_URL)
    if args.check:
        status = check_csv(GEOMETRY, GEOMETRY_FIELDS, geometry)
        reference = stamp(dispersion_rows(geometry), SOURCE_DOI, SOURCE_URL)
        return status | check_csv(REFERENCE, REFERENCE_FIELDS, reference)

    write_csv(GEOMETRY, GEOMETRY_FIELDS, geometry)
    reference = stamp(dispersion_rows(geometry), SOURCE_DOI, SOURCE_URL)
    write_csv(REFERENCE, REFERENCE_FIELDS, reference)
    print(f"wrote {GEOMETRY}（{len(geometry)} 行，{len(DIMERS)} 个二聚体）")
    print(f"wrote {REFERENCE}（{len(reference)} 行 = 原子数 × {len(METHODS)} 个方法）")
    for name, _ in DIMERS[:3]:
        energies = [row for row in reference if row["dimer"] == name]
        print(f"  {name}: E[{energies[0]['method']}] = {energies[0]['energy_hartree']} Ha")
    return 0


if __name__ == "__main__":
    sys.exit(main())
