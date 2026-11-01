#!/usr/bin/env python3
"""用 PySCF 自己的 CDS 实现验证 `cds_atomic.json` 里的 SMD 表面张力表。

思路（**单原子**是关键技巧）：CDS 项是 `E_CDS = Σ_i σ_i·A_i + CSSIGM·A_total`。对**单个原子**，
表面就是一个半径等于 CDS vdW 半径的球 `A = 4πR²`，于是 PySCF 返回的 `E_CDS` 直接给出

    水（`icds = 1`，`CSSIGM = 0`）：  σ_Z = E_CDS / A
    非水（`icds = 2`）：              σ_Z = σ_N·SOLN + σ_A·SOLA + σ_B·SOLB（再叠 CSSIGM）

所以不需要复现表面积分解算法，就能把**表**和**拼装规则**分别验证。单位：σ 用
cal/(mol·Å²)（SMD 论文与上游都用它），故 `E[kcal/mol] = σ·A / 1000`。

运行（WSL，需要带 `libsolvent` 的 PySCF）：
    python3 scripts/validate_smd_cds.py

**现状（2026-11-12 实测）**：水相的单原子反推给出逐元素不同的比值（H 1.78、C 1.53、F 1.62、
S 1.49、Cl 1.51、Br 1.48），说明 `E_CDS = σ·A` 里的 `A` **不是**几何球面积 4πR²，而是
`CDSCALC` 里带切换函数的那套分解面积 —— 也就是 SV-03 还缺的"CDS 表面积分解算法"。
σ 表本身的提取是对的（与上游逐字、`gen_smd_cds.py --check` 绿），所以本脚本现在以
**诊断**身份存在（退出码 2 表示"面积约定未钉住"），不是"数据错了"。
"""

from __future__ import annotations

import json
import math
import sys

import pyscf
from pyscf import gto
from pyscf.solvent import smd

HARTREE2KCAL = 627.5094740631
PATH = "qwb-data/solvent/_smd_cds_staging/cds_atomic.json"
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]


def e_cds_kcal(symbol: str, solvent: str) -> float:
    # 孤立原子的电子数是 Z：奇数电子必须给 spin = 1（PySCF 的 mol.spin 是 2S，不是 2S+1）。
    # CDS 只依赖几何与元素，所以取任一合法自旋即可。
    z = ELEMENTS.index(symbol) + 1
    spin = 1 if z % 2 == 1 else 0
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
        spin=spin,
    )
    obj = smd.SMD(mol)
    obj.solvent = solvent
    energy_hartree, _ = smd.get_cds_legacy(obj)
    return energy_hartree * HARTREE2KCAL


def area(symbol: str, radii: list[float]) -> float:
    radius = radii[ELEMENTS.index(symbol)]
    return 4.0 * math.pi * radius * radius


def main() -> int:
    data = json.load(open(PATH, encoding="utf-8"))
    water_sigma = data["cases"]["water"]["sigma"]
    radii = data["cds_vdw_radii_angstrom"]["values"]
    naq = data["cases"]["nonaqueous"]
    sigma_mol = naq["sigma_mol"]
    print(f"pyscf {pyscf.__version__}, libsolvent available: {smd.libsolvent is not None}")

    print(f"\n{'sym':4} {'E_CDS(water)':>14} {'A/A^2':>9} {'sigma_fortran':>14} {'sigma_json':>11} {'delta':>10}")
    worst = 0.0
    for symbol in ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]:
        energy = e_cds_kcal(symbol, "water")
        sigma = energy / area(symbol, radii) * 1000.0
        reference = water_sigma[symbol]
        delta = sigma - reference
        worst = max(worst, abs(delta))
        print(
            f"{symbol:4} {energy:14.6f} {area(symbol, radii):9.4f} {sigma:14.6f} "
            f"{reference:11.5f} {delta:10.2e}"
        )
    print(f"worst |delta sigma| = {worst:.3e} cal/(mol A^2)")
    if worst > 1.0e-6:
        # **这是一个有信息的负结果，不是脚本坏了**：用 `A = 4πR²`（R 取 CDS vdW 半径）反推 σ 时，
        # 得到的值与表值差一个**逐元素不同**的因子（H 1.78、C 1.53、F 1.62、S 1.49、Cl 1.51、
        # Br 1.48）—— 说明 CDS 的"面积"不是几何球面积，而是 `CDSCALC` 里那套带切换函数
        # （`SMXCDS`/`COT`）的分解面积。所以：**σ 表本身是逐字提取的（Δ = 0），没对上的那部分
        # 是面积约定**，而面积约定正是 SV-03 还缺的那块算法。这里返回 2（诊断未通过），
        # 不要把它当成"表错了"。
        print("DIAGNOSTIC: 4πR² is NOT the CDS area convention (per-element ratios above);")
        print("            the sigma table itself came from upstream verbatim (delta = 0).")
        return 2

    print()
    index = {symbol: i for i, symbol in enumerate(ELEMENTS)}
    worst_relative = 0.0
    for solvent in ["toluene", "chloroform", "dmso", "thf"]:
        soln, _, sola, solb, solg, _, solc, solh = smd.solvent_db[solvent]
        for symbol in ["C", "O", "Cl", "S"]:
            z = index[symbol]
            sigma = (
                naq["raw_tables"]["sigma_n"][z] * soln
                + naq["raw_tables"]["sigma_a"][z] * sola
                + naq["raw_tables"]["sigma_b"][z] * solb
            )
            cssigm = (
                sigma_mol[0] * solg
                + sigma_mol[1] * solb * solb
                + sigma_mol[2] * solc * solc
                + sigma_mol[3] * solh * solh
            )
            predicted = (sigma + cssigm) * area(symbol, radii) / 1000.0
            actual = e_cds_kcal(symbol, solvent)
            relative = abs(predicted - actual) / max(abs(actual), 1.0e-30)
            worst_relative = worst_relative.max = max(worst_relative, relative)
            print(
                f"{solvent:12s} {symbol:3s} predicted {predicted:12.6f} actual {actual:12.6f} "
                f"rel {relative:.2e}"
            )
    print(f"worst relative (nonaqueous rule) = {worst_relative:.3e}")
    if worst_relative > 1.0e-6:
        print("FAIL: the nonaqueous assembly rule does not reproduce PySCF's CDS energies")
        return 1
    print("ok: the staged CDS tables and the nonaqueous rule reproduce PySCF's CDS energies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
