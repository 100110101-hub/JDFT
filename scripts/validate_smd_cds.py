#!/usr/bin/env python3
"""用 PySCF 自己的 CDS 实现验证 `cds_atomic.json`：σ 表、非水拼装规则、**面积约定**。

用**单原子**把面积消掉（一个原子就是一个球），于是三件事都能独立验证：

    水（`icds = 1`，`CSSIGM = 0`）：  σ_Z = E_CDS / A,   A = 4π(R_Z + 0.4)²
    非水（`icds = 2`）：              σ_Z = σ_N·SOLN + σ_A·SOLA + σ_B·SOLB，再叠 CSSIGM·A

`0.4 Å` 这个偏移**不是抄来的**：先用 `4πR²` 反推时 σ 逐元素偏大 1.48~1.78 倍
（H 1.78、C 1.53、F 1.62、S 1.49、Cl 1.51、Br 1.48），换成 `4π(R+0.4)²` 之后水相与五个
非水溶剂一起回到 1e-7 量级 —— 所以 `A = 4π(R+0.4)²` 是**被这份脚本验证的约定**。

运行（WSL，需要带 `libsolvent` 的 PySCF；本机 2.14.0 已具备）：
    python3 scripts/validate_smd_cds.py
"""

from __future__ import annotations

import json
import math
import pathlib
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
# 只列 `smd.solvent_db` 里确实存在的溶剂（`thf`/`dmso` 在这个版本里用的是别的键名）。
NONAQUEOUS = ["toluene", "chloroform", "methanol", "acetonitrile", "ethanol"]


def solvent_descriptors_from_repo(name: str) -> dict[str, float]:
    """从**本仓库**的 `qwb-data/solvent/<name>.yaml` 读六个 SMD 描述符。

    这里刻意不用 PySCF 的 `solvent_db`：要验证的是"我们数据文件里的那六个数、叫那六个名字"
    能不能复现官方 CDS 能量。`mnsol.F` 的参数名与本仓库字段名的对应关系是

        SOLA = abraham_alpha, SOLB = abraham_beta, SOLG = gamma_cal_per_mol_angstrom2,
        SOLC = phi_sq,        SOLH = psi_sq,        SOLN = refractive_index_20c

    （PySCF 的 `solvent_db` 一行是 `[n20, n25, alpha, beta, gamma, epsilon, phi^2, psi^2]`，
    与 `get_cds_legacy` 里的解包顺序一致。）
    """
    import yaml  # noqa: PLC0415

    text = pathlib.Path(f"qwb-data/solvent/{name}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    smd_block = data["smd"]
    return {
        "SOLA": smd_block["abraham_alpha"],
        "SOLB": smd_block["abraham_beta"],
        "SOLG": smd_block["gamma_cal_per_mol_angstrom2"],
        "SOLC": smd_block["phi_sq"],
        "SOLH": smd_block["psi_sq"],
        "SOLN": smd_block["refractive_index_20c"],
    }


def e_cds_kcal(symbol: str, solvent: str) -> float:
    z = ELEMENTS.index(symbol) + 1
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
        spin=1 if z % 2 == 1 else 0,
    )
    obj = smd.SMD(mol)
    obj.solvent = solvent
    energy_hartree, _ = smd.get_cds_legacy(obj)
    return energy_hartree * HARTREE2KCAL


def main() -> int:
    data = json.load(open(PATH, encoding="utf-8"))
    offset = data["cds_area_convention"]["offset_angstrom"]
    sigma = data["cases"]["water"]["sigma"]
    radii = data["cds_vdw_radii_angstrom"]["values"]
    naq = data["cases"]["nonaqueous"]
    sigma_mol = naq["sigma_mol"]
    print(f"pyscf {pyscf.__version__}, libsolvent: {smd.libsolvent is not None}, offset = {offset} A")

    def area(symbol: str) -> float:
        return 4.0 * math.pi * (radii[ELEMENTS.index(symbol)] + offset) ** 2

    worst_sigma = 0.0
    for symbol in ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]:
        derived = e_cds_kcal(symbol, "water") / area(symbol) * 1000.0
        delta = derived - sigma[symbol]
        worst_sigma = max(worst_sigma, abs(delta))
        print(
            f"  water {symbol:3} sigma_derived={derived:11.6f} table={sigma[symbol]:11.5f} "
            f"delta={delta:+.2e}"
        )
    print(f"water: worst |delta sigma| = {worst_sigma:.3e} cal/(mol A^2)")

    worst_energy = 0.0
    for solvent in NONAQUEOUS:
        # **用本仓库的溶剂数据**（不是 PySCF 的库）：这样这条验证同时钉住"我们的六个描述符
        # 与上游同一个意思"。
        descriptors = solvent_descriptors_from_repo(solvent)
        soln = descriptors["SOLN"]
        sola = descriptors["SOLA"]
        solb = descriptors["SOLB"]
        solg = descriptors["SOLG"]
        solc = descriptors["SOLC"]
        solh = descriptors["SOLH"]
        for symbol in ["C", "O", "Cl", "S"]:
            index = ELEMENTS.index(symbol)
            sigma_i = (
                naq["raw_tables"]["sigma_n"][index] * soln
                + naq["raw_tables"]["sigma_a"][index] * sola
                + naq["raw_tables"]["sigma_b"][index] * solb
            )
            cssigm = (
                sigma_mol[0] * solg
                + sigma_mol[1] * solb * solb
                + sigma_mol[2] * solc * solc
                + sigma_mol[3] * solh * solh
            )
            predicted = (sigma_i + cssigm) * area(symbol) / 1000.0
            actual = e_cds_kcal(symbol, solvent)
            worst_energy = max(worst_energy, abs(predicted - actual))
    print(f"nonaqueous: worst |delta E_CDS| = {worst_energy:.3e} kcal/mol over {NONAQUEOUS}")

    if worst_sigma < 1.0e-3 and worst_energy < 1.0e-6:
        print("ok: sigma table, nonaqueous rule and A = 4*pi*(R+0.4)^2 all reproduce PySCF")
        return 0
    print("FAIL: the staged CDS data does not reproduce PySCF's CDS energies")
    return 1


if __name__ == "__main__":
    sys.exit(main())
