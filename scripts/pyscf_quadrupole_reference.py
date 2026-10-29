"""RS-08 的黑箱基准：用 PySCF 算 H2O / STO-3G 的无迹四极矩（原子单位 e·bohr²）。

为什么用参考实现而不是文献值：判据写的是「与文献差 < 5%」，但本仓库没有可核对的
文献表格（且不同文献的 traceless 约定/单位/原点各不相同）。PySCF 的 `int1e_rr` 是
同一物理量的独立实现，把它当基准比抄一个记不清出处的数更可靠 —— 这条理由写进
`docs/choices.md` 六十九。

约定（与 `qwb-props::quadrupole_traceless_debye_angstrom` 一致）：
    Theta_ij = 1/2 * sum_a q_a (3 r_i r_j - r^2 delta_ij)，原点 = 全零
    电子：-Tr(P · <mu|r_i r_j|nu>)，核：+Z
输出 CSV：system,component,value（原子单位）。
"""
import csv
import os

import numpy as np
from pyscf import gto, scf

OUT = os.path.join(os.path.dirname(__file__), "..", "qwb-core", "crates", "qwb-props",
                   "tests", "reference", "quadrupole.csv")

SYSTEMS = {
    "water": (
        "O 0.0 0.0 0.11779; H 0.0 0.75545 -0.47116; H 0.0 -0.75545 -0.47116",
        0,
    ),
    "co2": ("C 0 0 0; O 0 0 1.160; O 0 0 -1.160", 0),
    "methane": (
        "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; "
        "H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629",
        0,
    ),
}

rows = []
for name, (atom, spin) in SYSTEMS.items():
    mol = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=spin,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    density = mf.make_rdm1()
    nao = mol.nao_nr()
    rr = mol.intor("int1e_rr", comp=9).reshape(3, 3, nao, nao)
    electronic = np.einsum("bij,xyij->bxy", density[None], rr)[0]
    nuclear = np.zeros((3, 3))
    for index, coord in enumerate(mol.atom_coords(unit="Bohr")):
        symbol = mol.atom_symbol(index)
        z = gto.mole.charge(symbol)
        for i in range(3):
            for j in range(3):
                nuclear[i][j] += z * coord[i] * coord[j]
    moment = nuclear - electronic
    trace = np.trace(moment)
    theta = 0.5 * (3.0 * moment - trace * np.eye(3))
    for i in range(3):
        for j in range(i, 3):
            rows.append((name, f"{i}{j}", f"{theta[i][j]:.12e}"))
    print(f"{name}: E = {mf.e_tot:.10f} Ha, Theta_zz = {theta[2][2]:+.10f} a.u.")

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w", newline="\n", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["system", "component", "value_au"])
    writer.writerows(rows)
print(f"wrote {OUT} ({len(rows)} rows)")
