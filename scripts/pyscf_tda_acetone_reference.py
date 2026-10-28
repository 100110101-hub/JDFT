"""VF-01 的丙酮参考：TDA（CIS）激发能（PySCF）。

输出：qwb-core/crates/qwb-response/tests/reference/tda_acetone.csv
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import gto, scf, tdscf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "tda_acetone.csv"
)

HARTREE_TO_EV = 27.211386245988

# 丙酮实验几何（Å）：C=O 1.222、C–C 1.507、∠CCC 116.5°，甲基按四面体摆放
ACETONE = """
C   0.000000  0.000000  0.000000
O   0.000000  0.000000  1.222000
C   1.348000  0.000000 -0.450000
C  -1.348000  0.000000 -0.450000
H   1.700000  0.890000  0.070000
H   1.700000 -0.890000  0.070000
H   2.000000  0.000000 -1.300000
H  -1.700000  0.890000  0.070000
H  -1.700000 -0.890000  0.070000
H  -2.000000  0.000000 -1.300000
"""

BASIS = "sto-3g"


def load_yaml_basis(name):
    with open(f"qwb-data/basis/{name}.yaml", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    out = {}
    for entry in data["shells"]:
        shells = []
        for shell in entry["shells"]:
            l = shell["l"]
            exponents = np.array(shell["exponents"], dtype=float)
            for row in np.array(shell["coefficients"], dtype=float):
                shells.append([l] + [[float(e), float(c)] for e, c in zip(exponents, row)])
        out[entry["element"]] = shells
    return out


def main():
    mol = gto.M(atom=ACETONE, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis(BASIS)
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    e_hf = mf.kernel()
    print(f"(CH3)2CO/{BASIS}: nao = {mol.nao}, n_occ = {mol.nelectron // 2}, E_HF = {e_hf:.10f}")
    td = tdscf.rhf.TDA(mf)
    td.nstates = 4
    td.kernel()
    energies = np.asarray(td.e)
    oscillators = np.asarray(td.oscillator_strength())
    rows = []
    for index, (energy, oscillator) in enumerate(zip(energies, oscillators)):
        print(
            f"  {index}: {energy:.8f} Ha = {energy * HARTREE_TO_EV:8.4f} eV, f = {oscillator:.6f}"
        )
        rows.append(
            dict(
                species="C3H6O",
                basis=BASIS,
                method="tda",
                index=index,
                energy_hartree=f"{energy:.10f}",
                energy_ev=f"{energy * HARTREE_TO_EV:.6f}",
                oscillator_strength=f"{oscillator:.8f}",
                note="PySCF 2.14.0 RHF + tdscf.rhf.TDA（单重态，丙酮实验几何近似）",
            )
        )
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = [
        "species",
        "basis",
        "method",
        "index",
        "energy_hartree",
        "energy_ev",
        "oscillator_strength",
        "note",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")


if __name__ == "__main__":
    sys.exit(main())
