"""RS-02 的苯参考：TDA（CIS）激发能（PySCF）。

输出：qwb-core/crates/qwb-response/tests/reference/tda_benzene.csv
列：species,basis,method,index,energy_hartree,energy_ev,oscillator_strength,note
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import gto, scf, tdscf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "tda_benzene.csv"
)

HARTREE_TO_EV = 27.211386245988

BENZENE = """
C  0.000000  1.394100  0.000000
C  1.207400  0.697050  0.000000
C  1.207400 -0.697050  0.000000
C  0.000000 -1.394100  0.000000
C -1.207400 -0.697050  0.000000
C -1.207400  0.697050  0.000000
H  0.000000  2.481500  0.000000
H  2.149000  1.240750  0.000000
H  2.149000 -1.240750  0.000000
H  0.000000 -2.481500  0.000000
H -2.149000 -1.240750  0.000000
H -2.149000  1.240750  0.000000
"""

BASE = "sto-3g"


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
    mol = gto.M(atom=BENZENE, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis(BASE)
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    e_hf = mf.kernel()
    print(f"C6H6/{BASE}: nao = {mol.nao}, E_HF = {e_hf:.10f}")
    td = tdscf.rhf.TDA(mf)
    td.nstates = 6
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
                species="C6H6",
                basis=BASE,
                method="tda",
                index=index,
                energy_hartree=f"{energy:.10f}",
                energy_ev=f"{energy * HARTREE_TO_EV:.6f}",
                oscillator_strength=f"{oscillator:.8f}",
                note="PySCF 2.14.0 RHF + tdscf.rhf.TDA（单重态）",
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
