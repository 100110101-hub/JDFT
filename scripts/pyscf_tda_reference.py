"""RS-02/RS-03 的外部参考：TDA/CIS 的激发能与振子强度（PySCF）。

输出：qwb-core/crates/qwb-response/tests/reference/tda_excitations.csv
列：species,basis,method,n_states,index,energy_hartree,energy_ev,oscillator_strength,note
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import gto, scf, tdscf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "tda_excitations.csv"
)

WATER = """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
"""

HARTREE_TO_EV = 27.211386245988


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


def run(species, geometry, basis_name, n_states):
    mol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis(basis_name)
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    td = tdscf.rhf.TDA(mf)
    td.nstates = n_states
    td.kernel()
    energies = np.asarray(td.e)
    oscillators = np.asarray(td.oscillator_strength())
    print(f"{species}/{basis_name}: n_states={n_states}")
    for index, (energy, oscillator) in enumerate(zip(energies, oscillators)):
        print(
            f"  {index}: {energy:.8f} Ha = {energy*HARTREE_TO_EV:8.4f} eV, "
            f"f = {oscillator:.6f}"
        )
    return [
        dict(
            species=species,
            basis=basis_name,
            method="tda",
            n_states=n_states,
            index=index,
            energy_hartree=f"{energy:.10f}",
            energy_ev=f"{energy * HARTREE_TO_EV:.6f}",
            oscillator_strength=f"{oscillator:.8f}",
            note="PySCF 2.14.0 RHF + tdscf.rhf.TDA（单重态）",
        )
        for index, (energy, oscillator) in enumerate(zip(energies, oscillators))
    ]


def main():
    rows = []
    rows += run("H2O", WATER, "sto-3g", 5)
    rows += run("H2O", WATER, "def2-svp", 5)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = [
        "species",
        "basis",
        "method",
        "n_states",
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
