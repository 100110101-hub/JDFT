"""ST-01 的外部参考：**完整 TDDFT（RPA / TDHF）**的激发能与振子强度（PySCF 2.14）。

与 `pyscf_tda_reference.py` 的唯一区别是 `tdscf.rhf.TDH`（含 B 块与去激发耦合）而不是
`tdscf.rhf.TDA`。同一几何、同一（从 `qwb-data` 读的）基组，所以两边可以直接逐态比。

输出：qwb-core/crates/qwb-response/tests/reference/tddft_excitations.csv
列：species,basis,method,n_states,index,energy_hartree,energy_ev,oscillator_strength,note,
    source_doi,source_url
"""

import argparse
import io
import os
import sys

import numpy as np
import reference_sources
import yaml
from pyscf import gto, scf, tdscf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "tddft_excitations.csv"
)

FIELDS = (
    "species",
    "basis",
    "method",
    "n_states",
    "index",
    "energy_hartree",
    "energy_ev",
    "oscillator_strength",
    "note",
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
    td = tdscf.rhf.TDHF(mf)
    td.nstates = n_states
    td.kernel()
    energies = np.asarray(td.e)
    oscillators = np.asarray(td.oscillator_strength())
    print(f"{species}/{basis_name}: TDH n_states={n_states}")
    for index, (energy, oscillator) in enumerate(zip(energies, oscillators)):
        print(
            f"  {index}: {energy:.8f} Ha = {energy * HARTREE_TO_EV:8.4f} eV, "
            f"f = {oscillator:.6f}"
        )
    return [
        dict(
            species=species,
            basis=basis_name,
            method="tddft",
            n_states=n_states,
            index=index,
            energy_hartree=f"{energy:.10f}",
            energy_ev=f"{energy * HARTREE_TO_EV:.6f}",
            oscillator_strength=f"{oscillator:.8f}",
            note="PySCF 2.14.0 RHF + tdscf.rhf.TDHF（完整 RPA，含 B 块；单重态）",
        )
        for index, (energy, oscillator) in enumerate(zip(energies, oscillators))
    ]


def main(check_only: bool = False):
    rows = []
    rows += run("H2O", WATER, "sto-3g", 5)
    rows += run("H2O", WATER, "def2-svp", 5)
    rows = reference_sources.stamp(rows)
    fields = list(FIELDS)
    status = reference_sources.check_csv(CSV_PATH, fields, rows)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(CSV_PATH, fields, rows)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="只重算并比对文件，不写盘（DA-08）")
    arguments = parser.parse_args()
    sys.exit(main(arguments.check))
