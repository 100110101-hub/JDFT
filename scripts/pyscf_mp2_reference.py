"""用 PySCF 生成 MP2 参考（RHF 参考、闭壳层，无冻结核）。

输出：qwb-core/crates/qwb-scf/tests/reference/mp2_correlation.csv
列：species,basis,e_scf_hartree,e_correlation_hartree,e_total_hartree,note
"""

import argparse
import csv
import os
import sys

import numpy as np
from pyscf import gto, scf, mp

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-scf", "tests", "reference", "mp2_correlation.csv"
)

MOLECULES = {
    "H2": """
H 0.0 0.0 -0.37
H 0.0 0.0 0.37
""",
    "H2O": """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
""",
}

BASES = ["sto-3g", "def2-svp", "def2-tzvp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = []
    for species, geometry in MOLECULES.items():
        for basis in BASES:
            mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0, charge=0, spin=0)
            mf = scf.RHF(mol)
            mf.conv_tol = 1.0e-11
            e_scf = mf.kernel()
            pt = mp.MP2(mf)
            e_corr, _ = pt.kernel()
            print(f"{species:4s} {basis:9s}: E_SCF = {e_scf:.10f}  E_corr = {e_corr:.10f}  "
                  f"E_tot = {e_scf + e_corr:.10f}")
            rows.append(
                dict(
                    species=species,
                    basis=basis,
                    e_scf_hartree=f"{e_scf:.12f}",
                    e_correlation_hartree=f"{e_corr:.12f}",
                    e_total_hartree=f"{e_scf + e_corr:.12f}",
                    note="PySCF 2.14.0 RHF + mp.MP2（无冻结核）",
                )
            )

    if args.check:
        return 0
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = ["species", "basis", "e_scf_hartree", "e_correlation_hartree", "e_total_hartree", "note"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
