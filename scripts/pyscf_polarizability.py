"""RS-01 的外部参考：H2O 的静态极化率（PySCF，CPHF）。

输出：qwb-core/crates/qwb-response/tests/reference/polarizability.csv
列：species,basis,method,alpha_xx,alpha_yy,alpha_zz,alpha_iso,note
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import gto, scf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "polarizability.csv"
)

WATER = """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
"""


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
    rows = []
    for basis_name in ["sto-3g", "def2-svp"]:
        mol = gto.M(atom=WATER, unit="Angstrom", verbose=0)
        mol.basis = load_yaml_basis(basis_name)
        mol.cart = False
        mol.build()
        mf = scf.RHF(mol)
        mf.conv_tol = 1.0e-12
        mf.kernel()
        # PySCF 的解析极化率（CPHF）
        alpha = mf.polarizability()[0] if isinstance(mf.polarizability(), tuple) else mf.polarizability()
        alpha = np.asarray(alpha)
        iso = np.trace(alpha) / 3.0
        print(f"H2O/{basis_name}：α = \n{alpha}\n  iso = {iso:.8f}")
        rows.append(
            dict(
                species="H2O",
                basis=basis_name,
                method="cphf",
                alpha_xx=f"{alpha[0][0]:.10f}",
                alpha_yy=f"{alpha[1][1]:.10f}",
                alpha_zz=f"{alpha[2][2]:.10f}",
                alpha_iso=f"{iso:.10f}",
                note="PySCF 2.14.0 RHF + mf.polarizability()（CPHF）",
            )
        )
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = ["species", "basis", "method", "alpha_xx", "alpha_yy", "alpha_zz", "alpha_iso", "note"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")


if __name__ == "__main__":
    sys.exit(main())
