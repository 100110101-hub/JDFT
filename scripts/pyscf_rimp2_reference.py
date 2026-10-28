"""SC-05 的外部参考：H2O/def2-SVP 的 canonical MP2 与 RI-MP2 相关能（PySCF）。

输出：qwb-core/crates/qwb-scf/tests/reference/rimp2_correlation.csv
列：species,basis,aux_basis,e_scf_hartree,e_corr_canonical,e_corr_ri,ri_error,note
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import df, gto, mp, scf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-scf", "tests", "reference", "rimp2_correlation.csv"
)

WATER = """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
"""

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


def run(label, geometry, aux_name):
    mol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis("def2-svp")
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    e_scf = mf.kernel()

    pt = mp.MP2(mf)
    e_corr_canonical, _ = pt.kernel()

    auxmol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
    auxmol.basis = load_yaml_basis(aux_name)
    auxmol.cart = False
    auxmol.build()
    pt_ri = mp.MP2(mf).density_fit(auxbasis=auxmol.basis)
    e_corr_ri, _ = pt_ri.kernel()
    print(
        f"{label}: nao={mol.nao}, naux={auxmol.nao}, E_SCF={e_scf:.10f}, "
        f"E_corr(canonical)={e_corr_canonical:.10f}, E_corr(RI)={e_corr_ri:.10f}, "
        f"RI error={e_corr_ri - e_corr_canonical:+.3e}"
    )
    return dict(
        species=label,
        basis="def2-svp",
        aux_basis=aux_name,
        e_scf_hartree=f"{e_scf:.12f}",
        e_corr_canonical=f"{e_corr_canonical:.12f}",
        e_corr_ri=f"{e_corr_ri:.12f}",
        ri_error=f"{e_corr_ri - e_corr_canonical:.6e}",
        note="PySCF 2.14.0 RHF + mp.MP2（无冻结核；RI 版本用 density_fit 与 def2-svp-ri）",
    )


def main():
    rows = [
        run("H2O", WATER, "def2-svp-ri"),
        run("C6H6", BENZENE, "def2-svp-ri"),
    ]
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = [
        "species",
        "basis",
        "aux_basis",
        "e_scf_hartree",
        "e_corr_canonical",
        "e_corr_ri",
        "ri_error",
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
