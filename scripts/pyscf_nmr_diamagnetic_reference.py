"""Generate the RS-04 diamagnetic-shielding reference with PySCF.

sigma_dia(A) = 1/(3c^2) * sum_uv P_uv <u| 1/|r - R_A| |v>   [ppm]

PySCF's `int1e_rinv` is the 1/|r| attraction with the origin fixed at
(0, 0, 0) (verified: shifting an H atom to (5,5,5) bohr drops the value to
1/sqrt(75) = 0.11547).  So for each nucleus we rebuild the same molecule
translated by -R_A, which puts that nucleus on the origin; the basis function
ordering is unchanged, hence the same density matrix contracts it.

This is an independent path from the Rust implementation: PySCF evaluates the
1/r integrals itself, and the Rust side gets them from its own Gaussian
recursion via the unit-charge trick on the nuclear-attraction matrix.

Output: qwb-core/crates/qwb-props/tests/reference/nmr_diamagnetic.csv
"""
import io
import os

import numpy as np
from pyscf import gto, scf

C = 137.035999084
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-props", "tests", "reference", "nmr_diamagnetic.csv"
)

WATER = [
    ("O", [0.0, 0.0, 0.117790]),
    ("H", [0.0, 0.755450, -0.471160]),
    ("H", [0.0, -0.755450, -0.471160]),
]
D = 0.6290
METHANE = [
    ("C", [0.0, 0.0, 0.0]),
    ("H", [D, D, D]),
    ("H", [-D, -D, D]),
    ("H", [-D, D, -D]),
    ("H", [D, -D, -D]),
]


def shielding(label, atoms):
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    coords = mol.atom_coords(unit="Bohr")
    rows = []
    for index, ra in enumerate(coords):
        shifted = [[s, list(c - ra)] for s, c in zip([a[0] for a in atoms], coords)]
        mol2 = gto.M(atom=shifted, basis="sto-3g", unit="Bohr", verbose=0)
        expectation = np.einsum("ij,ij->", dm, mol2.intor("int1e_rinv"))
        ppm = expectation / (3.0 * C * C) * 1e6
        rows.append((label, index, atoms[index][0], expectation, ppm))
        print(f"  {label}[{index}] {atoms[index][0]}: <1/r_A>={expectation:.9f} "
              f"sigma_dia={ppm:.9f} ppm")
    return rows


all_rows = []
all_rows += shielding("water", WATER)
all_rows += shielding("methane", METHANE)

lines = ["system,atom_index,symbol,inverse_r_bohr,sigma_dia_ppm"]
for label, index, symbol, expectation, ppm in all_rows:
    lines.append(f"{label},{index},{symbol},{expectation:.12f},{ppm:.9f}")
io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(all_rows)} nuclei)")
