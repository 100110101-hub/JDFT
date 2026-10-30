"""Generate the RS-04 angular-momentum reference with PySCF.

We need N_alpha = <mu| ((r - O) x nabla)_alpha |nu>  with L = -i N.

This PySCF build has no `int1e_angmom`, but `int1e_irp` works: it returns the
9 components of  <mu| i r_beta p_gamma |nu> = <mu| r_beta d/dgamma |nu>
(p = -i d), so

    N_x = irp[y,z] - irp[z,y],  N_y = irp[z,x] - irp[x,z],
    N_z = irp[x,y] - irp[y,x].

The component order of the 9-vector is fixed by the angular momentum algebra
[N_x, N_y] ~ -N_z (an exact identity only for a complete basis, but the wrong
order is off by an order of magnitude: see the printed residual).

The exponents are read from qwb-data/basis/sto-3g.yaml and injected into
PySCF, so both sides use identical basis data (PySCF's built-in STO-3G table
is truncated to 8 significant digits, which alone moves the integrals by
~1e-8).  The origin is fixed at the coordinate origin, so each origin is
reached by translating the molecule.

Output: qwb-core/crates/qwb-integrals/tests/reference/angmom.csv
"""
import io
import os
import re

import numpy as np
from pyscf import gto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "qwb-core", "crates", "qwb-integrals", "tests",
                   "reference", "angmom.csv")
YAML = os.path.join(ROOT, "qwb-data", "basis", "sto-3g.yaml")
BOHR = 1.889726089838572

WATER = [
    ("O", [0.0, 0.0, 0.117790]),
    ("H", [0.0, 0.755450, -0.471160]),
    ("H", [0.0, -0.755450, -0.471160]),
]


def load_basis():
    """Parse the subset of the qwb-data YAML we need: elements with segmented shells."""
    basis = {}
    element = None
    angular = None
    exponents = None
    for raw in io.open(YAML, encoding="utf-8"):
        line = raw.rstrip("\n")
        match = re.match(r"  - element: (\w+)$", line)
        if match:
            element = match.group(1)
            basis[element] = []
            continue
        match = re.match(r"      - l: (\d+)$", line)
        if match:
            angular = int(match.group(1))
            exponents = None
            continue
        match = re.match(r"        exponents: \[(.*)\]$", line)
        if match:
            exponents = [float(v) for v in match.group(1).split(",")]
            continue
        match = re.match(r"          - \[(.*)\]$", line)
        if match and exponents is not None and angular is not None:
            coefficients = [float(v) for v in match.group(1).split(",")]
            shell = [angular] + [
                [exponent, coefficient]
                for exponent, coefficient in zip(exponents, coefficients)
            ]
            basis[element].append(shell)
    return basis


BASIS = load_basis()
print("basis loaded from qwb-data:",
      {k: len(v) for k, v in BASIS.items() if k in ("H", "O")})


def translated(atoms_angstrom, origin_bohr):
    shifted = []
    for symbol, xyz in atoms_angstrom:
        bohr = [v * BOHR - o for v, o in zip(xyz, origin_bohr)]
        shifted.append([symbol, list(bohr)])
    return gto.M(atom=shifted, basis=BASIS, unit="Bohr", verbose=0)


def angular(mol, order):
    irp = mol.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)
    if order == 1:
        irp = irp.transpose(1, 0, 2, 3)

    def comp(beta, gamma):
        return irp[beta, gamma]

    return np.array([
        comp(1, 2) - comp(2, 1),
        comp(2, 0) - comp(0, 2),
        comp(0, 1) - comp(1, 0),
    ])


def algebra_defect(n):
    anti = n[0] @ n[1] - n[1] @ n[0]
    return np.abs(anti + n[2]).max() / max(np.abs(n).max(), 1e-30)


coords = np.array([xyz for _, xyz in WATER]) * BOHR
centroid = (8.0 * coords[0] + coords[1] + coords[2]) / 10.0
origins = [("O_nucleus", coords[0]), ("charge_centroid", centroid)]

mol0 = translated(WATER, origins[0][1])
defects = [algebra_defect(angular(mol0, order)) for order in (0, 1)]
print(f"algebra residual: order 0 -> {defects[0]:.3e}, order 1 -> {defects[1]:.3e}")
ORDER = 0 if defects[0] < defects[1] else 1
print(f"using component order {ORDER} (0 means component = 3*beta + gamma)")

lines = ["system,origin_label,component,row,column,value"]
for label, origin in origins:
    mol = translated(WATER, origin)
    n = angular(mol, ORDER)
    print(f"water / {label}: nao={mol.nao}, origin_bohr="
          f"({origin[0]:.12f}, {origin[1]:.12f}, {origin[2]:.12f})")
    for component in range(3):
        for i in range(mol.nao):
            for j in range(mol.nao):
                lines.append(f"water,{label},{component},{i},{j},{n[component, i, j]:.16e}")
io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines) - 1} values)")
