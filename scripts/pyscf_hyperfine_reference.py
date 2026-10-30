"""Generate the RS-04 orbital-hyperfine reference with PySCF/cint.

Target:  M_alpha = <mu| ((r-R_A) x nabla)_alpha / |r-R_A|^3 |nu>

This PySCF build has no `int1e_angmom`, but cint does provide
`int1e_prinvxp` = "p* | rinv cross p |": three real antisymmetric components,
i.e. the eps_abg <d_b mu| 1/r |d_g nu> building block of the operator above.
The sign relation is established by the Rust test itself (it compares M and -M
and records which one matches), so the reference here is just cint's integrals.

Basis exponents are injected from qwb-data (PySCF's built-in STO-3G table has
only 8 significant digits).  `int1e_prinvxp` is centred on the coordinate
origin, so each nucleus is reached by translating the molecule.

Output: qwb-core/crates/qwb-integrals/tests/reference/hyperfine.csv
"""
import io
import os
import re

import numpy as np
from pyscf import gto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "qwb-core", "crates", "qwb-integrals", "tests",
                   "reference", "hyperfine.csv")
YAML = os.path.join(ROOT, "qwb-data", "basis", "sto-3g.yaml")
BOHR = 1.889726089838572

WATER = [
    ("O", [0.0, 0.0, 0.117790]),
    ("H", [0.0, 0.755450, -0.471160]),
    ("H", [0.0, -0.755450, -0.471160]),
]


def load_basis():
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
            basis[element].append(
                [angular] + [[e, c] for e, c in zip(exponents, coefficients)]
            )
    return basis


BASIS = load_basis()


def translated(origin_bohr):
    shifted = []
    for symbol, xyz in WATER:
        bohr = [v * BOHR - o for v, o in zip(xyz, origin_bohr)]
        shifted.append([symbol, list(bohr)])
    return gto.M(atom=shifted, basis=BASIS, unit="Bohr", verbose=0)


coords = np.array([xyz for _, xyz in WATER]) * BOHR
targets = [("O0", coords[0], 0), ("H1", coords[1], 1)]

lines = ["system,nucleus_index,component,row,column,value"]
for label, origin, index in targets:
    mol = translated(origin)
    arr = mol.intor("int1e_prinvxp")
    print(f"water / {label}: nao={mol.nao}, shape={arr.shape}, "
          f"|max|={np.abs(arr).max():.6f}")
    assert arr.shape == (3, mol.nao, mol.nao)
    for component in range(3):
        for i in range(mol.nao):
            for j in range(mol.nao):
                lines.append(f"{index},{component},{i},{j},{arr[component, i, j]:.16e}")
io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines) - 1} values)")
