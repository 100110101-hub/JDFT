#!/usr/bin/env python3
"""Build eeq2019.json: the EEQ(2019) charge model used by DFT-D4.

Sources (both pinned, byte-identical copies in raw/):
  * multicharge v0.6.0  src/multicharge/param/eeq2019.f90   element parameters
                        src/multicharge/model/eeq.f90       Coulomb kernel + RHS
                        src/multicharge/param.f90            model setup constants
                        src/multicharge/model/type.F90       linear solve / constraint
  * mctc-lib v0.7.0     src/mctc/ncoord/erf.f90             CN counting function
                        src/mctc/ncoord/type.f90           smooth CN cap
                        src/mctc/data/covrad.f90            CN radii of the EEQ model

Usage:
    python extract_eeq2019.py            # (re)write eeq2019.json
    python extract_eeq2019.py --check    # regenerate and compare byte-for-byte
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d4_parse import (  # noqa: E402
    MULTICHARGE_COMMIT,
    MCTC_COMMIT,
    RAW,
    ROOT,
    SYMBOLS,
    parse_fortran_array,
    parse_scalar_parameter,
    write_or_check,
)

EEQ = RAW / "multicharge/src/multicharge/param/eeq2019.f90"
NELEM = 103  # max_elem in eeq2019.f90

chi = parse_fortran_array(EEQ, "eeq_chi", NELEM)
eta = parse_fortran_array(EEQ, "eeq_eta", NELEM)
kcnchi = parse_fortran_array(EEQ, "eeq_kcnchi", NELEM)
rad = parse_fortran_array(EEQ, "eeq_rad", NELEM)

# model setup constants (multicharge/src/multicharge/param.f90:54-56)
cn_cutoff, _ = parse_scalar_parameter(
    RAW / "multicharge/src/multicharge/param.f90", "cutoff"
)
cn_exp, _ = parse_scalar_parameter(RAW / "multicharge/src/multicharge/param.f90", "cn_exp")
cn_max, _ = parse_scalar_parameter(
    RAW / "multicharge/src/multicharge/param.f90", "cn_max"
)

# mctc CN constants actually hit by the EEQ model
mctc_erf = RAW / "mctc-lib/src/mctc/ncoord/erf.f90"
erf_default_kcn, _ = parse_scalar_parameter(mctc_erf, "default_kcn")

# pi / sqrt(2/pi) as written in the Fortran (eeq.f90:61-63)
pi_expr = None
for idx, line in enumerate((RAW / "multicharge/src/multicharge/model/eeq.f90")
                           .read_text(encoding="utf-8").splitlines(), start=1):
    if "sqrtpi = sqrt(pi)" in line:
        pi_expr = idx
        break

elements = []
for z in range(1, NELEM + 1):
    elements.append(
        {
            "Z": z,
            "symbol": SYMBOLS[z - 1] if z <= len(SYMBOLS) else None,
            "chi": chi["values"][z - 1],
            "eta": eta["values"][z - 1],
            "kcnchi": kcnchi["values"][z - 1],
            "rad": rad["values"][z - 1],
            "source_lines": {
                "chi": chi["lines"][z - 1],
                "eta": eta["lines"][z - 1],
                "kcnchi": kcnchi["lines"][z - 1],
                "rad": rad["lines"][z - 1],
            },
        }
    )


def quote(relpath, first, last):
    lines = (ROOT / relpath).read_text(encoding="utf-8").splitlines()
    return {
        "file": relpath,
        "lines": f"{first}-{last}",
        "text": "\n".join(ln.rstrip() for ln in lines[first - 1:last]),
    }


doc = {
    "schema_version": 1,
    "description": (
        "EEQ(2019) charge model of DFT-D4: per-element chi, eta, CN scaling and charge "
        "widths, plus the equations actually implemented (damped erf Coulomb kernel, "
        "electronegativity RHS, Lagrange multiplier for the total charge)."
    ),
    "generated_by": "extract_eeq2019.py (stdlib only; --check compares byte-for-byte)",
    "upstream": {
        "multicharge": {
            "repo": "https://github.com/grimme-lab/multicharge",
            "tag": "v0.6.0",
            "commit": MULTICHARGE_COMMIT,
            "license_spdx": "Apache-2.0",
            "note": "pinned by dftd4 subprojects/multicharge.wrap",
        },
        "mctc-lib": {
            "repo": "https://github.com/grimme-lab/mctc-lib",
            "tag": "v0.7.0",
            "commit": MCTC_COMMIT,
            "license_spdx": "Apache-2.0",
            "note": "pinned by dftd4 subprojects/mctc-lib.wrap",
        },
    },
    "citation": {
        "doi": "10.1063/1.5090222",
        "note": "EEQ introduced in the DFT-D4 paper; cited in eeq2019.f90:16-20",
    },
    "element_range": {
        "max_elem": NELEM,
        "note": "eeq2019.f90:56  integer, parameter :: max_elem = 103",
    },
    "parameters": {
        "chi": {
            "array": "eeq_chi",
            "file": "raw/multicharge/src/multicharge/param/eeq2019.f90",
            "lines": f"{chi['decl_line']}-{chi['last_line']}",
            "units": "hartree/e",
            "meaning": "element electronegativity entering the RHS x_i = -chi_i + ...",
        },
        "eta": {
            "array": "eeq_eta",
            "file": "raw/multicharge/src/multicharge/param/eeq2019.f90",
            "lines": f"{eta['decl_line']}-{eta['last_line']}",
            "units": "hartree/e^2",
            "meaning": "chemical hardness, diagonal of the Coulomb matrix; can be negative",
        },
        "kcnchi": {
            "array": "eeq_kcnchi",
            "file": "raw/multicharge/src/multicharge/param/eeq2019.f90",
            "lines": f"{kcnchi['decl_line']}-{kcnchi['last_line']}",
            "units": "hartree/e",
            "meaning": "CN scaling of the electronegativity (x_i = -chi_i + kcnchi_i*sqrt(CN_i))",
        },
        "rad": {
            "array": "eeq_rad",
            "file": "raw/multicharge/src/multicharge/param/eeq2019.f90",
            "lines": f"{rad['decl_line']}-{rad['last_line']}",
            "units": "bohr",
            "meaning": (
                "charge width; enters the diagonal as sqrt(2/pi)/rad and the "
                "off-diagonal as gamma_ij = 1/(rad_i^2+rad_j^2). These radii are "
                "static: no charge or CN scaling is applied in the kernel."
            ),
        },
    },
    "equations": {
        "coulomb_matrix": {
            "diagonal": "A_ii = eta_i + sqrt(2/pi)/rad_i",
            "off_diagonal": (
                "A_ij = erf(sqrt(r_ij^2 * gamma_ij)) / sqrt(r_ij^2),  "
                "gamma_ij = 1/(rad_i^2 + rad_j^2)"
            ),
            "source": "raw/multicharge/src/multicharge/model/eeq.f90:331-366",
            "quote": quote(
                "raw/multicharge/src/multicharge/model/eeq.f90", 344, 366
            ),
            "radii_scaled": False,
            "radii_note": (
                "The kernel uses self%rad(izp) unchanged (eeq.f90:350 and eeq.f90:355); "
                "the only charge dependence of the model sits in the RHS via CN and in "
                "the reference weighting of D4, not in these radii."
            ),
        },
        "rhs": {
            "formula": (
                "x_i = -chi_i + kcnchi_i * CN_i / sqrt(CN_i + 1e-14);  x_{N+1} = Q_total"
            ),
            "source": "raw/multicharge/src/multicharge/model/eeq.f90:132-155",
            "quote": quote("raw/multicharge/src/multicharge/model/eeq.f90", 139, 155),
            "regularization": "reg = 1.0e-14_wp (eeq.f90:137, 163)",
        },
        "linear_system": {
            "formula": "A q = x with A (N+1)x(N+1) and x (N+1); q = first N entries",
            "source": "raw/multicharge/src/multicharge/model/type.F90:227-273",
            "quote": quote("raw/multicharge/src/multicharge/model/type.F90", 227, 273),
        },
        "constraint": {
            "formula": (
                "the extra row/column of ones with a zero corner is a Lagrange "
                "multiplier enforcing sum_i q_i = Q_total; A q = x therefore solves "
                "the constrained electronegativity equilibration"
            ),
            "source": "raw/multicharge/src/multicharge/model/eeq.f90:363-365",
            "quote": quote("raw/multicharge/src/multicharge/model/eeq.f90", 363, 365),
            "constraint_rhs": "x_{N+1} = mol%charge (eeq.f90:153)",
        },
        "cn_model_used_by_eeq": {
            "model": "plain error-function CN (cn_count%erf), no electronegativity weighting",
            "counting_function": (
                "CN_i = sum_{j!=i} 0.5*(1 + erf(-kcn*(r_ij-rc)/rc)), rc = rcov_i+rcov_j"
            ),
            "kcn": cn_exp,
            "kcn_default_in_mctc": erf_default_kcn,
            "cutoff_bohr": cn_cutoff,
            "cn_max": cn_max,
            "cn_max_application": (
                "CN is smoothly capped with cn' = log(1+exp(cn_max)) - "
                "log(1+exp(cn_max-CN)) (mctc-lib ncoord/type.f90:1124-1134); the D4 "
                "dispersion CN does NOT use this cap."
            ),
            "rcov_source": (
                "multicharge/src/multicharge/param.f90:65 rcov = get_covalent_rad("
                "mol%num); mctc-lib's accessor returns covalent_rad_d3 = "
                "4/3 * covalent_rad_2009 (mctc-lib data/covrad.f90:63-65, 90-104), "
                "i.e. the same 4/3-scaled Pyykko radii the D4 dispersion CN uses"
            ),
            "source": "raw/multicharge/src/multicharge/model/eeq.f90:97-98",
            "quote": quote(
                "raw/multicharge/src/multicharge/model/eeq.f90", 92, 99
            ),
            "setup_constants": {
                "cutoff": cn_cutoff,
                "cn_exp": cn_exp,
                "cn_max": cn_max,
                "source": "raw/multicharge/src/multicharge/param.f90:54-56",
            },
        },
        "d4_reference_charge": {
            "formula": "q_ref(Z, ref) = clsq(ref, Z)  (set_refq_eeq)",
            "source": "raw/dftd4/src/dftd4/reference.f90:222-239",
            "note": (
                "The same EEQ model is used to build the D4 reference charges; the "
                "molecular charges q_i enter the reference weighting as q_i + zeff_i."
            ),
        },
    },
    "constants": {
        "sqrt2pi": "sqrt(2/pi), used in A_ii",
        "pi_source_line": pi_expr,
    },
    "elements": elements,
    "validation": {
        "script": "validate_d4_data.py",
        "reference": (
            "qwb-core/crates/qwb-scf/tests/reference/d4_properties.csv (official "
            "dftd4 4.3.1 via scripts/dftd4_dispersion_reference.py)"
        ),
        "quantities": ["coordination number", "EEQ charge"],
        "result": (
            "reproduced to <= 4e-8 relative for all 9 reference dimers when the "
            "4/3-scaled Pyykko radii are used for the EEQ CN"
        ),
    },
    "quotes": [
        quote("raw/multicharge/src/multicharge/param/eeq2019.f90", 56, 56),
        quote("raw/multicharge/src/multicharge/param/eeq2019.f90", 107, 108),
        quote("raw/multicharge/src/multicharge/param/eeq2019.f90", 131, 132),
        quote("raw/multicharge/src/multicharge/param.f90", 54, 70),
        quote("raw/mctc-lib/src/mctc/ncoord/erf.f90", 106, 122),
    ],
}

out = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
print(f"elements: {len(elements)}  (max_elem={NELEM})")
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
sys.exit(0 if write_or_check(ROOT / "eeq2019.json", out, args.check) else 1)
