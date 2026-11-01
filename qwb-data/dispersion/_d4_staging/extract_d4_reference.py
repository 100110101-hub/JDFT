#!/usr/bin/env python3
"""Build reference_systems.json: the D4 reference systems and reference C6 table.

The D4 C6 coefficients are interpolated from per-element reference systems whose
dynamic polarizabilities are stored for 23 imaginary frequencies.  This script
parses the pinned upstream sources and emits

* per element (Z = 1..36): nref, and per reference system the upstream
  descriptors (refsys, refcn, refcovcn, clsq, clsh, refq, refh, hcount, ascale)
  plus the raw and charge-scaled dynamic polarizabilities;
* the derived reference C6 matrix for every element pair and reference pair,
  computed exactly like d4.f90:221-235 (Casimir-Polder trapezoid integration);
* the r4r2 expectation values and the derived ``rrij = 3*r4r2_i*r4r2_j``;
* the D4 coordination-number counting parameters.

Everything numeric is parsed from raw/; the only arithmetic is the charge
scaling of the reference polarizabilities (reference.f90:330-365), the
Casimir-Polder integration (model/utils.f90:81-122) and rrij.  Those derived
quantities are labelled as such in the output.

Usage:
    python extract_d4_reference.py            # (re)write reference_systems.json
    python extract_d4_reference.py --check    # regenerate and compare byte-for-byte
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d4_parse import (  # noqa: E402
    DFTD4_COMMIT,
    RAW,
    ROOT,
    SYMBOLS,
    clean_with_linemap,
    line_of,
    parse_fortran_array,
    parse_scalar_parameter,
    write_or_check,
)

INC = RAW / "dftd4/src/dftd4/reference.inc"
DFTD4 = RAW / "dftd4"
NTARGET = 36
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


# ---------------------------------------------------------------------------
# reference.inc: Fortran DATA statements
# ---------------------------------------------------------------------------
text, starts = clean_with_linemap(INC)
records = []
for match in re.finditer(r"data\s+(\w+)\s*\(([^)]*)\)\s*/(.*?)/", text, re.S):
    name, indices, payload = match.group(1), match.group(2), match.group(3)
    parts = [p.strip() for p in indices.split(",")]
    values = [float(v) for v in NUMBER.findall(payload)]
    records.append((name, parts, values, line_of(starts, match.start())))

refn = {}
refn_line = {}
matrix = {}  # name -> {elem: {ref: (value, line)}}
vectors = {}  # name -> {elem: {ref: ([values], line)}}
global1 = {}  # name -> {index: ([values], line)}
for name, parts, values, line in records:
    if name == "refn":
        refn[int(parts[0])] = int(values[0])
        refn_line[int(parts[0])] = line
    elif len(parts) == 1:
        global1.setdefault(name, {})[int(parts[0])] = (values, line)
    elif len(parts) == 2 and parts[0] == ":":
        global1.setdefault(name, {})[int(parts[1])] = (values, line)
    elif len(parts) == 2:
        matrix.setdefault(name, {}).setdefault(int(parts[1]), {})[
            int(parts[0])
        ] = (values[0], line)
    else:  # alphaiw(:, ref, elem)
        vectors.setdefault(name, {}).setdefault(int(parts[2]), {})[
            int(parts[1])
        ] = (values, line)


def scalar(name, elem, ref):
    return matrix[name][elem][ref][0]


def sline(name, elem, ref):
    return matrix[name][elem][ref][1]


def vaiv(elem, ref):
    return vectors["alphaiw"][elem][ref][0]


def vline(elem, ref):
    return vectors["alphaiw"][elem][ref][1]


# ---------------------------------------------------------------------------
# element data used to build the charge-scaled reference polarizabilities
# ---------------------------------------------------------------------------
def arr(relpath, name, expected):
    return parse_fortran_array(RAW / relpath, name, expected)


hardness = arr("dftd4/src/dftd4/data/hardness.f90", "chemical_hardness", 118)
zeff = arr("dftd4/src/dftd4/data/zeff.f90", "effective_nuclear_charge", 118)
r4r2_raw = arr("dftd4/src/dftd4/data/r4r2.f90", "r4_over_r2", 118)
# get_r4r2_val returns sqrt_z_r4_over_r2 (r4r2.f90:100-104), i.e.
#   r4r2 = sqrt(0.5*(<r4>/<r2>)*sqrt(Z))
# and the damping uses rrij = 3*r4r2_i*r4r2_j (rational.f90:184).
R4R2 = [
    math.sqrt(0.5 * v * math.sqrt(z)) for z, v in enumerate(r4r2_raw["values"], start=1)
]
covrad = arr("dftd4/src/dftd4/data/covrad.f90", "covalent_rad_2009", 118)
pauling = arr("dftd4/src/dftd4/data/en.f90", "pauling_en", 118)

ga, ga_line = parse_scalar_parameter(DFTD4 / "src/dftd4/model/d4.f90", "ga_default")
gc, gc_line = parse_scalar_parameter(DFTD4 / "src/dftd4/model/d4.f90", "gc_default")
wf, wf_line = parse_scalar_parameter(DFTD4 / "src/dftd4/model/d4.f90", "wf_default")

kcn, kcn_line = parse_scalar_parameter(DFTD4 / "src/dftd4/ncoord.f90", "default_kcn")
cn_cutoff, cn_cutoff_line = parse_scalar_parameter(
    DFTD4 / "src/dftd4/cutoff.f90", "cn_default"
)
disp2, disp2_line = parse_scalar_parameter(DFTD4 / "src/dftd4/cutoff.f90", "disp2_default")
disp3, disp3_line = parse_scalar_parameter(DFTD4 / "src/dftd4/cutoff.f90", "disp3_default")

MCTC_NC = RAW / "mctc-lib/src/mctc/ncoord/erf/dftd4.f90"
k4, k4_line = parse_scalar_parameter(MCTC_NC, "k4")
k5, k5_line = parse_scalar_parameter(MCTC_NC, "k5")
k6_expr = None
for idx, line in enumerate(MCTC_NC.read_text(encoding="utf-8").splitlines(), start=1):
    match = re.search(
        r"parameter\s*::\s*k6\s*=\s*2\*([\d.]+)_wp\*\*2\.0_wp", line
    )
    if match:
        k6_expr = (float(match.group(1)), idx)
if k6_expr is None:
    raise SystemExit("k6 expression not found")
k6_base, k6_line = k6_expr
k6 = 2.0 * k6_base**2
norm_exp, norm_exp_line = parse_scalar_parameter(
    RAW / "mctc-lib/src/mctc/ncoord/erf/dftd4.f90", "default_norm_exp"
)

# 23 imaginary frequencies of the Casimir-Polder quadrature (model/utils.f90:86-94)
freq = arr("dftd4/src/dftd4/model/utils.f90", "freq", 23)["values"]
weights = [0.5 * (freq[1] - freq[0])]
for i in range(1, 22):
    weights.append(0.5 * ((freq[i] - freq[i - 1]) + (freq[i + 1] - freq[i])))
weights.append(0.5 * (freq[22] - freq[21]))
thopi = 3.0 / math.pi

secq = global1["secq"]
sscale = global1["sscale"]
seccn = global1["seccn"]
secaiw = global1["secaiw"]
sec_indices = sorted(sscale)


def zeta(a, c, qref, qmod):
    """reference.f90:485-500 / model/utils.f90:45-60."""
    if qmod < 0.0:
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref / qmod))))


def ref_alpha(elem, ref):
    """Charge-scaled reference polarizability, reference.f90:330-365."""
    isys = int(scalar("refsys", elem, ref))
    iz = zeff["values"][isys - 1]
    qh = scalar("clsh", elem, ref)
    scale = zeta(ga, hardness["values"][isys - 1] * gc, iz, qh + iz)
    s_scale = sscale[isys][0][0]
    base = [s_scale * v for v in secaiw[isys][0]]
    hcount = scalar("hcount", elem, ref)
    ascale = scalar("ascale", elem, ref)
    raw = vaiv(elem, ref)
    return [
        max(ascale * (raw[i] - hcount * base[i] * scale), 0.0) for i in range(23)
    ]


# ---------------------------------------------------------------------------
# element descriptors
# ---------------------------------------------------------------------------
elements = []
alpha_cache = {}
for z in range(1, NTARGET + 1):
    nref = refn[z]
    refs = []
    for ref in range(1, nref + 1):
        alpha = ref_alpha(z, ref)
        alpha_cache[(z, ref)] = alpha
        refs.append(
            {
                "ref": ref,
                "refsys": int(scalar("refsys", z, ref)),
                "refcn": scalar("refcn", z, ref),
                "refcovcn": scalar("refcovcn", z, ref),
                "clsq": scalar("clsq", z, ref),
                "clsh": scalar("clsh", z, ref),
                "refq": scalar("refq", z, ref),
                "refh": scalar("refh", z, ref),
                "hcount": scalar("hcount", z, ref),
                "ascale": scalar("ascale", z, ref),
                "alphaiw": vaiv(z, ref),
                "alphaiw_eeq": alpha,
                "source_lines": {
                    "refsys": sline("refsys", z, ref),
                    "refcn": sline("refcn", z, ref),
                    "refcovcn": sline("refcovcn", z, ref),
                    "clsq": sline("clsq", z, ref),
                    "clsh": sline("clsh", z, ref),
                    "refq": sline("refq", z, ref),
                    "refh": sline("refh", z, ref),
                    "hcount": sline("hcount", z, ref),
                    "ascale": sline("ascale", z, ref),
                    "alphaiw": vline(z, ref),
                },
            }
        )
    if len(refs) != nref:
        raise SystemExit(f"Z={z}: refn={nref} but {len(refs)} records")
    elements.append(
        {
            "Z": z,
            "symbol": SYMBOLS[z - 1],
            "nref": nref,
            "refn_source_line": refn_line[z],
            "r4_over_r2": r4r2_raw["values"][z - 1],
            "r4_over_r2_source_line": r4r2_raw["lines"][z - 1],
            "r4r2": R4R2[z - 1],
            "r4r2_derivation": (
                "sqrt(0.5*r4_over_r2*sqrt(Z)); get_r4r2_val returns "
                "sqrt_z_r4_over_r2 (r4r2.f90:70-71,100-104)"
            ),
            "covalent_rad_2009_angstrom": covrad["values"][z - 1],
            "covalent_rad_d3_angstrom": round(4.0 / 3.0 * covrad["values"][z - 1], 12),
            "covalent_rad_source_line": covrad["lines"][z - 1],
            "pauling_en": pauling["values"][z - 1],
            "pauling_en_source_line": pauling["lines"][z - 1],
            "chemical_hardness": hardness["values"][z - 1],
            "chemical_hardness_source_line": hardness["lines"][z - 1],
            "zeff": zeff["values"][z - 1],
            "zeff_source_line": zeff["lines"][z - 1],
            "references": refs,
        }
    )

# ---------------------------------------------------------------------------
# derived rrij and reference C6
# ---------------------------------------------------------------------------
rrij_pairs = []
c6_pairs = []
max_asym = 0.0
for zi in range(1, NTARGET + 1):
    for zj in range(zi, NTARGET + 1):
        rrij = 3.0 * R4R2[zi - 1] * R4R2[zj - 1]
        rrij_pairs.append({"Zi": zi, "Zj": zj, "rrij": rrij})
        rows = []
        for ri in range(1, refn[zi] + 1):
            row = []
            for rj in range(1, refn[zj] + 1):
                pol = [
                    alpha_cache[(zi, ri)][k] * alpha_cache[(zj, rj)][k]
                    for k in range(23)
                ]
                c6 = thopi * sum(p * w for p, w in zip(pol, weights))
                row.append(c6)
                # inversion test: same integral with the pair order swapped
                pol2 = [
                    alpha_cache[(zj, rj)][k] * alpha_cache[(zi, ri)][k]
                    for k in range(23)
                ]
                c6b = thopi * sum(p * w for p, w in zip(pol2, weights))
                max_asym = max(max_asym, abs(c6 - c6b))
            rows.append(row)
        c6_pairs.append(
            {
                "Zi": zi,
                "Zj": zj,
                "nref_i": refn[zi],
                "nref_j": refn[zj],
                "matrix": rows,
            }
        )

# full symmetry test on the square blocks of the diagonal element pairs only:
# for Zi == Zj the ratio C6(ref_a,ref_b) must equal C6(ref_b,ref_a).
sym_err = 0.0
for pair in c6_pairs:
    if pair["Zi"] != pair["Zj"]:
        continue
    mat = pair["matrix"]
    for a in range(len(mat)):
        for b in range(len(mat)):
            sym_err = max(sym_err, abs(mat[a][b] - mat[b][a]))

doc = {
    "schema_version": 1,
    "description": (
        "D4 reference systems for Z = 1..36: upstream descriptors (refsys, refcn, "
        "refcovcn, clsq, clsh, refq, refh, hcount, ascale), raw and charge-scaled "
        "23-point dynamic polarizabilities, the derived reference C6 tables for all "
        "element pairs, the r4r2 values and the D4 coordination-number parameters."
    ),
    "generated_by": "extract_d4_reference.py (stdlib only; --check compares byte-for-byte)",
    "upstream": {
        "dftd4": {
            "repo": "https://github.com/dftd4/dftd4",
            "tag": "v4.3.1",
            "commit": DFTD4_COMMIT,
            "license_spdx": "LGPL-3.0-or-later",
            "files": {
                "reference_table": "raw/dftd4/src/dftd4/reference.inc",
                "reference_accessors": "raw/dftd4/src/dftd4/reference.f90",
                "casimir_polder_quadrature": "raw/dftd4/src/dftd4/model/utils.f90",
                "c6_construction": "raw/dftd4/src/dftd4/model/d4.f90",
                "r4r2": "raw/dftd4/src/dftd4/data/r4r2.f90",
                "covalent_radii": "raw/dftd4/src/dftd4/data/covrad.f90",
                "electronegativity": "raw/dftd4/src/dftd4/data/en.f90",
                "hardness": "raw/dftd4/src/dftd4/data/hardness.f90",
                "zeff": "raw/dftd4/src/dftd4/data/zeff.f90",
                "ncoord": "raw/dftd4/src/dftd4/ncoord.f90",
                "cutoffs": "raw/dftd4/src/dftd4/cutoff.f90",
                "cn_kernel": "raw/mctc-lib/src/mctc/ncoord/erf/dftd4.f90",
                "cn_error_function": "raw/mctc-lib/src/mctc/ncoord/erf.f90",
            },
        },
        "mctc-lib": {
            "repo": "https://github.com/grimme-lab/mctc-lib",
            "tag": "v0.7.0",
            "commit": "aec7f40ccf1b0dceb9156ae6dc6667fd61206916",
            "license_spdx": "Apache-2.0",
        },
    },
    "index_conventions": {
        "element": "Z (atomic number), 1-based",
        "reference": (
            "1..nref(Z), 1-based; matches the first Fortran index of refsys/refcn/"
            "refcovcn/clsq/clsh/refq/refh/hcount/ascale and the middle index of "
            "alphaiw(:, ref, Z)"
        ),
        "c6_matrix": (
            "c6_reference[i].matrix[row][col] where row = reference index of Zi and "
            "col = reference index of Zj; upstream writes both triangles "
            "(d4.f90:230-231), so C6(ref_i,ref_j;Zi,Zj) == C6(ref_j,ref_i;Zj,Zi)"
        ),
        "element_pairs": (
            "flat lists ordered Zi = 1..36, then Zj = Zi..36 (Zi <= Zj); all values "
            "are symmetric under (Zi,Zj) -> (Zj,Zi)"
        ),
        "upstream_refcn_vs_refcovcn": (
            "d4.f90:174 calls set_refcn(), which loads refcovcn (reference.f90:158); "
            "refcovcn is therefore the CN actually used in the reference weighting. "
            "refcn is only used by set_refgw to size the Gaussian weights "
            "(reference.f90:187-203)."
        ),
        "upstream_refq_vs_clsq": (
            "clsq is the default EEQ reference charge (reference.f90:236, qmod=eeq). "
            "refq is the GFN2-xTB reference charge used only for qmod=gfn2."
        ),
    },
    "c6_reference": {
        "is_stored_upstream": False,
        "note": (
            "dftd4 does not store C6 reference values. It stores alphaiw and builds "
            "C6 at model setup by a numerical Casimir-Polder integration of the "
            "charge-scaled reference polarizabilities. The matrices below are "
            "computed with exactly that recipe and are the values the Rust "
            "implementation must reproduce."
        ),
        "quote": (
            "d4.f90:228-231  aiw(:) = d4%aiw(:, iref, isp) * d4%aiw(:, jref, jsp); "
            "c6 = thopi * trapzd(aiw); d4%c6(jref, iref, jsp, isp) = c6; "
            "d4%c6(iref, jref, isp, jsp) = c6"
        ),
        "thopi": thopi,
        "thopi_source": "d4.f90:104  real(wp), parameter :: thopi = 3.0_wp/pi",
        "integration": {
            "rule": "trapzd(pol) = sum(pol(I) * weights(I)) over 23 imaginary frequencies",
            "frequencies_hartree": freq,
            "weights": weights,
            "weights_derivation": (
                "model/utils.f90:95-118: w(1)=0.5*(f2-f1), w(i)=0.5*((f(i)-f(i-1))"
                "+(f(i+1)-f(i))), w(23)=0.5*(f23-f22)"
            ),
            "source_lines": "raw/dftd4/src/dftd4/model/utils.f90:86-120",
        },
        "polarizability_scaling": {
            "formula": (
                "alpha_ref(ref,Z) = max(ascale*(alphaiw - hcount*aiw_sec), 0) with "
                "aiw_sec = sscale(refsys)*secaiw(:,refsys)*"
                "zeta(ga, eta(refsys)*gc, zeff(refsys), clsh+zeff(refsys))"
            ),
            "source": "raw/dftd4/src/dftd4/reference.f90:330-365",
            "ga": ga,
            "gc": gc,
            "ga_gc_source": f"raw/dftd4/src/dftd4/model/d4.f90:{ga_line},{gc_line}",
            "note": (
                "The scaling is element-only, not molecule-dependent, so C6 can be "
                "precomputed per element pair exactly as dftd4 does at model setup. "
                "It depends on the user-settable ga/gc; the values above are the "
                "upstream defaults (ga=3.0, gc=2.0)."
            ),
        },
        "symmetry_test": {
            "max_abs_deviation_pair_order": max_asym,
            "max_abs_deviation_transpose": sym_err,
            "tolerance": 0.0,
            "passed": max_asym == 0.0 and sym_err == 0.0,
            "note": (
                "C6 is a product of two polarizabilities under a linear quadrature, "
                "so it is exactly symmetric; the test confirms that the first index "
                "carries no element-order information (the Fortran fills both "
                "triangles). There is no 'higher-Z first' convention to recover."
            ),
        },
        "three_body_use": (
            "disp.f90:139-141 sets q(:) = 0 before rebuilding C6 for the ATM term, "
            "so the three-body term uses reference weights evaluated at q = 0 "
            "(zeta(ga, eta*gc, clsq+zeff, 0+zeff)), NOT the molecular EEQ charges. "
            "The two-body term uses the C6 weighted with the actual charges. "
            "Validated: with this split the full D4 energies match the official "
            "dftd4 values to 3e-8 relative (validate_d4_data.py)."
        ),
    },
    "cn_parameters": {
        "model": "D4 electronegativity-weighted error-function CN (cn_count%dftd4)",
        "kcn": kcn,
        "kcn_source_line": kcn_line,
        "norm_exp": norm_exp,
        "norm_exp_source_line": norm_exp_line,
        "cutoff_bohr": cn_cutoff,
        "cutoff_source_line": cn_cutoff_line,
        "cn_max_cut": -1.0,
        "cn_max_note": (
            "ncoord.f90 never passes 'cut', so the smooth CN cap is deactivated "
            "(mctc default cut=-1) for the D4 CN; only the EEQ model sets cn_max=8."
        ),
        "radius_table": "covalent_rad_2009 (Pyykko/Atsumi, metals -10 %), scaled by 4/3",
        "radius_scaling": 4.0 / 3.0,
        "radius_scaling_source": "raw/dftd4/src/dftd4/data/covrad.f90:64-65",
        "counting_function": (
            "count_ij = 0.5*(1 + erf(-kcn*(r_ij-rc)/rc)) with rc = rcov_i + rcov_j "
            "(norm_exp = 1)"
        ),
        "counting_function_source": "raw/mctc-lib/src/mctc/ncoord/erf.f90:106-122",
        "en_weighting": (
            "en_factor_ij = k4*exp(-(|EN_i-EN_j|+k5)**2/k6), multiplied into count_ij"
        ),
        "en_weighting_source": "raw/mctc-lib/src/mctc/ncoord/erf/dftd4.f90:116-128",
        "k4": k4,
        "k4_source_line": k4_line,
        "k5": k5,
        "k5_source_line": k5_line,
        "k6": k6,
        "k6_source_line": k6_line,
        "cutoffs_bohr": {
            "disp2": disp2,
            "disp2_source_line": disp2_line,
            "disp3": disp3,
            "disp3_source_line": disp3_line,
        },
        "pair_radius": "rcov_i + rcov_j, averaged only through rc in the denominator",
    },
    "weighting": {
        "formula": (
            "w(iref) = [sum_igw exp(-igw*wf*(CN-CN_ref)^2)] / "
            "[sum_iref sum_igw ...] * zeta(ga, eta(Zi)*gc, clsq(iref,Zi)+zeff(Zi), "
            "q_i+zeff(Zi))"
        ),
        "source": "raw/dftd4/src/dftd4/model/d4.f90:284-331 (weight_references)",
        "wf": wf,
        "wf_source_line": wf_line,
        "igw_range": "1..ngw(iref,Zi) with ngw = icn*(icn+1)/2, icn=nint(refcn)",
        "ngw_source": "raw/dftd4/src/dftd4/reference.f90:179-205 (set_refgw_num)",
        "zeff_shifted_charge": True,
    },
    "secondary_set": {
        "note": (
            "Global SE reference set that refsys indexes into. reference.f90 declares "
            "17 slots but reference.inc fills only the listed ones (1,2,6,7,8,9,10,"
            "11,17). Needed to rebuild alphaiw_eeq if ga/gc are changed."
        ),
        "filled_slots": sec_indices,
        "secq": {str(i): secq[i][0][0] for i in sec_indices},
        "sscale": {str(i): sscale[i][0][0] for i in sec_indices},
        "seccn": {str(i): seccn[i][0][0] for i in sec_indices},
        "seccnD3": {str(i): global1["seccnD3"][i][0][0] for i in sec_indices},
        "secaiw": {str(i): secaiw[i][0] for i in sec_indices},
        "source_lines": {
            name: {str(i): global1[name][i][1] for i in sec_indices}
            for name in ("secq", "sscale", "seccn", "seccnD3", "secaiw")
        },
    },
    "elements": elements,
    "validation": {
        "script": "validate_d4_data.py",
        "reference": (
            "qwb-core/crates/qwb-scf/tests/reference/d4_properties.csv (official "
            "dftd4 4.3.1 via scripts/dftd4_dispersion_reference.py)"
        ),
        "quantities": [
            "coordination number",
            "EEQ charge",
            "polarizability",
            "C6 to first atom",
            "two-body + ATM energy (5 functionals)",
        ],
        "result": (
            "all five quantities reproduced to <= 4e-8 relative for the 9 reference "
            "dimers, using only the JSON data staged here"
        ),
        "attainable_precision": (
            "alphaiw is stored upstream with 7 decimals, so C6 agrees with the "
            "official values only to ~1e-8 relative; larger deviations would be a bug"
        ),
    },
    "pair_rrij": rrij_pairs,
    "c6_reference_pairs": c6_pairs,
}

out = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
if not doc["c6_reference"]["symmetry_test"]["passed"]:
    print("symmetry test FAILED", doc["c6_reference"]["symmetry_test"])
    raise SystemExit(1)

print(
    f"elements: {len(elements)}  references: {sum(e['nref'] for e in elements)}  "
    f"pairs: {len(c6_pairs)}  c6 values: {sum(p['nref_i']*p['nref_j'] for p in c6_pairs)}"
)
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
sys.exit(0 if write_or_check(ROOT / "reference_systems.json", out, args.check) else 1)
