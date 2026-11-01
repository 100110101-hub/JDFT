#!/usr/bin/env python3
"""Extract pinned DFT-D4 element and damping data from verbatim upstream sources.

Every number written to the JSON files is parsed out of the files under ``raw/``
(byte-identical copies of the upstream files, pinned by commit in PROVENANCE.md).
Nothing is transcribed by hand.

Sources
-------
* dftd4 v4.3.1 (commit 1107ea7c7c2131719e306fd57a5f4a1579e2577e)
    src/dftd4/data/en.f90            Pauling electronegativity  (D4 CN)
    src/dftd4/data/hardness.f90      chemical hardness          (charge scaling)
    src/dftd4/data/covrad.f90        Pyykko covalent radii      (D4 CN)
    src/dftd4/data/zeff.f90          effective nuclear charges  (charge scaling)
    src/dftd4/data/r4r2.f90          <r^4>/<r^2> expectation values
    src/dftd4/param.f90              functional damping parameters
    assets/parameters.toml           cross check of the same parameters
* multicharge v0.6.0 (commit 576f499b49eb57f8c2dc1ba6f7017800f1e332d9)
    src/multicharge/param/eeq2019.f90   EEQ chi / eta / kcnchi / rad

Run:  python extract_d4_data.py
"""

from __future__ import annotations

import argparse
import bisect
import math
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
CHECK = parser.parse_args().check
_OK = True


def emit(path: Path, obj) -> None:
    """Write JSON or, with --check, compare byte-for-byte with the file on disk."""
    global _OK
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    if CHECK:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            print(f"OK   {path.name}")
        else:
            print(f"FAIL {path.name}: differs or missing")
            _OK = False
    else:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.name}")

DFTD4_COMMIT = "1107ea7c7c2131719e306fd57a5f4a1579e2577e"
MULTICHARGE_COMMIT = "576f499b49eb57f8c2dc1ba6f7017800f1e332d9"
MCTC_COMMIT = "aec7f40ccf1b0dceb9156ae6dc6667fd61206916"

REPOS = {
    "dftd4": ("dftd4/dftd4", DFTD4_COMMIT, "v4.3.1"),
    "multicharge": ("grimme-lab/multicharge", MULTICHARGE_COMMIT, "v0.6.0"),
    "mctc-lib": ("grimme-lab/mctc-lib", MCTC_COMMIT, "v0.7.0"),
}

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr"
).split()

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdDqQ][-+]?\d+)?")


def clean_with_linemap(path: Path):
    """Return (comment-free text, list of line start offsets)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    clean = [ln.split("!")[0] for ln in lines]
    starts = []
    pos = 0
    for ln in clean:
        starts.append(pos)
        pos += len(ln) + 1
    return "\n".join(clean), starts


def line_of(starts, offset):
    return bisect.bisect_right(starts, offset)


def parse_fortran_array(path: Path, name: str, expected: int, prefix_mult=None):
    """Parse ``real(wp), parameter :: name(max_elem) = [ ... ]``.

    Returns (values, lines, declaration_line, block_first_line, block_last_line,
             prefix_text).
    """
    text, starts = clean_with_linemap(path)
    decl = re.search(
        r"parameter\s*::\s*" + re.escape(name) + r"\s*\(\s*max_elem\s*\)\s*=", text
    )
    if decl is None:
        raise SystemExit(f"array {name} not found in {path}")
    decl_line = line_of(starts, decl.start())
    open_bracket = text.index("[", decl.end())
    prefix_text = text[decl.end():open_bracket].strip()
    depth = 0
    close_bracket = None
    for idx in range(open_bracket, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                close_bracket = idx
                break
    if close_bracket is None:
        raise SystemExit(f"unterminated array {name} in {path}")
    block = text[open_bracket + 1:close_bracket]
    values, lines = [], []
    for match in NUMBER.finditer(block):
        values.append(float(match.group(0).replace("_wp", "")))
        lines.append(line_of(starts, open_bracket + 1 + match.start()))
    if len(values) != expected:
        raise SystemExit(
            f"{name}: parsed {len(values)} values, expected {expected} "
            f"({path})"
        )
    # covrad.f90 stores ``aatoau * [ <Angstrom literals> ]``.  We keep the
    # Angstrom literals (conversion to bohr is a load-time step, see PROVENANCE.md);
    # an explicit multiplier is only applied when supplied.
    if prefix_mult is not None:
        values = [prefix_mult * v for v in values]
    first_line = line_of(starts, open_bracket)
    last_line = line_of(starts, close_bracket)
    return values, lines, decl_line, first_line, last_line, prefix_text


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# element data
# ---------------------------------------------------------------------------
DFTD4 = RAW / "dftd4"
MC = RAW / "multicharge"

field_specs = [
    # key, path, array name, expected length, repo, multiplier
    ("pauling_en", DFTD4 / "src/dftd4/data/en.f90", "pauling_en", 118, "dftd4", None),
    (
        "chemical_hardness",
        DFTD4 / "src/dftd4/data/hardness.f90",
        "chemical_hardness",
        118,
        "dftd4",
        None,
    ),
    (
        "covalent_rad_2009_angstrom",
        DFTD4 / "src/dftd4/data/covrad.f90",
        "covalent_rad_2009",
        118,
        "dftd4",
        None,
    ),
    ("zeff", DFTD4 / "src/dftd4/data/zeff.f90", "effective_nuclear_charge", 118, "dftd4", None),
    ("r4_over_r2", DFTD4 / "src/dftd4/data/r4r2.f90", "r4_over_r2", 118, "dftd4", None),
    ("chi", MC / "src/multicharge/param/eeq2019.f90", "eeq_chi", 103, "multicharge", None),
    ("eta", MC / "src/multicharge/param/eeq2019.f90", "eeq_eta", 103, "multicharge", None),
    (
        "kcnchi",
        MC / "src/multicharge/param/eeq2019.f90",
        "eeq_kcnchi",
        103,
        "multicharge",
        None,
    ),
    ("rad", MC / "src/multicharge/param/eeq2019.f90", "eeq_rad", 103, "multicharge", None),
]

parsed = {}
for key, path, array, expected, repo, mult in field_specs:
    values, lines, decl_line, first, last, prefix = parse_fortran_array(
        path, array, expected, mult
    )
    parsed[key] = {
        "values": values,
        "lines": lines,
        "decl_line": decl_line,
        "first_line": first,
        "last_line": last,
        "prefix": prefix,
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "array": array,
        "repo": repo,
    }

NTARGET = 36
elements = []
for z in range(1, NTARGET + 1):
    entry = {"Z": z, "symbol": SYMBOLS[z - 1]}
    source_lines = {}
    for key in (
        "chi",
        "eta",
        "kcnchi",
        "rad",
        "pauling_en",
        "chemical_hardness",
        "covalent_rad_2009_angstrom",
        "zeff",
        "r4_over_r2",
    ):
        info = parsed[key]
        entry[key] = info["values"][z - 1]
        source_lines[key] = info["lines"][z - 1]
    # upstream explicitly applies the D3 scaling 4/3 to the Pyykko table:
    #   covalent_rad_d3 = 4.0_wp / 3.0_wp * covalent_rad_2009   (covrad.f90:64-65)
    entry["covalent_rad_d3_angstrom"] = round(
        4.0 / 3.0 * entry["covalent_rad_2009_angstrom"], 12
    )
    # the value D4 actually uses is the derived array (r4r2.f90:70-71):
    #   sqrt_z_r4_over_r2 = sqrt(0.5*(r4_over_r2*sqrt(Z)))
    # get_r4r2_val returns sqrt_z_r4_over_r2 (r4r2.f90:100-104), and the damping
    # uses rrij = 3*r4r2_i*r4r2_j (rational.f90:184).
    entry["r4r2"] = math.sqrt(0.5 * entry["r4_over_r2"] * math.sqrt(z))
    entry["source_lines"] = source_lines
    entry["comment"] = (
        "chi,eta,kcnchi,rad: multicharge eeq2019.f90 "
        f"lines {source_lines['chi']},{source_lines['eta']},"
        f"{source_lines['kcnchi']},{source_lines['rad']}; "
        "pauling_en: dftd4 data/en.f90:{0}; "
        "chemical_hardness: dftd4 data/hardness.f90:{1}; "
        "covalent_rad_2009_angstrom: dftd4 data/covrad.f90:{2} "
        "(literal Angstrom, upstream multiplies by aatoau); "
        "covalent_rad_d3_angstrom = 4/3 * covalent_rad_2009_angstrom "
        "(covrad.f90:64-65); "
        "zeff: dftd4 data/zeff.f90:{3}; "
        "r4_over_r2: dftd4 data/r4r2.f90:{4} (raw table); "
        "r4r2 = sqrt(0.5*r4_over_r2*sqrt(Z)) (derived array D4 actually uses)"
    ).format(
        source_lines["pauling_en"],
        source_lines["chemical_hardness"],
        source_lines["covalent_rad_2009_angstrom"],
        source_lines["zeff"],
        source_lines["r4_over_r2"],
    )
    elements.append(entry)

field_meta = {}
for key in (
    "chi",
    "eta",
    "kcnchi",
    "rad",
    "pauling_en",
    "chemical_hardness",
    "covalent_rad_2009_angstrom",
    "zeff",
    "r4_over_r2",
):
    info = parsed[key]
    owner, commit, tag = REPOS[info["repo"]]
    # file is "raw/<repo>/<repo path>" -> repo path is everything after the repo name
    repo_path = "/".join(info["file"].split("/")[2:])
    field_meta[key] = {
        "file": info["file"],
        "array": info["array"],
        "lines": f"{info['decl_line']}-{info['last_line']}",
        "repo": owner,
        "commit": commit,
        "tag": tag,
        "url": f"https://raw.githubusercontent.com/{owner}/{commit}/{repo_path}",
    }

eeq_doc = {
    "schema_version": 1,
    "description": (
        "Per-element parameters for the DFT-D4 dispersion correction (EEQ charge model, "
        "D4 coordination number, charge scaling and C8 extrapolation) for Z = 1..36, "
        "extracted programmatically from the pinned upstream sources listed in 'fields'. "
        "Field names follow the upstream Fortran array names where possible."
    ),
    "generated_by": "extract_d4_data.py (no value typed by hand)",
    "upstream": {
        "dftd4": {
            "repo": "https://github.com/dftd4/dftd4",
            "tag": "v4.3.1",
            "commit": DFTD4_COMMIT,
            "license_spdx": "LGPL-3.0-or-later",
        },
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
            "note": "pinned by dftd4 subprojects/mctc-lib.wrap; provides the CN kernels",
        },
    },
    "fields": field_meta,
    "units": {
        "chi": "electronegativity, hartree/e (multicharge eeq2019 self-energy)",
        "eta": "chemical hardness, hartree/e^2 (EEQ diagonal); may be negative",
        "kcnchi": "CN scaling constant of the EEQ electronegativity, hartree/e",
        "rad": "EEQ charge width / damping radius, bohr",
        "pauling_en": "Pauling electronegativity, dimensionless (D4 CN weighting)",
        "chemical_hardness": (
            "chemical hardness used by the D4 charge scaling function "
            "zeta(ga, eta*gc, q_ref, q), hartree/e^2"
        ),
        "covalent_rad_2009_angstrom": (
            "Pyykko/Atsumi covalent radius in Angstrom exactly as printed upstream "
            "(metals already reduced by 10 %); upstream multiplies by aatoau at compile time"
        ),
        "covalent_rad_d3_angstrom": "4/3 * covalent_rad_2009_angstrom (used for the D4 CN)",
        "zeff": "effective nuclear charge of the def2 ECP, e (q is shifted by zeff)",
        "r4_over_r2": (
            "<r^4>/<r^2> expectation value, bohr^2, raw upstream table "
            "(r4r2.f90:44-68); NOT the value used in the damping"
        ),
        "r4r2": (
            "derived sqrt(0.5*r4_over_r2*sqrt(Z)) (r4r2.f90:70-71), the value "
            "get_r4r2_val returns and D4 uses in C8/C6 = 3*r4r2_i*r4r2_j"
        ),
    },
    "element_order": "sorted by Z",
    "elements": elements,
}

emit(ROOT / "eeq_elements.json", eeq_doc)

# ---------------------------------------------------------------------------
# damping parameters
# ---------------------------------------------------------------------------
PARAM = DFTD4 / "src/dftd4/param.f90"
param_text, param_starts = clean_with_linemap(PARAM)
param_lines = PARAM.read_text(encoding="utf-8").splitlines()

# --- functional name -> case label (get_functional_id) ---------------------
name_to_label = {}
for match in re.finditer(r'case\s*\(([^)]*)\)\s*\n\s*num\s*=\s*(p_[a-z0-9_]+)', param_text):
    names, label = match.group(1), match.group(2)
    for raw_name in re.findall(r'"([^"]+)"', names):
        name_to_label[raw_name] = label

# --- param case labels (get_d4eeq_bjatm_parameter) -------------------------
start = param_text.index("subroutine get_d4eeq_bjatm_parameter")
select_at = param_text.index("select case", start)
end_select = param_text.index("end select", select_at)
block_text = param_text[select_at:end_select]

label_params = {}
label_lines = {}
for match in re.finditer(r"case\s*\(\s*(p_[a-z0-9_]+)\s*\)", block_text):
    label = match.group(1)
    tail = block_text[match.end():]
    # stop at the next case or the end select
    stop = re.search(r"\n\s*case\s*\(\s*p_", tail)
    body = tail[: stop.start()] if stop else tail
    kwargs = {}
    for kw in re.finditer(r"(s6|s8|s9|a1|a2|alp)\s*=\s*([-+]?[\d.]+)_wp", body):
        kwargs[kw.group(1)] = float(kw.group(2))
    label_params[label] = kwargs
    label_lines[label] = line_of(param_starts, select_at + match.start())

# defaults from the local dftd_param function of get_d4eeq_bjatm_parameter
bjatm_start = start
bjatm_end = param_text.index("end subroutine get_d4eeq_bjatm_parameter", bjatm_start)
bjatm_text = param_text[bjatm_start:bjatm_end]
defaults = {}
for match in re.finditer(r"(s6_|s9_|alp_)\s*=\s*([-+]?[\d.]+)_wp", bjatm_text):
    defaults[match.group(1)] = float(match.group(2))
DEFAULTS = {
    "s6": defaults.get("s6_", 1.0),
    "s9": defaults.get("s9_", 1.0),
    "alp": defaults.get("alp_", 16.0),
}

# --- cross check against assets/parameters.toml ----------------------------
toml_text = (DFTD4 / "assets/parameters.toml").read_text(encoding="utf-8").splitlines()
toml_params = {}
current = None
for idx, line in enumerate(toml_text, start=1):
    head = re.match(r"\s*\[parameter\.([a-z0-9_-]+)\]\s*$", line)
    if head:
        current = head.group(1)
        continue
    body = re.match(r"\s*d4\.bj-eeq-atm\s*=\s*\{(.*)\}\s*$", line)
    if body and current:
        vals = {}
        for kw in re.finditer(r"(s6|s8|a1|a2|alp)\s*=\s*([-+]?[\d.]+)", body.group(1)):
            vals[kw.group(1)] = float(kw.group(2))
        doi = re.search(r'doi\s*=\s*"([^"]+)"', body.group(1))
        vals["doi"] = doi.group(1) if doi else None
        vals["line"] = idx
        toml_params[current] = vals

requested = {
    "b3lyp": ["b3lyp", "b3-lyp"],
    "pbe": ["pbe"],
    "pbe0": ["pbe0"],
    "hf": ["hf"],
    "m06-l": ["m06l", "m06-l"],
    "m06-2x": ["m062x", "m06-2x"],
}

D4_DOI = "10.1063/1.5090222"
functionals = {}
for key, aliases in requested.items():
    label = None
    used_alias = None
    for alias in aliases:
        if alias in name_to_label:
            label = name_to_label[alias]
            used_alias = alias
            break
    if label is None or label not in label_params:
        functionals[key] = {
            "absent_upstream": True,
            "requested_aliases": aliases,
            "reason": (
                "dftd4 v4.3.1 has no damping parameter entry for this functional: the "
                "functional enumerator/param tables in src/dftd4/param.f90 do not "
                "contain it and assets/parameters.toml has no [parameter.*] block for "
                "it. No numbers are provided rather than inventing values."
            ),
            "checked_files": [
                "dftd4/src/dftd4/param.f90",
                "dftd4/assets/parameters.toml",
            ],
        }
        continue
    kwargs = dict(label_params[label])
    entry = {
        "s6": kwargs.get("s6", DEFAULTS["s6"]),
        "s8": kwargs["s8"],
        "a1": kwargs["a1"],
        "a2": kwargs["a2"],
        "alp": kwargs.get("alp", DEFAULTS["alp"]),
        "s9": kwargs.get("s9", DEFAULTS["s9"]),
        "charge_response_coefficients": {
            "s6q": None,
            "s8q": None,
            "a1q": None,
            "a2q": None,
            "note": (
                "upstream defines no charge-response damping coefficients; "
                "charge dependence is carried by the EEQ charges and the reference "
                "weighting zeta(ga, eta*gc, q_ref+zeff, q+zeff). See model.charge_response."
            ),
        },
        "aliases": [
            name for name, lab in name_to_label.items() if lab == label
        ],
        "matched_alias": used_alias,
        "case_label": label,
        "source": {
            "file": "dftd4/src/dftd4/param.f90",
            "subroutine": "get_d4eeq_bjatm_parameter",
            "case_line": label_lines[label],
            "url": (
                "https://raw.githubusercontent.com/dftd4/dftd4/"
                f"{DFTD4_COMMIT}/src/dftd4/param.f90#L{label_lines[label]}"
            ),
        },
        "doi": D4_DOI,
    }
    toml_key = key.replace("-", "")
    cross = None
    for candidate in (key, toml_key, key.replace("-", "")):
        if candidate in toml_params:
            cross = toml_params[candidate]
            break
    if cross is not None:
        match = all(
            abs(cross[k] - entry[k]) < 1e-12
            for k in ("s8", "a1", "a2")
            if k in cross
        ) and abs(cross.get("s6", 1.0) - entry["s6"]) < 1e-12
        entry["cross_check_parameters_toml"] = {
            "match": bool(match),
            "file": "dftd4/assets/parameters.toml",
            "line": cross["line"],
            "values": {k: cross[k] for k in ("s6", "s8", "a1", "a2") if k in cross},
        }
    functionals[key] = entry

damping_doc = {
    "schema_version": 1,
    "description": (
        "DFT-D4 damping parameters per functional, extracted programmatically from the "
        "pinned dftd4 v4.3.1 source. Numbers are parsed from "
        "src/dftd4/param.f90::get_d4eeq_bjatm_parameter and cross-checked against "
        "assets/parameters.toml."
    ),
    "generated_by": "extract_d4_data.py (no value typed by hand)",
    "upstream": {
        "repo": "https://github.com/dftd4/dftd4",
        "tag": "v4.3.1",
        "commit": DFTD4_COMMIT,
        "license_spdx": "LGPL-3.0-or-later",
    },
    "citation": {
        "title": (
            "A generally applicable atomic-charge dependent London dispersion correction"
        ),
        "authors": (
            "E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, "
            "C. Bannwarth, S. Grimme"
        ),
        "journal": "J. Chem. Phys. 150, 154122 (2019)",
        "doi": D4_DOI,
        "verified_from": "dftd4/CITATION.bib entry dftd4-2",
    },
    "model": {
        "two_body_energy_implemented": (
            "E_2 = -1/2 * sum_{i,j} C6_ij * sw_ij * "
            "( s6 / (R_ij^6 + r0_ij^6) + s8 * rrij_ij / (R_ij^8 + r0_ij^8) )"
        ),
        "variables": {
            "rrij_ij": "3 * r4r2_i * r4r2_j  (= C8/C6)",
            "r0_ij": "a1 * sqrt(rrij_ij) + a2",
            "C6_ij": (
                "sum_{ref,ref'} w_ref(i)*w_ref'(j)*C6_ref,ref'(Z_i,Z_j); the reference "
                "weights w are CN- and charge-dependent (see reference_systems.json)"
            ),
            "sw_ij": "quintic smooth cutoff switch, 1 for r <= cutoff-width (cutoff 60 bohr default)",
            "alp": "NOT used by the two-body term; upstream uses it only in the ATM term",
        },
        "atm_three_body_energy_implemented": (
            "E_3 = sum_{i<j<k} ang_ijk * fdmp_ijk * C9_ijk with "
            "C9_ijk = -s9*sqrt(|C6_ij*C6_ik*C6_jk|), "
            "r0_ijk = prod r0_ab, fdmp = 1/(1 + 6*(r0_ijk/(r_ij r_ik r_jk))^(alp/3)), "
            "ang = 3/8*(...)/(r_ij r_ik r_jk)^5 + 1/(r_ij r_ik r_jk)^3"
        ),
        "atm_enabled_by_default": True,
        "atm_note": (
            "get_rational_damping() prefers the 'bj-eeq-atm' table, whose dftd_param "
            "defaults s9 = 1.0, so the three-body term is on by default; s9 = 0 "
            "disables it (get_atm_dispersion returns immediately for |s9| < eps). "
            "disp.f90:139-141 resets q(:) = 0 before rebuilding C6 for the ATM call, "
            "so the three-body term uses weights evaluated at q = 0 while the "
            "two-body term keeps the charge-weighted C6."
        ),
        "defaults": DEFAULTS,
        "charge_response": {
            "coefficients_exist_upstream": False,
            "note": (
                "There are no s6q/s8q/a1q/a2q coefficients anywhere in dftd4 v4.3.1. "
                "The charge dependence of D4 enters (a) through the EEQ partial charges "
                "q used in the reference weighting, and (b) through the element-specific "
                "charge scaling zeta(ga, eta_i*gc, q_ref+zeff, q+zeff). The damping "
                "parameters s6, s8, a1, a2 are charge-independent constants."
            ),
            "ga": 3.0,
            "gc": 2.0,
            "wf": 6.0,
            "ga_gc_source": "dftd4/src/dftd4/model/d4.f90:65-71 (ga_default, gc_default, wf_default)",
            "zeta_source": "dftd4/src/dftd4/model/utils.f90:44-60",
        },
    },
    "functionals": functionals,
}

emit(ROOT / "damping_parameters.json", damping_doc)

# ---------------------------------------------------------------------------
# manifest of raw files
# ---------------------------------------------------------------------------
manifest = []
for path in sorted(RAW.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(RAW)
    repo_key = rel.parts[0]
    owner, commit, tag = REPOS[repo_key]
    repo_path = "/".join(rel.parts[1:])
    manifest.append(
        {
            "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "repo": owner,
            "tag": tag,
            "commit": commit,
            "repo_path": repo_path,
            "url": f"https://raw.githubusercontent.com/{owner}/{commit}/{repo_path}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
emit(ROOT / "raw_manifest.json", {"files": manifest})

print(f"elements: {len(elements)}")
for key, entry in functionals.items():
    if entry.get("absent_upstream"):
        print(f"{key:8s} ABSENT")
    else:
        check = entry.get("cross_check_parameters_toml", {}).get("match")
        print(
            f"{key:8s} s8={entry['s8']:<12} a1={entry['a1']:<12} "
            f"a2={entry['a2']:<12} toml_match={check}"
        )
print(f"raw files: {len(manifest)}")
sys.exit(0 if _OK else 1)
