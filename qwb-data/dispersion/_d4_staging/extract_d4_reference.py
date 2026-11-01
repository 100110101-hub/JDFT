#!/usr/bin/env python3
"""Extract the D4 reference-system table from the pinned dftd4 ``reference.inc``.

The D4 C6 coefficients are not fitted per element pair; they are interpolated from
per-element reference systems (up to 7 per element) whose dynamic polarizabilities
are stored for 23 imaginary frequencies.  Without this table DFT-D4 cannot be
implemented, so the staging directory carries both the verbatim include file and
this machine-readable extract for Z = 1..36.

Every number is parsed from raw/dftd4/src/dftd4/reference.inc
(commit 1107ea7c7c2131719e306fd57a5f4a1579e2577e).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INC = ROOT / "raw/dftd4/src/dftd4/reference.inc"
COMMIT = "1107ea7c7c2131719e306fd57a5f4a1579e2577e"
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr"
).split()

text = true_text = INC.read_text(encoding="utf-8")
# strip Fortran comments but keep line structure for provenance of the first record
clean_lines = [ln.split("!")[0] for ln in text.splitlines()]
clean = "\n".join(clean_lines)

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")

records = []  # (name, indices, values, line)
for match in re.finditer(r"data\s+(\w+)\s*\(([^)]*)\)\s*/(.*?)/", clean, re.S):
    name, indices, payload = match.group(1), match.group(2), match.group(3)
    parts = [p.strip() for p in indices.split(",")]
    values = [float(v) for v in NUM.findall(payload)]
    line = clean.count("\n", 0, match.start()) + 1
    records.append((name, parts, values, line))

SCALARS = {}
MATRIX = {}  # name -> {elem: {ref: value}}
VECT = {}  # name -> {elem: {ref: [values]}}
VECT1 = {}  # name -> {index: value}  (secq, sscale, seccn, seccnD3)
for name, parts, values, line in records:
    if name == "refn":
        SCALARS[int(parts[0])] = values[0]
        continue
    if len(parts) == 1:
        VECT1.setdefault(name, {})[int(parts[0])] = values[0]
        continue
    if len(parts) == 2:
        ref, elem = int(parts[0]), int(parts[1])
        MATRIX.setdefault(name, {}).setdefault(elem, {})[ref] = values
    else:  # alphaiw(:, ref, elem)
        ref, elem = int(parts[1]), int(parts[2])
        VECT.setdefault(name, {}).setdefault(elem, {})[ref] = values

elements = []
for z in range(1, 37):
    nref = int(SCALARS.get(z, 0))
    refs = []
    for ref in range(1, nref + 1):
        entry = {"ref": ref}
        for name in ("refsys", "refcn", "refcovcn", "clsq", "clsh", "refq", "refh",
                     "hcount", "ascale"):
            block = MATRIX.get(name, {}).get(z, {}).get(ref)
            if block is None:
                continue
            vals = block
            entry[name] = int(vals[0]) if name == "refsys" else vals[0]
        aiw = VECT.get("alphaiw", {}).get(z, {}).get(ref)
        if aiw is not None:
            if len(aiw) != 23:
                raise SystemExit(f"alphaiw has {len(aiw)} points for Z={z} ref={ref}")
            entry["alphaiw"] = aiw
        refs.append(entry)
    if len(refs) != nref:
        raise SystemExit(f"Z={z}: refn={nref} but {len(refs)} reference records found")
    elements.append({"Z": z, "symbol": SYMBOLS[z - 1], "nref": nref, "references": refs})

# --- global secondary (SE) reference set, reference.inc:1-90 -----------------
se = {}
for name in ("secq", "sscale", "seccn", "seccnD3"):
    flat = VECT1.get(name, {})
    se[name] = [flat[i] for i in sorted(flat)]
secaiw = VECT.get("secaiw", {})
se_aiw = []
for ref in sorted(secaiw.get(1, {})):
    vals = secaiw[1][ref]
    if len(vals) != 23:
        raise SystemExit(f"secaiw ref={ref} has {len(secaiw[1][ref])} points")
    se_aiw.append(vals)

doc = {
    "schema_version": 1,
    "description": (
        "D4 reference systems (per element) and the global secondary (SE) reference "
        "set, extracted from dftd4 v4.3.1 src/dftd4/reference.inc. Used to build the "
        "reference C6 coefficients by numerical Casimir-Polder integration of the "
        "reference dynamic polarizabilities and to weight them by CN and charge."
    ),
    "generated_by": "extract_d4_reference.py (no value typed by hand)",
    "upstream": {
        "repo": "https://github.com/dftd4/dftd4",
        "tag": "v4.3.1",
        "commit": COMMIT,
        "file": "raw/dftd4/src/dftd4/reference.inc",
        "license_spdx": "LGPL-3.0-or-later",
        "note": (
            "reference.inc is a computer generated Fortran DATA include, stated so in "
            "its own header; it is included verbatim in raw/."
        ),
    },
    "declaration": {
        "file": "raw/dftd4/src/dftd4/reference.f90",
        "shapes": {
            "refn": "integer refn(118)",
            "refsys": "real(7,118) reference system index into the SE set",
            "refcn": "real(7,118) D3 coordination number of the reference",
            "refcovcn": "real(7,118) D4 (EN-scaled) CN of the reference; used as refcn",
            "clsq": "real(7,118) EEQ partial charge of the reference",
            "clsh": "real(7,118) EEQ 'h' value of the reference (enters the charge scaling)",
            "refq": "real(7,118) GFN2-xTB partial charge of the reference (qmod=gfn2)",
            "refh": "real(7,118) GFN2-xTB 'h' value (qmod=gfn2)",
            "hcount": "real(7,118) number of hydrogen atoms removed from the parent",
            "ascale": "real(7,118) scaling factor of the reference polarizability",
            "alphaiw": "real(23,7,118) dynamic polarizability at 23 frequencies",
        },
    },
    "alphaiw_frequencies": "dftd4/src/dftd4/model/utils.f90:86-94 (trapzd freq array)",
    "units": {
        "alphaiw": "atomic units (bohr^3), 23 imaginary frequencies in hartree",
        "refcn/refcovcn": "dimensionless coordination number",
        "clsq/refq": "elementary charge e",
        "ascale/hcount": "dimensionless",
    },
    "secondary_set": {
        "size": len(se.get("secq", [])),
        "note": (
            "global reference set referenced by refsys; the 23-point polarizabilities "
            "are secaiw and the scaling factor sscale (reference.f90 set_refalpha_eeq)"
        ),
        "secq": se.get("secq"),
        "sscale": se.get("sscale"),
        "seccn": se.get("seccn"),
        "seccnD3": se.get("seccnD3"),
        "secaiw": se_aiw,
    },
    "elements": elements,
}

(ROOT / "reference_systems.json").write_text(
    json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("elements:", len(elements), "total refs:", sum(e["nref"] for e in elements))
print("SE set size:", len(se.get("secq", [])))
