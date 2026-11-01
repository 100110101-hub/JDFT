#!/usr/bin/env python3
"""Validate the staged D4 data against the official dftd4 reference values.

Two independent checks, both against values produced by the official dftd4
4.3.1 Python bindings (see scripts/dftd4_dispersion_reference.py):

1. per-atom properties for 9 dimers (d4_properties.csv):
   coordination number, EEQ charge, polarizability, C6 to the first atom;
2. total two-body + ATM dispersion energy for those dimers and the five
   functionals staged here (d4_dispersion.csv).

Everything is recomputed **from the staged JSON files only**, so a mismatch
means the extracted data or an index convention is wrong.

Usage:
    python validate_d4_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REFERENCE = REPO / "qwb-core/crates/qwb-scf/tests/reference"
GEOMETRY = REFERENCE / "d3_dimers.csv"
PROPERTIES = REFERENCE / "d4_properties.csv"
DISPERSION = REFERENCE / "d4_dispersion.csv"
METHOD_MAP = {"m06l": "m06-l", "m06-2x": "m06-2x"}


def mctc_aatoau() -> float:
    """Reproduce mctc-lib's aatoau from its CODATA 2018 constants file."""
    text = (ROOT / "raw/mctc-lib/src/mctc/io/codata2018.f90").read_text(
        encoding="utf-8"
    )

    def value(name):
        match = re.search(
            r"::\s*" + name + r"\s*=\s*([-+]?[\d.eEdD+-]+)_wp", text
        )
        return float(match.group(1))

    hbar = value("reduced_Planck_constant")
    me = value("electron_mass")
    c = value("speed_of_light_in_vacuum")
    alpha = value("fine_structure_constant")
    bohr = hbar / (me * c * alpha)  # metre
    autoaa = bohr * 1e10  # Angstrom
    return 1.0 / autoaa


AA2BOHR = mctc_aatoau()

data = json.loads((ROOT / "reference_systems.json").read_text(encoding="utf-8"))
eeq = json.loads((ROOT / "eeq2019.json").read_text(encoding="utf-8"))
damp = json.loads((ROOT / "damping_parameters.json").read_text(encoding="utf-8"))

el = {e["Z"]: e for e in data["elements"]}
eeq_el = {e["Z"]: e for e in eeq["elements"]}
cnp = data["cn_parameters"]
ga = data["c6_reference"]["polarizability_scaling"]["ga"]
gc = data["c6_reference"]["polarizability_scaling"]["gc"]
wf = data["weighting"]["wf"]
c6_lookup = {(p["Zi"], p["Zj"]): p for p in data["c6_reference_pairs"]}
sym = {e["symbol"]: e["Z"] for e in data["elements"]}

TWO_PI = 2.0 / math.pi


def zeta(a, c, qref, qmod):
    if qmod < 0.0:
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref / qmod))))


def nint(x):
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def ngw_vector(z):
    """reference.f90:187-203 set_refgw_num."""
    refs = el[z]["references"]
    cnc = [0] * 20
    cnc[0] = 1
    for r in refs:
        cnc[min(nint(r["refcn"]), 19)] += 1
    out = []
    for r in refs:
        icn = cnc[min(nint(r["refcn"]), 19)]
        out.append(icn * (icn + 1) // 2)
    return out


def dist(mol, i, j):
    return math.dist(mol[i]["xyz"], mol[j]["xyz"])


def d4_cn(mol):
    """D4 CN: EN-weighted error function, kcn=7.5, rcov = 4/3 Pyykko."""
    n = len(mol)
    cn = [0.0] * n
    for i in range(n):
        zi = mol[i]["Z"]
        for j in range(n):
            if i == j:
                continue
            zj = mol[j]["Z"]
            r = dist(mol, i, j)
            rc = (
                el[zi]["covalent_rad_d3_angstrom"]
                + el[zj]["covalent_rad_d3_angstrom"]
            ) * AA2BOHR
            count = 0.5 * (1.0 + math.erf(-cnp["kcn"] * (r - rc) / rc))
            en = cnp["k4"] * math.exp(
                -((abs(el[zi]["pauling_en"] - el[zj]["pauling_en"]) + cnp["k5"]) ** 2)
                / cnp["k6"]
            )
            cn[i] += en * count
    return cn


def eeq_charges(mol, total_charge):
    """EEQ(2019) charges from the staged tables (multicharge eeq.f90)."""
    n = len(mol)
    # CN of the charge model: plain erf CN, D3 (4/3-scaled) Pyykko radii,
    # capped at 8.  mctc-lib's get_covalent_rad returns covalent_rad_d3
    # (data/covrad.f90:90-104), i.e. the same radii the D4 CN uses.
    cn = [0.0] * n
    for i in range(n):
        zi = mol[i]["Z"]
        for j in range(n):
            if i == j:
                continue
            zj = mol[j]["Z"]
            r = dist(mol, i, j)
            rcov = (
                el[zi]["covalent_rad_d3_angstrom"]
                + el[zj]["covalent_rad_d3_angstrom"]
            ) * AA2BOHR
            if r > 25.0:  # EEQ cutoff 25 bohr
                continue
            cn[i] += 0.5 * (1.0 + math.erf(-7.5 * (r - rcov) / rcov))
    cn_max = 8.0
    cn = [
        math.log(1.0 + math.exp(cn_max)) - math.log(1.0 + math.exp(cn_max - c))
        for c in cn
    ]

    m = n + 1
    a = [[0.0] * m for _ in range(m)]
    x = [0.0] * m
    for i in range(n):
        p = eeq_el[mol[i]["Z"]]
        a[i][i] = p["eta"] + math.sqrt(TWO_PI) / p["rad"]
        x[i] = -p["chi"] + p["kcnchi"] * cn[i] / math.sqrt(cn[i] + 1e-14)
        for j in range(n):
            if i == j:
                continue
            q = eeq_el[mol[j]["Z"]]
            r2 = sum((mol[i]["xyz"][k] - mol[j]["xyz"][k]) ** 2 for k in range(3))
            gam = 1.0 / (p["rad"] ** 2 + q["rad"] ** 2)
            a[i][j] = math.erf(math.sqrt(r2 * gam)) / math.sqrt(r2)
    for i in range(n):
        a[n][i] = 1.0
        a[i][n] = 1.0
    a[n][n] = 0.0
    x[n] = total_charge

    # Gaussian elimination with partial pivoting
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(a[r][col]))
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            x[col], x[piv] = x[piv], x[col]
        for r in range(col + 1, m):
            f = a[r][col] / a[col][col]
            if f == 0.0:
                continue
            for c in range(col, m):
                a[r][c] -= f * a[col][c]
            x[r] -= f * x[col]
    sol = [0.0] * m
    for i in range(m - 1, -1, -1):
        s = x[i] - sum(a[i][j] * sol[j] for j in range(i + 1, m))
        sol[i] = s / a[i][i]
    return sol[:n]


def reference_weights(mol, q, cn):
    weights = []
    for i, atom in enumerate(mol):
        zi = atom["Z"]
        refs = el[zi]["references"]
        ngw = ngw_vector(zi)
        gi = el[zi]["chemical_hardness"] * gc
        expw = []
        norm = 0.0
        for r, ref in enumerate(refs):
            s = 0.0
            for igw in range(1, ngw[r] + 1):
                s += math.exp(-igw * wf * (cn[i] - ref["refcovcn"]) ** 2)
            expw.append(s)
            norm += s
        norm = 1.0 / norm if abs(norm) > 0 else 0.0
        w = []
        for r, ref in enumerate(refs):
            gwk = expw[r] * norm
            w.append(
                gwk * zeta(ga, gi, ref["clsq"] + el[zi]["zeff"], q[i] + el[zi]["zeff"])
            )
        weights.append(w)
    return weights


def c6_matrix(mol, weights):
    n = len(mol)
    c6 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        zi = mol[i]["Z"]
        for j in range(n):
            zj = mol[j]["Z"]
            pair = c6_lookup[(min(zi, zj), max(zi, zj))]
            total = 0.0
            for r in range(len(weights[i])):
                for s in range(len(weights[j])):
                    if (zi, zj) == (pair["Zi"], pair["Zj"]):
                        total += weights[i][r] * weights[j][s] * pair["matrix"][r][s]
                    else:
                        total += weights[i][r] * weights[j][s] * pair["matrix"][s][r]
            c6[i][j] = total
    return c6


def energy_two_body(mol, c6, par):
    total = 0.0
    n = len(mol)
    s6, s8, a1, a2 = par["s6"], par["s8"], par["a1"], par["a2"]
    for i in range(n):
        for j in range(i + 1, n):
            r = dist(mol, i, j)
            rrij = 3.0 * el[mol[i]["Z"]]["r4r2"] * el[mol[j]["Z"]]["r4r2"]
            r0 = a1 * math.sqrt(rrij) + a2
            t6 = 1.0 / (r**6 + r0**6)
            t8 = 1.0 / (r**8 + r0**8)
            total -= c6[i][j] * (s6 * t6 + s8 * rrij * t8)
    return total


def energy_atm(mol, c6, par):
    if abs(par["s9"]) < 1e-16:
        return 0.0
    total = 0.0
    n = len(mol)
    alp3 = par["alp"] / 3.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                rij = dist(mol, i, j)
                rik = dist(mol, i, k)
                rjk = dist(mol, j, k)
                r2ij, r2ik, r2jk = rij * rij, rik * rik, rjk * rjk
                c9 = -par["s9"] * math.sqrt(
                    abs(c6[i][j] * c6[i][k] * c6[j][k])
                )
                r0ij = par["a1"] * math.sqrt(
                    3 * el[mol[i]["Z"]]["r4r2"] * el[mol[j]["Z"]]["r4r2"]
                ) + par["a2"]
                r0ik = par["a1"] * math.sqrt(
                    3 * el[mol[i]["Z"]]["r4r2"] * el[mol[k]["Z"]]["r4r2"]
                ) + par["a2"]
                r0jk = par["a1"] * math.sqrt(
                    3 * el[mol[j]["Z"]]["r4r2"] * el[mol[k]["Z"]]["r4r2"]
                ) + par["a2"]
                r0 = r0ij * r0ik * r0jk
                r1 = rij * rik * rjk
                r3 = r1**3
                r5 = r3 * r1 * r1
                fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** alp3)
                ang = (
                    0.375
                    * (r2ij + r2jk - r2ik)
                    * (r2ij - r2jk + r2ik)
                    * (-r2ij + r2jk + r2ik)
                    / r5
                    + 1.0 / r3
                )
                # get_atm_dispersion_energy: dE = rr*c9*triple/3 per atom,
                # three atoms each receive -dE  ->  delta E = -rr*c9
                total += -(ang * fdmp) * c9
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimer", default=None)
    args = parser.parse_args()

    geometry = {}
    with GEOMETRY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            geometry.setdefault(row["dimer"], []).append(row)

    targets = {}
    with PROPERTIES.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            targets.setdefault(row["dimer"], []).append(row)

    energies = {}
    with DISPERSION.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            energies.setdefault(row["dimer"], []).append(row)

    dimers = [args.dimer] if args.dimer else sorted(targets)
    worst = {}
    worst_rel = {}
    for dimer in dimers:
        atoms = geometry[dimer]
        mol = [
            {
                "Z": sym[a["element"]],
                "xyz": [
                    float(a["x_angstrom"]) * AA2BOHR,
                    float(a["y_angstrom"]) * AA2BOHR,
                    float(a["z_angstrom"]) * AA2BOHR,
                ],
            }
            for a in atoms
        ]
        total_charge = float(targets[dimer][0]["charge"])
        q = eeq_charges(mol, total_charge)
        cn = d4_cn(mol)
        weights = reference_weights(mol, q, cn)
        c6 = c6_matrix(mol, weights)
        # disp.f90:139-141 resets q to zero before rebuilding the C6 used by the
        # three-body term, so the ATM C6 are weighted with q = 0, not with the
        # molecular EEQ charges.
        c6_zero = c6_matrix(mol, reference_weights(mol, [0.0] * len(mol), cn))
        alpha = [
            sum(
                weights[i][r] * el[mol[i]["Z"]]["references"][r]["alphaiw_eeq"][0]
                for r in range(len(weights[i]))
            )
            for i in range(len(mol))
        ]
        for i, row in enumerate(targets[dimer]):
            ref = {
                "cn": float(row["coordination_number"]),
                "q": float(row["eeq_charge"]),
                "alpha": float(row["polarizability"]),
                "c6": float(row["c6_to_first_atom"]),
            }
            got = {"cn": cn[i], "q": q[i], "alpha": alpha[i], "c6": c6[i][0]}
            for key in ("cn", "q", "alpha", "c6"):
                dev = abs(got[key] - ref[key])
                if dev > worst.get(key, 0.0):
                    worst[key] = dev
                    worst_rel[key] = dev / abs(ref[key]) if ref[key] else 0.0
        for row in energies[dimer]:
            if abs(float(row["charge"])) > 0.5:
                continue  # staged properties are for the neutral dimers only
            method = METHOD_MAP.get(row["method"], row["method"])
            par = damp["functionals"].get(method)
            if par is None or par.get("absent_upstream"):
                continue
            e2 = energy_two_body(mol, c6, par)
            e3 = energy_atm(mol, c6_zero, par)
            ref_e = float(row["energy_hartree"])
            dev = abs((e2 + e3) - ref_e)
            key = "energy"
            if dev > worst.get(key, 0.0):
                worst[key] = dev
                worst_rel[key] = dev / abs(ref_e) if ref_e else 0.0

    print(f"aatoau used: {AA2BOHR!r}")
    fails = 0
    for key, label in (
        ("cn", "coordination number"),
        ("q", "EEQ charge"),
        ("alpha", "polarizability"),
        ("c6", "C6 to first atom"),
        ("energy", "two-body + ATM energy"),
    ):
        if key not in worst:
            print(f"{label:22s} not compared")
            continue
        dev = worst[key]
        rel = worst_rel[key]
        status = "OK" if rel < 1e-6 else "MISMATCH"
        print(f"{label:22s} max |dev| = {dev:.3e}  max rel = {rel:.3e}   {status}")
        if status != "OK":
            fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
