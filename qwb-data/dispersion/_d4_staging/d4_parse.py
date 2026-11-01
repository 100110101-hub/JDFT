#!/usr/bin/env python3
"""Shared parsing helpers for the DFT-D4 staging extractors.

The helpers here only read files that already exist under ``raw/``.  They never
transcribe numbers by hand: every literal comes from the pinned upstream text.

Used by ``extract_d4_reference.py`` and ``extract_eeq2019.py``.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

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
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
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


def parse_fortran_array(path: Path, name: str, expected: int):
    """Parse ``real(wp), parameter :: name(max_elem|23) = [ ... ]``.

    Returns a dict with values, per-value line numbers and block line range.
    """
    text, starts = clean_with_linemap(path)
    decl = re.search(
        r"parameter\s*::\s*" + re.escape(name) + r"\s*\(\s*[\w:]+\s*\)\s*=", text
    )
    if decl is None:
        raise SystemExit(f"array {name} not found in {path}")
    decl_line = line_of(starts, decl.start())
    open_bracket = text.index("[", decl.end())
    prefix = text[decl.end():open_bracket].strip()
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
    if expected is not None and len(values) != expected:
        raise SystemExit(
            f"{name}: parsed {len(values)} values, expected {expected} ({path})"
        )
    return {
        "values": values,
        "lines": lines,
        "decl_line": decl_line,
        "first_line": line_of(starts, open_bracket),
        "last_line": line_of(starts, close_bracket),
        "prefix": prefix,
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "array": name,
    }


def parse_scalar_parameter(path: Path, name: str):
    """Parse ``real(wp), parameter :: name = 1.0_wp`` -> (value, line)."""
    text, starts = clean_with_linemap(path)
    match = re.search(
        r"parameter\s*::\s*"
        + re.escape(name)
        + r"\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?)(?:_wp)?",
        text,
    )
    if match is None:
        raise SystemExit(f"scalar {name} not found in {path}")
    return float(match.group(1)), line_of(starts, match.start())


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_or_check(path: Path, text: str, check: bool) -> bool:
    """Write ``text`` to ``path``, or in check mode compare byte-for-byte."""
    if check:
        if not path.exists():
            print(f"FAIL {path.name}: missing")
            return False
        if path.read_text(encoding="utf-8") == text:
            print(f"OK   {path.name}")
            return True
        print(f"FAIL {path.name}: content differs from regenerated output")
        return False
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.name}")
    return True
