"""SV-03 / VF-02 的参考家族：FreeSolv 的水相实验水合自由能（CC0，可再分发）。

为什么是 FreeSolv 而不是 MNSOL：`project-phase2.md` §6 的 VF-02 要「8 溶剂 × 3 分子」，
权威来源是 Minnesota Solvation Database（v2012，3037 个实验 ΔG_solv、790 个溶质、92 个溶剂，
DOI 10.13020/3eks-j059）。但它的下载页现在被 Azure WAF 挡在 403（不可下载），许可也只对
非商业用途免费 —— 所以**不**把原始数据进仓库。FreeSolv（Mobley & Guthrie, J. Comput. Aided
Mol. Des. 28, 711 (2014)，DOI `FREESOLV_DOI`）是 **CC0** 的 642 个水相水合自由能，每个值
自带它引用的原始实验文献 —— 这一份产出**水这一列**的完整参考家族。

标准态：FreeSolv 的 `expt` 是 1 M 气体 ↔ 1 M 溶液的**水合自由能**，与本项目
`SolvationResult::total_hartree`（= `G_EP + G_CDS`，1 M ↔ 1 M）同一个约定 —— 不再加
`RT ln(RT/p°)`。这条写在 CSV 的 `standard_state` 列里，比较时不靠记忆。FreeSolv 的记录里
**没有**逐分子的温度字段（只有 `expt` 与它引用的文献），所以这一份不写 `temperature_k`
列：与其抄一个看起来很像的 298.15，不如承认这个数是 FreeSolv 的、并按它的约定比较。

两个产物（同一个 staging 快照）：

* `solvation_free_energy.csv` —— VF-02 的「水一列」（甲醇、乙腈 2 行，历史行不动）；
* `freesolv_hydration.csv` —— 本家族的**全量受限子集**：642 条里只留
  「元素 ⊆ {H, C, N, O, F} + 单一分子 + 净电荷 0」的那些（见 `filters`）。
  选这个元素集是因为基组覆盖面（2026-11-15 逐文件读 `qwb-data/basis/`）：
  `sto-3g` 与 `def2-tzvp` 有 H/C/N/O/F，`def2-svp`、`def2-svp-ri`、`6-311g-dp` 只有
  H/C/N/O（**没有 F**），`lanl2dz` 只有 H/O/Cu/Zn —— 所以 F 只在 STO-3G（和 def2-TZVP）
  上跑得动，这一点写进了报告，没有假设。

原子数/分子式怎么来的：FreeSolv 的 `smiles` 是唯一的结构真相源，但它不带原子数。这里用
脚本内的 SMILES 解析器（有机子集 + bracket 原子 + 环闭合/支链/键级/电荷）算分子式与总
原子数（含隐式氢）。这不是「大概差不多」：解析结果在开发时与 PubChem 的
`MolecularFormula` **逐条**对照过 641/642 条（第 642 条 `mobley_9112978` 的 PubChemID
指向 PubChem 的混合物记录 "ethane;prop-1-ene"，与它的 SMILES/`iupac` 不符 —— 那是
FreeSolv 的元数据瑕疵，分子式仍按 SMILES 取 C5H10）。对照脚本是一次性的外部校验，
不进门禁（门禁不联网）。

用法：

    python scripts/gen_freesolv_reference.py            # 重新生成两份 CSV
    python scripts/gen_freesolv_reference.py --check    # 只比对（需要 staging 里的原始 JSON）
"""

import argparse
import collections
import csv
import hashlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import FREESOLV_DOI as _FREESOLV_DOI, FREESOLV_URL as _FREESOLV_URL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "qwb-data", "reference", "_freesolv_staging", "raw", "database.json")

FREESOLV_DOI = _FREESOLV_DOI
FREESOLV_URL = _FREESOLV_URL

#: staging 快照的钉（字节数 + sha256）。**不匹配必须炸掉**：上游仓库的
#: `database.json` 是可变的，若静默接受另一份文件，`--check` 就在证明一个不属于
#: 这个 pin 的 CSV。2026-11-15 复核过（620,456 B）。
PINNED_BYTES = 620_456
PINNED_SHA256 = "9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a"

#: VF-02 的水一列（历史文件，行不动；`sv03_smd_engine` 与 `docs/status-v2.yaml` 引用它）。
LEGACY_OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "solvation_free_energy.csv"
)
#: SV-03 的 FreeSolv 全量受限子集。
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "freesolv_hydration.csv"
)

#: (溶质名, SMILES) —— 只取 VF-02 点名的那两个分子。
SOLUTES = [
    ("methanol", "CO"),
    ("acetonitrile", "CC#N"),
]

#: 标准态说明：FreeSolv 的 expt 是 1 M ↔ 1 M，与本项目 total_hartree 同一约定。
STANDARD_STATE = "1M_gas<->1M_solution"

LEGACY_HEADER = [
    "solute",
    "solvent",
    "quantity",
    "value_kcal_per_mol",
    "temperature_k",
    "standard_state",
    "freesolv_id",
    "expt_reference_doi",
    "raw_sha256",
    "source_doi",
    "source_url",
]

#: 新家族的列：标识（freesolv_id）、结构（smiles，**按上游原样**）、分子式/原子数、
#: 实验值 + 它的原始文献、pin、以及 DA-08 要求的来源两列（固定在最后）。
FAMILY_FIELDS = [
    "freesolv_id",
    "smiles",
    "iupac",
    "formula",
    "atom_count",
    "net_charge",
    "dg_hyd_kcal_per_mol",
    "standard_state",
    "expt_reference",
    "raw_sha256",
    "source_doi",
    "source_url",
]

#: `filters` 的判据：元素只能来自这套（见模块 docstring 里基组覆盖面的实测）。
ALLOWED_ELEMENTS = ("H", "C", "N", "O", "F")

# ---------------------------------------------------------------------------
# 极简 SMILES 解析（只为本家族的分子式/原子数服务；无 3D、无立体化学）
# ---------------------------------------------------------------------------

_ORGANIC = ("Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I")
_AROMATIC = ("c", "n", "o", "s", "p", "b")
_BOND = {"-": 1.0, "=": 2.0, "#": 3.0, "$": 4.0, ":": 1.5, "/": 1.0, "\\": 1.0}
_VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1, "B": 3, "P": 3, "S": 2}
_BRACKET = re.compile(
    r"^(?P<iso>\d+)?(?P<sym>[A-Z][a-z]?|[a-z][a-z]?)"
    r"(?P<chir>@{1,2})?(?P<h>H\d*)?(?P<chg>(?:\d*[+-]+)|(?:[+-]\d+))?(?::(?P<cls>\d+))?$"
)


def parse_smiles(smiles):
    """返回 `(元素计数 Counter, 净形式电荷, [价键溢出说明])`。

    隐式氢只为有机子集的**裸原子**补（bracket 原子按 SMILES 语义只用显式 H）。
    芳香原子按「芳香键 = 1.5 级」记账：`c` 只在「恰好两个环邻居、键级和 3.0」时带 1 个 H
    （苯/萘/取代位置都对得上）；`n`/`o`/`s`/`p` 不带隐式 H（吡咯型的 H 在 FreeSolv 里
    一律写成 `[nH]`）。这两条不是猜的：642 条 SMILES 的分子式与 PubChem 逐条对过。
    """
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("empty SMILES")
    atoms = []
    bonds = []
    ring = {}
    branch = []
    prev = None
    pending = None
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if char == "(":
            branch.append(prev)
            i += 1
            continue
        if char == ")":
            if not branch:
                raise ValueError(f"unbalanced ')' in {smiles!r}")
            prev = branch.pop()
            pending = None
            i += 1
            continue
        if char in _BOND:
            pending = _BOND[char]
            i += 1
            continue
        if char.isdigit() or char == "%":
            if char == "%":
                label = smiles[i + 1:i + 3]
                i += 3
            else:
                label = char
                i += 1
            if label in ring:
                other, order = ring.pop(label)
                if pending is not None:
                    order = pending
                if order is None:
                    order = 1.5 if (atoms[other]["aromatic"] and atoms[prev]["aromatic"]) else 1.0
                bonds.append((other, prev, order))
            else:
                ring[label] = (prev, pending)
            pending = None
            continue
        if char == "[":
            close = smiles.find("]", i)
            if close < 0:
                raise ValueError(f"unclosed '[' in {smiles!r}")
            body = smiles[i + 1:close]
            match = _BRACKET.match(body)
            if not match:
                raise ValueError(f"unsupported bracket atom [{body}] in {smiles!r}")
            symbol = match.group("sym")
            explicit_h = match.group("h")
            hcount = 0
            if explicit_h:
                hcount = int(explicit_h[1:]) if explicit_h[1:] else 1
            charge_text = match.group("chg") or ""
            charge = 0
            if charge_text:
                if charge_text[0].isdigit() or charge_text[-1].isdigit():
                    digits = "".join(c for c in charge_text if c.isdigit())
                    sign = 1 if "+" in charge_text else -1
                    charge = sign * (int(digits) if digits else 1)
                else:
                    charge = charge_text.count("+") - charge_text.count("-")
            atoms.append(
                {
                    "sym": symbol.capitalize() if symbol.islower() else symbol,
                    "h": hcount,
                    "charge": charge,
                    "aromatic": symbol.islower(),
                }
            )
            i = close + 1
        else:
            two = smiles[i:i + 2]
            if two in _ORGANIC:
                atoms.append({"sym": two, "h": None, "charge": 0, "aromatic": False})
                i += 2
            elif char in _ORGANIC:
                atoms.append({"sym": char, "h": None, "charge": 0, "aromatic": False})
                i += 1
            elif char in _AROMATIC:
                atoms.append({"sym": char.upper(), "h": None, "charge": 0, "aromatic": True})
                i += 1
            else:
                raise ValueError(f"unsupported character {char!r} at {i} in {smiles!r}")
        if prev is not None:
            order = pending
            if order is None:
                order = 1.5 if (atoms[prev]["aromatic"] and atoms[-1]["aromatic"]) else 1.0
            bonds.append((prev, len(atoms) - 1, order))
            pending = None
        prev = len(atoms) - 1
    if ring:
        raise ValueError(f"unclosed ring {sorted(ring)} in {smiles!r}")
    if branch:
        raise ValueError(f"unclosed branch in {smiles!r}")

    problems = []
    for index, atom in enumerate(atoms):
        if atom["h"] is not None:
            continue
        total = sum(order for (a, b, order) in bonds if a == index or b == index)
        if atom["aromatic"]:
            neighbours = sum(1 for (a, b, _) in bonds if a == index or b == index)
            if atom["sym"] == "C":
                atom["h"] = 1 if (neighbours == 2 and abs(total - 3.0) < 1e-9) else 0
            else:
                atom["h"] = 0
        else:
            atom["h"] = int(round(_VALENCE.get(atom["sym"], 0) - total))
        if atom["h"] < 0:
            # 中性硝基有两种写法：`[N+](=O)[O-]` 与 hypervalent 的 `N(=O)=O`（键级和 5）。
            # 后者不是错误，只是没有把电荷写出来 —— 隐式 H 同样为 0，分子式不受影响
            # （与 PubChem 的分子式逐条对过），所以不记为 problem。
            if atom["sym"] == "N" and abs(total - 5.0) < 1e-9:
                atom["h"] = 0
                continue
            problems.append(f"{smiles!r}: {atom['sym']} valence overflow (bond order {total})")
            atom["h"] = 0
    counts = collections.Counter()
    for atom in atoms:
        counts[atom["sym"]] += 1
        counts["H"] += atom["h"]
    net_charge = sum(atom["charge"] for atom in atoms)
    return counts, net_charge, problems


def molecular_formula(counts):
    """Hill 序分子式（含 H，例如 `C2H4O2`）；0 个的元素不写。"""
    counts = {symbol: number for symbol, number in counts.items() if number}
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(symbol for symbol in counts if symbol not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}" for symbol in order
    )


# ---------------------------------------------------------------------------
# staging 快照与两个产物
# ---------------------------------------------------------------------------


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_database():
    """读 staging 快照，**先核对 pin** —— 不一致就炸，不静默接受别的上游文件。"""
    if not os.path.exists(RAW):
        raise SystemExit(
            f"missing {os.path.relpath(RAW, ROOT)}; fetch it with "
            "curl -L https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json"
        )
    size = os.path.getsize(RAW)
    digest = file_sha256(RAW)
    if size != PINNED_BYTES or digest != PINNED_SHA256:
        raise SystemExit(
            "FREESOLV PIN MISMATCH: {path}\n  expected {want_bytes} B / sha256 {want}\n"
            "  actual   {got_bytes} B / sha256 {got}\n"
            "This is NOT the FreeSolv snapshot this family was generated from. Either restore the "
            "pinned file or update PINNED_BYTES/PINNED_SHA256 together with a regenerated family "
            "and a note about what changed upstream.".format(
                path=os.path.relpath(RAW, ROOT),
                want_bytes=PINNED_BYTES,
                want=PINNED_SHA256,
                got_bytes=size,
                got=digest,
            )
        )
    import json

    with io.open(RAW, encoding="utf-8") as handle:
        database = json.load(handle)
    return database, digest


def legacy_rows(database, digest):
    """VF-02 的水一列（2 行）—— 保持历史输出逐字节不变。"""
    out = []
    for solute, smiles in SOLUTES:
        matches = [
            (identifier, record)
            for identifier, record in database.items()
            if (record.get("smiles") or "").strip() == smiles
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"{solute}: expected exactly one FreeSolv entry for {smiles}, got {len(matches)}"
            )
        identifier, record = matches[0]
        experimental = record["expt"]
        if not isinstance(experimental, (int, float)):
            raise SystemExit(f"{solute}: expt is not a number ({experimental!r})")
        out.append(
            [
                solute,
                "water",
                "dg_hydration_experimental",
                f"{experimental:.2f}",
                "298.15",
                STANDARD_STATE,
                identifier,
                str(record.get("expt_reference", "")),
                digest,
                FREESOLV_DOI,
                FREESOLV_URL,
            ]
        )
    return out


def render_legacy(data):
    lines = [",".join(LEGACY_HEADER)]
    for row in data:
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def filters(database, digest):
    """按固定顺序过筛，返回 `(保留的行 dict 列表, 每一步的计数)`。

    顺序就是报告里的级联：全部 → 解析 → 单一分子 → 元素 → 净电荷 → 数值。
    """
    steps = collections.OrderedDict()
    steps["records"] = len(database)
    parsed = []
    for identifier, record in database.items():
        counts, charge, problems = parse_smiles(record["smiles"])
        parsed.append((identifier, record, counts, charge, problems))

    single = [row for row in parsed if "." not in row[1]["smiles"]]
    steps["dropped_multi_fragment"] = len(parsed) - len(single)
    allowed = [row for row in single if set(row[2]) - {"H"} <= set(ALLOWED_ELEMENTS)]
    steps["dropped_elements"] = len(single) - len(allowed)
    neutral = [row for row in allowed if row[3] == 0]
    steps["dropped_net_charge"] = len(allowed) - len(neutral)
    numeric = [row for row in neutral if isinstance(row[1].get("expt"), (int, float))]
    steps["dropped_non_numeric_expt"] = len(neutral) - len(numeric)
    steps["kept"] = len(numeric)

    # 价键记账的 problem 只在**保留下来**的记录上才是致命的：被元素筛掉的那些
    # （P/S/卤素）用不上它们的分子式，而 5 价/6 价 P、S 在 SMILES 里是合法写法。
    broken = [row[0] for row in numeric if row[4]]
    if broken:
        raise SystemExit(
            f"SMILES valence problems in {len(broken)} kept record(s): {broken[:5]}"
        )
    ignored = sum(1 for row in parsed if row[4])
    if ignored:
        print(f"note: {ignored} non-kept record(s) carry valence-convention notes (P/S/halogen)")

    rows = []
    for identifier, record, counts, charge, problems in numeric:
        rows.append(
            {
                "freesolv_id": identifier,
                "smiles": record["smiles"],
                "iupac": record.get("iupac") or "",
                "formula": molecular_formula(counts),
                "atom_count": sum(counts.values()),
                "net_charge": charge,
                "dg_hyd_kcal_per_mol": f"{record['expt']:.2f}",
                "standard_state": STANDARD_STATE,
                "expt_reference": record.get("expt_reference") or "",
                "raw_sha256": digest,
                "source_doi": FREESOLV_DOI,
                "source_url": FREESOLV_URL,
            }
        )
    rows.sort(key=lambda row: row["freesolv_id"])
    return rows, steps


def render_family(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FAMILY_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def compare(path, text, label):
    current = io.open(path, encoding="utf-8", newline="").read() if os.path.exists(path) else ""
    if current == text:
        rows = max(0, len(text.splitlines()) - 1)
        print(f"ok: {os.path.relpath(path, ROOT)} matches {label} ({rows} rows)")
        return True
    got = current.splitlines()
    want = text.splitlines()
    for index in range(max(len(got), len(want))):
        left = got[index] if index < len(got) else "<missing>"
        right = want[index] if index < len(want) else "<missing>"
        if left != right:
            print(
                f"STALE {os.path.relpath(path, ROOT)} line {index + 1}:\n"
                f"  file: {left}\n  gen : {right}"
            )
            break
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    args = parser.parse_args()

    database, digest = load_database()

    legacy_text = render_legacy(legacy_rows(database, digest))
    rows, steps = filters(database, digest)
    family_text = render_family(rows)

    print(
        "FreeSolv filter cascade (642 upstream records -> this family): "
        + ", ".join(f"{name}={count}" for name, count in steps.items())
    )
    print(
        f"  elements allowed {{H, C, N, O, F}}; snapshot {PINNED_BYTES} B sha256 {digest}"
    )
    print(
        "  atom-count distribution: "
        + ", ".join(
            f"{atoms} atoms={count}"
            for atoms, count in sorted(
                collections.Counter(row["atom_count"] for row in rows).items()
            )
        )
    )

    if args.check:
        ok = compare(LEGACY_OUTPUT, legacy_text, "FreeSolv")
        ok = compare(OUTPUT, family_text, "FreeSolv") and ok
        return 0 if ok else 1

    for path, text in ((LEGACY_OUTPUT, legacy_text), (OUTPUT, family_text)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        io.open(path, "w", encoding="utf-8", newline="\n").write(text)
        print(f"wrote {os.path.relpath(path, ROOT)} ({len(text.splitlines()) - 1} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
