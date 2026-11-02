"""VF-02 的一部分：FreeSolv 的水相实验水合自由能（CC0，可再分发）。

为什么是它而不是 MNSOL：`project-phase2.md` §6 的 VF-02 要「8 溶剂 × 3 分子」，权威来源是
Minnesota Solvation Database（v2012，3037 个实验 ΔG_solv、790 个溶质、92 个溶剂，DOI
10.13020/3eks-j059）。但它的下载页现在被 Azure WAF 挡在 403（不可下载），而它的许可只对
非商业用途免费、商业用途要付费 —— 所以**不能**把原始数据进仓库。

FreeSolv（Mobley & Guthrie, J. Comput. Aided Mol. Des. 28, 711 (2014), DOI
10.1007/s10822-014-9747-x）是**公共领域（CC0）**的 642 个水相水合自由能，每个值自带它引用的
原始实验文献 DOI。于是这一轮交付 VF-02 的**水一列**（甲醇、乙腈），其余七个溶剂如实记为
blocked。

标准态：FreeSolv 的 `expt` 是 1 M 气体 ↔ 1 M 溶液的**水合自由能**，与本项目的
`SolvationResult::total_hartree`（= `G_EP + G_CDS`，1 M ↔ 1 M）同一个约定 —— 不再加
`RT ln(RT/p°)`。这一条写在 CSV 的 `standard_state` 列里，比较时不靠记忆。

用法：

    python scripts/gen_freesolv_reference.py            # 重新生成 CSV
    python scripts/gen_freesolv_reference.py --check    # 只比对（需要 staging 里的原始 JSON）
"""

import argparse
import hashlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import FREESOLV_DOI as _FREESOLV_DOI, FREESOLV_URL as _FREESOLV_URL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "qwb-data", "reference", "_freesolv_staging", "raw", "database.json")
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "solvation_free_energy.csv"
)

FREESOLV_DOI = _FREESOLV_DOI
FREESOLV_URL = _FREESOLV_URL

#: (溶质名, SMILES, 本项目用的 qwb 输入里的元素集) —— 只取我们能在 STO-3G 上算的小分子。
SOLUTES = [
    ("methanol", "CO"),
    ("acetonitrile", "CC#N"),
]

#: 标准态说明：FreeSolv 的 expt 是 1 M ↔ 1 M，与本项目 total_hartree 同一约定。
STANDARD_STATE = "1M_gas<->1M_solution"

HEADER = [
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


def raw_sha256():
    digest = hashlib.sha256()
    with open(RAW, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows():
    import json

    if not os.path.exists(RAW):
        raise SystemExit(
            f"missing {os.path.relpath(RAW, ROOT)}; fetch it with "
            "curl -L https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json"
        )
    with io.open(RAW, encoding="utf-8") as handle:
        database = json.load(handle)
    digest = raw_sha256()
    out = []
    for solute, smiles in SOLUTES:
        matches = [
            (identifier, record)
            for identifier, record in database.items()
            if (record.get("smiles") or "").strip() == smiles
        ]
        if len(matches) != 1:
            raise SystemExit(f"{solute}: expected exactly one FreeSolv entry for {smiles}, got {len(matches)}")
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


def render(data):
    lines = [",".join(HEADER)]
    for row in data:
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    args = parser.parse_args()

    text = render(rows())
    if args.check:
        current = io.open(OUTPUT, encoding="utf-8").read() if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"FAIL {os.path.relpath(OUTPUT, ROOT)} does not match the staged FreeSolv snapshot")
            return 1
        print(f"ok: {os.path.relpath(OUTPUT, ROOT)} matches FreeSolv ({len(text.splitlines()) - 1} rows)")
        return 0

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(text)
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)} ({len(text.splitlines()) - 1} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
