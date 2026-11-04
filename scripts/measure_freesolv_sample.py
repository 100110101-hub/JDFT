"""SV-03：用 FreeSolv 的受限子集**实测** SMD 水合自由能的误差分布。

`scripts/measure_smd_dgsolv.py` 只对 VF-02 的两个手搭分子（甲醇、乙腈）对账 —— 两个点
算不出分布，SV-03 的 MAE 判据因此长期只能"看着像"。这一份把 `freesolv_hydration.csv`
（642 → 457 的受限子集）当参考，跑**一段显式定义的样本**，给出逐分子误差 + MAE/RMSE/最大偏差。

样本定义（可复现、没有主观挑选）
--------------------------------
`atom_count <= 8` 的全部行，按 `(atom_count, freesolv_id)` 排序。这是 457 行里原子数最少
的一段：4–8 个原子、17 个分子。用"最小的一段"而不是随机抽样，是为了把 HF/STO-3G 的成本
限制在可复现的量级里；代价是样本**偏小分子**，不代表 457 行的整体分布 —— 这一点在
输出的 NOTE 里说，不假装它是随机样本。

SMD 协议（与 `measure_smd_dgsolv.py` 逐字节一致，reuse 它的 driver）
------------------------------------------------------------------
气相几何优化（`jobtype: opt`，真空）→ 在该几何上做一次 SMD 单点（`solvent_model: smd`）
→ `ΔG_hyd = solvation.total_hartree × 627.5094740636`。`total_hartree` 是 **1 M ↔ 1 M**
的转移能，与 FreeSolv 的 `expt` 同一个约定（CSV 的 `standard_state` 列），所以**不加**
`RT ln(RT/p°)`。

起始几何从哪来（不是手搭的）
--------------------------
PubChem 的 3D conformer（PUG-REST `record_type=3d`，按 FreeSolv 记录里的 `PubChemID` 取）。
坐标摘要（「元素 + 4 位小数坐标」的 sha256，见 `coordinates_digest`）钉在
`GEOMETRY_PINS` 里：staging 里没有就重新取，取回的摘要与 pin 不符就**炸**（PubChem 改过
conformer 时不能静默换成另一个起点）。PubChem 的几何只当**起点**，每一行随后都由本项目
自己的优化器在目标层级上重新优化。SDF 的原子/元素计数还会与 CSV 的 `atom_count`/`formula`
核对 —— 这是原子数列的独立校验。

显式排除（不静默丢）
------------------
跑不动的分子（基组缺元素、SCF 不收敛、几何取不到/摘要不符）逐条打印原因并列出，
最后单独统计 `excluded`，不混进 MAE。

用法：

    python scripts/measure_freesolv_sample.py                    # 17 个分子，HF/STO-3G
    python scripts/measure_freesolv_sample.py --basis def2-svp --method pbe
    python scripts/measure_freesolv_sample.py --max-atoms 9      # 26 个分子（更慢）
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_smd_dgsolv as driver  # noqa: E402  （同一套 write_input/read_xyz/run_qwb）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "freesolv_hydration.csv"
)
STAGING = os.path.join(ROOT, "qwb-data", "reference", "_freesolv_staging")
DATABASE = os.path.join(STAGING, "raw", "database.json")
GEOMETRY_DIR = os.path.join(STAGING, "geom3d")

HARTREE_PER_KCAL = 627.509_474_063_6
#: `cargo run` 的构建档：debug 下这几步慢一个数量级（实测：同一个 SMD 单点，debug 与
#: release 走同一条代码路径）。物理层级不变，档位写进输出。
PROFILE = "release"
#: 样本的上界：`atom_count <= SAMPLE_MAX_ATOMS`（默认 8 → 17 个分子，见模块 docstring）。
SAMPLE_MAX_ATOMS = 8

#: PubChem 3D conformer 的坐标摘要（`coordinates_digest` 的定义见下）：元素 + 4 位小数坐标。
#: 2026-11-15 取回；`--refresh-geometry` 之外的路径都必须命中这里。
GEOMETRY_PINS = {
    "mobley_1636752": "6bd6dc62a2b28b2fa8263da9784750315dd73caabb35cc14aba22c789e108d1d",
    "mobley_1717215": "79135b1f23f3c5725df8fa23ef622cd505110e1dc870474d4d966e722be2eee1",
    "mobley_1952272": "54d03f6c61a7a8e9c802c2ee928ed984b4c33026a148dc2bc5e4dce10e3cced9",
    "mobley_1967551": "1538b53e5f78d44842533243846e26ffec01e325d267f1718c357adf663fa857",
    "mobley_2008055": "93a5417d4009525b02b2af3a52748a2e7bd6aff5f9ef056d2fca0008f6caa558",
    "mobley_2146331": "8cc963c8be45a0f098421bd406c4d42f8da90e403d70ef94c221acf18871ab8c",
    "mobley_3034976": "36fff66364e76c60414aaaee9c4653b17f4531c9d2846fcf66cd0a6071111ba3",
    "mobley_3370989": "811a7ea4e0260f6363a8c4efee8aa2db5429e6d9da7bdc4e1778d58a7ab64f56",
    "mobley_5631798": "e12f485b6d4a0425382f88be94f47f5ad16236b243e3b327071f3015aa4f33b5",
    "mobley_6091882": "a5a7c7c479c38136385eda2d556927a5521711b54c0e3950749f82a37ece5e17",
    "mobley_6714389": "5c6682e133f37faebe74ec4ec5f000e9bd13bc3edcf06333bdc855e1a40d8bf2",
    "mobley_676247": "d1c3f110dd46a6686b824c177330348188dc8ee1a8db066f681b7cd87738e639",
    "mobley_7261305": "de6d1ffcb9ccac294b4f6c2fe94997a730f26fac31558a33aa6910dedb19178e",
    "mobley_7532833": "de3c0c53b78892dff3464f0a5522e57f05b3bb0d80ac373fc25c34ffcdbfe69f",
    "mobley_8260524": "5874368cb9eac848eda5c9db51aebbc78fa12c764ab7c7f7f29f2e43e603c6a4",
    "mobley_9029594": "3f3d5b53ee98707b4105790a9ff6d6a4b8d379090ba88fd7066e6f316351819a",
    "mobley_9055303": "75157f8063028229fda0404ab6b6ac299d800a0e73766eac0f72513bb8e9c61a",
}


def family_rows():
    import csv

    with io.open(FAMILY, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_sample(rows, max_atoms):
    chosen = [row for row in rows if int(row["atom_count"]) <= max_atoms]
    chosen.sort(key=lambda row: (int(row["atom_count"]), row["freesolv_id"]))
    return chosen


def formula_counts(text):
    counts = {}
    for symbol, number in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + (int(number) if number else 1)
    return counts


def sdf_atoms(text):
    """V2000 SDF → `[(symbol, x, y, z)]`（只取元素与坐标，不解析键）。"""
    lines = text.splitlines()
    if len(lines) < 5 or "V2000" not in lines[3]:
        raise ValueError("not a V2000 SDF")
    n_atoms = int(lines[3][0:3])
    atoms = []
    for line in lines[4:4 + n_atoms]:
        symbol = line[31:34].strip()
        atoms.append((symbol, float(line[0:10]), float(line[10:20]), float(line[20:30])))
    return atoms


def coordinates_digest(atoms):
    digest = hashlib.sha256()
    for symbol, x, y, z in atoms:
        digest.update(f"{symbol} {x:.4f} {y:.4f} {z:.4f}\n".encode())
    return digest.hexdigest()


def fetch_geometry(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/SDF?record_type=3d"
    with urllib.request.urlopen(url, timeout=120) as handle:
        payload = handle.read()
    os.makedirs(GEOMETRY_DIR, exist_ok=True)
    with open(os.path.join(GEOMETRY_DIR, f"{int(cid)}.sdf"), "wb") as handle:
        handle.write(payload)
    return payload.decode("utf-8", "replace")


def starting_geometry(row, cid):
    """读（或取回并 pin 住）PubChem 的 3D conformer，并核对元素计数。"""
    path = os.path.join(GEOMETRY_DIR, f"{int(cid)}.sdf")
    if os.path.exists(path):
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", "replace")
    else:
        text = fetch_geometry(cid)
    atoms = sdf_atoms(text)
    digest = coordinates_digest(atoms)
    want = GEOMETRY_PINS.get(row["freesolv_id"])
    if want is None:
        raise ValueError(f"no geometry pin for {row['freesolv_id']}; add one before measuring")
    if digest != want:
        raise ValueError(
            f"PubChem conformer for CID {cid} changed: coordinates digest {digest} != pinned {want}"
        )
    seen = {}
    for symbol, *_ in atoms:
        seen[symbol] = seen.get(symbol, 0) + 1
    expected = formula_counts(row["formula"])
    if seen != expected or len(atoms) != int(row["atom_count"]):
        raise ValueError(
            f"geometry/CSV disagree: SDF {len(atoms)} atoms {seen} vs CSV "
            f"{row['atom_count']} atoms {expected}"
        )
    return [(symbol, (x, y, z)) for symbol, x, y, z in atoms]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--method", default="hf")
    parser.add_argument("--max-atoms", type=int, default=SAMPLE_MAX_ATOMS)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    args = parser.parse_args()

    if not os.path.exists(DATABASE):
        raise SystemExit(f"missing {os.path.relpath(DATABASE, ROOT)} (the pinned FreeSolv snapshot)")
    with io.open(DATABASE, encoding="utf-8") as handle:
        database = json.load(handle)

    rows = family_rows()
    sample = select_sample(rows, args.max_atoms)
    if args.limit:
        sample = sample[: args.limit]
    print(f"family: {len(rows)} rows; sample: atom_count <= {args.max_atoms} -> {len(sample)} molecules")
    print(
        f"level: {args.method.upper()}/{args.basis} ({PROFILE} build); "
        "geometry: qwb opt (vacuum) then SMD single point"
    )
    print(
        f"{'id':16s} {'formula':9s} {'N':>2s} {'this work':>10s} {'experiment':>11s} {'error':>8s}"
    )

    workdir = tempfile.mkdtemp(prefix="qwb-freesolv-")
    deviations = []
    measured = []
    excluded = []
    for row in sample:
        identifier = row["freesolv_id"]
        cid = database[identifier]["PubChemID"]
        try:
            atoms = starting_geometry(row, cid)
            driver.write_input(
                os.path.join(workdir, f"{identifier}_opt.yaml"),
                atoms,
                args.basis,
                args.method,
                "opt",
                "vacuum",
                None,
            )
            driver.run_qwb(workdir, f"{identifier}_opt", profile=PROFILE)
            optimized = driver.read_xyz(
                os.path.join(workdir, "out", f"{identifier}_opt", "artifacts", "optimized.xyz")
            )
            driver.write_input(
                os.path.join(workdir, f"{identifier}_smd.yaml"),
                optimized,
                args.basis,
                args.method,
                "sp",
                "water",
                "smd",
            )
            solvated = driver.run_qwb(workdir, f"{identifier}_smd", profile=PROFILE)
            ours = solvated["solvation"]["total_hartree"] * HARTREE_PER_KCAL
        except (SystemExit, ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            reason = " ".join(str(exc).split())[:220]
            excluded.append((identifier, type(exc).__name__, reason))
            print(f"{identifier:16s} {'':9s} {'':2s} {'EXCLUDED':>10s} {row['dg_hyd_kcal_per_mol']:>11s}   {reason}")
            continue
        experiment = float(row["dg_hyd_kcal_per_mol"])
        deviation = ours - experiment
        deviations.append(deviation)
        measured.append((identifier, row["formula"], int(row["atom_count"]), ours, experiment, deviation))
        print(
            f"{identifier:16s} {row['formula']:9s} {row['atom_count']:>2s} {ours:10.3f} "
            f"{experiment:11.2f} {deviation:+8.3f}"
        )

    if not deviations:
        print("no molecule produced a number; nothing to report")
        return 1

    count = len(deviations)
    mae = sum(abs(value) for value in deviations) / count
    rmse = math.sqrt(sum(value * value for value in deviations) / count)
    bias = sum(deviations) / count
    worst = max(deviations, key=abs)
    worst_row = measured[deviations.index(worst)]
    print(
        f"\nMAE = {mae:.3f} kcal/mol, RMSE = {rmse:.3f}, max |error| = {abs(worst):.3f} "
        f"({worst_row[0]} {worst_row[1]}: {worst:+.3f}), mean signed error (bias) = {bias:+.3f} "
        f"over {count} molecule(s), {len(excluded)} excluded"
    )
    positive = sum(1 for value in deviations if value > 0)
    negative = sum(1 for value in deviations if value < 0)
    print(
        f"sign: {positive} positive (our ΔG_hyd is more positive than experiment) / {negative} negative; "
        f"|error| <= 1.0 kcal/mol for {sum(1 for value in deviations if abs(value) <= 1.0)}/{count}"
    )
    if excluded:
        print("excluded explicitly:")
        for identifier, kind, reason in excluded:
            print(f"  {identifier}: {kind}: {reason}")
    sizes = [row[2] for row in measured]
    print(
        "NOTE: the sample is the smallest-atom segment "
        f"({min(sizes)}-{max(sizes)} atoms of the {len(rows)}-row family), not a random sample; "
        "it is biased towards small rigid molecules."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
