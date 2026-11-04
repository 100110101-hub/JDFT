#!/usr/bin/env python
"""RP-01 势垒测量驱动（**不是**参考数据生成器）。

这个脚本把「三个反应的势垒」这件事变成可重跑的命令：它只负责

1. 按给定的层级（method/basis/grid）写 `input.yaml`；
2. 用 `qwb` 依次跑极小点优化（`jobtype: opt`）与鞍点搜索（`jobtype: ts`）；
3. 把每一步的**墙钟**、`result.json` 的 `energy_hartree`、`ts_result` 的原样字段
   （iterations / max|g| / n_imaginary / imaginary_wavenumber_cm）抄进一份 JSON。

**它不产出参考 CSV，也不写任何文献值** —— 文献值不在这个脚本里，因为仓库里
从来没有钉过它们（见报告）。脚本只报「本机实测」的那一栏，避免把自算值伪装成文献。

用法::

    python scripts/rp01_barriers.py --level hf/sto-3g --phase all --out results/rp01-hf-sto3g
    python scripts/rp01_barriers.py --level pbe/def2-svp --grid coarse --phase all ...

`--phase minima|ts|all` 允许只跑一半；`--seed-from` 允许把另一层级收敛出来的几何
（`artifacts/optimized.xyz`）当下一层级的初猜 —— 抬层级时这能省掉一半的步数。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 1 Hartree = 627.5094740631 kcal/mol（CODATA 2018 的换算，脚本里显式写死，
# 报告里同时给 Hartree 原值，读者可以用任何常数重算）。
HARTREE_TO_KCAL = 627.5094740631

ROOT = Path(__file__).resolve().parents[1]
QWB = ROOT / "qwb-core" / "target" / "release" / "qwb.exe"

# 几何（Å）。 reactant / product 是极小点的初猜；ts 是鞍点搜索的初猜。
# 键长取自常见值，只作为初猜 —— 收敛后的几何与能量才是数据。
GEOMETRIES = {
    # R1: HCN <-> HNC（1,2-H 迁移）
    "r1_reactant": [
        ("C", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 1.156),
        ("H", 0.0, 0.0, -1.066),
    ],
    "r1_product": [
        ("H", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 0.994),
        ("C", 0.0, 0.0, 2.163),
    ],
    # 1,2-H 迁移的鞍点量级几何：C–N 1.17、C–H 1.19、∠H–C–N 100°
    # （H 不再同时贴住 C 与 N —— 那样搜出来的驻点带 2 个虚频）
    "r1_ts": [
        ("C", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 1.170),
        ("H", 1.1719, 0.0, -0.2066),
    ],
    # R2: CH3NC <-> CH3CN（甲基从 N 迁到 C）
    "r2_reactant": [
        ("C", 0.0, 0.0, 0.0),
        ("H", 1.028, 0.0, -0.363),
        ("H", -0.514, 0.891, -0.363),
        ("H", -0.514, -0.891, -0.363),
        ("N", 0.0, 0.0, 1.450),
        ("C", 0.0, 0.0, 2.620),
    ],
    "r2_product": [
        ("C", 0.0, 0.0, 0.0),
        ("H", 1.028, 0.0, -0.363),
        ("H", -0.514, 0.891, -0.363),
        ("H", -0.514, -0.891, -0.363),
        ("C", 0.0, 0.0, 1.460),
        ("N", 0.0, 0.0, 2.620),
    ],
    "r2_ts": [
        ("N", 0.0, 0.0, 0.0),
        ("C", 0.0, 0.0, 1.200),
        ("C", 1.860, 0.0, 0.600),
        ("H", 1.4967, 1.0277, 0.600),
        ("H", 1.4967, -0.5138, -0.2900),
        ("H", 1.4967, -0.5138, 1.4900),
    ],
    # R3: H2CO -> H2 + CO（四中心协同）
    "r3_reactant": [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.210),
        ("H", 0.940, 0.0, -0.680),
        ("H", -0.940, 0.0, -0.680),
    ],
    "r3_ts": [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.150),
        ("H", 0.500, 0.0, -1.500),
        ("H", -0.500, 0.0, -1.500),
    ],
    # 产物侧的两块碎片分开算（H2 与 CO 单独一个 SCF 最干净）
    "r3_product_h2": [
        ("H", 0.0, 0.0, 0.0),
        ("H", 0.0, 0.0, 0.740),
    ],
    "r3_product_co": [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.130),
    ],
}

MINIMA = [
    "r1_reactant",
    "r1_product",
    "r2_reactant",
    "r2_product",
    "r3_reactant",
    "r3_product_h2",
    "r3_product_co",
]
SADDLES = ["r1_ts", "r2_ts", "r3_ts"]


def write_input(species, level, grid, out_dir, geometry=None):
    atoms = geometry or GEOMETRIES[species]
    lines = ["molecule:", "  atoms:"]
    for symbol, x, y, z in atoms:
        lines.append(f"    - symbol: {symbol}")
        lines.append(f"      position_angstrom: [{x!r}, {y!r}, {z!r}]")
    lines += [
        "  charge: 0",
        "  multiplicity: 1",
        f"basis_name: {level['basis']}",
        f"method_name: {level['method']}",
        f"grid_level: {grid}",
        "solvent: vacuum",
        f"jobtype: {'ts' if species in SADDLES else 'opt'}",
        "options:",
        "  scf_max_iter: 200",
        "  scf_energy_threshold_hartree: 1.0e-9",
        "  scf_density_threshold: 1.0e-7",
        "  diis_history: 8",
        "  threads: 0",
        "  max_opt_steps: 200",
        "  opt_gradient_threshold_hartree_per_bohr: 4.5e-4",
        "  freq_step_angstrom: 0.01",
        "  ecp_name: null",
    ]
    if species in SADDLES:
        lines += ["  ts:", "    max_steps: 40", "    gradient_threshold_hartree_per_bohr: 4.5e-4"]
    path = out_dir / f"{species}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_xyz(path):
    """把 `artifacts/optimized.xyz` 读回 [(symbol, x, y, z)]（Å）。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def run_one(species, level, grid, work_dir, seed_dir, record):
    input_dir = work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    geometry = None
    if seed_dir is not None:
        # 结果目录是 `--out DIR` + 一层输入文件名，所以两种布局都试。
        for seed in (
            seed_dir / species / species / "artifacts" / "optimized.xyz",
            seed_dir / species / "artifacts" / "optimized.xyz",
        ):
            if seed.exists():
                geometry = read_xyz(seed)
                record["seeded_from"] = str(seed)
                break
    input_path = write_input(species, level, grid, input_dir, geometry)
    # `--out DIR` 之下还会套一层「输入文件名」的目录（K3 的结果目录约定），
    # 所以真正的 result.json 在 DIR/<stem>/result.json。
    out_dir = work_dir / species / species
    command = [str(QWB), ("ts" if species in SADDLES else "run"), str(input_path), "--out", str(work_dir / species)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT / "qwb-core"))
    elapsed = time.perf_counter() - started
    record.update(
        {
            "species": species,
            "command": " ".join(command),
            "wall_clock_s": round(elapsed, 3),
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout.strip().splitlines()[-6:],
            "stderr_tail": completed.stderr.strip().splitlines()[-6:],
        }
    )
    result_path = out_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record["energy_hartree"] = result.get("energy_hartree")
        record["scf_converged"] = result.get("scf_converged")
        block = result.get("ts_result")
        if block is not None:
            record["ts_result"] = {
                key: block[key]
                for key in (
                    "iterations",
                    "energy_hartree",
                    "max_gradient_hartree_per_bohr",
                    "n_imaginary",
                    "imaginary_wavenumber_cm",
                )
            }
        # 优化得到的步数与收敛标记（`optimization.json`）
        opt_path = out_dir / "artifacts" / "optimization.json"
        if opt_path.exists():
            opt = json.loads(opt_path.read_text(encoding="utf-8"))
            record["opt"] = {
                key: opt.get(key)
                for key in ("steps", "converged", "max_gradient_hartree_per_bohr")
                if key in opt
            }
    log_path = out_dir / "log.txt"
    if log_path.exists():
        record["log"] = log_path.read_text(encoding="utf-8").strip().splitlines()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--grid", default="normal")
    parser.add_argument("--phase", choices=["minima", "ts", "all"], default="all")
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-from", default=None)
    args = parser.parse_args()

    level = {"method": args.method, "basis": args.basis}
    work_dir = Path(args.out)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = None
    if args.seed_from:
        seed_dir = Path(args.seed_from)
        if not seed_dir.is_absolute():
            seed_dir = ROOT / seed_dir

    wanted = list(SADDLES) if args.phase == "ts" else list(MINIMA) if args.phase == "minima" else MINIMA + SADDLES
    if args.species:
        wanted = [name for name in wanted if name in args.species]

    records = []
    for species in wanted:
        print(f"[{species}] {args.method}/{args.basis}/{args.grid} ...", flush=True)
        record = run_one(species, level, args.grid, work_dir, seed_dir, {"level": {}, "grid": args.grid})
        record["level"] = level
        print(
            f"[{species}] exit={record['exit_code']} {record['wall_clock_s']}s "
            f"E={record.get('energy_hartree')}",
            flush=True,
        )
        records.append(record)
        (work_dir / "measurements.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )

    summary = build_summary(records)
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_summary(records):
    by_species = {record["species"]: record for record in records}

    def energy(name):
        record = by_species.get(name)
        if record is None or record.get("energy_hartree") is None:
            return None
        return record["energy_hartree"]

    summary = {"level": records[0]["level"], "grid": records[0]["grid"], "reactions": {}}
    for reaction, reactant, product, saddle in (
        ("r1_hcn_hcnc", "r1_reactant", "r1_product", "r1_ts"),
        ("r2_ch3nc_ch3cn", "r2_reactant", "r2_product", "r2_ts"),
        ("r3_h2co_h2_co", "r3_reactant", None, "r3_ts"),
    ):
        e_reactant = energy(reactant)
        if product is None:
            parts = [energy("r3_product_h2"), energy("r3_product_co")]
            e_product = None if any(part is None for part in parts) else sum(parts)
        else:
            e_product = energy(product)
        e_saddle = energy(saddle)
        block = by_species.get(saddle, {}).get("ts_result")
        entry = {
            "reactant": {"species": reactant, "energy_hartree": e_reactant},
            "product": {"species": product or "r3_product_h2 + r3_product_co", "energy_hartree": e_product},
            "saddle": {"species": saddle, "energy_hartree": e_saddle},
            "imaginary_wavenumber_cm": None if block is None else block["imaginary_wavenumber_cm"],
            "n_imaginary": None if block is None else block["n_imaginary"],
            "ts_iterations": None if block is None else block["iterations"],
            "max_gradient_hartree_per_bohr": None if block is None else block["max_gradient_hartree_per_bohr"],
            "wall_clock_s": {
                "reactant": by_species.get(reactant, {}).get("wall_clock_s"),
                "product": by_species.get(product, {}).get("wall_clock_s")
                if product
                else {
                    "r3_product_h2": by_species.get("r3_product_h2", {}).get("wall_clock_s"),
                    "r3_product_co": by_species.get("r3_product_co", {}).get("wall_clock_s"),
                },
                "saddle": by_species.get(saddle, {}).get("wall_clock_s"),
            },
        }
        if e_reactant is not None and e_saddle is not None:
            entry["barrier_hartree"] = e_saddle - e_reactant
            entry["barrier_kcal"] = (e_saddle - e_reactant) * HARTREE_TO_KCAL
        if e_reactant is not None and e_product is not None:
            entry["reaction_energy_kcal"] = (e_product - e_reactant) * HARTREE_TO_KCAL
        summary["reactions"][reaction] = entry
    return summary


if __name__ == "__main__":
    sys.exit(main())
