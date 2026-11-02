"""把「我们的 SMD ΔG_solv」与 `tests/reference/solvation_free_energy.csv` 的实验值对账。

流程（SMD 协议的最小版本）：气相几何优化（`jobtype: opt`，本项目当前层级）→ 在该几何上做
一次 SMD 单点（`solvent_model: smd`）→ `ΔG_solv = solvation.total_hartree × 627.5095`。
`total_hartree` 是 **1 M ↔ 1 M** 的转移能，与 FreeSolv 的 `expt` 同一个约定（CSV 的
`standard_state` 列就是这么写的），所以这里**不加** `RT ln(RT/p°)`。

这不是闸门的一部分：一次要跑 2 次几何优化 + 2 次 SMD 单点，几十秒到一分钟量级。它的输出
（每个体系的实测偏差与 MAE）写进 `docs/status-v2.yaml` 的 VF-02 与 `docs/error-budget.md`。

用法：

    python scripts/measure_smd_dgsolv.py            # 跑一遍，打印对账表
    python scripts/measure_smd_dgsolv.py --basis 6-311g(d,p) --method pbe
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "solvation_free_energy.csv"
)
HARTREE_PER_KCAL = 627.509_474_063_6

#: 起始几何（Å）；优化会把它们带到当前层级的极小点，起始值只要合理即可。
GEOMETRIES = {
    "methanol": [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.43)),
        ("H", (0.94, 0.0, 1.85)),
        ("H", (-0.36, 0.89, -0.36)),
        ("H", (-0.36, -0.89, -0.36)),
        ("H", (0.0, 0.0, -1.09)),
    ],
    "acetonitrile": [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (0.0, 0.0, 1.46)),
        ("N", (0.0, 0.0, 2.62)),
        ("H", (-0.51, 0.89, 0.0)),
        ("H", (-0.51, -0.89, 0.0)),
        ("H", (1.02, 0.0, 0.0)),
    ],
}

OPTIONS_TEMPLATE = """  scf_max_iter: 300
  scf_energy_threshold_hartree: 1.0e-9
  scf_density_threshold: 1.0e-7
  diis_history: 8
  threads: 0
  max_opt_steps: 200
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
"""


def write_input(path, atoms, basis, method, jobtype, solvent, solvent_model):
    lines = ["molecule:", "  atoms:"]
    for symbol, (x, y, z) in atoms:
        lines.append(f"    - symbol: {symbol}")
        lines.append(f"      position_angstrom: [{x}, {y}, {z}]")
    lines.append("  charge: 0")
    lines.append("  multiplicity: 1")
    lines.append(f"basis_name: {basis}")
    lines.append(f"method_name: {method}")
    lines.append("grid_level: normal")
    lines.append(f"solvent: {solvent}")
    lines.append(f"jobtype: {jobtype}")
    lines.append("options:")
    body = OPTIONS_TEMPLATE
    if solvent_model:
        body += f"  solvent_model: {solvent_model}\n"
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n" + body)


def read_xyz(path):
    with io.open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    count = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        atoms.append((fields[0], tuple(float(value) for value in fields[1:4])))
    return atoms


def run_qwb(workdir, name):
    # 工作目录是临时目录，`qwb` 按相对路径找不到 qwb-data —— 用 QWB_DATA_ROOT 指回去。
    environment = dict(os.environ)
    environment["QWB_DATA_ROOT"] = os.path.join(ROOT, "qwb-data")
    environment["QWB_AUTO_FETCH"] = "0"
    binary = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", os.path.join(ROOT, "qwb-core", "Cargo.toml"),
         "-p", "qwb-cli", "--", "run", f"{name}.yaml", "--out", "out"],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
    )
    if binary.returncode != 0:
        raise SystemExit(f"qwb run {name} failed:\n{binary.stdout}\n{binary.stderr}")
    result_path = os.path.join(workdir, "out", name, "result.json")
    with io.open(result_path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--method", default="hf")
    args = parser.parse_args()

    with io.open(REFERENCE, encoding="utf-8") as handle:
        reference = [row for row in csv.DictReader(handle) if row["quantity"] == "dg_hydration_experimental"]

    workdir = tempfile.mkdtemp(prefix="qwb-smd-dgsolv-")
    deviations = []
    print(f"{'solute':14s} {'solvent':8s} {'this work':>10s} {'experiment':>11s} {'deviation':>10s}")
    for row in reference:
        solute = row["solute"]
        if solute not in GEOMETRIES:
            raise SystemExit(f"no starting geometry for `{solute}`")
        write_input(os.path.join(workdir, f"{solute}_opt.yaml"), GEOMETRIES[solute],
                    args.basis, args.method, "opt", "vacuum", None)
        optimized = run_qwb(workdir, f"{solute}_opt")
        atoms = read_xyz(os.path.join(workdir, "out", f"{solute}_opt", "artifacts", "optimized.xyz"))
        write_input(os.path.join(workdir, f"{solute}_smd.yaml"), atoms,
                    args.basis, args.method, "sp", row["solvent"], "smd")
        solvated = run_qwb(workdir, f"{solute}_smd")
        block = solvated["solvation"]
        ours = block["total_hartree"] * HARTREE_PER_KCAL
        experiment = float(row["value_kcal_per_mol"])
        deviation = ours - experiment
        deviations.append(deviation)
        print(f"{solute:14s} {row['solvent']:8s} {ours:10.3f} {experiment:11.2f} {deviation:+10.3f}")

    mae = sum(abs(value) for value in deviations) / len(deviations)
    print(f"\nMAE = {mae:.3f} kcal/mol over {len(deviations)} system(s); "
          f"geometry + single point at {args.method.upper()}/{args.basis}, "
          f"standard state 1 M (FreeSolv convention; no RT ln(RT/p°) added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
