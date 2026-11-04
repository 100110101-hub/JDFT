#!/usr/bin/env python
"""RP-01：给「局部本征向量跟随」找初猜 —— 先沿一个几何参数扫能量，再从上坡点起搜鞍点。

`qwb ts` 只做**局部**搜索（`qwb-engine/src/ts.rs` 的头注释写清了这一点）。手搭的初猜对
两个反应落在了极小点盆地里（实测：H2CO 的初猜搜出 0 个虚频；CH3NC 的 40 步没收敛），
所以这里先用**同一套几何参数**扫一遍单点能量，把能量最高的那个参数值当鞍点初猜。

两个反应的参数化（都是「反应坐标」的直接写法，不是经验公式）：

* R3 `h2co`：H₂ 单元沿 −z 轴离开 C，参数 `d` = H₂ 中心到 C 的距离；H–H 与 C–O 固定。
* R2 `ch3nc`：甲基沿以 N–C 中点为心的圆弧从 N 侧绕到 C 侧，参数 `beta` = 圆弧角；
  甲基的 C₃ 轴始终背离 N–C 轴（伞形朝外）。

用法::

    python scripts/rp01_ts_guesses.py --reaction h2co --method hf --basis sto-3g --out .tmp-rp01-lvl/r3_guess
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWB = ROOT / "qwb-core" / "target" / "release" / "qwb.exe"
HARTREE_TO_KCAL = 627.5094740631


def rotor(beta_deg, radius, center_z, ch=1.09, umb=109.4712206):
    """甲基：C 在圆弧上，C₃ 轴背离 N–C 中点，三个 H 绕该轴按四面体角张开。"""
    beta = math.radians(beta_deg)
    # 圆弧：从 N 侧（beta=0 在 z 轴负侧）绕过 +x 到 C 侧。
    cx = radius * math.sin(beta)
    cz = center_z - radius * math.cos(beta)
    # C₃ 轴指向远离中点的方向：从 (0,0,center_z) 指向甲基碳的单位向量。
    ax = math.sin(beta)
    az = -math.cos(beta)
    # 与轴垂直的两支基：e1 = ẑ×u（在这里沿 y），e2 = u×e1。
    e1 = (0.0, 1.0, 0.0)
    e2 = (az, 0.0, -ax)  # u × e1 = (ax,0,az) × (0,1,0) = (-az, 0, ax) → 取反号等价
    cos_umb = math.cos(math.radians(umb))
    sin_umb = math.sin(math.radians(umb))
    atoms = [("C", cx, 0.0, cz)]
    for phi_deg in (0.0, 120.0, 240.0):
        phi = math.radians(phi_deg)
        dx = ch * (cos_umb * ax + sin_umb * (math.cos(phi) * e1[0] + math.sin(phi) * e2[0]))
        dy = ch * (cos_umb * 0.0 + sin_umb * (math.cos(phi) * e1[1] + math.sin(phi) * e2[1]))
        dz = ch * (cos_umb * az + sin_umb * (math.cos(phi) * e1[2] + math.sin(phi) * e2[2]))
        atoms.append(("H", cx + dx, dy, cz + dz))
    return atoms


def ch3nc_at(beta_deg, radius=1.86, nc=1.20):
    """CH₃NC→CH₃CN 的弧线几何：N 在原点、C 在 +z（N–C 固定在 `nc`），甲基在圆弧上。"""
    atoms = [("N", 0.0, 0.0, 0.0), ("C", 0.0, 0.0, nc)]
    atoms += rotor(beta_deg, radius, nc / 2.0)
    return atoms


def h2co_at(d, hh=1.05, co=1.15):
    """H₂CO→H₂+CO：C 在原点、O 在 +z（C–O 固定），H₂ 单元沿 −z 离开，参数 `d`。"""
    return [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, co),
        ("H", hh / 2.0, 0.0, -d),
        ("H", -hh / 2.0, 0.0, -d),
    ]


def write_yaml(path, atoms, method, basis, grid, jobtype, max_steps=40):
    lines = ["molecule:", "  atoms:"]
    for symbol, x, y, z in atoms:
        lines.append(f"    - symbol: {symbol}")
        lines.append(f"      position_angstrom: [{x!r}, {y!r}, {z!r}]")
    lines += [
        "  charge: 0",
        "  multiplicity: 1",
        f"basis_name: {basis}",
        f"method_name: {method}",
        f"grid_level: {grid}",
        "solvent: vacuum",
        f"jobtype: {jobtype}",
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
    if jobtype == "ts":
        lines += [
            "  ts:",
            f"    max_steps: {max_steps}",
            "    gradient_threshold_hartree_per_bohr: 4.5e-4",
        ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def single_point(atoms, method, basis, grid, out_dir, tag):
    inputs = out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    yaml_path = inputs / f"{tag}.yaml"
    write_yaml(yaml_path, atoms, method, basis, grid, "sp")
    completed = subprocess.run(
        [str(QWB), "run", str(yaml_path), "--out", str(out_dir / "sp" / tag)],
        capture_output=True,
        text=True,
        cwd=str(ROOT / "qwb-core"),
    )
    result_path = out_dir / "sp" / tag / tag / "result.json"
    energy = None
    if result_path.exists():
        energy = json.loads(result_path.read_text(encoding="utf-8"))["energy_hartree"]
    return energy, completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction", required=True, choices=["ch3nc", "h2co"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--grid", default="normal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reaction == "ch3nc":
        samples = [
            (f"beta{beta:03.0f}", ch3nc_at(beta), {"beta_deg": beta})
            for beta in (40.0, 55.0, 70.0, 85.0, 100.0, 115.0, 130.0, 145.0)
        ]
    else:
        # 二维：H₂ 单元离 C 的距离 `d` × H–H 距离。一维（固定 H–H = 1.05）的最高点
        # 落在离真正鞍点很远的地方，搜出来的"鞍点"虚频只有 −138 cm⁻¹（那是另一个模）。
        samples = [
            (f"d{d * 100:03.0f}h{hh * 100:03.0f}", h2co_at(d, hh=hh), {"d_angstrom": d, "hh_angstrom": hh})
            for d in (1.10, 1.30, 1.50, 1.70, 1.90)
            for hh in (0.85, 1.00)
        ]

    scanned = []
    for tag, atoms, parameter in samples:
        energy, code = single_point(atoms, args.method, args.basis, args.grid, out_dir, tag)
        print(f"{tag}: exit={code} E = {energy}", flush=True)
        scanned.append({"tag": tag, "parameter": parameter, "energy_hartree": energy, "atoms": atoms})

    usable = [point for point in scanned if point["energy_hartree"] is not None]
    best = max(usable, key=lambda point: point["energy_hartree"])
    lowest = min(usable, key=lambda point: point["energy_hartree"])
    print(
        f"highest: {best['tag']} E = {best['energy_hartree']} "
        f"(+{(best['energy_hartree'] - lowest['energy_hartree']) * HARTREE_TO_KCAL:.2f} kcal/mol over lowest)",
        flush=True,
    )

    ts_yaml = out_dir / "ts_guess.yaml"
    write_yaml(ts_yaml, best["atoms"], args.method, args.basis, args.grid, "ts", args.max_steps)
    completed = subprocess.run(
        [str(QWB), "ts", str(ts_yaml), "--out", str(out_dir / "ts")],
        capture_output=True,
        text=True,
        cwd=str(ROOT / "qwb-core"),
    )
    result_path = out_dir / "ts" / "ts_guess" / "result.json"
    block = None
    energy = None
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        block = result.get("ts_result")
        energy = result.get("energy_hartree")
    summary = {
        "reaction": args.reaction,
        "level": {"method": args.method, "basis": args.basis, "grid": args.grid},
        "scan": [{key: point[key] for key in ("tag", "parameter", "energy_hartree")} for point in scanned],
        "guess": {"tag": best["tag"], "parameter": best["parameter"]},
        "ts": {"exit_code": completed.returncode, "energy_hartree": energy, "block": block},
        "ts_stdout_tail": completed.stdout.strip().splitlines()[-6:],
        "ts_stderr_tail": completed.stderr.strip().splitlines()[-6:],
    }
    (out_dir / "guess.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
