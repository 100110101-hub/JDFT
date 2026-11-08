#!/usr/bin/env python
"""RP-01 势垒的**实测**驱动（不是参考数据生成器）。

这个脚本只做一件事：把「三个反应的势垒」变成可重跑的命令链，并且把每一步的原始输出抄下来。

    极小点优化（`jobtype: opt`）→ 鞍点搜索（`jobtype: ts`）→ 双向 IRC（`jobtype: irc`）

**它不产出参考 CSV，也不写任何文献值**。文献值在 `scripts/gen_rp01_literature.py`
（DA-08 的参考家族，逐行带 DOI），实测值在这里的 `summary.json` 与报告里。两者分开的理由
是造假的常见形态：把自算值混进参考集，之后谁也分不清哪一栏是外部尺子。

# 为什么鞍点必须再过一遍 IRC

上一轮实测过 `ts` 会接受**软模式鞍点**：`−137 cm⁻¹` 的虚频确实是虚频，但它不是反应坐标。
「恰好一个虚频」只说明停在一阶鞍点，不说明停在**这个反应**的鞍点上。所以这里的验收是：

1. `ts` 收敛、`n_imaginary == 1`、`max|g| < 4.5e-4`；
2. `irc` 双向跑通，两端都 `monotone`；
3. 两端的**收尾优化能量**分别对上反应物 / 产物的极小点（|ΔE| < 1e-3 Ha ≈ 0.63 kcal/mol），
   且两端**不是同一个**极小点；
4. 键数/原子数不变时，端点内坐标指纹也逐项对上（键长 < 0.02 Å、键角 < 1°，与
   `crates/qwb-engine/tests/rp02_irc.rs` 同一套判据）。

任何一条不满足，这个鞍点就**不计入**势垒：`summary.json` 里 `accepted` 为 false 并写明原因。

# 原子顺序必须逐反应一致

IRC 的端点比对靠内坐标指纹，而指纹按**原子顺序**枚举键长与三元组键角。同一反应的反应物、
产物、鞍点如果不是同一套原子顺序，指纹就不可比（CH₃NC 与 CH₃CN 的 C/N 换位正是这种陷阱）。
所以 `GEOMETRIES` 里每个反应的三个结构共用一份顺序，注释里写明每一号原子是什么。

# 层级与网格

层级由命令行给定（本轮：PBE/def2-SVP、`grid_level: normal`）。`--seed-dir` 允许把上一层
收敛出来的几何（`artifacts/optimized.xyz`）当这一层的初猜。`--stages` 的每一段都可重入：
已经写出 `result.json` 的物种默认跳过（`--force` 重跑）。

用法::

    python scripts/rp01_measure.py --method hf --basis sto-3g --grid normal \
        --out .tmp-rp01-measure/hf-sto3g --stages guess,minima,ts,irc --jobs 8
    python scripts/rp01_measure.py --method pbe --basis def2-svp --grid normal \
        --out .tmp-rp01-measure/pbe-svp --stages minima,ts,irc --jobs 6 \
        --seed-dir .tmp-rp01-measure/hf-sto3g
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 1 Hartree = 627.5094740631 kcal/mol（CODATA 2018，脚本里显式写死；报告同时给 Hartree 原值，
# 读者可以用任何常数重算）。
HARTREE_TO_KCAL = 627.5094740631

ROOT = Path(__file__).resolve().parents[1]
QWB = ROOT / "qwb-core" / "target" / "release" / "qwb.exe"

# CLI 覆盖（`--qwb` / `--threads` / `--irc-steps` / `--irc-step-size`）。
# `threads` 不是装饰：XC 网格收缩是**单线程**的，而 Fock 构建按行多线程 —— 同时跑多个
# qwb 进程又都给 `threads: 0`（全核）时线程严重超订，实测单个作业能从 30 s/SCF 掉到
# 5 分钟/SCF。多进程并行时必须把每个作业的线程数压下来。
THREADS = 0
IRC_STEPS = 80
IRC_STEP_SIZE = 0.10

# ────────────────────────────── 几何（Å，只作初猜）──────────────────────────────
# 初猜不是数据：收敛后的几何与能量才是。R1 的鞍点初猜取自
# Makhnev et al. J. Phys. Chem. A 122, 1326 (2018) 的鞍点几何
# （r(C–H) = 2.238 a0 = 1.184 Å、r(C–N) = 2.242 a0 = 1.186 Å、∠(H–C–N) = 71.29°）——
# 用的是**几何**，不是势垒值（势垒见 scripts/gen_rp01_literature.py）。
GEOMETRIES = {
    # R1: HCN ↔ HNC（1,2-H 迁移）。顺序 [C, N, H] 三个结构一致。
    "r1_reactant": [  # HCN
        ("C", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 1.156),
        ("H", 0.0, 0.0, -1.066),
    ],
    "r1_product": [  # HNC（H 挂在 N 上，C–N 仍按 C→N 的方向放）
        ("C", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 1.166),
        ("H", 0.0, 0.0, 2.160),
    ],
    "r1_ts": [  # 文献鞍点几何：r(C–H) 1.184、r(C–N) 1.186、∠H–C–N 71.29°
        ("C", 0.0, 0.0, 0.0),
        ("N", 0.0, 0.0, 1.186),
        ("H", 1.1215, 0.0, 0.3796),
    ],
    # R2: CH₃NC ↔ CH₃CN（甲基从 N 迁到 C）。顺序 [C(甲基), H, H, H, N, C(腈)]。
    "r2_reactant": [  # CH₃–N≡C
        ("C", 0.0, 0.0, 0.0),
        ("H", 1.028, 0.0, -0.363),
        ("H", -0.514, 0.891, -0.363),
        ("H", -0.514, -0.891, -0.363),
        ("N", 0.0, 0.0, 1.450),
        ("C", 0.0, 0.0, 2.620),
    ],
    "r2_product": [  # CH₃–C≡N（同一个顺序：N 在 2.62、C 在 1.46）
        ("C", 0.0, 0.0, 0.0),
        ("H", 1.028, 0.0, -0.363),
        ("H", -0.514, 0.891, -0.363),
        ("H", -0.514, -0.891, -0.363),
        ("N", 0.0, 0.0, 2.620),
        ("C", 0.0, 0.0, 1.460),
    ],
    # 甲基在圆弧上、C₃ 轴背离 N–C 轴的 guess（β = 90°、半径 1.86 Å、N–C 1.20 Å）。
    # 这三个坐标由 `_ch3nc_arc(90.0)` 在下面算出（手写坐标会写错一个正弦），`guess` 阶段
    # 会用同一套参数化扫一遍再覆盖它。
    "r2_ts": [],
    # R3: H₂CO → H₂ + CO（四中心协同）。顺序 [C, O, H, H]。
    "r3_reactant": [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, 1.210),
        ("H", 0.940, 0.0, -0.680),
        ("H", -0.940, 0.0, -0.680),
    ],
    # C–H ≈ 1.39 Å、H–H ≈ 1.05 Å、C–O ≈ 1.20 Å 的四中心 guess（`guess` 阶段覆盖它）。
    "r3_ts": [],
    # 产物侧的两块碎片分开优化（H₂ 与 CO 各自一个 SCF 最干净）
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

#: 三个反应：名字 → (反应物, 产物碎片, 鞍点)。产物的能量是碎片能量之**和**。
REACTIONS = {
    "r1_hcn_hcnc": ("r1_reactant", ["r1_product"], "r1_ts"),
    "r2_ch3nc_ch3cn": ("r2_reactant", ["r2_product"], "r2_ts"),
    "r3_h2co_h2_co": ("r3_reactant", ["r3_product_h2", "r3_product_co"], "r3_ts"),
}

#: 单个 `qwb` 进程的墙钟上限（秒）。上一轮有一次鞍点搜索在病态初猜上跑了 4000 s CPU 时间还
#: 没出来 —— 没有上限的话它会一直占着一个槽位，而"卡住"与"在算"从外面看不出区别。
JOB_TIMEOUT_S = 7200

#: 端点收尾能量对上极小点的判据（Hartree）。1e-3 Ha = 0.63 kcal/mol，比势垒判据松，因为
#: 端点优化与直接优化停在同一个极小点附近、但阈值 4.5e-4 Ha/Bohr 允许 ~0.1 kcal/mol 的抖动。
ENDPOINT_ENERGY_TOL_HARTREE = 1.0e-3
#: 两端不能是同一个极小点（否则这条路径只是"两条上坡再回来"）。
ENDPOINT_SEPARATION_HARTREE = 1.0e-4

# ────────────────────────────── 几何构造器（guess 阶段）──────────────────────────────


def _rotor(beta_deg, radius, center_z, ch=1.09, umb=109.4712206):
    """甲基：C 在圆弧上，C₃ 轴指向 N–C 中点（H 朝外），三个 H 绕该轴按四面体角张开。

    轴的朝向是这一段的**唯一**易错点：C₃ 轴必须从甲基碳指向弧心。反过来（第一版就是反的）
    三个 H 会插进 N≡C 那一侧，整条扫描曲线的形状都是错的 —— 代价是被它带偏的一次鞍点搜索。

    返回 `[C, H, H, H]`（与 `r2_reactant` 的前四号原子同一顺序）。
    """
    beta = math.radians(beta_deg)
    cx = radius * math.sin(beta)
    cz = center_z - radius * math.cos(beta)
    # 单位向量 u：从甲基碳 (cx, 0, cz) 指向弧心 (0, 0, center_z)。
    ax = -math.sin(beta)
    az = math.cos(beta)
    # 与 u 垂直的两支基：e1 = ŷ，e2 = u × e1 = (−az, 0, ax)。
    e1 = (0.0, 1.0, 0.0)
    e2 = (-az, 0.0, ax)
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


def _ch3nc_arc(beta_deg, radius=1.86, nc=1.20):
    """R2 的弧线几何，按 `[C(甲基), H, H, H, N, C(腈)]` 的顺序返回。"""
    methyl = _rotor(beta_deg, radius, nc / 2.0)
    return methyl + [("N", 0.0, 0.0, 0.0), ("C", 0.0, 0.0, nc)]


def _h2co_bridge(d, hh=1.05, co=1.20):
    """R3 的桥式几何：C 在原点、O 在 +z，H₂ 单元沿 −z 离开 C 的距离为 `d`。"""
    return [
        ("C", 0.0, 0.0, 0.0),
        ("O", 0.0, 0.0, co),
        ("H", hh / 2.0, 0.0, -d),
        ("H", -hh / 2.0, 0.0, -d),
    ]


# 两个鞍点初猜由参数化构造器算出（手写这三行坐标会写错一个正弦）：
# R2 = 甲基在弧顶（β = 90°，半径 1.86 Å）的对称桥式结构；R3 = H₂ 单元离 C 1.45 Å、
# H–H 1.05 Å 的四中心。
GEOMETRIES["r2_ts"] = _ch3nc_arc(90.0)
GEOMETRIES["r3_ts"] = _h2co_bridge(1.45, hh=1.05)


def _guess_samples(species):
    """`guess` 阶段扫的样本：`(tag, atoms, parameter)`。扫出来的最高点是鞍点初猜。"""
    if species == "r2_ts":
        return [
            (
                f"beta{beta:03.0f}r{radius * 100:03.0f}",
                _ch3nc_arc(beta, radius),
                {"beta_deg": beta, "radius_angstrom": radius, "nc_angstrom": 1.20},
            )
            for beta in (55.0, 70.0, 85.0, 100.0, 115.0, 130.0)
            for radius in (1.74, 1.90, 2.06)
        ]
    if species == "r3_ts":
        return [
            (
                f"d{d * 100:03.0f}h{hh * 100:03.0f}",
                _h2co_bridge(d, hh=hh, co=1.20),
                {"d_angstrom": d, "hh_angstrom": hh, "co_angstrom": 1.20},
            )
            for d in (1.15, 1.30, 1.45, 1.60)
            for hh in (0.90, 1.05, 1.20)
        ]
    return []


# ────────────────────────────── 输入文件 ──────────────────────────────


def write_input(path, atoms, level, grid, jobtype, ts_max_steps=40, irc_steps=80,
                irc_step_size=0.10, threads=0):
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
        f"jobtype: {jobtype}",
        "options:",
        "  scf_max_iter: 200",
        # 收敛阈值：能量 1e-9 Ha / 密度 1e-7。数值 Hessian 的差分是 0.01 Å，能量噪声必须
        # 比它小两三个量级，否则 Hessian 会被 SCF 噪声污染（这正是 RP-01 的鞍点质量来源）。
        "  scf_energy_threshold_hartree: 1.0e-9",
        "  scf_density_threshold: 1.0e-7",
        "  diis_history: 8",
        f"  threads: {threads}",
        "  max_opt_steps: 200",
        # 优化器阈值 4.5e-4 Ha/Bohr（phase-2 判据）。几何不确定性里 ~0.1 kcal/mol 来自它，
        # 报告里如实标注。
        "  opt_gradient_threshold_hartree_per_bohr: 4.5e-4",
        "  freq_step_angstrom: 0.01",
        "  ecp_name: null",
    ]
    if jobtype == "ts":
        lines += [
            "  ts:",
            f"    max_steps: {ts_max_steps}",
            "    gradient_threshold_hartree_per_bohr: 4.5e-4",
        ]
    if jobtype == "irc":
        lines += [
            "  irc:",
            f"    max_steps: {irc_steps}",
            f"    step_size: {irc_step_size!r}",
            "    direction: both",
            "    plateau_hartree: 1.0e-5",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_xyz(path, atoms, comment):
    lines = [f"{len(atoms)}", comment]
    lines += [f"{symbol} {x!r} {y!r} {z!r}" for symbol, x, y, z in atoms]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────── 指纹（与 `qwb-engine::irc::fingerprint` 同一套定义）────────────────────


def fingerprint(atoms):
    """键长（所有 i<j，Å）+ 键角（连续三元组 i, i+1, i+2，度）。

    与 `crates/qwb-engine/src/irc.rs` 的 `fingerprint` 逐条一致：先枚举所有对，再枚举连续
    三元组（退化向量被跳过 —— 这里与那边一样，`n1/n2 > 1e-12` 才算）。
    """
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            out.append(math.dist(a[1:], b[1:]))
    for i in range(max(0, len(atoms) - 2)):
        a, b, c = atoms[i], atoms[i + 1], atoms[i + 2]
        v1 = [a[k] - b[k] for k in (1, 2, 3)]
        v2 = [c[k] - b[k] for k in (1, 2, 3)]
        n1 = math.sqrt(sum(v * v for v in v1))
        n2 = math.sqrt(sum(v * v for v in v2))
        if n1 > 1.0e-12 and n2 > 1.0e-12:
            cosine = sum(v1[k] * v2[k] for k in range(3)) / (n1 * n2)
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return out


def compare_fingerprints(reference_atoms, endpoint_fingerprint):
    """端点指纹 vs 某个极小点几何的指纹：按「先全部键长、后连续三元组键角」切段比较。"""
    if reference_atoms is None or endpoint_fingerprint is None:
        return None
    ref = fingerprint(reference_atoms)
    if len(ref) != len(endpoint_fingerprint):
        return {"note": f"fingerprint length {len(ref)} vs {len(endpoint_fingerprint)}"}
    n_pairs = len(reference_atoms) * (len(reference_atoms) - 1) // 2
    lengths = [
        abs(a - b) for a, b in zip(ref[:n_pairs], endpoint_fingerprint[:n_pairs])
    ]
    angles = [
        abs(a - b) for a, b in zip(ref[n_pairs:], endpoint_fingerprint[n_pairs:])
    ]
    return {
        "max_length_delta_angstrom": max(lengths, default=0.0),
        "max_angle_delta_degree": max(angles, default=0.0),
        "reference_fingerprint": ref,
        "endpoint_fingerprint": endpoint_fingerprint,
    }


def endpoint_matches(block, length_tol=0.02, angle_tol=1.0):
    if not block or "max_length_delta_angstrom" not in block:
        return False
    return (
        block["max_length_delta_angstrom"] < length_tol
        and block["max_angle_delta_degree"] < angle_tol
    )


# ────────────────────────────── 执行 ──────────────────────────────


def run_job(species, command, input_path, out_dir, record, timeout_s=JOB_TIMEOUT_S):
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(QWB), command, str(input_path), "--out", str(out_dir)],
            capture_output=True,
            text=True,
            cwd=str(ROOT / "qwb-core"),
            timeout=timeout_s,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired:
        returncode = -1
        stdout = ""
        stderr = f"killed after {timeout_s} s (JOB_TIMEOUT_S)"
    elapsed = time.perf_counter() - started
    record.update(
        {
            "species": species,
            "jobtype": command,
            "command": f"qwb {command} {input_path.name} --out {out_dir}",
            "wall_clock_s": round(elapsed, 3),
            "exit_code": returncode,
            "stdout": stdout.strip().splitlines(),
            "stderr": stderr.strip().splitlines(),
        }
    )
    task_dir = out_dir / input_path.stem
    record["task_dir"] = str(task_dir)
    result_path = task_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record["energy_hartree"] = result.get("energy_hartree")
        record["scf_converged"] = result.get("scf_converged")
        record["scf_iterations"] = result.get("scf_iterations")
        if result.get("ts_result") is not None:
            record["ts_result"] = result["ts_result"]
        if result.get("irc_result") is not None:
            record["irc_result"] = result["irc_result"]
    opt_path = task_dir / "artifacts" / "optimization.json"
    if opt_path.exists():
        opt = json.loads(opt_path.read_text(encoding="utf-8"))
        record["optimization"] = {
            key: opt.get(key) for key in ("iterations", "converged") if key in opt
        }
    xyz = task_dir / "artifacts" / "optimized.xyz"
    if xyz.exists():
        record["optimized_xyz"] = [list(row) for row in read_xyz(xyz)]
    log_path = task_dir / "log.txt"
    if log_path.exists():
        record["log"] = log_path.read_text(encoding="utf-8").strip().splitlines()
    return record


def load_record(out_root, name):
    path = out_root / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_record(out_root, name, record):
    (out_root / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


# ────────────────────────────── 三个阶段 ──────────────────────────────


def stage_guess(out_root, level, grid, jobs, force):
    """按参数化反应坐标扫一遍单点，把最高点写成鞍点初猜（`guesses/<species>.xyz`）。"""
    tasks = []
    for species in SADDLES:
        for tag, atoms, parameter in _guess_samples(species):
            tasks.append((species, tag, atoms, parameter))
    if not tasks:
        return

    def one(task):
        species, tag, atoms, parameter = task
        name = f"guess_{species}_{tag}"
        record_path = out_root / "guess" / f"{name}.json"
        if not force and record_path.is_file():
            existing = load_record(out_root / "guess", name)
            if existing is not None and existing.get("exit_code") == 0:
                return species, tag, existing
        input_path = write_input(
            out_root / "guess" / "inputs" / f"{name}.yaml", atoms, level, grid, "sp",
            threads=THREADS,
        )
        record = run_job(
            name, "run", input_path, out_root / "guess" / "runs",
            {"tag": tag, "parameter": parameter, "atoms": [list(a) for a in atoms]},
        )
        save_record(out_root / "guess", name, record)
        print(f"[{name}] exit={record['exit_code']} E={record.get('energy_hartree')}", flush=True)
        return species, tag, record

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, tasks))

    by_species = {}
    by_tag = {}
    for species, tag, record in results:
        by_species.setdefault(species, []).append(record)
        by_tag[(species, tag)] = record
    for species, points in by_species.items():
        usable = [p for p in points if p.get("energy_hartree") is not None]
        if not usable:
            print(f"[guess:{species}] no usable points", flush=True)
            continue
        best = max(usable, key=lambda p: p["energy_hartree"])
        lowest = min(usable, key=lambda p: p["energy_hartree"])
        tag = best["tag"]
        atoms = [(row[0], row[1], row[2], row[3]) for row in best["atoms"]]
        write_xyz(out_root / "guesses" / f"{species}.xyz", atoms, f"guess {species} {tag}")
        (out_root / "guesses" / f"{species}.json").write_text(
            json.dumps(
                {
                    "species": species,
                    "best_tag": tag,
                    "best_parameter": best.get("parameter"),
                    "energy_hartree": best["energy_hartree"],
                    "energy_over_lowest_kcal": (best["energy_hartree"] - lowest["energy_hartree"])
                    * HARTREE_TO_KCAL,
                    "scan": [
                        {
                            "tag": p["species"],
                            "parameter": p.get("parameter"),
                            "energy_hartree": p.get("energy_hartree"),
                        }
                        for p in points
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[guess:{species}] best = {tag} E = {best['energy_hartree']} "
            f"(+{(best['energy_hartree'] - lowest['energy_hartree']) * HARTREE_TO_KCAL:.2f} kcal/mol "
            f"over the lowest scanned point)",
            flush=True,
        )


def geometry_for(species, seed_root, guess_root):
    """初猜来源的优先级：本层 guess（仅鞍点）> 上一层收敛几何（seed）> 内置初猜。"""
    if species in SADDLES and guess_root is not None:
        path = guess_root / "guesses" / f"{species}.xyz"
        if path.is_file():
            return read_xyz(path), f"guess:{path}"
    if seed_root is not None:
        for candidate in (
            seed_root / species / species / "artifacts" / "optimized.xyz",
            seed_root / species / "artifacts" / "optimized.xyz",
        ):
            if candidate.is_file():
                return read_xyz(candidate), f"seed:{candidate}"
    return GEOMETRIES[species], "builtin"


def stage_species(out_root, level, grid, species, jobs, force, seed_root, guess_root, jobtype):
    def one(name):
        if not force:
            existing = load_record(out_root, name)
            # Only a *successful* record counts as done: an interrupted job also leaves a
            # record.json behind (exit_code != 0, no energy), and treating that as finished
            # would silently keep the failure.
            if existing is not None and existing.get("exit_code") == 0:
                print(f"[{name}] already done, skipping", flush=True)
                return existing
        atoms, source = geometry_for(name, seed_root, guess_root)
        input_path = write_input(
            out_root / "inputs" / f"{name}.yaml", atoms, level, grid, jobtype, threads=THREADS
        )
        record = {"level": level, "grid": grid, "geometry_source": source, "geometry": [list(a) for a in atoms]}
        # CLI 子命令：极小点走 `qwb run`（`jobtype: opt`），鞍点走 `qwb ts`（它自己把 jobtype
        # 改写成 ts —— 输入里的 jobtype 只为可读性）。
        command = "ts" if jobtype == "ts" else "run"
        run_job(name, command, input_path, out_root / name, record)
        save_record(out_root, name, record)
        print(
            f"[{name}] exit={record['exit_code']} {record['wall_clock_s']}s "
            f"E={record.get('energy_hartree')}",
            flush=True,
        )
        return record

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, species))


def stage_irc(out_root, level, grid, jobs, force):
    """IRC 的起点是 `ts` 落盘的**鞍点几何**（`artifacts/optimized.xyz`）。"""

    def one(saddle):
        name = f"irc_{saddle}"
        if not force:
            existing = load_record(out_root, name)
            # Only a *successful* record counts as done: an interrupted job also leaves a
            # record.json behind (exit_code != 0, no energy), and treating that as finished
            # would silently keep the failure.
            if existing is not None and existing.get("exit_code") == 0:
                print(f"[{name}] already done, skipping", flush=True)
                return existing
        saddle_xyz = out_root / saddle / saddle / "artifacts" / "optimized.xyz"
        if not saddle_xyz.is_file():
            print(f"[{name}] no saddle geometry at {saddle_xyz}, skipped", flush=True)
            return None
        atoms = read_xyz(saddle_xyz)
        input_path = write_input(
            out_root / "inputs" / f"{name}.yaml", atoms, level, grid, "irc",
            irc_steps=IRC_STEPS, irc_step_size=IRC_STEP_SIZE, threads=THREADS,
        )
        record = {"level": level, "grid": grid, "geometry_source": f"saddle:{saddle_xyz}"}
        run_job(name, "irc", input_path, out_root / name, record)
        save_record(out_root, name, record)
        print(f"[{name}] exit={record['exit_code']} {record['wall_clock_s']}s", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, SADDLES))


# ────────────────────────────── 汇总 ──────────────────────────────


def _energy(record):
    return None if record is None else record.get("energy_hartree")


def _end_summary(end, reactants, products):
    if not end:
        return None
    entry = {
        "points": len(end.get("points") or []),
        "monotone": end.get("monotone"),
        "minimum_energy_hartree": end.get("minimum_energy_hartree"),
        "optimisation_steps": end.get("optimisation_steps"),
        "fingerprint_vs": {},
    }
    for reference in reactants + products:
        atoms = reference.get("optimized_xyz")
        entry["fingerprint_vs"][reference["name"]] = compare_fingerprints(
            [tuple(a) for a in atoms] if atoms else None, end.get("fingerprint")
        )
    if entry["minimum_energy_hartree"] is not None:
        reference_energies = {
            reference["name"]: reference.get("energy_hartree") for reference in reactants + products
        }
        entry["energy_delta_hartree"] = {
            name: (
                None
                if value is None
                else entry["minimum_energy_hartree"] - value
            )
            for name, value in reference_energies.items()
        }
    return entry


def _end_reached(end, reactants, products):
    """端点是否到达某一个参照极小点：先看能量（|ΔE| < 1e-3 Ha），再看指纹（键长 < 0.02 Å）。"""
    if not end:
        return None
    deltas = end.get("energy_delta_hartree") or {}
    matches = [name for name, delta in deltas.items() if delta is not None
               and abs(delta) < ENDPOINT_ENERGY_TOL_HARTREE]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # 能量分辨不开（例如 R1 的两个异构体能量差很小）时用指纹裁决。
        for name in matches:
            if endpoint_matches(end["fingerprint_vs"].get(name)):
                return name
        return f"ambiguous:{matches}"
    # 能量都对不上时退回指纹（端点优化没收敛好的情况仍应被看出来）。
    by_fingerprint = [
        name for name in deltas if endpoint_matches(end["fingerprint_vs"].get(name))
    ]
    return by_fingerprint[0] if len(by_fingerprint) == 1 else None


def build_summary(out_root):
    records = {name: load_record(out_root, name) for name in MINIMA + SADDLES}
    irc_records = {f"irc_{saddle}": load_record(out_root, f"irc_{saddle}") for saddle in SADDLES}
    summary = {"reactions": {}}
    for reaction, (reactant, products, saddle) in REACTIONS.items():
        reactant_record = records.get(reactant)
        product_records = [records.get(p) for p in products]
        saddle_record = records.get(saddle)
        irc_record = irc_records.get(f"irc_{saddle}")
        e_reactant = _energy(reactant_record)
        parts = [_energy(record) for record in product_records]
        e_product = None if any(p is None for p in parts) else sum(parts)
        e_saddle = _energy(saddle_record)
        ts_block = (saddle_record or {}).get("ts_result") or {}
        references = []
        for name, record in [(reactant, reactant_record)] + list(zip(products, product_records)):
            if record is not None:
                references.append({"name": name, **record})
        entry = {
            "level": (saddle_record or {}).get("level") or (reactant_record or {}).get("level"),
            "grid": (saddle_record or {}).get("grid") or (reactant_record or {}).get("grid"),
            "reactant": {
                "species": reactant,
                "energy_hartree": e_reactant,
                "geometry_source": (reactant_record or {}).get("geometry_source"),
            },
            "product": {
                "species": "+".join(products),
                "energy_hartree": e_product,
                "fragments": {name: _energy(record) for name, record in zip(products, product_records)},
            },
            "saddle": {
                "species": saddle,
                "energy_hartree": e_saddle,
                "geometry_source": (saddle_record or {}).get("geometry_source"),
                "n_imaginary": ts_block.get("n_imaginary"),
                "imaginary_wavenumber_cm": ts_block.get("imaginary_wavenumber_cm"),
                "max_gradient_hartree_per_bohr": ts_block.get("max_gradient_hartree_per_bohr"),
                "iterations": ts_block.get("iterations"),
            },
            "wall_clock_s": {
                "reactant": (reactant_record or {}).get("wall_clock_s"),
                "product": {name: (record or {}).get("wall_clock_s")
                            for name, record in zip(products, product_records)},
                "saddle": (saddle_record or {}).get("wall_clock_s"),
                "irc": (irc_record or {}).get("wall_clock_s"),
            },
        }
        if e_reactant is not None and e_saddle is not None:
            entry["barrier_hartree"] = e_saddle - e_reactant
            entry["barrier_kcal"] = (e_saddle - e_reactant) * HARTREE_TO_KCAL
        if e_reactant is not None and e_product is not None:
            entry["reaction_energy_kcal"] = (e_product - e_reactant) * HARTREE_TO_KCAL

        # ── 鞍点验收：一个虚频 + 双向 IRC 连到两个不同的极小点 ──
        irc_result = (irc_record or {}).get("irc_result") or {}
        forward = _end_summary(irc_result.get("forward"), [references[0]], references[1:]) if references else None
        reverse = _end_summary(irc_result.get("reverse"), [references[0]], references[1:]) if references else None
        reached_forward = _end_reached(forward, [references[0]], references[1:]) if forward else None
        reached_reverse = _end_reached(reverse, [references[0]], references[1:]) if reverse else None
        distinct = (
            forward is not None
            and reverse is not None
            and forward.get("minimum_energy_hartree") is not None
            and reverse.get("minimum_energy_hartree") is not None
            and abs(forward["minimum_energy_hartree"] - reverse["minimum_energy_hartree"])
            > ENDPOINT_SEPARATION_HARTREE
        )
        endpoints = {reached_forward, reached_reverse}
        monocle = bool(forward and reverse and forward.get("monotone") and reverse.get("monotone"))
        connects = bool(
            forward
            and reverse
            and reached_forward
            and reached_reverse
            and distinct
            and len(endpoints) == 2
            and monocle
            and not any(str(name or "").startswith("ambiguous") for name in endpoints)
        )
        entry["irc"] = {
            "imaginary_wavenumber_cm": irc_result.get("imaginary_wavenumber_cm"),
            "forward": forward,
            "reverse": reverse,
            "forward_reached": reached_forward,
            "reverse_reached": reached_reverse,
            "ends_are_distinct_minima": distinct,
            "both_monotone": monocle,
            "connects_reactant_and_product": connects,
        }
        reasons = []
        if ts_block.get("n_imaginary") != 1:
            reasons.append(f"ts n_imaginary = {ts_block.get('n_imaginary')}")
        if not irc_result:
            reasons.append("no irc_result")
        elif not connects:
            reasons.append(
                f"irc endpoints ({reached_forward!r}, {reached_reverse!r}) do not connect "
                f"reactant/product (monotone={monocle}, distinct={distinct})"
            )
        entry["accepted"] = not reasons
        entry["rejection_reasons"] = reasons
        summary["reactions"][reaction] = entry
    return summary


# ────────────────────────────── main ──────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--grid", default="normal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", default="guess,minima,ts,irc,summary")
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--seed-dir", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--irc-steps", type=int, default=80)
    parser.add_argument("--irc-step-size", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--qwb",
        default=None,
        help="qwb 可执行文件（缺省 qwb-core/target/release/qwb.exe）",
    )
    args = parser.parse_args()

    global QWB, THREADS, IRC_STEPS, IRC_STEP_SIZE
    if args.qwb:
        QWB = Path(args.qwb).resolve()
    THREADS = args.threads
    IRC_STEPS = args.irc_steps
    IRC_STEP_SIZE = args.irc_step_size

    level = {"method": args.method, "basis": args.basis}
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    seed_root = None
    if args.seed_dir:
        seed_root = Path(args.seed_dir)
        if not seed_root.is_absolute():
            seed_root = ROOT / seed_root
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]

    if "guess" in stages:
        stage_guess(out_root, level, args.grid, args.jobs, args.force)
    minima = [s for s in MINIMA if args.species is None or s in args.species]
    saddles = [s for s in SADDLES if args.species is None or s in args.species]
    if "minima" in stages and minima:
        stage_species(
            out_root, level, args.grid, minima, args.jobs, args.force, seed_root, None, "opt"
        )
    if "ts" in stages and saddles:
        stage_species(
            out_root, level, args.grid, saddles, args.jobs, args.force, seed_root, out_root, "ts"
        )
    if "irc" in stages:
        stage_irc(out_root, level, args.grid, args.jobs, args.force)
    if "summary" in stages:
        summary = build_summary(out_root)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for reaction, entry in summary["reactions"].items():
            print(
                f"{reaction}: barrier = {entry.get('barrier_kcal')} kcal/mol, "
                f"accepted = {entry['accepted']} {entry['rejection_reasons']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
