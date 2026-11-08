#!/usr/bin/env python3
"""RP-01 的**鞍点初猜**（PySCF 侧，WSL 里跑）。

**它不是测量，也不是参考值。** 报出的势垒一律来自 `qwb ts` / `qwb irc` 在 PBE/def2-SVP 上
收敛的鞍点；这个脚本只解决一个**成本**问题：`qwb ts` 是局部搜索，每一步重算完整数值
Hessian（`6N` 次解析梯度），从坏初猜出发要几十步 —— PBE/def2-SVP 上这个乘积是小时量级
（实测见报告）。所以先用项目钉住的黑盒（PySCF 2.14.0）把几何推到鞍点附近，交给 qwb 收尾
（1–3 步）。用另一套实现给初猜是本项目一贯的做法 —— `rp01_measure.py --seed-dir` 就是
同样的事，只是换层级当种子。

# 算法（与 `qwb-engine::ts` 同一个思路，便宜在 Hessian 只算一次）

1. 每个几何一次**解析梯度**（PySCF 的 RKS/RHF 解析梯度，Ha/Bohr）；
2. 初始 Hessian 用中心差分数值差分（`6N` 次梯度），之后用 **BFGS 更新**
   （质量加权坐标：`H̃ ← H̃ + ỹỹᵀ/(ỹᵀs̃) − (H̃s̃)(H̃s̃)ᵀ/(s̃ᵀH̃s̃)`）；
3. 质量加权对角化，**最低本征方向的本征值取 −|λ|**（沿反应坐标上坡），其余照常下山；
4. 步长上限 0.10 Å、线搜索折半（能量只升不降时减半，最多 4 次）；
5. 收敛判据 `max|g| < 4.5e-4 Ha/Bohr`（**与 qwb 的 `ts` 同一条判据**）；
6. 收敛后再做一次完整数值 Hessian，报波数并核对"恰好一个虚频"。

**内部一律用 Bohr**（PySCF 的梯度是 Ha/Bohr、Hessian 是 Ha/Bohr²，Å 只出现在读写 xyz 时）。
第一版把 Bohr 当 Å 加进坐标里，位移小了 0.53 倍 —— 那是个静默的错误步长。

这个脚本**不判收**：它的输出只是一个 `.xyz`，成不成由 qwb 的 `ts`（一个虚频）与 `irc`
（双向连到反应物/产物）说了算。

用法::

    wsl -e python3 scripts/rp01_pyscf_ts_guess.py --species r2_ts --method pbe --basis def2-svp \
        --start .tmp-rp01-measure/dry-hf/guesses/r2_ts.xyz --out .tmp-rp01-guess
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pyscf import dft, gto, scf

#: Å → Bohr（CODATA 2018）。参考值文件的 `note` 里也写同一个常数。
ANGSTROM_TO_BOHR = 1.8897261246257702
#: `sqrt(Ha / (Bohr² · amu))` → cm⁻¹（与 `qwb-engine::hessian::WAVE_NUMBER_FACTOR` 同一个数）。
WAVE_NUMBER_FACTOR = 5140.4871
#: 标准原子量（与 `qwb-engine::hessian::atomic_mass` 同一套）。
ATOMIC_MASS = {"H": 1.008, "He": 4.002602, "C": 12.011, "N": 14.007, "O": 15.999,
               "F": 18.998403}
#: qwb `ts` 的收敛判据（Ha/Bohr），两边同一条。
GRADIENT_THRESHOLD = 4.5e-4
#: 数值 Hessian 的差分步长（Bohr）—— 与 qwb 的 `freq_step_angstrom: 0.01` 同量级。
HESSIAN_STEP_BOHR = 0.01 * ANGSTROM_TO_BOHR
#: 单步位移上限（Bohr）= 0.10 Å。
MAX_STEP_BOHR = 0.10 * ANGSTROM_TO_BOHR


def read_xyz(path):
    lines = [line for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    count = int(lines[0].split()[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        atoms.append((parts[0], np.array([float(x) for x in parts[1:4]]) * ANGSTROM_TO_BOHR))
    return atoms


def write_xyz(path, atoms, comment):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n{comment}\n")
        for symbol, position in atoms:
            # `float(...)` 是必须的：numpy 标量的 `repr` 是 `np.float64(...)`，写进 xyz 之后
            # 谁读谁崩（本项目自己就崩过一次）。
            handle.write(
                f"{symbol} {float(position[0] / ANGSTROM_TO_BOHR)!r} "
                f"{float(position[1] / ANGSTROM_TO_BOHR)!r} "
                f"{float(position[2] / ANGSTROM_TO_BOHR)!r}\n"
            )


def energy_gradient(method, basis, symbols, coords_bohr, grid_level=3):
    """返回 (E Ha, g Ha/Bohr)。每个几何重建一次分子与 SCF（不留状态，最不容易出错）。"""
    atom = [
        (symbol, tuple(position / ANGSTROM_TO_BOHR))
        for symbol, position in zip(symbols, coords_bohr)
    ]
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    if method == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = method
        mf.grids.level = grid_level
    mf.conv_tol = 1.0e-9
    mf.max_cycle = 200
    mf.verbose = 0
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("pySCF SCF did not converge")
    gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float).reshape(-1)
    return float(energy), gradient


def numerical_hessian(method, basis, symbols, coords_bohr, grid_level=3):
    """中心差分数值 Hessian（Ha/Bohr²），`2 × 3N` 次梯度；`H[i][j] = ∂g_i/∂x_j`。"""
    dof = 3 * len(symbols)
    hessian = np.zeros((dof, dof))
    for atom in range(len(symbols)):
        for axis in range(3):
            column = 3 * atom + axis
            gradients = []
            for sign in (+1.0, -1.0):
                shifted = np.array(coords_bohr, dtype=float)
                shifted[atom, axis] += sign * HESSIAN_STEP_BOHR
                gradients.append(
                    energy_gradient(method, basis, symbols, shifted, grid_level)[1]
                )
            hessian[:, column] = (gradients[0] - gradients[1]) / (2.0 * HESSIAN_STEP_BOHR)
    return 0.5 * (hessian + hessian.T)


def masses_of(symbols):
    return np.array([ATOMIC_MASS[symbol] for symbol in symbols])


def wavenumbers(masses, mass_weighted_hessian):
    eigenvalues = np.linalg.eigvalsh(mass_weighted_hessian)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * WAVE_NUMBER_FACTOR


def run(species, start_atoms, method, basis, out_dir, grid_level=3, max_steps=40):
    symbols = [atom[0] for atom in start_atoms]
    coords = np.array([atom[1] for atom in start_atoms], dtype=float)
    masses = masses_of(symbols)
    weights = np.sqrt(np.repeat(masses, 3))
    energy, gradient = energy_gradient(method, basis, symbols, coords, grid_level)
    hessian = numerical_hessian(method, basis, symbols, coords, grid_level)
    weighted_hessian = hessian / np.outer(weights, weights)
    history = [
        {"step": 0, "energy_hartree": energy, "max_gradient": float(np.abs(gradient).max())}
    ]
    print(
        f"[{species}] start E = {energy:.10f} Ha, max|g| = {np.abs(gradient).max():.3e}",
        flush=True,
    )
    for step in range(1, max_steps + 1):
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_hessian)
        g_tilde = gradient.ravel() / weights
        step_tilde = np.zeros_like(g_tilde)
        for index, value in enumerate(eigenvalues):
            direction = eigenvectors[:, index]
            effective = -abs(value) if index == 0 else value
            if abs(effective) < 1.0e-6:
                effective = 1.0e-6
            step_tilde -= (g_tilde @ direction) / effective * direction
        step_cartesian = (step_tilde / weights).reshape(-1, 3)
        norm = np.linalg.norm(step_cartesian)
        if norm > MAX_STEP_BOHR:
            step_cartesian *= MAX_STEP_BOHR / norm
        factor = 1.0
        accepted = None
        for _ in range(4):
            trial = coords + factor * step_cartesian
            try:
                trial_energy, trial_gradient = energy_gradient(
                    method, basis, symbols, trial, grid_level
                )
            except RuntimeError:
                factor *= 0.5
                continue
            if trial_energy <= energy or factor < 0.2:
                accepted = (trial, trial_energy, trial_gradient, factor)
                break
            factor *= 0.5
        if accepted is None:
            print(f"[{species}] line search failed at step {step}", flush=True)
            break
        trial, trial_energy, trial_gradient, factor = accepted
        # BFGS 更新，全程在质量加权坐标里（s̃ 单位 Bohr·√amu，ỹ 单位 Ha/(Bohr·√amu)）。
        s_tilde = (trial - coords).ravel() * weights
        y_tilde = (trial_gradient - gradient).ravel() / weights
        hs = weighted_hessian @ s_tilde
        curvature = s_tilde @ y_tilde
        if abs(curvature) > 1.0e-12:
            weighted_hessian += (
                np.outer(y_tilde, y_tilde) / curvature - np.outer(hs, hs) / (s_tilde @ hs)
            )
        coords, energy, gradient = trial, trial_energy, trial_gradient
        max_gradient = float(np.abs(gradient).max())
        history.append(
            {
                "step": step,
                "energy_hartree": energy,
                "max_gradient": max_gradient,
                "line_search_factor": factor,
                "lowest_eigenvalue": float(eigenvalues[0]),
            }
        )
        print(
            f"[{species}] step {step}: E = {energy:.10f} Ha, max|g| = {max_gradient:.3e}, "
            f"lowest eig = {eigenvalues[0]:.6e}, ls = {factor}",
            flush=True,
        )
        if max_gradient < GRADIENT_THRESHOLD:
            break
    # 收尾：重新算一次完整 Hessian（BFGS 更新过的 Hessian 不能用来数虚频）。
    final_hessian = numerical_hessian(method, basis, symbols, coords, grid_level)
    wave = wavenumbers(masses, final_hessian / np.outer(weights, weights))
    imaginary = [float(w) for w in wave if w < -100.0]
    atoms = [(symbol, position) for symbol, position in zip(symbols, coords)]
    path = os.path.join(out_dir, f"{species}.xyz")
    write_xyz(path, atoms, f"pyscf {method}/{basis} ts guess for {species}")
    record = {
        "species": species,
        "method": method,
        "basis": basis,
        "grid_level": grid_level,
        "final_energy_hartree": energy,
        "max_gradient_hartree_per_bohr": float(np.abs(gradient).max()),
        "converged": float(np.abs(gradient).max()) < GRADIENT_THRESHOLD,
        "wavenumbers_cm": [float(w) for w in wave],
        "imaginary_below_minus_100_cm": imaginary,
        "history": history,
        "xyz": path,
    }
    with open(os.path.join(out_dir, f"{species}.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "converged": record["converged"],
                "max_gradient_hartree_per_bohr": record["max_gradient_hartree_per_bohr"],
                "imaginary_below_minus_100_cm": imaginary,
            },
            indent=2,
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--species", required=True)
    parser.add_argument("--method", default="pbe")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--start", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    atoms = read_xyz(args.start)
    record = run(
        args.species,
        atoms,
        args.method,
        args.basis,
        args.out,
        grid_level=args.grid_level,
        max_steps=args.max_steps,
    )
    return 0 if record["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
