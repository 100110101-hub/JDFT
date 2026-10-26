#!/usr/bin/env python3
"""用 **PySCF** 独立算参考值（只做黑盒对照，§14 #12）。

project.md 的参考值政策（§7.0）：能量判据优先用**外部**来源。T2 一直卡在
"H2O / PBE / def2-SVP 的总能量在可引用表里查不到"，而本机 Windows 侧装不上 PySCF
（Python 3.12 没有 wheel，源码构建失败）。这条路现在走通了：**WSL 的 Ubuntu 22.04
自带 Python 3.10，`pip install --user pyscf` 有现成 wheel**。

这个脚本是本仓库里唯一允许"调用别的电子结构代码"的地方 —— 它只**产出数字**，
不把任何代码/数据拷进 `qwb-data`；我们的实现一行都不依赖它（§14 #12 的边界）。

用法（在 WSL 里跑，输出 JSON）：

    python3 scripts/pyscf_reference.py --json

依赖：`python3 -m pip install --user pyscf`（WSL）。
"""

from __future__ import annotations

import argparse
import json
import sys

# 与 qwb-core 的测试保持**逐字一致**的几何与模型化学。
WATER_ANGSTROM = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.757, 0.586)),
    ("H", (0.0, -0.757, 0.586)),
]


def run(xc: str, basis: str, grid_level: int, conv_tol: float) -> dict:
    from pyscf import dft, gto

    mol = gto.M(
        atom=[(symbol, position) for symbol, position in WATER_ANGSTROM],
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )
    mean_field = dft.RKS(mol)
    mean_field.xc = xc
    mean_field.grids.level = grid_level
    mean_field.conv_tol = conv_tol
    mean_field.conv_tol_grad = conv_tol
    energy = mean_field.kernel()
    return {
        "code": "pyscf",
        "version": __import__("pyscf").__version__,
        "xc": xc,
        "basis": basis,
        "geometry_angstrom": WATER_ANGSTROM,
        "grids_level": grid_level,
        "conv_tol": conv_tol,
        "nao": int(mol.nao),
        "nelectron": int(mol.nelectron),
        "energy_hartree": float(energy),
        "converged": bool(mean_field.converged),
        "mo_energies_hartree": [float(value) for value in mean_field.mo_energy],
        "dipole_debye": [float(value) for value in mean_field.dip_moment(verbose=0)],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="PySCF 的 dft.grids.level（3 = 默认，约等于 (75,302)；4 = 更细）",
    )
    parser.add_argument("--conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--json", action="store_true", help="只打印 JSON")
    args = parser.parse_args(argv)

    payload = run(args.xc, args.basis, args.grid_level, args.conv_tol)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"{payload['code']} {payload['version']}  {payload['xc']}/{payload['basis']}  "
            f"grid level {payload['grids_level']}  nao = {payload['nao']}"
        )
        print(f"E = {payload['energy_hartree']:.12f} Ha  (converged = {payload['converged']})")
    return 0 if payload["converged"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
