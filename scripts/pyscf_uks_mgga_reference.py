#!/usr/bin/env python3
"""SC-06：开壳层 UKS × meta-GGA 的 **SCF 总能量**参考值（PySCF 2.14.0 / libxc 7.0.0）。

为什么需要这一份：SC-06 的判据是「至少 1 个开壳层体系在 M06-2X/D3 下跑通」。逐点测试
（`qwb-xc/tests/sc06_spin_mgga.rs`、`sc06_spin_m06_correlation.rs`）钉的是**公式**，
这里钉的是**整条链路**（`uks` 的 τ 积累 + Fock 的 τ 矩阵元 + 自旋相关的 ζ 插值 +
hybrid 的精确交换比例）—— 三者任何一处写错，总能量都会跑偏远不止 0.5 mHa。

体系：O₂ 三重态（16 个电子、9α/7β），STO-3G，键长 1.208 Å（与
`qwb-scf/tests/fx01_tau_pipeline.rs` 的 `o2()` 同一个几何）。选它是因为**开壳层**、
足够小（10 个基函数）、且三重态的 α/β 不对称能让自旋通道的错误暴露出来。

Run (from Windows, PySCF 只在 WSL 里):
  wsl -e bash -lc "cd /mnt/d/python_II/Jaguar && python3 scripts/pyscf_uks_mgga_reference.py"
  wsl -e bash -lc "cd /mnt/d/python_II/Jaguar && python3 scripts/pyscf_uks_mgga_reference.py --check"
"""
from __future__ import annotations

import argparse
import os
import sys

from pyscf import dft, gto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_sources  # noqa: E402

ROOT = reference_sources.ROOT
CSV_PATH = os.path.join(
    ROOT, "qwb-core/crates/qwb-scf/tests/reference/uks_mgga.csv"
)

GEOMETRY = "O 0.0 0.0 0.0; O 0.0 0.0 1.208"
BASIS = "sto-3g"
CHARGE = 0
SPIN = 2  # triplet: 2S = 2
# PySCF 的默认网格（level 3）与 qwb 的 `grid-normal` 尺度相当：FX-02 在 H2O 上实测两头
# 相差 ~1e-5 Ha，所以 0.5 mHa 的判据里网格不是瓶颈。
GRID_LEVEL = 3

FUNCTIONALS = [("m06-l", "MGGA_C_M06_L"), ("m06-2x", "MGGA_C_M06_2X")]

FIELDS = [
    "case",
    "functional",
    "libxc_id",
    "geometry_angstrom",
    "basis",
    "charge",
    "spin_2s",
    "grid_level",
    "energy_hartree",
]


def evaluate(xc: str) -> float:
    mol = gto.M(
        atom=GEOMETRY,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    energy = mf.kernel()
    assert mf.converged, f"{xc}: UKS did not converge"
    # 开壳层自洽性：电子数与多重度
    n_alpha = (mol.nelectron + mol.spin) // 2
    n_beta = mol.nelectron - n_alpha
    assert (n_alpha, n_beta) == (9, 7), (n_alpha, n_beta)
    print(f"{xc:10} E(UKS) = {energy:.12f} Ha  n_alpha/n_beta = {n_alpha}/{n_beta}")
    return float(energy)


def build_rows():
    rows = []
    for xc, libxc_id in FUNCTIONALS:
        energy = evaluate(xc)
        rows.append(
            {
                "case": "o2_triplet",
                "functional": xc,
                "libxc_id": libxc_id,
                "geometry_angstrom": GEOMETRY,
                "basis": BASIS,
                "charge": CHARGE,
                "spin_2s": SPIN,
                "grid_level": GRID_LEVEL,
                "energy_hartree": f"{energy:.12f}",
            }
        )
    return reference_sources.stamp(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    if args.check:
        return reference_sources.check_csv(CSV_PATH, FIELDS, rows)

    reference_sources.write_csv(CSV_PATH, FIELDS, rows)
    print(f"wrote {reference_sources.rel(CSV_PATH)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
