#!/usr/bin/env python3
"""CuH / LANL2DZ / PBE 的 PySCF 2.14.0 复算（用于 T5 的 Cu ECP 钉的独立标定）。

几何/基组/ECP/泛函与 `qwb-core/crates/qwb-engine/tests/t5_ecp.rs` 的 `cuh(1.46)` 逐项对齐：
Cu 在原点、H 在 z = 1.46 Å，LANL2DZ（Cu 10 芯 ECP，z_eff = 19；H 全电子），PBE。

用法（WSL，PySCF 2.14.0）：

    python3 scripts/pyscf_cuh_lanl2dz_reference.py            # 打印数字
    python3 scripts/pyscf_cuh_lanl2dz_reference.py --check    # 与 tests/reference 的 CSV 比
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import PYSCF_DOI, PYSCF_URL, rel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "cuh_lanl2dz_pbe_ecp.csv"
)

HEADER = ["quantity", "value_hartree", "tolerance_hartree", "source", "source_doi", "source_url"]
TOLERANCE = "0.005"
SOURCE = "pyscf-2.14.0-pbe"
DISTANCE = 1.46


def energy(grid_level: int) -> float:
    from pyscf import dft, gto

    mol = gto.M(
        atom=[("Cu", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, DISTANCE))],
        unit="Angstrom",
        basis="lanl2dz",
        ecp="lanl2dz",
        charge=0,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol)
    method.xc = "PBE"
    method.grids.level = grid_level
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-7
    method.max_cycle = 500
    value = method.kernel()
    if not method.converged:
        solver = method.newton()
        solver.conv_tol = 1.0e-11
        value = solver.kernel()
        method = solver
    if not method.converged:
        raise RuntimeError(f"PySCF did not converge (grid {grid_level})")
    return float(value)


def rows():
    coarse = energy(1)
    normal = energy(2)
    out = [
        {"quantity": "cuh_total_energy_grid_coarse", "value_hartree": f"{coarse:.10f}"},
        {"quantity": "cuh_total_energy_grid_normal", "value_hartree": f"{normal:.10f}"},
    ]
    for row in out:
        row["tolerance_hartree"] = TOLERANCE
        row["source"] = SOURCE
        row["source_doi"] = PYSCF_DOI
        row["source_url"] = PYSCF_URL
    return out


COMMENTS = """\
# T5：CuH / LANL2DZ（Cu 10 芯 ECP，z_eff = 19）/ PBE —— **PySCF 2.14.0 的复算值**
#
# 生成器：scripts/pyscf_cuh_lanl2dz_reference.py（`--check` 用同一份代码重算并逐字比对）。
# 几何 r(Cu–H) = 1.46 Å，Cu 在原点、H 在 +z；泛函 PBE；`grids.level = 1`（对应本项目的
# grid-coarse）与 `2`（grid-normal）。RKS（Cu 的 z_eff = 19 加 H 的 1 个电子 = 20，闭壳层单重态，与 Molecule::new(..., 0, 1) 的 multiplicity 一致）。
#
# 为什么需要这份 CSV：t5_ecp.rs 的两个 CuH 能量是**回归钉**（internal-independent-path），
# 2026-11 修掉 ECP 的 `r` 幂次约定与通道语义之后它们整体移动，重新钉必须用**外部标尺**
# （PySCF）而不是本项目自己的输出。
"""


def render(data) -> str:
    lines = COMMENTS.rstrip("\n").split("\n")
    lines.append(",".join(HEADER))
    for row in data:
        lines.append(",".join(row[column] for column in HEADER))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    text = render(rows())
    if args.write:
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(text)
        print(f"wrote {rel(OUTPUT)}")
        return 0
    if args.check:
        current = io.open(OUTPUT, encoding="utf-8").read() if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"FAIL {rel(OUTPUT)} does not match a fresh PySCF run")
            return 1
        print(f"ok: {rel(OUTPUT)} matches PySCF")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
