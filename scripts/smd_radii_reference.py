#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 PySCF 的 `smd.smd_radii(alpha)` 取 **SMD 参数化半径** 的参考值。

`smd_radii` 是 Marenich–Cramer–Truhlar 2009 eq. 16 的实现：`radii.VDW` 基础表 + SMD 的逐元素
覆盖项 + O 依赖溶剂 Abraham α 的规则。它返回 **bohr**（函数里先乘 BOHR 再除回去），所以这份
参考值也按 bohr 存 —— 单位换算交给 Rust 侧的 `BOHR_PER_ANGSTROM`（CODATA 2018），两边不各自
定义一个常数。

运行（WSL，需要 PySCF 2.14.0）：

    python3 scripts/smd_radii_reference.py            # 写 CSV
    python3 scripts/smd_radii_reference.py --check    # 逐字比对
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_sources  # noqa: E402

OUTPUT = os.path.join(
    reference_sources.ROOT,
    "qwb-core/crates/qwb-props/tests/reference/smd_parameterised_radii.csv",
)
FIELDS = ("symbol", "atomic_number", "alpha", "radius_bohr", "radius_angstrom")

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]
#: 三种 α：0.0（低氢键酸性，O 走"变宽"分支）、0.43（阈值，正好落在等号上）、0.5（高 α）。
ALPHAS = (0.0, 0.43, 0.5)


def rows() -> list[dict[str, str]]:
    import pyscf
    from pyscf.data import radii as radii_module
    from pyscf.solvent import smd

    print(f"pyscf {pyscf.__version__}")
    out: list[dict[str, str]] = []
    for alpha in ALPHAS:
        radii = smd.smd_radii(alpha)
        for index, symbol in enumerate(ELEMENTS, start=1):
            out.append(
                {
                    "symbol": symbol,
                    "atomic_number": str(index),
                    "alpha": f"{alpha:.2f}",
                    "radius_bohr": f"{float(radii[index]):.12g}",
                    # 同一份数据换成 Å：PySCF 的 `smd_radii` 先 Å→bohr 再返回 bohr，
                    # 这里乘回它自己的 `BOHR`，于是 Å 列与"源码里写的 Å"只差浮点往返。
                    "radius_angstrom": f"{float(radii[index]) * radii_module.BOHR:.17g}",
                }
            )
    return reference_sources.stamp(out)


def main(argv: list[str]) -> int:
    table = rows()
    if "--check" in argv:
        return reference_sources.check_csv(OUTPUT, FIELDS, table)
    reference_sources.write_csv(OUTPUT, FIELDS, table)
    print(f"wrote {reference_sources.rel(OUTPUT)} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
