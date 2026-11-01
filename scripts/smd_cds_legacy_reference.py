#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 PySCF 2.14.0 的 `libsolvent`（`pyscf.solvent.smd.get_cds_legacy`）取**多原子** CDS 参考。

# 为什么需要这份参考

单原子把表面积消掉之后，`E_CDS = σ·A` 里的 σ 已经被钉住（`scripts/validate_smd_cds.py`：
水相 σ 1.6e-5 cal/(mol·Å²)、五个非水溶剂 3.8e-7 kcal/mol）；但**面积**从来没有在多原子
体系上验证过 —— 我们的面积是 Fibonacci 点采样，上游是 `CDSCALC`→`DAREAL` 自己的一套。

`get_cds_legacy` 返回 `(E_CDS [Ha], dE/dR [Ha/Bohr])`，**不含** `areacds`（接口签名是
能量 + 梯度），所以面积没法直接读。这里取巧：选**表面张力只有零阶项**的体系 ——
H/F/Cl/S 的 H 修正 `HSIGMA(9)/HSIGMA(17)/HSIGMA(16)` 全为 0、成对项 `SIGMA(100+pair)`
只对 C/N/Si/P 邻居生效（`choices` 一百六十五 四）—— 于是

    E_CDS = Σ_i σ(Z_i)·A_i / 1000

只差**面积**。任何含 C–H/C–O/C–N 的分子都要先补上 H/成对修正才能比，所以本文件只收
水、HF、HCl、H₂S 四个体系（都满足"零阶即全部"）。

# 列

长表：一行 = (体系, 溶剂, 原子序号, 元素, x, y, z, 该体系的 E_CDS, 该原子三个梯度分量)。
重复写体系和能量是有意的：Rust 侧按 `system` 分组就能直接重建分子，不需要另一个几何文件。

运行（WSL，需要带 `libsolvent` 的 PySCF 2.14.0）：

    python3 scripts/smd_cds_legacy_reference.py            # 写 CSV
    python3 scripts/smd_cds_legacy_reference.py --check    # 逐字比对（不改文件）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_sources  # noqa: E402

OUTPUT = os.path.join(
    reference_sources.ROOT,
    "qwb-core/crates/qwb-props/tests/reference/smd_cds_legacy.csv",
)
FIELDS = (
    "system",
    "solvent",
    "atom_index",
    "symbol",
    "x",
    "y",
    "z",
    "e_cds_kcal_per_mol",
    "grad_x",
    "grad_y",
    "grad_z",
)

#: Hartree → kcal/mol。与 Fortran 里的 `TOKCAL = 627.509451` 差 3.7e-8 相对（那一位是
#: 1986 年的常数），在我们的 1e-3 判据下无关；这里用 CODATA 2018 的值，与仓库其他脚本一致。
HARTREE2KCAL = 627.5094740631

#: 体系：(名称, [(元素, x, y, z)])，单位 Å。几何随意但要远离线性依赖（CDS 只看几何）。
SYSTEMS = (
    (
        "water",
        (("O", 0.0, 0.0, 0.11779), ("H", 0.0, 0.75545, -0.47116), ("H", 0.0, -0.75545, -0.47116)),
    ),
    ("hydrogen-fluoride", (("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9168))),
    ("hydrogen-chloride", (("Cl", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.2745))),
    (
        "hydrogen-sulfide",
        (("S", 0.0, 0.0, 0.1025), ("H", 0.0, 0.9661, -0.8283), ("H", 0.0, -0.9661, -0.8283)),
    ),
)

#: 只做水相（`icds = 1`）：非水还要叠 `CSSIGM·A_total`，那是另一个自由度；零阶表验证
#: 不需要它，加进来只会让判据更难区分"σ 错"与"面积错"。
SOLVENT = "water"


def rows() -> list[dict[str, str]]:
    import pyscf
    from pyscf import gto
    from pyscf.solvent import smd

    if smd.libsolvent is None:
        raise SystemExit("smd_cds_legacy_reference: PySCF here has no libsolvent")
    print(f"pyscf {pyscf.__version__}, libsolvent available")

    out: list[dict[str, str]] = []
    for name, atoms in SYSTEMS:
        mol = gto.M(
            atom="; ".join(f"{symbol} {x} {y} {z}" for symbol, x, y, z in atoms),
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        obj = smd.SMD(mol)
        obj.solvent = SOLVENT
        energy, gradient = smd.get_cds_legacy(obj)
        energy_kcal = float(energy) * HARTREE2KCAL
        print(f"  {name:20s} E_CDS = {energy_kcal:.6f} kcal/mol")
        for index, (symbol, x, y, z) in enumerate(atoms):
            gx, gy, gz = (float(value) for value in gradient[index])
            out.append(
                {
                    "system": name,
                    "solvent": SOLVENT,
                    "atom_index": str(index),
                    "symbol": symbol,
                    "x": f"{x:.10f}",
                    "y": f"{y:.10f}",
                    "z": f"{z:.10f}",
                    "e_cds_kcal_per_mol": f"{energy_kcal:.12g}",
                    "grad_x": f"{gx:.12g}",
                    "grad_y": f"{gy:.12g}",
                    "grad_z": f"{gz:.12g}",
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
