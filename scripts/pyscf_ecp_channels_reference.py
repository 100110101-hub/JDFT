#!/usr/bin/env python3
"""PySCF 2.14.0 的 LANL2DZ ECP 矩阵，**按 l 通道拆开**（I-05 的外部标尺）。

为什么要有这一份：`tests/i05_ecp.rs` 原来只比"本项目的两条实现"（解析 vs 半数值），
两条实现共享同一套数据解释，所以**猜错数据约定时两条会一起错**（2026-11 实测：Zn 的
`Tr(S⁻¹V_ecp)` 本项目 `+0.485` vs PySCF `-24.797`，而内部两条实现互差 `3e-14`）。
这里把 PySCF/libcint 的同一个矩阵也按通道拆开，作为外部对照。

通道怎么拆（**测量**出来的 libcint 语义，不是猜的）：

    ECPscalar = V_local + Σ_l (V_l P_l)

* `ECPscalar` 只带局域块（导出里的最高 l，`l = -1` 的 UL 块）→ `local`；
* `ECPscalar` 带局域 + 第 l 块、减掉只带局域的那一份 → `delta_l{l}`，即 `V_l P_l`；
* 两者相加逐项复现 `ecp='lanl2dz'` 的 `ECPscalar`（差 `8.9e-16`，见脚本打印）。

`value_hartree` 取每个 (shell_i, shell_j) 块的 `(first_i, first_j)` 元：球对称算符在一中心
下该块是 `v·I`，所以这一个元就代表整块（Rust 侧的测试同时断言块内是常数）。

用法（WSL，PySCF 2.14.0）：

    python3 scripts/pyscf_ecp_channels_reference.py --write
    python3 scripts/pyscf_ecp_channels_reference.py --check
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import PYSCF_DOI, PYSCF_URL, check_csv, rel, write_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-scf", "tests", "reference", "ecp_lanl2dz_channels.csv"
)

FIELDS = ["element", "channel", "ao_i", "ao_j", "value_hartree"]
SPIN = {"Zn": 0, "Cu": 1}


def build(element, blocks):
    from pyscf import gto

    return gto.M(
        atom=[(element, (0.0, 0.0, 0.0))],
        unit="Angstrom",
        basis="lanl2dz",
        ecp={element: blocks},
        spin=SPIN[element],
        verbose=0,
    )


def reference(element):
    from pyscf import gto

    return gto.M(
        atom=[(element, (0.0, 0.0, 0.0))],
        unit="Angstrom",
        basis="lanl2dz",
        ecp="lanl2dz",
        spin=SPIN[element],
        verbose=0,
    )


def filtered(mol, element, keep):
    """只保留 keep 里列出的通道块（-1 = 局域 UL）。"""
    nelec, blocks = mol._ecp[element]
    new = [[l, [list(order) for order in orders]] for l, orders in blocks if l in keep]
    assert len(new) == len(keep), (element, keep)
    return build(element, [nelec, new])


def shells_by_l(mol):
    """`{l: [该 l 的全部 AO 下标]}`（球谐；`ao_loc_nr` 的两个相邻值之差就是该壳的函数数）。"""
    loc = [int(x) for x in mol.ao_loc_nr()]
    groups = {}
    for i in range(mol.nbas):
        l = int(mol._bas[i, 1])
        groups.setdefault(l, []).extend(range(loc[i], loc[i + 1]))
    return groups


def rows():
    out = []
    for element in ("Zn", "Cu"):
        mol = reference(element)
        total = mol.intor("ECPscalar")
        local = filtered(mol, element, [-1]).intor("ECPscalar")
        channels = {b[0] for b in mol._ecp[element][1] if b[0] >= 0}
        per = {l: filtered(mol, element, [-1, l]).intor("ECPscalar") - local for l in channels}
        reconstruction = local + sum(per.values())
        print(
            f"{element}: V_local + Σ_l V_l P_l 与 ecp='lanl2dz' 的 max|diff| = "
            f"{np.abs(reconstruction - total).max():.3e}（矩阵尺度 {np.abs(total).max():.3e}）"
        )
        matrices = {"total": total, "local": local}
        for l in sorted(channels):
            matrices[f"delta_l{l}"] = per[l]
        groups = shells_by_l(mol)
        for channel, matrix in matrices.items():
            for l in sorted(groups):
                functions = groups[l]
                for i in functions:
                    for j in functions:
                        out.append(
                            {
                                "element": element,
                                "channel": channel,
                                "ao_i": i,
                                "ao_j": j,
                                "value_hartree": f"{matrix[i, j]:.17e}",
                            }
                        )
    return out


def stamped(rows_):
    out = []
    for row in rows_:
        row = dict(row)
        row["source_doi"] = PYSCF_DOI
        row["source_url"] = PYSCF_URL
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = stamped(rows())
    if args.write:
        write_csv(OUTPUT, FIELDS, data)
        print(f"wrote {rel(OUTPUT)} ({len(data)} rows)")
        return 0
    if args.check:
        return check_csv(OUTPUT, FIELDS, data)
    print(f"{len(data)} rows; use --write or --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
