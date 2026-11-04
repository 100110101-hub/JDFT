#!/usr/bin/env python3
"""RP-02 的 PySCF oracle：HCN 的四条 bisect 轴 + ZnO/LANL2DZ 的位能曲线。

这是**测量**脚本（不是 `--check` 生成器）：它只打印 JSON/表，用来给四个轴帧和 ZnO 的
三行提供独立标尺。WSL 里跑（PySCF 2.14.0）：

    python3 scripts/rp02_oracle.py --part hcn --json > /tmp/rp02_hcn.json
    python3 scripts/rp02_oracle.py --part zno --json > /tmp/rp02_zno.json

几何与 `qwb-core/crates/qwb-scf/tests/rp02_hcn_pbe_fallback.rs` 一致：C 在原点、
N 沿 +z 1.17 Å、H 在 x–z 平面里 1.19 Å 处，`angle` 是 ∠H–C–N（180° 是直线对照）。
ZnO 的几何与 `qwb-core/crates/qwb-engine/tests/reference/zno_lanl2dz_pbe_ecp.csv` 一致：
Zn 在原点、O 沿 +z。LANL2DZ 只在 Zn 上带 ECP（18 芯，z_eff = 12），O 是全电子。

本项目的泛函 id 与 PySCF/libxc 名字的对应（`qwb-data/functional/*.yaml`）：

| 本项目 | 组成 | PySCF |
| --- | --- | --- |
| `pbe` | PBE 交换 + PBE 相关 | `PBE` |
| `b88` | **只有** B88 交换（无相关） | `B88` |
| `svwn5` | Slater α=2/3 交换 + VWN 相关 | `lda_x,lda_c_vwn_rpa` |
| `hf` | 无 XC 项 | `scf.RHF` |

`svwn5` 那一行是**测出来**的，不是猜的：本项目的 `vwn5_*` 参数（`x0 = -0.409286,
b = 13.072, c = 42.7198`）在 libxc 里对应 `LDA_C_VWN_RPA`，而不是 `LDA_C_VWN`
（VWN5 的 `x0 = -0.10498, b = 3.72744, c = 12.9352`）。实测 H2O/STO-3G：

```text
本项目 svwn5                      E = -74.931053570619
pyscf lda_x,lda_c_vwn   (VWN5)    E = -74.734847398054
pyscf lda_x,lda_c_vwn_rpa         E = -74.931056429697   ← 差 2.9e-6（网格）
```

所以拿 PySCF 的 `SVWN` 当本项目的 `svwn5` 会凭空造出 0.2 Ha 的"偏差"。
（这是 RP-02 之外的既有命名问题，本脚本只负责把标尺对准，不去改数据文件。）
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def hcn(angle_deg: float):
    a = math.radians(angle_deg)
    return [
        ("C", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, 1.17)),
        ("H", (1.19 * math.sin(a), 0.0, 1.19 * math.cos(a))),
    ]


def zno(distance_angstrom: float):
    return [("Zn", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, distance_angstrom))]


def run(atom, xc: str, basis: str, grid_level: int, conv_tol: float = 1.0e-11):
    from pyscf import dft, gto, scf

    ecp = "lanl2dz" if basis == "lanl2dz" else None
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        ecp=ecp,
        charge=0,
        spin=0,
        verbose=0,
    )
    method = scf.RHF(mol) if xc == "hf" else dft.RKS(mol)
    if xc != "hf":
        method.xc = xc
        method.grids.level = grid_level
    method.conv_tol = conv_tol
    method.conv_tol_grad = min(1.0e-7, conv_tol**0.5)
    method.max_cycle = 500
    energy = method.kernel()
    if not method.converged:
        # LANL2DZ 的 ZnO 在纯 DIIS 下会停住（本项目的缺陷也是这一类"停住"），
        # 用二阶方法把同一个解收敛出来：这只影响**怎么到解**，不影响解本身。
        solver = scf.RHF(mol) if xc == "hf" else dft.RKS(mol)
        if xc != "hf":
            solver.xc = xc
            solver.grids.level = grid_level
        solver.conv_tol = conv_tol
        solver.max_cycle = 500
        energy = solver.newton().kernel()
        method = solver
    summary = {k: float(v) for k, v in method.scf_summary.items()}
    return {
        "xc": xc,
        "basis": basis,
        "grids_level": grid_level,
        "nao": int(mol.nao),
        "nelectron": int(mol.nelectron),
        "energy_hartree": float(energy),
        "converged": bool(method.converged),
        "e_nuc": float(method.energy_nuc()),
        "e_xc": summary.get("exc", 0.0),
        "e_coul": summary["coul"],
    }


def hcn_rows():
    rows = []
    for angle in (90.0, 110.0, 135.0, 160.0, 180.0):
        rows.append(
            dict(axis="geometry", angle_deg=angle, **run(hcn(angle), "pbe", "def2-svp", 3))
        )
    for xc in ("B88", "lda_x,lda_c_vwn_rpa", "hf"):
        rows.append(dict(axis="functional", angle_deg=90.0, **run(hcn(90.0), xc, "def2-svp", 3)))
    for basis in ("sto-3g", "6-311g(d,p)", "def2-tzvp"):
        rows.append(dict(axis="basis", angle_deg=90.0, **run(hcn(90.0), "pbe", basis, 3)))
    for level in (1, 2, 3):
        rows.append(dict(axis="grid", angle_deg=90.0, **run(hcn(90.0), "pbe", "def2-svp", level)))
    return rows


def zno_rows():
    rows = []
    for distance in (1.60, 1.65, 1.70, 1.75, 1.78, 1.80, 1.82, 1.85, 1.88, 1.90, 1.95):
        rows.append(dict(r_angstrom=distance, **run(zno(distance), "PBE", "lanl2dz", 3)))
    # 细扫：确定离散最小值的位置（"1.85 Å 是平衡点" 是伪解时代的说法，见 CSV 的 note）。
    for distance in (1.74, 1.76, 1.77, 1.79, 1.81):
        rows.append(dict(r_angstrom=distance, **run(zno(distance), "PBE", "lanl2dz", 3)))
    for level in (1, 2):
        rows.append(dict(r_angstrom=1.70, **run(zno(1.70), "PBE", "lanl2dz", level)))
        rows.append(dict(r_angstrom=1.85, **run(zno(1.85), "PBE", "lanl2dz", level)))
    return rows


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("hcn", "zno"), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from pyscf import __version__

    rows = hcn_rows() if args.part == "hcn" else zno_rows()
    payload = {"code": "pyscf", "version": __version__, "part": args.part, "rows": rows}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for row in rows:
            label = row.get("angle_deg", row.get("r_angstrom"))
            axis = row.get("axis", "scan")
            print(
                f"{axis:>10s} {label:7.2f} {row['xc']:>7s}/{row['basis']:<10s} "
                f"L{row['grids_level']} E = {row['energy_hartree']:+.10f}  "
                f"conv={row['converged']}  nao={row['nao']}  ne={row['nelectron']}"
            )
    return 0 if all(row["converged"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
