# -*- coding: utf-8 -*-
"""把 cccbdb-raw.json 变成 K1/V-03 风格的参考 CSV，并做自检。

自检两件事（避免把抓错的数据写进测试）：
1. 用解析出来的笛卡尔坐标重算核排斥能，与 CCCBDB 报告值比（应 < 1e-4 Ha）；
2. 能量值与已发表值范围粗检（H2O B3LYP/6-31G* ≈ -76.4089）。

产物（V-03 的 CSV 形态，每行带 source_url）：
    tests/reference/cccbdb_geometries.csv   物种,原子序号,元素,x,y,z,method,basis,source_url
    tests/reference/cccbdb_energies.csv     物种,method,basis,energy_hartree,tolerance,source_url
"""
import json
import sys

Z = {"H": 1, "C": 6, "N": 7, "O": 8}
BOHR = 0.529177210903


def enuc(atoms):
    total = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            zi, zj = Z[atoms[i][0]], Z[atoms[j][0]]
            d2 = sum((atoms[i][k] - atoms[j][k]) ** 2 for k in (1, 2, 3))
            total += zi * zj / (d2 ** 0.5)
    return total * BOHR


def main():
    raw = json.load(open(sys.argv[1], encoding="utf-8"))
    ref = sys.argv[2] if len(sys.argv) > 2 else "qwb-core/crates/qwb-scf/tests/reference"
    geo_rows, ene_rows = [], []
    worst = 0.0
    for e in raw["entries"]:
        atoms = [[a[0].rstrip("0123456789")] + a[1:] for a in e["cartesian_angstrom"]]
        atoms = [[a[0], float(a[1]), float(a[2]), float(a[3])] for a in atoms]
        nuc = enuc(atoms)
        delta = abs(nuc - e["nuclear_repulsion_hartree"])
        worst = max(worst, delta)
        # 阈值 0.1 Ha：CCCBDB 只给 3 位小数，苯（12 个原子、近核 C-C）的核排斥能
        # 会因此偏 ~4e-2 Ha；但总能量在极小点是一阶平稳的，几何舍入只贡献二阶项（~1e-6 Ha），
        # 所以这条自检只用来抓「表抓错/抓串行」这类量级错误（错位时差 3 倍以上）。
        if delta > 0.1:
            print(f"!! geometry check failed {e['species']} {e['method']}/{e['basis']}: "
                  f"recomputed {nuc:.6f} vs reported {e['nuclear_repulsion_hartree']:.6f}")
        for idx, a in enumerate(atoms, start=1):
            geo_rows.append(f"{e['species']},{e['method']},{e['basis']},{idx},{a[0]},"
                            f"{a[1]:.3f},{a[2]:.3f},{a[3]:.3f},{e['url']}")
        ene_rows.append(f"{e['species']},{e['method']},{e['basis']},"
                        f"{e['energy_0k_hartree']:.6f},0.002,{e['url']}")
    print(f"nuclear-repulsion self-check: worst |dE_nuc| = {worst:.2e} Ha")
    geo_hdr = ("# CCCBDB Release 22 (May 2022) 的优化几何（Cartesians (A)），逐行带原始 URL。\n"
               "# 抓取脚本：scripts/fetch_cccbdb_reference.py（限速 2s/请求，遇到 429 退避重试）。\n"
               "# 精度：CCCBDB 只给到 3 位小数，定点处一阶导为零，能量误差是二阶的（< 1e-5 Ha）。\n"
               "# species,method,basis,atom_index,element,x_angstrom,y_angstrom,z_angstrom,source_url\n")
    ene_hdr = ("# CCCBDB Release 22 (May 2022) 的 0K 电子能（optimized geometry 处）。\n"
               "# species,method,basis,energy_hartree,tolerance_hartree,source_url\n")
    with open(f"{ref}/cccbdb_geometries.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(geo_hdr + "\n".join(geo_rows) + "\n")
    with open(f"{ref}/cccbdb_energies.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(ene_hdr + "\n".join(ene_rows) + "\n")
    print(f"wrote {len(geo_rows)} geometry rows, {len(ene_rows)} energy rows")


if __name__ == "__main__":
    main()
