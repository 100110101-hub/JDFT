"""RS-04 / DA-08 的 NMR 屏蔽参考（PySCF 黑盒），含几何。

sigma_dia(A) = 1/(3c^2) * sum_uv P_uv <u| 1/|r - R_A| |v>   [ppm]
sigma_p(A)   = 「两个独立虚扰动」的有限场二阶导（4 点模板），约定同 RS-04：
               W = (1/2) L_O、V = L_A/|r-R_A|^3，核中心共原规范。
               两个扰动必须**各自缩放**：把两者绑到同一个 lambda 会让能量变成
               E0 + lam^2 q(W ± V)，两半精确抵消（真踩过，见 docs/choices.md 八十一）。

`int1e_rinv` 的原点在 (0,0,0)，所以逐核把分子平移到「该核在原点」再算 —— 基函数顺序不变，
同一个密度矩阵可以直接缩并。这一步与 Rust 侧完全独立（Rust 用自己的 Gaussian 递推 +
核吸引的 unit-charge 技巧）。

**为什么要多写一份几何文件**：NMR 参考值只有在「同一个几何」下才有意义，而参考值要同时被
Python 生成器和 Rust 测试使用。把几何硬编码两遍就会漂。所以几何只在**本脚本**定义一次，
另写 `nmr_molecules.csv`（system,atom_index,symbol,x_angstrom,y_angstrom,z_angstrom），
Rust 测试从那里读 —— 单一真相源。

体系（DA-08 点名的三个）：H2O / CH4 / CH3OH，全部 STO-3G。选 STO-3G 不是图省事：RS-04 的
dia 参考一直是 STO-3G，para 用同一基组才能拼成同一行；cc-pVTZ 的 12 次复 SCF/核 换不来信息。
甲醇几何：C-O 1.4246 Å、C-O-H 108.5°、O-H 0.9638 Å、C-H 1.0936 Å、三个甲基 H 走四面体角。
这是**本项目的约定几何**（不是抄某个实验值），所以几何本身也进了 CSV，可由脚本重建。

另外**保留 RS-04 的 cc-pVTZ 水锚点**：顺磁项的第一份验收基准就是 cc-pVTZ 水
（`docs/status-v2.yaml` 的 RS-04 条目写着 O 416.143947 / −106.338538 / 309.805409），
DA-08 不能把已经用过的参考值弄丢，所以它继续留在 `paramagnetic.csv` 里（只进这一个文件：
dia 家族与几何文件都是 STO-3G 三体系）。

用法：
    python3 scripts/pyscf_nmr_shieldings_reference.py [--check]

输出（相对仓库根）：
    qwb-core/crates/qwb-props/tests/reference/nmr_molecules.csv     （几何）
    qwb-core/crates/qwb-props/tests/reference/nmr_diamagnetic.csv   （dia，RS-04 测试读它）
    qwb-core/crates/qwb-response/tests/reference/paramagnetic.csv   （dia/para/total）
"""
import argparse
import os
import sys

import numpy as np
from pyscf import gto, scf

import reference_sources

C = 137.035999084
BASIS = "sto-3g"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPS_REF = os.path.join("qwb-core", "crates", "qwb-props", "tests", "reference")
RESPONSE_REF = os.path.join("qwb-core", "crates", "qwb-response", "tests", "reference")
GEOMETRY_OUT = os.path.join(ROOT, PROPS_REF, "nmr_molecules.csv")
DIA_OUT = os.path.join(ROOT, PROPS_REF, "nmr_diamagnetic.csv")
PARA_OUT = os.path.join(ROOT, RESPONSE_REF, "paramagnetic.csv")

TETRA = 0.6290
CH_BOND = 1.0936
METHANOL_TETRA = CH_BOND / np.sqrt(3.0)

SYSTEMS = {
    "water": [
        ("O", [0.0, 0.0, 0.117790]),
        ("H", [0.0, 0.755450, -0.471160]),
        ("H", [0.0, -0.755450, -0.471160]),
    ],
    "methane": [
        ("C", [0.0, 0.0, 0.0]),
        ("H", [TETRA, TETRA, TETRA]),
        ("H", [-TETRA, -TETRA, TETRA]),
        ("H", [-TETRA, TETRA, -TETRA]),
        ("H", [TETRA, -TETRA, -TETRA]),
    ],
    "methanol": [
        ("C", [0.0, 0.0, 0.0]),
        ("O", [0.0, 0.0, 1.4246]),
        ("H", [0.0, 0.9638 * np.sin(np.radians(108.5)),
               1.4246 + 0.9638 * np.cos(np.radians(108.5))]),
        ("H", [METHANOL_TETRA, METHANOL_TETRA, -METHANOL_TETRA]),
        ("H", [-METHANOL_TETRA, -METHANOL_TETRA, -METHANOL_TETRA]),
        ("H", [-METHANOL_TETRA, METHANOL_TETRA, METHANOL_TETRA]),
    ],
}

GEOMETRY_FIELDS = ("system", "atom_index", "symbol", "x_angstrom", "y_angstrom", "z_angstrom")
DIA_FIELDS = ("system", "basis", "atom_index", "symbol", "inverse_r_bohr", "sigma_dia_ppm")
PARA_FIELDS = (
    "system",
    "basis",
    "nucleus_index",
    "symbol",
    "sigma_dia_ppm",
    "sigma_p_ppm",
    "sigma_total_ppm",
    "lambda",
    "mu",
)


def shielding(label, basis, atoms, geometry_rows, dia_rows, para_rows, lam=0.02, mu=0.02):
    """算一个体系的逐核 dia/para，往提供的行列表里追加（None = 该文件不要这些行）。"""
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    solver = scf.RHF(mol)
    solver.verbose = 0
    solver.run(conv_tol=1e-11)
    density = solver.make_rdm1()
    if density.ndim == 3:
        density = density.sum(axis=0)
    coords = mol.atom_coords(unit="Bohr")
    symbols = [atom[0] for atom in atoms]
    print(f"{label}/{basis}: nao = {mol.nao}, {len(atoms)} nuclei")
    for index, symbol in enumerate(symbols):
        if geometry_rows is not None:
            geometry_rows.append(
                dict(
                    system=label,
                    atom_index=index,
                    symbol=symbol,
                    x_angstrom=f"{atoms[index][1][0]:.6f}",
                    y_angstrom=f"{atoms[index][1][1]:.6f}",
                    z_angstrom=f"{atoms[index][1][2]:.6f}",
                )
            )
        ra = coords[index]
        shifted = [[s, list(c - ra)] for s, c in zip(symbols, coords)]
        mol2 = gto.M(atom=shifted, basis=basis, unit="Bohr", verbose=0)
        expectation = float(np.einsum("ij,ij->", density, mol2.intor("int1e_rinv")))
        sigma_dia = expectation / (3.0 * C * C) * 1e6
        sigma_p, lam_used, mu_used = paramagnetic(mol, basis, coords, symbols, index, lam, mu)
        print(
            f"  {symbol}{index}: dia {sigma_dia:9.3f}"
            f"  para {sigma_p:9.3f}  total {sigma_dia + sigma_p:9.3f} ppm"
        )
        if dia_rows is not None:
            dia_rows.append(
                dict(
                    system=label,
                    basis=basis,
                    atom_index=index,
                    symbol=symbol,
                    inverse_r_bohr=f"{expectation:.12f}",
                    sigma_dia_ppm=f"{sigma_dia:.9f}",
                )
            )
        row = dict(
            system=label,
            basis=basis,
            nucleus_index=index,
            symbol=symbol,
            sigma_dia_ppm=f"{sigma_dia:.6f}",
            sigma_p_ppm=f"{sigma_p:.6f}",
            sigma_total_ppm=f"{sigma_dia + sigma_p:.6f}",
            mu=mu_used,
        )
        row["lambda"] = lam_used
        para_rows.append(row)


def paramagnetic(mol, basis, coords, symbols, index, lam, mu):
    """核 `index` 的各向同性顺磁屏蔽（两个独立虚扰动的 4 点有限差分）。"""
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    def energy(perturbations, scales):
        hb = h0.copy()
        for operator, scale in zip(perturbations, scales):
            hb = hb + 1j * scale * operator
        zeros = np.zeros_like(hb)
        solver = scf.GHF(mol)
        solver.get_hcore = lambda *args: np.block([[hb, zeros], [zeros, hb]])
        solver.verbose = 0
        solver.conv_tol = 1e-11
        solver.max_cycle = 200
        solver.run()
        return solver.e_tot

    ra = coords[index]
    shifted = [[s, list(c - ra)] for s, c in zip(symbols, coords)]
    m2 = gto.M(atom=shifted, basis=basis, unit="Bohr", verbose=0)
    irp = m2.intor("int1e_irp").reshape(3, 3, mol.nao, mol.nao)
    hyper_all = -m2.intor("int1e_prinvxp")
    isotropic = 0.0
    for component in range(3):
        other = [(component + 1) % 3, (component + 2) % 3]
        ang = 0.5 * (irp[other[0], other[1]] - irp[other[1], other[0]])
        hyper = hyper_all[component]
        isotropic += (
            energy((ang, hyper), (lam, mu))
            - energy((ang, hyper), (lam, -mu))
            - energy((ang, hyper), (-lam, mu))
            + energy((ang, hyper), (-lam, -mu))
        ) / (4.0 * lam * mu)
    return -isotropic / 3.0 / (C * C) * 1e6, lam, mu


def reference_rows(check_only: bool) -> int:
    geometry_rows = []
    dia_rows = []
    para_rows = []
    for label, atoms in SYSTEMS.items():
        shielding(label, BASIS, atoms, geometry_rows, dia_rows, para_rows)
    # RS-04 的 cc-pVTZ 水锚点：只进 paramagnetic.csv（DA-08 家族仍以 STO-3G 三体系为准）。
    shielding("water", "cc-pvtz", SYSTEMS["water"], None, None, para_rows)

    geometry_rows = reference_sources.stamp(geometry_rows)
    dia_rows = reference_sources.stamp(dia_rows)
    para_rows = reference_sources.stamp(para_rows)
    status = reference_sources.check_csv(GEOMETRY_OUT, GEOMETRY_FIELDS, geometry_rows)
    status |= reference_sources.check_csv(DIA_OUT, DIA_FIELDS, dia_rows)
    status |= reference_sources.check_csv(PARA_OUT, PARA_FIELDS, para_rows)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(GEOMETRY_OUT, GEOMETRY_FIELDS, geometry_rows)
    reference_sources.write_csv(DIA_OUT, DIA_FIELDS, dia_rows)
    reference_sources.write_csv(PARA_OUT, PARA_FIELDS, para_rows)
    for path, count in (
        (GEOMETRY_OUT, len(geometry_rows)),
        (DIA_OUT, len(dia_rows)),
        (PARA_OUT, len(para_rows)),
    ):
        print(f"写入 {path}（{count} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="只重算并比对文件，不写盘（DA-08）"
    )
    arguments = parser.parse_args()
    sys.exit(reference_rows(arguments.check))
