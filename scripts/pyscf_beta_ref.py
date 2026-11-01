"""ST-02 的外部参考：水的静态一阶超极化率 beta（有限场 + CPHF，PySCF 2.14）。

做法与 Rust 侧同一条路线，但用 PySCF 自己的 SCF 与 CPHF：

    H(F)     = H0 - F*r_k          （mf.get_hcore 覆盖）
    alpha(F) = 4 dT (A+B)^-1 d     （mf.gen_response + cphf.solve；本环境的 PySCF
                                    没有 pyscf.prop，所以不能用 mf.polarizability()）
    beta_ijk = (4*D(F/2) - D(F))/3, D(F) = [alpha(F) - alpha(-F)]/(2F)

为什么参考也用有限场：Orr-Ward 的态求和要激发态间的偶极矩阵元，本轮没做；而有限场是两边
都能独立跑同一条定义。方法匹配的参考能把「实现对不对」与「方法好不好」分开。

输出：qwb-core/crates/qwb-response/tests/reference/beta.csv
列：species,basis,component,beta_au,note,source_doi,source_url
"""

import argparse
import os
import sys

import numpy as np
import reference_sources
import yaml
from pyscf import gto, scf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-response", "tests", "reference", "beta.csv"
)

FIELDS = ("species", "basis", "component", "beta_au", "note")

WATER = """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
"""

COARSE = 2.5e-3


def load_yaml_basis(name):
    with open(f"qwb-data/basis/{name}.yaml", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    out = {}
    for entry in data["shells"]:
        shells = []
        for shell in entry["shells"]:
            l = shell["l"]
            exponents = np.array(shell["exponents"], dtype=float)
            for row in np.array(shell["coefficients"], dtype=float):
                shells.append([l] + [[float(e), float(c)] for e, c in zip(exponents, row)])
        out[entry["element"]] = shells
    return out


def dipole_in_field(mol, dip, h0, field):
    """`field` 是长度 3 的向量（a.u.）：返回该场下的偶极矩 `mu_i = Tr(P d_i)`。"""
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.get_hcore = lambda *args: h0 - sum(field[k] * dip[k] for k in range(3))
    mf.kernel()
    density = mf.make_rdm1()
    return np.asarray([float(np.einsum("ij,ji->", density, dip[k])) for k in range(3)])


def alpha_in_field(mol, dip, h0, axis, field, inner=1.0e-3):
    """`alpha_ij(F_k)`：在场沿 `axis` 时，对**另一个**方向的场做中心差分。

    α 的指标 (i, j)：i 是偶极分量、j 是内层差分方向；外层 `axis` 是 β 的场方向 k。
    """
    alpha = np.zeros((3, 3))
    for j in range(3):
        plus = [0.0, 0.0, 0.0]
        minus = [0.0, 0.0, 0.0]
        plus[axis] = field
        minus[axis] = field
        plus[j] += inner
        minus[j] -= inner
        mu_plus = dipole_in_field(mol, dip, h0, plus)
        mu_minus = dipole_in_field(mol, dip, h0, minus)
        alpha[:, j] = (mu_plus - mu_minus) / (2.0 * inner)
    return alpha


def beta_tensor(mol, dip, h0, coarse):
    per_axis = {}
    for axis in range(3):
        plus_fine = alpha_in_field(mol, dip, h0, axis, coarse / 2.0)
        minus_fine = alpha_in_field(mol, dip, h0, axis, -coarse / 2.0)
        plus_coarse = alpha_in_field(mol, dip, h0, axis, coarse)
        minus_coarse = alpha_in_field(mol, dip, h0, axis, -coarse)
        d_fine = (plus_fine - minus_fine) / coarse
        d_coarse = (plus_coarse - minus_coarse) / (2.0 * coarse)
        per_axis[axis] = (4.0 * d_fine - d_coarse) / 3.0
    tensor = np.zeros((3, 3, 3))
    for axis, value in per_axis.items():
        # β_ijk：i 偶极、j 内层、k 外层
        tensor[:, :, axis] = 0.5 * (value + value.T)
    beta_i = np.array(
        [
            sum(tensor[i, j, j] + tensor[j, i, j] + tensor[j, j, i] for j in range(3))
            for i in range(3)
        ]
    )
    beta_par = 0.6 * sum(tensor[2, i, i] + tensor[i, 2, i] + tensor[i, i, 2] for i in range(3))
    return tensor, float(np.linalg.norm(beta_i)), float(beta_par)


def run(species, geometry, basis_name):
    mol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis(basis_name)
    mol.cart = False
    mol.build()
    # int1e_r 的分量形状是 (3, nao, nao)：一次取完，再按轴索引。
    dip = np.asarray(mol.intor("int1e_r"))
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    tensor, beta_vec, beta_par = beta_tensor(mol, dip, h0, COARSE)
    print(
        f"{species}/{basis_name}: beta_zzz = {tensor[2, 2, 2]:.6f}, "
        f"beta_zxx = {tensor[2, 0, 0]:.6f}, beta_vec = {beta_vec:.6f}, beta_par = {beta_par:.6f}"
    )
    rows = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rows.append(
                    dict(
                        species=species,
                        basis=basis_name,
                        component=f"tensor_{i}{j}{k}",
                        beta_au=f"{tensor[i][j][k]:.10f}",
                        note="PySCF 2.14.0 RHF + get_hcore 有限场 + CPHF(gen_response) + Richardson",
                    )
                )
    rows.append(
        dict(
            species=species,
            basis=basis_name,
            component="beta_vec",
            beta_au=f"{beta_vec:.10f}",
            note="beta_vec = sqrt(sum_i (sum_j (beta_ijj+beta_jij+beta_jji))^2)",
        )
    )
    rows.append(
        dict(
            species=species,
            basis=basis_name,
            component="beta_parallel",
            beta_au=f"{beta_par:.10f}",
            note="beta_parallel = (3/5) sum_i (beta_zii + beta_izi + beta_iiz), z = dipole axis",
        )
    )
    return rows


def main(check_only: bool = False):
    rows = run("H2O", WATER, "sto-3g")
    rows = reference_sources.stamp(rows)
    fields = list(FIELDS)
    status = reference_sources.check_csv(CSV_PATH, fields, rows)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(CSV_PATH, fields, rows)
    print(f"written {CSV_PATH} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare only (no write)")
    arguments = parser.parse_args()
    sys.exit(main(arguments.check))
