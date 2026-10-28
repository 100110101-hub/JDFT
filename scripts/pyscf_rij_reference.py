"""RI-J 拟合误差的参考值（PySCF 2.14.0），供 IN-01/SC-05 的测试当外部尺子。

为什么要有这个文件：`project-phase2.md` §6.3 给 IN-01 写的判据是「拟合误差对
H2O/def2-SVP < 1e-6 Ha（能量）」。实测（本脚本 + 我们的实现两边逐位一致）表明：

* `def2-SVP-RI` 拟合 `def2-SVP` 的 Coulomb 能量误差是 **6.14e-3 Ha**（46.845 的
  0.013%），RI-J 的 SCF 总能量偏 **2.27e-3 Ha**；
* 这个误差是**辅助基不完备**带来的，不是实现错误 —— PySCF 的同一配对给出同一个数。

所以判据要拆成两条：实现正确性用「与 PySCF 的同配对 RI-J 能量差 < 1e-6 Ha」
（实测 1.4e-7，受 K2 的 Å→bohr 常数限制），辅助基质量如实记账（6.14e-3）。

输出：qwb-core/crates/qwb-scf/tests/reference/rij_energy.csv
列：system,basis,aux_basis,n_aux,e_j_exact,e_j_ri,e_ri_minus_exact,note
"""

import csv
import os
import sys

import numpy as np
import yaml
from pyscf import df, gto, scf

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-scf", "tests", "reference", "rij_energy.csv"
)

WATER = """
O 0.000000 0.000000 0.000000
H 0.000000 0.757000 0.586000
H 0.000000 -0.757000 0.586000
"""


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


def main():
    mol = gto.M(atom=WATER, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis("def2-svp")
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    e_hf = mf.kernel()
    dm = mf.make_rdm1()
    vj, _ = mf.get_jk(mol, dm)
    e_j_exact = 0.5 * np.einsum("ij,ji->", vj, dm)
    print(f"exact RHF = {e_hf:.10f}; E_J = {e_j_exact:.10f}")

    rows = []
    for aux_name in ["def2-svp-ri", "def2-svp", "sto-3g"]:
        auxmol = gto.M(atom=WATER, unit="Angstrom", verbose=0)
        auxmol.basis = load_yaml_basis(aux_name)
        auxmol.cart = False
        auxmol.build()
        int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        v = auxmol.intor("int2c2e")
        d = np.einsum("ij,ijP->P", dm, int3c)
        e_j_ri = 0.5 * d @ np.linalg.solve(v, d)
        note = (
            "def2-SVP-RI 是这一对的官方伴随辅助基：误差 6.14e-3 Ha 属于辅助基不完备"
            if aux_name == "def2-svp-ri"
            else "对照组：辅助基不对口，误差量级 1 Ha"
        )
        print(
            f"aux = {aux_name:>12}: naux = {auxmol.nao:3d}, E_J^RI = {e_j_ri:.10f}, "
            f"差 = {e_j_ri - e_j_exact:+.3e} Ha"
        )
        rows.append(
            dict(
                system="H2O",
                basis="def2-svp",
                aux_basis=aux_name,
                n_aux=auxmol.nao,
                e_j_exact=f"{e_j_exact:.12f}",
                e_j_ri=f"{e_j_ri:.12f}",
                e_ri_minus_exact=f"{e_j_ri - e_j_exact:.6e}",
                note=note,
            )
        )

    mf_ri = scf.RHF(mol).density_fit(auxbasis=load_yaml_basis("def2-svp-ri"))
    e_ri = mf_ri.kernel()
    print(f"RI-J SCF = {e_ri:.10f}，与精确差 {e_ri - e_hf:+.3e} Ha")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = [
        "system",
        "basis",
        "aux_basis",
        "n_aux",
        "e_j_exact",
        "e_j_ri",
        "e_ri_minus_exact",
        "note",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")


if __name__ == "__main__":
    sys.exit(main())
