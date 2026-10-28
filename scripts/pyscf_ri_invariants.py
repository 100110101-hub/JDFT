"""用 PySCF 生成 IN-01 的外部参考：RI 积分的不变量。

为什么要用**不变量**而不是逐元素对照：两边的球谐分量排序可能不同（本项目自己的
直角→球谐变换是一套 Gram–Schmidt，libcint 是另一套），逐元素对比会假失败。
下面两个量在"壳层内部做任意正交变换"下都不变：

* `V = (P|Q)` 的**特征值**（`V → U V Uᵀ`）；
* 3 中心 `(ij|P)` 作为 `(n_aux × n_orb²)` 矩阵的**奇异值**
  （辅助侧 `U_P`、轨道侧的 `U_i ⊗ U_j` 都是正交阵）。

输出：qwb-core/crates/qwb-integrals/tests/reference/ri_invariants.csv
列：system,basis,aux_basis,quantity,index,value
"""

import csv
import os
import sys

import numpy as np
from pyscf import gto

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-integrals", "tests", "reference", "ri_invariants.csv"
)

WATER = """
O 0.000000 0.000000 0.000000
H 0.000000 0.757000 0.586000
H 0.000000 -0.757000 0.586000
"""

CASES = [
    ("H2O", "def2-svp", "def2-svp-ri", WATER),
    ("H2O", "sto-3g", "sto-3g", WATER),
]


def load_yaml_basis(name):
    import yaml

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
    rows = []
    for system, basis, aux, geometry in CASES:
        mol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
        mol.basis = load_yaml_basis(basis)
        mol.cart = False
        mol.build()
        auxmol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
        auxmol.basis = load_yaml_basis(aux)
        auxmol.cart = False
        auxmol.build()

        naux, nao = auxmol.nao, mol.nao
        v = auxmol.intor("int2c2e")
        from pyscf import df as pyscf_df

        int3c = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        print(f"{system}/{basis}+{aux}: nao={nao}, naux={naux}, int3c.shape={int3c.shape}")

        eig_v = np.sort(np.linalg.eigvalsh(v))
        # `df.incore.aux_e2` 的布局固定是 (nao, nao, naux)——AO 对在前、辅助指标在后。
        # 所以**先**判这个形状：STO-3G 那种 nao == naux == 7 的三维等长情形下，
        # 先判 (naux, nao, nao) 会把 (j, P) 错当成一对，最小奇异值直接差 5 倍。
        if int3c.shape == (nao, nao, naux):
            b = int3c.reshape(nao * nao, naux).T
        elif int3c.shape == (naux, nao, nao):  # pragma: no cover - 兼容其他布局
            b = int3c.reshape(naux, nao * nao)
        else:  # pragma: no cover - 形状只有这两种可能
            raise SystemExit(f"unexpected int3c shape {int3c.shape}")
        singular = np.sort(np.linalg.svd(b, compute_uv=False))

        for index, value in enumerate(eig_v):
            rows.append(
                dict(
                    system=system,
                    basis=basis,
                    aux_basis=aux,
                    quantity="V_eigenvalue",
                    index=index,
                    value=f"{value:.12e}",
                )
            )
        for index, value in enumerate(singular):
            rows.append(
                dict(
                    system=system,
                    basis=basis,
                    aux_basis=aux,
                    quantity="three_center_singular_value",
                    index=index,
                    value=f"{value:.12e}",
                )
            )
        print(
            f"   V: eig[0]={eig_v[0]:.10e} eig[-1]={eig_v[-1]:.10e}; "
            f"3c: sv[0]={singular[-1]:.10e} sv[-1]={singular[0]:.10e}"
        )

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = ["system", "basis", "aux_basis", "quantity", "index", "value"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")


if __name__ == "__main__":
    sys.exit(main())
