"""RS-04（GIAO dia）：libcint 的 `int1e_giao_a11part` / `int1e_a01gp` 逐元素参考值。

# 为什么要这个文件

GIAO 屏蔽的**抗磁**（dia）项在 PySCF 里是

```python
h11 = mol.intor('int1e_giao_a11part', comp=9).reshape(3,3,nao,nao)
e11 = numpy.einsum('xyij,ij->xy', h11, dm0)
e11 = e11 - numpy.eye(3) * e11.trace()
h11 = mol.intor('int1e_a01gp', comp=9).reshape(3,3,nao,nao)
e11 = e11 + numpy.einsum('xyij,ij->xy', h11, dm0)
```

（转写自 `pyscf/prop/nmr/rhf.py` v1.7.6；本环境的 PySCF 2.14.0 没有 `pyscf.prop`，见
`docs/choices.md` 八十一。）本文件把这两个 9 分量张量积分从 libcint 逐元素导出，供
`qwb-integrals` 的 `giao_dia` 模块对照。

# 两个算子的定义（本文件同时是**测量**记录）

两者都由「水 + 三个核 + 9 个 (α,β) 块」的全矩阵拟合确认，不是抄公式：

1. `int1e_giao_a11part[α,β] = −½ ⟨μ| (r−R_A)_α (r−R_ν)_β / |r−R_A|³ |ν⟩`
   （`R_ν` 是 **ket** 壳层中心；`max|A_αβ − A_βα| = 1.785e-1`，所以它**不是** α↔β 对称的）。
2. `int1e_a01gp[α,β] = −½ ⟨μ| [(R_ν−R_μ) × r]_α · ((r−R_A) × ∇)_β / |r−R_A|³ |ν⟩`
   —— 绝对电子坐标 `r` 的「相位导数 × 超精细算符」。两个可测后果（水/STO-3G 实测）：
   * 对整体平移**不**不变（把分子平移到核在原点，`max|Δa01gp| = 4.147e-2`，
     而 `max|a01gp| = 1.806e-1`；同一测量里 `a11part` 的差只有 2.3e-16）；
   * 同一中心上的矩阵元为零（`(R_ν−R_μ) = 0`）。

两块都是 `with_rinv_origin(R_A)` 的约定（frame A），也就是 PySCF 自己 `giao_dia` 的用法。

# 球谐 d 的 AO 顺序（本文件**不用**，因为下面两个基组都只有 s/p；记在这里是因为它决定了
# 「能不能直接拿 def2-SVP 的 libcint 矩阵逐元素对比」）
#
# 本 crate 的球谐 d 顺序（`qwb-core/crates/qwb-basis/src/spherical.rs::d_transform`）：
#
#     [ (x²−y²)·√3/2, (2z²−x²−y²)/2, xy, xz, yz ]
#
# libcint 的是 `[ xy, yz, z², xz, x²−y² ]`。两者是同一组实球谐的**纯置换**（`[4,2,0,3,1]`，
# 无相位差；用重叠矩阵的「d 行 × 干净 s/p 列」指纹逐行配对，残差 5.6e-17…1.1e-16）。
# def2-SVP 上这两个算子按该置换重排后与本 crate 的实现差 8.7e-15…1.8e-14（本轮实测）；
# 但把 def2-SVP 的 24 个 AO 也写进 CSV 会让这个参考文件从 1.2 MB 涨到 3.5 MB，所以「更大的
# 那个基组」选的是 LANL2DZ（13 AO，双 ζ + 两个 p 壳层）。LANL2DZ 对 O 的 ECP 与这些
# **一电子**算子无关：这里只用到它的基函数。
#
# # 用法

    python3 scripts/pyscf_giao_dia_reference.py [--check]

输出：`qwb-core/crates/qwb-integrals/tests/reference/giao_dia.csv`
"""

import argparse
import io
import os
import re
import sys

# 参考值要逐字节可复现：把 BLAS 线程钉成 1（与其它 pyscf_*_reference.py 同一约定）。
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy
from pyscf import gto

import reference_sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-integrals", "tests", "reference", "giao_dia.csv"
)
BOHR = 1.889726089838572

#: 与 Rust 测试共用的几何（Å；单一真相源就在本文件）。
SYSTEM = "water"
GEOMETRY = [
    ("O", [0.0, 0.0, 0.117790]),
    ("H", [0.0, 0.755450, -0.471160]),
    ("H", [0.0, -0.755450, -0.471160]),
]

#: 球谐 d 的 AO 顺序（本文件**不用**，因为下面两个基组都只有 s/p；记在这里是因为它决定了
#: 「能不能直接拿 def2-SVP 的 libcint 矩阵逐元素对比」）：
#:
#: ```text
#: libcint:  [ xy, yz, z², xz, x²−y² ]
#: 本 crate: [ (x²−y²)·√3/2, (2z²−x²−y²)/2, xy, xz, yz ]   (qwb-basis/src/spherical.rs)
#: ```
#:
#: 两者是同一组实球谐的**纯置换**（`[4,2,0,3,1]`，无相位差；用重叠矩阵的「d 行 × 干净
#: s/p 列」指纹逐行配对，残差 5.6e-17…1.1e-16）。def2-SVP 上这两个算子按该置换重排后与
#: 本 crate 的实现差 8.7e-15…1.8e-14（本轮实测）；但把 def2-SVP 的 24 个 AO 也写进 CSV
#: 会让这个参考文件从 1.2 MB 涨到 3.5 MB，所以更大的那个基组选的是 LANL2DZ（13 AO，
#: 双 ζ + 两个 p 壳层）。LANL2DZ 对 O 的 ECP 与这些**一电子**算子无关：这里只用到它的
#: 基函数。
BASES = ("sto-3g", "lanl2dz")

FIELDS = (
    "system",
    "basis",
    "nucleus_index",
    "operator",
    "alpha",
    "beta",
    "row",
    "column",
    "value",
)

OPERATORS = ("a11part", "a01gp")


def load_basis(name: str) -> dict:
    """从 `qwb-data/basis/<name>.yaml` 读指数/系数（PySCF 内建表精度不够）。"""
    path = os.path.join(ROOT, "qwb-data", "basis", name + ".yaml")
    basis = {}
    element = None
    angular = None
    exponents = None
    for raw in io.open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        match = re.match(r"  - element: (\w+)$", line)
        if match:
            element = match.group(1)
            basis[element] = []
            continue
        match = re.match(r"      - l: (\d+)$", line)
        if match:
            angular = int(match.group(1))
            exponents = None
            continue
        match = re.match(r"        exponents: \[(.*)\]$", line)
        if match:
            exponents = [float(v) for v in match.group(1).split(",")]
            continue
        match = re.match(r"          - \[(.*)\]$", line)
        if match and exponents is not None and angular is not None:
            coefficients = [float(v) for v in match.group(1).split(",")]
            basis[element].append(
                [angular] + [[e, c] for e, c in zip(exponents, coefficients)]
            )
    return basis


def make_mol(name: str):
    mol = gto.M(
        atom=[[symbol, numpy.array(xyz) * BOHR] for symbol, xyz in GEOMETRY],
        basis=load_basis(name),
        unit="Bohr",
        verbose=0,
    )
    mol.cart = False
    return mol


def ao_permutation(mol) -> list:
    """本 crate 的 AO 下标 → libcint 的 AO 下标。

    两个选定的基组都只有 s/p 壳层，所以这是恒等映射；`l = 2` 的分支保留下来是因为它把
    「本 crate 与 libcint 的球谐 d 顺序不同」这件事写成了可执行的断言（见上面的注释）。
    """
    permutation = list(range(mol.nao))
    first = 0
    for shell in range(mol.nbas):
        angular = mol.bas_angular(shell)
        count = 2 * angular + 1
        if angular == 2:
            for local, source in enumerate((4, 2, 0, 3, 1)):
                permutation[first + local] = first + source
        first += count
    assert first == mol.nao, (first, mol.nao)
    return permutation


def rows() -> list:
    out = []
    for name in BASES:
        mol = make_mol(name)
        permutation = ao_permutation(mol)
        coords = numpy.array([xyz for _, xyz in GEOMETRY]) * BOHR
        print(f"{SYSTEM}/{name}: nao = {mol.nao}")
        for nucleus in range(mol.natm):
            with mol.with_rinv_origin(coords[nucleus]):
                a11 = mol.intor("int1e_giao_a11part", comp=9).reshape(
                    3, 3, mol.nao, mol.nao
                )
                a01 = mol.intor("int1e_a01gp", comp=9).reshape(3, 3, mol.nao, mol.nao)
            asymmetry = numpy.abs(a11 - numpy.transpose(a11, (1, 0, 2, 3))).max()
            print(
                f"  nucleus {nucleus} ({mol.atom_symbol(nucleus)}): "
                f"max|a11| = {numpy.abs(a11).max():.9f}, "
                f"max|a01| = {numpy.abs(a01).max():.9f}, "
                f"max|a11_ab - a11_ba| = {asymmetry:.9f}"
            )
            for operator, tensor in (("a11part", a11), ("a01gp", a01)):
                for alpha in range(3):
                    for beta in range(3):
                        for row in range(mol.nao):
                            for column in range(mol.nao):
                                out.append(
                                    dict(
                                        system=SYSTEM,
                                        basis=name,
                                        nucleus_index=nucleus,
                                        operator=operator,
                                        alpha=alpha,
                                        beta=beta,
                                        row=row,
                                        column=column,
                                        value=f"{tensor[alpha, beta, permutation[row], permutation[column]]:.16e}",
                                    )
                                )
    return out


def main(check_only: bool) -> int:
    stamped = reference_sources.stamp(rows())
    status = reference_sources.check_csv(OUT, FIELDS, stamped)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(OUT, FIELDS, stamped)
    print(f"写入 {reference_sources.rel(OUT)}（{len(stamped)} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只重算并比对文件，不写盘")
    arguments = parser.parse_args()
    sys.exit(main(arguments.check))
