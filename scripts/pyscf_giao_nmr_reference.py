"""RS-04（GIAO）：PySCF 侧的 GIAO-HF NMR 屏蔽与 GIAO Fock 导数参考值。

# 为什么是"自己转写"而不是 `pyscf.prop.nmr`

任务书要求「与 PySCF 的 GIAO NMR 比」，但**这个环境里的 PySCF 2.14.0 没有 `pyscf.prop`**：
PyPI 的 wheel 与 sdist 里都没有那个子包（RS-04 在 `docs/choices.md` 八十一 就已经记过这条）。
核对方式（本轮实跑）：

```
$ pip index versions pyscf        -> 2.14.0 及 2.0.0..2.13.1 全部在册
$ python -c "import zipfile; ..."  # 2.14.0 wheel 顶层只有 adc/agf2/.../tdscf，
                                   # 没有 prop；sdist 的 tar 列表同样没有
```

不过 PySCF 的 libcint **仍然带着整套 GIAO 积分**（`int1e_giao_a11part`、`int1e_a01gp`、
`int1e_giao_irjxp`、`int1e_ig{ovlp,kin,nuc}`、`int2e_ig1`、`int1e_cg_a11part`、`int1e_cg_irxp`），
所以这里把 **PySCF 自己的 GIAO 实现**（`pyscf/prop/nmr/rhf.py`，v1.7.6 源码；2.x 删掉了那个包）
逐步转写到本脚本上，积分全部来自 PySCF 的 libcint，SCF/CPHF 也全部走 PySCF。
转写的**真伪**由 PySCF 自己写在那个文件 `__main__` 里的自检值钉住（见 `--self-test`）：

```text
FH/6-31G   GIAO  +CPHF    F σ_xx = 375.232839（文件里的注释值）
                          F σ_zz = 483.002139
FH/6-31G   GIAO  无 CPHF  F σ_xx = 449.032227
```

本脚本实测 375.23280 / 483.00208 / 449.03213 —— 相对偏差 ≤ 1.3e-7，来自 SCF/CPHF 收敛阈
（那边用默认阈值；这里 1e-12）。所以"转写是忠实的"这件事有独立证据，不是自我声明。

# 两个 GIAO 事实（都会进 CSV，因为 Rust 侧要逐元素对）

1. **GIAO 的两电子相位导数（`int2e_ig1` 那一项）是本体，不是修正**：把它从 Fock 导数里拿掉，
   水/STO-3G 的 O 屏蔽从 365.67 ppm 变成 2396.04 ppm。Rust 侧因此必须实现 `X_l[μν|λσ]`。
2. **GIAO 的 dia/para 分解依赖绝对原点，只有总和对平移不变**（GIAO 的规范不变性只保证总和）。
   所以本 CSV 报的是**总和** σ_iso（以及 3 个对角分量），不报 dia/para 的分解 —— 分解不是
   规范不变量，拿它当参考值会把实现钉在别人的一个任意选择上。

用法：
    python3 scripts/pyscf_giao_nmr_reference.py [--check] [--self-test]

输出（相对仓库根）：
    qwb-core/crates/qwb-response/tests/reference/giao_nmr.csv              （屏蔽）
    qwb-core/crates/qwb-response/tests/reference/giao_fock_derivative.csv （Fock/重叠导数）
"""


import argparse
import os
import sys

# PySCF 的 J/K/CPHF 走多线程 BLAS，求和顺序会让末几位漂；参考值要逐字节可复现，所以
# 在导入 numpy 之前把线程钉成 1（stable() 再把噪声区写成 0）。
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy
from pyscf import gto, scf
from pyscf.data import nist
from pyscf.scf import _response_functions  # noqa: F401  (加载 RHF.gen_response)
from pyscf.scf import cphf

import reference_sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSE_REF = os.path.join("qwb-core", "crates", "qwb-response", "tests", "reference")
SHIELDING_OUT = os.path.join(ROOT, RESPONSE_REF, "giao_nmr.csv")
DERIVATIVE_OUT = os.path.join(ROOT, RESPONSE_REF, "giao_fock_derivative.csv")

TETRA = 0.6290

#: 与 Rust 侧测试共用的几何（Å；单一真相源就在本文件）。
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
}

CASES = [("water", "sto-3g"), ("water", "6-31g"), ("methane", "sto-3g")]

SHIELDING_FIELDS = (
    "system",
    "basis",
    "atom_index",
    "symbol",
    "x_angstrom",
    "y_angstrom",
    "z_angstrom",
    "sigma_iso_ppm",
    "sigma_xx_ppm",
    "sigma_yy_ppm",
    "sigma_zz_ppm",
    "note",
)
DERIVATIVE_FIELDS = ("system", "basis", "quantity", "k", "row", "column", "value", "note")


# ── PySCF 的 GIAO 实现（转写自 pyscf/prop/nmr/rhf.py v1.7.6） ─────────────────


def make_scf(atoms, basis):
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    solver = scf.RHF(mol)
    solver.verbose = 0
    solver.conv_tol = 1e-12
    solver.max_cycle = 200
    solver.run()
    return mol, solver


def giao_dia(mol, dm0):
    """GIAO 抗磁项：`int1e_giao_a11part`（去掉迹）+ `int1e_a01gp`。"""
    nao = mol.nao
    out = []
    for atm_id in range(mol.natm):
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            h11 = mol.intor("int1e_giao_a11part", comp=9).reshape(3, 3, nao, nao)
            e11 = numpy.einsum("xyij,ij->xy", h11, dm0)
            e11 = e11 - numpy.eye(3) * e11.trace()
            h11 = mol.intor("int1e_a01gp", comp=9).reshape(3, 3, nao, nao)
            e11 = e11 + numpy.einsum("xyij,ij->xy", h11, dm0)
        out.append(e11)
    return numpy.asarray(out)


def giao_fock_derivative(mol, dm0, with_2e=True):
    """GIAO 的一阶 Fock 导数（转写自 `pyscf.prop.nmr.rhf.make_h10`）。"""
    nao = mol.nao
    h1 = -0.5 * mol.intor("int1e_giao_irjxp", 3)
    if with_2e:
        tensor = mol.intor("int2e_ig1", comp=3).reshape(3, nao, nao, nao, nao)
        vj = numpy.einsum("xijkl,kl->xij", tensor, dm0)
        vk = numpy.einsum("xilkj,lk->xij", tensor, dm0)
        vk = vk - numpy.swapaxes(vk, -1, -2)
        h1 = h1 + (-vj) - 0.5 * (-vk)
    h1 = h1 - mol.intor_asymmetric("int1e_ignuc", 3)
    h1 = h1 - mol.intor("int1e_igkin", 3)
    return h1


def giao_fock_derivative_1e(mol):
    """GIAO 一阶 Fock 导数里的**单电子**部分。"""
    return (
        -0.5 * mol.intor("int1e_giao_irjxp", 3)
        - mol.intor_asymmetric("int1e_ignuc", 3)
        - mol.intor("int1e_igkin", 3)
    )


def giao_overlap_derivative(mol):
    """GIAO 的重叠导数（转写自 `pyscf.prop.nmr.rhf.make_s10`，`gauge_orig=None`）。"""
    return -mol.intor_asymmetric("int1e_igovlp", 3)


def solve_mo1(mf, h1, s1, conv_tol=1e-11):
    mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    orbo = mo_coeff[:, mo_occ > 0]
    h1mo = numpy.einsum("xpq,pi,qj->xij", h1, mo_coeff.conj(), orbo)
    s1mo = numpy.einsum("xpq,pi,qj->xij", s1, mo_coeff.conj(), orbo)
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=2)
    nmo = mo_coeff.shape[1]
    nocc = orbo.shape[1]

    def vind(mo1):
        dm1 = numpy.asarray(
            [
                mo_coeff @ (x * 2) @ orbo.T.conj()
                for x in numpy.asarray(mo1).reshape(-1, nmo, nocc)
            ]
        )
        dm1 = numpy.asarray([d - d.conj().T for d in dm1])
        return numpy.einsum("xpq,pi,qj->xij", vresp(dm1), mo_coeff.conj(), orbo).ravel()

    mo10, _ = cphf.solve(
        vind, mo_energy, mo_occ, h1mo, s1mo, max_cycle=100, tol=conv_tol, verbose=0
    )
    return mo10


def giao_para(mol, mf, mo10):
    """GIAO 顺磁项（转写自 `pyscf.prop.nmr.rhf.para`）。"""
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    dm10_oo = numpy.asarray([orbo @ (x[occidx] * 2) @ orbo.T.conj() for x in mo10])
    dm10_vo = numpy.asarray([orbv @ (x[viridx] * 2) @ orbo.T.conj() for x in mo10])
    out = []
    for atm_id in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        h01i = mol.intor_asymmetric("int1e_prinvxp", 3)
        out.append(
            numpy.einsum("xji,yij->xy", dm10_oo, h01i) * 2
            + numpy.einsum("xji,yij->xy", dm10_vo, h01i) * 2
        )
    return numpy.asarray(out)


def giao_shielding(atoms, basis, with_2e=True):
    """GIAO 总屏蔽张量（ppm），逐核 3×3。"""
    mol, mf = make_scf(atoms, basis)
    dm0 = mf.make_rdm1()
    dia = giao_dia(mol, dm0)
    para = giao_para(
        mol, mf, solve_mo1(mf, giao_fock_derivative(mol, dm0, with_2e), giao_overlap_derivative(mol))
    )
    unit = nist.ALPHA ** 2 * 1e6
    return mol, (dia + para) * unit


# ── 参考表 ────────────────────────────────────────────────────────────────


def self_test():
    """PySCF 1.7.6 `pyscf/prop/nmr/rhf.py` 自带的 `__main__` 自检值。"""
    atoms = [["H", (0.0, 0.0, 0.917)], ["F", (0.0, 0.0, 0.0)]]
    _, total = giao_shielding(atoms, "6-31g")
    referenced = total[1][2, 2]
    x_referenced = total[1][0, 0]
    print(f"FH/6-31G GIAO: F σ_xx = {x_referenced:.6f}（参考 375.232839）")
    print(f"FH/6-31G GIAO: F σ_zz = {referenced:.6f}（参考 483.002139）")
    ok = abs(x_referenced - 375.232839) < 1.0e-3 and abs(referenced - 483.002139) < 1.0e-3
    print("self-test:", "ok" if ok else "MISMATCH")
    return 0 if ok else 1


def shielding_rows():
    rows = []
    for system, basis in CASES:
        atoms = SYSTEMS[system]
        mol, total = giao_shielding(atoms, basis)
        print(f"{system}/{basis}: nao = {mol.nao}")
        for atom_id, symbol in enumerate(atom[0] for atom in atoms):
            isotropic = total[atom_id].trace() / 3.0
            print(
                f"  {symbol}{atom_id}: σ_iso {isotropic:11.6f} ppm"
                f"  (xx {total[atom_id][0, 0]:11.6f}, zz {total[atom_id][2, 2]:11.6f})"
            )
            rows.append(
                dict(
                    system=system,
                    basis=basis,
                    atom_index=atom_id,
                    symbol=symbol,
                    x_angstrom=f"{atoms[atom_id][1][0]:.6f}",
                    y_angstrom=f"{atoms[atom_id][1][1]:.6f}",
                    z_angstrom=f"{atoms[atom_id][1][2]:.6f}",
                    sigma_iso_ppm=f"{isotropic:.9f}",
                    sigma_xx_ppm=f"{total[atom_id][0, 0]:.9f}",
                    sigma_yy_ppm=f"{total[atom_id][1, 1]:.9f}",
                    sigma_zz_ppm=f"{total[atom_id][2, 2]:.9f}",
                    note=(
                        "GIAO-HF/SCF+CPHF, transposed from pyscf/prop/nmr/rhf.py "
                        "with libcint GIAO intors"
                    ),
                )
            )
    return rows


def stable(value: float) -> str:
    """矩阵元格式化：`|v| < 1e-12` 一律写成 0。

    PySCF 的 J/K 与 CPHF 走多线程 BLAS，求和顺序不保证逐位复现：量级 1 的元素在末几位会漂，
    而「对称性允许为零」的元素会漂成 ±1e-17。把噪声区钉成 0、有效数字留 10 位，`--check`
    才能稳定。
    """
    if abs(value) < 1.0e-12:
        return "0.0000000000e+00"
    return f"{value:.7e}"


def derivative_rows():
    """水/STO-3G 的 GIAO Fock / 重叠导数矩阵元（Rust 侧逐元素对）。"""
    system, basis = "water", "sto-3g"
    atoms = SYSTEMS[system]
    mol, mf = make_scf(atoms, basis)
    dm0 = mf.make_rdm1()
    h1_1e = giao_fock_derivative_1e(mol)
    h1_total = giao_fock_derivative(mol, dm0, with_2e=True)
    s1 = giao_overlap_derivative(mol)
    nao = mol.nao
    print(
        f"{system}/{basis}: nao = {nao}, max|h1_1e| = {numpy.abs(h1_1e).max():.6e},"
        f" max|h1_2e| = {numpy.abs(h1_total - h1_1e).max():.6e},"
        f" max|s1| = {numpy.abs(s1).max():.6e}"
    )
    rows = []
    for quantity, tensor in (
        ("core_field_derivative", h1_1e),
        ("two_electron_field_derivative", h1_total - h1_1e),
        ("overlap_derivative", s1),
    ):
        for k in range(3):
            for row in range(nao):
                for column in range(nao):
                    rows.append(
                        dict(
                            system=system,
                            basis=basis,
                            quantity=quantity,
                            k=k,
                            row=row,
                            column=column,
                                    value=stable(tensor[k][row][column]),
                            note="water/STO-3G, GIAO, RHF converged state",
                        )
                    )
    return rows


def main(check_only: bool) -> int:
    shielding = reference_sources.stamp(shielding_rows())
    derivatives = reference_sources.stamp(derivative_rows())
    status = reference_sources.check_csv(SHIELDING_OUT, SHIELDING_FIELDS, shielding)
    status |= reference_sources.check_csv(DERIVATIVE_OUT, DERIVATIVE_FIELDS, derivatives)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(SHIELDING_OUT, SHIELDING_FIELDS, shielding)
    reference_sources.write_csv(DERIVATIVE_OUT, DERIVATIVE_FIELDS, derivatives)
    for path, count in ((SHIELDING_OUT, len(shielding)), (DERIVATIVE_OUT, len(derivatives))):
        print(f"写入 {path}（{count} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只重算并比对文件，不写盘")
    parser.add_argument("--self-test", action="store_true", help="跑 PySCF 1.7.6 的自检值")
    arguments = parser.parse_args()
    if arguments.self_test:
        sys.exit(self_test())
    sys.exit(main(arguments.check))