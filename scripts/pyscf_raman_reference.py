"""RS-06 的外部尺子：**独立**算一遍 Raman 活性（PySCF + numpy），逐模对照。

判据（`project-phase2.md` §6.6）要的是「与文献 Raman 峰**相对强度**排序一致（至少
H2O/CH4）」。本脚本给的是**第一半**的可执行尺子：同一个几何、同一基组下，用一套与本仓库
**完全无关**的实现算 `A_k = 45 ᾱ'² + 7 γ'²`。两套实现一致只说明「我们没算错」；与实验
的对照是另一半（`scripts/gen_raman_experiment.py`）。

# 独立在哪里

| 环节 | 本仓库（`qwb-core`） | 这里 |
|---|---|---|
| 简正模 | `qwb_engine::hessian::numerical_hessian`（**数值** Hessian，中心差分解析梯度） | `mf.Hessian()`（PySCF 的**解析** Hessian）+ `thermo.harmonic_analysis` |
| 极化率 | `qwb_response::static_polarizability`（自研 CPHF，`(A+B)` 矩阵自由迭代） | 有限场（`±0.001 a.u.` 的中心差分，`H += ∓λ·∫r`） |
| `∂α/∂R` | `qwb_engine::raman::polarizability_derivatives`（中心差分，`δ = 0.01 bohr`） | 本脚本第 3 步（同样的中心差分、同样的 δ，但用上面的有限场 α） |
| 投影 | `raman_activities`（Rust） | 本脚本第 4 步（numpy，`norm_mode` 直接来自 PySCF） |

所以两套实现只在「有限差分」这一层是共同的数学（这是个事实，不是重复利用代码）：
**Hessian、SCF、CPHF/有限场、投影**四条都各自独立。

# 单位与约定

* `α` 由 PySCF 的 `dip_moment(unit='A.U.')` 给出（`e·bohr`），有限场差分得到 `bohr³`；
* `∂α/∂x` 单位 `bohr²`；乘 PySCF 的 `nist.BOHR**2` 换成 `Å²`，再投影到
  `norm_mode`（= `L/√m`，单位 `amu^{-1/2}`）上得到 `∂α/∂Q`，单位 `Å²/√amu`；
* 活性 `A = 45 ᾱ'² + 7 γ'²`，单位 `Å⁴/amu`（Placzek，与 Gaussian `Freq=Raman` 同一定义）；
* 相对强度 `I ∝ A (ν̃0−ν̃)⁴ / [ν̃ (1−e^{−hcν̃/kT})]`，`ν̃0 = 1e7/532 nm`、`T = 298.15 K`，
  归一化到最强 = 100。这一栏是为了让「相对强度排序」这句话可核对 —— 活性与相对强度
  **不是**同一个数。

几何是**每个基组自己的 STO-3G/def2-SVP HF 优化点**（简单 BFGS 在这个脚本里跑，收敛到
`max|grad| < 1e-8`），因为频率/活性都是几何的函数，两边必须用同一个点；优化后的坐标
打印成 12 位小数写进 `GEOMETRIES`，Rust 侧逐字复用。

用法：

    python scripts/pyscf_raman_reference.py            # 重新生成 CSV
    python scripts/pyscf_raman_reference.py --check    # 只比对，不写盘
"""

import argparse
import io
import os
import sys

# **单线程是 `--check` 的前提**（实测，不是猜测）：多线程下同一份代码两次运行的活性
# 相对差到 5e-5（water/STO-3G 的 A₁ 一次 9.26552062、一次 9.26503783），而 `--check`
# 是**逐字节**比对。单线程下两次独立进程的输出逐位相同（9.26455034 / 47.80681473 /
# 21.55104830 两次一致），而且这几个小体系反而更快（9.5 s vs 26 s —— 线程开销 > 并行收益）。
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import (  # noqa: E402
    PYSCF_DOI,
    PYSCF_URL,
    check_csv,
    rel,
    stamp,
    write_csv,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "raman_activities.csv"
)

#: 中心差分步长（bohr）——与本仓库 `hessian::HESSIAN_DELTA_BOHR` 是同一个数。
DELTA_BOHR = 1.0e-2

#: 有限场的场强（a.u.）。
FIELD_AU = 1.0e-3

#: 相对强度用的激发光（532 nm = 倍频 Nd:YAG）与温度。
EXCITATION_NM = 532.0
TEMPERATURE_K = 298.15

#: 每个体系的 HF 优化几何（Å，12 位小数）。由 `--optimise` 打印，两边逐字复用。
GEOMETRIES = {
    ("water", "sto-3g"): [
        ("O", (0.0, 0.0, 0.150_168_163_058)),
        ("H", (0.0, 0.758_080_671_347, -0.485_634_081_529)),
        ("H", (0.0, -0.758_080_671_347, -0.485_634_081_529)),
    ],
    ("methane", "sto-3g"): [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.625_276_284_795, 0.625_276_284_795, 0.625_276_284_795)),
        ("H", (-0.625_276_284_795, -0.625_276_284_795, 0.625_276_284_795)),
        ("H", (-0.625_276_284_795, 0.625_276_284_795, -0.625_276_284_795)),
        ("H", (0.625_276_284_795, -0.625_276_284_795, -0.625_276_284_795)),
    ],
    ("water", "def2-svp"): [
        ("O", (0.0, 0.0, 0.109_228_829_465)),
        ("H", (0.0, 0.750_347_091_264, -0.465_164_414_732)),
        ("H", (0.0, -0.750_347_091_264, -0.465_164_414_732)),
    ],
}

HEADER = [
    "system",
    "method",
    "basis",
    "quantity",
    "index",
    "value",
    "unit",
    "source_doi",
    "source_url",
]


def optimise(atom, basis, verbose=False):
    """最简单的 BFGS（解析梯度），把几何收到 `max|grad| < 1e-8` 并打印 12 位小数。"""
    import numpy as np
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    x = mol.atom_coords().copy()
    n_dof = 3 * mol.natm
    g = mf.nuc_grad_method().kernel().reshape(-1)
    hess_inv = np.eye(n_dof) * 0.5
    for _ in range(80):
        step = -hess_inv.dot(g)
        if np.abs(step).max() > 0.1:
            step *= 0.1 / np.abs(step).max()
        x_new = x + step.reshape(-1, 3)
        mol.set_geom_(x_new, unit="Bohr")
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        g_new = mf.nuc_grad_method().kernel().reshape(-1)
        s = x_new.reshape(-1) - x.reshape(-1)
        y = g_new - g
        sy = s.dot(y)
        if abs(sy) > 1e-14:
            rho = 1.0 / sy
            hess_inv = (np.eye(n_dof) - rho * np.outer(s, y)).dot(hess_inv).dot(
                np.eye(n_dof) - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)
        x, g = x_new, g_new
        if np.abs(g).max() < 1e-8:
            break
    if verbose:
        from pyscf.data import nist

        print(f"# {basis}: max|grad| = {np.abs(g).max():.3e} (coords in Å below)")
        for symbol, position in zip(
            [mol.atom_symbol(index) for index in range(mol.natm)], x * nist.BOHR
        ):
            print(
                f'        ("{symbol}", '
                f"({position[0]:.12f}, {position[1]:.12f}, {position[2]:.12f})),"
            )
    return np.abs(g).max()


def polarizability_au(mol):
    """`α_ij = ∂μ_i/∂F_j`（原子单位）：有限场中心差分，`H_core ∓ λ·∫r`。"""
    import numpy as np
    from pyscf import scf

    mf0 = scf.RHF(mol).run(conv_tol=1e-12)
    hcore = mf0.get_hcore()
    dipoles_ao = mol.intor("int1e_r", comp=3)
    alpha = np.zeros((3, 3))
    for axis in range(3):
        moments = []
        for sign in (+1.0, -1.0):
            mf = scf.RHF(mol)
            mf.get_hcore = lambda *args, s=sign, a=axis: hcore - s * FIELD_AU * dipoles_ao[a]
            mf.conv_tol = 1e-12
            mf.kernel()
            moments.append(mf.dip_moment(unit="A.U.", verbose=0))
        # PySCF 的 `-λ∫r` 方向：μ(+λ) 与 μ(−λ) 的差除以 2λ 给出 **负** 的 α，取反即正定。
        alpha[:, axis] = (moments[1] - moments[0]) / (2.0 * FIELD_AU)
    return alpha


def polarizability_derivatives(mol):
    """`∂α/∂x_c`（`bohr²`），中心差分，步长 `DELTA_BOHR`（bohr）。"""
    import numpy as np
    from pyscf import gto
    from pyscf.data import nist

    coordinates = mol.atom_coords(unit="Bohr").copy()
    derivatives = []
    for atom in range(mol.natm):
        for axis in range(3):
            values = []
            for sign in (+1.0, -1.0):
                moved = coordinates.copy()
                moved[atom, axis] += sign * DELTA_BOHR
                displaced = gto.M(
                    atom=list(
                        zip(
                            [mol.atom_symbol(index) for index in range(mol.natm)],
                            moved,
                        )
                    ),
                    basis=mol.basis,
                    unit="Bohr",
                    verbose=0,
                )
                values.append(polarizability_au(displaced))
            derivatives.append((values[0] - values[1]) / (2.0 * DELTA_BOHR))
    return np.asarray(derivatives)


def activity_from_derivative(derivative):
    """Placzek：`45 ᾱ'² + 7 γ'²`（`derivative` 单位 `Å²/√amu`）。"""
    import numpy as np

    trace = np.trace(derivative) / 3.0
    anisotropy = 0.5 * (
        (derivative[0, 0] - derivative[1, 1]) ** 2
        + (derivative[1, 1] - derivative[2, 2]) ** 2
        + (derivative[2, 2] - derivative[0, 0]) ** 2
    ) + 3.0 * (
        derivative[0, 1] ** 2 + derivative[0, 2] ** 2 + derivative[1, 2] ** 2
    )
    return 45.0 * trace * trace + 7.0 * anisotropy


def rows():
    """一个体系一组行：静态 α（标量）+ 每个振动模的波数、活性与相对强度。"""
    import numpy as np
    from pyscf import gto, scf
    from pyscf.data import nist
    from pyscf.hessian import thermo

    out = []
    for (name, basis), geometry in GEOMETRIES.items():
        if not geometry:
            raise SystemExit(
                f"{name}/{basis} 的几何还没跑出来：先用 `--optimise {name} {basis}` 打印坐标，"
                "填进 GEOMETRIES"
            )
        mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        alpha = polarizability_au(mol)
        derivatives = polarizability_derivatives(mol)

        hessian = mf.Hessian().kernel()
        analysis = thermo.harmonic_analysis(mol, hessian)
        wavenumbers = [float(value) for value in analysis["freq_wavenumber"]]
        modes = np.asarray(analysis["norm_mode"], dtype=float)  # (n_modes, natm, 3) = L/√m
        activities = []
        for mode in modes:
            derivative = np.einsum("c,cij->ij", mode.reshape(-1), derivatives) * nist.BOHR**2
            activities.append(float(activity_from_derivative(derivative)))
        thermal = 1.438776877 / TEMPERATURE_K
        intensities = [
            a * (1.0e7 / EXCITATION_NM - w) ** 4 / (w * (1.0 - np.exp(-thermal * w)))
            for a, w in zip(activities, wavenumbers)
        ]
        strongest = max(intensities)
        for index, (wavenumber, activity, intensity) in enumerate(
            zip(wavenumbers, activities, intensities), start=1
        ):
            out.append([name, "hf", basis, "wavenumber", str(index), f"{wavenumber:.6f}", "cm^-1"])
            out.append(
                [name, "hf", basis, "raman_activity", str(index), f"{activity:.8f}", "angstrom^4/amu"]
            )
            out.append(
                [
                    name,
                    "hf",
                    basis,
                    "raman_relative_intensity",
                    str(index),
                    f"{100.0 * intensity / strongest:.6f}",
                    "relative",
                ]
            )
        out.append(
            [name, "hf", basis, "polarizability_iso", "0", f"{np.trace(alpha) / 3.0:.10f}", "bohr^3"]
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    parser.add_argument(
        "--optimise", nargs=2, metavar=("SYSTEM", "BASIS"), help="只打印某体系的 HF 优化几何"
    )
    args = parser.parse_args()

    if args.optimise:
        name, basis = args.optimise
        atoms = {
            "water": [
                ("O", (0.0, 0.0, 0.1173)),
                ("H", (0.0, 0.7572, -0.4692)),
                ("H", (0.0, -0.7572, -0.4692)),
            ],
            "methane": [
                ("C", (0.0, 0.0, 0.0)),
                ("H", (0.6291, 0.6291, 0.6291)),
                ("H", (-0.6291, -0.6291, 0.6291)),
                ("H", (-0.6291, 0.6291, -0.6291)),
                ("H", (0.6291, -0.6291, -0.6291)),
            ],
        }[name]
        optimise(atoms, basis, verbose=True)
        return 0

    data = rows()
    fieldnames = HEADER[:-2]
    dict_rows = [dict(zip(fieldnames, row)) for row in data]
    stamped = stamp(dict_rows)
    if args.check:
        return check_csv(OUTPUT, fieldnames, stamped)
    write_csv(OUTPUT, fieldnames, stamped)
    print(f"wrote {rel(OUTPUT)} ({len(stamped)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
