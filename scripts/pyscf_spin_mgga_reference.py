#!/usr/bin/env python3
"""自旋分辨 meta-GGA 的逐点参考值（SC-06 的第二半要对着它实现）。

**为什么要先落参考值**：开壳层的 M06-L / M06-2X 现在被 `qwb-xc` 拒绝
（`MggaXM062x` 没有自旋分辨实现），要做它就得有**外部尺子**逐点对齐 —— 与
`spin_xc_pointwise.csv`（LDA/GGA）同一套路，只是每个自旋块多一行 τ。

来源（逐行写进 CSV，DA-08 的要求）：
- `source_doi = 10.1063/5.0006074`（PySCF，`pyscf.dft.libxc` 只是 libxc 的绑定）；
- `source_url = https://github.com/pyscf/pyscf`。

**真正的泛函定义在 libxc**（Lehtola et al., SoftwareX 2018, DOI 10.1016/j.softx.2018.06.003）：
`MGGA_X_M06_L` / `MGGA_C_M06_L` / `MGGA_X_M06_2X` / `MGGA_C_M06_2X`。
CSV 只记「libxc 在这一点给出什么」，不冒充文献表值。

用法（WSL，需要 pyscf）：
    python3 scripts/pyscf_spin_mgga_reference.py            # 生成 CSV
    python3 scripts/pyscf_spin_mgga_reference.py --check     # 与磁盘逐字比较
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_sources import PYSCF_DOI, PYSCF_URL, check_csv, stamp, write_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-xc", "tests", "reference", "spin_mgga_pointwise.csv"
)

FIELDS = (
    "case",
    "functional",
    "libxc_id",
    "rho_alpha",
    "rho_beta",
    "sigma_alpha",
    "sigma_beta",
    "sigma_alpha_beta",
    "tau_alpha",
    "tau_beta",
    "e_density",
    "v_rho_alpha",
    "v_rho_beta",
    "d_sigma_alpha",
    "d_sigma_beta",
    "d_sigma_alpha_beta",
    "d_tau_alpha",
    "d_tau_beta",
)

#: 要钉住的组件。**交换与相关分开**：交换的自旋标度是精确的（`e_σ = e(2ρ_σ, 4σ_σ, 2τ_σ)`），
#: 相关不是（M06 的相关是 ζ 相关的 Stoll 分解），分开列才能分别判定是谁没对上。
#: ⚠ 实测：**这个 libxc 构建里 M06 / M06-2X / M06-HF 的交换项不存在**
#: （`MGGA_X_M06`、`MGGA_X_M06_2X`、`MGGA_X_M062X` 全部 `KeyError`），只有
#: `MGGA_X_M06_L` 与四个相关项（M06-L / M06 / M06-2X / M06-HF）可用。
#: 所以这里钉的是**能钉的部分**：
#:   * M06-L 的交换 + 相关（一对完整的 meta-GGA 组件）；
#:   * M06 / M06-2X 的**相关** —— 这正是自旋化最难的半边（ζ 相关的 Stoll 分解），
#:     交换那半边有精确的自旋标度 `e_σ = e_closed(2ρ_σ, 4σ_σσ, 2τ_σ)`，可以对着
#:     **自己的闭壳层实现**逐点验，不需要外部尺子。
COMPONENTS = (
    ("m06-l exchange", "MGGA_X_M06_L"),
    ("m06-l correlation", "MGGA_C_M06_L"),
    ("m06 correlation", "MGGA_C_M06"),
    ("m06-2x correlation", "MGGA_C_M06_2X"),
)

#: 逐点样本：`(名字, ρ_α, ρ_β, σ_αα 系数, σ_ββ 系数, σ_αβ 符号, τ 相对 τ_unif 的余量)`。
#:
#: 用**相对量**而不是绝对值，是为了让样本落在物理区间里（τ ≥ τ_W ≥ 0，σ 与 ρ 同量级），
#: 同时覆盖 ζ = 0 / 0.37 / 1 三种自旋极化。
CASES = (
    # ζ = 0（闭壳层极限：自旋分辨实现必须与闭壳层逐位一致）
    ("closed_shell_dense", 1.0, 1.0, 0.05, 0.05, 0.01, 0.20),
    ("closed_shell_sparse", 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-7, 0.05),
    # ζ ≈ 0.37（弱极化）
    ("weakly_polarized", 1.2, 0.55, 0.08, 0.02, 0.0, 0.35),
    # ζ ≈ 1（完全极化：β 通道接近 0，导数最挑剔）
    ("fully_polarized", 1.5, 1.0e-8, 0.10, 0.0, 0.0, 0.30),
    # 大梯度（σ 与 ρ^{8/3} 同量级）
    ("large_gradient", 0.8, 0.45, 0.40, 0.15, 0.05, 0.60),
)


def spin_uniform_tau(rho: float) -> float:
    """均匀电子气的 τ（自旋标度后）：`τ_unif = (3/10)(3π²)^{2/3} ρ^{5/3}`。"""
    return 0.3 * (3.0 * math.pi * math.pi) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)


def samples():
    """把 CASES 展开成 libxc 需要的七个量。"""
    rows = []
    for name, rho_a, rho_b, s_a, s_b, s_ab_sign, tau_margin in CASES:
        total = rho_a + rho_b
        # σ 取成与 ρ^{8/3} 成比例（这样约化梯度 s 是 O(1) 的量级）
        sigma_aa = s_a * total ** (8.0 / 3.0)
        sigma_bb = s_b * total ** (8.0 / 3.0)
        sigma_ab = s_ab_sign * math.sqrt(max(sigma_aa, 0.0) * max(sigma_bb, 0.0))
        tau_alpha = spin_uniform_tau(rho_a) * (1.0 + tau_margin)
        tau_beta = spin_uniform_tau(rho_b) * (1.0 + tau_margin) if rho_b > 1.0e-7 else 1.0e-12
        rows.append(
            {
                "case": name,
                "rho_alpha": rho_a,
                "rho_beta": rho_b,
                "sigma_alpha": sigma_aa,
                "sigma_beta": sigma_bb,
                "sigma_alpha_beta": sigma_ab,
                "tau_alpha": tau_alpha,
                "tau_beta": tau_beta,
            }
        )
    return rows


def evaluate(libxc, libxc_id, sample):
    """一点、一个组件：返回 `(e, v_α, v_β, dσ_αα, dσ_ββ, dσ_αβ, dτ_α, dτ_β)`。

    PySCF 的自旋密度块是 `(6, N)`：`(den, gx, gy, gz, lapl, tau)`。梯度块用
    `σ` 反推一个满足 `σ_αα = |∇ρ_α|²`、`σ_αβ = ∇ρ_α·∇ρ_β` 的排布（与
    `pyscf_spin_xc_reference.py` 的 `gradients` 同构，只是多了 τ 行）。
    """
    import numpy as np

    sigma_aa = sample["sigma_alpha"]
    sigma_bb = sample["sigma_beta"]
    sigma_ab = sample["sigma_alpha_beta"]
    grad_a = (math.sqrt(max(sigma_aa, 0.0)), 0.0, 0.0)
    grad_b = (
        sigma_ab / math.sqrt(max(sigma_aa, 1.0e-30)),
        math.sqrt(max(sigma_bb - (sigma_ab**2) / max(sigma_aa, 1.0e-30), 0.0)),
        0.0,
    )

    def block(rho, grad, tau):
        return np.array([[rho], [grad[0]], [grad[1]], [grad[2]], [0.0], [tau]])

    rho = np.stack(
        [
            block(sample["rho_alpha"], grad_a, sample["tau_alpha"]),
            block(sample["rho_beta"], grad_b, sample["tau_beta"]),
        ]
    )
    out = libxc.eval_xc(libxc_id, rho, spin=1, deriv=1)
    # PySCF 对 meta-GGA 的 `deriv=1` 返回 `(exc, vxc, None, None)`，其中 `vxc` 是一个
    # **列表**：`[0]` 是 `v_rho`（形状 (1,2)：α、β），`[1]` 是 `v_sigma`
    # （(1,3)：顺序是 (uu, ud, dd)，与 libxc 一致），`[2]` 是 laplacian 的导数（自旋
    # 情形下是 None），`[3]` 是 `v_tau`（(1,2)：α、β）。**不是** `out[2]`/`out[3]` ——
    # 那两个在自旋 meta-GGA 下是 None（踩过：按直觉取会得到 TypeError）。
    exc = float(out[0][0])
    e = exc * (sample["rho_alpha"] + sample["rho_beta"])
    v_rho, v_sigma, _v_lapl, v_tau = out[1][0], out[1][1], out[1][2], out[1][3]
    v_alpha = float(v_rho[0][0])
    v_beta = float(v_rho[0][1])
    d_sigma_aa = float(v_sigma[0][0])
    d_sigma_ab = float(v_sigma[0][1])
    d_sigma_bb = float(v_sigma[0][2])
    d_tau_alpha = float(v_tau[0][0])
    d_tau_beta = float(v_tau[0][1])
    return (
        e,
        v_alpha,
        v_beta,
        d_sigma_aa,
        d_sigma_bb,
        d_sigma_ab,
        d_tau_alpha,
        d_tau_beta,
    )


def build_rows():
    from pyscf.dft import libxc  # noqa: PLC0415

    rows = []
    for sample in samples():
        for label, libxc_id in COMPONENTS:
            values = evaluate(libxc, libxc_id, sample)
            row = {
                "case": sample["case"],
                "functional": label,
                "libxc_id": libxc_id,
                "rho_alpha": f"{sample['rho_alpha']:.12e}",
                "rho_beta": f"{sample['rho_beta']:.12e}",
                "sigma_alpha": f"{sample['sigma_alpha']:.12e}",
                "sigma_beta": f"{sample['sigma_beta']:.12e}",
                "sigma_alpha_beta": f"{sample['sigma_alpha_beta']:.12e}",
                "tau_alpha": f"{sample['tau_alpha']:.12e}",
                "tau_beta": f"{sample['tau_beta']:.12e}",
                "e_density": f"{values[0]:.12e}",
                "v_rho_alpha": f"{values[1]:.12e}",
                "v_rho_beta": f"{values[2]:.12e}",
                "d_sigma_alpha": f"{values[3]:.12e}",
                "d_sigma_beta": f"{values[4]:.12e}",
                "d_sigma_alpha_beta": f"{values[5]:.12e}",
                "d_tau_alpha": f"{values[6]:.12e}",
                "d_tau_beta": f"{values[7]:.12e}",
            }
            rows.append(row)
    return stamp(rows, PYSCF_DOI, PYSCF_URL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    if args.check:
        return check_csv(CSV_PATH, FIELDS, rows)

    write_csv(CSV_PATH, FIELDS, rows)
    print(f"wrote {CSV_PATH}（{len(rows)} 行 = {len(CASES)} 个样本 × {len(COMPONENTS)} 个组件）")
    for row in rows[:4]:
        print(f"  {row['case']:22s} {row['functional']:20s} e = {row['e_density']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
