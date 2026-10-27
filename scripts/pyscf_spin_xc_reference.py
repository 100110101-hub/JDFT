"""探测 WSL 里 PySCF/libxc 的自旋分辨 XC 求值，并导出逐点参考数据。

用法：
    python3 scripts/pyscf_spin_xc_reference.py            # 生成 CSV
    python3 scripts/pyscf_spin_xc_reference.py --check    # 只打印摘要

输出：qwb-core/crates/qwb-xc/tests/reference/spin_xc_pointwise.csv

列：functional,rho_alpha,rho_beta,sigma_alpha,sigma_beta,sigma_alpha_beta,
    exc_hartree,v_alpha,v_beta,d_sigma_alpha,d_sigma_beta,d_sigma_alpha_beta,note

约定（与 qwb-xc 的自旋接口一致）：
- `exc_hartree` 是**每体积**能量密度 e（E = ∫ e dV），不是「每电子」ε；
- v_σ = ∂e/∂ρ_σ，d_sigma_* = ∂e/∂σ_*，σ_αβ = ∇ρ_α·∇ρ_β；
  libxc 的 vsigma 顺序是 `(uu, ud, dd)`，CSV 按名字拆开写；
- LDA 行给 σ = 0。

梯度用共线构造：∇ρ_α = (a,0,0)、∇ρ_β = (b,0,0)，
于是 σ_αα = a²、σ_ββ = b²、σ_αβ = a·b，且闭壳层的
σ = σ_αα + σ_ββ + 2σ_αβ = |∇(ρ_α+ρ_β)|²。（要求 σ_αβ² ≤ σ_αα σ_ββ。）
"""

import argparse
import csv
import math
import os
import sys

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-xc", "tests", "reference", "spin_xc_pointwise.csv"
)

RHO_PAIRS = [
    (0.5, 0.5),
    (0.4, 0.3),
    (0.3, 0.1),
    (0.2, 0.05),
    (0.1, 0.01),
    (0.02, 0.005),
    (1.0, 0.5),
    (2.0, 0.1),
    (0.5, 0.0),
    (0.05, 0.0),
    (0.001, 0.001),
]

SIGMA_TRIPLES = [
    (0.0, 0.0, 0.0),
    (0.01, 0.01, 0.0),
    (0.05, 0.02, 0.01),
    (0.2, 0.1, -0.03),
    (1.0, 0.0, 0.0),
]

LDA_FUNCTIONALS = ["VWN5", "LDA_C_VWN_RPA", "LDA_C_PW_MOD", "LDA_X"]
GGA_FUNCTIONALS = ["GGA_C_PBE", "GGA_X_PBE"]

NOTES = {
    "VWN5": "libxc 的 VWN5（本项目 svwn5.yaml 用的其实是 RPA 参数集）",
    "LDA_C_VWN_RPA": "VWN RPA（本项目 svwn5.yaml 的实际实现）",
    "LDA_C_PW_MOD": "PW92 modified 相关（PBE 的 LDA 基）",
    "LDA_X": "Slater 交换（LDA-X，α=2/3）",
    "GGA_C_PBE": "PBE 相关（ζ 插值 + 梯度修正）",
    "GGA_X_PBE": "PBE 交换（自旋标度精确）",
}


def gradients(sigma_aa, sigma_bb, sigma_ab):
    """构造两个三维梯度向量，使它们的三个 σ 与给定值一致。

    σ_αα = |a|²、σ_ββ = |b|²、σ_αβ = a·b。取
        a = (√σ_αα, 0, 0)、b = (σ_αβ/√σ_αα, √(σ_ββ − σ_αβ²/σ_αα), 0)
    （Cauchy–Schwarz 保证根号内非负）。返回 (a_vec, b_vec)。
    """
    import numpy as np

    if sigma_aa <= 0.0:
        a = np.zeros(3)
        b = np.array([0.0, math.sqrt(max(sigma_bb, 0.0)), 0.0])
        return a, b
    root = math.sqrt(sigma_aa)
    first = sigma_ab / root
    second_squared = max(sigma_bb - first * first, 0.0)
    a = np.array([root, 0.0, 0.0])
    b = np.array([first, math.sqrt(second_squared), 0.0])
    return a, b


def spin_rho(rho_a, rho_b, a, b, gga):
    """两套密度数组：LDA 给 (1,1)（只有 den），GGA 给 (4,1)
    （den, gx, gy, gz, lapl, tau）。"""
    import numpy as np

    def block(rho, grad):
        if not gga:
            return np.array([[rho]])
        values = np.zeros((4, 1))
        values[0, 0] = rho
        values[1, 0] = grad[0]
        values[2, 0] = grad[1]
        values[3, 0] = grad[2]
        return values

    return block(rho_a, a), block(rho_b, b)


def closed_rho(rho, a, b, gga):
    import numpy as np

    if not gga:
        return np.array([[rho]])
    values = np.zeros((4, 1))
    values[0, 0] = rho
    values[1, 0] = a[0] + b[0]
    values[2, 0] = a[1] + b[1]
    values[3, 0] = a[2] + b[2]
    return values


def evaluate_spin(libxc, name, rho_a, rho_b, sig_aa, sig_bb, sig_ab):
    gga = name.startswith("GGA")
    a, b = gradients(sig_aa, sig_bb, sig_ab)
    rho = spin_rho(rho_a, rho_b, a, b, gga)
    out = libxc.eval_xc(name, rho, spin=1, deriv=1)
    exc, vxc = out[0], out[1]
    e = float(exc[0]) * (rho_a + rho_b)
    v = vxc[0][0]
    d_sigma = vxc[1][0] if gga else (0.0, 0.0, 0.0)
    return e, float(v[0]), float(v[1]), d_sigma


def evaluate_closed(libxc, name, rho, sig_aa, sig_bb, sig_ab):
    gga = name.startswith("GGA")
    a, b = gradients(sig_aa, sig_bb, sig_ab)
    out = libxc.eval_xc(name, closed_rho(rho, a, b, gga), spin=0, deriv=1)
    return float(out[0][0]) * rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default=CSV_PATH)
    args = parser.parse_args()

    try:
        from pyscf.dft import libxc
    except Exception as exc:  # pragma: no cover - 环境问题直接报出
        print(f"无法导入 pyscf.dft.libxc：{exc}", file=sys.stderr)
        return 2

    rows = []
    print("=== 自旋分辨（spin=1）逐点求值 ===")
    for name in LDA_FUNCTIONALS:
        for rho_a, rho_b in RHO_PAIRS:
            e, v_a, v_b, _ = evaluate_spin(libxc, name, rho_a, rho_b, 0.0, 0.0, 0.0)
            rows.append(
                dict(
                    functional=name,
                    rho_alpha=rho_a,
                    rho_beta=rho_b,
                    sigma_alpha=0.0,
                    sigma_beta=0.0,
                    sigma_alpha_beta=0.0,
                    exc_hartree=e,
                    v_alpha=v_a,
                    v_beta=v_b,
                    d_sigma_alpha=0.0,
                    d_sigma_beta=0.0,
                    d_sigma_alpha_beta=0.0,
                    note=NOTES[name] + "；spin=1；libxc via pyscf",
                )
            )
    for name in GGA_FUNCTIONALS:
        for rho_a, rho_b in RHO_PAIRS:
            for sig_aa, sig_bb, sig_ab in SIGMA_TRIPLES:
                e, v_a, v_b, d_sigma = evaluate_spin(
                    libxc, name, rho_a, rho_b, sig_aa, sig_bb, sig_ab
                )
                rows.append(
                    dict(
                        functional=name,
                        rho_alpha=rho_a,
                        rho_beta=rho_b,
                        sigma_alpha=sig_aa,
                        sigma_beta=sig_bb,
                        sigma_alpha_beta=sig_ab,
                        exc_hartree=e,
                        v_alpha=v_a,
                        v_beta=v_b,
                        d_sigma_alpha=float(d_sigma[0]),
                        d_sigma_beta=float(d_sigma[2]),
                        d_sigma_alpha_beta=float(d_sigma[1]),
                        note=NOTES[name] + "；spin=1；libxc via pyscf",
                    )
                )
    print(f"自旋行数：{len(rows)}")

    print("=== 闭壳层 vs ζ=0 退化（应当逐点相同）===")
    for name in LDA_FUNCTIONALS + GGA_FUNCTIONALS:
        for rho in (0.6, 0.1):
            sig = 0.02 if name.startswith("GGA") else 0.0
            spin_e = evaluate_spin(libxc, name, 0.5 * rho, 0.5 * rho, sig, sig, sig)[0]
            closed_e = evaluate_closed(libxc, name, rho, sig, sig, sig)
            print(
                f"{name:12s} ρ={rho:.3f} σ={sig:.3f}: spin {spin_e:.12f} vs "
                f"closed {closed_e:.12f} 差 {spin_e - closed_e:+.3e}"
            )

    print("=== 自洽性：v_σ 是否等于 ∂e/∂ρ_σ（中心差分）===")
    delta = 1.0e-6
    for name in ("LDA_C_VWN_RPA", "LDA_C_PW_MOD", "GGA_C_PBE", "GGA_X_PBE", "LDA_X"):
        for rho_a, rho_b in ((0.3, 0.1), (0.5, 0.05)):
            sig = 0.02 if name.startswith("GGA") else 0.0
            _, v_a, v_b, _ = evaluate_spin(libxc, name, rho_a, rho_b, sig, sig, 0.5 * sig)
            plus = evaluate_spin(libxc, name, rho_a + delta, rho_b, sig, sig, 0.5 * sig)[0]
            minus = evaluate_spin(libxc, name, rho_a - delta, rho_b, sig, sig, 0.5 * sig)[0]
            fd = (plus - minus) / (2.0 * delta)
            plus = evaluate_spin(libxc, name, rho_a, rho_b + delta, sig, sig, 0.5 * sig)[0]
            minus = evaluate_spin(libxc, name, rho_a, rho_b - delta, sig, sig, 0.5 * sig)[0]
            fdb = (plus - minus) / (2.0 * delta)
            print(
                f"{name:12s} ρ=({rho_a},{rho_b}): v_a {v_a:+.9f} vs {fd:+.9f} "
                f"(差 {v_a - fd:+.2e})；v_b {v_b:+.9f} vs {fdb:+.9f} (差 {v_b - fdb:+.2e})"
            )

    if args.check:
        print("--check：不写 CSV")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = [
        "functional",
        "rho_alpha",
        "rho_beta",
        "sigma_alpha",
        "sigma_beta",
        "sigma_alpha_beta",
        "exc_hartree",
        "v_alpha",
        "v_beta",
        "d_sigma_alpha",
        "d_sigma_beta",
        "d_sigma_alpha_beta",
        "note",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {args.out}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
