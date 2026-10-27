"""用 PySCF/libxc 生成**非极化** M06 交换的逐点参考（FX-02）。

输出：qwb-core/crates/qwb-xc/tests/reference/meta_gga_exchange.csv
约定：`exc_hartree` 是**每体积**能量密度 `e = ρ·ε`；`v_* = ∂e/∂*`；
只取 libxc 的**单一交换功能**（不掺 0.54 精确交换）。
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-xc", "tests", "reference", "meta_gga_exchange.csv"
)
K = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0)

POINTS = [
    (0.5, 0.0, 1.0),
    (0.5, 0.10, 1.0),
    (0.5, 0.10, 0.6),
    (1.0, 0.05, 1.0),
    (1.0, 0.05, 0.3),
    (0.2, 0.02, 1.0),
    (2.0, 0.30, 0.8),
    (0.05, 0.001, 1.2),
    (3.0, 0.5, 0.5),
    (0.1, 0.5, 0.1),
]
FUNCTIONALS = ["mgga_x_m06_l", "hyb_mgga_x_m06_2x"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from pyscf.dft import libxc

    rows = []
    for name in FUNCTIONALS:
        for rho, sigma_scale, tau_scale in POINTS:
            sigma = sigma_scale * rho ** (8.0 / 3.0)
            tau = tau_scale * K * rho ** (5.0 / 3.0)
            values = np.zeros((6, 1))
            values[0, 0] = rho
            values[1, 0] = math.sqrt(sigma)
            values[5, 0] = tau
            out = libxc.eval_xc(name, values, spin=0, deriv=1)
            energy = float(out[0][0]) * rho
            v_rho = float(out[1][0][0])
            v_sigma = float(out[1][1][0])
            v_tau = float(out[1][3][0])
            rows.append(
                dict(
                    functional=name,
                    rho=f"{rho!r}",
                    sigma=f"{sigma!r}",
                    tau=f"{tau!r}",
                    exc_hartree=f"{energy:.15e}",
                    v_rho=f"{v_rho:.15e}",
                    v_sigma=f"{v_sigma:.15e}",
                    v_tau=f"{v_tau:.15e}",
                )
            )
            print(
                f"{name:18s} ρ={rho:5.2f} σ={sigma:.3e} τ={tau:.3e}: e={energy:+.10e}"
            )
    if args.check:
        return 0
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = ["functional", "rho", "sigma", "tau", "exc_hartree", "v_rho", "v_sigma", "v_tau"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
