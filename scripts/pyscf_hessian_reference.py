"""F-01/F-02 与 `jobtype: freq` 的外部尺子：PySCF 的**解析** Hessian → 简正模频率。

本项目的数值 Hessian（`qwb_engine::hessian`）与 PySCF 的解析 Hessian 是同一种量在不同
实现下的两种算法，所以这条对照能同时钉住「质量加权 + 对角化 + 波数换算」这一整条链：
差 1 cm⁻¹ 以内说明实现一致；系统性偏差则说明某一步的因子错了（历史上这个坑是 5140.49
的换算常数与「哪些模算振动」的判定）。

几何是 **H2O/STO-3G/HF 的优化几何**（本项目 `qwb run` 的 `opt` 输出，R = 0.9898 Å、
∠ = 100.6°），不是实验几何 —— 因为频率是几何的函数，两边必须用同一个点。

用法：

    python scripts/pyscf_hessian_reference.py            # 重新生成 CSV
    python scripts/pyscf_hessian_reference.py --check    # 只比对新旧，不写盘
"""

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import PYSCF_DOI, PYSCF_URL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "hessian_water.csv"
)

#: H2O/STO-3G/HF 的优化几何（Å）：由 `qwb run` 的 `opt` 产出，与本仓库的测试共用同一个点。
WATER = [
    ("O", (0.0, 0.0, 0.148940223535)),
    ("H", (0.0, 0.758058893909, -0.486735104178)),
    ("H", (0.0, -0.758058893909, -0.486735104178)),
]

HEADER = ["system", "method", "basis", "quantity", "index", "value", "unit", "source_doi", "source_url"]


def rows():
    """PySCF 解析 Hessian 的频率与约化质量（3N−6 = 3 个振动模）。"""
    import numpy as np
    from pyscf import gto, scf
    from pyscf.hessian import thermo

    mol = gto.M(atom=WATER, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    hessian = mf.Hessian().kernel()
    analysis = thermo.harmonic_analysis(mol, hessian)
    frequencies = np.asarray(analysis["freq_wavenumber"], dtype=float)
    masses = np.asarray(analysis["reduced_mass"], dtype=float)
    out = []
    for index, (frequency, mass) in enumerate(zip(frequencies, masses), start=1):
        out.append(["water", "hf", "sto-3g", "wavenumber", str(index), f"{frequency:.6f}", "cm^-1"])
        out.append(["water", "hf", "sto-3g", "reduced_mass", str(index), f"{mass:.6f}", "amu"])
    for row in out:
        row.extend([PYSCF_DOI, PYSCF_URL])
    return out


def render(data):
    lines = [",".join(HEADER)]
    for row in data:
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    args = parser.parse_args()

    text = render(rows())
    if args.check:
        current = io.open(OUTPUT, encoding="utf-8").read() if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"FAIL {os.path.relpath(OUTPUT, ROOT)} does not match a fresh PySCF run")
            return 1
        print(f"ok: {os.path.relpath(OUTPUT, ROOT)} matches PySCF ({len(text.splitlines()) - 1} rows)")
        return 0

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(text)
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)} ({len(text.splitlines()) - 1} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
