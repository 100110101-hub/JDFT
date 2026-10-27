"""用 PySCF 生成 UKS（O2/OH，STO-3G）的参考能量与 ⟨S²⟩。

用法：
    python3 scripts/pyscf_uks_reference.py            # 生成 CSV
    python3 scripts/pyscf_uks_reference.py --check    # 只打印

输出：qwb-core/crates/qwb-scf/tests/reference/uks_open_shell.csv

# 功能名对应关系（很重要）

本项目的 `svwn5.yaml` 用的其实是 libxc 的 **VWN RPA** 参数集，所以对应的 PySCF
写法是 `LDA_X,LDA_C_VWN_RPA`，不是 `SVWN`（后者是 libxc 的 `LDA_C_VWN`，
两者在 rs ≈ 0.7 处差 0.02 Ha/e）。PBE 没有这个歧义。

几何与 `qwb-core/examples/o2_uhf_sto3g.yaml` / `oh_uhf_sto3g.yaml` 逐字一致。
"""

import argparse
import csv
import os
import sys

import numpy as np
from pyscf import dft, gto

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-scf", "tests", "reference", "uks_open_shell.csv"
)

MOLECULES = {
    # R = 1.208 Å，与 examples/o2_uhf_sto3g.yaml 一致
    "O2": dict(
        atom="""
O 0.0 0.0 0.0
O 0.0 0.0 1.208
""",
        multiplicity=3,
    ),
    # R = 0.980 Å
    "OH": dict(
        atom="""
O 0.0 0.0 0.0
H 0.0 0.0 0.980
""",
        multiplicity=2,
    ),
}

# 每个体系的容差：残差来源是**网格**（我们的 grid-fine ↔ PySCF grids.level = 4），
# 不是公式 —— O2 在 grid-veryfine 上差 1.0e-7，OH 差 1.1e-6（OH 的密度更弥散，
# 尾部积分对网格更敏感）。数值写进 CSV，测试按行读。
TOLERANCES = {"O2": "1.0e-6", "OH": "5.0e-6"}

# 本项目功能名 → PySCF 的 xc 字符串
FUNCTIONALS = {
    "svwn5": ("LDA_X,LDA_C_VWN_RPA", "本项目的 svwn5.yaml 是 libxc 的 VWN RPA 参数集"),
    "pbe": ("PBE", "PBE = GGA_X_PBE + GGA_C_PBE"),
}


def run_uks(species, functional):
    spec = MOLECULES[species]
    xc, note = FUNCTIONALS[functional]
    mol = gto.M(
        atom=spec["atom"],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
        spin=spec["multiplicity"] - 1,
        charge=0,
    )
    def build():
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.conv_tol = 1.0e-12
        mf.conv_tol_grad = 1.0e-9
        mf.grids.level = 4
        mf.max_cycle = 400
        mf.verbose = 0
        return mf

    mf = build()
    energy = mf.kernel()
    if not mf.converged:
        # 开壳层小体系偶尔会卡在 DIIS 上：加能级移动再试，最后用二阶 SCF 兜底。
        print(f"  {species}/{functional}: DIIS 未收敛，试能级移动")
        mf = build()
        mf.level_shift = 0.5
        mf.max_cycle = 800
        energy = mf.kernel()
    if not mf.converged:
        print(f"  {species}/{functional}: 能级移动仍未收敛，试 Newton")
        mf = build()
        energy = mf.newton().kernel()
    if not mf.converged:
        # 不编数字：把失败如实写进 CSV（能量 nan + note），Rust 侧的测试跳过这些行。
        print(f"  {species}/{functional}: PySCF UKS 三种策略都没收敛 -> 记 nan")
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            xc,
            note + "；PySCF UKS 未收敛（DIIS / 能级移动 / Newton 都失败），不冒充参考值",
        )
    # ⟨S²⟩：UKS 用 (P_α − P_β) 的解析值
    dm = mf.make_rdm1()
    from pyscf.scf import uhf as pyscf_uhf

    # ⟨S²⟩：直接用 P_α/P_β 的公式（PySCF 的 spin_square 版本间签名不同）
    overlap = mol.intor_symmetric("int1e_ovlp")
    pa_s = dm[0] @ overlap
    pb_s = dm[1] @ overlap
    s_z = 0.5 * (np.trace(pa_s) - np.trace(pb_s))
    ss = s_z * (s_z + 1.0) + np.trace(pb_s) - np.trace(pa_s @ pb_s)
    n_alpha = float(np.trace(dm[0] @ mol.intor_symmetric("int1e_ovlp")))
    n_beta = float(np.trace(dm[1] @ mol.intor_symmetric("int1e_ovlp")))
    return energy, float(ss), n_alpha, n_beta, xc, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = []
    for species in MOLECULES:
        for functional in FUNCTIONALS:
            energy, s2, n_alpha, n_beta, xc, note = run_uks(species, functional)
            if energy != energy:  # NaN
                print(f"{species:3s} {functional:6s}: 未收敛（记 nan）")
            else:
                print(
                    f"{species:3s} {functional:6s}: E = {energy:.10f}  <S^2> = {s2:.9f}  "
                    f"n_alpha = {n_alpha:.0f}  n_beta = {n_beta:.0f}  (pyscf xc = {xc})"
                )
            rows.append(
                dict(
                    species=species,
                    functional=functional,
                    basis="sto-3g",
                    multiplicity=MOLECULES[species]["multiplicity"],
                    energy_hartree=f"{energy:.12f}",
                    s_squared=f"{s2:.12f}",
                    n_alpha=f"{n_alpha:.0f}",
                    n_beta=f"{n_beta:.0f}",
                    tolerance_hartree=TOLERANCES[species],
                    source_url="PySCF 2.14.0（libxc）",
                    note=note,
                )
            )

    if args.check:
        print("--check：不写 CSV")
        return 0

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = [
        "species",
        "functional",
        "basis",
        "multiplicity",
        "energy_hartree",
        "s_squared",
        "n_alpha",
        "n_beta",
        "tolerance_hartree",
        "source_url",
        "note",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
