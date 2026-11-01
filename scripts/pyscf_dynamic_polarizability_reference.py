"""ST-02 的外部参考：**动态极化率** `α(ω)`（PySCF 2.14 的 TDHF 态 + 态求和）。

为什么参考用「PySCF 的态 + 同一条态求和公式」而不是"文献里的 α(ω) 数值"：

* 文献的 α(ω) 通常给的是实验/高精度理论值，与本项目的方法层级（HF 参考 + RPA + 小基组）
  **不可比**（§6 的容差又写的是「与文献差 < 10%」，两种读法天差地别，处置见
  `docs/status-v2.md` 的 ST-02 条目）；
* 方法匹配的参考能把「实现对不对」和「方法好不好」分开：PySCF 的 TDHF 态与本项目是同一个
  近似，态求和公式是极化率的定义。若两边差 1e-5，说明本项目的 RPA 态与跃迁偶极矩是对的。

输出：qwb-core/crates/qwb-response/tests/reference/dynamic_polarizability.csv
列：species,basis,omega_au,alpha_iso_au,lowest_excitation_hartree,note,source_doi,source_url
"""

import argparse
import os
import sys

import numpy as np
import reference_sources
import yaml
from pyscf import gto, scf, tdscf

CSV_PATH = os.path.join(
    "qwb-core",
    "crates",
    "qwb-response",
    "tests",
    "reference",
    "dynamic_polarizability.csv",
)

FIELDS = (
    "species",
    "basis",
    "omega_au",
    "alpha_iso_au",
    "lowest_excitation_hartree",
    "note",
)

WATER = """
O 0.000000 0.000000 0.117790
H 0.000000 0.755450 -0.471160
H 0.000000 -0.755450 -0.471160
"""

FREQUENCIES = (0.0, 0.05, 0.1, 0.2)
HARTREE_TO_EV = 27.211386245988


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


def run(species, geometry, basis_name):
    mol = gto.M(atom=geometry, unit="Angstrom", verbose=0)
    mol.basis = load_yaml_basis(basis_name)
    mol.cart = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    td = tdscf.rhf.TDHF(mf)
    td.nstates = mf.mo_occ.tolist().count(2) * (mol.nao - mf.mo_occ.tolist().count(2))
    td.kernel()
    energies = np.asarray(td.e)
    oscillators = np.asarray(td.oscillator_strength())
    lowest = float(energies.min())
    print(f"{species}/{basis_name}: {len(energies)} TDHF states, lowest = {lowest:.6f} Ha")
    rows = []
    for omega in FREQUENCIES:
        # α_iso(ω) = Σ_n f_n / (ω_n² − ω²)：由振子强度 f_n = (2/3) ω_n |μ_n|² 与
        # 态求和公式 α_iso(ω) = (1/3)Σ_n 2ω_n|μ_n|²/(ω_n²−ω²) 合并而来。
        denominator = energies**2 - omega**2
        assert np.all(denominator > 0.0), f"omega {omega} hits a pole"
        alpha = float(np.sum(oscillators / denominator))
        print(f"  omega = {omega:.2f}: alpha_iso = {alpha:.10f} a.u.")
        rows.append(
            dict(
                species=species,
                basis=basis_name,
                omega_au=f"{omega:.4f}",
                alpha_iso_au=f"{alpha:.10f}",
                lowest_excitation_hartree=f"{lowest:.10f}",
                note=(
                    "PySCF 2.14.0 RHF + tdscf.rhf.TDHF 的全部态，"
                    "α_iso(ω) = Σ_n f_n/(ω_n²−ω²)（态求和）"
                ),
            )
        )
    return rows


def main(check_only: bool = False):
    rows = []
    for basis_name in ("sto-3g", "def2-svp"):
        rows += run("H2O", WATER, basis_name)
    rows = reference_sources.stamp(rows)
    fields = list(FIELDS)
    status = reference_sources.check_csv(CSV_PATH, fields, rows)
    if status == 0 or check_only:
        return status
    reference_sources.write_csv(CSV_PATH, fields, rows)
    print(f"写入 {CSV_PATH}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="只重算并比对文件，不写盘（DA-08）")
    arguments = parser.parse_args()
    sys.exit(main(arguments.check))
