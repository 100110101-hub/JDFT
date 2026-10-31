"""ST-05 的外部参考：开壳层自由基在**核位置上**的自旋密度（PySCF UKS）。

判据的诚实读法：`A_iso` 与实验的比较需要实验原文（本轮没有取到可引用的），所以 ST-05 记
blocked；但"自旋密度这条流水线对不对"是**可以独立验证**的 —— 用 PySCF 的 UKS 算同一个
体系、同一个基组、同一个泛函，把每个核位置上的 `ρ_s = Σ (P_α − P_β) φ_μ φ_ν` 写出来，
两边逐核比。这比"和记忆里的文献值比"硬得多，也不会把自算值冒充文献值。

输出：qwb-core/crates/qwb-props/tests/reference/epr_spin_density.csv
列：species,basis,functional,nucleus_index,symbol,spin_density_au,note,source_doi,source_url
"""

import argparse
import os
import sys

import numpy as np
import reference_sources
import yaml
from pyscf import dft, gto

CSV_PATH = os.path.join(
    "qwb-core", "crates", "qwb-props", "tests", "reference", "epr_spin_density.csv"
)

FIELDS = (
    "species",
    "basis",
    "functional",
    "nucleus_index",
    "symbol",
    "spin_density_au",
    "note",
)

BOND = 1.079  # CH3 的 C-H 键长（Å，实验 r_e 量级）
ANGLE = np.radians(120.0)
METHYL = [
    ("C", [0.0, 0.0, 0.0]),
    ("H", [BOND, 0.0, 0.0]),
    ("H", [BOND * np.cos(ANGLE), BOND * np.sin(ANGLE), 0.0]),
    ("H", [BOND * np.cos(ANGLE), -BOND * np.sin(ANGLE), 0.0]),
]

OH = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.980])]


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


def spin_density_at_nuclei(atoms, basis_name, functional, multiplicity):
    # `gto.M(...)` 会立刻 build，所以 `spin` 必须在构造时就给（N_alpha - N_beta）。
    mol = gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        unit="Angstrom",
        spin=multiplicity - 1,
        verbose=0,
    )
    mol.basis = load_yaml_basis(basis_name)
    mol.cart = False
    mol.build()
    if functional == "uhf":
        from pyscf import scf

        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = functional
        mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.kernel()
    assert mf.converged, f"{functional}: SCF did not converge"
    dm = mf.make_rdm1()
    dm_alpha, dm_beta = dm[0], dm[1]
    spin = dm_alpha - dm_beta
    coords = mol.atom_coords(unit="Bohr")
    ao = dft.numint.eval_ao(mol, coords)  # (n_nuclei, nao)
    values = np.einsum("ni,ij,nj->n", ao, spin, ao)
    return values


def main(check_only: bool = False):
    rows = []
    for label, atoms, multiplicity, functional in [
        ("CH3", METHYL, 2, None),
        ("OH", OH, 2, None),
    ]:
        # `uhf` 行是给我们自己的 UHF 对照用的（快、可复现）；`pbe` 行留给将来 UKS 收敛问题
        # 解决后使用 —— 本轮我们自己的 UKS 在 CH3 上 200 步不收敛（实测 >10 min），
        # 所以那两行**暂时没有对应测试**，但留着参考数据无害。
        for basis, func in [("def2-svp", "uhf"), ("def2-svp", "pbe"), ("sto-3g", "uhf"), ("sto-3g", "svwn")]:
            values = spin_density_at_nuclei(atoms, basis, func, multiplicity)
            print(f"{label}/{basis}/{func}:")
            for index, (symbol, _) in enumerate(atoms):
                print(f"  {symbol}{index}: rho_s = {values[index]:+.6f}")
                rows.append(
                    dict(
                        species=label,
                        basis=basis,
                        functional=func,
                        nucleus_index=index,
                        symbol=symbol,
                        spin_density_au=f"{values[index]:.10f}",
                        note=(
                            "PySCF 2.14.0 scf.UHF 的 P_alpha-P_beta 在核位置的 AO 值缩并"
                            if func == "uhf"
                            else "PySCF 2.14.0 dft.UKS 的 P_alpha-P_beta 在核位置的 AO 值缩并"
                        ),
                    )
                )
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
