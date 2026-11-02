"""SMD 的**整条非极性 + 静电**外部尺子：PySCF 的 SMD（同一几何、同一基组）。

本项目与 PySCF 都实现了「SMD 半径 + C-PCM 型表面方程 + G_CDS」，但两边的表面表示不同
（PySCF 用 SWIG 解析表面，我们用 Fibonacci tesserae + 球内侧切换带），所以：

* **G_CDS 应当对得很紧**（它只依赖 SASA 与描述符，两边算的都是同一个式子）；
* **静电那一半会差几百分之一到千分之几 Ha**（表面表示不同），这个差值必须**量出来**并写进
  `docs/error-budget.md`，而不是拿一个宽松容差把它盖住。

产物 `qwb-core/crates/qwb-engine/tests/reference/smd_water.csv` 每行带 source_doi/source_url。

用法：

    python scripts/pyscf_smd_reference.py            # 重新生成 CSV
    python scripts/pyscf_smd_reference.py --check    # 只比对，不写盘
"""

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import PYSCF_DOI, PYSCF_URL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT, "qwb-core", "crates", "qwb-engine", "tests", "reference", "smd_water.csv"
)

#: 与 `qwb-core/examples/h2o_smd.yaml` 逐字一致的几何（Å）。
WATER = [
    ("O", (0.0, 0.0, 0.11779)),
    ("H", (0.0, 0.75545, -0.47116)),
    ("H", (0.0, -0.75545, -0.47116)),
]

HARTREE_PER_KCAL = 627.5094740631

HEADER = ["system", "solvent", "method", "basis", "quantity", "value", "unit", "source_doi", "source_url"]


def rows():
    from pyscf import gto, scf
    from pyscf.solvent import smd

    mol = gto.M(atom=WATER, basis="sto-3g", unit="Angstrom", verbose=0)
    gas = scf.RHF(mol).run(conv_tol=1e-11)
    solvent = smd.SMD(mol, "water")
    solvent.build()
    cds = float(solvent.get_cds())

    mf = scf.RHF(mol).SMD(solvent_obj=solvent)
    mf.conv_tol = 1e-11
    mf.run()

    e_gas = float(gas.e_tot)
    e_smd = float(mf.e_tot)
    total = e_smd - e_gas
    electrostatic = total - cds

    out = [
        ["water", "water", "hf", "sto-3g", "energy_gas_hartree", f"{e_gas:.10f}", "hartree"],
        ["water", "water", "hf", "sto-3g", "energy_smd_hartree", f"{e_smd:.10f}", "hartree"],
        ["water", "water", "hf", "sto-3g", "cds_kcal_per_mol", f"{cds * HARTREE_PER_KCAL:.6f}", "kcal/mol"],
        ["water", "water", "hf", "sto-3g", "electrostatic_kcal_per_mol",
         f"{electrostatic * HARTREE_PER_KCAL:.6f}", "kcal/mol"],
        ["water", "water", "hf", "sto-3g", "total_kcal_per_mol", f"{total * HARTREE_PER_KCAL:.6f}", "kcal/mol"],
    ]
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
