#!/usr/bin/env python3
"""用 **PySCF** 生成开壳层参考值（UHF / ROHF / RHF，只做黑盒对照，§14 #12）。

为什么需要它：M1（开壳层）的验收要判 `⟨S²⟩` 与能量，而仓库现有的参考集全是**闭壳层**
（CCCBDB 的 STO-3G 七个体系）。开壳层的判据不能靠"看起来收敛了"——必须有第二个代码的
同一个数，否则 UHF 的 bug（漏掉交换项、自旋通道错位）会安静地算出"合理"的能量。

这个脚本只**产数字**：不把 PySCF 的任何代码/数据拷进 `qwb-data`，我们的实现也不依赖它。
生成的 CSV 进 `qwb-core/crates/qwb-scf/tests/reference/`，每行带 `source_url`。

用法（WSL 里跑）：

    python3 scripts/pyscf_open_shell_reference.py --check   # 只校验已入库的 CSV 与 PySCF 一致
    python3 scripts/pyscf_open_shell_reference.py           # 打印 CSV（人看/重生成用）

依赖：`python3 -m pip install --user pyscf`（WSL；本机 2.14.0）。
"""

from __future__ import annotations

import argparse
import sys

#: 与 Rust 测试**逐字一致**的几何（Å）。改这里就必须同步改
#: `qwb-core/crates/qwb-scf/tests/sc01_uhf.rs`。
SPECIES: list[dict] = [
    {
        "name": "H2",
        "atoms": [("H", (0.0, 0.0, -0.37)), ("H", (0.0, 0.0, 0.37))],
        "charge": 0,
        "multiplicity": 1,
        "methods": ["rhf", "uhf"],
        "guesses": ["atom"],
        "note": "closed shell: UHF must reproduce RHF exactly (degeneracy check)",
    },
    {
        "name": "O2",
        "atoms": [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.208))],
        "charge": 0,
        "multiplicity": 3,
        "methods": ["uhf", "rohf"],
        # O2 的 UHF 有**两个解**：hcore 初猜落到 +0.255 Ha 的较高解，原子密度初猜落到变分极小。
        # 两个都记下来 —— 前者证明"同一初猜下与独立实现等价"，后者是真正的目标值。
        "guesses": ["atom", "hcore"],
        "note": "triplet; UHF is spin-contaminated by the degenerate pi*, ROHF is spin-pure",
    },
    {
        "name": "OH",
        "atoms": [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.970))],
        "charge": 0,
        "multiplicity": 2,
        "methods": ["uhf", "rohf"],
        "guesses": ["atom"],
        "note": "doublet radical; the classic UHF spin-contamination case",
    },
]

BASIS = "sto-3g"
#: 与 `h2_sto3g.csv` 等既有参考集一致的容差（1e-6 是"我们的实现 vs PySCF"的目标，
#: 不是文献不确定度）。
TOLERANCE_HARTREE = 1.0e-6

HEADER = f"""# 开壳层参考集（M1 / SC-01）。**外部来源**：PySCF 2.14.0 黑盒计算（§14 #12）。
#
# 生成：`python3 scripts/pyscf_open_shell_reference.py`（WSL；需要 pyscf）。
# 校验：`python3 scripts/pyscf_open_shell_reference.py --check`（门禁**不**跑它：
#       PySCF 只装在 WSL 里，Windows 侧没有；数值已经钉在这份 CSV 里）。
#
# basis 一律 {BASIS}；几何（Å）与 `qwb-core/crates/qwb-scf/tests/sc01_uhf.rs` 逐字一致：
#   H2  z = ±0.37           （闭壳层退化：UHF 必须复现 RHF）
#   O2  z = 0, 1.208        （三重态；UHF 有自旋污染，ROHF 是纯自旋态）
#   OH  z = 0, 0.970        （双峰自由基）
#
# 列：species,method,basis,guess,multiplicity,energy_hartree,s_squared,tolerance_hartree,source_url,note
species,method,basis,guess,multiplicity,energy_hartree,s_squared,tolerance_hartree,source_url,note
"""

SOURCE_URL = "https://pyscf.org/"


def rows() -> list[str]:
    from pyscf import gto, scf

    out: list[str] = []
    for entry in SPECIES:
        spin = entry["multiplicity"] - 1
        mol = gto.M(
            atom=[(symbol, position) for symbol, position in entry["atoms"]],
            unit="Angstrom",
            basis=BASIS,
            charge=entry["charge"],
            spin=spin,
            verbose=0,
        )
        for method in entry["methods"]:
            for guess in entry["guesses"]:
                mean_field = {
                    "rhf": scf.RHF,
                    "uhf": scf.UHF,
                    "rohf": scf.ROHF,
                }[method](mol)
                mean_field.conv_tol = 1.0e-11
                mean_field.conv_tol_grad = 1.0e-9
                mean_field.init_guess = guess
                energy = mean_field.kernel()
                s_squared, _ = mean_field.spin_square()
                out.append(
                    "{},{},{},{},{},{:.12f},{:.12f},{:g},{},{}".format(
                        entry["name"],
                        method.upper(),
                        BASIS,
                        guess,
                        entry["multiplicity"],
                        float(energy),
                        float(s_squared),
                        TOLERANCE_HARTREE,
                        SOURCE_URL,
                        entry["note"] + " (PySCF black-box reference, not our implementation)",
                    )
                )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验入库 CSV 与本脚本一致")
    parser.add_argument("--csv", default=None, help="要校验的 CSV 路径（配合 --check）")
    args = parser.parse_args(argv)

    body = HEADER + "\n".join(rows()) + "\n"
    if not args.check:
        sys.stdout.write(body)
        return 0

    from pathlib import Path

    target = Path(args.csv or "qwb-core/crates/qwb-scf/tests/reference/uhf_open_shell.csv")
    if not target.is_file():
        print(f"missing {target}", file=sys.stderr)
        return 1
    existing = target.read_text(encoding="utf-8")
    if existing != body:
        print(f"{target} is out of date; rerun this script and re-commit", file=sys.stderr)
        for old, new in zip(existing.splitlines(), body.splitlines()):
            if old != new:
                print(f"  - {old}\n  + {new}", file=sys.stderr)
                break
        return 1
    print(f"ok: {target} matches PySCF")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
