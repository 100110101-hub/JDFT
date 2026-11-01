"""DA-05：把 SMD 的 4 项非极性参数写进 8 个溶剂文件（从 MNSOL 数据表转录）。

数据来源（逐值可核）：
* **Minnesota Solvation Database (MNSOL)**，`https://comp.chem.umn.edu/solvation/mnsddb.pdf`
  —— 表里的 `[n, n25, alpha, beta, gamma, epsilon, phi, psi]`，与
  `pyscf/solvent/_solvent_data.py` 的 `solvent_db` 逐字一致（本脚本的值就是从那里抄的，
  并已用 ε 与 n 和我们既有溶剂文件交叉核对过）。
* SMD 原文：A. V. Marenich, C. J. Cramer, D. G. Truhlar, J. Phys. Chem. B 113, 6378 (2009),
  `doi:10.1021/jp810292n` —— **四项非极性项的公式**（γ、φ²、ψ² 与 Abraham α/β/n）出自这里。

单位：MNSOL 的 `gamma` 是 **cal/(mol·Å²)**（甲醇 31.77 ⇒ 22.07 mN/m），本脚本按原文单位
落盘（字段名 `gamma_cal_per_mol_angstrom2`），不做单位换算 —— 换了单位就再也对不上原表。

水的 γ/φ²/ψ² 在表里是 **−1 占位**（SMD 对水走单独的 CDS 处理），所以
`nonpolar_terms_available: false` 是**有意义的状态**，不是 0。

用法：
    .venv/Scripts/python.exe -X utf8 scripts/gen_smd_params.py           # 写进溶剂文件
    .venv/Scripts/python.exe -X utf8 scripts/gen_smd_params.py --check    # 只校验
"""

from __future__ import annotations

import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVENTS = os.path.join(ROOT, "qwb-data", "solvent")

#: MNSOL / SMD 2009 的非极性参数：`n(20°C)`, `n(25°C)`, `alpha`, `beta`,
#: `gamma [cal/(mol·Å²)]`, `epsilon`, `phi^2`, `psi^2`。
#:
#: 这八个溶剂是最初 DA-04 落地的八个（水/甲醇/乙醇/乙腈/DMSO/氯仿/甲苯/THF），
#: 键名对应 `qwb-data/solvent/*.yaml` 的文件名主干。
ROWS = {
    "water": [1.3328, 1.3323, 0.82, 0.35, -1.0, 78.3553, -1.0, -1.0],
    "methanol": [1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0],
    "ethanol": [1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0],
    "acetonitrile": [1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0],
    "dmso": [1.4783, 1.4783, 0.00, 0.88, 61.78, 46.826, 0.0, 0.0],
    "chloroform": [1.4459, 1.4431, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75],
    "toluene": [1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0],
    "thf": [1.4050, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0],
}

SOURCE = (
    "MNSOL (Minnesota Solvation Database, https://comp.chem.umn.edu/solvation/mnsddb.pdf) "
    "row [n20, n25, alpha, beta, gamma, epsilon, phi^2, psi^2]; nonpolar terms per "
    "Marenich-Cramer-Truhlar, J. Phys. Chem. B 113, 6378 (2009), doi:10.1021/jp810292n. "
    "gamma is in cal/(mol*Angstrom^2) as published; -1 marks a placeholder (water uses "
    "the separate CDS treatment, so its nonpolar terms are unavailable)."
)


def fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e16:
        return f"{value:.1f}"
    return repr(value)


def smd_block(name: str) -> str:
    n20, n25, alpha, beta, gamma, _epsilon, phi, psi = ROWS[name]
    available = gamma > 0.0 and phi >= 0.0 and psi >= 0.0
    lines = [
        "smd:",
        f"  gamma_cal_per_mol_angstrom2: {fmt(gamma) if gamma > 0.0 else 'null'}",
        f"  phi_sq: {fmt(phi) if phi >= 0.0 else 'null'}",
        f"  psi_sq: {fmt(psi) if psi >= 0.0 else 'null'}",
        f"  abraham_alpha: {fmt(alpha)}",
        f"  abraham_beta: {fmt(beta)}",
        f"  refractive_index_20c: {fmt(n20)}",
        f"  refractive_index_25c: {fmt(n25)}",
        f"  nonpolar_terms_available: {'true' if available else 'false'}",
        f'  source: "{SOURCE}"',
    ]
    return "\n".join(lines)


def replace_block(text: str, name: str) -> str:
    """把文件里**已有的** `smd:` 块原地换掉。

    注意：八个溶剂文件在 DA-04/DA-05 的前期工作里**已经带一个 `smd:` 块**
    （只有 `nonpolar_terms_available` 与 `source`，因为当时表还没拿到）。所以这里必须
    「替换这一段」而不是「从头追加」—— 早先按 `smd:` 切分再拼接的写法把块后面的
    `scale`/`atomic_radii` 一起截掉了（`qwb data verify` 立刻报 missing field
    `atomic_radii`，这正是它该抓的那类错）。
    """
    lines = text.split("\n")
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "smd:"), None
    )
    assert start is not None, f"{name}: no `smd:` block to replace"
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1
    return "\n".join(lines[:start] + smd_block(name).split("\n") + lines[end:])


def patch(path: str, name: str) -> None:
    text = io.open(path, encoding="utf-8").read()
    io.open(path, "w", encoding="utf-8", newline="\n").write(replace_block(text, name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    problems = []
    for name in ROWS:
        path = os.path.join(SOLVENTS, f"{name}.yaml")
        if not os.path.isfile(path):
            problems.append(f"{name}: missing {path}")
            continue
        if args.check:
            # 语义比较：`qwb-stamp-data` 会重排 YAML，逐字比对会假红。
            import yaml  # noqa: PLC0415

            current = yaml.safe_load(io.open(path, encoding="utf-8").read()).get("smd")
            expected = yaml.safe_load(smd_block(name))["smd"]
            for key, value in expected.items():
                if current is None or key not in current:
                    problems.append(f"{name}: smd.{key} is missing")
                elif current[key] != value:
                    problems.append(f"{name}: smd.{key} = {current[key]!r} != {value!r}")
            for key in (current or {}):
                if key not in expected:
                    problems.append(f"{name}: smd.{key} is not produced by the generator")
            continue
        patch(path, name)
        print(f"patched {os.path.relpath(path, ROOT)}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print(f"ok: {len(ROWS)} solvent file(s) carry the SMD nonpolar parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
