#!/usr/bin/env python3
"""把已验证的 CDS 数据（`_smd_cds_staging/cds_atomic.json`）落成 K1 形状的数据文件
`qwb-data/solvent/smd_atomic.yaml`（SV-03 的加性数据块）。

**为什么是"元素级的一个文件"而不是抄进 8 个溶剂文件**：见 `docs/choices.md` 一百六十二 ——
σ 表、CDS vdW 半径、面积约定都与溶剂无关（唯一的溶剂依赖是 O 的半径通过 Abraham α 进入，那属于
"SMD 参数化半径"，与本文件是**两套**半径，故不混）。抄 8 份意味着 8 处保持一致。

本文件**只做搬运**：数值全部来自 `cds_atomic.json`（它由 `gen_smd_cds.py` 从上游 `mnsol.F`
逐字提取、并被 `validate_smd_cds.py` 对 PySCF 的 `libsolvent` 验证到 1e-5/1e-7）。
`--check` 做语义比较（逐值相等，容差 0）。

用法：
    python3 scripts/gen_smd_atomic.py            # 生成/覆盖 qwb-data/solvent/smd_atomic.yaml
    python3 scripts/gen_smd_atomic.py --check    # 与磁盘语义比较（需要 PyYAML）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(ROOT, "qwb-data", "solvent", "_smd_cds_staging", "cds_atomic.json")
# 目的地**暂时**在 staging：`qwb-data/solvent/*.yaml` 会被当作 `Solvent` 校验，而 SMD 的
# 元素级块需要自己的 K1 类型（K1 只做加性扩展：新 `DataKind` + 新 schema + registry 的
# `smd_atomics` 段）。与 D4 的 `d4-staged.yaml` 同一处境，所以同一个做法。
OUTPUT = os.path.join(ROOT, "qwb-data", "solvent", "_smd_cds_staging", "smd_atomic.yaml")

NOTES = (
    "SMD 的 CDS（cavity dispersion solvent structure）项所需的元素级数据，从 PySCF 2.14.0 的 "
    "pyscf/lib/solvent/mnsol/mnsol.F 逐字提取（Apache-2.0），并用同一份 PySCF 的 libsolvent "
    "验证：水相 sigma 最差 1.615e-5 cal/(mol A^2)、五个非水溶剂的 E_CDS 最差 3.797e-7 kcal/mol。"
    "三块内容：(1) CDS 的 vdW 半径（BONDI 表，H 取 Bondi 1964、其余取 Mantina 等 CRC 表）——"
    "注意它与 SMD 的“参数化半径”不是同一套（后者 C 1.85 / N 1.89 / O 依赖 Abraham alpha）；"
    "(2) 水相的逐元素 sigma（CSSIGM = 0）；(3) 非水相 sigma 由 sigma_N/sigma_A/sigma_B 与溶剂的 "
    "Abraham 描述符线性组合而成（外加与元素无关的 sigma_mol 常数项）。面积约定 "
    "A = 4*pi*(R + 0.4 A)^2 是反推并验证出来的（不是从论文抄的）：用 4*pi*R^2 时 sigma 会逐元素"
    "偏大 1.48~1.78 倍，加 0.4 A 后与官方回到 1e-7 相对以内。"
)


def build() -> dict:
    with open(STAGING, encoding="utf-8") as handle:
        staging = json.load(handle)
    water = staging["cases"]["water"]
    naq = staging["cases"]["nonaqueous"]
    radii = staging["cds_vdw_radii_angstrom"]["values"]
    element_symbols = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
    ]
    data = {
        "name": "smd-atomic",
        "version": "1.0.0",
        "kind": "smd-atomic",
        "notes": NOTES,
        "citation": [
            {
                "title": (
                    "Universal solvation model based on solute electron density and on a "
                    "continuum model of the solvent defined by the bulk dielectric constant "
                    "and atomic surface tensions (SMD)"
                ),
                "doi": "10.1021/jp810292n",
            },
            {
                "title": "Atomic radii of the elements (Mantina et al., CRC Handbook 2010-2011)",
                "doi": "10.1021/j100785a001",
            },
        ],
        "license": {
            "spdx": "Apache-2.0",
            "commercial_ok": True,
            "redistribute_ok": True,
        },
        "created": "2026-11-12T00:00:00Z",
        "elements": element_symbols,
        "cds": {
            "area_convention": staging["cds_area_convention"]["formula"],
            "area_offset_angstrom": staging["cds_area_convention"]["offset_angstrom"],
            "vdw_radii_angstrom": {symbol: radii[index] for index, symbol in enumerate(element_symbols)},
            "water": {
                "sigma_cal_per_mol_angstrom2": dict(water["sigma"]),
                "hsigma_cal_per_mol_angstrom2": dict(water["hsigma"]),
            },
            "nonaqueous": {
                "sigma_formula": naq["sigma_formula"],
                "hsigma_formula": naq["hsigma_formula"],
                "constant_formula": naq["constant_formula"],
                "descriptor_names": list(naq["descriptor_names"]),
                "sigma_n": list(naq["raw_tables"]["sigma_n"]),
                "sigma_a": list(naq["raw_tables"]["sigma_a"]),
                "sigma_b": list(naq["raw_tables"]["sigma_b"]),
                "hsigma_n": list(naq["raw_tables"]["hsigma_n"]),
                "hsigma_a": list(naq["raw_tables"]["hsigma_a"]),
                "hsigma_b": list(naq["raw_tables"]["hsigma_b"]),
                "sigma_mol": list(naq["sigma_mol"]),
            },
        },
        "provenance": {
            "generator": "scripts/gen_smd_atomic.py",
            "upstream": [
                {
                    "file": staging["upstream"]["file"],
                    "url": staging["upstream"]["url"],
                    "sha256": staging["upstream"]["sha256"],
                }
            ],
        },
    }
    return data


def emit(value, indent: int = 0, out: list[str] | None = None) -> list[str]:
    """够用的 YAML 发射器（文件随后会被 `qwb-stamp-data` 重新序列化）。"""
    import math  # noqa: PLC0415

    out = [] if out is None else out
    pad = "  " * indent

    def scalar(item) -> str:
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, float):
            text = repr(item)
            if "e" in text or "E" in text:
                mantissa, _, exponent = text.partition("e" if "e" in text else "E")
                if "." not in mantissa:
                    mantissa += ".0"
                text = f"{mantissa}e{exponent}"
            elif "." not in text:
                text += ".0"
            return text
        if isinstance(item, int):
            return str(item)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False)
        raise SystemExit(f"unsupported scalar {type(item)}")

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                out.append(f"{pad}{key}:")
                emit(item, indent + 1, out)
            elif isinstance(item, list) and not item:
                out.append(f"{pad}{key}: []")
            elif isinstance(item, dict) and not item:
                out.append(f"{pad}{key}: {{}}")
            else:
                out.append(f"{pad}{key}: {scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                first = True
                for key, sub in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(sub, (dict, list)) and sub:
                        out.append(f"{prefix}{key}:")
                        emit(sub, indent + 2, out)
                    else:
                        out.append(f"{prefix}{key}: {scalar(sub)}")
            else:
                out.append(f"{pad}- {scalar(item)}")
    else:
        out.append(f"{pad}{scalar(value)}")
    return out


def compare(want, got, where: str = "") -> list[str]:
    problems = []
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: expected mapping"]
        for key, value in want.items():
            if key not in got:
                if key == "sha256" and where == "":
                    continue
                problems.append(f"{where}.{key}: missing")
                continue
            problems += compare(value, got[key], f"{where}.{key}")
        for key in got:
            if key not in want and not (key == "sha256" and where == ""):
                problems.append(f"{where}.{key}: unexpected")
    elif isinstance(want, list):
        if not isinstance(got, list) or len(want) != len(got):
            return [f"{where}: list length mismatch"]
        for index, (a, b) in enumerate(zip(want, got)):
            problems += compare(a, b, f"{where}[{index}]")
    elif isinstance(want, float):
        if isinstance(got, str):
            try:
                got = float(got)
            except ValueError:
                return [f"{where}: non-numeric {got!r}"]
        if not isinstance(got, (int, float)) or float(want) != float(got):
            problems.append(f"{where}: {want!r} vs {got!r}")
    elif want != got:
        problems.append(f"{where}: {want!r} vs {got!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build()
    body = "\n".join(emit(data)) + "\n"
    if args.check:
        import yaml  # noqa: PLC0415

        with open(OUTPUT, encoding="utf-8") as handle:
            on_disk = yaml.safe_load(handle)
        problems = compare(data, on_disk)
        if problems:
            print(f"{len(problems)} 处不一致：")
            for problem in problems[:10]:
                print("  -", problem)
            return 1
        print(f"ok: {OUTPUT} matches the generator (semantic, tolerance 0)")
        return 0
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    print(f"wrote {OUTPUT} ({len(body)} bytes)")
    print("下一步：qwb-core/target/debug/qwb-stamp-data.exe qwb-data/solvent/smd_atomic.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
