#!/usr/bin/env python3
"""把 BSE（Basis Set Exchange）导出的 JSON 转成 qwb-data 的 K1 基组 YAML。

project.md §8.1 允许「BSE 网站导出」，禁止「从其他项目代码仓库复制文件」。
本脚本是**唯一**的转录路径：数字来自 BSE 导出文件，不手抄，因此
`tests/d03_sto3g_vs_bse.rs` 才能做逐项 diff。

用法（在仓库根）：

    python scripts/bse_to_qwb_data.py \\
      --input  qwb-core/crates/qwb-registry/tests/fixtures/bse/sto-3g.json \\
      --output qwb-data/basis/sto-3g.yaml \\
      --name sto-3g --version 1.0.0 \\
      --title "Self-Consistent Molecular-Orbital Methods. I. ..." \\
      --doi 10.1063/1.1672392 \\
      --created 2026-10-22T23:16:10Z \\
      --notes "..."

输出**不含** `sha256`：打完 YAML 后跑 `qwb-stamp-data` 打戳（G2 要求工具生成）。

映射规则（BSE → K1）：
- 一个 `electron_shells[]` 条目 → 若干个 `Shell`：`angular_momentum` 有几项就拆几个
  （`[0,1]` 的 2sp 壳层拆成 l=0 与 l=1 两个 Shell，共用指数、各取一列系数）。
- `angular_momentum` 只有一项但系数有多列 → 通用收缩（一列一个收缩函数）。
- `coefficients` 的列序与 `angular_momentum` 的顺序一一对应。
- 系数符号按 BSE 原样保留（2s 的内层系数本来就是负的）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

# BSE 用原子序数当键；K1 用元素符号。
ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
}


def num(text: str) -> float:
    """BSE 的数字是字符串（如 `0.3425250914E+01`）。"""
    return float(text)


def fmt(value: float) -> str:
    """最短的精确往返表示（YAML 里人能读、且不丢精度）。"""
    return repr(float(value))


def split_shells(entry: dict) -> list[dict]:
    """把一个 BSE electron_shell 拆成 K1 的 Shell 列表。"""
    if entry.get("function_type") not in ("gto", "gto_spherical"):
        raise SystemExit(f"unsupported function_type: {entry.get('function_type')!r}")
    # BSE 的 `region` 只是个标签：`valence` 表示这是价基（LANL2DZ 这类 ECP 基组全都这么标），
    # `unknown` 出现在老导出里。K1 的 `Shell` 没有"区域"概念，所以这几个都按整壳层转录。
    # 其它取值（例如 `core`、`semicore`）意味着导出被裁剪过，必须显式拒绝。
    region = entry.get("region")
    if region not in (None, "", "valence", "unknown"):
        raise SystemExit(f"unsupported region: {region!r} (only full-range/valence shells)")

    exps = [num(e) for e in entry["exponents"]]
    columns = [[num(c) for c in column] for column in entry["coefficients"]]
    angular = [int(l) for l in entry["angular_momentum"]]

    for column in columns:
        if len(column) != len(exps):
            raise SystemExit("coefficient column length != exponent count")

    shells: list[dict] = []
    if len(columns) == len(angular):
        # 业界最常见的形状：每个角动量一列系数。
        for l, column in zip(angular, columns):
            shells.append({"l": l, "exponents": exps, "columns": [column]})
    elif len(angular) == 1 and len(columns) > 1:
        # 通用收缩：同一个 l、多列系数。
        shells.append({"l": angular[0], "exponents": exps, "columns": columns})
    else:
        raise SystemExit(
            f"cannot map angular_momentum={angular} with {len(columns)} coefficient columns"
        )
    return shells


def render(args: argparse.Namespace, per_element: "OrderedDict[str, list[dict]]") -> str:
    lines: list[str] = []
    add = lines.append

    add(f"name: {args.name}")
    add(f"version: {args.version}")
    add(f"notes: {json.dumps(args.notes, ensure_ascii=False)}")
    add("citation:")
    add(f"  - title: {json.dumps(args.title, ensure_ascii=False)}")
    if args.doi:
        add(f"    doi: {json.dumps(args.doi)}")
    if args.url:
        add(f"    url: {json.dumps(args.url)}")
    add("license:")
    add(f"  spdx: {args.spdx}")
    add(f"  commercial_ok: {str(args.commercial_ok).lower()}")
    add(f"  redistribute_ok: {str(args.redistribute_ok).lower()}")
    add(f"created: {args.created}")
    add("elements: [" + ", ".join(per_element.keys()) + "]")
    add(f"angular: {args.angular}")
    add(f"normalization: {args.normalization}")
    add(f"role: {args.role}")
    add("shells:")

    for symbol, shells in per_element.items():
        add(f"  - element: {symbol}")
        add("    shells:")
        for shell in shells:
            add(f"      - l: {shell['l']}")
            add("        exponents: [" + ", ".join(fmt(e) for e in shell["exponents"]) + "]")
            add("        coefficients:")
            for column in shell["columns"]:
                add("          - [" + ", ".join(fmt(c) for c in column) + "]")
            add(f"        contraction_type: {args.contraction_type}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="BSE JSON 导出文件")
    parser.add_argument("--output", required=True, help="输出的 K1 YAML 路径")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--title", required=True)
    parser.add_argument("--doi", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--spdx", default="Apache-2.0")
    parser.add_argument("--commercial-ok", dest="commercial_ok", action="store_true", default=True)
    parser.add_argument(
        "--no-commercial",
        dest="commercial_ok",
        action="store_false",
        help="原始条款没写清商用许可时用这个（§14 #20 的严格读法）",
    )
    parser.add_argument(
        "--no-redistribute",
        dest="redistribute_ok",
        action="store_false",
        default=True,
        help="外部来源条款未确认时用这个（§14 #20）",
    )
    parser.add_argument("--created", required=True)
    parser.add_argument("--angular", default="spherical", choices=["spherical", "cartesian"])
    parser.add_argument("--normalization", default="l2", choices=["l2", "none"])
    parser.add_argument("--role", default="orbital", choices=["orbital", "ri-j", "ri-jk", "ecp-fit"])
    parser.add_argument("--contraction-type", dest="contraction_type", default="segmented")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    # `function_types` 是整份导出的清单：ECP 基组（LANL2DZ 等）会同时带 `scalar_ecp`，
    # 那是 ECP 参数（由 scripts/bse_ecp_to_qwb_data.py 转录），不影响基组转录。
    # 但 `electron_shells` 里只允许 gto / gto_spherical —— 别的（如 `sto`）不能假装成 gto。
    orbital_types = {"gto", "gto_spherical"}
    seen = set(data.get("function_types") or [])
    unexpected = seen - orbital_types - {"scalar_ecp"}
    if unexpected:
        raise SystemExit(f"unsupported function_types: {sorted(unexpected)}")
    for key, element in data["elements"].items():
        for entry in element.get("electron_shells") or []:
            if entry.get("function_type") not in orbital_types:
                raise SystemExit(
                    f"element {key}: unsupported electron_shell function_type "
                    f"{entry.get('function_type')!r}"
                )
    if "gto_spherical" in seen and args.angular != "spherical":
        raise SystemExit("BSE export contains spherical shells; use --angular spherical")

    elements = data["elements"]
    per_element: "OrderedDict[str, list[dict]]" = OrderedDict()
    for key in sorted(elements, key=int):
        z = int(key)
        symbol = ELEMENT_SYMBOLS.get(z)
        if symbol is None:
            raise SystemExit(f"element Z={z} outside the H..Kr table")
        shells: list[dict] = []
        for entry in elements[key]["electron_shells"]:
            shells.extend(split_shells(entry))
        shells.sort(key=lambda shell: shell["l"])
        if not shells:
            raise SystemExit(f"element Z={z} has no shells")
        per_element[symbol] = shells

    text = render(args, per_element)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    total = sum(len(s) for s in per_element.values())
    print(
        f"wrote {args.output}: {len(per_element)} elements "
        f"({', '.join(per_element)}), {total} shells"
    )
    print("next: cargo run -q -p qwb-registry --bin qwb-stamp-data -- <that file>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
