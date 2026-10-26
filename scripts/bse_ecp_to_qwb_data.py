#!/usr/bin/env python3
"""把 BSE 导出的 **ECP 参数**转成 qwb-data 的 K1 `ecp` YAML。

project.md §8.1/§8.2 允许「BSE 网站导出 + 一手文献核对」。这个脚本是 ECP 的唯一转录路径：
数字来自 BSE 导出文件（`ecp_potentials`），不手抄，因此 `tests/d07_lanl2dz.rs` 才能对导出
逐项 diff。

用法（在仓库根）：

    python scripts/bse_ecp_to_qwb_data.py \\
      --input qwb-core/crates/qwb-registry/tests/fixtures/bse/lanl2dz.json \\
      --output qwb-data/ecp/lanl2dz-cu.yaml \\
      --name lanl2dz-cu --element Cu \\
      --title "..." --doi 10.1063/1.448975 --created 2026-10-26T14:00:00Z

映射规则（BSE → K1）：

* `ecp_electrons` → `n_core_electrons`；`z_eff = Z − n_core`（K1 的校验器会复核）。
* BSE 的每个 `ecp_potentials[]` 块是一个角量子数通道 `l`，块内的三元组
  `(r_exponent, gaussian_exponent, coefficient)` 就是 K1 的 `EcpTerm{power, exponent, coefficient}`。
* **局域通道 = 该元素给出的最大 `l`**（EMSL/Stuttgart 的通行约定：其它 `l` 走
  `V_ECP = V_local + Σ_{l<L}(V_l − V_local)P_l`）。这一点写进 `local_part.id` 与 `notes`，
  不让读者猜。
* `r_power_convention` 按 BSE 的定义写 `"r^n"`：BSE/EMSL 的 `r_exponents` 就是
  `V(r) = Σ c r^n exp(−α r²)` 里的 `n`（与 Stuttgart 原始表格的 `r^(n-2)` 差 2 次，
  §8.2 要求这个陷阱显式声明）。

输出**不含** `sha256`：打完 YAML 后跑 `cargo run -q -p qwb-registry --bin qwb-stamp-data -- <file>`。
"""

from __future__ import annotations

import argparse
import json
import sys

ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
}

ATOMIC_NUMBER = {symbol: z for z, symbol in ELEMENT_SYMBOLS.items()}


def num(text: str) -> float:
    return float(text)


def fmt(value: float) -> str:
    return repr(float(value))


def render(args: argparse.Namespace, element: dict, terms_by_l: dict[int, list[dict]]) -> str:
    local_l = max(terms_by_l)
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
    add(f"element: {args.element}")
    add(f"n_core_electrons: {args.n_core_electrons}")
    add(f"z_eff: {args.z_eff}")
    add('r_power_convention: "r^n"')
    add("local_part:")
    add(f'  id: "l={local_l} (BSE local channel: the highest l in the export)"')
    add("  terms:")
    for term in terms_by_l[local_l]:
        add(f"    - power: {term['power']}")
        add(f"      exponent: {fmt(term['exponent'])}")
        add(f"      coefficient: {fmt(term['coefficient'])}")
    add("nonlocal_parts:")
    for l in sorted(terms_by_l):
        if l == local_l:
            continue
        add(f"  - l: {l}")
        add(f'    id: "l={l} (BSE projector channel)"')
        add("    terms:")
        for term in terms_by_l[l]:
            add(f"      - power: {term['power']}")
            add(f"        exponent: {fmt(term['exponent'])}")
            add(f"        coefficient: {fmt(term['coefficient'])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="BSE JSON 导出文件（含 ecp_potentials）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--element", required=True, help="元素符号（一个文件一个元素，K1 的 Ecp 就是单元素）")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--title", required=True)
    parser.add_argument("--doi", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--spdx", default="NOASSERTION")
    parser.add_argument("--commercial-ok", dest="commercial_ok", action="store_true", default=False)
    parser.add_argument("--redistribute-ok", dest="redistribute_ok", action="store_true", default=False)
    parser.add_argument("--created", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if args.element not in ATOMIC_NUMBER:
        raise SystemExit(f"未知元素 {args.element!r}")
    z = ATOMIC_NUMBER[args.element]
    entry = data["elements"].get(str(z))
    if entry is None:
        raise SystemExit(f"导出文件里没有 {args.element}（Z={z}）")
    potentials = entry.get("ecp_potentials") or []
    if not potentials:
        raise SystemExit(f"{args.element} 在导出文件里没有 ecp_potentials")

    args.n_core_electrons = int(entry["ecp_electrons"])
    args.z_eff = float(z - args.n_core_electrons)

    terms_by_l: dict[int, list[dict]] = {}
    for block in potentials:
        if block.get("ecp_type") != "scalar_ecp":
            raise SystemExit(f"unsupported ecp_type: {block.get('ecp_type')!r}")
        angular = [int(l) for l in block["angular_momentum"]]
        if len(angular) != 1:
            raise SystemExit(f"ecp block covers several l: {angular}")
        l = angular[0]
        powers = [int(n) for n in block["r_exponents"]]
        exponents = [num(e) for e in block["gaussian_exponents"]]
        columns = block["coefficients"]
        if len(columns) != 1:
            # 每个 l 只有一组系数；多组会意味着"自旋相关 ECP"，那是另一个数据形态。
            raise SystemExit(f"l={l}: {len(columns)} coefficient columns (expected 1)")
        coefficients = [num(c) for c in columns[0]]
        if not (len(powers) == len(exponents) == len(coefficients)):
            raise SystemExit(f"l={l}: powers/exponents/coefficients 长度不一致")
        # §8.2 的陷阱：BSE 用 r^n，Stuttgart 原始表用 r^(n-2)。这里显式检查常见形态。
        if any(power < -2 or power > 2 for power in powers):
            raise SystemExit(f"l={l}: 出现意外的 r 幂次 {powers}（BSE 约定应为 -2..2）")
        terms_by_l[l] = [
            {"power": power, "exponent": exponent, "coefficient": coefficient}
            for power, exponent, coefficient in zip(powers, exponents, coefficients)
        ]

    text = render(args, entry, terms_by_l)
    import os

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(
        f"wrote {args.output}: {args.element} n_core={args.n_core_electrons} "
        f"z_eff={args.z_eff:g}, channels l={sorted(terms_by_l)} "
        f"(local = {max(terms_by_l)})"
    )
    print("next: cargo run -q -p qwb-registry --bin qwb-stamp-data -- <that file>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
