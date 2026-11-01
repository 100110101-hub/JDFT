#!/usr/bin/env python3
"""独立复核 D4 的**上游提取**（DA-06）：不信任提取脚本，自己再解析一遍上游 Fortran/TOML。

为什么需要这一份：`_d4_staging/*.json` 是由 `extract_d4_data.py` **程序化**生成的（不手抄数字），
但"程序化生成"不等于"生成对了"—— 解析器的偏移一格、把 `real(wp), parameter :: foo(94) = [...]`
的续行数错一行，都会安静地给出**看起来合理**的数字。所以这里用一套**独立的、最小化的**解析器
重新读同一批上游文件，逐个数字对比，任何不一致都点名到元素/字段。

对照的两块：
1. **逐元素块**（Z = 1..36）：`multicharge` 的 `eeq2019.f90`（χ、η、kcnchi、rad），dftd4 的
   `data/en.f90`（Pauling 电负性）、`data/hardness.f90`（化学硬度）、`data/r4r2.f90`、
   `data/zeff.f90`、`data/covrad.f90`；
2. **泛函阻尼块**：dftd4 的 `assets/parameters.toml`（上游认为这是参数的规范来源，
   `param.f90` 是从它生成的）里 `[parameter.<label>]` 的 s6/s8/a1/a2/alp。

用法（在仓库根目录）：
    python3 scripts/check_d4_extract.py            # 打印逐项对比与结论
    python3 scripts/check_d4_extract.py --quiet    # 只打印结论

数据缺失（没下载 `_d4_staging/raw/`）时退出码 2，而不是假装通过。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(ROOT, "qwb-data", "dispersion", "_d4_staging")
RAW = os.path.join(STAGING, "raw")

EEQ_F90 = os.path.join(RAW, "multicharge", "src", "multicharge", "param", "eeq2019.f90")
DFTD4_DATA = {
    "pauling_en": (
        os.path.join(RAW, "dftd4", "src", "dftd4", "data", "en.f90"),
        "pauling_en",
    ),
    "chemical_hardness": (
        os.path.join(RAW, "dftd4", "src", "dftd4", "data", "hardness.f90"),
        "chemical_hardness",
    ),
    "r4r2": (os.path.join(RAW, "dftd4", "src", "dftd4", "data", "r4r2.f90"), "r4_over_r2"),
    "zeff": (
        os.path.join(RAW, "dftd4", "src", "dftd4", "data", "zeff.f90"),
        "effective_nuclear_charge",
    ),
    "covalent_rad_2009_angstrom": (
        os.path.join(RAW, "dftd4", "src", "dftd4", "data", "covrad.f90"),
        "covalent_rad_2009",
    ),
}
PARAMS_TOML = os.path.join(RAW, "dftd4", "assets", "parameters.toml")

#: `parameters.toml` 的键 ↔ JSON 字段。
TOML_KEYS = {"s6": "s6", "s8": "s8", "a1": "a1", "a2": "a2", "alp": "alp"}


def read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def fortran_array(text: str, name: str) -> list[float]:
    """把 `real(wp), parameter :: <name>(...) = [ ... ]`（含 `&` 续行与 `_wp` 后缀）里的数读出来。"""
    # 声明右侧可能有前缀表达式（例如 `covalent_rad_2009(max_elem) = aatoau * [ ... ]`），
    # 所以只要求「名字(维度) = 任意非方括号前缀 [」。
    match = re.search(rf"\b{re.escape(name)}\s*\([^)]*\)\s*=\s*[^\[]*\[", text)
    if match is None:
        raise SystemExit(f"check_d4_extract: array `{name}` not found")
    start = match.end()
    body = []
    depth = 1
    for char in text[start:]:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            body.append(char)
    body = body[:-1] if body else body
    cleaned = "".join(body).replace("&", " ").replace("\n", " ")
    cleaned = re.sub(r"_wp\b", "", cleaned)
    values = []
    for token in cleaned.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def toml_parameter_tables(text: str):
    """解析上游 `parameters.toml` 的两层结构：

    * `[default.parameter]` 里是**默认**（`d4.bj-eeq-atm = { s6=1.0, s9=1.0, alp=16.0, ... }`）；
    * `[parameter.<label>]` 里逐个泛函只写被覆盖的键（多数是 s8/a1/a2，个别还覆盖 s6）。

    返回 `(blocks, defaults)`，`blocks[label][variant] = {key: float}`。
    """
    blocks: dict[str, dict[str, dict[str, float]]] = {}
    defaults: dict[str, dict[str, float]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        header = re.match(r"^\[([^\]]+)\]$", line)
        if header:
            current = header.group(1)
            if current.startswith("parameter."):
                blocks.setdefault(current[len("parameter."):], {})
            continue
        entry = re.match(r"^([A-Za-z0-9_.\-]+)\s*=\s*\{(.*)\}$", line)
        if entry is None or current is None:
            continue
        key, body = entry.group(1), entry.group(2)
        values = {}
        for part in body.split(","):
            pair = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*$", part)
            if pair:
                values[pair.group(1)] = float(pair.group(2))
        if current == "default.parameter":
            defaults[key] = values
        elif current.startswith("parameter."):
            blocks[current[len("parameter."):]][key] = values
    return blocks, defaults


def compare(label: str, reference: list[float], actual: list[float], quiet: bool) -> list[str]:
    problems = []
    if len(reference) != len(actual):
        problems.append(f"{label}: upstream has {len(reference)} entries, extract has {len(actual)}")
        return problems
    worst = 0.0
    where = None
    for index, (want, got) in enumerate(zip(reference, actual)):
        delta = abs(want - got)
        if delta > worst:
            worst, where = delta, index
    if worst > 1.0e-12:
        problems.append(f"{label}: max |Δ| = {worst:.3e} at index {where}")
    if not quiet:
        print(
            f"  {label:28s} n = {len(reference):3d}   max |Δ| = {worst:.2e}"
            + (f"   (index {where})" if worst > 1.0e-12 else "")
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    missing = [
        path
        for path in [EEQ_F90, PARAMS_TOML, *(entry[0] for entry in DFTD4_DATA.values())]
        if not os.path.isfile(path)
    ]
    if missing:
        print("缺失上游文件（先跑 `_d4_staging/extract_d4_data.py` 下载）:")
        for path in missing:
            print(f"  {path}")
        return 2

    elements = json.loads(read(os.path.join(STAGING, "eeq_elements.json")))["elements"]
    elements = sorted(elements, key=lambda item: item["Z"])
    assert [item["Z"] for item in elements] == list(range(1, 37)), "extract must cover Z=1..36"
    functionals = json.loads(read(os.path.join(STAGING, "damping_parameters.json")))["functionals"]

    problems: list[str] = []

    if not args.quiet:
        print("== 逐元素块（multicharge eeq2019 + dftd4 data/*.f90）==")
    eeq_text = read(EEQ_F90)
    for field, array in [
        ("chi", "eeq_chi"),
        ("eta", "eeq_eta"),
        ("kcnchi", "eeq_kcnchi"),
        ("rad", "eeq_rad"),
    ]:
        upstream = fortran_array(eeq_text, array)[:36]
        extracted = [item[field] for item in elements]
        problems += compare(f"eeq2019.{field}", upstream, extracted, args.quiet)

    for field, (path, array) in DFTD4_DATA.items():
        upstream = fortran_array(read(path), array)[:36]
        extracted = [item[field] for item in elements]
        problems += compare(f"dftd4.{field}", upstream, extracted, args.quiet)

    # `covalent_rad_d3` 不是字面量数组，而是派生的 `4/3 * covalent_rad_2009`（covrad.f90:64），
    # 所以按派生关系核对，而不是按数组解析。
    derived = [4.0 / 3.0 * item["covalent_rad_2009_angstrom"] for item in elements]
    problems += compare(
        "derived.4/3*covrad_2009",
        derived,
        [item["covalent_rad_d3_angstrom"] for item in elements],
        args.quiet,
    )

    if not args.quiet:
        print("== 泛函阻尼块（assets/parameters.toml）==")
    blocks, defaults = toml_parameter_tables(read(PARAMS_TOML))
    variant = "d4.bj-eeq-atm"  # dftd4 的默认 D4 变体（与 DampingParam(method=...) 对应）
    base = defaults.get(variant, {})
    for name, values in functionals.items():
        if values.get("absent_upstream"):
            print(f"  {name:28s} 上游缺失（已记录为 absent_upstream）")
            continue
        label = None
        for candidate in [values.get("matched_alias") or "", name]:
            if candidate and candidate in blocks:
                label = candidate
                break
        if label is None:
            problems.append(f"{name}: no [parameter.*] block in parameters.toml")
            continue
        merged = dict(base)
        merged.update(blocks[label].get(variant, {}))
        for key, field in TOML_KEYS.items():
            if key not in merged:
                problems.append(f"{name}: parameters.toml (default+`{label}`) has no `{key}`")
                continue
            got = values[field]
            delta = abs(merged[key] - got)
            if delta > 1.0e-12:
                problems.append(f"{name}.{key}: toml {merged[key]!r} vs extract {got!r}")
            if not args.quiet:
                print(f"  {name:12s} {key:4s} toml = {merged[key]:+.10f}  Δ = {delta:.1e}")

    print()
    if problems:
        print(f"FAIL: {len(problems)} 处不一致")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"ok: 36 个元素的 {4 + len(DFTD4_DATA)} 组参数与 "
        f"{len([k for k, v in functionals.items() if not v.get('absent_upstream')])} 个泛函的阻尼参数"
        " 与上游逐个数字一致"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
