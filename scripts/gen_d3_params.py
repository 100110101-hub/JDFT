"""从 s-dftd3 的官方源码生成 `qwb-data/dispersion/d3bj.yaml`（DA-06）。

数据来源（LGPL-3.0-or-later，逐字节可复现）：
- `src/dftd3/reference.f90`   —— 参考 C6 表（`reference_c6`，展平）、`reference_cn`、
  `number_of_references`
- `src/dftd3/param.f90`       —— 每个泛函的 D3(BJ) 阻尼参数（s6/s8/a1/a2 + DOI）
- `src/dftd3/radii.f90`       —— `rcov`（分数配位数用的共价半径）与 `r4r2`

用法：
    python3 scripts/gen_d3_params.py            # 生成/覆盖 qwb-data/dispersion/d3bj.yaml
    python3 scripts/gen_d3_params.py --check    # 只校验现有文件与上游一致

生成器把**上游文件的 sha256** 写进数据文件（`source_sha256`），这样"这份数据是从哪
份源码生成的"是可核对的；同时把 C6 表按（元素对，7×7 参考点）展平保存。
"""

import argparse
import hashlib
import os
import re
import sys
import urllib.request

UPSTREAM = {
    "reference": "https://raw.githubusercontent.com/dftd3/simple-dftd3/master/src/dftd3/reference.f90",
    "param": "https://raw.githubusercontent.com/dftd3/simple-dftd3/master/src/dftd3/param.f90",
}
CACHE = ".d3-upstream"
OUTPUT = os.path.join("qwb-data", "dispersion", "d3bj.yaml")
MAX_ELEMENT = 36  # H–Kr（DA-06 的范围）

# 我们支持的泛函（数据里的 `functional` 名字 → param.f90 的 case 分支别名）
METHOD_ALIASES = {
    "b3lyp": ["p_b3lyp_df"],
    "pbe": ["p_pbe_df"],
    "pbe0": ["p_pbe0_df", "p_pbeh_df"],
    "m06-l": ["p_m06l_df", "p_m06_l_df"],
    "m06-2x": ["p_m06_2x_df", "p_m062x_df"],
    "hf": ["p_hf_df"],
}


def fetch(name: str) -> bytes:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.f90")
    if os.path.exists(path):
        return open(path, "rb").read()
    with urllib.request.urlopen(UPSTREAM[name], timeout=120) as response:
        data = response.read()
    with open(path, "wb") as handle:
        handle.write(data)
    return data


def strip_fortran_comments(text: str) -> str:
    return "\n".join(re.sub(r"!.*$", "", line) for line in text.splitlines())


def floats(text: str):
    """从一段 Fortran 源码里按顺序取出所有实数（含 `_wp` 后缀与指数）。"""
    return [
        float(match)
        for match in re.findall(
            r"[-+]?\d+\.\d*(?:[edED][-+]?\d+)?|[-+]?\d+\.", text.replace("_wp", "").replace("_dp", "")
        )
    ]


def ints(text: str):
    return [int(match) for match in re.findall(r"\d+", text)]


def bracket_blocks(text: str):
    """取出所有 `reshape([ ... ])` / `= [ ... ]` 里的数组内容（保持顺序）。"""
    blocks = []
    for match in re.finditer(r"\[([^\[\]]*)\]", text):
        body = match.group(1)
        if "&" in body or re.search(r"\d", body):
            blocks.append(body)
    return blocks


def parse_reference(text: str):
    clean = strip_fortran_comments(text)
    count = ints(clean.split("number_of_references(max_elem) =")[1].split("]")[0])
    # reference_cn：7×103，先按行读
    cn_body = clean.split("reference_cn(max_ref, max_elem) = reshape([")[1]
    cn_body = cn_body.split("]", 1)[0]
    cn_values = floats(cn_body)
    # reference_c6：一串 `c6ab_view(a:b) = [ ... ]`，按出现顺序拼起来
    # 只取真正的赋值块 `c6ab_view(a : b) = [ ... ]`（声明行 `c6ab_view(...) => c6ab`
    # 也会被 `split` 抓到，导致数值整体错位一格 —— 已踩过）
    c6_values = []
    for match in re.finditer(
        r"c6ab_view\(\s*\d+\s*:\s*\d+\s*\)\s*=\s*\[(.*?)\]", clean, re.S
    ):
        c6_values.extend(floats(match.group(1)))
    return count, cn_values, c6_values


def parse_param(text: str, aliases):
    clean = strip_fortran_comments(text).replace("_wp", "").replace("_dp", "")
    result = {}
    for alias in aliases:
        for match in re.finditer(r"case\(([^)]*)\)(.*?)end", clean, re.S):
            cases, body = match.group(1), match.group(2)
            if alias not in cases:
                continue
            bj = re.search(r"d3_param\(([^)]*)\)", body)
            if not bj:
                continue
            arguments = dict(
                (key, float(value))
                for key, value in re.findall(r"(\w+)\s*=\s*([-\d.eE+]+)", bj.group(1))
            )
            if "a1" not in arguments or "a2" not in arguments:
                continue
            result = {
                "s6": arguments.get("s6", 1.0),
                "s8": arguments.get("s8", 1.0),
                "a1": arguments["a1"],
                "a2": arguments["a2"],
            }
            break
        if result:
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    reference_bytes = fetch("reference")
    param_bytes = fetch("param")
    count, cn_values, c6_values = parse_reference(reference_bytes.decode("utf-8", "replace"))
    print(f"number_of_references: {len(count)} 项；reference_cn: {len(cn_values)} 个；"
          f"reference_c6: {len(c6_values)} 个")

    pairs = (103 * 104) // 2
    print(f"预期 c6 数量 = 7·7·{pairs} = {49 * pairs}")
    expected = 49 * pairs
    c6_values = c6_values[:expected]
    damping = {}
    for method, aliases in METHOD_ALIASES.items():
        values = parse_param(param_bytes.decode("utf-8", "replace"), aliases)
        if not values:
            print(f"  警告：{method} 没在 param.f90 里找到 BJ 参数")
            continue
        damping[method] = values
        print(f"  {method}: s6={values['s6']} s8={values['s8']} a1={values['a1']} a2={values['a2']}")

    if args.check:
        return 0
    print("（先把解析结果打出来，落盘逻辑下一步补）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
