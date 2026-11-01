"""从 s-dftd3 的官方源码生成 `qwb-data/dispersion/d3bj.yaml`（DA-06）。

数据来源（逐字节可复现，上游文件的 sha256 写进数据文件的 `provenance`）：

- `src/dftd3/reference.f90`   —— 参考 C6 表（`reference_c6`）、`reference_cn`、
  `number_of_references`
- `src/dftd3/param.f90`       —— 每个泛函的 D3(BJ) 阻尼参数（s6/s8/a1/a2）：方法名到
  `p_*` 枚举的映射在 `get_method_id`，参数在 `get_rational_damping` 的 `select case`
- `src/dftd3/data/r4r2.f90`   —— 官方 `r4_over_r2` 表（118 个元素；只取 H–Kr）。
  早先几轮按 `data/r4r2.f90`、`data_r4r2.f90` 找都 404，正确路径是 `src/dftd3/data/r4r2.f90`。
- `src/dftd3/ncoord.f90`      —— 分数配位数的**公式**（不是猜的）：
  `countf = 1/(1 + exp(-kcn*(rc/r - 1)))`，`kcn = 16`、`rc = rcov_i + rcov_j`、`r` 为 Bohr。
  注意是 **`rc/r`**，不是 `r/rc` —— 早先几轮用反了比值，这是 CN 一直没复现的根因。
- `src/dftd3/model.f90`       —— 参考体系的权重：`gw = exp(-wf*(dcn^2 - min(dcn^2)))`
  再归一化（log-sum-exp，`wf_default = 4.0`）；不是拉格朗日插值。
- `src/dftd3/damping/rational.f90` —— 两点项：`rrij = 3*r4r2_i*r4r2_j`、
  `r0ij = a1*sqrt(rrij) + a2`、`E = -C6*(s6/(r^6+r0^6) + s8*rrij/(r^8+r0^8))`。
- mctc-lib `src/mctc/data/covrad.f90` —— `rcov`（Pyykko & Atsumi 2009 的共价半径，Å，
  **D3 用 4/3 倍**：`covalent_rad_d3 = 4/3 * covalent_rad_2009`）。

数据文件里存的是**上游公布的值**（Å、无量纲 r4r2、Ha·Bohr^6 的 C6、纯数阻尼参数），
单位换算（Å→Bohr）留给加载器，用 K2 的 `BOHR_PER_ANGSTROM`（CODATA 2018）。

用法：
    python3 scripts/gen_d3_params.py            # 生成/覆盖 qwb-data/dispersion/d3bj.yaml
    python3 scripts/gen_d3_params.py --check     # 只校验现有文件与上游逐项一致

`--check` 是**语义**比较（逐项浮点相等，容差 0），与 YAML 排版无关 —— 因为文件随后会被
`qwb-stamp-data` 重新序列化，排版会变，值不会。它需要 PyYAML（门禁用 `.venv` 的解释器）。
"""

import argparse
import hashlib
import math
import os
import re
import sys
import urllib.request

UPSTREAM = {
    "reference": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/reference.f90",
    "param": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/param.f90",
    "r4r2": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/data/r4r2.f90",
    "ncoord": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/ncoord.f90",
    "model": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/model.f90",
    "rational": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/damping/rational.f90",
    "covrad": "https://raw.githubusercontent.com/awvwgk/mctc-lib/main/src/mctc/data/covrad.f90",
    "vdwrad": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/data/vdwrad.f90",
    "zero": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/damping/zero.f90",
}

#: D3 原始论文（C6 参考表与 BJ 阻尼的出处）。2026-11-09 用 api.crossref.org 核对过。
D3_DOI = "10.1063/1.3382344"
D3_TITLE = (
    "A consistent and accurate ab initio parametrization of density functional "
    "dispersion correction (DFT-D) for the 94 elements H-Pu"
)
#: Pyykko & Atsumi 2009（CN 用的共价半径）。题名/卷/页用 Crossref 核对过：
#: 2026-11-09 查 api.crossref.org/works/10.1002/chem.200800987 得到
#: "Molecular Single-Bond Covalent Radii for Elements 1-118", Chem. Eur. J. 15, 186-197。
#: **注意是 Single-Bond 那篇**（Double-Bond 是另一个 DOI），mctc-lib 的文件头把它写成
#: "15, 2009, 188-197"，页首略有出入，以 Crossref 的为准。
COVRAD_DOI = "10.1002/chem.200800987"
COVRAD_TITLE = "Molecular Single-Bond Covalent Radii for Elements 1-118"

#: CN 计数函数的常量（`ncoord.f90` 的 `default_kcn`）。
CN_KCN = 16.0
#: 零阻尼（D3(0)）覆盖的泛函：**比 BJ 多两个** —— M06-L 与 M06-2X 在上游只有零阻尼
#: 参数（rs6 = 1.581 / 1.619、s8 = 0），这也是 M2 那条「M06-2X-D3」断言唯一可用的路径。
METHODS_ZERO = {
    "b3lyp": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "hf": "hf",
    "m06-l": "m06l",
    "m06-2x": "m062x",
}

#: 参考体系权重的宽度因子（`model.f90` 的 `wf_default`）。
REFERENCE_WF = 4.0
#: 两点项的实空间截断（`cutoff.f90` 的 `disp2_default`，Bohr）。
DISP2_CUTOFF_BOHR = 60.0
#: CN 的实空间截断（`cutoff.f90` 的 `cn_default`，Bohr）。
CN_CUTOFF_BOHR = 40.0
#: 参考点个数（`reference.f90` 的 `max_ref`）。
MAX_REF = 7

CACHE = ".d3-upstream"
OUTPUT = os.path.join("qwb-data", "dispersion", "d3bj.yaml")
#: 零阻尼（D3(0)）那一份：它需要**成对 vdW 半径表**与 rs6/rs8/alp，所以是独立文件
#: （K1 的一份数据 = 一个自包含文件）。参考 C6/CN/r4r2 与 BJ 那份逐字节相同 ——
#: 同一个生成器从同一个解析结果写出来，不可能漂移。
OUTPUT_ZERO = os.path.join("qwb-data", "dispersion", "d3zero.yaml")

# H–Kr：DA-06 覆盖的元素范围（与 registry 的 `elements` 一致）。
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]

# 本数据支持的泛函（数据文件里的 `functional` 键 → param.f90 的方法名）。
# 名字是 `get_method_id` 里的**规范化形式**（小写、去掉 `-`），与 QWB 的泛函名一致。
#
# ⚠ **M06-L / M06-2X 不在里面，这是上游的事实不是遗漏**：s-dftd3 的
# `get_rational_damping`（BJ）里没有 `p_m06l_df` / `p_m062x_df` 分支，它们只出现在
# `get_zero_damping`（零阻尼，`rs6=1.581` / `1.619`、`s8=0.0`）。也就是说 M06 系列
# **没有公开的 BJ 参数**，要给它做色散只能用零阻尼 —— 零阻尼的 R0 要用 vdW 半径表
# （`get_vdw_rad`），是另一份数据 + 另一套阻尼函数，归 DA-06 的剩余部分。
METHODS = {
    "b3lyp": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "hf": "hf",
}


def fetch(name: str) -> bytes:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.f90")
    if os.path.exists(path):
        return open(path, "rb").read()
    with urllib.request.urlopen(UPSTREAM[name], timeout=180) as response:
        data = response.read()
    with open(path, "wb") as handle:
        handle.write(data)
    return data


def strip_fortran_comments(text: str) -> str:
    return "\n".join(re.sub(r"!.*$", "", line) for line in text.splitlines())


def floats(text: str):
    """按顺序取出所有实数（含 `_wp` / `_dp` 后缀、`d`/`e` 指数）。"""
    body = text.replace("_wp", "").replace("_dp", "").replace("_WP", "")
    return [
        float(match)
        for match in re.findall(r"[-+]?\d*\.\d+(?:[edED][-+]?\d+)?|[-+]?\d+\.", body)
    ]


def ints(text: str):
    return [int(match) for match in re.findall(r"\d+", text)]


def select_cases(text: str):
    """把 Fortran `select case(...)` 表拆成 `[(labels, body), ...]`。

    不能用「`case(...)` + 非贪婪 `.*?end`」的正则：`end` 在几十个 `case` 之后才出现，
    匹配会把整个表吞成一块（第 13–20 轮一直没提取出阻尼参数就是这个原因）。
    """
    clean = strip_fortran_comments(text).replace("_wp", "").replace("_dp", "")
    tables = []
    for select in re.finditer(r"select case\([^)]*\)(.*?)\n\s*end select", clean, re.S):
        body = select.group(1)
        parts = re.split(r"(?m)^\s*case\s*\(([^)]*)\)", body)
        cases = []
        for index in range(1, len(parts) - 1, 2):
            labels = [label.strip().strip('"\'') for label in parts[index].split(",")]
            cases.append((labels, parts[index + 1]))
        tables.append(cases)
    return tables


def parse_reference(text: str):
    """解析 C6 参考表、参考 CN 与每个元素的参考点个数。"""
    clean = strip_fortran_comments(text)
    count = ints(clean.split("number_of_references(max_elem) =")[1].split("]")[0])

    cn_body = clean.split("reference_cn(max_ref, max_elem) = reshape([")[1]
    cn_body = cn_body.split("]", 1)[0]
    cn_values = floats(cn_body)

    # 只取赋值块 `c6ab_view(a : b) = [ ... ]`；声明行 `c6ab_view(...) => c6ab` 也含
    # 方括号以外的匹配，早先按 split 抓会整体错位一格（已踩过）。
    c6_values = []
    for match in re.finditer(
        r"c6ab_view\(\s*\d+\s*:\s*\d+\s*\)\s*=\s*\[(.*?)\]", clean, re.S
    ):
        c6_values.extend(floats(match.group(1)))
    return count, cn_values, c6_values


def parse_r4r2(text: str):
    """`r4_over_r2(max_elem) = [ ... ]` 的逐元素值，已换成**模型实际消费的形式**。

    这里踩过一个大坑：`r4_over_r2` 是**公布值**（H 8.0589、O 4.7566 …），但
    `get_r4r2_val` 返回的不是它，而是同一文件里派生出来的

        sqrt_z_r4_over_r2(i) = sqrt(0.5 * r4_over_r2(i) * sqrt(i))

    （`r4r2.f90` 第 69–70 行）。`damping/rational.f90` 里的
    `rrij = 3*r4r2(izp)*r4r2(jzp)` 用的是**这个派生值**。早先几轮直接拿公布值
    代进去，C8/C6 差了 2–16 倍（H 差 16.1、Kr 差 2.07），这就是色散能量一直对不上
    官方实现的真正原因 —— 不是 CN、不是表索引、不是单位。反解 r4r2 失败也是同一个
    原因：反解出来的量本来就带 `sqrt(0.5*sqrt(Z))` 因子，拿去填公布值的表当然对不上。
    """
    clean = strip_fortran_comments(text)
    start = clean.index("r4_over_r2(max_elem) = [")
    end = clean.index("]", start)
    published = [float(value) for value in re.findall(r"([0-9]+\.[0-9]+)_wp", clean[start:end])]
    if "sqrt(0.5_wp*(r4_over_r2*[(sqrt(real(idum,wp)),idum=1,max_elem)]))" not in text:
        raise SystemExit("r4r2.f90 no longer derives sqrt_z_r4_over_r2 the way we assume")
    return published


def parse_covrad(text: str):
    """Pyykko 2009 的共价半径（Å，上游 `covalent_rad_2009`，未乘 4/3）。"""
    clean = strip_fortran_comments(text)
    start = clean.index("covalent_rad_2009(max_elem) = aatoau * [")
    end = clean.index("]", start)
    values = floats(clean[start:end])
    if len(values) != 118:
        raise SystemExit(f"covalent_rad_2009: expected 118 values, got {len(values)}")
    return values


def parse_vdwrad(text: str):
    """成对 vdW 半径表（`vdwrad`，Å；上游乘 `aatoau` 转 Bohr）。

    这是一张 **元素对** 表（`i + j(j-1)/2` 列主序下三角，与 C6 表同一套配对索引），
    不是「两原子半径相加」—— `get_vdw_rad_pair_num` 直接查表（`get_vdw_rad_num(i)`
    才是 `0.5*vdwrad(i,i)`）。零阻尼的 `R0_ij` 用的就是这张表的成对值。
    """
    clean = strip_fortran_comments(text)
    start = clean.index("vdwrad(max_elem*(1+max_elem)/2) = aatoau * [")
    end = clean.index("]", start)
    values = floats(clean[start:end])
    pairs = (103 * 104) // 2
    if len(values) != pairs:
        raise SystemExit(f"vdwrad: expected {pairs} pair values, got {len(values)}")
    return values


def parse_zero_damping(text: str, method_ids):
    """`get_zero_damping` 的 `p_*` → (s6, s8, rs6, rs8, alp)。

    `d3_param` 的缺省值是 s6 = 1、s8 = 1、rs6 = 1、rs8 = 1、alp = 14（`param.f90` 顶部的
    类型定义），上游对每个方法只写非缺省的那几项（M06-L 只写 rs6=1.581、s8=0）。
    """
    clean = strip_fortran_comments(text).replace("_wp", "").replace("_dp", "")
    start = clean.index("subroutine get_zero_damping")
    table = clean[start : clean.index("end subroutine get_zero_damping", start)]
    result = {}
    for labels, body in select_cases(table)[0]:
        match = re.search(r"d3_param\(([^)]*)\)", body)
        if not match:
            continue
        arguments = {
            key: float(value)
            for key, value in re.findall(r"(\w+)\s*=\s*([-+0-9.eEdD]+)", match.group(1))
        }
        params = {
            "s6": arguments.get("s6", 1.0),
            "s8": arguments.get("s8", 1.0),
            "rs6": arguments.get("rs6", 1.0),
            "rs8": arguments.get("rs8", 1.0),
            "alp": arguments.get("alp", 14.0),
        }
        for label in labels:
            if label in method_ids.values() and label not in result:
                result[label] = params
    return result


def parse_method_ids(text: str):
    """`get_method_id` 里的「规范化方法名 → `p_*` 枚举」表。"""
    clean = strip_fortran_comments(text)
    start = clean.index("function get_method_id")
    table = clean[start : clean.index("end function get_method_id", start)]
    mapping = {}
    for match in re.finditer(r"case\(([^)]*)\)\s*;\s*id = (\w+)", table):
        for name in match.group(1).split(","):
            mapping[name.strip().strip('"\'')] = match.group(2)
    return mapping


def parse_damping(text: str, method_ids):
    """`get_rational_damping` 的 `p_*` → `d3_param(a1=…, s8=…, a2=…)`。"""
    clean = strip_fortran_comments(text).replace("_wp", "").replace("_dp", "")
    start = clean.index("subroutine get_rational_damping")
    table = clean[start : clean.index("end subroutine get_rational_damping", start)]
    result = {}
    for labels, body in select_cases(table)[0]:
        bj = re.search(r"d3_param\(([^)]*)\)", body)
        if not bj:
            continue
        arguments = {
            key: float(value)
            for key, value in re.findall(r"(\w+)\s*=\s*([-+0-9.eEdD]+)", bj.group(1))
        }
        if "a1" not in arguments or "a2" not in arguments:
            continue
        for label in labels:
            if label in method_ids.values() and label not in result:
                result[label] = {
                    "s6": arguments.get("s6", 1.0),
                    "s8": arguments.get("s8", 1.0),
                    "a1": arguments["a1"],
                    "a2": arguments["a2"],
                }
    return result


def c6_pair_index(z_hi: int, z_lo: int) -> int:
    """`reference.f90` 的 `get_c6`：列主序下三角，`ic = min + max*(max-1)/2`（1 基）。"""
    return z_lo + (z_hi * (z_hi - 1)) // 2


def build():
    """从上游源码算出数据文件的全部内容（纯 Python 结构，不打戳）。"""
    raw = {name: fetch(name) for name in UPSTREAM}
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    text = {name: data.decode("utf-8", "replace") for name, data in raw.items()}

    reference_count, reference_cn, reference_c6 = parse_reference(text["reference"])
    r4r2_all = parse_r4r2(text["r4r2"])
    covrad_all = parse_covrad(text["covrad"])
    method_ids = parse_method_ids(text["param"])
    damping_all = parse_damping(text["param"], method_ids)

    if len(reference_cn) != MAX_REF * 103:
        raise SystemExit(f"reference_cn: expected {MAX_REF * 103} values, got {len(reference_cn)}")
    pairs_total = (103 * 104) // 2
    if len(reference_c6) != MAX_REF * MAX_REF * pairs_total:
        raise SystemExit(f"reference_c6: expected {MAX_REF * MAX_REF * pairs_total} values, "
                         f"got {len(reference_c6)}")
    if len(r4r2_all) != 118:
        raise SystemExit(f"r4_over_r2: expected 118 values, got {len(r4r2_all)}")
    if CN_KCN != 16.0:  # 与 ncoord.f90 的 default_kcn 交叉核对
        raise SystemExit("kcn drifted from ncoord.f90")
    if "rc/r1 - 1.0" not in text["ncoord"]:
        raise SystemExit("ncoord.f90 no longer uses rc/r — re-derive the counting function")

    n_elements = len(ELEMENTS)
    data = {
        "name": "d3bj",
        "version": "1.0.0",
        "kind": "d3bj",
        "elements": list(ELEMENTS),
        "coordination_number": {
            "model": "exp",
            "kcn": CN_KCN,
            "scale": 4.0 / 3.0,
            "cutoff_bohr": CN_CUTOFF_BOHR,
            "radii_angstrom": {
                symbol: covrad_all[z - 1] for z, symbol in enumerate(ELEMENTS, start=1)
            },
        },
        "reference": {
            "wf": REFERENCE_WF,
            "weight": "gaussian",
        },
        "r4r2": {
            symbol: math.sqrt(0.5 * r4r2_all[z - 1] * math.sqrt(z))
            for z, symbol in enumerate(ELEMENTS, start=1)
        },
        "r4r2_published": {
            symbol: r4r2_all[z - 1] for z, symbol in enumerate(ELEMENTS, start=1)
        },
        "reference_cn": {
            symbol: reference_cn[
                (z - 1) * MAX_REF : (z - 1) * MAX_REF + reference_count[z - 1]
            ]
            for z, symbol in enumerate(ELEMENTS, start=1)
        },
        "reference_c6": {},
        "damping_parameters": {},
        "dispersion": {"cutoff_bohr": DISP2_CUTOFF_BOHR, "damping": "rational"},
        "provenance": {
            "generator": "scripts/gen_d3_params.py",
            "upstream": [
                {"file": path.split("/", 3)[-1], "url": UPSTREAM[name], "sha256": digests[name]}
                for name, path in (
                    ("reference", UPSTREAM["reference"]),
                    ("param", UPSTREAM["param"]),
                    ("r4r2", UPSTREAM["r4r2"]),
                    ("ncoord", UPSTREAM["ncoord"]),
                    ("model", UPSTREAM["model"]),
                    ("rational", UPSTREAM["rational"]),
                    ("covrad", UPSTREAM["covrad"]),
                )
            ],
        },
    }

    for symbol in ELEMENTS:
        nref = len(data["reference_cn"][symbol])
        if nref != reference_count[ELEMENTS.index(symbol)]:
            raise SystemExit(f"{symbol}: reference count mismatch")
        data["reference_cn"][symbol] = list(data["reference_cn"][symbol])

    for z_hi in range(1, n_elements + 1):
        for z_lo in range(1, z_hi + 1):
            ic = c6_pair_index(z_hi, z_lo)
            base = (ic - 1) * MAX_REF * MAX_REF
            key = f"{ELEMENTS[z_hi - 1]}-{ELEMENTS[z_lo - 1]}"
            # 原样保存 7×7 块的 **Fortran 展平顺序**（第一维最快）：`values[a + 7*b]`
            # 对应 `reference_c6(a+1, b+1, ic)`。曾经按 (iref, jref) 双重循环重排，
            # 得到的列表其实是转置过的，于是异核元素对（O-H）的 C6 差了 40%
            # 而同核（H-H/O-O）看不出来 —— 这是个只用同核体系测不出来的错。
            data["reference_c6"][key] = list(reference_c6[base : base + MAX_REF * MAX_REF])

    for functional, method in METHODS.items():
        token = method_ids.get(method)
        if token is None or token not in damping_all:
            raise SystemExit(f"{functional}: no BJ damping in param.f90 (token {token})")
        data["damping_parameters"][functional] = dict(damping_all[token])

    # ── 零阻尼（D3(0)）那一份 ──
    vdwrad_all = parse_vdwrad(text["vdwrad"])
    zero_all = parse_zero_damping(text["param"], method_ids)
    data_zero = {
        key: value
        for key, value in data.items()
        if key in (
            "elements",
            "coordination_number",
            "reference",
            "r4r2",
            "r4r2_published",
            "reference_cn",
            "reference_c6",
        )
    }
    data_zero["name"] = "d3zero"
    data_zero["version"] = "1.0.0"
    data_zero["kind"] = "d3zero"
    data_zero["zero_damping_parameters"] = {}
    data_zero["vdw_radii_angstrom"] = {}
    data_zero["dispersion"] = {"cutoff_bohr": DISP2_CUTOFF_BOHR, "damping": "zero"}
    data_zero["provenance"] = {
        "generator": data["provenance"]["generator"],
        "upstream": data["provenance"]["upstream"]
        + [
            {"file": UPSTREAM["vdwrad"].split("/", 3)[-1], "url": UPSTREAM["vdwrad"],
             "sha256": digests["vdwrad"]},
            {"file": UPSTREAM["zero"].split("/", 3)[-1], "url": UPSTREAM["zero"],
             "sha256": digests["zero"]},
        ],
    }
    for functional, method in METHODS_ZERO.items():
        token = method_ids.get(method)
        if token is None or token not in zero_all:
            raise SystemExit(f"{functional}: no zero damping in param.f90 (token {token})")
        data_zero["zero_damping_parameters"][functional] = dict(zero_all[token])
    for z_hi in range(1, n_elements + 1):
        for z_lo in range(1, z_hi + 1):
            ic = c6_pair_index(z_hi, z_lo)
            key = f"{ELEMENTS[z_hi - 1]}-{ELEMENTS[z_lo - 1]}"
            data_zero["vdw_radii_angstrom"][key] = vdwrad_all[ic - 1]

    return data, data_zero, digests


def format_float(value: float) -> str:
    """最短可回读的十进制表示（整数值写成 `1.0` 而不是 `1`，保持浮点类型）。"""
    if value == int(value) and abs(value) < 1e16:
        return f"{value:.1f}"
    return repr(value)


def emit(data) -> str:
    """把数据写成 K1 形状的 YAML（不用 PyYAML：输出要点是可读、稳定、逐行可 diff）。"""
    lines = []
    lines.append(f"name: {data['name']}")
    lines.append(f"version: {data['version']}")
    lines.append(f"kind: {data['kind']}")
    lines.append(
        'notes: "D3(BJ) dispersion parameters for H-Kr, transcribed from the s-dftd3 '
        "sources listed in provenance (each file pinned by sha256). Per the D3 paper "
        "C6 from the 7x7 reference table interpolated with Gaussian weights on the "
        "fractional coordination number (kcn = 16, rcov = 4/3 Pyykko 2009), C8/C6 = "
        "3*r4r2_i*r4r2_j, and rational (Becke-Johnson) damping r0 = a1*sqrt(C8/C6) + a2. "
        "C6 values are in Ha*Bohr^6; radii are the Pyykko single-bond covalent radii in Angstrom "
        "exactly as the upstream table stores them (metals already reduced by 10 percent), converted "
        "with the K2 BOHR_PER_ANGSTROM constant at load time. Only functionals "
        "with published BJ parameters are listed -- M06-L and M06-2X have no BJ entry "
        'upstream (zero damping only), so they are deliberately absent here."'
    )
    lines.append("citation:")
    lines.append(f"  - title: {D3_TITLE}")
    lines.append(f"    doi: {D3_DOI}")
    lines.append(f"  - title: {COVRAD_TITLE}")
    lines.append(f"    doi: {COVRAD_DOI}")
    lines.append("license:")
    lines.append("  spdx: LGPL-3.0-or-later")
    lines.append("  commercial_ok: true")
    lines.append("  redistribute_ok: true")
    lines.append("created: 2026-11-09T00:00:00Z")

    lines.append("elements:")
    for symbol in data["elements"]:
        lines.append(f"  - {symbol}")

    cn = data["coordination_number"]
    lines.append("coordination_number:")
    lines.append(f"  model: {cn['model']}")
    lines.append(f"  kcn: {format_float(cn['kcn'])}")
    lines.append(f"  scale: {format_float(cn['scale'])}")
    lines.append(f"  cutoff_bohr: {format_float(cn['cutoff_bohr'])}")
    lines.append("  radii_angstrom:")
    for symbol in data["elements"]:
        lines.append(f"    {symbol}: {format_float(cn['radii_angstrom'][symbol])}")

    lines.append("reference:")
    lines.append(f"  wf: {format_float(data['reference']['wf'])}")
    lines.append(f"  weight: {data['reference']['weight']}")

    # `r4r2` 是模型消费的派生值；`r4r2_published` 是同一文件里公布的表，两者一起落盘
    # 才能一眼看出「这是派生的，不是转录错的」。
    lines.append("r4r2:")
    for symbol in data["elements"]:
        lines.append(f"  {symbol}: {format_float(data['r4r2'][symbol])}")
    lines.append("r4r2_published:")
    for symbol in data["elements"]:
        lines.append(f"  {symbol}: {format_float(data['r4r2_published'][symbol])}")

    lines.append("reference_cn:")
    for symbol in data["elements"]:
        values = ", ".join(format_float(value) for value in data["reference_cn"][symbol])
        lines.append(f"  {symbol}: [{values}]")

    lines.append("reference_c6:")
    for key, values in data["reference_c6"].items():
        lines.append(f"  {key}: [" + ", ".join(format_float(value) for value in values) + "]")

    if data["kind"] == "d3bj":
        lines.append("damping_parameters:")
        for functional, params in data["damping_parameters"].items():
            lines.append(f"  {functional}:")
            for field in ("s6", "s8", "a1", "a2"):
                lines.append(f"    {field}: {format_float(params[field])}")
            lines.append(f"    doi: {D3_DOI}")
    else:
        # 成对 vdW 半径：零阻尼的 R0_ij 直接查这张表（**不是**两原子半径相加）。
        lines.append("vdw_radii_angstrom:")
        for key, value in data["vdw_radii_angstrom"].items():
            lines.append(f"  {key}: {format_float(value)}")
        lines.append("zero_damping_parameters:")
        for functional, params in data["zero_damping_parameters"].items():
            lines.append(f"  {functional}:")
            for field in ("s6", "s8", "rs6", "rs8", "alp"):
                lines.append(f"    {field}: {format_float(params[field])}")
            lines.append(f"    doi: {D3_DOI}")

    lines.append("dispersion:")
    lines.append(f"  damping: {data['dispersion']['damping']}")
    lines.append(f"  cutoff_bohr: {format_float(data['dispersion']['cutoff_bohr'])}")

    lines.append("provenance:")
    lines.append(f"  generator: {data['provenance']['generator']}")
    lines.append("  upstream:")
    for entry in data["provenance"]["upstream"]:
        lines.append(f"    - file: {entry['file']}")
        lines.append(f"      url: {entry['url']}")
        lines.append(f"      sha256: {entry['sha256']}")
    return "\n".join(lines) + "\n"


def compare(expected: str, path: str) -> int:
    """把生成结果与磁盘上的文件做**逐项**比较（容差 0，与排版无关）。"""
    import yaml  # noqa: PLC0415 —— 只有 --check 才需要（门禁用 .venv 的解释器）

    want = yaml.safe_load(expected)
    got = yaml.safe_load(open(path, encoding="utf-8").read())
    problems = []

    def walk(a, b, where):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                problems.append(f"{where}: expected a mapping, got {type(b).__name__}")
                return
            for key in a:
                if key not in b:
                    problems.append(f"{where}.{key}: missing from the data file")
                else:
                    walk(a[key], b[key], f"{where}.{key}")
            for key in b:
                # 顶层 `sha256` 是 `qwb-stamp-data` 打的内容摘要，生成器不管它
                # （它由 Rust 侧的规范化规则算，见 qwb-registry 的 sha256 模块）。
                if key not in a and not (where in ("d3bj", "d3zero") and key == "sha256"):
                    problems.append(f"{where}.{key}: not produced by the generator")
        elif isinstance(a, list):
            if not isinstance(b, list):
                problems.append(f"{where}: expected a list, got {type(b).__name__}")
                return
            if len(a) != len(b):
                problems.append(f"{where}: length {len(b)} != expected {len(a)}")
            for index, (left, right) in enumerate(zip(a, b)):
                walk(left, right, f"{where}[{index}]")
        elif isinstance(a, bool):
            if bool(b) != a:
                problems.append(f"{where}: {b!r} != expected {a!r}")
        elif isinstance(a, float) or isinstance(a, int):
            if isinstance(b, bool) or not isinstance(b, (int, float)):
                problems.append(f"{where}: expected a number, got {b!r}")
            elif float(a) != float(b):
                problems.append(f"{where}: {b!r} != expected {a!r}")
        elif a != b:
            problems.append(f"{where}: {b!r} != expected {a!r}")

    walk(want, got, "d3bj")
    if problems:
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  … and {len(problems) - 40} more")
        print(f"FAIL: {len(problems)} item(s) differ from a fresh generation")
        return 1
    items = count_items(want)
    print(f"ok: {items} items match the upstream sources exactly (tolerance 0)")
    return 0


def count_items(node) -> int:
    if isinstance(node, dict):
        return sum(count_items(value) for value in node.values())
    if isinstance(node, list):
        return sum(count_items(value) for value in node)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data, data_zero, digests = build()
    body = emit(data)
    body_zero = emit(data_zero)

    print("上游 sha256：")
    for name, digest in digests.items():
        print(f"  {name}: {digest}")
    print(
        f"元素 {len(data['elements'])} 个（H–Kr）；C6 元素对 {len(data['reference_c6'])} 个"
        f"（应为 {len(ELEMENTS) * (len(ELEMENTS) + 1) // 2}）；每个 {MAX_REF * MAX_REF} 个参考点"
    )
    for functional, params in data["damping_parameters"].items():
        print(
            f"  {functional}: s6={params['s6']} s8={params['s8']} "
            f"a1={params['a1']} a2={params['a2']}"
        )
    for symbol in ("H", "C", "N", "O", "Kr"):
        print(
            f"  {symbol}: r4r2={data['r4r2'][symbol]} "
            f"rcov={data['coordination_number']['radii_angstrom'][symbol]} A, "
            f"nref={len(data['reference_cn'][symbol])}"
        )
    print(f"  C6(H,H) 第 1 个参考点 = {data['reference_c6']['H-H'][0]}")
    print(f"  C6(C,C) 第 1 个参考点 = {data['reference_c6']['C-C'][0]}")

    print(f"  零阻尼 vdW 半径（成对）：{len(data_zero['vdw_radii_angstrom'])} 个元素对，"
          f"H-H = {data_zero['vdw_radii_angstrom']['H-H']} A")
    for functional, params in data_zero["zero_damping_parameters"].items():
        print(
            f"  零阻尼 {functional}: s6={params['s6']} s8={params['s8']} "
            f"rs6={params['rs6']} rs8={params['rs8']} alp={params['alp']}"
        )

    if args.check:
        status = 0
        for body_text, path in ((body, OUTPUT), (body_zero, OUTPUT_ZERO)):
            if not os.path.exists(path):
                print(f"FAIL: {path} does not exist (run without --check first)")
                status = 1
                continue
            print(f"--- {path} ---")
            status |= compare(body_text, path)
        return status

    for body_text, path in ((body, OUTPUT), (body_zero, OUTPUT_ZERO)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)  # 旧文件可能带 sha256：整份重写，避免残留字段
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body_text)
        print(f"wrote {path}（{len(body_text)} bytes）；下一步：qwb-stamp-data {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
