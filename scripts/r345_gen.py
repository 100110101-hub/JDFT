import io

path = "scripts/gen_d3_params.py"
text = io.open(path, encoding="utf-8").read()


def sub(old, new):
    global text
    assert old in text, f"ANCHOR MISSING: {old[:80]!r}"
    text = text.replace(old, new, 1)


# ① 上游清单里加 vdwrad 与 damping/zero.f90
sub(
    '''    "covrad": "https://raw.githubusercontent.com/awvwgk/mctc-lib/main/src/mctc/data/covrad.f90",
}''',
    '''    "covrad": "https://raw.githubusercontent.com/awvwgk/mctc-lib/main/src/mctc/data/covrad.f90",
    "vdwrad": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/data/vdwrad.f90",
    "zero": "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/damping/zero.f90",
}''',
)

# ② 输出路径与零阻尼的方法表
sub(
    'OUTPUT = os.path.join("qwb-data", "dispersion", "d3bj.yaml")',
    'OUTPUT = os.path.join("qwb-data", "dispersion", "d3bj.yaml")\n'
    '#: 零阻尼（D3(0)）的那一份。它需要 **成对 vdW 半径表** 与 rs6/rs8/alp，\n'
    '#: 所以是独立文件（K1 的一份数据 = 一个自包含文件）；参考 C6/CN/r4r2 与 BJ 那份\n'
    '#: 逐字节相同（同一个生成器从同一个解析结果写出来，不可能漂移）。\n'
    'OUTPUT_ZERO = os.path.join("qwb-data", "dispersion", "d3zero.yaml")',
)

sub(
    '''METHODS = {
    "b3lyp": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "hf": "hf",
}''',
    '''METHODS = {
    "b3lyp": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "hf": "hf",
}

# 零阻尼（D3(0)）覆盖的泛函：**比 BJ 多两个** —— M06-L 与 M06-2X 在上游只有零阻尼
# 参数（rs6 = 1.581 / 1.619、s8 = 0），这也是 M2 那条「M06-2X-D3」断言唯一可用的路径。
METHODS_ZERO = {
    "b3lyp": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "hf": "hf",
    "m06-l": "m06l",
    "m06-2x": "m062x",
}''',
)

# ③ vdwrad 解析 + 零阻尼参数解析
sub(
    '''def parse_method_ids(text: str):''',
    '''def parse_vdwrad(text: str):
    """成对 vdW 半径表（`vdwrad`，Å；上游乘 `aatoau` 转 Bohr）。

    这是一张 **元素对** 表（`i + j(j-1)/2` 列主序下三角，与 C6 表同一套配对索引），
    不是「两原子半径相加」—— `get_vdw_rad_pair_num` 直接查表，`get_vdw_rad_num(i)`
    才是 `0.5*vdwrad(i,i)`。零阻尼的 `R0_ij` 用的就是这张表的成对值。
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

    `d3_param` 的缺省值是 s6 = 1、s8 = 1、rs6 = 1、rs8 = 1、alp = 14（`param.f90` 顶部
    的类型定义），上游对每个方法只写非缺省的那几项。
    """
    clean = strip_fortran_comments(text).replace("_wp", "").replace("_dp", "")
    start = clean.index("subroutine get_zero_damping")
    table = clean[start : clean.index("end subroutine get_zero_damping", start)]
    result = {}
    for labels, body in select_cases(table)[0]:
        match = re.search(r"d3_param\\(([^)]*)\\)", body)
        if not match:
            continue
        arguments = {
            key: float(value)
            for key, value in re.findall(r"(\\w+)\\s*=\\s*([-+0-9.eEdD]+)", match.group(1))
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


def parse_method_ids(text: str):''',
)

io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("generator: parsers added")
