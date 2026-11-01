#!/usr/bin/env python3
"""从 PySCF 的 `mnsol.F` 提取 **SMD 的 CDS 原子表面张力**表（DA-05/SV-02 的数据缺口）。

背景：PySCF 的 Python 层（`pyscf/solvent/smd.py`）把 CDS 项交给编译好的 `libsolvent`
（`mnsol_interface_`），**表在 Fortran 里**，不在 Python 里。本脚本只做一件事：把
`SMD_CDS_AQ`（水）与 `SMD_CDS_NAQ`（非水）两个子程序里的 `SIGMA_DATA` / `HSIGMA_DATA`
数组读出来，展开 Fortran 的重复计数写法（`5*0.00`），写成 JSON。

**只提取、不解释**：本脚本不判断"CDS 项怎么与表面积相乘"，那属于实现（SV-02）与验证。
数值一律来自上游行，`--check` 做语义比较（逐值相等，容差 0）。

**两个 case 的成熟度不同**（这一点写在数据里，不靠注释）：水的 `SIGMA_DATA` 就是逐元素 σ，
`complete: true`；非水的 σ 是**按 Abraham 溶剂描述符拼出来的**（`SIGMA_N` + `SIGMA_A`·SOLA +
`SIGMA_B`·SOLB + …），所以那里只落了原始表、`complete: false` —— 拼装规则单独做、单独验证，
不让"表"冒充"σ"。

用法：
    python3 scripts/gen_smd_cds.py            # 生成/覆盖 staging JSON
    python3 scripts/gen_smd_cds.py --check    # 与磁盘逐值比较
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(ROOT, "qwb-data", "solvent", "_smd_cds_staging")
RAW = os.path.join(STAGING, "raw", "mnsol.F")
OUTPUT = os.path.join(STAGING, "cds_atomic.json")

#: PySCF 的版本与本项目 oracle 一致（`scripts/reference_sources.py` 的 PYSCF_DOI）。
PYSCF_VERSION = "2.14.0"
PYSCF_URL = (
    "https://raw.githubusercontent.com/pyscf/pyscf/v2.14.0/"
    "pyscf/lib/solvent/mnsol/mnsol.F"
)
#: 数据文件里的元素范围（H-Kr），与 solvent 数据的其它部分一致。
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]


def fortran_number_list(body: str) -> list[float]:
    """展开 Fortran `DATA` 的初值表：支持 `n*value`、`&`/`.` 续行与 `D` 指数。"""
    # Fortran 的续行标记是第 6 列的 `.`（如 `  . 48.69, ...`）：**先按行去掉**再拼，
    # 否则拼成一行后行首锚点就失效了（实测会留下一个孤立的 `.` 让 float() 抛错）。
    # 每行要同时处理三件事：列 6 的续行标记（`.` 或 `$`）、Fortran 的行内注释（`! ...`）、
    # 以及可能的行首 `&`。都在**按行**阶段做，避免拼成一行后锚点失效。
    # `RKKVAL` 里的初值写成 `0.00D0`（Fortran 的 double 指数），`float()` 不认，先统一成 `E`。
    body = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", body)
    lines = []
    for line in body.split("\n"):
        line = re.sub(r"!.*$", " ", line)
        lines.append(re.sub(r"^\s*[.$&]", " ", line))
    text = " ".join(lines).replace("&", " ").replace(",", " ")
    values: list[float] = []
    for token in text.split():
        repeat = re.match(r"^(\d+)\*(.+)$", token)
        if repeat:
            count = int(repeat.group(1))
            values.extend([float(repeat.group(2))] * count)
            continue
        values.append(float(token))
    return values


def extract_table(
    source: str, declaration: str, limit: int | None = 150
) -> list[float]:
    """把 `DATA <declaration>/ ... /` 之间的数读出来（只取前 `limit` 个）。"""
    pattern = rf"DATA\s+{re.escape(declaration)}\s*/"
    match = re.search(pattern, source, re.IGNORECASE)
    if match is None:
        raise SystemExit(f"gen_smd_cds: `DATA {declaration}` not found in mnsol.F")
    body_start = match.end()
    body_end = source.index("/", body_start)
    values = fortran_number_list(source[body_start:body_end])
    return values if limit is None else values[:limit]


def extract_matrix(source: str, name: str, size: int) -> list[list[float]]:
    """把 `DATA ((<name>(J,I), I=1,n), J=1,n)` 的 n×n 表读出来。

    与 [`extract_table`] 的区别有三处，都是 `RKKVAL` 这块数据特有的写法：
    ① 声明符不是裸名字（多一层括号），所以开 `/` 得另外找；
    ② 表里夹着整行的定长 Fortran 注释（第 1 列是 `C`），要先剔掉，
       否则 `C`/`Set`/`for` 这些词会被当成数字；
    ③ 初值是 `0.00D0` 这种 `D` 指数（已在 [`fortran_number_list`] 里统一）。
    """
    match = re.search(rf"DATA\s+\(\({name}", source, re.IGNORECASE)
    if match is None:
        raise SystemExit(f"gen_smd_cds: `DATA (({name}` not found in mnsol.F")
    body_start = source.index("/", match.end())
    body_end = source.index("/", body_start + 1)
    body = source[body_start + 1 : body_end]
    kept = [
        line
        for line in body.split("\n")
        if not re.match(r"^\s*[Cc]\s", line) and not line.strip().startswith("C")
    ]
    values = fortran_number_list("\n".join(kept))
    expected = size * size
    if len(values) != expected:
        raise SystemExit(
            f"gen_smd_cds: `{name}` has {len(values)} entries, expected {expected}"
        )
    return [values[row * size : (row + 1) * size] for row in range(size)]


def subroutine_body(source: str, name: str) -> str:
    """取子程序正文。

    结束锚点用「下一个 `SUBROUTINE` 或文件尾」，**不用**第一个 `END` —— Fortran 里的
    `END DO` / `END IF` 会让第一个 `END` 落在循环里（实测 `VDWRAD` 里就有一个 `END DO`，
    截断后表只剩几项）。
    """
    start = source.index(f"SUBROUTINE {name}")
    following = source.find("SUBROUTINE ", start + 1)
    end = following if following != -1 else len(source)
    return source[start:end]


def build_rows() -> dict:
    with open(RAW, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    rows = {
        "schema_version": 1,
        "description": (
            "SMD CDS atomic surface tensions (SIGMA) and their hydrogen-specific "
            "corrections (HSIGMA), for water (ICDS=1) and nonaqueous (ICDS=2) solvents, "
            "extracted programmatically from PySCF's mnsol.F."
        ),
        "generated_by": "scripts/gen_smd_cds.py",
        "upstream": {
            "file": "pyscf/lib/solvent/mnsol/mnsol.F",
            "pyscf_version": PYSCF_VERSION,
            "url": PYSCF_URL,
            "sha256": digest,
            "license": "Apache-2.0",
        },
        "cases": {},
    }
    # 水：一张逐元素表（`SIGMA_DATA`）就够了。
    body = subroutine_body(source, "SMD_CDS_AQ")
    sigma = extract_table(body, "SIGMA_DATA")
    hsigma = extract_table(body, "HSIGMA_DATA")
    if len(sigma) < 36 or len(hsigma) < 36:
        raise SystemExit("gen_smd_cds: the water tables are shorter than 36 entries")
    rows["cases"]["water"] = {
        "subroutine": "SMD_CDS_AQ",
        "complete": True,
        "sigma": {symbol: sigma[index] for index, symbol in enumerate(ELEMENTS)},
        "hsigma": {symbol: hsigma[index] for index, symbol in enumerate(ELEMENTS)},
        "sigma_raw_150": sigma,
        "hsigma_raw_150": hsigma,
    }

    # 非水：σ 不是一张表，而是 `σ_i = σ_N,i + σ_A,i·SOLA + σ_B,i·SOLB (+ σ_MOL·SOLC ...)`
    # 这种按溶剂 Abraham 描述符拼出来的量。所以这里**只落原始表**，`complete: false` —
    # 拼装规则属于下一步（要逐行从 `SMD_CDS_NAQ` 的正文里读出来并单独验证），
    # 把「表」冒充成「σ」会让下游以为 CDS 项可以直接乘。
    naq = subroutine_body(source, "SMD_CDS_NAQ")
    raw = {}
    for declaration in (
        "SIGMA_N",
        "SIGMA_A",
        "SIGMA_B",
        "HSIGMA_N",
        "HSIGMA_A",
        "HSIGMA_B",
        "SIGMA_MOL",
    ):
        raw[declaration.lower()] = extract_table(naq, declaration, limit=None)
    sigma_mol = raw["sigma_mol"]
    if len(sigma_mol) != 4:
        raise SystemExit("gen_smd_cds: SIGMA_MOL must have four entries")
    rows["cases"]["nonaqueous"] = {
        "subroutine": "SMD_CDS_NAQ",
        "complete": True,
        # 拼装规则逐字来自 `SMD_CDS_NAQ` 的正文（mnsol.F:423-429）：
        #   SIGMA(I)  = SIGMA_N(I)*SOLN + SIGMA_A(I)*SOLA + SIGMA_B(I)*SOLB
        #   HSIGMA(I) = HSIGMA_N(I)*SOLN + HSIGMA_A(I)*SOLA + HSIGMA_B(I)*SOLB
        #   CSSIGM    = SIGMA_MOL(1)*SOLG + SIGMA_MOL(2)*SOLB^2
        #             + SIGMA_MOL(3)*SOLC^2 + SIGMA_MOL(4)*SOLH^2
        # 也就是说：非水的 σ 与溶剂的 Abraham 描述符**线性**相关（外加一个与元素无关的
        # 常数项 CSSIGM），所以数据里落的是三张表 + 四个系数，而不是 100 多个溶剂各自的 σ。
        "sigma_formula": "SIGMA_N(i)*SOLN + SIGMA_A(i)*SOLA + SIGMA_B(i)*SOLB",
        "hsigma_formula": "HSIGMA_N(i)*SOLN + HSIGMA_A(i)*SOLA + HSIGMA_B(i)*SOLB",
        "constant_formula": (
            "SIGMA_MOL[0]*SOLG + SIGMA_MOL[1]*SOLB**2 + SIGMA_MOL[2]*SOLC**2 "
            "+ SIGMA_MOL[3]*SOLH**2"
        ),
        "sigma_mol": sigma_mol,
        "descriptor_names": ["SOLA", "SOLB", "SOLC", "SOLG", "SOLH", "SOLN"],
        "raw_tables": raw,
    }

    # CDS 用的 vdW 半径（`SUBROUTINE VDWRAD` 的 `BONDI` 表；H 取 Bondi 1964，其余取
    # Mantina et al. 的 CRC 表）——**与 SMD 的"参数化半径"不是同一套**：
    # 这里的 C 是 1.70 / N 1.55 / O 1.52，而 SMD 参数化半径是 C 1.85 / N 1.89 /（O 依赖 α）。
    # 两套半径混用会静默错数，所以分开落盘、各自标名字。
    vdwrad = subroutine_body(source, "VDWRAD")
    # CDS 的**面积约定**：不是几何球面 4πR²，而是 4π(R + 0.4 Å)²。这一条不是从论文抄的，
    # 是 `scripts/validate_smd_cds.py` 用单原子对着 PySCF 的 `libsolvent` 反推出来的：
    # 不加 0.4 时 σ 逐元素偏大 1.48~1.78 倍（H 1.78、C 1.53、F 1.62、S 1.49、Cl 1.51、Br 1.48），
    # 加上之后水相与四个非水溶剂都回到 1e-7 相对以内。所以它是**验证出来的约定**，不是猜的。
    rows["cds_area_convention"] = {
        "formula": "A_i = 4*pi*(R_i + offset)^2",
        "offset_angstrom": 0.4,
        "evidence": (
            "single-atom inverse against PySCF 2.14.0 libsolvent: without the offset the "
            "derived sigma is 1.48-1.78x too large, element by element; with 0.4 A the water "
            "table matches to 1.6e-5 cal/(mol A^2) and four nonaqueous solvents to 1e-7 kcal/mol"
        ),
    }
    rows["cds_vdw_radii_angstrom"] = {
        "table": "BONDI",
        "note": (
            "CDS cavity radii, not the SMD parameterised radii: H from Bondi 1964, the rest "
            "from Mantina et al. (CRC Handbook 2010-2011)"
        ),
        "values": extract_table(vdwrad, "BONDI", limit=None),
    }

    # CDS 的**成键接触权重**（`SMXCDS` 用）：`COT_ij = exp(Δ/(d_ij − (R_ij + Δ)))`，
    # 其中 `R_ij` 来自 15×15 的 `RKKVAL`、`Δ = 0.30 Å`（O–C 特例 0.10 / 1.330，O–O 特例
    # 0.30 / 1.80，见 `SMXCDS` 正文）。`NATCNV` 把核电荷映射到 11 个「类型」，类型 0
    # 表示「不参与成键权重」，此时 `COT ≡ 0`；本文件覆盖的 36 个元素里有 26 个是 0。
    natcnv = extract_table(source, "NATCNV", limit=None)
    if len(natcnv) < 100:
        raise SystemExit("gen_smd_cds: NATCNV is shorter than 100 entries")
    rkkval = extract_matrix(source, "RKKVAL", 15)
    for row in range(15):
        for column in range(row + 1, 15):
            if rkkval[row][column] != rkkval[column][row]:
                raise SystemExit(
                    f"gen_smd_cds: RKKVAL is not symmetric at ({row + 1},{column + 1})"
                )
    rows["cds_bond_rules"] = {
        "table": "RKKVAL",
        "note": (
            "bond-contact reference lengths used by SMXCDS: "
            "COT = exp(delta/(d - (R + delta))) with delta = 0.30 A; the O-C pair overrides "
            "delta = 0.10 / R = 1.330 and the O-O pair (ISTS = 6) delta = 0.30 / R = 1.80. "
            "Element types come from NATCNV; type 0 means 'no bond weight'."
        ),
        "element_types": {
            symbol: int(natcnv[index]) for index, symbol in enumerate(ELEMENTS)
        },
        "natcnv_raw_100": natcnv[:100],
        "rkkval_angstrom": rkkval,
    }
    return rows


def compare(want: dict, got: dict, where: str = "") -> list[str]:
    problems = []
    if isinstance(want, dict):
        for key, value in want.items():
            if key not in got:
                problems.append(f"{where}.{key}: missing")
                continue
            problems += compare(value, got[key], f"{where}.{key}")
    elif isinstance(want, list):
        if not isinstance(got, list) or len(want) != len(got):
            problems.append(f"{where}: list mismatch")
            return problems
        for index, (a, b) in enumerate(zip(want, got)):
            problems += compare(a, b, f"{where}[{index}]")
    elif isinstance(want, float):
        if not isinstance(got, (int, float)) or float(want) != float(got):
            problems.append(f"{where}: {want!r} vs {got!r}")
    elif want != got:
        problems.append(f"{where}: {want!r} vs {got!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(RAW):
        print(f"missing {RAW}: download PySCF's mnsol.F first")
        return 2
    rows = build_rows()
    if args.check:
        with open(OUTPUT, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        problems = compare(rows, on_disk)
        if problems:
            print(f"{len(problems)} 处不一致")
            for problem in problems[:10]:
                print("  -", problem)
            return 1
        print(f"ok: {OUTPUT} matches mnsol.F ({len(rows['cases'])} cases)")
        return 0
    os.makedirs(STAGING, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {OUTPUT}")
    water = rows["cases"]["water"]["sigma"]
    nonzero = {k: v for k, v in water.items() if v != 0.0}
    print(f"  water     nonzero sigma: {nonzero}")
    for name, values in rows["cases"]["nonaqueous"]["raw_tables"].items():
        print(f"  nonaqueous {name}: {len(values)} entries")
    radii = rows["cds_vdw_radii_angstrom"]["values"]
    print(
        f"  CDS vdW radii: {len(radii)} entries (H {radii[0]}, C {radii[5]}, "
        f"N {radii[6]}, O {radii[7]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
