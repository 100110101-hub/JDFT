"""RS-04 第三条判据的实验侧：CH4 / CH3OH 的 ¹H 与 ¹³C 化学位移（TMS = 0）。

## 为什么用这一份来源

`project-phase2.md` §6 的 RS-04 要「¹H 化学位移与实验差 < 0.5 ppm（CH4 / CH3OH 的更宽）」。
实验值必须是**可核对的文献值**，不是记忆里的"常识"，也不是我们自己算的。取用：

    G. R. Fulmer, A. J. M. Miller, N. H. Sherden, H. E. Gottlieb, A. Nudelman,
    B. M. Stoltz, J. E. Bercaw, K. I. Goldberg,
    "NMR Chemical Shifts of Trace Impurities: Common Laboratory Solvents, Organics, and
     Gases in Deuterated Solvents Relevant to the Organometallic Chemist",
    Organometallics 29 (9), 2176–2179 (2010).  DOI 10.1021/om100106e

它的支持信息里两张大表（Table S1 = ¹H，Table S2 = ¹³C）把同一批化合物在 **12 种氘代溶剂**
里的位移并排列出，其中就同时有 methane 与 methanol —— 这正是"一次来源、两种核、两个分子"
能对齐的地方。ACS 的 SI 直链被 403 挡住，所以 `source_url` 指向 CaltechAUTHORS 上的那份
支持信息（同一个文件、同一份内容，可下载）。

**取的是 CDCl3 那一列**（第 3 列；两个表的列序相同：THF-d8, CD2Cl2, CDCl3, toluene-d8,
C6D6, C6D5Cl, (CD3)2CO, (CD3)2SO, CD3CN, TFE-d3, CD3OD, D2O）。列的识别靠表头同一行给的
残余溶剂信号（¹H 7.26 ppm、¹³C 77.16 ppm）核对过，不靠数位置。

## 这一份数据的**相态**（重要，不许在比较时忘掉）

实验值是 **CDCl3 稀溶液**的位移（Fulmer 表里 methane 是把气体溶进溶剂测的），而本项目的
GIAO 计算是**真空中孤立分子**。CSV 的 `phase` / `solvent` 两列把这件事写死在数据里；
`solvent_spread_ppm` 列给出同一张表里 12 种溶剂之间的**实测跨度** —— 于是"溶剂效应有多大"
是一个引用了来源的数字，而不是一句解释性的话。

## TMS 的两行

TMS 是这套 ppm 刻度的零点（`δ = σ(TMS) − σ(sample)`），所以它的位移是 0.00 ppm **按定义**，
不是测量值。CSV 里留两行把它写实：比较公式的两端都能在这份文件里找到。

用法：

    python scripts/gen_nmr_shifts_experimental.py            # 重新生成 CSV
    python scripts/gen_nmr_shifts_experimental.py --check    # 只比对（不写盘）
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import FULMER_DOI, FULMER_URL, check_csv, rel, stamp, write_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT,
    "qwb-core",
    "crates",
    "qwb-engine",
    "tests",
    "reference",
    "nmr_shifts_experimental.csv",
)

#: `solvent_spread_ppm` = 同一张表里 12 种氘代溶剂之间的 [最小, 最大]（含端点，逐字读自 SI）。
#: 它**不是**误差棒，是"溶液相把位移挪动多少"的实测跨度。
COLUMNS = [
    "molecule",
    "nucleus",
    "site",
    "shift_ppm",
    "phase",
    "solvent",
    "temperature",
    "solvent_spread_ppm",
    "note",
]

#: 逐字读自支持信息 Table S1 / Table S2 的 CDCl3 一列（第 3 列）。
#: 一个数一个数对过原始 PDF 的那一行：
#:
#:   Table S1（¹H）: "methane CH4 s 0.19 0.21 0.22 0.17 0.16 0.15 0.17 0.20 0.20 0.18 0.20 0.18"
#:                   "methanol CH3 s  3.27 3.42 3.49 3.03 3.07 3.25 3.31 3.16 3.28 3.44 3.34 3.34"
#:   Table S2（¹³C）: "methane CH4 –4.90 –4.33 –4.63 –4.34 –4.29 –4.33 –5.33 –4.01 –4.61 –5.88 –4.90 -"
#:                   "methanol CH3  49.64 50.45 50.41 49.90 49.97 49.66 49.77 48.59 49.90 50.67 49.86 49.50"
ROWS = [
    {
        "molecule": "methane",
        "nucleus": "1H",
        "site": "CH4",
        "shift_ppm": "0.22",
        "phase": "solution",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "0.15-0.22",
        "note": "Table S1 / CDCl3 column",
    },
    {
        "molecule": "methane",
        "nucleus": "13C",
        "site": "CH4",
        "shift_ppm": "-4.63",
        "phase": "solution",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "-5.88--4.01",
        "note": "Table S2 / CDCl3 column",
    },
    {
        "molecule": "methanol",
        "nucleus": "1H",
        "site": "CH3",
        "shift_ppm": "3.49",
        "phase": "solution",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "3.03-3.49",
        "note": "Table S1 / CDCl3 column (the OH proton is exchange-broadened; not used)",
    },
    {
        "molecule": "methanol",
        "nucleus": "13C",
        "site": "CH3",
        "shift_ppm": "50.41",
        "phase": "solution",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "48.59-50.67",
        "note": "Table S2 / CDCl3 column",
    },
    {
        "molecule": "tms",
        "nucleus": "1H",
        "site": "CH3",
        "shift_ppm": "0.00",
        "phase": "definition",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "",
        "note": "zero of the ppm scale by definition (delta = sigma(TMS) - sigma)",
    },
    {
        "molecule": "tms",
        "nucleus": "13C",
        "site": "CH3",
        "shift_ppm": "0.00",
        "phase": "definition",
        "solvent": "chloroform-d",
        "temperature": "not_stated",
        "solvent_spread_ppm": "",
        "note": "zero of the ppm scale by definition (delta = sigma(TMS) - sigma)",
    },
]


def rows():
    return stamp(ROWS, FULMER_DOI, FULMER_URL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写盘")
    args = parser.parse_args()

    if args.check:
        return check_csv(OUTPUT, COLUMNS, rows())

    write_csv(OUTPUT, COLUMNS, rows())
    count = sum(1 for _ in io.open(OUTPUT, encoding="utf-8", newline="")) - 1
    print(f"wrote {rel(OUTPUT)} ({count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
