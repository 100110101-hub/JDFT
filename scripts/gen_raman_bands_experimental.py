"""RS-06（Raman）的**文献侧**：水（气相）三个基频的**绝对 Raman 强度**（Å⁴/amu，实验值）。

`project-phase2.md` §6 的 RS-06 判据要求「per-mode Raman **relative intensity** ordering 与
**文献**一致，至少 H2O/CH4」。本文件钉住这条判据里**可引用**的那一半 —— 文献值本身 ——
并如实说明 CH4 那一半为什么**没有**出现在 CSV 里。

## 一、水：这一份来源与它的原话

来源（正式版）：

    M. Walter, M. Moseler,
    "Ab Initio Wavelength-Dependent Raman Spectra: Placzek Approximation and Beyond",
    J. Chem. Theory Comput. 16 (1), 576–586 (2019).  DOI 10.1021/acs.jctc.9b00584

正式版在 ACS 付费墙后；本文件读的是**作者自存的 arXiv 全文**（`arXiv:1806.03840v2`，
13 Jun 2019，`source_url` 指它），逐字读自该 PDF 的 **Table 2**（第 15 页）。原文照抄
（`pdftotext -layout` 的输出，含其排版错位；脚注上标原样保留）：

    Table 2: Vibrational frequencies in cm−1 and Placzek absolute Raman intensities for
    h̄ωL /Ee → 0 in Å4 /amu. A width of γ = 0.2 eV is applied for the calculations at ωLexp .

                                               ωv
                                               IRam (0)               IRam (ωLexp ≡ 514.5nm)
         mode        ours      exp.    ours          others         ours others        exp.f
                               a    b             c      d     e                 e
         v2 (1A1 ) 1585 1595 , 1638     1.4    0.8 , 0.9 , 1.1       1.4     1.1      0.9±0.2
                               a    b             c      d      e                 e
         v1 (2A1 ) 3747 3657 , 3832    112 109 , 120 , 111           127     129      108±14
                               a    b              c    d    e                   e
         v3 (1B2 ) 3846 3756 , 3943     25      26 , 30 , 26          28      29     19.2±2.1
             a
               exp. vapor , exp. harmonic , PW92 , BP86 , PBE , from ref. 47
                          71 b            72 c        3 d        73 e      74 f

所以三个数是 **IRam(ωL = 514.5 nm) 那一组的 `exp.` 列**：v2 = 0.9±0.2、v1 = 108±14、
v3 = 19.2±2.1 Å⁴/amu。同一篇的 **Table 3**（Albrecht 各分项，ωL → 0）又重复了同一列实验值
（`v2 0.9 / v1 108 / v3 19.2`），两表互相对得上，这不是一处分行读错的产物：

    Table 3: Absolute Raman intensities for ωL = 0 in the Albrecht approximations from our
    calculations and from Gong et al. 30 in Å4 /amu.

          mode    exp 47   Albrecht   Albrecht 30   Albr. A    Albr. A 30   B+C     B + C 30
           v2       0.9         2.7          1.1         0.5          0.1     5.3        1.5
           v1       108         103          105         7.6         11.7      56      48.3
           v3      19.2          30         25.7           0            0      30      25.7

频率两列也一并抄进来了（`wavenumber_vapor_cm1` = `exp. vapor` 脚注 a，`wavenumber_harmonic_cm1`
= `exp. harmonic` 脚注 b）：1595/1638（v2）、3657/3832（v1）、3756/3943（v3）。

### 这条来源的**链条**必须写清楚（不许含糊）

表中 `exp.` 列的脚注是 `f`，而脚注表把 `f` 映射为 `from ref. 47`；ref. 47 是

    A. Stirling, "Raman intensities from Kohn–Sham calculations",
    J. Chem. Phys. 104, 1254–1262 (1996).

也就是说：**这几个实验值是 Walter & Moseler 从 Stirling (1996) 转引的**，我们没有读到
Stirling 原文（AIP 付费墙）。CSV 的 `note` 列把这条转引链写死在每一行里。数值本身是
我们从**已取回的** arXiv PDF 里逐字读到的，不是记忆、不是插图目测。

### 相态 / 温度 / 波长（比较时不许忘掉）

* **相态：气相**（`exp. vapor` 是这一列自己的标注；本项目的计算也是孤立分子，可比）。
* **激发波长：514.5 nm**（表头 `ωLexp ≡ 514.5nm`）。
* **温度：来源没有给**（CSV 里写 `not_stated`）。
* 级别：**实验值**（不是计算值）。同一张表里 `ours/PW92/BP86/PBE` 那几列才是计算。

## 二、CH4：为什么不在这份 CSV 里（以及读到了什么）

判据的另一半是 CH4。**没有**找到任何"逐模式、可机读、且带 DOI"的 CH4 实验强度表。逐个
候选与结局见本轮报告；这里只记最有用的三条**已取回**文字（它们不是 CSV 行，因为都没有
"逐模式数值表"这一性质，且 Hibben 那本 1939 年的书在 Crossref 里**没有 DOI**，
只有一篇 1940 年的书评 `10.1002/ange.19400531511` —— 拿书评的 DOI 冒充书的来源就是伪造）：

1. J. H. Hibben, *The Raman Effect and Its Chemical Applications* (Reinhold, 1939), p. 126
   （Internet Archive `dli.ernet.474921`，公有领域扫描；OCR 与页图 `n128` 互校过，逐字）：

       "The strongest line appearing in methane is at Δν̄ 2914(10) (ρ1). Two other weak lines
        Δν̄ 3022(2) (ρ3) and 3071(1) (2ν2) are present. ... In liquid methane Dauré 46 and
        McLennan, Smith and Wilhelm 108 have observed additional lines."

   即基频只看到 ν1(2914, 强度 10) 与 ν3(3022, 强度 2)，ν2 与 ν4 **没观测到**（同页表 3 里
   CH4 的 ν2、ν4 都写 `unobs.`）。https://archive.org/details/dli.ernet.474921

2. V.-H. Le, 博士论文（Université de Lorraine, 2020），p. 28（HAL `tel-03153454`，全文可下）：

       "La raie située vers 2917 cm−1, correspondant au mode de vibration ν1, est la plus
        intense. ... Les raies des autres modes de vibration du CH4 (ν2, ν3 et ν4), sont de
        très faible intensité, comparées à celle de la raie ν1."

3. M. A. Thomas, H. L. Welsh, "The Raman spectrum of methane", Can. J. Phys. 38, 1291 (1960)
   （Internet Archive `sim_canadian-journal-of-physics_1960-10_38_10`，OCR 可读）：

       "The totally symmetric CH stretching frequency, v1, was observed as a strong unresolved
        Q branch with a maximum at 2917.0±0.1 cm−1."

   摘要同时给出带原点 `v1 = 2916.7, v2 = 1533.6, v3 = 3018.9 cm−1`。

**结论（写进报告，不写进 CSV）**：CH4 的文献只支持"ν1 是最强带，ν2/ν3/ν4 弱得多"这个
**序**，不支持逐模式数值。因此 CH4 的比较只能在"最强带是谁"这一层做，而 H2O 那一份可以
做到逐模式数值 + 序。

## 三、与本文数值的序比对（比较时按这里的顺序，别按频率顺序）

本项目的 Hartree–Fock/STO-3G（`tests/reference/raman_activities.csv` 是 PySCF 侧的同一批数）：

* 水：2169.67 cm⁻¹(ν2, 9.26543) / 4139.68(ν1, 47.81225) / 4390.72(ν3, 21.55134) Å⁴/amu；
* 甲烷：1675.42(ν4×3, 3.84220) / 1903.48(ν2×2, 21.66784) / 3525.59(ν1, 41.72481) /
  3786.13(ν3×3, 21.28044) Å⁴/amu。

**（a）按 activity（Å⁴/amu，与文献同一物理量）**

* 水：本文 ν1 > ν3 > ν2；文献 108 > 19.2 > 0.9 → **序一致**。但量值差得很远：
  本文 ν2 是文献的 10.3 倍、ν1 只有文献的 0.44 倍、ν3 是 1.12 倍 —— 弯曲模被严重高估。
* 甲烷：本文 ν1 > ν2 ≳ ν3 > ν4；文献只有"ν1 最强、其余很弱、1939 年只看到 ν3(2/10)、
  ν2 与 ν4 未观测到"。**"ν1 最强"一致**；**ν2 的名次不一致**（本文 ν2 是 ν1 的 0.52，
  文献里 ν2 低于探测限）。

**（b）按 532 nm 相对强度（本项目口径 I ∝ S·(ν0−ν)⁴/ν，与 CSV 里
`raman_relative_intensity` 逐位吻合：水 61.23/100/39.66，甲烷 21.26/100/69.43/30.78）**

* 水：本文 ν1(100) > ν2(61.2) > ν3(39.7)；把**同一口径**用在文献 activity + 文献气相频率上
  得 ν1(100) > ν3(16.9) > ν2(3.2) → **ν2/ν3 名次相反**。
  原因不是公式，而是（a）里那个 10 倍的 ν2 高估：ν⁴/ν 权重对 ν2/ν3 只有 3.6 倍（用本文
  频率）或 4.1 倍（用文献频率），在文献里翻不动 21 倍的 activity 差（19.2/0.9），在本项目
  里却翻得动 2.33 倍的差（21.55134/9.26543）。
* 甲烷：本文 ν2(100) > ν1(69.4) > ν3(30.8) > ν4(21.3) → **ν1 掉到第二，与"ν1 是最强带"
  的文献结论直接冲突**。

所以：**activity 口径下水的序与文献一致、甲烷只在"最强带"这一层一致；一旦按本项目
532 nm 相对强度口径，两者的序都对不上文献**。这不是"来源不够好"能解释的差别，是弯曲模
activity 被高估的后果，报告里必须点名。

用法：

    python scripts/gen_raman_bands_experimental.py            # 重新生成 CSV
    python scripts/gen_raman_bands_experimental.py --check    # 只比对（不写盘）
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import (  # noqa: E402
    RAMAN_INTENSITY_DOI,
    RAMAN_INTENSITY_URL,
    check_csv,
    rel,
    stamp,
    write_csv,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT,
    "qwb-core",
    "crates",
    "qwb-engine",
    "tests",
    "reference",
    "raman_bands_experimental.csv",
)

COLUMNS = [
    "molecule",
    "phase",
    "mode",
    "symmetry",
    "wavenumber_vapor_cm1",
    "wavenumber_harmonic_cm1",
    "raman_intensity",
    "uncertainty",
    "unit",
    "excitation_nm",
    "temperature",
    "note",
]

#: 逐字读自 arXiv:1806.03840v2 的 Table 2（第 15 页，`exp.` 列，`IRam(ωLexp ≡ 514.5nm)` 那一组）。
#: `note` 里的转引链（`f = from ref. 47` = Stirling, JCP 104, 1254 (1996)）是这一份数据最弱的
#: 一环：实验值本身我们没能读到一手来源。行里不许把这件事抹掉。
ROWS = [
    {
        "molecule": "water",
        "phase": "gas",
        "mode": "v2 (bending)",
        "symmetry": "1A1",
        "wavenumber_vapor_cm1": "1595",
        "wavenumber_harmonic_cm1": "1638",
        "raman_intensity": "0.9",
        "uncertainty": "0.2",
        "unit": "angstrom^4/amu",
        "excitation_nm": "514.5",
        "temperature": "not_stated",
        "note": "Table 2 exp. column (IRam at omega_L = 514.5 nm); footnote f = 'from ref. 47' = Stirling, JCP 104, 1254 (1996); repeated in Table 3 exp column (omega_L -> 0)",
    },
    {
        "molecule": "water",
        "phase": "gas",
        "mode": "v1 (symmetric stretch)",
        "symmetry": "2A1",
        "wavenumber_vapor_cm1": "3657",
        "wavenumber_harmonic_cm1": "3832",
        "raman_intensity": "108",
        "uncertainty": "14",
        "unit": "angstrom^4/amu",
        "excitation_nm": "514.5",
        "temperature": "not_stated",
        "note": "Table 2 exp. column (IRam at omega_L = 514.5 nm); footnote f = 'from ref. 47' = Stirling, JCP 104, 1254 (1996); repeated in Table 3 exp column (omega_L -> 0)",
    },
    {
        "molecule": "water",
        "phase": "gas",
        "mode": "v3 (antisymmetric stretch)",
        "symmetry": "1B2",
        "wavenumber_vapor_cm1": "3756",
        "wavenumber_harmonic_cm1": "3943",
        "raman_intensity": "19.2",
        "uncertainty": "2.1",
        "unit": "angstrom^4/amu",
        "excitation_nm": "514.5",
        "temperature": "not_stated",
        "note": "Table 2 exp. column (IRam at omega_L = 514.5 nm); footnote f = 'from ref. 47' = Stirling, JCP 104, 1254 (1996); repeated in Table 3 exp column (omega_L -> 0)",
    },
]


def rows():
    return stamp(ROWS, RAMAN_INTENSITY_DOI, RAMAN_INTENSITY_URL)


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
