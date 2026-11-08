"""RP-01 的外部尺子：三个反应的势垒，逐行钉在**来源自己说过的话**上（DA-08）。

`project-phase2.md` §6 的 RP-01 判据是「三个反应（HCN↔HNC、CH₃NC↔CH₃CN、H₂CO→H₂+CO）的
势垒与**文献**差 < 1.5 kcal/mol（DFT 水平）」。这个判据要成立，先得有三条**可引用的**文献值
—— 而且必须知道每个数的**理论级别**与**含不含零点能**，否则"差 1.5 kcal/mol"是一句空话。
`docs/status-v2.yaml` 里的 "M06-2X/6-31G(d)-class" 是本项目自己写的猜测，不是文献值，本轮
用下面三份来源替换它。

## 三条来源与它们各自说的原话

**(1) HCN ↔ HNC** —— V. Yu. Makhnev, A. A. Kyuberis, N. F. Zobov, L. Lodi, J. Tennyson,
O. L. Polyansky, "High Accuracy ab Initio Calculations of Rotational–Vibrational Levels of the
HCN/HNC System", J. Phys. Chem. A **122**, 1326–1343 (2018). DOI `10.1021/acs.jpca.7b10483`。

摘要原话：

  "The height of the isomerisation barrier, the isomerisation energy and the dissociation
   energy were computed using a number of models; our best results are 16 809.4 cm−1,
   5 312.8 cm−1 and 43 729 cm−1, respectively."

级别（正文）：全局 PES 的主表面是 ic-MRCI（Pople 修正，aug-cc-pCV6Z，全电子相关，1541 个
几何点的拟合标准偏差 2.6 cm⁻¹），上面再加 BODC 与标量相对论修正面。判据用它报的
16 809.4 cm⁻¹（Table 13 的 "+P fix" 一列；同表四列跨度 16 765.9–16 809.4 cm⁻¹）。

**零点能：不含**。Table 13 的分解是 `∆Ebarrier = ∆EBO + ∆EBODC + ∆Erel`
（16 821.2 + 7.3 − 19.1 = 16 809.4 ✓），而同一个来源的 Table 12 里零点能是**单独一行**
（`∆EZPE = −109.1 / −98.7 / −108.5 / −100.3 cm⁻¹`）。所以这条势垒是可以与本项目
**电子**势垒直接比的 —— 这是三个反应里唯一不需要 ZPE 折算的一个。

**(2) H₂CO → H₂ + CO** —— D. Feller, M. Dupuis, B. C. Garrett, "Barrier for the H2CO→H2+CO
reaction: A discrepancy between high-level electronic structure calculations and experiment",
J. Chem. Phys. **113**, 218–226 (2000). DOI `10.1063/1.481788`。

摘要原话：

  "Our best estimate of the barrier height, including zero point vibrational effects, is
   81.9±0.3 kcal/mol, almost 3 kcal/mol larger than the experimental value of 79.2±0.8 kcal/mol.
   This estimate includes corrections for the effects of finite basis set truncation (which is
   negligible at the quintuple zeta level), higher order correlation recovery, core/valence
   correlation, and scalar relativistic effects."

**零点能：含**（"including zero point vibrational effects" 是来源自己写的）。这条与本项目的
电子势垒**不可直接比** —— CSV 的 `zpe_included` 列把这件事写死在数据里，比较时不许忘掉。
来源同时给出实验值 79.2 ± 0.8 kcal/mol，作为第二行（`role = experimental`）。

**(3) CH₃NC → CH₃CN** —— L. T. Redmon, G. D. Purvis, R. J. Bartlett, "The unimolecular
isomerization of methyl isocyanide to methyl cyanide (acetonitrile)", J. Chem. Phys. **69**,
5386–5392 (1978). DOI `10.1063/1.436568`。

摘要原话：

  "Using double-zeta plus polarization (DZP) contracted Gaussian basis sets (63 CGTO's),
   correlated calculations are made by evaluating all second-, third-, and fourth-order
   diagrams that occur due to single, double, and quadruple excitations [SDQ–MBPT(4)]. ...
   The SDQ–MBPT(4) model predicts the enthalpy of isomerization to be −22.7 kcal/mole
   [exp(−23.7 ± 0.14)]. The activation barrier is found to be ∼41 kcal/mole (exp 38.4).
   The inclusion of quadrupole excitation diagrams has little effect on the enthalpy of
   isomerization but increases the activation barrier by 1 kcal/mole."

级别：SDQ–MBPT(4)/DZP。**零点能：来源没写**（摘要里把电子能差直接叫作 "enthalpy" 并与
实验比，但没说是否加了 ZPE）—— CSV 里 `zpe_included = not_stated`，这是这一条最弱的地方，
报告里必须点名。另外这个数在来源里就是 **"∼41"**，而且来源自己说四重激发贡献 **1 kcal/mol**
的量级 —— 也就是说**这条尺子本身的精度（±1 kcal/mol）就已经吃掉了 1.5 kcal/mol 判据的
大半**。不是我们算得不够准，是这条文献值不够精确。

## 三条来源的可读范围（如实记账）

* (1) 读到的是 arXiv/仓库绿 OA 的**接受稿 PDF**（UCL Discovery，`source_url` 指它），全文
  可读 —— 所以 Table 13 的分解与 Table 12 的 ZPE 行都是**读到的**，不是猜的。
* (2)(3) 只有**摘要**（AIP 正文在付费墙后，项目没有订阅）。摘要是来源自己的话，所以
  "它说的是什么"这一栏是可靠的；但 (2) 的 **ΔZPE 数值**（鞍点与反应物的谐振频率）读不到，
  所以本项目的比较只能用"电子 vs 含 ZPE"并把这个差**另行量化**（报告里说明用的是什么）。
* 三份来源的 DOI 都用 Crossref 逐条核过题名/卷/页/年份（见 `reference_sources.py` 的常量）。

用法：

    python scripts/gen_rp01_literature.py            # 重新生成 CSV
    python scripts/gen_rp01_literature.py --check    # 只比对（不写盘）
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_sources import (  # noqa: E402
    RP01_H2CO_DOI,
    RP01_H2CO_URL,
    RP01_HCN_DOI,
    RP01_HCN_URL,
    RP01_MECN_DOI,
    RP01_MECN_URL,
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
    "rp01_literature_barriers.csv",
)

#: 波数 → kcal/mol 的两个常数（CODATA 2018）。**只用在这一个转换上**，而且每行同时保留
#: `value_as_stated`（来源里的原值与原单位），所以读者可以用任何常数重算，不必相信这个。
CM_PER_HARTREE = 219474.631363
KCAL_PER_HARTREE = 627.5094740631

COLUMNS = [
    "reaction",
    "role",
    "barrier_kcal_per_mol",
    "uncertainty_kcal_per_mol",
    "value_as_stated",
    "unit_as_stated",
    "zpe_included",
    "level_of_theory",
    "quote",
    "note",
]

#: 逐行钉住的文献值。`role = computed` 是来源算的，`role = experimental` 是来源引的实验值
#: （只作参照，**不是** RP-01 判据要比的那一栏）。
ROWS = [
    {
        "reaction": "r1_hcn_hcnc",
        "role": "computed",
        "barrier_kcal_per_mol": f"{16809.4 * KCAL_PER_HARTREE / CM_PER_HARTREE:.4f}",
        "uncertainty_kcal_per_mol": f"{25.0 * KCAL_PER_HARTREE / CM_PER_HARTREE:.4f}",
        "value_as_stated": "16809.4",
        "unit_as_stated": "cm^-1",
        "zpe_included": "no",
        "level_of_theory": (
            "ic-MRCI/aug-cc-pCV6Z (all electrons correlated; Pople size-extensivity "
            "correction) + BODC + scalar-relativistic correction surfaces. 16809.4 cm-1 is the "
            "'+P fix.' column of Table 13; the global PES uses the relaxed-reference variant "
            "(+P rel. = 16796.3 cm-1)."
        ),
        "quote": (
            "The height of the isomerisation barrier, the isomerisation energy and the "
            "dissociation energy were computed using a number of models; our best results "
            "are 16 809.4 cm-1, 5 312.8 cm-1 and 43 729 cm-1, respectively."
        ),
        "note": (
            "Table 13 (+P fix column) 16 809.4 = 16 821.2 (Delta-E_BO) + 7.3 (Delta-E_BODC) "
            "- 19.1 (Delta-E_rel); no Delta-E_ZPE row in that table, while Table 12 of the "
            "same paper lists Delta-E_ZPE separately -> the barrier is ZPE-free. The four "
            "size-extensivity variants span 16 765.9-16 809.4 cm-1 (0.12 kcal/mol); the "
            "paper states the size-extensivity error is 'about 25 cm-1'."
        ),
    },
    {
        "reaction": "r2_ch3nc_ch3cn",
        "role": "computed",
        "barrier_kcal_per_mol": "41.0",
        "uncertainty_kcal_per_mol": "1.0",
        "value_as_stated": "~41",
        "unit_as_stated": "kcal/mol",
        "zpe_included": "not_stated",
        "level_of_theory": "SDQ-MBPT(4)/DZP (63 CGTOs), MBPT(2,3,4) diagram summation",
        "quote": (
            "The SDQ-MBPT(4) model predicts the enthalpy of isomerization to be -22.7 kcal/mole "
            "[exp(-23.7 +- 0.14)]. The activation barrier is found to be ~41 kcal/mole "
            "(exp 38.4). The inclusion of quadrupole excitation diagrams has little effect on "
            "the enthalpy of isomerization but increases the activation barrier by 1 kcal/mole."
        ),
        "note": (
            "The source states the barrier only as '~41'; the 1.0 kcal/mol uncertainty is the "
            "paper's own statement that the quadruple-excitation diagrams are worth 1 kcal/mol "
            "(and SDQ already contains Q, so which of 40/41 is the SDQ number is ambiguous in "
            "the abstract). Whether ZPE is included is not stated in the abstract; the abstract "
            "calls an electronic energy difference an 'enthalpy'. Only the abstract was "
            "readable (AIP paywall), so this row's resolution cannot be improved from the "
            "full text here."
        ),
    },
    {
        "reaction": "r2_ch3nc_ch3cn",
        "role": "experimental",
        "barrier_kcal_per_mol": "38.4",
        "uncertainty_kcal_per_mol": "",
        "value_as_stated": "38.4",
        "unit_as_stated": "kcal/mol",
        "zpe_included": "yes",
        "level_of_theory": "experimental activation energy (as quoted by the source)",
        "quote": "The activation barrier is found to be ~41 kcal/mole (exp 38.4).",
        "note": (
            "Quoted by the same source as the comparison value; no uncertainty is given in the "
            "abstract. Not the RP-01 criterion value (RP-01 compares to the computed literature "
            "barrier), kept in the file so the provenance of the comparison is visible."
        ),
    },
    {
        "reaction": "r3_h2co_h2_co",
        "role": "computed",
        "barrier_kcal_per_mol": "81.9",
        "uncertainty_kcal_per_mol": "0.3",
        "value_as_stated": "81.9 +- 0.3",
        "unit_as_stated": "kcal/mol",
        "zpe_included": "yes",
        "level_of_theory": (
            "CCSD(T)/CCSDT-type correlation with CBS extrapolation, core/valence correlation, "
            "scalar relativistic; best estimate at the CCSD(T)/CBS-limit level"
        ),
        "quote": (
            "Our best estimate of the barrier height, including zero point vibrational effects, "
            "is 81.9+-0.3 kcal/mol, almost 3 kcal/mol larger than the experimental value of "
            "79.2+-0.8 kcal/mol. This estimate includes corrections for the effects of finite "
            "basis set truncation (which is negligible at the quintuple zeta level), higher "
            "order correlation recovery, core/valence correlation, and scalar relativistic "
            "effects."
        ),
        "note": (
            "Includes zero point vibrational effects (the source says so), so it is NOT directly "
            "comparable to an electronic barrier; the comparison must be labelled "
            "electronic-vs-ZPE-inclusive unless Delta-ZPE is quantified. The full text is "
            "paywalled, so the harmonic frequencies needed to derive Delta-ZPE were not "
            "readable from this source."
        ),
    },
    {
        "reaction": "r3_h2co_h2_co",
        "role": "experimental",
        "barrier_kcal_per_mol": "79.2",
        "uncertainty_kcal_per_mol": "0.8",
        "value_as_stated": "79.2+-0.8",
        "unit_as_stated": "kcal/mol",
        "zpe_included": "yes",
        "level_of_theory": "experimental (as quoted by the source)",
        "quote": (
            "Our best estimate of the barrier height, including zero point vibrational effects, "
            "is 81.9+-0.3 kcal/mol, almost 3 kcal/mol larger than the experimental value of "
            "79.2+-0.8 kcal/mol."
        ),
        "note": (
            "The experimental value quoted by the same source; itself ZPE/thermally based, and "
            "the source's point is precisely that theory and experiment disagree by ~3 kcal/mol "
            "on this reaction. Kept as provenance, not as the RP-01 criterion value."
        ),
    },
]


def rows():
    """补上两列来源；每一行按它属于哪个反应挑 DOI/URL。"""
    out = []
    for row in ROWS:
        if row["reaction"] == "r1_hcn_hcnc":
            doi, url = RP01_HCN_DOI, RP01_HCN_URL
        elif row["reaction"] == "r2_ch3nc_ch3cn":
            doi, url = RP01_MECN_DOI, RP01_MECN_URL
        else:
            doi, url = RP01_H2CO_DOI, RP01_H2CO_URL
        out.extend(stamp([row], doi, url))
    return out


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
