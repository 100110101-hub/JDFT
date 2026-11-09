"""参考集 CSV 的来源标记（DA-08）：一份来源，三条约束。

DA-08 的判据是「每行有 `source_doi`/`source_url`；脚本可重建（`scripts/gen_*.py --check`
绿）；缺来源即门禁失败」。三者分别落在：

* **每行**：本模块的 [`stamp`] 给每一行补上两列，生成器不许自己手写这两列；
* **可重建**：每个 `scripts/pyscf_*_reference.py` 都有 `--check`，用同一份代码重算并逐字
  比对文件（`--check` 不写盘）；
* **门禁失败**：`scripts/check_reference_sources.py` 在 `scripts/verify.ps1` /
  `scripts/verify.sh` 里跑，只依赖标准库（不 import PySCF），所以 Windows 门禁也能查。

## 来源怎么填

参考值不是抄来的常数，而是**用一个独立实现算出来的**（PySCF 2.14.0 黑盒，§14 #12）。
所以 `source_doi` 填的是那个实现所依据的**论文**（Crossref 已核对题名与卷页），
`source_url` 填它的仓库地址；具体版本与算法写在每行的 `note` 里。

这两件事必须分开记：`source_doi` 说明「这个数是用什么东西算的」，`note` 说明
「怎么算的（哪个模块、什么近似）」。把版本号塞进 DOI 字段、或者用「PySCF」当 DOI，
都是错的。
"""

from __future__ import annotations

import csv
import io
import os

__all__ = [
    "FAMILIES",
    "GENERATORS",
    "DFTD3_DOI",
    "DFTD3_URL",
    "DFTD4_DOI",
    "DFTD4_URL",
    "S22_DOI",
    "S22_URL",
    "FULMER_DOI",
    "FULMER_URL",
    "RP01_HCN_DOI",
    "RP01_HCN_URL",
    "RP01_H2CO_DOI",
    "RP01_H2CO_URL",
    "RP01_MECN_DOI",
    "RP01_MECN_URL",
    "PYSCF_DOI",
    "PYSCF_URL",
    "SOURCE_COLUMNS",
    "check_csv",
    "rel",
    "stamp",
    "write_csv",
]

#: PySCF 2.x 的论文（Sun et al., J. Chem. Phys. 153, 024109 (2020)）。
#: 2026-11-08 用 Crossref `api.crossref.org/works/10.1063/5.0006074` 核对过题名与卷期。
PYSCF_DOI = "10.1063/5.0006074"
#: PySCF 源码仓库（参考值的实际执行环境是这里的 2.14.0）。
PYSCF_URL = "https://github.com/pyscf/pyscf"

#: D3(BJ) 的原始论文（Grimme et al., J. Chem. Phys. 132, 154104 (2010)）。
#: 2026-11-09 用 Crossref `api.crossref.org/works/10.1063/1.3382344` 核对过题名与卷号。
DFTD3_DOI = "10.1063/1.3382344"
#: 官方 s-dftd3 实现（色散参考值的黑盒尺子）。
DFTD3_URL = "https://github.com/dftd3/simple-dftd3"

#: D4 的原始论文（Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme,
#: J. Chem. Phys. 150, 154122 (2019)）。
#: 2026-11-10 用 Crossref `api.crossref.org/works/10.1063/1.5090222` 核对过题名（
#: "A generally applicable atomic-charge dependent London dispersion correction"）、
#: 卷 150 / 期 15 / 文章号 154122 与年份 2019。
DFTD4_DOI = "10.1063/1.5090222"
#: 官方 dftd4 实现（D4 参考值的黑盒尺子；本机装的是 pip 包 dftd4 4.3.1）。
DFTD4_URL = "https://github.com/dftd4/dftd4"

#: S22 弱相互作用二聚体集的**几何**来源（Jurečka, Šponer, Černý, Hobza, PCCP 8, 1985 (2006)）。
#: 2026-11-13 用 Crossref `api.crossref.org/works/10.1039/b600027d` 核对过题名、卷 8 与页码 1985–1993。
S22_DOI = "10.1039/b600027d"
#: 参考相互作用能取自 Takatani 等的"基组一致修订值"（JCP 132, 144104 (2010)），
#: 同上用 Crossref 核对过（`10.1063/1.3378024`）；数据文件本身从 ASE 的 `ase/data/s22.py`
#: 取回并按 sha256 钉住（LGPL-2.1-or-later，文件名与引用保留在 staging JSON 里）。
#: 这里记 URL 用 ASE 的 GitLab 源（数据文件的位置），DOI 用 Takatani 的（数值的位置）。
S22_URL = "https://gitlab.com/ase/ase/-/raw/master/ase/data/s22.py"

#: FreeSolv（Mobley & Guthrie, J. Comput. Aided Mol. Des. 28, 711 (2014)）—— **CC0** 的
#: 水相水合自由能；每个值自带它引用的原始实验文献 DOI（写进 CSV 的 `expt_reference_doi`）。
#: 2026-11-13 用 Crossref 核过题名与期刊："FreeSolv: a database of experimental and calculated
#: hydration free energies, with input files"。
FREESOLV_DOI = "10.1007/s10822-014-9747-x"
FREESOLV_URL = "https://github.com/MobleyLab/FreeSolv"

#: RS-04（¹H/¹³C 化学位移 vs 实验）：Fulmer, Miller, Sherden, Gottlieb, Nudelman, Stoltz,
#: Bercaw, Goldberg, Organometallics 29, 2176–2179 (2010) —— 常见溶剂/有机物/**气体**在
#: 12 种氘代溶剂里的 ¹H 与 ¹³C 化学位移表（支持信息 Table S1/S2）。数的是**溶液**里的位移
#: （CDCl3 那一列），不是气相 —— 这个差别写在 CSV 的 `phase`/`solvent` 两列里，比较时不许
#: 凭记忆把它抹掉。2026-11-14 用 Crossref `api.crossref.org/works/10.1021/om100106e` 核对过
#: 题名（"NMR Chemical Shifts of Trace Impurities: ..."）、卷 29 / 期 9 / 页 2176–2179 与
#: 八位作者；URL 是 CaltechAUTHORS 上那份支持信息的**直链**（ACS 的 SI 地址被 403 挡住）。
FULMER_DOI = "10.1021/om100106e"
FULMER_URL = (
    "https://authors.library.caltech.edu/records/q3m5b-fec90/files/"
    "om100106e_si_001.pdf?download=1"
)

#: RP-01（势垒 vs 文献）：三条**外部**来源，一条反应一条。这三个 DOI 都用 Crossref
#: `api.crossref.org/works/<doi>` 核对过题名、卷、页、年份与摘要原文（2026-11-15）。
#:
#: 1. HCN ↔ HNC：Makhnev et al., J. Phys. Chem. A 122, 1326–1343 (2018)。值 16 809.4 cm⁻¹
#:    （ic-MRCI/aug-cc-pCV6Z + BODC + 标量相对论，**不含** ZPE）。URL 指 UCL Discovery 上
#:    那份**接受稿 PDF** —— 全文可读，Table 12/13 的分解就是从那上面读的。
RP01_HCN_DOI = "10.1021/acs.jpca.7b10483"
RP01_HCN_URL = "https://discovery.ucl.ac.uk/id/eprint/10047737/1/hcn27.pdf"
#: 2. CH₃NC → CH₃CN：Redmon, Purvis, Bartlett, J. Chem. Phys. 69, 5386–5392 (1978)。
#:    值 "∼41 kcal/mol"（SDQ–MBPT(4)/DZP，ZPE 未说明）。正文在付费墙后，只读到摘要。
RP01_MECN_DOI = "10.1063/1.436568"
RP01_MECN_URL = "https://doi.org/10.1063/1.436568"
#: 3. H₂CO → H₂ + CO：Feller, Dupuis, Garrett, J. Chem. Phys. 113, 218–226 (2000)。
#:    值 81.9 ± 0.3 kcal/mol（**含** ZPE，来源原话）。正文在付费墙后，只读到摘要。
RP01_H2CO_DOI = "10.1063/1.481788"
RP01_H2CO_URL = "https://doi.org/10.1063/1.481788"

#: 每一行都必须带的两个来源列（加在最后一列之后，位置固定）。
SOURCE_COLUMNS = ("source_doi", "source_url")

#: DA-08 已经交付的「家族文件」—— 门禁只强制这批（历史数值转储如 `angmom.csv` 没有
#: 逐行来源可言，硬塞进去只会变成噪声；它们的出处写在生成脚本与账本里）。

#: 每个家族文件由哪个脚本重生成（`--check` 用同一份代码比对）。门禁会检查脚本存在且
#: 带 `--check` 分支 —— 「可重建」不许只是嘴上说说。
GENERATORS = {
    "qwb-core/crates/qwb-response/tests/reference/tda_excitations.csv":
        "scripts/pyscf_tda_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/tda_benzene.csv":
        "scripts/pyscf_tda_benzene_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/tda_formaldehyde.csv":
        "scripts/pyscf_tda_formaldehyde_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/tda_acetone.csv":
        "scripts/pyscf_tda_acetone_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/tddft_excitations.csv":
        "scripts/pyscf_tdh_reference.py",
    # F-01/F-02 + `jobtype: freq`：PySCF 的**解析** Hessian 给出的简正模频率与约化质量
    # （本项目的数值 Hessian 对同一条链的独立实现）。
    # SV-03：PySCF 的 SMD（同一几何/基组）给出的 G_CDS、静电与总溶剂化能 —— 本项目的
    # SMD 接线（C-PCM 表面 + eq. 4.77）对这条整条链的独立实现。
    # VF-02：水相实验水合自由能（FreeSolv，CC0）。8 溶剂 × 3 分子的完整集卡在 MNSOL 的
    # WAF 403（见 status-v2.yaml 的 VF-02），所以先交付水这一列。
    "qwb-core/crates/qwb-engine/tests/reference/solvation_free_energy.csv":
        "scripts/gen_freesolv_reference.py",
    # SV-03：FreeSolv 的**全量受限子集**（642 → 457：元素 ⊆ {H,C,N,O,F}、单一分子、净电荷 0）。
    # 与上面那一份同源同一个 pin（`database.json` 620,456 B / sha256 9133b243…），列不同：
    # 这一份每行带 freesolv_id / SMILES / 分子式 / 原子数 / 实验 ΔG_hyd 与它引用的原始文献，
    # 是 SV-03 判据被**实测**而不是被 2 行样本猜出来的依据。
    "qwb-core/crates/qwb-engine/tests/reference/freesolv_hydration.csv":
        "scripts/gen_freesolv_reference.py",
    "qwb-core/crates/qwb-engine/tests/reference/smd_water.csv":
        "scripts/pyscf_smd_reference.py",
    "qwb-core/crates/qwb-engine/tests/reference/hessian_water.csv":
        "scripts/pyscf_hessian_reference.py",
    # RS-06（Raman 活性）：PySCF 侧的**解析 Hessian**（模式）+ **有限场极化率**（α）+
    # numpy 投影给出的 `45 ᾱ'² + 7 γ'²`。本仓库那一半是数值 Hessian + 自研 CPHF 的 α +
    # 中心差分 —— 两条路只在「中心差分」这一层是共同的数学（Hessian、SCF、极化率、投影
    # 四条都是各自独立的实现）。
    "qwb-core/crates/qwb-engine/tests/reference/raman_activities.csv":
        "scripts/pyscf_raman_reference.py",
    "qwb-core/crates/qwb-props/tests/reference/epr_spin_density.csv":
        "scripts/pyscf_epr_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/dynamic_polarizability.csv":
        "scripts/pyscf_dynamic_polarizability_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/beta.csv":
        "scripts/pyscf_beta_ref.py",
    "qwb-core/crates/qwb-props/tests/reference/nmr_molecules.csv":
        "scripts/pyscf_nmr_shieldings_reference.py",
    "qwb-core/crates/qwb-props/tests/reference/nmr_diamagnetic.csv":
        "scripts/pyscf_nmr_shieldings_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/paramagnetic.csv":
        "scripts/pyscf_nmr_shieldings_reference.py",
    # RS-04（GIAO）：PySCF 侧的 GIAO-HF 屏蔽总和与 GIAO Fock/重叠导数。PySCF 2.14.0 的这个
    # 构建**没有 `pyscf.prop`**（PyPI 的 wheel 与 sdist 都没有那个子包，见
    # `docs/choices.md` 八十一），所以生成器把 PySCF 自己的 GIAO 实现
    # （`pyscf/prop/nmr/rhf.py` v1.7.6 源码）逐步转写到脚本里：积分走 PySCF 的 libcint
    # GIAO 积分（`int1e_giao_a11part` / `int1e_a01gp` / `int1e_giao_irjxp` / `int1e_ig*` /
    # `int2e_ig1`），SCF/CPHF 也走 PySCF。转写的真伪由那个文件 `__main__` 里的自检值钉住
    # （`--self-test`：FH/6-31G 的 375.232839 / 483.002139 / 449.032227）。
    "qwb-core/crates/qwb-response/tests/reference/giao_nmr.csv":
        "scripts/pyscf_giao_nmr_reference.py",
    "qwb-core/crates/qwb-response/tests/reference/giao_fock_derivative.csv":
        "scripts/pyscf_giao_nmr_reference.py",
    # RS-04（GIAO dia）：libcint 的 `int1e_giao_a11part` / `int1e_a01gp` 逐元素参考值
    # （水 × {STO-3G, LANL2DZ} × 三个核 × 9 个 (α,β) 块 × 两个算子 = 11772 个元素）。
    # 两个算子的定义都是**测量**出来的（见生成脚本顶部：a01gp 是「相位导数 × 超精细算符」，
    # 含绝对电子坐标 r，所以对整体平移不不变 —— 参考值因此只能在 libcint 的
    # `with_rinv_origin` 约定下取）。
    "qwb-core/crates/qwb-integrals/tests/reference/giao_dia.csv":
        "scripts/pyscf_giao_dia_reference.py",
    # DA-06 / FX-04：色散校正的参考能量与解析梯度来自官方 s-dftd3（黑盒），几何与参考值
    # 出自同一个脚本（`d3_dimers.csv` 是几何的单一真相源）。
    "qwb-core/crates/qwb-scf/tests/reference/d3_dimers.csv":
        "scripts/dftd3_dispersion_reference.py",
    "qwb-core/crates/qwb-scf/tests/reference/d3_dispersion.csv":
        "scripts/dftd3_dispersion_reference.py",
    "qwb-core/crates/qwb-xc/tests/reference/spin_mgga_pointwise.csv":
        "scripts/pyscf_spin_mgga_reference.py",
    "qwb-core/crates/qwb-scf/tests/reference/d3zero_dispersion.csv":
        "scripts/dftd3_dispersion_reference.py",
    # SC-06：开壳层 UKS × meta-GGA 的 **SCF 总能量**（O2 三重态 / STO-3G），走 PySCF 的
    # `dft.UKS` 黑盒。逐点参考（`spin_mgga_pointwise.csv`）钉公式，这一份钉接线。
    "qwb-core/crates/qwb-scf/tests/reference/uks_mgga.csv":
        "scripts/pyscf_uks_mgga_reference.py",
    # DA-06（D4）：官方 dftd4 4.3.1 黑盒给出的色散能/梯度与中间量（CN、EEQ 电荷、
    # 极化率、C6）。D4 的验收判据同样是「与官方实现一致」，所以参考值只能来自它本身。
    "qwb-core/crates/qwb-scf/tests/reference/d4_dispersion.csv":
        "scripts/dftd4_dispersion_reference.py",
    "qwb-core/crates/qwb-scf/tests/reference/d4_properties.csv":
        "scripts/dftd4_dispersion_reference.py",
    # SV-03：多原子 CDS 能量的官方参考（PySCF 2.14.0 `libsolvent` 的 `get_cds_legacy`，
    # 返回能量 + 梯度、**不含**面积）。只收「表面张力只有零阶项」的体系（H/F/Cl/S），
    # 于是差异只剩表面分解本身。
    "qwb-core/crates/qwb-props/tests/reference/smd_cds_legacy.csv":
        "scripts/smd_cds_legacy_reference.py",
    # SV-03：SMD 参数化半径（eq. 16）的官方参考值（PySCF 的 `smd.smd_radii`，单位 bohr）。
    "qwb-core/crates/qwb-props/tests/reference/smd_parameterised_radii.csv":
        "scripts/smd_radii_reference.py",
    # M2 ④：S22 弱相互作用二聚体集（几何 + CCSD(T)/CBS CP 参考相互作用能），
    # 数据来自 ASE 的 `ase/data/s22.py`（sha256 钉住），DOI 见 `S22_DOI`。
    "qwb-core/crates/qwb-engine/tests/reference/s22_dimers.csv":
        "scripts/gen_s22_reference.py",
    # RS-04（第三条判据：¹H/¹³C 化学位移 vs **实验**）：CH4 / CH3OH 的 ¹H 与 ¹³C 位移，
    # 以及 TMS 的定义零点。这一份**不是**算出来的，是从 pinned 文献里逐字抄下来手写的
    # （所以没有 `--check` 之外的可重建性可言）：生成器里那几行常量就是 CSV 的唯一真相源，
    # `--check` 保证 CSV 与那几行常量逐字一致。出处见 `FULMER_DOI` / `FULMER_URL`
    # （Organometallics 29, 2176 (2010) 的支持信息 Table S1/S2 的 CDCl3 一列）。
    "qwb-core/crates/qwb-engine/tests/reference/nmr_shifts_experimental.csv":
        "scripts/gen_nmr_shifts_experimental.py",
    # T5：ZnO / LANL2DZ（Zn 18 芯 ECP，z_eff = 12）/ PBE 的三行。它们**曾经**是本项目
    # 自己的回归钉（`internal-independent-path`），而那三行实测就是 `H_core` 初猜密度的
    # 能量 —— RP-02 的缺陷把初猜能量当成收敛解返回了（证据见 CSV 的 note 与
    # `scripts/pyscf_zno_lanl2dz_reference.py` 的文件头）。现在这三行是 PySCF 2.14.0 的
    # 复算值（基组与 ECP 与 qwb-data 逐壳层/逐项一致），所以按 PySCF 的来源标注。
    "qwb-core/crates/qwb-engine/tests/reference/zno_lanl2dz_pbe_ecp.csv":
        "scripts/pyscf_zno_lanl2dz_reference.py",
    # T5：CuH / LANL2DZ（Cu 10 芯 ECP，z_eff = 19）/ PBE 在 r(Cu–H) = 1.46 Å 的两行
    # （grid level 1 ↔ 本项目 grid-coarse，level 2 ↔ grid-normal）。这一份是 2026-11 修掉
    # ECP 的 `r` 幂次约定与通道语义之后重新钉 CuH 能量时**必需的独立标尺**：两个 CuH 能量
    # 本身是回归钉（本项目的输出、1e-6），而"钉在 oracle 上"这件事由这一份 + 5 mHa 断言证明。
    "qwb-core/crates/qwb-engine/tests/reference/cuh_lanl2dz_pbe_ecp.csv":
        "scripts/pyscf_cuh_lanl2dz_reference.py",
    # I-05：LANL2DZ ECP 矩阵**按 l 通道拆开**的外部对照（Cu、Zn，各含局域 + ΔV_l 通道）。
    # 这一份存在的理由是"两条内部实现会一起错"：解析与半数值共享同一套数据解释，
    # 2026-11 实测它们互差 3e-14 而同时与 PySCF 差 8 个量级以上。
    "qwb-core/crates/qwb-scf/tests/reference/ecp_lanl2dz_channels.csv":
        "scripts/pyscf_ecp_channels_reference.py",
    # RP-01：三个反应的**文献势垒**（HCN↔HNC / CH₃NC↔CH₃CN / H₂CO→H₂+CO）。这一份不是算
    # 出来的，是逐字抄自三篇论文的摘要与表格（所以 `--check` 证明的是"CSV 与那几行常量逐字
    # 一致"，不是"能被重算"）：生成器里那几行常量是唯一真相源。每条都带 `zpe_included` 与
    # `level_of_theory`，因为"差 1.5 kcal/mol"这句话少了这两栏就不能比。出处见
    # `RP01_HCN_DOI` / `RP01_MECN_DOI` / `RP01_H2CO_DOI`。
    "qwb-core/crates/qwb-engine/tests/reference/rp01_literature_barriers.csv":
        "scripts/gen_rp01_literature.py",
}

FAMILIES = tuple(GENERATORS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(path: str) -> str:
    """仓库相对路径（`FAMILIES` 里的写法）。"""
    return os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, "/")


def stamp(rows, doi: str = PYSCF_DOI, url: str = PYSCF_URL):
    """给每一行补上两列来源；返回新列表（不改入参）。"""
    out = []
    for row in rows:
        row = dict(row)
        row["source_doi"] = doi
        row["source_url"] = url
        out.append(row)
    return out


def write_csv(path: str, fields, rows) -> None:
    """按给定列序写 CSV（LF、无 CR）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buffer = io.StringIO()
    # `lineterminator="\n"`：csv 模块默认写 CRLF，而仓库的 CSV 走 git 的 LF 归一化 ——
    # 不显式指定的话，"生成器写出的字节"与"clone 下来的字节"永远不相等，
    # `--check` 会在干净检出上误报 STALE（这类坑只有在新克隆里才暴露）。
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields) + list(SOURCE_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    io.open(path, "w", encoding="utf-8", newline="\n").write(buffer.getvalue())


def check_csv(path: str, fields, rows) -> int:
    """`--check` 的实现：把同样的行渲染一遍，与文件逐字比对。

    返回进程退出码：0 = 一致；1 = 文件过时（打印第一处差异所在行）。
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields) + list(SOURCE_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    expected = buffer.getvalue()
    if not os.path.isfile(path):
        print(f"MISSING {path}")
        return 1
    actual = io.open(path, encoding="utf-8", newline="").read()
    if actual == expected:
        print(f"ok: {rel(path)} matches the generator ({len(rows)} rows)")
        return 0
    got_rows = actual.splitlines()
    want_rows = expected.splitlines()
    for index in range(max(len(got_rows), len(want_rows))):
        got = got_rows[index] if index < len(got_rows) else "<missing>"
        want = want_rows[index] if index < len(want_rows) else "<missing>"
        if got != want:
            print(f"STALE {rel(path)} line {index + 1}:\n  file: {got}\n  gen : {want}")
            break
    return 1
