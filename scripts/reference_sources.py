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
    "qwb-core/crates/qwb-engine/tests/reference/smd_water.csv":
        "scripts/pyscf_smd_reference.py",
    "qwb-core/crates/qwb-engine/tests/reference/hessian_water.csv":
        "scripts/pyscf_hessian_reference.py",
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
