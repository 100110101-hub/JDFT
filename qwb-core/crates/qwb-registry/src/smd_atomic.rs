//! SMD 的**元素级**数据（`kind: smd-atomic`）：CDS 的 vdW 半径、逐元素表面张力 σ、面积约定。
//!
//! # 为什么单独一类
//!
//! SMD 的两块数据是**元素级**的：CDS 的 vdW 半径（BONDI/Mantina 表）与表面张力 σ。它们与具体
//! 溶剂无关（唯一的溶剂依赖是 O 的半径通过 Abraham `α` 进入，而那是**参数化半径**的另一套，
//! 不属于本文件）。把它们抄进 `solvent/*.yaml` 的八份副本里，等于同一张表要在八处保持一致 ——
//! 所以走**加性扩展**：一个独立文件 + 独立类型 + 独立 schema（与 `kind: d4` 同一个做法）。
//!
//! # 数据是怎么来的（这一点必须留在类型里）
//!
//! 全部数值逐字来自 PySCF 2.14.0 的 `pyscf/lib/solvent/mnsol/mnsol.F`（Apache-2.0），由
//! `scripts/gen_smd_cds.py` 解析、`scripts/gen_smd_atomic.py` 落盘，并由
//! `scripts/validate_smd_cds.py` 对**同一个** PySCF 的 `libsolvent` 验证：水相 σ 最差
//! `1.615e-5 cal/(mol·Å²)`、五个非水溶剂的 `E_CDS` 最差 `3.797e-7 kcal/mol`。
//!
//! 面积约定 `A = 4π(R + 0.4 Å)²` 里那个 `0.4` **不是从论文抄的**：用 `4πR²` 反推时 σ 逐元素
//! 偏大 1.48~1.78 倍，加 0.4 Å 之后与官方回到 1e-7 —— 所以它以数据字段的形式存在，并带上
//! 来由（见 `cds.area_offset_angstrom` 与数据文件的 notes）。
#![deny(unsafe_code)]

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::dispersion::UpstreamSource;
use crate::elements;

/// CDS 的水相参数（`SIGMA_DATA`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdWaterSigma {
    /// 逐元素表面张力 σ（cal/(mol·Å²)）。
    pub sigma_cal_per_mol_angstrom2: BTreeMap<String, f64>,
    /// 逐元素的氢修正 `HSIGMA`（cal/(mol·Å²)）。
    pub hsigma_cal_per_mol_angstrom2: BTreeMap<String, f64>,
}

/// CDS 的非水相参数：σ 是「三张表 × 溶剂 Abraham 描述符」的线性组合，外加一个常数项。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdNonaqueousSigma {
    /// σ 的拼装公式（上游 `mnsol.F` 的原文，人读用）。
    pub sigma_formula: String,
    /// `HSIGMA` 的拼装公式。
    pub hsigma_formula: String,
    /// 与元素无关的常数项公式（`CSSIGM`）。
    pub constant_formula: String,
    /// 描述符顺序：`SOLA/SOLB/SOLC/SOLG/SOLH/SOLN`。
    pub descriptor_names: Vec<String>,
    /// `SIGMA_N`（长度 = 元素数）。
    pub sigma_n: Vec<f64>,
    /// `SIGMA_A`。
    pub sigma_a: Vec<f64>,
    /// `SIGMA_B`。
    pub sigma_b: Vec<f64>,
    /// `HSIGMA_N`。
    pub hsigma_n: Vec<f64>,
    /// `HSIGMA_A`。
    pub hsigma_a: Vec<f64>,
    /// `HSIGMA_B`。
    pub hsigma_b: Vec<f64>,
    /// `SIGMA_MOL`（四个系数）。
    pub sigma_mol: Vec<f64>,
}

/// CDS 的**成键接触规则**（上游 `SMXCDS`）：15×15 的参考键长表 + 元素类型映射。
///
/// 成键权重是 `COT_ij = exp(Δ/(d_ij − (R_ij + Δ)))`（`d ≥ R + Δ` 时为 0），
/// `R_ij = rkkval_angstrom[type_i − 1][type_j − 1]`、`Δ = 0.30 Å`；两处特例（O–S 用
/// `Δ = 0.10 / R = 1.330`、`ISTS = 6` 时 O–O 用 `Δ = 0.30 / R = 1.80`）写在 `note` 里，
/// 由实现侧按上游正文落实。类型编号为 0 的元素**不参与**成键权重（`COT ≡ 0`）——
/// 本文件覆盖的 36 个元素里有 26 个是 0，这与上游 `NATCNV` 一致。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdBonds {
    /// 上游表名（`RKKVAL`）。
    pub table: String,
    /// 提取说明与两处特例（人读用）。
    pub note: String,
    /// 元素符号 → 类型编号，1..=15；0 表示不参与成键权重。
    pub element_types: BTreeMap<String, usize>,
    /// 15×15 的参考键长（Å），按上游类型编号索引（0 基：`[类型 − 1][类型 − 1]`）。
    pub rkkval_angstrom: Vec<Vec<f64>>,
}

/// CDS 项的元素级数据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdCds {
    /// 面积约定（人类可读的形式，如 `A_i = 4*pi*(R_i + offset)^2`）。
    pub area_convention: String,
    /// 面积约定里的半径偏移（Å）。
    pub area_offset_angstrom: f64,
    /// CDS 用的 vdW 半径（Å）。**不是** SMD 的参数化半径。
    pub vdw_radii_angstrom: BTreeMap<String, f64>,
    /// 水相参数。
    pub water: SmdWaterSigma,
    /// 非水相参数。
    pub nonaqueous: SmdNonaqueousSigma,
    /// 成键接触规则（`SMXCDS` 的 `COT` 权重）。
    pub bonds: SmdBonds,
}

/// 溯源。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdAtomicProvenance {
    /// 生成脚本。
    pub generator: String,
    /// 上游文件（含 sha256）。
    pub upstream: Vec<UpstreamSource>,
}

/// `kind: smd-atomic` 的数据文件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdAtomic {
    /// 数据键名（`smd-atomic`）。
    pub name: String,
    /// 数据版本。
    pub version: String,
    /// 数据种类（恒为 `smd-atomic`）。
    pub kind: String,
    /// 引用来源。
    pub citation: Vec<Citation>,
    /// 许可声明。
    pub license: License,
    /// 创建时间（ISO 8601 UTC）。
    pub created: String,
    /// 内容摘要（工具生成）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// 备注。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// 覆盖的元素（符号，按原子序数升序）。
    pub elements: Vec<String>,
    /// CDS 数据。
    pub cds: SmdCds,
    /// 溯源。
    pub provenance: SmdAtomicProvenance,
}

impl_data_common!(SmdAtomic);

impl SmdAtomic {
    /// 完整校验（`qwb data verify` 用）。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 结构校验：元素顺序、各表与元素数一致、参数为正、溯源可查。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.kind.trim() != "smd-atomic" {
            issues.push(format!("kind must be `smd-atomic`, got `{}`", self.kind));
        }
        if self.elements.is_empty() {
            issues.push("elements must not be empty".into());
        }
        let mut previous = 0u32;
        for symbol in &self.elements {
            match elements::atomic_number(symbol) {
                None => issues.push(format!("unknown element symbol `{symbol}`")),
                Some(z) => {
                    if z <= previous {
                        issues.push(format!(
                            "elements must be sorted by Z ascending (got {symbol} after Z={previous})"
                        ));
                    }
                    previous = z;
                }
            }
        }
        let count = self.elements.len();

        // 每个元素都要有半径与水相 σ（缺一个就是索引错位或漏抄）。
        for symbol in &self.elements {
            match self.cds.vdw_radii_angstrom.get(symbol) {
                None => issues.push(format!(
                    "cds.vdw_radii_angstrom has no entry for `{symbol}`"
                )),
                Some(radius) => {
                    if !radius.is_finite() || *radius <= 0.0 {
                        issues.push(format!("cds.vdw_radii_angstrom[{symbol}] must be positive"));
                    }
                }
            }
            if !self
                .cds
                .water
                .sigma_cal_per_mol_angstrom2
                .contains_key(symbol)
            {
                issues.push(format!(
                    "cds.water.sigma_cal_per_mol_angstrom2 has no entry for `{symbol}`"
                ));
            }
            if !self
                .cds
                .water
                .hsigma_cal_per_mol_angstrom2
                .contains_key(symbol)
            {
                issues.push(format!(
                    "cds.water.hsigma_cal_per_mol_angstrom2 has no entry for `{symbol}`"
                ));
            }
        }

        for (label, values) in [
            ("sigma_n", &self.cds.nonaqueous.sigma_n),
            ("sigma_a", &self.cds.nonaqueous.sigma_a),
            ("sigma_b", &self.cds.nonaqueous.sigma_b),
            ("hsigma_n", &self.cds.nonaqueous.hsigma_n),
            ("hsigma_a", &self.cds.nonaqueous.hsigma_a),
            ("hsigma_b", &self.cds.nonaqueous.hsigma_b),
        ] {
            if values.len() != count {
                issues.push(format!(
                    "cds.nonaqueous.{label} has {} entries, expected {count}",
                    values.len()
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                issues.push(format!("cds.nonaqueous.{label} has a non-finite entry"));
            }
        }
        if self.cds.nonaqueous.sigma_mol.len() != 4 {
            issues.push(format!(
                "cds.nonaqueous.sigma_mol has {} entries, expected 4",
                self.cds.nonaqueous.sigma_mol.len()
            ));
        }
        if self.cds.nonaqueous.descriptor_names.len() != 6 {
            issues.push(format!(
                "cds.nonaqueous.descriptor_names has {} entries, expected 6",
                self.cds.nonaqueous.descriptor_names.len()
            ));
        }
        if !self.cds.area_offset_angstrom.is_finite() || self.cds.area_offset_angstrom <= 0.0 {
            issues.push("cds.area_offset_angstrom must be positive and finite".into());
        }
        // 成键规则：15×15、有限非负、对称；每个覆盖元素都要有类型（0 = 不参与）。
        let bonds = &self.cds.bonds;
        if bonds.table.trim() != "RKKVAL" {
            issues.push(format!(
                "cds.bonds.table must be `RKKVAL`, got `{}`",
                bonds.table
            ));
        }
        if bonds.rkkval_angstrom.len() != 15 {
            issues.push(format!(
                "cds.bonds.rkkval_angstrom has {} rows, expected 15",
                bonds.rkkval_angstrom.len()
            ));
        } else {
            for (row, values) in bonds.rkkval_angstrom.iter().enumerate() {
                if values.len() != 15 {
                    issues.push(format!(
                        "cds.bonds.rkkval_angstrom row {row} has {} entries, expected 15",
                        values.len()
                    ));
                    continue;
                }
                for (column, value) in values.iter().enumerate() {
                    if !value.is_finite() || *value < 0.0 {
                        issues.push(format!(
                            "cds.bonds.rkkval_angstrom[{row}][{column}] must be non-negative \
                             and finite"
                        ));
                    }
                    let mirrored = bonds.rkkval_angstrom[column][row];
                    if (value - mirrored).abs() > 1.0e-12 {
                        issues.push(format!(
                            "cds.bonds.rkkval_angstrom must be symmetric at [{row}][{column}]"
                        ));
                    }
                }
            }
        }
        for symbol in &self.elements {
            match bonds.element_types.get(symbol) {
                None => issues.push(format!(
                    "cds.bonds.element_types has no entry for `{symbol}`"
                )),
                Some(kind) => {
                    if *kind > 15 {
                        issues.push(format!(
                            "cds.bonds.element_types[{symbol}] must be <= 15, got {kind}"
                        ));
                    }
                }
            }
        }
        if self.provenance.upstream.is_empty() {
            issues.push("provenance.upstream must not be empty".into());
        }
        for source in &self.provenance.upstream {
            if source.sha256.len() != 64 || !source.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                issues.push(format!(
                    "provenance.upstream `{}` has a malformed sha256",
                    source.file
                ));
            }
        }
        issues
    }
}
