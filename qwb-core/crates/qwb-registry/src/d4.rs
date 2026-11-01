//! DFT-D4（Grimme 2019）的数据类型：`kind: d4`（DA-06）。
//!
//! # 为什么是**另一个类型**而不是给 [`crate::dispersion::Dispersion`] 加字段
//!
//! D4 与 D3 的数据形状不一样，不是"多几列"：
//!
//! * C6 **不是**元素对常量（D3 是 666 对 × 49 值的参考表），而是每个参考体系一对
//!   `alpha(iω)`（23 个虚频），在加载时由 `3/π·trapzd` 的 Casimir–Polder 积分现算；
//! * 每个参考权重带**电荷缩放** `ζ(ga, η_i·gc, q_ref+zeff_i, q_i+zeff_i)`，所以模型需要
//!   EEQ 电荷；上游没有任何 `s6q/s8q/a1q/a2q` 系数；
//! * 阻尼是**有理型但没有零阻尼的 `alp` 指数**（`alp` 只是 ATM 三体项的指数）；
//! * CN 参数、参考体系数、`zeff`/`r4r2` 的语义都与 D3 不同。
//!
//! 把两套塞进同一个 struct 会让 D3 的字段全变成 `Option`，而 D3 的加载路径本来是对的
//! （它已经过 `d3_dispersion.csv` / `d3zero_dispersion.csv` 的验收）。所以 D4 走**自己的
//! 类型 + 自己的 K1 schema 文件**（`d4dispersion.schema.*`），D3 一个字不动 —— 这是
//! 「K1 只做加性扩展」的做法。
//!
//! # 这个类型只管**结构**
//!
//! 能量/梯度/EEQ 求解是 `qwb-scf` 的活（下一段）。这里校验的是"数据自洽"：元素数与各数组
//! 长度一致、每个参考体系的 `alpha(iω)` 点数等于频率网格、`r4r2` 为正、ATM 用 `q = 0` 建
//! C6（上游 `disp.f90:139-141`，用分子电荷会偏最多 30%）、以及要的泛函都在（缺失的记在
//! `unsupported_functionals` 里而不是编造）。
#![deny(unsafe_code)]

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::dispersion::DampingParameters;
use crate::elements;

/// D4 的标量常数（全部来自上游，见数据文件的 `provenance`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4Constants {
    /// Å→Bohr，mctc-lib 的 `aatoau`（CODATA-2018 推导，**不是** K2 的 `BOHR_PER_ANGSTROM`）。
    pub aatoau: f64,
    /// `3/π`（Casimir–Polder 积分的前因子）。
    pub thopi: f64,
    /// `ζ` 的宽度参数。
    pub ga: f64,
    /// `ζ` 的硬度缩放。
    pub gc: f64,
    /// 参考权重的 CN 高斯宽度。
    pub wf: f64,
}

/// EEQ(2019) 的逐元素参数（数组下标 = 原子序数 − 1，与 `elements` 同序）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4Eeq {
    /// 模型名（当前只有 `eeq2019`）。
    pub model: String,
    /// 电负性 χ。
    pub chi: Vec<f64>,
    /// 化学硬度 η（对角项；可以为负，上游如此）。
    pub eta: Vec<f64>,
    /// χ 的 CN 缩放。
    pub kcnchi: Vec<f64>,
    /// 电荷宽度/阻尼半径（**不做** CN 缩放）。
    pub rad: Vec<f64>,
}

/// D4 的配位数参数（电负性加权的误差函数 CN）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4CoordinationNumber {
    /// 计数函数 `½(1+erf(−kcn(R−rc)/rc))` 的 `kcn`。
    pub kcn: f64,
    /// 截断半径（Bohr）。
    pub cutoff_bohr: f64,
    /// 计数函数的长度幂（上游 `norm_exp`，当前为 1）。
    pub norm_exp: f64,
    /// 共价半径缩放（上游 4/3）。
    pub radius_scaling: f64,
    /// 电负性加权的 `k4`。
    pub k4: f64,
    /// 电负性加权的 `k5`。
    pub k5: f64,
    /// 电负性加权的 `k6`。
    pub k6: f64,
}

/// `alpha(iω)` 的积分网格（23 个虚频）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4FrequencyGrid {
    /// 虚频（Hartree），升序。
    pub hartree: Vec<f64>,
    /// `trapzd` 的权重。
    pub trapzd_weights: Vec<f64>,
}

/// 一个元素的一个参考体系。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4ReferenceSystem {
    /// 参考序号（1-based，与上游 `refsys` 一致）。
    pub r#ref: u32,
    /// 全局 SE 参考集里的槽位号。
    pub refsys: u32,
    /// 该参考体系的 CN（上游 `refcovcn`，权重用它）。
    pub refcovcn: f64,
    /// 该参考体系的 EEQ 电荷。
    pub clsq: f64,
    /// 极化率缩放因子。
    pub ascale: f64,
    /// 原始 `alpha(iω)`。
    pub alpha_iw: Vec<f64>,
    /// 电荷缩放后的 `alpha(iω)`（`alpha_iw_eeq`）。
    pub alpha_iw_eeq: Vec<f64>,
}

/// 一个元素的全部参考体系与其元素级参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4ElementReference {
    /// 元素符号。
    pub element: String,
    /// 公布的 `<r⁴>/<r²>`（透明起见一并落盘）。
    pub r4_over_r2: f64,
    /// 模型消费的 `sqrt(0.5·<r⁴>/<r²>·sqrt(Z))`。
    pub r4r2: f64,
    /// Pauling 电负性（CN 的 EN 加权用）。
    pub pauling_en: f64,
    /// 化学硬度（上游 `hardness`）。
    pub chemical_hardness: f64,
    /// 有效核电荷。
    pub zeff: f64,
    /// 2009 版共价半径（Å，上游字面值）。
    pub covalent_rad_2009_angstrom: f64,
    /// 参考体系列表。
    pub reference_systems: Vec<D4ReferenceSystem>,
}

/// 上游没有参数的泛函（**不编造数值**，只记原因）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4UnsupportedFunctional {
    /// 为什么没有。
    pub reason: String,
    /// 查过哪些文件。
    pub checked_files: Vec<String>,
}

/// D4 的两体项设置与 ATM 三体项参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4PairwiseSettings {
    /// 两体项截断（Bohr）。
    pub cutoff_bohr: f64,
    /// ATM 三体项。
    pub atm: D4AtmSettings,
}

/// ATM 三体项（上游默认开，`s9 = 1`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4AtmSettings {
    /// 默认是否启用。
    pub enabled_by_default: bool,
    /// `s9`。
    pub s9: f64,
    /// 阻尼指数（上游 `alp = 16`；**不是**零阻尼那个 `alp`）。
    pub alp: f64,
    /// ATM 里重建 C6 用的电荷：上游是 **0**（`disp.f90:139-141`）。
    pub charge_for_c6: f64,
}

/// 上游文件的一条来源记录。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4UpstreamSource {
    /// 上游相对路径。
    pub file: String,
    /// 抓取 URL（钉到 commit）。
    pub url: String,
    /// 逐字节 sha256。
    pub sha256: String,
    /// 字节数。
    pub bytes: u64,
}

/// 溯源：文件清单 + 每个仓库的 tag/commit。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4Provenance {
    /// 上游文件。
    pub upstream: Vec<D4UpstreamSource>,
    /// 仓库 → `tag / commit`。
    pub pinned: BTreeMap<String, String>,
}

/// `kind: d4` 的数据文件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct D4Dispersion {
    /// 数据键名。
    pub name: String,
    /// 数据版本。
    pub version: String,
    /// 数据种类（恒为 `d4`）。
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
    /// 标量常数。
    pub constants: D4Constants,
    /// EEQ 参数。
    pub eeq: D4Eeq,
    /// CN 参数。
    pub coordination_number: D4CoordinationNumber,
    /// 频率网格。
    pub frequency_grid: D4FrequencyGrid,
    /// 逐元素参考体系。
    pub reference: Vec<D4ElementReference>,
    /// 各泛函的阻尼参数（`s6/s8/a1/a2`；**没有**电荷响应系数）。
    pub damping_parameters: BTreeMap<String, DampingParameters>,
    /// 上游没有参数的泛函。
    #[serde(default)]
    pub unsupported_functionals: BTreeMap<String, D4UnsupportedFunctional>,
    /// 两体项与 ATM。
    pub dispersion: D4PairwiseSettings,
    /// 溯源。
    pub provenance: D4Provenance,
}

impl_data_common!(D4Dispersion);

impl D4Dispersion {
    /// 完整校验（`qwb data verify` 用）。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 结构校验：只查"数据自洽"，不算能量。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.kind.trim() != "d4" {
            issues.push(format!(
                "kind must be `d4` (this loader implements DFT-D4), got `{}`",
                self.kind
            ));
        }
        if self.elements.is_empty() {
            issues.push("elements must not be empty".into());
        }
        // 元素必须严格按原子序数升序，且每项都是已知符号：数组下标的含义全指望这一点。
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
        let n = self.elements.len();
        for (label, values) in [
            ("eeq.chi", &self.eeq.chi),
            ("eeq.eta", &self.eeq.eta),
            ("eeq.kcnchi", &self.eeq.kcnchi),
            ("eeq.rad", &self.eeq.rad),
        ] {
            if values.len() != n {
                issues.push(format!(
                    "{label} has {} entries, expected {n}",
                    values.len()
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                issues.push(format!("{label} has a non-finite entry"));
            }
        }
        if self.eeq.rad.iter().any(|value| *value <= 0.0) {
            issues.push("eeq.rad must be positive (it is a Coulomb damping width)".into());
        }
        if !self.coordination_number.kcn.is_finite() || self.coordination_number.kcn <= 0.0 {
            issues.push("coordination_number.kcn must be positive".into());
        }
        if !self.coordination_number.cutoff_bohr.is_finite()
            || self.coordination_number.cutoff_bohr <= 0.0
        {
            issues.push("coordination_number.cutoff_bohr must be positive".into());
        }
        if !self.coordination_number.radius_scaling.is_finite()
            || self.coordination_number.radius_scaling <= 0.0
        {
            issues.push("coordination_number.radius_scaling must be positive".into());
        }

        let points = self.frequency_grid.hartree.len();
        if points < 2 {
            issues.push("frequency_grid.hartree needs at least two points".into());
        }
        if self.frequency_grid.trapzd_weights.len() != points {
            issues.push(format!(
                "frequency_grid.trapzd_weights has {} entries, expected {points}",
                self.frequency_grid.trapzd_weights.len()
            ));
        }
        if self
            .frequency_grid
            .hartree
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        {
            issues.push("frequency_grid.hartree must be strictly ascending".into());
        }

        for (label, value) in [
            ("constants.aatoau", self.constants.aatoau),
            ("constants.thopi", self.constants.thopi),
            ("constants.ga", self.constants.ga),
            ("constants.gc", self.constants.gc),
            ("constants.wf", self.constants.wf),
        ] {
            if !value.is_finite() || value <= 0.0 {
                issues.push(format!("{label} must be positive and finite"));
            }
        }

        if self.reference.len() != n {
            issues.push(format!(
                "reference has {} entries, expected {n} (one per element)",
                self.reference.len()
            ));
        }
        for (index, entry) in self.reference.iter().enumerate() {
            let expected = self.elements.get(index).map(String::as_str).unwrap_or("?");
            if entry.element != expected {
                issues.push(format!(
                    "reference[{index}] is `{}` but elements[{index}] is `{expected}`",
                    entry.element
                ));
            }
            if !entry.r4r2.is_finite() || entry.r4r2 <= 0.0 {
                issues.push(format!(
                    "reference[{index}].r4r2 must be positive and finite"
                ));
            }
            if !entry.zeff.is_finite() || entry.zeff <= 0.0 {
                issues.push(format!(
                    "reference[{index}].zeff must be positive and finite"
                ));
            }
            if !entry.covalent_rad_2009_angstrom.is_finite()
                || entry.covalent_rad_2009_angstrom <= 0.0
            {
                issues.push(format!(
                    "reference[{index}].covalent_rad_2009_angstrom must be positive"
                ));
            }
            if entry.reference_systems.is_empty() {
                issues.push(format!("reference[{index}] has no reference systems"));
            }
            for system in &entry.reference_systems {
                if system.alpha_iw.len() != points {
                    issues.push(format!(
                        "reference[{index}].reference_systems[{}].alpha_iw has {} points, expected {points}",
                        system.r#ref,
                        system.alpha_iw.len()
                    ));
                }
                if system.alpha_iw_eeq.len() != points {
                    issues.push(format!(
                        "reference[{index}].reference_systems[{}].alpha_iw_eeq has {} points, expected {points}",
                        system.r#ref,
                        system.alpha_iw_eeq.len()
                    ));
                }
                if system.alpha_iw_eeq.iter().any(|value| *value < 0.0) {
                    issues.push(format!(
                        "reference[{index}].reference_systems[{}].alpha_iw_eeq must be non-negative",
                        system.r#ref
                    ));
                }
            }
        }

        if self.damping_parameters.is_empty() {
            issues.push("damping_parameters must not be empty".into());
        }
        for (name, parameters) in &self.damping_parameters {
            for (label, value) in [
                ("s6", parameters.s6),
                ("s8", parameters.s8),
                ("a1", parameters.a1),
                ("a2", parameters.a2),
            ] {
                if !value.is_finite() {
                    issues.push(format!("damping_parameters.{name}.{label} is not finite"));
                }
            }
        }

        if !self.dispersion.cutoff_bohr.is_finite() || self.dispersion.cutoff_bohr <= 0.0 {
            issues.push("dispersion.cutoff_bohr must be positive".into());
        }
        if !self.dispersion.atm.alp.is_finite() || self.dispersion.atm.alp <= 0.0 {
            issues.push("dispersion.atm.alp must be positive".into());
        }
        if !self.dispersion.atm.s9.is_finite() {
            issues.push("dispersion.atm.s9 is not finite".into());
        }
        // 这不是笔误：上游在 ATM 里用 **q = 0** 重建 C6（`disp.f90:139-141`），
        // 用分子电荷会偏最多 30%。把它写进校验，改坏了就红。
        if self.dispersion.atm.charge_for_c6 != 0.0 {
            issues.push(
                "dispersion.atm.charge_for_c6 must be 0 (upstream rebuilds the ATM C6 with q = 0)"
                    .into(),
            );
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
        if !self.provenance.pinned.contains_key("dftd4/dftd4") {
            issues.push("provenance.pinned must record the dftd4 commit".into());
        }
        issues
    }
}
