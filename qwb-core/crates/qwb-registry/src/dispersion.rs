//! 色散校正参数（§4 K1 / DA-06）：D3(BJ) 的元素对参数。
//!
//! # 这份数据回答什么
//!
//! D3(BJ) 把色散能写成原子对之和
//!
//! ```text
//! E = Σ_{i<j} -C6_ij · ( s6/(r^6 + R0^6) + s8·(C8/C6)_ij/(r^8 + R0^8) )
//! R0 = a1·sqrt(C8/C6) + a2,   (C8/C6)_ij = 3·r4r2_i·r4r2_j
//! ```
//!
//! 其中 `C6_ij` 不是元素常数，而是「参考体系」上的插值：
//!
//! ```text
//! C6_ij = Σ_{k,l} w_i(k)·w_j(l)·C6_ref(k, l; 元素对)
//! w_i(k) ∝ exp(−wf·(CN_ref(k) − CN_i)²)          （log-sum-exp 归一化）
//! CN_i  = Σ_{j≠i} 1/(1 + exp(−kcn·(rc_i+rc_j)/r − kcn))
//! ```
//!
//! 所以数据文件必须同时带：参考 C6 表、参考 CN、每元素的参考点个数、CN 的常量与半径、
//! r4r2、以及每个泛函的阻尼参数。缺一项都算不出来 —— 这也是 DA-06 卡了很久的原因。
//!
//! # 两个容易搞错的约定（都用官方实现反解验证过，见 docs/choices.md 一百四十一）
//!
//! 1. **`r4r2` 是派生值**，不是 `r4r2.f90` 里公布的 `<r⁴>/<r²>`：上游的
//!    `get_r4r2_val(i) = sqrt(0.5·published(i)·sqrt(i))`。数据文件里两列都存
//!    （[`Dispersion::r4r2`] 是派生值，[`Dispersion::r4r2_published`] 是公布值），
//!    用错哪一个都会让 C8/C6 差 2–16 倍。
//! 2. **参考块的第一维是高 Z 元素**：每个元素对固定 7×7 个槽位，实际用到的只是
//!    「高 Z 元素参考点数 × 低 Z 元素参考点数」的左上角子块，第一维对应**原子序数更大**
//!    的那个元素。同核元素对在转置下不变，所以只有异核对才能发现搞反了。

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;
use crate::sha256::is_sha256_hex;

/// 每个元素对的参考点槽位数（上游 `reference.f90` 的 `max_ref`）。
pub const REFERENCE_POINTS: usize = 7;

/// 一个元素对的参考 C6 表长度（7×7，Fortran 展平顺序，第一维最快）。
pub const PAIR_C6_LEN: usize = REFERENCE_POINTS * REFERENCE_POINTS;

/// 配位数模型（目前只有上游的指数计数函数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum CoordinationModel {
    /// `count = 1/(1+exp(−kcn·(rc/r − 1)))`。
    Exp,
}

/// 参考体系权重模型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum WeightModel {
    /// `w ∝ exp(−wf·(CN_ref − CN)²)`，按最邻近参考点平移后归一化（log-sum-exp）。
    Gaussian,
}

/// 阻尼函数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DampingModel {
    /// Becke–Johnson 有理阻尼（D3(BJ)）。
    Rational,
    /// 零阻尼（D3(0)）；数据里还没有（要 vdW 半径表），保留取值位。
    Zero,
}

/// 配位数参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CoordinationNumber {
    /// 计数函数模型。
    pub model: CoordinationModel,
    /// 陡峭度 `kcn`（上游 16）。
    pub kcn: f64,
    /// 共价半径的缩放（上游 `4/3`，即 D3 用的是 Pyykko 半径的 4/3 倍）。
    pub scale: f64,
    /// 实空间截断，单位 Bohr。
    pub cutoff_bohr: f64,
    /// 各元素共价半径，单位 Å（键：元素符号）。上游表里金属已经降过 10%。
    pub radii_angstrom: BTreeMap<String, f64>,
}

/// 参考体系插值的参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ReferenceInterpolation {
    /// 权重宽度 `wf`（上游 4.0）。
    pub wf: f64,
    /// 权重模型。
    pub weight: WeightModel,
}

/// 一个泛函的 BJ 阻尼参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct DampingParameters {
    /// `s6`（BJ 下恒为 1）。
    pub s6: f64,
    /// `s8`。
    pub s8: f64,
    /// `a1`（乘 `sqrt(C8/C6)`）。
    pub a1: f64,
    /// `a2`，单位 Bohr。
    pub a2: f64,
    /// 该组参数的出处（DOI）。
    pub doi: String,
}

/// 两点项的整体设置。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PairwiseSettings {
    /// 阻尼模型。
    pub damping: DampingModel,
    /// 两点项的实空间截断，单位 Bohr。
    pub cutoff_bohr: f64,
}

/// 一条上游溯源记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct UpstreamSource {
    /// 上游文件相对路径。
    pub file: String,
    /// 取回地址。
    pub url: String,
    /// 该文件的 sha256。
    pub sha256: String,
}

/// 数据的溯源声明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Provenance {
    /// 生成脚本（仓库相对路径）。
    pub generator: String,
    /// 上游文件清单。
    pub upstream: Vec<UpstreamSource>,
}

/// 色散参数（`qwb-data/dispersion/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Dispersion {
    /// 数据键名。
    pub name: String,
    /// 数据版本（独立 semver）。
    pub version: String,
    /// 数据种类（`d3bj`）。
    pub kind: String,
    /// 引用来源，至少 1 条。
    pub citation: Vec<Citation>,
    /// 许可声明。
    pub license: License,
    /// 创建/修订时间，ISO 8601 UTC。
    pub created: String,
    /// 内容摘要（工具生成，见 [`crate::sha256`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// 备注/来源说明（可选）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// 覆盖的元素（元素符号，按原子序数升序）。
    pub elements: Vec<String>,
    /// 配位数参数。
    pub coordination_number: CoordinationNumber,
    /// 参考体系插值参数。
    pub reference: ReferenceInterpolation,
    /// 每元素的 `sqrt(r4/r2)`：**派生值** `sqrt(0.5·published·sqrt(Z))`（模型消费的就是它）。
    pub r4r2: BTreeMap<String, f64>,
    /// 同一元素的公布 `<r⁴>/<r²>`（透明起见一并落盘）。
    pub r4r2_published: BTreeMap<String, f64>,
    /// 每元素的参考 CN（长度 = 该元素的参考点个数，≤ 7）。
    pub reference_cn: BTreeMap<String, Vec<f64>>,
    /// 元素对的参考 C6 表，键是 `高Z-低Z`（如 `H-H`、`O-H`），值 49 个（Ha·Bohr⁶）。
    pub reference_c6: BTreeMap<String, Vec<f64>>,
    /// 各泛函的阻尼参数。
    pub damping_parameters: BTreeMap<String, DampingParameters>,
    /// 两点项设置。
    pub dispersion: PairwiseSettings,
    /// 溯源。
    pub provenance: Provenance,
}

impl_data_common!(Dispersion);

impl Dispersion {
    /// 完整校验（`qwb data verify` 用）。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 元素对在 `reference_c6` 里的键：**高 Z 在前**（块内第一维就是高 Z 元素）。
    #[must_use]
    pub fn pair_key(symbol_a: &str, symbol_b: &str) -> Option<String> {
        let za = elements::atomic_number(symbol_a)?;
        let zb = elements::atomic_number(symbol_b)?;
        let (hi, lo) = if za >= zb {
            (symbol_a, symbol_b)
        } else {
            (symbol_b, symbol_a)
        };
        Some(format!("{hi}-{lo}"))
    }

    /// 取一个元素对的参考 C6 表（49 个值，Fortran 展平顺序、第一维为高 Z 元素）。
    ///
    /// # Errors
    ///
    /// 元素符号不认识，或该元素对不在数据里。
    pub fn pair_c6(&self, symbol_a: &str, symbol_b: &str) -> Result<&[f64], String> {
        let key = Self::pair_key(symbol_a, symbol_b)
            .ok_or_else(|| format!("unknown element symbol in pair ({symbol_a}, {symbol_b})"))?;
        self.reference_c6
            .get(&key)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("reference_c6 has no entry for {key}"))
    }

    /// 取一个泛函的阻尼参数。
    #[must_use]
    pub fn damping(&self, functional: &str) -> Option<&DampingParameters> {
        self.damping_parameters.get(functional)
    }

    /// 某元素的参考点个数。
    #[must_use]
    pub fn reference_count(&self, symbol: &str) -> Option<usize> {
        self.reference_cn.get(symbol).map(Vec::len)
    }

    /// 某元素在 CN 插值里用的共价半径（Bohr，已乘 `scale` 与 Å→Bohr）。
    ///
    /// # Errors
    ///
    /// 元素不在 `radii_angstrom` 里。
    pub fn covalent_radius_bohr(&self, symbol: &str) -> Result<f64, String> {
        let angstrom = self
            .coordination_number
            .radii_angstrom
            .get(symbol)
            .ok_or_else(|| format!("coordination_number.radii_angstrom has no `{symbol}`"))?;
        Ok(angstrom * self.coordination_number.scale * qwb_engine_api::molecule::BOHR_PER_ANGSTROM)
    }

    /// 结构性检查：**不含** G2 门禁（`Registry::open` 用，§8.6）。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.kind.trim() != "d3bj" {
            issues.push(format!(
                "kind must be `d3bj` (this loader only implements D3(BJ)), got {:?}",
                self.kind
            ));
        }

        // ── 元素表 ──
        if self.elements.is_empty() {
            issues.push("elements must not be empty".into());
        }
        let mut previous = 0u32;
        let mut seen = std::collections::BTreeSet::new();
        for symbol in &self.elements {
            match elements::atomic_number(symbol) {
                Some(number) => {
                    if !seen.insert(symbol.clone()) {
                        issues.push(format!("elements contains `{symbol}` twice"));
                    }
                    if number <= previous {
                        issues.push(format!(
                            "elements must be in ascending atomic-number order; `{symbol}` \
                             breaks it"
                        ));
                    }
                    previous = number;
                }
                None => issues.push(format!("elements contains unknown symbol `{symbol}`")),
            }
        }

        // ── 配位数参数 ──
        let cn = &self.coordination_number;
        if !cn.kcn.is_finite() || cn.kcn <= 0.0 {
            issues.push("coordination_number.kcn must be positive and finite".into());
        }
        if !cn.scale.is_finite() || cn.scale <= 0.0 {
            issues.push("coordination_number.scale must be positive and finite".into());
        }
        if !cn.cutoff_bohr.is_finite() || cn.cutoff_bohr <= 0.0 {
            issues.push("coordination_number.cutoff_bohr must be positive and finite".into());
        }
        issues.extend(check_element_map(
            "coordination_number.radii_angstrom",
            &cn.radii_angstrom,
            &self.elements,
            true,
        ));

        // ── 参考插值参数 ──
        if !self.reference.wf.is_finite() || self.reference.wf <= 0.0 {
            issues.push("reference.wf must be positive and finite".into());
        }

        // ── r4r2 两张表 ──
        issues.extend(check_element_map("r4r2", &self.r4r2, &self.elements, true));
        issues.extend(check_element_map(
            "r4r2_published",
            &self.r4r2_published,
            &self.elements,
            true,
        ));
        for (symbol, published) in &self.r4r2_published {
            let Some(derived) = self.r4r2.get(symbol) else {
                continue;
            };
            let Some(number) = elements::atomic_number(symbol) else {
                continue;
            };
            let expected = (0.5 * published * f64::from(number).sqrt()).sqrt();
            if (derived - expected).abs() > 1e-12 * expected.abs().max(1.0) {
                issues.push(format!(
                    "r4r2.{symbol} = {derived} does not equal sqrt(0.5*sqrt(Z)*published) = \
                     {expected}; the kernel consumes the derived value (docs/choices.md 141)"
                ));
            }
        }

        // ── 参考 CN ──
        for symbol in &self.elements {
            match self.reference_cn.get(symbol) {
                None => issues.push(format!("reference_cn has no entry for `{symbol}`")),
                Some(values) => {
                    if values.is_empty() || values.len() > REFERENCE_POINTS {
                        issues.push(format!(
                            "reference_cn.{symbol} must have 1..={REFERENCE_POINTS} values, got {}",
                            values.len()
                        ));
                    }
                    if values.iter().any(|value| !value.is_finite()) {
                        issues.push(format!("reference_cn.{symbol} contains a non-finite value"));
                    }
                }
            }
        }
        for symbol in self.reference_cn.keys() {
            if !self.elements.contains(symbol) {
                issues.push(format!("reference_cn contains `{symbol}`, not in elements"));
            }
        }

        // ── 参考 C6 表：元素对数、每个 49 个值、全有限非负 ──
        let n = self.elements.len();
        let expected_pairs = n * (n + 1) / 2;
        if self.reference_c6.len() != expected_pairs {
            issues.push(format!(
                "reference_c6 must hold {expected_pairs} element pairs ({} elements), got {}",
                n,
                self.reference_c6.len()
            ));
        }
        for (index, high) in self.elements.iter().enumerate() {
            for low in &self.elements[..=index] {
                let key = format!("{high}-{low}");
                match self.reference_c6.get(&key) {
                    None => issues.push(format!("reference_c6 is missing the pair `{key}`")),
                    Some(values) => {
                        if values.len() != PAIR_C6_LEN {
                            issues.push(format!(
                                "reference_c6.{key} must hold {PAIR_C6_LEN} values, got {}",
                                values.len()
                            ));
                        }
                        if let Some(position) = values
                            .iter()
                            .position(|value| !value.is_finite() || *value < 0.0)
                        {
                            issues.push(format!(
                                "reference_c6.{key}[{position}] must be finite and >= 0"
                            ));
                        }
                    }
                }
            }
        }
        for key in self.reference_c6.keys() {
            let mut parts = key.split('-');
            let (Some(high), Some(low), None) = (parts.next(), parts.next(), parts.next()) else {
                issues.push(format!("reference_c6 key `{key}` is not of the form HI-LO"));
                continue;
            };
            let (Some(zh), Some(zl)) =
                (elements::atomic_number(high), elements::atomic_number(low))
            else {
                issues.push(format!("reference_c6 key `{key}` has an unknown element"));
                continue;
            };
            if zh < zl {
                issues.push(format!(
                    "reference_c6 key `{key}` must put the higher-Z element first \
                     (the first block dimension is the higher-Z one)"
                ));
            }
            if !self.elements.iter().any(|known| known == high)
                || !self.elements.iter().any(|known| known == low)
            {
                issues.push(format!(
                    "reference_c6 key `{key}` uses an element outside `elements`"
                ));
            }
        }

        // ── 阻尼参数 ──
        if self.damping_parameters.is_empty() {
            issues.push("damping_parameters must not be empty".into());
        }
        for (functional, params) in &self.damping_parameters {
            if !params.s6.is_finite() || params.s6 < 0.0 {
                issues.push(format!(
                    "damping_parameters.{functional}.s6 must be finite and >= 0"
                ));
            }
            if !params.s8.is_finite() || params.s8 < 0.0 {
                issues.push(format!(
                    "damping_parameters.{functional}.s8 must be finite and >= 0"
                ));
            }
            if !params.a1.is_finite() {
                issues.push(format!("damping_parameters.{functional}.a1 must be finite"));
            }
            if !params.a2.is_finite() || params.a2 < 0.0 {
                issues.push(format!(
                    "damping_parameters.{functional}.a2 must be finite and >= 0 (Bohr)"
                ));
            }
            if params.doi.trim().is_empty() {
                issues.push(format!(
                    "damping_parameters.{functional}.doi must not be empty"
                ));
            }
        }

        // ── 两点项设置 ──
        if !self.dispersion.cutoff_bohr.is_finite() || self.dispersion.cutoff_bohr <= 0.0 {
            issues.push("dispersion.cutoff_bohr must be positive and finite".into());
        }
        if self.dispersion.damping == DampingModel::Zero {
            issues.push(
                "dispersion.damping = zero is not implemented (needs vdW radii); only rational \
                 (BJ) data is accepted"
                    .into(),
            );
        }

        // ── 溯源 ──
        if self.provenance.generator.trim().is_empty() {
            issues.push("provenance.generator must not be empty".into());
        }
        if self.provenance.upstream.is_empty() {
            issues.push("provenance.upstream must list the source files (DA-06)".into());
        }
        for (index, source) in self.provenance.upstream.iter().enumerate() {
            if source.file.trim().is_empty() {
                issues.push(format!(
                    "provenance.upstream[{index}].file must not be empty"
                ));
            }
            if !source.url.starts_with("https://") {
                issues.push(format!(
                    "provenance.upstream[{index}].url must be https, got {:?}",
                    source.url
                ));
            }
            if !is_sha256_hex(&source.sha256) {
                issues.push(format!(
                    "provenance.upstream[{index}].sha256 must be 64 lowercase hex chars"
                ));
            }
        }

        issues
    }
}

/// 检查「元素符号 → 数值」的映射是否恰好覆盖 `elements`。
fn check_element_map(
    field: &str,
    map: &BTreeMap<String, f64>,
    expected: &[String],
    positive: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    if map.len() != expected.len() {
        issues.push(format!(
            "{field} must cover exactly the {} element(s) in `elements`, got {}",
            expected.len(),
            map.len()
        ));
    }
    for symbol in expected {
        match map.get(symbol) {
            None => issues.push(format!("{field} has no entry for `{symbol}`")),
            Some(value) => {
                if !value.is_finite() {
                    issues.push(format!("{field}.{symbol} must be finite"));
                } else if positive && *value <= 0.0 {
                    issues.push(format!("{field}.{symbol} must be positive"));
                }
            }
        }
    }
    for symbol in map.keys() {
        if !expected.contains(symbol) {
            issues.push(format!("{field} contains `{symbol}`, not in elements"));
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::{Dispersion, PAIR_C6_LEN};
    use crate::common::{Citation, License};
    use std::collections::BTreeMap;

    fn sample() -> Dispersion {
        let elements = vec!["H".to_string(), "He".to_string()];
        let mut radii = BTreeMap::new();
        radii.insert("H".into(), 0.32);
        radii.insert("He".into(), 0.46);
        let mut r4r2_published: BTreeMap<String, f64> = BTreeMap::new();
        r4r2_published.insert("H".into(), 8.0589);
        r4r2_published.insert("He".into(), 3.4698);
        let mut r4r2: BTreeMap<String, f64> = BTreeMap::new();
        for (symbol, published) in &r4r2_published {
            let z: f64 = if symbol == "H" { 1.0 } else { 2.0 };
            r4r2.insert(symbol.clone(), (0.5 * published * z.sqrt()).sqrt());
        }
        let mut reference_cn = BTreeMap::new();
        reference_cn.insert("H".into(), vec![0.9118, 0.0]);
        reference_cn.insert("He".into(), vec![0.0]);
        let mut reference_c6 = BTreeMap::new();
        reference_c6.insert("H-H".into(), vec![1.0; PAIR_C6_LEN]);
        reference_c6.insert("He-H".into(), vec![1.0; PAIR_C6_LEN]);
        reference_c6.insert("He-He".into(), vec![1.0; PAIR_C6_LEN]);
        let mut damping_parameters = BTreeMap::new();
        damping_parameters.insert(
            "b3lyp".into(),
            super::DampingParameters {
                s6: 1.0,
                s8: 1.9889,
                a1: 0.3981,
                a2: 4.4211,
                doi: "10.1063/1.3382344".into(),
            },
        );
        Dispersion {
            name: "d3bj".into(),
            version: "1.0.0".into(),
            kind: "d3bj".into(),
            citation: vec![Citation::with_doi("D3", "10.1063/1.3382344")],
            license: License::new("LGPL-3.0-or-later", true, true),
            created: "2026-11-09T00:00:00Z".into(),
            sha256: Some("a".repeat(64)),
            notes: None,
            elements,
            coordination_number: super::CoordinationNumber {
                model: super::CoordinationModel::Exp,
                kcn: 16.0,
                scale: 4.0 / 3.0,
                cutoff_bohr: 40.0,
                radii_angstrom: radii,
            },
            reference: super::ReferenceInterpolation {
                wf: 4.0,
                weight: super::WeightModel::Gaussian,
            },
            r4r2,
            r4r2_published,
            reference_cn,
            reference_c6,
            damping_parameters,
            dispersion: super::PairwiseSettings {
                damping: super::DampingModel::Rational,
                cutoff_bohr: 60.0,
            },
            provenance: super::Provenance {
                generator: "scripts/gen_d3_params.py".into(),
                upstream: vec![super::UpstreamSource {
                    file: "src/dftd3/reference.f90".into(),
                    url: "https://raw.githubusercontent.com/dftd3/simple-dftd3/main/x.f90".into(),
                    sha256: "b".repeat(64),
                }],
            },
        }
    }

    #[test]
    fn sample_is_valid() {
        let data = sample();
        assert!(data.check().is_empty(), "{:?}", data.check());
        assert_eq!(data.reference_count("H"), Some(2));
        assert_eq!(Dispersion::pair_key("H", "He").as_deref(), Some("He-H"));
        assert_eq!(Dispersion::pair_key("He", "H").as_deref(), Some("He-H"));
        assert_eq!(data.damping("b3lyp").map(|p| p.s8), Some(1.9889));
        assert!(data.covalent_radius_bohr("H").unwrap() > 0.0);
    }

    #[test]
    fn structure_check_catches_broken_data() {
        let mut data = sample();
        data.elements = vec!["H".into()];
        assert!(
            data.check_structure()
                .iter()
                .any(|message| message.contains("reference_c6 must hold"))
        );

        let mut data = sample();
        data.reference_c6.insert("He-H".into(), vec![1.0; 3]);
        assert!(
            data.check_structure()
                .iter()
                .any(|message| message.contains("must hold 49 values"))
        );

        let mut data = sample();
        data.r4r2.insert("H".into(), 8.0589);
        assert!(
            data.check_structure()
                .iter()
                .any(|message| message.contains("does not equal sqrt"))
        );

        let mut data = sample();
        data.elements = vec!["He".into(), "H".into()];
        assert!(
            data.check_structure()
                .iter()
                .any(|message| message.contains("ascending"))
        );

        let mut data = sample();
        data.reference_c6.remove("He-H");
        data.reference_c6
            .insert("H-He".into(), vec![1.0; PAIR_C6_LEN]);
        let issues = data.check_structure();
        assert!(
            issues
                .iter()
                .any(|message| message.contains("higher-Z element first")),
            "{issues:?}"
        );

        let mut data = sample();
        data.kind = "d4".into();
        assert!(
            data.check_structure()
                .iter()
                .any(|message| message.contains("kind must be"))
        );
    }
}
