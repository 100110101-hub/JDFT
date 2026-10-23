//! XC 网格数据（§4 K1 / §8.4）。
//!
//! §8.4 要求“代码不含硬编码点数”：等级 → 参数的映射全部在这里的数据里。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};

/// 网格等级名（与 `engine-api::GridLevel` 对应）。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum GridLevelName {
    /// `coarse`。
    Coarse,
    /// `normal`。
    Normal,
    /// `fine`。
    Fine,
    /// `veryfine`。
    VeryFine,
}

/// 角向求积规则（K1 增量，v1.2：**可选字段** `angular_rule`，默认 `lebedev`）。
///
/// Lebedev 点集没有闭式生成法、需要按阶数制表；在表引入 qwb-data 之前，
/// 等级可以显式声明用 `gauss-product`（Gauss–Legendre(cosθ) × 均匀 φ），
/// 它同样满足 `Σw = 4π`，且不需要外部数据。**规则写在数据里**，不在代码里猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AngularRule {
    /// Lebedev 球面求积（需要 `lebedev_order` 对应的点集表）。
    #[default]
    Lebedev,
    /// Gauss–Legendre(cosθ) × 均匀 φ 乘积规则（无需外部表）。
    GaussProduct,
}

impl AngularRule {
    /// 数据里的字符串形式。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            AngularRule::Lebedev => "lebedev",
            AngularRule::GaussProduct => "gauss-product",
        }
    }
}

/// 合法的 Lebedev 阶数（§8.4：点数必须来自数据，但这些是数学上存在的阶数）。
pub const LEBEDEV_ORDERS: [u32; 32] = [
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202,
    1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810,
];

/// 一个网格等级的定义（`qwb-data/grid/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GridLevelDef {
    /// 数据键名（数据目录下的标识）。
    pub name: String,
    /// 数据版本（独立 semver；§8.6 要求变更即 bump）。
    pub version: String,
    /// 引用来源，至少 1 条；每条需 `title` + `doi` 或 `url`。
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
    /// 等级。
    pub level: GridLevelName,
    /// 每个原子的径向点数。
    pub radial_points: u32,
    /// 角向求积规则（默认 `lebedev`）。
    #[serde(default)]
    pub angular_rule: AngularRule,
    /// Lebedev 角向阶数（`angular_rule: gauss-product` 时作为角向点数的预算）。
    pub lebedev_order: u32,
    /// Becke 分区锐度参数（§8.4 陷阱显式化）。
    pub becke_sharpness: f64,
    /// 原子半径缩放因子；`None` 表示用代码内建表（§8.4 允许）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radii_scale: Option<f64>,
    /// 计入网格的原子截断半径，单位 Bohr。
    pub cutoff: f64,
}

impl_data_common!(GridLevelDef);

impl GridLevelDef {
    /// 完整校验（`qwb data verify` 用）：G2 门禁 + 结构检查。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 结构性检查：**不含** G2 门禁（`Registry::open` 用，见 §8.6）。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.radial_points < 10 {
            issues.push(format!(
                "radial_points = {} is too small (Euler-Maclaurin 需 >= 10)",
                self.radial_points
            ));
        }
        if !LEBEDEV_ORDERS.contains(&self.lebedev_order) {
            issues.push(format!(
                "lebedev_order = {} is not a standard Lebedev order",
                self.lebedev_order
            ));
        }
        if !self.becke_sharpness.is_finite() || self.becke_sharpness <= 0.0 {
            issues.push("becke_sharpness must be positive and finite (§8.4)".into());
        }
        if let Some(scale) = self.radii_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            issues.push("radii_scale must be positive and finite when present".into());
        }
        if !self.cutoff.is_finite() || self.cutoff <= 0.0 {
            issues.push("cutoff must be positive and finite (Bohr)".into());
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::{GridLevelDef, GridLevelName, LEBEDEV_ORDERS};
    use crate::common::{Citation, License};

    pub(crate) fn normal_grid() -> GridLevelDef {
        GridLevelDef {
            name: "grid-normal".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_url(
                "Becke 1988 partitioning",
                "https://doi.org/10.1063/1.454033",
            )],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("d".repeat(64)),
            notes: None,
            angular_rule: super::AngularRule::GaussProduct,
            level: GridLevelName::Normal,
            radial_points: 75,
            lebedev_order: 302,
            becke_sharpness: 0.6,
            radii_scale: Some(0.8),
            cutoff: 20.0,
        }
    }

    #[test]
    fn normal_grid_is_valid() {
        let g = normal_grid();
        assert!(g.check().is_empty(), "{:?}", g.check());
        assert!(LEBEDEV_ORDERS.contains(&g.lebedev_order));
    }

    #[test]
    fn check_catches_broken_grids() {
        let mut g = normal_grid();
        g.lebedev_order = 300;
        assert!(g.check().iter().any(|m| m.contains("Lebedev")));

        let mut g = normal_grid();
        g.radial_points = 5;
        assert!(g.check().iter().any(|m| m.contains("radial_points")));

        let mut g = normal_grid();
        g.becke_sharpness = 0.0;
        assert!(g.check().iter().any(|m| m.contains("becke_sharpness")));

        let mut g = normal_grid();
        g.cutoff = f64::INFINITY;
        assert!(g.check().iter().any(|m| m.contains("cutoff")));
    }
}
