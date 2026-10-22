//! 溶剂数据（§4 K1 / §8.5）。
//!
//! 第一阶段只做**接口与数据**，不做完整 PCM：`cavity_model` 必须显式声明，
//! loader 不因为它不认识就报错（§8.5），由 `Engine::run` 返回 `Unsupported`。

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;

/// 空腔模型（§8.5：必须显式声明，禁止隐式默认）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum CavityModel {
    /// 无空腔（真空；理论上不该作为溶剂条目出现）。
    None,
    /// 导体屏蔽模型（C-PCM 一族）。
    Pcm,
    /// SMD（§2.2 非目标；保留数据位以便 loader 通过、引擎返回 Unsupported）。
    Smd,
}

/// 溶剂（`qwb-data/solvent/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Solvent {
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
    /// 相对介电常数 `epsilon`。
    pub epsilon: f64,
    /// 温度，单位 K（K1 字段名 `temperature_K`）。
    #[serde(rename = "temperature_K")]
    pub temperature_k: f64,
    /// 各元素原子半径，单位 Å（键：元素符号）。
    pub atomic_radii: BTreeMap<String, f64>,
    /// 空腔模型。
    pub cavity_model: CavityModel,
    /// 半径缩放因子。
    pub scale: f64,
}

impl_data_common!(Solvent);

impl Solvent {
    /// 完整校验（`qwb data verify` 用）：G2 门禁 + 结构检查。
    ///
    /// ε 与文献一致的检查由 D-14 用基准值做，这里只查自洽性。
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

        if !self.epsilon.is_finite() || self.epsilon < 1.0 {
            issues.push("epsilon must be finite and >= 1 (vacuum = 1)".into());
        }
        if !self.temperature_k.is_finite() || self.temperature_k <= 0.0 {
            issues.push("temperature_K must be positive and finite".into());
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            issues.push("scale must be positive and finite".into());
        }
        if self.atomic_radii.is_empty() {
            issues.push("atomic_radii must not be empty (§4 K1)".into());
        }
        for (element, radius) in &self.atomic_radii {
            if elements::atomic_number(element).is_none() {
                issues.push(format!("atomic_radii contains unknown symbol `{element}`"));
            }
            if !radius.is_finite() || *radius <= 0.0 {
                issues.push(format!(
                    "atomic_radii.{element} must be positive and finite (Å)"
                ));
            }
        }
        if matches!(self.cavity_model, CavityModel::None) {
            issues.push(
                "cavity_model = none: a solvent entry must declare a real cavity model (§8.5)"
                    .into(),
            );
        }
        issues
    }

    /// 是否为第一阶段引擎能实际处理的空腔模型（PCM 接口尚未实现，§8.5）。
    #[must_use]
    pub fn is_engine_supported(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{CavityModel, Solvent};
    use crate::common::{Citation, License};

    pub(crate) fn water() -> Solvent {
        let mut radii = BTreeMap::new();
        radii.insert("H".to_string(), 1.2);
        radii.insert("O".to_string(), 1.5);
        Solvent {
            name: "water".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_url(
                "NIST Chemistry WebBook",
                "https://webbook.nist.gov/",
            )],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("e".repeat(64)),
            notes: None,
            epsilon: 78.3553,
            temperature_k: 298.15,
            atomic_radii: radii,
            cavity_model: CavityModel::Pcm,
            scale: 1.2,
        }
    }

    #[test]
    fn water_matches_literature_epsilon() {
        let w = water();
        assert!((w.epsilon - 78.3553).abs() < 1e-9, "§8.5 参考值");
        assert!(w.check().is_empty(), "{:?}", w.check());
        assert!(!w.is_engine_supported(), "§8.5: PCM 未实现");
    }

    #[test]
    fn check_catches_broken_solvents() {
        let mut w = water();
        w.epsilon = 0.5;
        assert!(w.check().iter().any(|m| m.contains("epsilon")));

        let mut w = water();
        w.temperature_k = 0.0;
        assert!(w.check().iter().any(|m| m.contains("temperature_K")));

        let mut w = water();
        w.atomic_radii.insert("Xx".into(), 1.0);
        assert!(w.check().iter().any(|m| m.contains("unknown symbol")));

        let mut w = water();
        w.cavity_model = CavityModel::None;
        assert!(w.check().iter().any(|m| m.contains("cavity_model")));

        let mut w = water();
        w.atomic_radii.clear();
        assert!(
            w.check()
                .iter()
                .any(|m| m.contains("atomic_radii must not be empty"))
        );
    }
}
