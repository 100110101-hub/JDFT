//! 泛函数据（§4 K1 / §8.3）。
//!
//! 数据侧只放**参数与组合**，数学形式在代码侧（X-01 的 `FunctionalTerm`）。
//! 一个泛函不允许在多处重写：`components[].id` 是代码侧组件的唯一键。

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};

/// 泛函的一个组成项（如 B3LYP 的 0.8·B88 + 0.2·HF + 0.81·LYP + 0.19·VWN5）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct FunctionalComponent {
    /// 代码侧组件 id（如 `GgaB88`、`HfExchange`）。
    pub id: String,
    /// 混合权重。
    pub weight: f64,
}

/// 该泛函需要哪些密度量（决定网格与数据结构，不允许运行时猜）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct FunctionalRequires {
    /// 需要密度 `rho`。
    pub rho: bool,
    /// 需要密度梯度不变量 `sigma = |∇rho|²`。
    pub sigma: bool,
    /// 需要 `∇²rho`（LYP 一类；meta-GGA 也常一起给）。
    pub laplacian: bool,
    /// 需要动能密度 `τ = ½ Σ_μν P_μν ∇φ_μ·∇φ_ν`（meta-GGA，FX-01）。
    ///
    /// `#[serde(default)]`：阶段一的数据文件没有这个键，缺省 `false` 才能原样解析
    /// —— 这是**加性**契约扩展的唯一可行写法（旧数据不改、新数据显式声明）。
    #[serde(default)]
    pub kinetic_density: bool,
}

/// 数值稳定性的截断规则（§8.3 第三道门禁）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct StabilityRules {
    /// 密度下限（低于此值按此值处理），单位 `e/Bohr³`。
    pub density_floor: f64,
    /// `sigma` 上限，单位 `(e/Bohr³)²`。
    pub sigma_ceiling: f64,
}

/// 泛函（`qwb-data/functional/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Functional {
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
    /// 组成项，必须非空。
    pub components: Vec<FunctionalComponent>,
    /// 需要的密度量。
    pub requires: FunctionalRequires,
    /// 参数字典（如 PBE 的 `kappa = 0.804`、`mu = 0.21951`）。
    pub parameters: BTreeMap<String, f64>,
    /// 每个参数的来源（`参数名 -> 出处`）。K1 v1.1 新增可选字段：
    /// D-09 的验收要求“参数逐个带 citation”，而 `citation` 是文件级的，
    /// 无法定位到单个参数，所以补这张表（§10.3 允许新增可选字段）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameter_sources: BTreeMap<String, String>,
    /// 稳定性截断规则。
    pub stability: StabilityRules,
}

impl_data_common!(Functional);

impl Functional {
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

        if self.components.is_empty() {
            issues.push("components must not be empty".into());
        }
        for (i, component) in self.components.iter().enumerate() {
            if component.id.trim().is_empty() {
                issues.push(format!("components[{i}].id must not be empty"));
            }
            if !component.weight.is_finite() {
                issues.push(format!("components[{i}].weight must be finite"));
            }
        }

        if !(self.requires.rho
            || self.requires.sigma
            || self.requires.laplacian
            || self.requires.kinetic_density)
        {
            issues.push(
                "requires must set at least one of rho/sigma/laplacian/kinetic_density".into(),
            );
        }
        if self.requires.laplacian {
            issues.push(
                "requires.laplacian = true is out of first-phase scope (§2.2: no meta-GGA); \
                 Engine::run will return Unsupported"
                    .into(),
            );
        }

        for (key, value) in &self.parameters {
            if key.trim().is_empty() {
                issues.push("parameters contains an empty key".into());
            }
            if !value.is_finite() {
                issues.push(format!("parameters.{key} must be finite"));
            }
        }

        for key in self.parameter_sources.keys() {
            if !self.parameters.contains_key(key) {
                issues.push(format!(
                    "parameter_sources.{key} does not name a parameter in `parameters` (dangling source)"
                ));
            }
        }

        if !self.stability.density_floor.is_finite() || self.stability.density_floor <= 0.0 {
            issues.push("stability.density_floor must be positive and finite (§8.3)".into());
        }
        if !self.stability.sigma_ceiling.is_finite() || self.stability.sigma_ceiling <= 0.0 {
            issues.push("stability.sigma_ceiling must be positive and finite (§8.3)".into());
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Functional, FunctionalComponent, FunctionalRequires, StabilityRules};
    use crate::common::{Citation, License};

    pub(crate) fn pbe() -> Functional {
        let mut parameters = BTreeMap::new();
        parameters.insert("kappa".to_string(), 0.804);
        parameters.insert("mu".to_string(), 0.21951);
        Functional {
            name: "pbe".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_doi(
                "Perdew-Burke-Ernzerhof 1996",
                "10.1103/PhysRevLett.77.3865",
            )],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("c".repeat(64)),
            notes: None,
            components: vec![
                FunctionalComponent {
                    id: "GgaPbeX".into(),
                    weight: 1.0,
                },
                FunctionalComponent {
                    id: "GgaPbeC".into(),
                    weight: 1.0,
                },
            ],
            requires: FunctionalRequires {
                rho: true,
                sigma: true,
                laplacian: false,
                kinetic_density: false,
            },
            parameters,
            parameter_sources: BTreeMap::new(),
            stability: StabilityRules {
                density_floor: 1.0e-12,
                sigma_ceiling: 1.0e8,
            },
        }
    }

    #[test]
    fn pbe_is_valid() {
        let f = pbe();
        assert!(f.check().is_empty(), "{:?}", f.check());
        assert_eq!(f.parameters.get("kappa"), Some(&0.804));
    }

    /// FX-01：`requires.kinetic_density` 是**加性**扩展 —— 阶段一的数据文件里没有
    /// 这个键，必须原样解析成 `false`，否则升级 = 所有旧数据失效。
    #[test]
    fn kinetic_density_defaults_to_false_for_old_data() {
        let old_yaml = r#"
name: pbe
version: 1.0.0
citation:
  - title: PBE
    doi: 10.1103/PhysRevLett.77.3865
license: {spdx: Apache-2.0, commercial_ok: true, redistribute_ok: true}
created: 2025-01-10T08:00:00Z
components:
  - {id: GgaPbeX, weight: 1.0}
requires: {rho: true, sigma: true, laplacian: false}
parameters: {kappa: 0.804}
stability: {density_floor: 1.0e-12, sigma_ceiling: 1.0e8}
"#;
        let parsed: Functional = serde_yaml::from_str(old_yaml).expect("旧数据必须仍可解析");
        assert!(!parsed.requires.kinetic_density);
        assert!(
            parsed.check_structure().is_empty(),
            "{:?}",
            parsed.check_structure()
        );

        // 显式声明 τ：不再需要 σ（meta-GGA 至少要有 rho + τ）
        let meta_yaml = old_yaml.replace(
            "requires: {rho: true, sigma: true, laplacian: false}",
            "requires: {rho: true, sigma: true, laplacian: false, kinetic_density: true}",
        );
        let meta: Functional = serde_yaml::from_str(&meta_yaml).expect("meta-GGA 数据");
        assert!(meta.requires.kinetic_density);
        assert!(
            meta.check_structure().is_empty(),
            "声明 τ 不该被结构性检查抱怨：{:?}",
            meta.check_structure()
        );
    }

    /// 只有 τ（没有 rho）也算"声明了需求"，不能报「requires must set ...」。
    #[test]
    fn tau_only_requirement_is_not_empty() {
        let mut f = pbe();
        f.requires = FunctionalRequires {
            rho: true,
            sigma: false,
            laplacian: false,
            kinetic_density: true,
        };
        assert!(f.check_structure().is_empty(), "{:?}", f.check_structure());
        f.requires.rho = false;
        assert!(
            f.check_structure()
                .iter()
                .all(|issue| !issue.contains("at least one")),
            "τ 也算需求：{:?}",
            f.check_structure()
        );
    }

    #[test]
    fn laplacian_requirement_is_out_of_scope() {
        let mut f = pbe();
        f.requires.laplacian = true;
        assert!(f.check().iter().any(|m| m.contains("meta-GGA")));
    }

    #[test]
    fn check_catches_broken_functionals() {
        let mut f = pbe();
        f.components.clear();
        assert!(
            f.check()
                .iter()
                .any(|m| m.contains("components must not be empty"))
        );

        let mut f = pbe();
        f.requires = FunctionalRequires {
            rho: false,
            sigma: false,
            laplacian: false,
            kinetic_density: false,
        };
        assert!(f.check().iter().any(|m| m.contains("at least one")));

        let mut f = pbe();
        f.stability.density_floor = 0.0;
        assert!(f.check().iter().any(|m| m.contains("density_floor")));

        let mut f = pbe();
        f.parameters.insert("bad".into(), f64::NAN);
        assert!(f.check().iter().any(|m| m.contains("must be finite")));
    }
}
