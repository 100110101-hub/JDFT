//! ECP 数据（§4 K1 / §8.2）。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;

/// `r` 幂次约定（§8.2 显式声明的陷阱）。
///
/// Stuttgart 系的表格常用 `r^(n-2)`，其他来源常用 `r^n`。字符差异会直接改变
/// 非局域项的量级，所以必须在数据里声明，由加载器统一转成内部约定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum RPowerConvention {
    /// 表格里写的是 `r^n`。
    #[serde(rename = "r^n")]
    Rn,
    /// 表格里写的是 `r^(n-2)`。
    #[serde(rename = "r^(n-2)")]
    RnMinus2,
}

/// 一个 ECP 高斯项：`coefficient * r^power * exp(-exponent * r^2)`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EcpTerm {
    /// `r` 的幂次 `n`（按 `r_power_convention` 解释）。
    pub power: i32,
    /// 高斯指数，单位 Bohr^-2。
    pub exponent: f64,
    /// 系数。
    pub coefficient: f64,
}

/// 局域部分。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EcpLocalPart {
    /// 该数据块在来源中的标识（便于逐项核对）。
    pub id: String,
    /// 项列表。
    pub terms: Vec<EcpTerm>,
}

/// 非局域部分（按角量子数 `l` 分组）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EcpNonlocalPart {
    /// 角量子数 `l`。
    pub l: u32,
    /// 该数据块在来源中的标识。
    pub id: String,
    /// 项列表。
    pub terms: Vec<EcpTerm>,
}

/// ECP（`qwb-data/ecp/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Ecp {
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
    /// 元素符号。
    pub element: String,
    /// 芯电子数。
    pub n_core_electrons: u32,
    /// 有效核电荷 `Z_eff = Z - n_core_electrons`。
    pub z_eff: f64,
    /// `r` 幂次约定。
    pub r_power_convention: RPowerConvention,
    /// 局域部分。
    pub local_part: EcpLocalPart,
    /// 非局域部分，按 `l` 升序且不重复。
    pub nonlocal_parts: Vec<EcpNonlocalPart>,
}

impl_data_common!(Ecp);

impl Ecp {
    /// 校验 `z_eff == Z - n_core` 是否与元素表一致（T5 断言的前置条件）。
    ///
    /// # Errors
    ///
    /// 元素未知、`z_eff` 不自洽时返回说明字符串。
    pub fn z_eff_consistent(&self) -> Result<(), String> {
        let z = elements::atomic_number(&self.element)
            .ok_or_else(|| format!("unknown element `{}`", self.element))?;
        let expected = f64::from(z) - f64::from(self.n_core_electrons);
        if (expected - self.z_eff).abs() > 1e-9 {
            return Err(format!(
                "z_eff = {} but Z({}) - n_core({}) = {}",
                self.z_eff, self.element, self.n_core_electrons, expected
            ));
        }
        Ok(())
    }

    /// 语义校验（G2 的 ECP 部分）。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();

        match self.z_eff_consistent() {
            Ok(()) => {}
            Err(msg) => issues.push(msg),
        }
        if self.n_core_electrons == 0 {
            issues.push(
                "n_core_electrons must be >= 1 (a full-electron potential is not an ECP)".into(),
            );
        }
        if !self.z_eff.is_finite() || self.z_eff <= 0.0 {
            issues.push("z_eff must be positive and finite".into());
        }
        if self.local_part.terms.is_empty() {
            issues.push("local_part.terms must not be empty".into());
        }
        check_terms(&mut issues, "local_part", &self.local_part.terms);

        if self.nonlocal_parts.is_empty() {
            issues.push(
                "nonlocal_parts must not be empty (an ECP without nonlocal parts is local-only)"
                    .into(),
            );
        }
        let mut seen_l = std::collections::BTreeSet::new();
        for (i, part) in self.nonlocal_parts.iter().enumerate() {
            if !seen_l.insert(part.l) {
                issues.push(format!("nonlocal_parts[{i}] duplicates l = {}", part.l));
            }
            if part.l > 4 {
                issues.push(format!(
                    "nonlocal_parts[{i}].l = {} exceeds first-phase limit 4",
                    part.l
                ));
            }
            if part.terms.is_empty() {
                issues.push(format!("nonlocal_parts[{i}].terms must not be empty"));
            }
            check_terms(&mut issues, &format!("nonlocal_parts[{i}]"), &part.terms);
        }
        issues
    }
}

fn check_terms(issues: &mut Vec<String>, at: &str, terms: &[EcpTerm]) {
    for (j, term) in terms.iter().enumerate() {
        if !term.exponent.is_finite() || term.exponent <= 0.0 {
            issues.push(format!(
                "{at}.terms[{j}].exponent must be positive and finite"
            ));
        }
        if !term.coefficient.is_finite() {
            issues.push(format!("{at}.terms[{j}].coefficient must be finite"));
        }
        if term.power < 0 {
            issues.push(format!("{at}.terms[{j}].power must be >= 0"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Ecp, EcpLocalPart, EcpNonlocalPart, EcpTerm, RPowerConvention};
    use crate::common::{Citation, License};

    fn cu_ecp() -> Ecp {
        Ecp {
            name: "lanl2dz".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_doi("Hay & Wadt 1985", "10.1063/1.447079")],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("b".repeat(64)),
            notes: None,
            element: "Cu".into(),
            n_core_electrons: 28,
            z_eff: 1.0,
            r_power_convention: RPowerConvention::RnMinus2,
            local_part: EcpLocalPart {
                id: "local".into(),
                terms: vec![EcpTerm {
                    power: 2,
                    exponent: 1.5,
                    coefficient: -3.0,
                }],
            },
            nonlocal_parts: vec![EcpNonlocalPart {
                l: 2,
                id: "d".into(),
                terms: vec![EcpTerm {
                    power: 2,
                    exponent: 2.5,
                    coefficient: 0.5,
                }],
            }],
        }
    }

    #[test]
    fn copper_lanl2dz_is_consistent() {
        let e = cu_ecp();
        assert!(e.check().is_empty(), "{:?}", e.check());
        assert!(e.z_eff_consistent().is_ok());
    }

    #[test]
    fn z_eff_must_equal_z_minus_n_core() {
        let mut e = cu_ecp();
        e.z_eff = 11.0;
        assert!(e.z_eff_consistent().is_err());
        assert!(e.check().iter().any(|m| m.contains("z_eff")));
    }

    #[test]
    fn check_catches_broken_parts() {
        let mut e = cu_ecp();
        e.nonlocal_parts.clear();
        assert!(e.check().iter().any(|m| m.contains("nonlocal_parts")));

        let mut e = cu_ecp();
        e.element = "U".into();
        assert!(e.check().iter().any(|m| m.contains("unknown element")));

        let mut e = cu_ecp();
        e.local_part.terms[0].exponent = -1.0;
        assert!(e.check().iter().any(|m| m.contains("exponent")));

        let mut e = cu_ecp();
        e.nonlocal_parts.push(e.nonlocal_parts[0].clone());
        assert!(e.check().iter().any(|m| m.contains("duplicates l")));
    }
}
