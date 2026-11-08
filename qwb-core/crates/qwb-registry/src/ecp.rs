//! ECP 数据（§4 K1 / §8.2）。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;

/// `r` 幂次约定（§8.2 显式声明的陷阱）。
///
/// Stuttgart 系的表格常用 `r^(n-2)`，其他来源常用 `r^n`。字符差异会直接改变
/// 非局域项的量级，所以必须在数据里声明，由加载器统一转成内部约定。
///
/// **内部约定是 `r^n`**（`qwb_scf::ecp` 把 `power` 直接当 `r` 的指数用），所以
/// `r^(n-2)` 的数据在 [`Ecp::normalise`] 里把每一项的 `power` 减 2 —— 归一后的
/// 内部幂次因此落在 `[-2, 2]`（`r^-2 … r^2`），这正是 `qwb_scf::ecp::radial_power`
/// 显式覆盖的范围。
///
/// # 为什么 BSE 属于 `r^(n-2)`
///
/// BSE 的 `r_exponents` 值 `n` 对应的算符项是 `c r^(n-2) e^(-ζ r^2)`：
///
/// * BSE 自己的 CRYSTAL 写出器在写"完整幂次"时做 `rexp - 2`，并注明
///   "r exponent in CRYSTAL format is the full one: it does NOT have the r^{-2}
///   prefactor as in other formats"（`basis_set_exchange/writers/crystal.py`）；
/// * libcint/PySCF 的 ECP 块同样把 `r_exponents` 当成 `n` 去算 `r^(n-2)`
///   （实测：单个 Gauss 项 `rorder = 1` 给出的积分等于 `c r^-1 e^(-ζr^2)` 的解析值）。
///
/// LANL2DZ 的局域通道正好印证：Zn 的 `-18.0`（= `n_core`）落在 `r^(1-2) = r^-1` 上，
/// 也就是 `-Z_core/r` 那一项。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum RPowerConvention {
    /// 表格里写的是 `r^n`（= 内部约定，加载后一定是它）。
    #[serde(rename = "r^n")]
    Rn,
    /// 表格里写的是 `r^(n-2)`（BSE 的 `r_exponents` 就是这一种）。
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
///
/// **语义（§8.2 的第二个陷阱）**：这些块是**相对局域通道的差分** `ΔV_l = V_l − V_L`
/// （`L` = 导出里最大的 `l`，见 [`Ecp::local_part`]），不是绝对 `V_l`。EMSL/BSE 的
/// `ecp_potentials` 就是这么给的，PySCF/libcint 也按这个语义算：实测
/// `ECPscalar == V_local + Σ_l V_l P_l`（差 `8.9e-16`，见
/// `qwb-core/crates/qwb-scf/tests/i05_ecp_oracle.rs`）。所以算子
/// `V_ECP = V_L + Σ_{l<L} ΔV_l P_l`，通道那一项**不许**再减一次 `V_L` ——
/// 2026-11 之前就是这么错的，Zn 的 d 通道因此只剩 1/18。
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

impl_data_common!(Ecp, normalise: Ecp::normalise);

impl Ecp {
    /// 把 `r` 幂次归一成**内部约定** `r^n`（加载器在 `check_structure` 之前调用）。
    ///
    /// `r^(n-2)` 的数据（BSE 导出的 ECP 就是这一种，见 [`RPowerConvention`]）每一项的
    /// `power` 减 2，然后把 `r_power_convention` 改写成 [`RPowerConvention::Rn`]，所以
    /// 归一之后的数据只有一种形态，消费方不需要再分支。幂等：已经是 `r^n` 的数据原样返回。
    ///
    /// 归一后的内部幂次是 `[-2, 2]`（局域/投影通道里的 `r^-2`、`r^-1` 项），
    /// `qwb_scf::ecp` 的两条实现都按 `r^n` 用它们。
    pub fn normalise(&mut self) {
        let offset = match self.r_power_convention {
            RPowerConvention::Rn => return,
            RPowerConvention::RnMinus2 => -2,
        };
        for term in &mut self.local_part.terms {
            term.power += offset;
        }
        for part in &mut self.nonlocal_parts {
            for term in &mut part.terms {
                term.power += offset;
            }
        }
        self.r_power_convention = RPowerConvention::Rn;
    }

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
        // 原始文件里是表格的幂次（`r^n` 或 `r^(n-2)`，两者都有 `power >= 0`）；归一之后
        // 是内部 `r^n`，BSE 那一套会落成 `r^-2 … r^0`。两条路径共用一个下界 −2：
        // `r^(-3)` 会让 `∫ r^m e^(-γr²)` 在原点发散（见 `qwb_scf::ecp::radial_integral`），
        // 所以比 −2 更负的幂次是数据错误。
        if term.power < -2 {
            issues.push(format!(
                "{at}.terms[{j}].power = {} must be >= -2 (r^-2 is the most singular \
                 integrable factor for a Gaussian ECP)",
                term.power
            ));
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
