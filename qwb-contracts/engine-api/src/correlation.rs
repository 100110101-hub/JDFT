//! K9 · 开壳层与相关方法（phase-2 §4.4）。
//!
//! 两块结果（[`SpinResult`] / [`CorrelationResult`]）+ 两个输入选项（[`ScfType`] /
//! [`Mp2Options`]）。设计约束来自 phase-2 的判决：
//!
//! - **开壳层的自旋污染必须报出来**（§4.4 门禁 ②）：`⟨S²⟩` 与偏差进 [`SpinResult`]，
//!   不允许"收敛了就不提"。
//! - **相关方法只做 canonical MP2 与 RI-MP2**（§13 #26）：[`Mp2Kind`] 只有两个变体，
//!   RI 的拟合误差必须落盘（[`CorrelationResult::ri_error_hartree`]，K10 留痕）。
//! - `scf_type` 只决定**自旋处理**（R/U/RO），HF 还是 DFT 由 `method_name` 决定；
//!   两者的自洽性由 [`crate::JobInput::validate`] 检查（避免"uhf + pbe"这种自相矛盾的输入）。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// SCF 类型（phase-2 §4.1 的 `TaskOptions.scf_type`）。
///
/// 前三个是 HF 变体，后两个是 KS 变体：与 `method_name == "hf"` 的一致性由
/// [`crate::JobInput::validate`] 强制（phase-2 §4.4）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ScfType {
    /// 闭壳层 RHF。
    Rhf,
    /// 限制性开壳层 ROHF。
    Rohf,
    /// 非限制性 UHF。
    Uhf,
    /// 闭壳层 RKS（DFT）。
    Rks,
    /// 非限制性 UKS（DFT）。
    Uks,
}

impl ScfType {
    /// 冻结的小写名（进 `input.yaml`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rhf => "rhf",
            Self::Rohf => "rohf",
            Self::Uhf => "uhf",
            Self::Rks => "rks",
            Self::Uks => "uks",
        }
    }

    /// 是否 HF 变体（`true` 时要求 `method_name == "hf"`）。
    #[must_use]
    pub fn is_hartree_fock(self) -> bool {
        matches!(self, Self::Rhf | Self::Rohf | Self::Uhf)
    }

    /// 是否开壳层（U/RO；`true` 时 `multiplicity > 1` 才合法）。
    #[must_use]
    pub fn is_open_shell(self) -> bool {
        matches!(self, Self::Rohf | Self::Uhf | Self::Uks)
    }

    /// 是否是**非限制性**（双套系数）——`rohf` 不是。
    #[must_use]
    pub fn is_unrestricted(self) -> bool {
        matches!(self, Self::Uhf | Self::Uks)
    }
}

/// 相关方法（phase-2 §13 #26：只有这两个）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Mp2Kind {
    /// Canonical MP2（4 中心积分，慢但无拟合误差）。
    Mp2,
    /// RI-MP2（密度拟合；拟合误差必须落盘）。
    #[serde(rename = "ri-mp2")]
    RiMp2,
}

impl Mp2Kind {
    /// 冻结的拼写（`mp2` / `ri-mp2`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mp2 => "mp2",
            Self::RiMp2 => "ri-mp2",
        }
    }
}

/// MP2 选项（`TaskOptions.mp2`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Mp2Options {
    /// 方法（canonical / RI）。
    pub kind: Mp2Kind,
    /// 辅助基名（RI 必填；canonical 忽略）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_basis_name: Option<String>,
}

impl Default for Mp2Options {
    fn default() -> Self {
        Self {
            kind: Mp2Kind::RiMp2,
            aux_basis_name: None,
        }
    }
}

impl Mp2Options {
    /// 用方法构造。
    #[must_use]
    pub fn new(kind: Mp2Kind) -> Self {
        Self {
            kind,
            aux_basis_name: None,
        }
    }

    /// 补上辅助基。
    #[must_use]
    pub fn with_aux_basis(mut self, aux_basis_name: impl Into<String>) -> Self {
        self.aux_basis_name = Some(aux_basis_name.into());
        self
    }

    /// 校验：RI 必须有辅助基。
    ///
    /// # Errors
    ///
    /// `kind == RiMp2` 而 `aux_basis_name` 缺失/空白时返回 [`EngineError::InvalidInput`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.kind == Mp2Kind::RiMp2 {
            match self.aux_basis_name.as_deref() {
                Some(name) if !name.trim().is_empty() => {}
                _ => {
                    return Err(EngineError::InvalidInput(
                        "options.mp2.aux_basis_name is required for ri-mp2".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// 自旋结果（K9）：`⟨S²⟩` 与污染必须无条件报出来。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SpinResult {
    /// α 电子数。
    pub n_alpha: u32,
    /// β 电子数。
    pub n_beta: u32,
    /// 自旋多重度（`2S+1`）。
    pub multiplicity: u32,
    /// `⟨S²⟩` 的期望值。
    pub s_squared: f64,
    /// 污染量 `⟨S²⟩ − S(S+1)`；闭壳层/纯自旋态应 ≈ 0。
    pub spin_contamination: f64,
    /// 是否检测到破对称解（同一体系有多种 SCF 解时的选择）。
    pub broken_symmetry: bool,
}

impl SpinResult {
    /// 构造。
    #[must_use]
    pub fn new(n_alpha: u32, n_beta: u32, multiplicity: u32, s_squared: f64) -> Self {
        let spin = (f64::from(multiplicity) - 1.0) / 2.0;
        Self {
            n_alpha,
            n_beta,
            multiplicity,
            s_squared,
            spin_contamination: s_squared - spin * (spin + 1.0),
            broken_symmetry: false,
        }
    }

    /// 标记破对称解。
    #[must_use]
    pub fn with_broken_symmetry(mut self, broken_symmetry: bool) -> Self {
        self.broken_symmetry = broken_symmetry;
        self
    }

    /// 污染是否在可接受范围（`|⟨S²⟩ − S(S+1)| < tol`）。
    #[must_use]
    pub fn is_spin_pure(&self, tol: f64) -> bool {
        self.spin_contamination.abs() < tol
    }
}

/// 相关能结果块（K9）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CorrelationResult {
    /// 方法（`mp2` / `ri-mp2`）。
    pub method: Mp2Kind,
    /// 二阶相关能，单位 Hartree（负值）。
    pub e2_hartree: f64,
    /// 含相关的总能量，单位 Hartree（= SCF + `e2`）。
    pub e_total_hartree: f64,
    /// RI 用的辅助基（canonical 为 `None`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_basis_name: Option<String>,
    /// RI 拟合误差（与 canonical 同基组的差），Hartree；K10 要求它落盘。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ri_error_hartree: Option<f64>,
}

impl CorrelationResult {
    /// 构造（`aux_basis_name` / `ri_error_hartree` 默认空）。
    #[must_use]
    pub fn new(kind: Mp2Kind, e2_hartree: f64, e_total_hartree: f64) -> Self {
        Self {
            method: kind,
            e2_hartree,
            e_total_hartree,
            aux_basis_name: None,
            ri_error_hartree: None,
        }
    }

    /// 补上辅助基。
    #[must_use]
    pub fn with_aux_basis(mut self, aux_basis_name: impl Into<String>) -> Self {
        self.aux_basis_name = Some(aux_basis_name.into());
        self
    }

    /// 补上 RI 拟合误差。
    #[must_use]
    pub fn with_ri_error(mut self, ri_error_hartree: f64) -> Self {
        self.ri_error_hartree = Some(ri_error_hartree);
        self
    }

    /// 自洽性检查：`e_total − (e_total − e2) ≈ e2`，且相关能为负、有限。
    ///
    /// # Errors
    ///
    /// 数值非有限或相关能不为负时返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.e2_hartree.is_finite() || !self.e_total_hartree.is_finite() {
            return Err(EngineError::Internal(
                "correlation result contains a non-finite energy".into(),
            ));
        }
        if self.e2_hartree > 0.0 {
            return Err(EngineError::Internal(format!(
                "mp2 correlation energy must be negative, got {}",
                self.e2_hartree
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CorrelationResult, Mp2Kind, Mp2Options, ScfType, SpinResult};

    #[test]
    fn scf_type_spelling_and_roles_are_frozen() {
        assert_eq!(ScfType::Uks.as_str(), "uks");
        assert_eq!(serde_json::to_string(&ScfType::Rohf).unwrap(), "\"rohf\"");
        assert!(ScfType::Uhf.is_hartree_fock());
        assert!(!ScfType::Uks.is_hartree_fock());
        assert!(ScfType::Uks.is_open_shell());
        assert!(ScfType::Uks.is_unrestricted());
        assert!(!ScfType::Rohf.is_unrestricted());
        assert!(!ScfType::Rhf.is_open_shell());
    }

    #[test]
    fn mp2_kind_spelling_and_aux_requirement() {
        assert_eq!(Mp2Kind::RiMp2.as_str(), "ri-mp2");
        assert_eq!(
            serde_json::to_string(&Mp2Kind::RiMp2).unwrap(),
            "\"ri-mp2\""
        );
        assert!(Mp2Options::new(Mp2Kind::RiMp2).validate().is_err());
        assert!(
            Mp2Options::new(Mp2Kind::RiMp2)
                .with_aux_basis("def2-svp-ri")
                .validate()
                .is_ok()
        );
        assert!(Mp2Options::new(Mp2Kind::Mp2).validate().is_ok());
    }

    #[test]
    fn spin_result_reports_contamination() {
        let triplet = SpinResult::new(5, 5, 3, 2.0);
        assert!((triplet.spin_contamination - 0.0).abs() < 1e-12);
        assert!(triplet.is_spin_pure(1e-6));

        let contaminated = SpinResult::new(5, 4, 3, 2.4);
        assert!((contaminated.spin_contamination - 0.4).abs() < 1e-12);
        assert!(!contaminated.is_spin_pure(1e-6));
        assert!(!contaminated.broken_symmetry);

        let json = serde_json::to_string(&contaminated).expect("serialize");
        let back: SpinResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, contaminated);
    }

    #[test]
    fn correlation_result_checks_its_sign() {
        let ok =
            CorrelationResult::new(Mp2Kind::RiMp2, -0.31, -76.33).with_aux_basis("def2-svp-ri");
        assert!(ok.validate().is_ok());
        assert_eq!(ok.ri_error_hartree, None);

        let with_error = ok.clone().with_ri_error(-1.0e-5);
        assert_eq!(with_error.ri_error_hartree, Some(-1.0e-5));

        let positive = CorrelationResult::new(Mp2Kind::Mp2, 0.1, -76.0);
        assert!(positive.validate().is_err());
    }
}
