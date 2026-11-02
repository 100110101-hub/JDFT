//! K8c · 过渡态搜索的输入选项与结果块（phase-2 `RP-01`）。
//!
//! # 为什么结果块里带「虚频个数」和「唯一虚频的波数」
//!
//! 过渡态的定义是**一阶鞍点**：`max|grad|` 小只说明停在了驻点上，极小点也满足它。所以
//! RP-01 的判据是两条一起看（`max|grad| < 4.5e-4` **且** Hessian 恰有一个虚频），这个块
//! 就必须同时报出这两个数 —— 只报梯度会让"收敛到了一个极小点"看起来像成功。
//!
//! 虚频用**负波数**表示（与 `VibrationalFrequencies` 同一个约定），`imaginary_wavenumber_cm`
//! 因此是一个负数；`n_imaginary` 是独立的整数，两者不一致时 `validate` 会报错。
use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 过渡态搜索的选项（`options.ts`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TsOptions {
    /// 最多多少步（每步 = 一次数值 Hessian）。
    pub max_steps: u32,
    /// 收敛判据：`max|grad|`（Ha/Bohr）。阶段二的判据是 4.5e-4。
    pub gradient_threshold_hartree_per_bohr: f64,
}

impl TsOptions {
    /// 缺省：100 步、判据 4.5e-4 Ha/Bohr（与 T3 的力判据一致）。
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_steps: 100,
            gradient_threshold_hartree_per_bohr: 4.5e-4,
        }
    }

    /// 改步数上限。
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 改收敛判据。
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.gradient_threshold_hartree_per_bohr = threshold;
        self
    }
}

impl Default for TsOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// 过渡态搜索的结果块。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TsResult {
    /// 迭代步数（每步一次数值 Hessian）。
    pub iterations: u32,
    /// 收敛后的能量（Hartree）。
    pub energy_hartree: f64,
    /// 收敛后的 `max|grad|`（Ha/Bohr）。
    pub max_gradient_hartree_per_bohr: f64,
    /// 虚频个数（一阶鞍点必须是 1）。
    ///
    /// **按"真虚频"计**：3N 个本征值里有 3 个平动 + 3 个转动是零模，数值噪声让它们落在 0
    /// 两侧（实测 NH3/STO-3G 的平面几何上有 −16.4 与 −0.016 cm⁻¹ 两个噪声点）。所以这里
    /// 只统计波数低于 **−100 cm⁻¹** 的那些（与 `freq` 的判据同一个阈值），否则"恰好一个虚频"
    /// 这条判据会把所有分子都判成失败。
    pub n_imaginary: u32,
    /// 最负的虚频波数（cm⁻¹，**负值**）；没有真虚频时为 0。
    pub imaginary_wavenumber_cm: f64,
}

/// "真虚频"的波数阈值（cm⁻¹）：零模数值噪声实测到 −48.7（H2O）、−16.4（NH3），
/// 而鞍点的伞形/伸缩虚频是 −893…−1081 量级，−100 把两者分开。
pub const IMAGINARY_THRESHOLD_CM: f64 = -100.0;

impl TsResult {
    /// 构造（`n_imaginary` 显式传入：零模噪声会让"数负本征值"给出错误的个数）。
    #[must_use]
    pub fn new(
        iterations: u32,
        energy_hartree: f64,
        max_gradient_hartree_per_bohr: f64,
        n_imaginary: u32,
        imaginary_wavenumber_cm: f64,
    ) -> Self {
        Self {
            iterations,
            energy_hartree,
            max_gradient_hartree_per_bohr,
            n_imaginary,
            imaginary_wavenumber_cm,
        }
    }

    /// 这个结果是不是一阶鞍点（RP-01 的两条判据一起看）。
    #[must_use]
    pub fn is_first_order_saddle(&self, threshold: f64) -> bool {
        self.n_imaginary == 1 && self.max_gradient_hartree_per_bohr < threshold
    }

    /// 自洽性检查。
    ///
    /// # Errors
    ///
    /// 非有限数、`max|grad|` 为负、`n_imaginary` 与虚频波数符号矛盾时返回
    /// [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        for (name, value) in [
            ("energy_hartree", self.energy_hartree),
            (
                "max_gradient_hartree_per_bohr",
                self.max_gradient_hartree_per_bohr,
            ),
            ("imaginary_wavenumber_cm", self.imaginary_wavenumber_cm),
        ] {
            if !value.is_finite() {
                return Err(EngineError::Internal(format!("ts.{name} is not finite")));
            }
        }
        if self.max_gradient_hartree_per_bohr < 0.0 {
            return Err(EngineError::Internal(
                "ts.max_gradient_hartree_per_bohr must be non-negative".into(),
            ));
        }
        if self.n_imaginary == 0 && self.imaginary_wavenumber_cm != 0.0 {
            return Err(EngineError::Internal(format!(
                "ts: n_imaginary = 0 but imaginary_wavenumber_cm = {}",
                self.imaginary_wavenumber_cm
            )));
        }
        if self.n_imaginary > 0 && self.imaginary_wavenumber_cm >= 0.0 {
            return Err(EngineError::Internal(format!(
                "ts: n_imaginary = {} but imaginary_wavenumber_cm = {} (must be negative)",
                self.n_imaginary, self.imaginary_wavenumber_cm
            )));
        }
        if self.n_imaginary > 0 && self.imaginary_wavenumber_cm > IMAGINARY_THRESHOLD_CM {
            return Err(EngineError::Internal(format!(
                "ts: imaginary_wavenumber_cm = {} is above the -100 cm^-1 threshold; \
                 zero-mode numerical noise must not be counted as an imaginary mode",
                self.imaginary_wavenumber_cm
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{TsOptions, TsResult};

    #[test]
    fn options_default_to_the_phase_two_criterion() {
        let options = TsOptions::new();
        assert_eq!(options.max_steps, 100);
        assert!((options.gradient_threshold_hartree_per_bohr - 4.5e-4).abs() < 1e-15);
        assert_eq!(TsOptions::default(), options);
        assert_eq!(options.clone().with_max_steps(5).max_steps, 5);
        assert!(
            (options
                .with_threshold(1e-3)
                .gradient_threshold_hartree_per_bohr
                - 1e-3)
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn a_first_order_saddle_needs_both_criteria() {
        let saddle = TsResult::new(12, -76.3, 1.0e-4, 1, -523.4);
        assert_eq!(saddle.n_imaginary, 1);
        assert!(saddle.is_first_order_saddle(4.5e-4));
        assert!(saddle.validate().is_ok());

        // 梯度小但没有虚频 = 停在了极小点，不算过渡态。
        let minimum = TsResult::new(3, -76.4, 1.0e-5, 0, 0.0);
        assert_eq!(minimum.n_imaginary, 0);
        assert!(!minimum.is_first_order_saddle(4.5e-4));

        // 梯度没收敛（虚频对）也不算。
        let loose = TsResult::new(3, -76.4, 5.0e-3, 1, -523.4);
        assert!(!loose.is_first_order_saddle(4.5e-4));
    }

    #[test]
    fn validate_catches_the_usual_drift() {
        let mut block = TsResult::new(12, -76.3, 1.0e-4, 1, -523.4);
        block.n_imaginary = 0;
        assert!(block.validate().is_err(), "n_imaginary 与波数必须一致");

        // 零模噪声（−16 cm⁻¹）不算虚频。
        assert!(
            TsResult::new(1, -76.3, 1.0e-4, 1, -16.4)
                .validate()
                .is_err()
        );

        let mut block = TsResult::new(12, -76.3, 1.0e-4, 1, -523.4);
        block.max_gradient_hartree_per_bohr = -1.0;
        assert!(block.validate().is_err());

        let mut block = TsResult::new(12, -76.3, 1.0e-4, 1, -523.4);
        block.energy_hartree = f64::NAN;
        assert!(block.validate().is_err());
    }

    #[test]
    fn block_roundtrips_through_json() {
        let block = TsResult::new(12, -76.3, 1.0e-4, 1, -523.4);
        let json = serde_json::to_string(&block).expect("serialize");
        let back: TsResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);
        assert!(json.contains("\"n_imaginary\":1"));
    }
}
