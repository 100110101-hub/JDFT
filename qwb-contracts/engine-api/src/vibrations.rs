//! K7 · 振动频率与热化学的结果块（`jobtype: freq`，phase-2 `M7` 的接线部分）。
//!
//! 数值来自 `qwb_engine::hessian`（数值 Hessian → 质量加权简正模 → 谐振频率 → 理想气体
//! 热化学）。这里**只承载结果**，并且像其它阶段二块一样是**加性**的：旧结果没有这些键仍
//! 能解析（`serde(default)`），新结果缺省不写出（`skip_serializing_if`）。
//!
//! # 三条必须写进类型里的约定
//!
//! 1. **虚频用负波数表示**，不许写成"正数 + 标记"两套真值 —— 判据与显示都只认符号。
//!    [`VibrationalMode::imaginary`] 只是**派生**的布尔，给读者省一次判断，
//!    [`VibrationalFrequencies::validate`] 会核对它与 `wavenumber_cm` 的符号一致。
//! 2. **热化学带标准态**：`pressure_pa` 是配分函数里的压强（本项目默认 101 325 Pa = 1 atm），
//!    与相变/1 M 标准态的换算**不是**同一件事，读数的人必须能把它区分开。
//! 3. **`method` 是字符串而不是枚举**：现在只有 `numerical-hessian`，将来 RS-05 的解析
//!    Hessian 会加一个值 —— 加值不是加字段，旧读者不会因为多一个字符串就解析失败。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 1 Hartree 的焦耳数（每摩尔）。CODATA 2018，与 `qwb_engine::hessian` 用的是同一个数 ——
/// 两个 crate 各有一份常量，所以数值必须逐字相同（测试里核对）。
pub const HARTREE_J_PER_MOL: f64 = 2_625_499.639;

/// 一个简正模。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct VibrationalMode {
    /// 波数（cm⁻¹）。**虚频为负**。
    pub wavenumber_cm: f64,
    /// 红外强度（km/mol）；`None` = 未算（例如 `options.freq_ir = false`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ir_intensity_km_per_mol: Option<f64>,
    /// 是否虚频（由 `wavenumber_cm < 0` 派生，`validate` 会核对）。
    pub imaginary: bool,
}

impl VibrationalMode {
    /// 由波数构造（`imaginary` 自动按符号取，`ir_intensity` 由 [`Self::with_ir`] 补）。
    #[must_use]
    pub fn new(wavenumber_cm: f64) -> Self {
        Self {
            wavenumber_cm,
            ir_intensity_km_per_mol: None,
            imaginary: wavenumber_cm < 0.0,
        }
    }

    /// 补上红外强度。
    #[must_use]
    pub fn with_ir(mut self, intensity: f64) -> Self {
        self.ir_intensity_km_per_mol = Some(intensity);
        self
    }
}

/// 振动分析结果块。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct VibrationalFrequencies {
    /// 频率的来源（`numerical-hessian` / 将来 `analytic-hessian`）。
    pub method: String,
    /// 简正模，按波数升序（虚频在前）。
    pub modes: Vec<VibrationalMode>,
    /// 零点能（cm⁻¹），只含振动模（平移/转动不算）。
    pub zero_point_cm: f64,
    /// 振动模个数（非线性 `3N−6`、线性 `3N−5`）。
    pub vibrational_modes: usize,
    /// 笛卡尔自由度数 `3N`。
    pub degrees_of_freedom: usize,
}

impl VibrationalFrequencies {
    /// 构造（`method` 取冻结字符串）。
    #[must_use]
    pub fn new(
        method: impl Into<String>,
        modes: Vec<VibrationalMode>,
        zero_point_cm: f64,
        vibrational_modes: usize,
        degrees_of_freedom: usize,
    ) -> Self {
        Self {
            method: method.into(),
            modes,
            zero_point_cm,
            vibrational_modes,
            degrees_of_freedom,
        }
    }

    /// 虚频个数。
    #[must_use]
    pub fn imaginary_count(&self) -> usize {
        self.modes.iter().filter(|mode| mode.imaginary).count()
    }

    /// 自洽性检查（写出 `result.json` 之前跑）。
    ///
    /// # Errors
    ///
    /// 出现非有限数、模数超过 `3N`、模的个数与 `vibrational_modes` 不一致、
    /// 或 `imaginary` 与波数符号矛盾时返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.modes.len() != self.vibrational_modes {
            return Err(EngineError::Internal(format!(
                "vibrational_frequencies: {} modes but vibrational_modes = {}",
                self.modes.len(),
                self.vibrational_modes
            )));
        }
        if self.vibrational_modes > self.degrees_of_freedom {
            return Err(EngineError::Internal(
                "vibrational_frequencies: more vibrational modes than 3N degrees of freedom".into(),
            ));
        }
        for (index, mode) in self.modes.iter().enumerate() {
            if !mode.wavenumber_cm.is_finite()
                || mode
                    .ir_intensity_km_per_mol
                    .is_some_and(|value| !value.is_finite())
            {
                return Err(EngineError::Internal(format!(
                    "vibrational_frequencies.modes[{index}] is not finite"
                )));
            }
            if mode.imaginary != (mode.wavenumber_cm < 0.0) {
                return Err(EngineError::Internal(format!(
                    "vibrational_frequencies.modes[{index}]: imaginary = {} contradicts \
                     wavenumber = {} (an imaginary mode is a NEGATIVE wavenumber)",
                    mode.imaginary, mode.wavenumber_cm
                )));
            }
        }
        if !self.zero_point_cm.is_finite() || self.zero_point_cm < 0.0 {
            return Err(EngineError::Internal(
                "vibrational_frequencies: zero_point_cm must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// 理想气体热化学（谐振近似 + 刚性转子），298.15 K / 1 atm 由调用方写进来。
///
/// 单位约定：能量 `_hartree`、熵 `_j_per_mol_k`、温度 `_k`、压强 `_pa`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Thermochemistry {
    /// 温度（K）。
    pub temperature_k: f64,
    /// 配分函数用的压强（Pa）。
    pub pressure_pa: f64,
    /// 零点能（Hartree）。
    pub zero_point_hartree: f64,
    /// 内能热修正 `U(T) − E_elec`（含 ZPE，Hartree）。
    pub internal_energy_hartree: f64,
    /// 焓热修正 `H(T) − E_elec`（含 ZPE 与 `PV = kT`，Hartree）。
    pub enthalpy_hartree: f64,
    /// 熵（J/(mol·K)），含平动 + 转动 + 振动。
    pub entropy_j_per_mol_k: f64,
    /// 吉布斯自由能 `G = E_elec + H_corr − T·S`（Hartree）。
    pub gibbs_hartree: f64,
    /// 转动常数（cm⁻¹）；线性分子只有一个。
    pub rotational_constants_cm: Vec<f64>,
    /// 用到的**转动对称数 σ**（水 2、氨 3、甲烷 12、同核双原子 2）。
    ///
    /// 写进结果块而不是只写在日志里：S 与 G 都依赖它，读数的人必须能看出这个数是多少。
    /// 自动判定只覆盖原子/同核双原子；其余分子取 `options.symmetry_number`（缺省 1）。
    pub symmetry_number: u32,
}

impl Thermochemistry {
    /// 构造（字段多，按热化学表意顺序给）。
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature_k: f64,
        pressure_pa: f64,
        zero_point_hartree: f64,
        internal_energy_hartree: f64,
        enthalpy_hartree: f64,
        entropy_j_per_mol_k: f64,
        gibbs_hartree: f64,
        rotational_constants_cm: Vec<f64>,
        symmetry_number: u32,
    ) -> Self {
        Self {
            temperature_k,
            pressure_pa,
            zero_point_hartree,
            internal_energy_hartree,
            enthalpy_hartree,
            entropy_j_per_mol_k,
            gibbs_hartree,
            rotational_constants_cm,
            symmetry_number,
        }
    }

    /// 自洽性检查。
    ///
    /// # Errors
    ///
    /// 非有限数、温度/压强非正、熵为负、或 `G` 与 `H_corr − T·S` 对不上（> 1e-9 Ha）时返回
    /// [`EngineError::Internal`]。最后一条是有意的：`G` 是**派生量**，写出来的数必须与
    /// 写出来的 `H`、`S`、`T` 自洽，否则两份数据会各自"看起来正常"。
    pub fn validate(&self, electronic_energy_hartree: f64) -> Result<(), EngineError> {
        for (name, value) in [
            ("temperature_k", self.temperature_k),
            ("pressure_pa", self.pressure_pa),
            ("zero_point_hartree", self.zero_point_hartree),
            ("internal_energy_hartree", self.internal_energy_hartree),
            ("enthalpy_hartree", self.enthalpy_hartree),
            ("entropy_j_per_mol_k", self.entropy_j_per_mol_k),
            ("gibbs_hartree", self.gibbs_hartree),
        ] {
            if !value.is_finite() {
                return Err(EngineError::Internal(format!(
                    "thermochemistry.{name} is not finite"
                )));
            }
        }
        if self.temperature_k <= 0.0 || self.pressure_pa <= 0.0 {
            return Err(EngineError::Internal(
                "thermochemistry: temperature and pressure must be positive".into(),
            ));
        }
        if self.entropy_j_per_mol_k < 0.0 {
            return Err(EngineError::Internal(
                "thermochemistry: entropy must be non-negative".into(),
            ));
        }
        if self.symmetry_number == 0 {
            return Err(EngineError::Internal(
                "thermochemistry: symmetry_number must be >= 1 (a rotation symmetry number)".into(),
            ));
        }
        if self.zero_point_hartree < 0.0 {
            return Err(EngineError::Internal(
                "thermochemistry: zero_point_hartree must be non-negative".into(),
            ));
        }
        let expected = electronic_energy_hartree + self.enthalpy_hartree
            - self.temperature_k * self.entropy_j_per_mol_k / HARTREE_J_PER_MOL;
        if (self.gibbs_hartree - expected).abs() > 1.0e-9 {
            return Err(EngineError::Internal(format!(
                "thermochemistry: gibbs_hartree = {} contradicts E + H_corr − T·S = {}",
                self.gibbs_hartree, expected
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Thermochemistry, VibrationalFrequencies, VibrationalMode};

    fn water() -> VibrationalFrequencies {
        VibrationalFrequencies::new(
            "numerical-hessian",
            vec![
                VibrationalMode::new(1648.5).with_ir(21.4),
                VibrationalMode::new(3802.1).with_ir(1.8),
                VibrationalMode::new(3901.7).with_ir(65.2),
            ],
            4676.15,
            3,
            9,
        )
    }

    #[test]
    fn imaginary_modes_are_negative_wavenumbers() {
        let mut block = water();
        assert_eq!(block.imaginary_count(), 0);
        assert!(block.validate().is_ok());

        block.modes.insert(0, VibrationalMode::new(-512.0));
        block.vibrational_modes = 4;
        assert_eq!(block.imaginary_count(), 1);
        assert!(block.modes[0].imaginary);
        assert!(block.validate().is_ok(), "负波数 = 虚频，自洽");

        // 反过来写"正波数但标记为虚"必须被拒。
        let mut bad = water();
        bad.modes[0].imaginary = true;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn mode_count_must_match_the_declared_number() {
        let mut block = water();
        block.vibrational_modes = 4;
        assert!(block.validate().is_err());
        let mut block = water();
        block.degrees_of_freedom = 2;
        assert!(block.validate().is_err());
        let mut block = water();
        block.zero_point_cm = -1.0;
        assert!(block.validate().is_err());
    }

    #[test]
    fn thermochemistry_must_be_internally_consistent() {
        // H2O/STO-3G/HF 的量级：E = -74.9631 Ha、ZPE ≈ 0.0245 Ha、H_corr ≈ 0.0263 Ha、
        // S ≈ 45.1 J/(mol·K)。
        let electronic = -74.963_146_409_2;
        let enthalpy = 0.026_3;
        let entropy = 45.1;
        let gibbs = electronic + enthalpy - 298.15 * entropy / 2_625_499.639;
        let block = Thermochemistry::new(
            298.15,
            101_325.0,
            0.024_5,
            enthalpy,
            enthalpy,
            entropy,
            gibbs,
            vec![27.88, 14.51, 9.48],
            2,
        );
        assert!(block.validate(electronic).is_ok());

        let mut bad = block.clone();
        bad.gibbs_hartree += 1.0e-6;
        assert!(bad.validate(electronic).is_err(), "G 与 H、S、T 必须自洽");

        let mut bad = block.clone();
        bad.entropy_j_per_mol_k = -1.0;
        assert!(bad.validate(electronic).is_err());
    }

    #[test]
    fn blocks_roundtrip_through_json() {
        let block = water();
        let json = serde_json::to_string(&block).expect("serialize");
        assert!(json.contains("\"method\":\"numerical-hessian\""));
        let back: VibrationalFrequencies = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);

        // 没有红外强度时该键不出现（加性契约：旧读者不受影响）。
        let plain = VibrationalFrequencies::new(
            "numerical-hessian",
            vec![VibrationalMode::new(5476.8)],
            2738.4,
            1,
            6,
        );
        let json = serde_json::to_string(&plain).expect("serialize");
        assert!(!json.contains("ir_intensity"), "{json}");
    }
}
