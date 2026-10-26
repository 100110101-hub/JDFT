//! K8 · 溶剂化的输入选项与结果块（phase-2 §4.3）。
//!
//! 三个约束来自 phase-2 的判决：
//!
//! - **模型是显式选择**（§8.2）：[`SolventModel`] 没有"默认猜一个"的分支 —— 输入不给就用
//!   数据里 `cavity_model` 的默认，数据也不给就是 `vacuum`。
//! - **能量分解必须能逐项对账**：SMD 的四项（静电/空化/色散/排斥）每一项都可选，
//!   但 `total = Σ 各项` 必须能被 [`SolvationResult::decomposition_gap_hartree`] 查出来。
//! - **`converged` 是独立判定**（`S-06` 风险）：表面电荷迭代不收敛时不得悄悄返回能量。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 隐式溶剂模型（K2 输入侧的词汇；K1 数据的 `cavity_model` 由引擎映射过来）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SolventModel {
    /// 气相（无溶剂）。
    Vacuum,
    /// COSMO / C-PCM 一族（导体屏蔽近似）。
    Cosmo,
    /// SMD（COSMO + 4 项非极性校正 + 参数化半径）。
    Smd,
    /// Poisson–Boltzmann（含离子强度；phase-2 `stretch`）。
    Pbf,
}

impl SolventModel {
    /// 冻结的小写名（进 `input.yaml` / `result.json`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vacuum => "vacuum",
            Self::Cosmo => "cosmo",
            Self::Smd => "smd",
            Self::Pbf => "pbf",
        }
    }

    /// 是否需要空腔/表面（真空不需要）。
    #[must_use]
    pub fn needs_cavity(self) -> bool {
        !matches!(self, Self::Vacuum)
    }
}

/// 空腔几何（表面积 / 体积 / tesserae 数）——三者总是一起出现，合成一个结构避免
/// `SolvationResult::new` 参数爆炸，也让"空腔校验"集中在一个地方。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CavityGeometry {
    /// 可及表面积，单位 Å²。
    pub surface_area_angstrom2: f64,
    /// 可及体积，单位 Å³。
    pub surface_volume_angstrom3: f64,
    /// 表面 tesserae 数（收敛性与可复现性都依赖它）。
    pub n_tesserae: usize,
}

impl CavityGeometry {
    /// 构造。
    #[must_use]
    pub fn new(
        surface_area_angstrom2: f64,
        surface_volume_angstrom3: f64,
        n_tesserae: usize,
    ) -> Self {
        Self {
            surface_area_angstrom2,
            surface_volume_angstrom3,
            n_tesserae,
        }
    }
}

/// 溶剂化结果块（K8）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SolvationResult {
    /// 用的模型。
    pub model: SolventModel,
    /// 溶剂键名（`qwb-data/solvent/` 的 name）。
    pub solvent_name: String,
    /// 含溶剂的总能量，单位 Hartree。
    pub total_hartree: f64,
    /// 静电（极化 + 表面电荷）贡献，单位 Hartree（负值 = 稳定化）。
    pub electrostatic_hartree: f64,
    /// 空化项（SMD 才有），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cavitation_hartree: Option<f64>,
    /// 色散项（SMD 才有），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispersion_hartree: Option<f64>,
    /// 排斥项（SMD 才有），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repulsion_hartree: Option<f64>,
    /// 可及表面积，单位 Å²。
    pub surface_area_angstrom2: f64,
    /// 可及体积，单位 Å³。
    pub surface_volume_angstrom3: f64,
    /// 表面 tesserae 数（收敛性与可复现性都依赖它）。
    pub n_tesserae: usize,
    /// 表面电荷迭代步数。
    pub iterations: usize,
    /// 是否收敛（独立判定，不允许"迭代用完就算收敛"）。
    pub converged: bool,
}

impl SolvationResult {
    /// 构造（非极性三项默认空，`converged` 默认 `false`）。
    #[must_use]
    pub fn new(
        model: SolventModel,
        solvent_name: impl Into<String>,
        total_hartree: f64,
        electrostatic_hartree: f64,
        cavity: CavityGeometry,
        iterations: usize,
    ) -> Self {
        Self {
            model,
            solvent_name: solvent_name.into(),
            total_hartree,
            electrostatic_hartree,
            cavitation_hartree: None,
            dispersion_hartree: None,
            repulsion_hartree: None,
            surface_area_angstrom2: cavity.surface_area_angstrom2,
            surface_volume_angstrom3: cavity.surface_volume_angstrom3,
            n_tesserae: cavity.n_tesserae,
            iterations,
            converged: false,
        }
    }

    /// 补上 SMD 的三项非极性贡献。
    #[must_use]
    pub fn with_nonpolar(
        mut self,
        cavitation_hartree: f64,
        dispersion_hartree: f64,
        repulsion_hartree: f64,
    ) -> Self {
        self.cavitation_hartree = Some(cavitation_hartree);
        self.dispersion_hartree = Some(dispersion_hartree);
        self.repulsion_hartree = Some(repulsion_hartree);
        self
    }

    /// 标记收敛。
    #[must_use]
    pub fn with_converged(mut self, converged: bool) -> Self {
        self.converged = converged;
        self
    }

    /// 已报出的非极性项之和（未报的按 0 计）。
    #[must_use]
    pub fn nonpolar_sum_hartree(&self) -> f64 {
        self.cavitation_hartree.unwrap_or(0.0)
            + self.dispersion_hartree.unwrap_or(0.0)
            + self.repulsion_hartree.unwrap_or(0.0)
    }

    /// 对账差额：`total − (electrostatic + nonpolar_sum)`。
    ///
    /// phase-2 §7（M5 断言）要求它 `< 1e-10`；非零说明能量分解与总能量不是同一套数。
    #[must_use]
    pub fn decomposition_gap_hartree(&self) -> f64 {
        self.total_hartree - (self.electrostatic_hartree + self.nonpolar_sum_hartree())
    }

    /// 结构自洽性检查（`result.json` 写出前的门禁）。
    ///
    /// # Errors
    ///
    /// 数值非有限、表面面积/体积非正、或模型与"有没有非极性项"矛盾时返回
    /// [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        for (name, value) in [
            ("total_hartree", self.total_hartree),
            ("electrostatic_hartree", self.electrostatic_hartree),
            ("surface_area_angstrom2", self.surface_area_angstrom2),
            ("surface_volume_angstrom3", self.surface_volume_angstrom3),
        ] {
            if !value.is_finite() {
                return Err(EngineError::Internal(format!(
                    "solvation.{name} is not finite"
                )));
            }
        }
        if !self.model.needs_cavity() {
            return Err(EngineError::Internal(
                "solvation result with model `vacuum` should not be emitted (K8)".into(),
            ));
        }
        if self.surface_area_angstrom2 <= 0.0 || self.surface_volume_angstrom3 <= 0.0 {
            return Err(EngineError::Internal(
                "solvation surface area/volume must be positive".into(),
            ));
        }
        if self.model == SolventModel::Smd && self.cavitation_hartree.is_none() {
            return Err(EngineError::Internal(
                "smd solvation requires the nonpolar (cavitation/dispersion/repulsion) terms"
                    .into(),
            ));
        }
        for (name, value) in [
            ("cavitation_hartree", self.cavitation_hartree),
            ("dispersion_hartree", self.dispersion_hartree),
            ("repulsion_hartree", self.repulsion_hartree),
        ] {
            if let Some(value) = value
                && !value.is_finite()
            {
                return Err(EngineError::Internal(format!(
                    "solvation.{name} is not finite"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CavityGeometry, SolvationResult, SolventModel};

    fn cosmo() -> SolvationResult {
        SolvationResult::new(
            SolventModel::Cosmo,
            "water",
            -76.4,
            -0.02,
            CavityGeometry::new(42.0, 18.0, 1024),
            7,
        )
        .with_converged(true)
    }

    #[test]
    fn model_spelling_is_frozen() {
        assert_eq!(SolventModel::Smd.as_str(), "smd");
        assert_eq!(
            serde_json::to_string(&SolventModel::Pbf).unwrap(),
            "\"pbf\""
        );
        assert!(!SolventModel::Vacuum.needs_cavity());
        assert!(SolventModel::Cosmo.needs_cavity());
    }

    #[test]
    fn decomposition_gap_uses_every_reported_term() {
        let block = cosmo();
        // total 与 electrostatic 之间差 0.02 → gap 不为 0（测试自己造的数不满足对账）
        assert!((block.decomposition_gap_hartree() + 0.0).abs() > 0.0);
        let consistent = block.clone();
        assert!(consistent.nonpolar_sum_hartree().abs() < 1e-15);

        let smd = SolvationResult::new(
            SolventModel::Smd,
            "water",
            -76.41,
            -0.02,
            CavityGeometry::new(42.0, 18.0, 1024),
            9,
        )
        .with_nonpolar(0.001, -0.002, 0.0005);
        let expected = -0.02 + 0.001 - 0.002 + 0.0005;
        assert!((smd.nonpolar_sum_hartree() - (0.001 - 0.002 + 0.0005)).abs() < 1e-15);
        assert!((smd.decomposition_gap_hartree() - (-76.41 - expected)).abs() < 1e-12);
    }

    #[test]
    fn validate_rejects_vacuum_and_unaccounted_smd() {
        assert!(cosmo().validate().is_ok());

        let vacuum = SolvationResult::new(
            SolventModel::Vacuum,
            "vacuum",
            0.0,
            0.0,
            CavityGeometry::new(1.0, 1.0, 1),
            1,
        );
        assert!(vacuum.validate().is_err(), "vacuum 不该出溶剂化块");

        let smd_no_nonpolar = SolvationResult::new(
            SolventModel::Smd,
            "water",
            -76.4,
            -0.02,
            CavityGeometry::new(42.0, 18.0, 1024),
            9,
        );
        assert!(smd_no_nonpolar.validate().is_err());

        let mut bad = cosmo();
        bad.surface_area_angstrom2 = 0.0;
        assert!(bad.validate().is_err());

        let mut bad = cosmo();
        bad.total_hartree = f64::NAN;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn block_roundtrips_through_json() {
        let block = cosmo();
        let json = serde_json::to_string(&block).expect("serialize");
        assert!(!json.contains("cavitation"), "{json}");
        let back: SolvationResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);
    }
}
