//! K8 · 溶剂化的输入选项与结果块（phase-2 §4.3）。
//!
//! 三个约束来自 phase-2 的判决：
//!
//! - **模型是显式选择**（§8.2）：[`SolventModel`] 没有"默认猜一个"的分支 —— 输入不给就用
//!   数据里 `cavity_model` 的默认，数据也不给就是 `vacuum`。
//! - **能量分解必须能逐项对账**：SMD 的非极性部分可以逐项报，但
//!   `total = Σ 各项` 必须能被 [`SolvationResult::decomposition_gap_hartree`] 查出来。
//! - **`converged` 是独立判定**（`S-06` 风险）：表面电荷迭代不收敛时不得悄悄返回能量。
//!
//! # 一处契约订正（第 387 轮，`choices` 一百八十二）
//!
//! 这个块最初照着"静电 / 空化 / 色散 / 排斥 四项"设计，把 SMD 的非极性部分当成三个独立的
//! 能量项。**SMD 不是那样**：Marenich–Cramer–Truhlar 2009 的非极性部分是**一项**
//! `G_CDS`（cavity-dispersion-solvent structure），由 eq. 4.77 给出，可分成两半 ——
//! 每原子表面张力那一半 `Σ_j σ_j A_j` 与分子表面张力那一半 `σ^[M] Σ_j A_j`。
//! 那"四项"其实是 eq. 4.80 里**分子表面张力**的四个溶剂描述符系数（γ、φ²、ψ²、β²），
//! 不是四个能量项。所以这里**加性地**补上 [`SolvationResult::cds_hartree`]（以及它的两个
//! 分量），并把 `cavitation/dispersion/repulsion` 保留为**未使用**的字段（不删：K2 是加性
//! 演进的契约，删字段会让旧读者炸；但 SMD 不再用它们，`validate` 也不再要求它们）。

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
    /// SMD：C-PCM/COSMO 静电 + **一项**非极性 `G_CDS`（eq. 4.77）+ 参数化半径（eq. 16）。
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

/// 气体常数（J/(mol·K)，CODATA 2018）。
const GAS_CONSTANT_J_PER_MOL_K: f64 = 8.314_462_618;
/// 1 Hartree 的焦耳数（每摩尔），与 `vibrations.rs` 用的是同一个数。
const HARTREE_J_PER_MOL: f64 = 2_625_499.639;
/// 标准态换算用的温度（K）与压强（Pa）：SMD 的 298.15 K / 1 atm。
const STANDARD_TEMPERATURE_K: f64 = 298.15;
const STANDARD_PRESSURE_PA: f64 = 101_325.0;

/// `RT·ln(RT/p°)`：1 atm 理想气体 → 1 mol/dm³ 的吉布斯能差（Hartree）。
///
/// 在 298.15 K / 101 325 Pa 上 `RT/p° = 24.465 L/mol`，所以
/// `RT ln(24.465) = 7.926 kJ/mol = +1.8943 kcal/mol = +3.0188e-3 Ha`。
/// 写成函数而不是字面量：温度/压强的依赖关系要看得见，将来支持变温时只改这一个地方。
#[must_use]
pub fn standard_state_1atm_hartree() -> f64 {
    let rt = GAS_CONSTANT_J_PER_MOL_K * STANDARD_TEMPERATURE_K;
    // RT/p° 的 SI 单位是 m³/mol，换成 L/mol（×1e3）才能得到文献里的 24.465。
    let molar_volume_litres = rt / STANDARD_PRESSURE_PA * 1.0e3;
    rt * molar_volume_litres.ln() / HARTREE_J_PER_MOL
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
    /// 空化项（SMD **不用**这个字段，见模块头的契约订正），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cavitation_hartree: Option<f64>,
    /// 色散项（SMD **不用**这个字段），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispersion_hartree: Option<f64>,
    /// 排斥项（SMD **不用**这个字段），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repulsion_hartree: Option<f64>,
    /// SMD 的非极性项 `G_CDS`（cavity-dispersion-solvent structure，eq. 4.77），单位 Hartree。
    ///
    /// 它**不是**上面三项之和：CDS 是一个整体表达式，下面的两半是它的可核对分解。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cds_hartree: Option<f64>,
    /// `G_CDS` 的每原子表面张力那一半 `Σ_j σ_j A_j`（eq. 4.78），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cds_atomic_hartree: Option<f64>,
    /// `G_CDS` 的分子表面张力那一半 `σ^[M] Σ_j A_j`（eq. 4.80；水相为 0），单位 Hartree。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cds_molecular_hartree: Option<f64>,
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
            cds_hartree: None,
            cds_atomic_hartree: None,
            cds_molecular_hartree: None,
            surface_area_angstrom2: cavity.surface_area_angstrom2,
            surface_volume_angstrom3: cavity.surface_volume_angstrom3,
            n_tesserae: cavity.n_tesserae,
            iterations,
            converged: false,
        }
    }

    /// 补上 SMD 的三项非极性贡献。
    ///
    /// **SMD 不用这条路**（契约订正后它只有一项 `G_CDS`）；保留给将来真正把非极性拆成
    /// 空化/色散/排斥的模型（例如 PBF 的扩展）。SMD 请用 [`Self::with_cds`]。
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

    /// `total_hartree` 的口径说明，以及到 **1 atm 气体标准态**的换算项。
    ///
    /// `total_hartree` 是 **1 M ↔ 1 M** 的转移自由能（`G_EP + G_CDS`）：气体与溶液都用
    /// 1 mol/dm³ 的标准态，分子内的热化学项在两边**互相抵消**，所以这里没有振动/热修正。
    ///
    /// 文献与数据库里的溶剂化自由能常常写成 `ΔG*_S`（1 atm 气体 → 1 M 溶液），两者差一项
    /// `RT·ln(RT/p°)`（Marenich–Cramer–Truhlar 2009；Imperial 学位论文 eq. 4.64–4.65）：
    /// 在 298.15 K / 101 325 Pa 上 `RT/p° = 24.465 L/mol`，于是该项是
    /// **+1.8943 kcal/mol**。比较之前先确认对方的约定，不要拿两个不同标准态的数直接相减。
    #[must_use]
    pub fn standard_state_hartree(&self) -> f64 {
        standard_state_1atm_hartree()
    }

    /// 1 atm 气体标准态下的溶剂化自由能（`total_hartree + standard_state_hartree`）。
    #[must_use]
    pub fn delta_g_solv_1atm_hartree(&self) -> f64 {
        self.total_hartree + standard_state_1atm_hartree()
    }

    /// 补上 SMD 的非极性项 `G_CDS` 与它的两半（eq. 4.77）。
    ///
    /// `cds_hartree` 由调用方给出，而不是这里把两半加起来：这样
    /// [`Self::cds_gap_hartree`] 才真的在核对"报出来的总数 = 报出来的两半之和"。
    #[must_use]
    pub fn with_cds(
        mut self,
        cds_hartree: f64,
        cds_atomic_hartree: f64,
        cds_molecular_hartree: f64,
    ) -> Self {
        self.cds_hartree = Some(cds_hartree);
        self.cds_atomic_hartree = Some(cds_atomic_hartree);
        self.cds_molecular_hartree = Some(cds_molecular_hartree);
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
            + self.cds_hartree.unwrap_or(0.0)
    }

    /// `G_CDS` 的自洽差额：`cds − (cds_atomic + cds_molecular)`。
    ///
    /// eq. 4.77 的两半必须加起来就是报出来的 CDS；差得开说明分解与总数不是同一套数。
    #[must_use]
    pub fn cds_gap_hartree(&self) -> f64 {
        match (
            self.cds_hartree,
            self.cds_atomic_hartree,
            self.cds_molecular_hartree,
        ) {
            (Some(cds), Some(atomic), Some(molecular)) => cds - (atomic + molecular),
            _ => 0.0,
        }
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
        if self.model == SolventModel::Smd
            && self.cds_hartree.is_none()
            && self.cavitation_hartree.is_none()
        {
            return Err(EngineError::Internal(
                "smd solvation requires its nonpolar term `cds_hartree` (eq. 4.77); the \
                 cavitation/dispersion/repulsion fields are not used by SMD"
                    .into(),
            ));
        }
        for (name, value) in [
            ("cavitation_hartree", self.cavitation_hartree),
            ("dispersion_hartree", self.dispersion_hartree),
            ("repulsion_hartree", self.repulsion_hartree),
            ("cds_hartree", self.cds_hartree),
            ("cds_atomic_hartree", self.cds_atomic_hartree),
            ("cds_molecular_hartree", self.cds_molecular_hartree),
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
    fn the_standard_state_term_is_the_documented_1_89_kcal() {
        // RT ln(RT/p°) = 24.465 · ln / Hartree：文献记 +1.89 kcal/mol（eq. 4.64–4.65）。
        let hartree = crate::solvation::standard_state_1atm_hartree();
        let kcal = hartree * 627.509_474_063_6;
        assert!(
            (kcal - 1.8943).abs() < 1.0e-3,
            "标准态项 {kcal:.4} kcal/mol 与文献 1.89 不符"
        );
        let block = SolvationResult::new(
            SolventModel::Cosmo,
            "water",
            -0.02,
            -0.02,
            CavityGeometry::new(42.0, 18.0, 1024),
            7,
        );
        assert!((block.standard_state_hartree() - hartree).abs() < 1e-15);
        assert!((block.delta_g_solv_1atm_hartree() - (-0.02 + hartree)).abs() < 1e-15);
    }

    #[test]
    fn cds_parts_must_add_up_to_the_reported_cds() {
        let block = SolvationResult::new(
            SolventModel::Smd,
            "water",
            0.0,
            0.0,
            CavityGeometry::new(42.0, 18.0, 1024),
            9,
        )
        .with_cds(0.0012, 0.0010, 0.0002);
        assert!(block.validate().is_ok());
        assert!(block.cds_gap_hartree().abs() < 1e-15);
        assert!((block.nonpolar_sum_hartree() - 0.0012).abs() < 1e-15);

        let mismatched = SolvationResult::new(
            SolventModel::Smd,
            "water",
            0.0,
            0.0,
            CavityGeometry::new(42.0, 18.0, 1024),
            9,
        )
        .with_cds(0.0012, 0.0010, 0.0001);
        // 契约不因此报错（总数才是判据），但差额必须能查出来 —— 这正是它的用途。
        assert!((mismatched.cds_gap_hartree() - 0.0001).abs() < 1e-15);

        // 没有 CDS 也没有旧的三项 → 拒绝。
        let bare = SolvationResult::new(
            SolventModel::Smd,
            "water",
            0.0,
            0.0,
            CavityGeometry::new(42.0, 18.0, 1024),
            9,
        );
        assert!(bare.validate().is_err());
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
