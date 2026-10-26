//! K7 · 响应与性质的结果块（phase-2 §4.2）。
//!
//! 这些类型**只承载结果**，数值由 `qwb-response` / `qwb-props` 产生。它们挂在
//! [`crate::JobResult`] 的**可选块**上：旧结果目录没有这些键仍然解析（`serde(default)`），
//! 新结果写出时缺省不出现（`skip_serializing_if`）。
//!
//! 三道不变量的语义写在这里（判据见 phase-2 §4.2 / §9.2 G9）：
//!
//! - [`ExcitedStates::sum_rules`]：Thomas–Reiche–Kuhn（f-sum）自检；
//! - [`NmrShieldings::length_gauge_spread_ppm`]：规范一致性（越小越可信）；
//! - [`Polarizability::isotropic_au`] / [`Polarizability::anisotropy_au`]：转动不变量
//!   ——分子整体旋转后这两个数不变，是"张量算对了"的必要条件。
//!
//! 单位后缀（§12.2）：能量 `_hartree`、波长 `_nm`、屏蔽 `_ppm`、极化率 `_au`（原子单位）、
//! 偶极 `_debye`、四极 `_debye_angstrom`、八极 `_debye_angstrom2`、规范原点 `_bohr`。

use serde::{Deserialize, Serialize};

/// 激发态方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ExcitedStateMethod {
    /// CIS（HF 参考的组态相互作用单激发）。
    Cis,
    /// TDA（Tamm–Dancoff 近似的 TDDFT；phase-2 #27 定的默认）。
    Tda,
    /// 完整 TDDFT（含耦合项；phase-2 `ST-01`）。
    Tddft,
}

impl ExcitedStateMethod {
    /// 冻结的小写名（进 `result.json`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cis => "cis",
            Self::Tda => "tda",
            Self::Tddft => "tddft",
        }
    }
}

/// 一个跃迁的主要轨道贡献（`from`/`to` 是 MO 序号，`weight` 是系数平方）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TransitionContribution {
    /// 出发轨道序号（0 基，按能级升序）。
    pub from: usize,
    /// 到达轨道序号（0 基）。
    pub to: usize,
    /// 该行列在跃迁中的权重（系数平方，0..=1）。
    pub weight: f64,
}

impl TransitionContribution {
    /// 构造一个贡献项。
    #[must_use]
    pub fn new(from: usize, to: usize, weight: f64) -> Self {
        Self { from, to, weight }
    }
}

/// 一个电子跃迁。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Transition {
    /// 激发态序号（1 基，按能量升序）。
    pub index: usize,
    /// 激发能，单位 Hartree。
    pub energy_hartree: f64,
    /// 对应的波长，单位 nm（由能量换算，便于显示）。
    pub wavelength_nm: f64,
    /// 振子强度（无量纲，长度规范）。
    pub oscillator_strength: f64,
    /// 旋转强度（VCD/ECD 用；`None` = 未计算）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotatory_strength: Option<f64>,
    /// 主要轨道贡献（按权重降序，通常取前 3 个）。
    #[serde(default)]
    pub dominant: Vec<TransitionContribution>,
}

impl Transition {
    /// 构造一个跃迁（`rotatory_strength`/`dominant` 默认空）。
    #[must_use]
    pub fn new(
        index: usize,
        energy_hartree: f64,
        wavelength_nm: f64,
        oscillator_strength: f64,
    ) -> Self {
        Self {
            index,
            energy_hartree,
            wavelength_nm,
            oscillator_strength,
            rotatory_strength: None,
            dominant: Vec::new(),
        }
    }

    /// 补上主要轨道贡献。
    #[must_use]
    pub fn with_dominant(mut self, dominant: Vec<TransitionContribution>) -> Self {
        self.dominant = dominant;
        self
    }
}

/// f-sum 自检（Thomas–Reiche–Kuhn）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SumRules {
    /// 已算出的各态振子强度之和。
    pub trk_oscillator: f64,
    /// 体系电子数（TRK 的完整求和极限；有限基/有限态数下只会更小）。
    pub n_electrons: f64,
}

impl SumRules {
    /// 构造。
    #[must_use]
    pub fn new(trk_oscillator: f64, n_electrons: f64) -> Self {
        Self {
            trk_oscillator,
            n_electrons,
        }
    }

    /// TRK 差额（`n_electrons - trk_oscillator`）。完备基 + 全部态时趋于 0；
    /// 截断态数时为正。phase-2 的门禁是"它必须被报出来"，不是"它必须是 0"。
    #[must_use]
    pub fn deficit(&self) -> f64 {
        self.n_electrons - self.trk_oscillator
    }
}

/// 激发态结果块（K7）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ExcitedStates {
    /// 用的方法（`cis` / `tda` / `tddft`）。
    pub method: ExcitedStateMethod,
    /// 请求的态数。
    pub n_states: usize,
    /// 跃迁列表（按能量升序）。
    pub transitions: Vec<Transition>,
    /// f-sum 自检。
    pub sum_rules: SumRules,
}

impl ExcitedStates {
    /// 构造。
    #[must_use]
    pub fn new(
        method: ExcitedStateMethod,
        n_states: usize,
        transitions: Vec<Transition>,
        sum_rules: SumRules,
    ) -> Self {
        Self {
            method,
            n_states,
            transitions,
            sum_rules,
        }
    }

    /// 能量最低的跃迁（列表为空时 `None`）。
    #[must_use]
    pub fn lowest(&self) -> Option<&Transition> {
        self.transitions.first()
    }
}

/// 一个原子核的屏蔽张量（GIAO）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NucleusShielding {
    /// 元素符号。
    pub element: String,
    /// 该元素在分子中的序号（0 基，按 `Molecule.atoms` 顺序）。
    pub index: usize,
    /// 各向同性屏蔽，单位 ppm。
    pub isotropic_ppm: f64,
    /// 各向异性（`Δσ = σ_zz − (σ_xx+σ_yy)/2`），单位 ppm。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anisotropy_ppm: Option<f64>,
    /// 完整的 3×3 屏蔽张量，单位 ppm（行 = 第一个指标）。
    pub tensor_ppm: [[f64; 3]; 3],
}

impl NucleusShielding {
    /// 构造。
    #[must_use]
    pub fn new(
        element: impl Into<String>,
        index: usize,
        isotropic_ppm: f64,
        tensor_ppm: [[f64; 3]; 3],
    ) -> Self {
        Self {
            element: element.into(),
            index,
            isotropic_ppm,
            anisotropy_ppm: None,
            tensor_ppm,
        }
    }

    /// 补上各向异性。
    #[must_use]
    pub fn with_anisotropy(mut self, anisotropy_ppm: f64) -> Self {
        self.anisotropy_ppm = Some(anisotropy_ppm);
        self
    }
}

/// 化学位移的参考物（如 TMS 的 ¹H/¹³C 屏蔽）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NmrReference {
    /// 参考物名称（如 `TMS`）。
    pub compound: String,
    /// 该参考物在同一方法/基组下的屏蔽，单位 ppm。
    pub shift_ppm: f64,
}

impl NmrReference {
    /// 构造。
    #[must_use]
    pub fn new(compound: impl Into<String>, shift_ppm: f64) -> Self {
        Self {
            compound: compound.into(),
            shift_ppm,
        }
    }
}

/// NMR 屏蔽结果块（K7）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NmrShieldings {
    /// 规范/方法名；phase-2 只支持 `giao`（§13 #30）。
    pub method: String,
    /// 规范原点，单位 Bohr（GIAO 下结果对原点不该敏感——这正是
    /// [`Self::length_gauge_spread_ppm`] 检查的东西）。
    pub gauge_origin_bohr: [f64; 3],
    /// 每个核的屏蔽。
    pub nuclei: Vec<NucleusShielding>,
    /// 化学位移参考物（`None` = 只报屏蔽，不换算位移）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<NmrReference>,
    /// 换不同规范原点重算时各向同性屏蔽的最大变化，单位 ppm。
    ///
    /// phase-2 §4.2 的门禁是 `< 1 ppm`；它必须被报出来，否则"GIAO 算对了"无法验证。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_gauge_spread_ppm: Option<f64>,
}

impl NmrShieldings {
    /// 构造 GIAO 结果块。
    #[must_use]
    pub fn new(gauge_origin_bohr: [f64; 3], nuclei: Vec<NucleusShielding>) -> Self {
        Self {
            method: "giao".to_string(),
            gauge_origin_bohr,
            nuclei,
            reference: None,
            length_gauge_spread_ppm: None,
        }
    }

    /// 补上化学位移参考物。
    #[must_use]
    pub fn with_reference(mut self, reference: NmrReference) -> Self {
        self.reference = Some(reference);
        self
    }

    /// 补上规范一致性数字。
    #[must_use]
    pub fn with_gauge_spread(mut self, spread_ppm: f64) -> Self {
        self.length_gauge_spread_ppm = Some(spread_ppm);
        self
    }

    /// 化学位移（`δ = σ_ref − σ`，ppm）；没有参考物时 `None`。
    #[must_use]
    pub fn chemical_shift_ppm(&self, isotropic_ppm: f64) -> Option<f64> {
        self.reference
            .as_ref()
            .map(|reference| reference.shift_ppm - isotropic_ppm)
    }
}

/// 极化率（静态 + 可选动态），单位原子单位（a.u. = Bohr³）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Polarizability {
    /// 静态极化率张量，a.u.（行 = 第一个指标）。
    pub static_au: [[f64; 3]; 3],
    /// 动态部分的频率，单位 Hartree（空 = 只算了静态）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frequencies_hartree: Vec<f64>,
    /// 与 [`Self::frequencies_hartree`] 一一对应的张量。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dynamic_au: Vec<[[f64; 3]; 3]>,
    /// 各向同性分量 `α_iso = (α_xx+α_yy+α_zz)/3`，a.u.
    pub isotropic_au: f64,
    /// 各向异性 `Δα`（转动不变量），a.u.
    pub anisotropy_au: f64,
}

impl Polarizability {
    /// 从静态张量构造，并算出两个转动不变量。
    ///
    /// # Errors
    ///
    /// 张量含非有限值时返回 [`crate::EngineError::Internal`]。
    pub fn from_static_tensor(static_au: [[f64; 3]; 3]) -> Result<Self, crate::EngineError> {
        if static_au.iter().flatten().any(|v| !v.is_finite()) {
            return Err(crate::EngineError::Internal(
                "polarizability tensor contains a non-finite value".into(),
            ));
        }
        let trace = static_au[0][0] + static_au[1][1] + static_au[2][2];
        let isotropic = trace / 3.0;
        // Δα² = ½[ (α_xx−α_yy)² + (α_yy−α_zz)² + (α_zz−α_xx)² + 6(α_xy²+α_xz²+α_yz²) ]
        let mut sum = 0.0;
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            sum += (static_au[i][i] - static_au[j][j]).powi(2);
        }
        sum += 6.0 * (static_au[0][1].powi(2) + static_au[0][2].powi(2) + static_au[1][2].powi(2));
        Ok(Self {
            static_au,
            frequencies_hartree: Vec::new(),
            dynamic_au: Vec::new(),
            isotropic_au: isotropic,
            anisotropy_au: (0.5 * sum).sqrt(),
        })
    }

    /// 补上动态极化率（频率与张量必须等长）。
    ///
    /// # Errors
    ///
    /// 两个列表长度不等时返回 [`crate::EngineError::Internal`]。
    pub fn with_dynamic(
        mut self,
        frequencies_hartree: Vec<f64>,
        dynamic_au: Vec<[[f64; 3]; 3]>,
    ) -> Result<Self, crate::EngineError> {
        if frequencies_hartree.len() != dynamic_au.len() {
            return Err(crate::EngineError::Internal(format!(
                "polarizability: {} frequencies but {} tensors",
                frequencies_hartree.len(),
                dynamic_au.len()
            )));
        }
        self.frequencies_hartree = frequencies_hartree;
        self.dynamic_au = dynamic_au;
        Ok(self)
    }
}

/// 多极展开（K7）。偶极沿用 [`crate::JobResult::dipole_debye`]，这里放更高阶。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Multipoles {
    /// traceless 四极矩 `Θ_ij`，单位 Debye·Å（对称、迹为零）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quadrupole_traceless_debye_angstrom: Option<[[f64; 3]; 3]>,
    /// 八极矩（3×3×3，全对称 + 缩并条件），单位 Debye·Å²。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub octupole_traceless_debye_angstrom2: Option<[[[f64; 3]; 3]; 3]>,
    /// 展开原点，单位 Å（多极矩对原点是敏感的，必须记下来）。
    pub origin_angstrom: [f64; 3],
}

impl Multipoles {
    /// 构造（原点必需，高阶矩可后补）。
    #[must_use]
    pub fn new(origin_angstrom: [f64; 3]) -> Self {
        Self {
            quadrupole_traceless_debye_angstrom: None,
            octupole_traceless_debye_angstrom2: None,
            origin_angstrom,
        }
    }

    /// 补上 traceless 四极矩。
    #[must_use]
    pub fn with_quadrupole(mut self, quadrupole: [[f64; 3]; 3]) -> Self {
        self.quadrupole_traceless_debye_angstrom = Some(quadrupole);
        self
    }

    /// 四极矩的迹（traceless 约定下应 ≈ 0；作为实现自检用）。
    #[must_use]
    pub fn quadrupole_trace(&self) -> Option<f64> {
        self.quadrupole_traceless_debye_angstrom
            .map(|q| q[0][0] + q[1][1] + q[2][2])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExcitedStateMethod, ExcitedStates, Multipoles, NmrReference, NmrShieldings,
        NucleusShielding, Polarizability, SumRules, Transition, TransitionContribution,
    };

    fn transitions() -> Vec<Transition> {
        vec![
            Transition::new(1, 0.3, 412.0, 0.05)
                .with_dominant(vec![TransitionContribution::new(2, 3, 0.92)]),
            Transition::new(2, 0.4, 310.0, 0.31),
        ]
    }

    #[test]
    fn excited_states_roundtrip_and_sum_rules() {
        let block = ExcitedStates::new(
            ExcitedStateMethod::Tda,
            transitions().len(),
            transitions(),
            SumRules::new(7.4, 10.0),
        );
        let json = serde_json::to_string(&block).expect("serialize");
        assert!(json.contains("\"method\":\"tda\""), "{json}");
        let back: ExcitedStates = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);
        assert_eq!(back.lowest().map(|t| t.index), Some(1));
        assert!((back.sum_rules.deficit() - 2.6).abs() < 1e-12);
    }

    #[test]
    fn nmr_block_reports_shifts_and_gauge_spread() {
        let block = NmrShieldings::new(
            [0.0, 0.0, 0.0],
            vec![
                NucleusShielding::new(
                    "H",
                    0,
                    31.0,
                    [[31.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 32.0]],
                )
                .with_anisotropy(1.5),
                NucleusShielding::new(
                    "O",
                    1,
                    300.0,
                    [[300.0, 0.0, 0.0], [0.0, 299.0, 0.0], [0.0, 0.0, 301.0]],
                ),
            ],
        )
        .with_reference(NmrReference::new("TMS", 31.9))
        .with_gauge_spread(0.02);
        assert_eq!(block.method, "giao");
        assert!((block.chemical_shift_ppm(30.0).expect("shift") - 1.9).abs() < 1e-12);
        assert!(block.length_gauge_spread_ppm.expect("spread") < 1.0);
        let json = serde_json::to_string(&block).expect("serialize");
        let back: NmrShieldings = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);
    }

    #[test]
    fn polarizability_invariants_are_computed_and_rotation_stable() {
        let diagonal = Polarizability::from_static_tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 20.0],
        ])
        .expect("isotropic");
        assert!((diagonal.isotropic_au - 40.0 / 3.0).abs() < 1e-12);
        assert!((diagonal.anisotropy_au - 10.0).abs() < 1e-12);

        // 把张量按 z 轴转 90°（x↔y）：不变量必须不变。
        let rotated = Polarizability::from_static_tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 20.0],
        ])
        .expect("rotated");
        assert!((rotated.isotropic_au - diagonal.isotropic_au).abs() < 1e-12);
        assert!((rotated.anisotropy_au - diagonal.anisotropy_au).abs() < 1e-12);

        assert!(
            Polarizability::from_static_tensor([[f64::NAN, 0.0, 0.0], [0.0; 3], [0.0; 3]]).is_err()
        );

        let with_dyn = diagonal
            .clone()
            .with_dynamic(vec![0.1], vec![[[1.0; 3]; 3]])
            .expect("dynamic");
        assert_eq!(with_dyn.dynamic_au.len(), 1);
        assert!(
            diagonal
                .with_dynamic(vec![0.1, 0.2], vec![[[1.0; 3]; 3]])
                .is_err()
        );
    }

    #[test]
    fn multipoles_carry_their_origin() {
        let block = Multipoles::new([0.0, 0.0, 0.0]).with_quadrupole([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        assert!((block.quadrupole_trace().expect("trace") - 0.0).abs() < 1e-12);
        let json = serde_json::to_string(&block).expect("serialize");
        let back: Multipoles = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, block);
        assert!(block.octupole_traceless_debye_angstrom2.is_none());
    }

    #[test]
    fn omitted_optionals_stay_out_of_json() {
        let block = Multipoles::new([0.0, 0.0, 0.0]);
        let json = serde_json::to_string(&block).expect("serialize");
        assert!(!json.contains("quadrupole"), "{json}");
        assert!(!json.contains("octupole"), "{json}");
    }
}
