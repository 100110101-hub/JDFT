//! 作业输入（K2）。

use serde::{Deserialize, Serialize};

use crate::correlation::{Mp2Options, ScfType};
use crate::error::EngineError;
use crate::molecule::Molecule;
use crate::properties::ExcitedStateMethod;
use crate::solvation::SolventModel;

/// XC 网格等级（数据侧见 `grid.schema.yaml`）。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum GridLevel {
    /// `coarse`。
    Coarse,
    /// `normal`（默认）。
    #[default]
    Normal,
    /// `fine`。
    Fine,
    /// `veryfine`。
    VeryFine,
}

impl GridLevel {
    /// 冻结的小写名（与 `grid.schema.yaml` 的 `level` 取值一致）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Coarse => "coarse",
            Self::Normal => "normal",
            Self::Fine => "fine",
            Self::VeryFine => "veryfine",
        }
    }
}

/// 作业类型（K3 状态机之外的第二个维度：算什么）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum JobType {
    /// 单点能。
    #[serde(alias = "Sp", alias = "SP")]
    Sp,
    /// 几何优化。
    #[serde(alias = "Opt", alias = "OPT")]
    Opt,
    /// 频率。
    #[serde(alias = "Freq", alias = "FREQ")]
    Freq,
}

impl JobType {
    /// 冻结的小写名（K4 `input.yaml` 的 `jobtype` 取值）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sp => "sp",
            Self::Opt => "opt",
            Self::Freq => "freq",
        }
    }
}

/// 数值控制参数。全部有 `Default`，下游用 `Default::default()` 后按字段覆盖。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TaskOptions {
    /// SCF 最大迭代步数。
    pub scf_max_iter: u32,
    /// 能量收敛阈值，单位 Hartree（判据 `|dE| < 阈值`）。
    pub scf_energy_threshold_hartree: f64,
    /// 密度矩阵收敛阈值（判据 `max|dP_ij| < 阈值`）。
    pub scf_density_threshold: f64,
    /// DIIS 历史长度。
    pub diis_history: u32,
    /// 线程数；`0` 表示自动（`rayon` 默认）。
    pub threads: u32,
    /// 几何优化最大步数。
    pub max_opt_steps: u32,
    /// 优化收敛阈值（最大力分量），单位 Hartree/Bohr（T3 门禁值）。
    pub opt_gradient_threshold_hartree_per_bohr: f64,
    /// 数值 Hessian 的位移步长，单位 Å。
    pub freq_step_angstrom: f64,
    /// 与基组配对的 ECP 名称；`None` 表示全电子。
    pub ecp_name: Option<String>,
    // ── 阶段二（K8/K9）的**可选输入**：缺省时行为与阶段一逐位相同。──
    /// SCF 类型（自旋处理）；`None` = 由引擎按 `method_name` + 多重度决定（阶段一行为）。
    ///
    /// 与 `method_name` 的自洽性由 [`JobInput::validate`] 强制：HF 变体要求
    /// `method_name == "hf"`，KS 变体要求不是 `"hf"`（phase-2 §4.4）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scf_type: Option<ScfType>,
    /// 相关方法（MP2 / RI-MP2）；`None` = 只做 SCF。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mp2: Option<Mp2Options>,
    /// 激发态（CIS / TDA / TDDFT）；`None` = 不算激发态。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excited_states: Option<ExcitedStatesOptions>,
    /// 溶剂模型；`None` = 用 `qwb-data/solvent/` 里该溶剂的 `cavity_model` 默认
    /// （`solvent: vacuum` 时永远是气相，不接受覆盖）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solvent_model: Option<SolventModel>,
}

/// 激发态选项（`TaskOptions.excited_states`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ExcitedStatesOptions {
    /// 请求的态数。
    pub n_states: u32,
    /// 方法（`cis` / `tda` / `tddft`）。
    pub method: ExcitedStateMethod,
}

impl Default for ExcitedStatesOptions {
    fn default() -> Self {
        Self {
            n_states: 10,
            method: ExcitedStateMethod::Tda,
        }
    }
}

impl ExcitedStatesOptions {
    /// 构造。
    #[must_use]
    pub fn new(n_states: u32, method: ExcitedStateMethod) -> Self {
        Self { n_states, method }
    }
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            scf_max_iter: 128,
            scf_energy_threshold_hartree: 1.0e-8,
            scf_density_threshold: 1.0e-6,
            diis_history: 8,
            threads: 0,
            max_opt_steps: 100,
            opt_gradient_threshold_hartree_per_bohr: 4.5e-4,
            freq_step_angstrom: 0.01,
            ecp_name: None,
            scf_type: None,
            mp2: None,
            excited_states: None,
            solvent_model: None,
        }
    }
}

impl TaskOptions {
    /// 用 SCF 类型覆盖自旋处理（阶段二 K9；`None` = 引擎按方法/多重度决定）。
    ///
    /// `non_exhaustive` 结构体不允许下游写 struct literal，所以新选项一律用 builder 挂上。
    #[must_use]
    pub fn with_scf_type(mut self, scf_type: ScfType) -> Self {
        self.scf_type = Some(scf_type);
        self
    }

    /// 挂上相关方法（阶段二 K9）。
    #[must_use]
    pub fn with_mp2(mut self, mp2: Mp2Options) -> Self {
        self.mp2 = Some(mp2);
        self
    }

    /// 挂上激发态选项（阶段二 K7）。
    #[must_use]
    pub fn with_excited_states(mut self, excited_states: ExcitedStatesOptions) -> Self {
        self.excited_states = Some(excited_states);
        self
    }

    /// 覆盖溶剂模型（阶段二 K8；`solvent: vacuum` 时只能是 `vacuum`）。
    #[must_use]
    pub fn with_solvent_model(mut self, solvent_model: SolventModel) -> Self {
        self.solvent_model = Some(solvent_model);
        self
    }

    /// 校验取值区间。
    ///
    /// # Errors
    ///
    /// 任一阈值非正、迭代上限为 0、位移步长非正时返回 [`EngineError::InvalidInput`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        let positive = |name: &str, v: f64| -> Result<(), EngineError> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(EngineError::InvalidInput(format!(
                    "options.{name} must be a positive finite number, got {v}"
                )))
            }
        };
        if self.scf_max_iter == 0 {
            return Err(EngineError::InvalidInput(
                "options.scf_max_iter must be >= 1".into(),
            ));
        }
        if self.diis_history == 0 {
            return Err(EngineError::InvalidInput(
                "options.diis_history must be >= 1".into(),
            ));
        }
        if self.max_opt_steps == 0 {
            return Err(EngineError::InvalidInput(
                "options.max_opt_steps must be >= 1".into(),
            ));
        }
        positive(
            "scf_energy_threshold_hartree",
            self.scf_energy_threshold_hartree,
        )?;
        positive("scf_density_threshold", self.scf_density_threshold)?;
        positive(
            "opt_gradient_threshold_hartree_per_bohr",
            self.opt_gradient_threshold_hartree_per_bohr,
        )?;
        positive("freq_step_angstrom", self.freq_step_angstrom)?;
        // ── 阶段二新增选项的区间检查（缺省时全部跳过，旧输入行为不变）──
        if let Some(mp2) = &self.mp2 {
            mp2.validate()?;
        }
        if let Some(excited) = &self.excited_states
            && excited.n_states == 0
        {
            return Err(EngineError::InvalidInput(
                "options.excited_states.n_states must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// 一次作业的完整输入（K2 冻结；同时是 `input.yaml` 的形态，见 K4）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct JobInput {
    /// 分子。
    pub molecule: Molecule,
    /// 基组键名（`qwb-data/basis/` 下的 name）。
    pub basis_name: String,
    /// 泛函键名（`qwb-data/functional/` 下的 name；HF 用 `"hf"`）。
    pub method_name: String,
    /// XC 网格等级。
    pub grid_level: GridLevel,
    /// 溶剂键名；`"vacuum"` 表示气相。
    pub solvent: String,
    /// 作业类型。
    pub jobtype: JobType,
    /// 数值控制。
    pub options: TaskOptions,
}

impl JobInput {
    /// 用必需字段构造输入；其余维度可用字段赋值覆盖（结构体是 `non_exhaustive`）。
    #[must_use]
    pub fn new(
        molecule: Molecule,
        basis_name: impl Into<String>,
        method_name: impl Into<String>,
        grid_level: GridLevel,
        solvent: impl Into<String>,
        jobtype: JobType,
        options: TaskOptions,
    ) -> Self {
        Self {
            molecule,
            basis_name: basis_name.into(),
            method_name: method_name.into(),
            grid_level,
            solvent: solvent.into(),
            jobtype,
            options,
        }
    }

    /// 校验：分子自洽、键名非空、选项区间合法。
    ///
    /// 阶段二（phase-2 §4.4）再加三条**跨字段**检查（只在显式给新选项时生效）：
    /// ① `scf_type` 与 `method_name` 必须同类（HF 变体 ↔ `hf`）；
    /// ② 开壳层 `scf_type` 要求 `multiplicity > 1`，闭壳层要求 `multiplicity == 1`；
    /// ③ MP2 只支持闭壳层参考（`rhf`/`rks`），开壳层 MP2 还没实现。
    ///
    /// # Errors
    ///
    /// 分子非法、基组/泛函/溶剂键为空、选项越界、或上述跨字段关系不成立时返回
    /// [`EngineError::InvalidInput`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        self.molecule.validate()?;
        for (label, value) in [
            ("basis_name", &self.basis_name),
            ("method_name", &self.method_name),
            ("solvent", &self.solvent),
        ] {
            if value.trim().is_empty() {
                return Err(EngineError::InvalidInput(format!(
                    "{label} must not be empty"
                )));
            }
        }
        self.options.validate()?;
        self.validate_phase2_options()
    }

    /// 阶段二选项的跨字段校验（抽出成独立方法，便于单测逐条覆盖）。
    fn validate_phase2_options(&self) -> Result<(), EngineError> {
        let is_hf = self.method_name.trim().eq_ignore_ascii_case("hf");
        if let Some(scf_type) = self.options.scf_type {
            if scf_type.is_hartree_fock() != is_hf {
                return Err(EngineError::InvalidInput(format!(
                    "options.scf_type = `{}` contradicts method_name = `{}` \
                     (HF variants need \"hf\", KS variants need a functional)",
                    scf_type.as_str(),
                    self.method_name
                )));
            }
            let multiplicity = self.molecule.multiplicity;
            if scf_type.is_open_shell() && multiplicity <= 1 {
                return Err(EngineError::InvalidInput(format!(
                    "options.scf_type = `{}` is open-shell but multiplicity = {multiplicity}",
                    scf_type.as_str()
                )));
            }
            if !scf_type.is_open_shell() && multiplicity != 1 {
                return Err(EngineError::InvalidInput(format!(
                    "options.scf_type = `{}` is closed-shell but multiplicity = {multiplicity}",
                    scf_type.as_str()
                )));
            }
            if self.options.mp2.is_some() && scf_type.is_open_shell() {
                return Err(EngineError::InvalidInput(
                    "options.mp2 currently supports closed-shell references only (rhf/rks)".into(),
                ));
            }
        }
        if let Some(model) = self.options.solvent_model {
            let in_vacuum = self.solvent.trim().eq_ignore_ascii_case("vacuum");
            if in_vacuum && model != SolventModel::Vacuum {
                return Err(EngineError::InvalidInput(format!(
                    "options.solvent_model = `{}` but `solvent: vacuum`; \
                     name a solvent in `solvent:` first",
                    model.as_str()
                )));
            }
            if !in_vacuum && model == SolventModel::Vacuum {
                return Err(EngineError::InvalidInput(format!(
                    "options.solvent_model = `vacuum` contradicts `solvent: {}`",
                    self.solvent
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ExcitedStatesOptions, GridLevel, JobInput, JobType, TaskOptions};
    use crate::correlation::{Mp2Kind, Mp2Options, ScfType};
    use crate::molecule::{Atom, Molecule};
    use crate::properties::ExcitedStateMethod;
    use crate::solvation::SolventModel;

    fn h2_input() -> JobInput {
        JobInput::new(
            Molecule::new(
                vec![
                    Atom::new("H", [0.0, 0.0, -0.37]),
                    Atom::new("H", [0.0, 0.0, 0.37]),
                ],
                0,
                1,
            ),
            "sto-3g",
            "svwn5",
            GridLevel::Normal,
            "vacuum",
            JobType::Sp,
            TaskOptions::default(),
        )
    }

    #[test]
    fn defaults_are_inside_the_documented_bands() {
        let o = TaskOptions::default();
        assert_eq!(o.scf_max_iter, 128);
        assert_eq!(o.diis_history, 8);
        assert!((o.scf_energy_threshold_hartree - 1.0e-8).abs() < f64::EPSILON);
        assert!((o.opt_gradient_threshold_hartree_per_bohr - 4.5e-4).abs() < 1e-18);
        assert_eq!(o.threads, 0);
        assert!(o.ecp_name.is_none());
        assert!(o.validate().is_ok());
    }

    #[test]
    fn validate_rejects_bad_options_and_blank_names() {
        let o = TaskOptions {
            scf_max_iter: 0,
            ..TaskOptions::default()
        };
        assert!(o.validate().is_err());

        let o = TaskOptions {
            scf_energy_threshold_hartree: 0.0,
            ..TaskOptions::default()
        };
        assert!(o.validate().is_err());

        let o = TaskOptions {
            freq_step_angstrom: f64::NAN,
            ..TaskOptions::default()
        };
        assert!(o.validate().is_err());

        let mut input = h2_input();
        input.basis_name = "  ".into();
        assert!(input.validate().is_err());

        assert!(h2_input().validate().is_ok());
    }

    #[test]
    fn jobtype_and_gridlevel_serde_spelling_is_frozen() {
        // K4/input.yaml 约定：小写，且 K2 的 Sp/Opt/Freq 拼写必须仍可解析（alias）。
        assert_eq!(serde_json::to_string(&JobType::Sp).unwrap(), "\"sp\"");
        assert_eq!(serde_json::to_string(&JobType::Opt).unwrap(), "\"opt\"");
        assert_eq!(serde_json::to_string(&JobType::Freq).unwrap(), "\"freq\"");
        assert_eq!(
            serde_json::from_str::<JobType>("\"Sp\"").unwrap(),
            JobType::Sp
        );
        assert_eq!(
            serde_json::from_str::<GridLevel>("\"veryfine\"").unwrap(),
            GridLevel::VeryFine
        );
        assert!(serde_json::from_str::<GridLevel>("\"VeryFine\"").is_err());
    }

    #[test]
    fn input_yaml_shape_roundtrips_through_json() {
        let json = serde_json::to_string(&h2_input()).unwrap();
        assert!(json.contains("\"basis_name\":\"sto-3g\""));
        assert!(json.contains("\"position_angstrom\""));
        let back: JobInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h2_input());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // deny_unknown_fields：拼错字段名必须报错，不能被静默忽略。
        let json = r#"{"molecule":{"atoms":[{"symbol":"H","position_angstrom":[0,0,0]}],
            "charge":0,"multiplicity":1},"basis_name":"sto-3g","method_name":"svwn5",
            "grid_level":"normal","solvent":"vacuum","jobtype":"sp",
            "options":{"scf_max_iter":128,"scf_energy_threshold_hartree":1e-8,
            "scf_density_threshold":1e-6,"diis_history":8,"threads":0,"max_opt_steps":100,
            "opt_gradient_threshold_hartree_per_bohr":4.5e-4,"freq_step_angstrom":0.01,
            "ecp_name":null,"typo_field":1}}"#;
        assert!(serde_json::from_str::<JobInput>(json).is_err());
    }

    // ───────── 阶段二：新增选项必须是**纯加性**的（phase-2 §4.1）─────────

    #[test]
    fn phase1_options_json_still_parses_without_the_new_keys() {
        // 阶段一写出的 input.yaml / task.json 里没有 scf_type/mp2/... 这些键：
        // 必须照常解析，且新字段都是 None（默认行为逐位不变）。
        let json = r#"{"scf_max_iter":200,"scf_energy_threshold_hartree":1e-8,
            "scf_density_threshold":1e-6,"diis_history":8,"threads":0,"max_opt_steps":100,
            "opt_gradient_threshold_hartree_per_bohr":4.5e-4,"freq_step_angstrom":0.01,
            "ecp_name":"lanl2dz-cu"}"#;
        let options: TaskOptions = serde_json::from_str(json).expect("旧选项必须仍可解析");
        assert!(options.scf_type.is_none());
        assert!(options.mp2.is_none());
        assert!(options.excited_states.is_none());
        assert!(options.solvent_model.is_none());
        assert!(options.validate().is_ok());
        assert_eq!(options.ecp_name.as_deref(), Some("lanl2dz-cu"));
    }

    #[test]
    fn new_options_serialize_only_when_set() {
        let default_json = serde_json::to_string(&TaskOptions::default()).unwrap();
        for key in ["scf_type", "mp2", "excited_states", "solvent_model"] {
            assert!(
                !default_json.contains(key),
                "缺省不该出现 {key}: {default_json}"
            );
        }

        let options = TaskOptions {
            scf_type: Some(ScfType::Uks),
            mp2: Some(Mp2Options::new(Mp2Kind::RiMp2).with_aux_basis("def2-svp-ri")),
            excited_states: Some(ExcitedStatesOptions::new(5, ExcitedStateMethod::Tda)),
            solvent_model: Some(SolventModel::Cosmo),
            ..TaskOptions::default()
        };
        let json = serde_json::to_string(&options).unwrap();
        assert!(json.contains("\"scf_type\":\"uks\""), "{json}");
        assert!(json.contains("\"mp2\""), "{json}");
        assert!(json.contains("\"n_states\":5"), "{json}");
        assert!(json.contains("\"solvent_model\":\"cosmo\""), "{json}");
        let back: TaskOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back, options);
    }

    #[test]
    fn scf_type_must_match_method_family() {
        // HF 变体 ↔ method_name == "hf"
        let mut input = h2_input(); // svwn5
        input.options.scf_type = Some(ScfType::Rhf);
        assert!(input.validate().is_err(), "svwn5 + rhf 自相矛盾");

        let mut input = h2_input();
        input.method_name = "hf".into();
        input.options.scf_type = Some(ScfType::Rhf);
        assert!(input.validate().is_ok());

        let mut input = h2_input();
        input.method_name = "hf".into();
        input.options.scf_type = Some(ScfType::Uks);
        assert!(input.validate().is_err(), "hf + uks 自相矛盾");
    }

    #[test]
    fn open_shell_scf_type_requires_a_multiplicity() {
        let mut input = h2_input(); // 单重态
        input.options.scf_type = Some(ScfType::Uks);
        assert!(input.validate().is_err(), "单重态 + uks 不合法");

        let mut input = h2_input();
        input.molecule.multiplicity = 3;
        input.options.scf_type = Some(ScfType::Uks);
        assert!(input.validate().is_ok());

        let mut input = h2_input();
        input.molecule.multiplicity = 3;
        input.options.scf_type = Some(ScfType::Rks);
        assert!(input.validate().is_err(), "三重态 + rks 不合法");
    }

    #[test]
    fn mp2_is_closed_shell_only_for_now() {
        let mut input = h2_input();
        input.options.mp2 = Some(Mp2Options::new(Mp2Kind::Mp2));
        assert!(input.validate().is_ok(), "闭壳层 MP2 可以");

        let mut input = h2_input();
        input.molecule.multiplicity = 3;
        input.options.scf_type = Some(ScfType::Uks);
        input.options.mp2 = Some(Mp2Options::new(Mp2Kind::Mp2));
        let message = input
            .validate()
            .expect_err("开壳层 MP2 还没实现")
            .to_string();
        assert!(message.contains("closed-shell"), "{message}");
    }

    #[test]
    fn solvent_model_must_agree_with_the_solvent_name() {
        let mut input = h2_input(); // solvent: vacuum
        input.options.solvent_model = Some(SolventModel::Cosmo);
        assert!(input.validate().is_err(), "真空里不该给 cosmo");

        let mut input = h2_input();
        input.options.solvent_model = Some(SolventModel::Vacuum);
        assert!(
            input.validate().is_ok(),
            "显式 vacuum 与 solvent: vacuum 一致"
        );

        let mut input = h2_input();
        input.solvent = "water".into();
        input.options.solvent_model = Some(SolventModel::Cosmo);
        assert!(input.validate().is_ok());

        let mut input = h2_input();
        input.solvent = "water".into();
        input.options.solvent_model = Some(SolventModel::Vacuum);
        assert!(input.validate().is_err(), "水里不该给 vacuum 模型");
    }

    #[test]
    fn excited_states_needs_a_positive_state_count() {
        let mut input = h2_input();
        input.options.excited_states = Some(ExcitedStatesOptions::new(0, ExcitedStateMethod::Tda));
        assert!(input.validate().is_err());

        let mut input = h2_input();
        input.options.excited_states = Some(ExcitedStatesOptions::new(3, ExcitedStateMethod::Cis));
        assert!(input.validate().is_ok());
    }
}
