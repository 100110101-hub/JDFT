//! 作业输入（K2）。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;
use crate::molecule::Molecule;

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
        }
    }
}

impl TaskOptions {
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
    /// # Errors
    ///
    /// 分子非法、基组/泛函/溶剂键为空、选项越界时返回 [`EngineError::InvalidInput`]。
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
        self.options.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::{GridLevel, JobInput, JobType, TaskOptions};
    use crate::molecule::{Atom, Molecule};

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
}
