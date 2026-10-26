//! K10 · 数值路径的报告与精度等级选项（phase-2 §4.5）。
//!
//! 三条门禁（等价性 / 单调收敛 / 确定性）要求**每一个近似都必须留痕**：
//! 用了哪条路径、哪一级精度、网格/拟合/配点是多少、与解析路径差多少。
//! 这些数字如果只写在日志里就等于没有 —— 它们在 [`crate::JobResult::numerics_report`] 里。
//!
//! 语义约定：
//!
//! - [`NumericsPath::Analytic`] 是**基准**，不参与"等价"比较（`accuracy_level` 仍然记录，
//!   因为它决定网格/配点的实际用量）；
//! - `ri` 与 `ps` 两个子块按路径出现：RI 路径只有 `ri`，PS 路径只有 `ps`；
//! - `wall_time_s` 是墙钟，不是 CPU 时间（K10 的"值不值得"要按墙钟判）。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 数值路径（K10）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum NumericsPath {
    /// 解析积分 + 网格 XC（阶段一的路径，也是 K10 的基准）。
    #[default]
    Analytic,
    /// 密度拟合（RI）：Fock 构建走 3 中心积分。
    Ri,
    /// pseudospectral：Coulomb/exchange 在网格上算。
    Pseudospectral,
}

impl NumericsPath {
    /// 冻结的小写名。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Analytic => "analytic",
            Self::Ri => "ri",
            Self::Pseudospectral => "pseudospectral",
        }
    }

    /// 是否是与解析路径等价的近似（要过 K10 条件 ①）。
    #[must_use]
    pub fn is_approximation(self) -> bool {
        !matches!(self, Self::Analytic)
    }
}

/// 精度等级选项（`TaskOptions.numerics`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NumericsOptions {
    /// 路径；`None` = 解析路径（阶段一行为）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<NumericsPath>,
    /// 精度等级 `1..=5`（语义在 `qwb-data/numerics/*.yaml`）。
    pub accuracy_level: u8,
}

impl Default for NumericsOptions {
    fn default() -> Self {
        Self {
            path: None,
            accuracy_level: 3,
        }
    }
}

impl NumericsOptions {
    /// 构造。
    #[must_use]
    pub fn new(path: NumericsPath, accuracy_level: u8) -> Self {
        Self {
            path: Some(path),
            accuracy_level,
        }
    }

    /// 校验等级区间（phase-2 §8.5 只有 1..=5）。
    ///
    /// # Errors
    ///
    /// `accuracy_level` 不在 `1..=5` 时返回 [`EngineError::InvalidInput`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(1..=5).contains(&self.accuracy_level) {
            return Err(EngineError::InvalidInput(format!(
                "options.numerics.accuracy_level must be 1..=5, got {}",
                self.accuracy_level
            )));
        }
        Ok(())
    }
}

/// 使用的网格（`NumericsReport.grid`）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GridSummary {
    /// 每原子径向点数。
    pub radial: u32,
    /// 角向点数。
    pub angular: u32,
    /// 网格截断半径，单位 Bohr。
    pub cutoff_bohr: f64,
}

impl GridSummary {
    /// 构造。
    #[must_use]
    pub fn new(radial: u32, angular: u32, cutoff_bohr: f64) -> Self {
        Self {
            radial,
            angular,
            cutoff_bohr,
        }
    }
}

/// RI 路径的留痕（`NumericsReport.ri`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RiSummary {
    /// 辅助基名。
    pub aux_basis_name: String,
    /// 拟合误差（与解析路径的总能量差），单位 Hartree。
    pub fit_error_hartree: f64,
}

impl RiSummary {
    /// 构造。
    #[must_use]
    pub fn new(aux_basis_name: impl Into<String>, fit_error_hartree: f64) -> Self {
        Self {
            aux_basis_name: aux_basis_name.into(),
            fit_error_hartree,
        }
    }
}

/// pseudospectral 路径的留痕（`NumericsReport.ps`）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PsSummary {
    /// 每原子配点数。
    pub collocation_points_per_atom: u32,
    /// 是否用了去别名。
    pub de_aliasing: bool,
    /// Coulomb 项的误差，单位 Hartree。
    pub coulomb_error_hartree: f64,
    /// 交换项的误差，单位 Hartree。
    pub exchange_error_hartree: f64,
}

impl PsSummary {
    /// 构造。
    #[must_use]
    pub fn new(
        collocation_points_per_atom: u32,
        de_aliasing: bool,
        coulomb_error_hartree: f64,
        exchange_error_hartree: f64,
    ) -> Self {
        Self {
            collocation_points_per_atom,
            de_aliasing,
            coulomb_error_hartree,
            exchange_error_hartree,
        }
    }

    /// 两项误差之和（K10 条件 ① 比较的是总能量，这一项便于看是哪边贡献的）。
    #[must_use]
    pub fn total_error_hartree(&self) -> f64 {
        self.coulomb_error_hartree + self.exchange_error_hartree
    }
}

/// 数值路径报告（K10；进 `result.json`，缺省不出现）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NumericsReport {
    /// 用的路径。
    pub path: NumericsPath,
    /// 精度等级 `1..=5`。
    pub accuracy_level: u8,
    /// 网格（XC 求积；任何路径都有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridSummary>,
    /// RI 留痕（只有 RI 路径有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ri: Option<RiSummary>,
    /// PS 留痕（只有 PS 路径有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ps: Option<PsSummary>,
    /// 本次结果的总能量，单位 Hartree（与 `JobResult.energy_hartree` 一致，便于对账）。
    pub energy_hartree: f64,
    /// 墙钟时间，单位秒。
    pub wall_time_s: f64,
}

impl NumericsReport {
    /// 构造（`grid`/`ri`/`ps` 默认空）。
    #[must_use]
    pub fn new(
        path: NumericsPath,
        accuracy_level: u8,
        energy_hartree: f64,
        wall_time_s: f64,
    ) -> Self {
        Self {
            path,
            accuracy_level,
            grid: None,
            ri: None,
            ps: None,
            energy_hartree,
            wall_time_s,
        }
    }

    /// 补上网格。
    #[must_use]
    pub fn with_grid(mut self, grid: GridSummary) -> Self {
        self.grid = Some(grid);
        self
    }

    /// 补上 RI 留痕。
    #[must_use]
    pub fn with_ri(mut self, ri: RiSummary) -> Self {
        self.ri = Some(ri);
        self
    }

    /// 补上 PS 留痕。
    #[must_use]
    pub fn with_ps(mut self, ps: PsSummary) -> Self {
        self.ps = Some(ps);
        self
    }

    /// K10 条件 ① 的判定：本报告的误差是否在给定容差内。
    ///
    /// 解析路径恒为 `true`（它就是基准）；近似路径缺对应留痕时返回 `false`
    /// —— "没有数字"不能当成"通过"。
    #[must_use]
    pub fn within_tolerance(&self, ri_tol_hartree: f64, ps_tol_hartree: f64) -> bool {
        match self.path {
            NumericsPath::Analytic => true,
            NumericsPath::Ri => self
                .ri
                .as_ref()
                .is_some_and(|ri| ri.fit_error_hartree.abs() <= ri_tol_hartree),
            NumericsPath::Pseudospectral => self
                .ps
                .as_ref()
                .is_some_and(|ps| ps.total_error_hartree().abs() <= ps_tol_hartree),
        }
    }

    /// 结构自洽性（写进 `result.json` 之前）。
    ///
    /// # Errors
    ///
    /// 等级越界、能量/时间非有限、或路径与留痕不匹配（RI 路径却没有 `ri`）时
    /// 返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(1..=5).contains(&self.accuracy_level) {
            return Err(EngineError::Internal(format!(
                "numerics report accuracy_level {} is out of 1..=5",
                self.accuracy_level
            )));
        }
        if !self.energy_hartree.is_finite() || !self.wall_time_s.is_finite() {
            return Err(EngineError::Internal(
                "numerics report contains a non-finite number".into(),
            ));
        }
        if self.wall_time_s < 0.0 {
            return Err(EngineError::Internal(
                "numerics report wall_time_s must be >= 0".into(),
            ));
        }
        match self.path {
            NumericsPath::Ri if self.ri.is_none() => {
                return Err(EngineError::Internal(
                    "numerics report for `ri` path is missing the `ri` block (K10 requires the trace)"
                        .into(),
                ));
            }
            NumericsPath::Pseudospectral if self.ps.is_none() => {
                return Err(EngineError::Internal(
                    "numerics report for `pseudospectral` path is missing the `ps` block \
                     (K10 requires the trace)"
                        .into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GridSummary, NumericsOptions, NumericsPath, NumericsReport, PsSummary, RiSummary};

    #[test]
    fn path_spelling_and_predicate_are_frozen() {
        assert_eq!(NumericsPath::Analytic.as_str(), "analytic");
        assert_eq!(
            serde_json::to_string(&NumericsPath::Pseudospectral).unwrap(),
            "\"pseudospectral\""
        );
        assert!(!NumericsPath::Analytic.is_approximation());
        assert!(NumericsPath::Ri.is_approximation());
        assert_eq!(NumericsPath::default(), NumericsPath::Analytic);
    }

    #[test]
    fn options_default_to_the_analytic_path_at_level_three() {
        let options = NumericsOptions::default();
        assert!(options.path.is_none());
        assert_eq!(options.accuracy_level, 3);
        assert!(options.validate().is_ok());
        assert!(!serde_json::to_string(&options).unwrap().contains("path"));

        for bad in [0u8, 6] {
            let options = NumericsOptions {
                accuracy_level: bad,
                ..NumericsOptions::default()
            };
            assert!(options.validate().is_err(), "level {bad} 应当被拒");
        }
    }

    #[test]
    fn analytic_report_is_the_reference() {
        let report = NumericsReport::new(NumericsPath::Analytic, 3, -1.157, 1.2)
            .with_grid(GridSummary::new(75, 302, 20.0));
        assert!(report.validate().is_ok());
        assert!(report.within_tolerance(1.0e-6, 5.0e-5));
        assert!(report.ri.is_none());
        assert!(report.ps.is_none());
    }

    #[test]
    fn approximation_needs_its_trace_and_the_tolerance() {
        let ri = NumericsReport::new(NumericsPath::Ri, 3, -76.33, 4.0)
            .with_ri(RiSummary::new("def2-svp-ri", -2.0e-7));
        assert!(ri.validate().is_ok());
        assert!(ri.within_tolerance(1.0e-6, 5.0e-5));
        assert!(!ri.within_tolerance(1.0e-9, 5.0e-5));

        let missing_trace = NumericsReport::new(NumericsPath::Ri, 3, -76.33, 4.0);
        assert!(missing_trace.validate().is_err(), "没有留痕不能算通过");
        assert!(!missing_trace.within_tolerance(1.0e-3, 1.0e-3));

        let ps = NumericsReport::new(NumericsPath::Pseudospectral, 4, -76.33, 9.0)
            .with_ps(PsSummary::new(120, true, -1.0e-6, 2.0e-6));
        assert!(ps.validate().is_ok());
        assert!((ps.ps.expect("ps").total_error_hartree() - 1.0e-6).abs() < 1e-12);
        assert!(ps.within_tolerance(1.0e-6, 5.0e-5));
    }

    #[test]
    fn report_roundtrips_and_omits_absent_blocks() {
        let report = NumericsReport::new(NumericsPath::Analytic, 5, -1.157, 0.5)
            .with_grid(GridSummary::new(140, 974, 20.0));
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(!json.contains("\"ri\""), "{json}");
        assert!(!json.contains("\"ps\""), "{json}");
        let back: NumericsReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);

        let bad = NumericsReport::new(NumericsPath::Analytic, 3, f64::NAN, 1.0);
        assert!(bad.validate().is_err());
    }
}
