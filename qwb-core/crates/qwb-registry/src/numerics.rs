//! 精度等级与等价容差（K10 的数据侧，phase-2 §8.5）。
//!
//! K10 的三条门禁（等价性 / 单调收敛 / 确定性留痕）需要两个"数据"才能执行：
//!
//! 1. **精度等级** `level = 1..=5` 的语义：每一级绑一个 XC 网格等级（**只引用名字**，
//!    网格参数仍只有 `grid/*.yaml` 一份）与 pseudospectral 路径的配置；
//! 2. **等价容差**：RI / pseudospectral 与解析路径的允许偏差（Hartree）。
//!
//! 为什么容差进数据而不是写死在代码里：它是**验收判据**，改判据要留痕（`CHANGELOG` +
//! `DATA_LICENSES`），和在代码里改一个常量不是一回事。
//!
//! 单调性（K10 条件 ②）在数据层就能查一半：等级越高，PS 配点数必须越多；去别名一旦打开
//! 就不许关回去。真正"误差随等级下降"的检查要等实现（`VF-06` 的等价性套件），这里只保证
//! 数据结构不会自相矛盾。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};

/// pseudospectral 路径的配置（K10；`IN-03` 会消费它）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PsSettings {
    /// 每个原子的配点数（pseudospectral 的 collocation grid）。
    pub collocation_points_per_atom: u32,
    /// 是否启用去别名（de-aliasing）。打开之后不许在更高等级关回去。
    pub de_aliasing: bool,
}

impl PsSettings {
    /// 构造。
    #[must_use]
    pub fn new(collocation_points_per_atom: u32, de_aliasing: bool) -> Self {
        Self {
            collocation_points_per_atom,
            de_aliasing,
        }
    }
}

/// 一个精度等级（K10 / phase-2 §8.5）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct AccuracyLevel {
    /// 等级，`1` 最快、`5` 最准。
    pub level: u8,
    /// 推荐的 XC 网格等级名（**引用** `grid/*.yaml` 的 `name`，不重复参数）。
    pub grid_level: String,
    /// pseudospectral 配置；`None` = 该等级不用于 PS 路径。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ps: Option<PsSettings>,
    /// 人类可读说明（可选）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl AccuracyLevel {
    /// 构造。
    #[must_use]
    pub fn new(level: u8, grid_level: impl Into<String>) -> Self {
        Self {
            level,
            grid_level: grid_level.into(),
            ps: None,
            description: None,
        }
    }

    /// 补上 pseudospectral 配置。
    #[must_use]
    pub fn with_ps(mut self, ps: PsSettings) -> Self {
        self.ps = Some(ps);
        self
    }

    /// 补上说明。
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// 等价容差（K10 条件 ①）：近似路径与解析路径的允许总能量差，单位 Hartree。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EquivalenceTolerance {
    /// RI（密度拟合）路径的容差。
    pub ri: f64,
    /// pseudospectral 路径的容差（比 RI 松，phase-2 §9.3）。
    pub pseudospectral: f64,
}

impl EquivalenceTolerance {
    /// 构造。
    #[must_use]
    pub fn new(ri: f64, pseudospectral: f64) -> Self {
        Self { ri, pseudospectral }
    }
}

/// 数值路径的精度等级与等价容差（`qwb-data/numerics/<name>.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NumericsDef {
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
    /// 精度等级表（至少一级，等级号唯一且递增）。
    pub levels: Vec<AccuracyLevel>,
    /// 等价容差。
    pub equivalence_tolerance: EquivalenceTolerance,
}

impl_data_common!(NumericsDef);

impl NumericsDef {
    /// 完整校验（`qwb data verify` 用）：G2 门禁 + 结构 + 单调性。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues.extend(self.check_monotone());
        issues
    }

    /// 结构性检查（**不含** G2 门禁，`Registry::open` 用；§8.6 的 load ≠ verify）。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.levels.is_empty() {
            issues.push("levels must not be empty (K10 needs at least one level)".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for entry in &self.levels {
            if !(1..=5).contains(&entry.level) {
                issues.push(format!(
                    "level {} is out of range: accuracy levels are 1..=5 (phase-2 §8.5)",
                    entry.level
                ));
            }
            if !seen.insert(entry.level) {
                issues.push(format!("duplicate accuracy level {}", entry.level));
            }
            if entry.grid_level.trim().is_empty() {
                issues.push(format!(
                    "level {}: grid_level must not be empty",
                    entry.level
                ));
            }
            if let Some(ps) = entry.ps
                && ps.collocation_points_per_atom == 0
            {
                issues.push(format!(
                    "level {}: ps.collocation_points_per_atom must be >= 1",
                    entry.level
                ));
            }
        }

        for (label, tolerance) in [
            ("ri", self.equivalence_tolerance.ri),
            ("pseudospectral", self.equivalence_tolerance.pseudospectral),
        ] {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                issues.push(format!(
                    "equivalence_tolerance.{label} must be positive and finite"
                ));
            } else if tolerance >= 1.0e-3 {
                // 容差松到 1e-3 Ha 就不再是"等价"，而是"另一个方法"。
                issues.push(format!(
                    "equivalence_tolerance.{label} = {tolerance} is too loose \
                     (must be < 1e-3 Ha, phase-2 §4.5)"
                ));
            }
        }
        issues
    }

    /// K10 条件 ② 的数据侧一半：配点数随等级严格增加、去别名只增不减。
    #[must_use]
    pub fn check_monotone(&self) -> Vec<String> {
        let mut issues = Vec::new();
        let mut previous: Option<&AccuracyLevel> = None;
        let mut sorted: Vec<&AccuracyLevel> = self.levels.iter().collect();
        sorted.sort_by_key(|entry| entry.level);
        for entry in &sorted {
            if let (Some(previous), Some(current)) = (previous.and_then(|p| p.ps), entry.ps) {
                if current.collocation_points_per_atom <= previous.collocation_points_per_atom {
                    issues.push(format!(
                        "level {}: ps.collocation_points_per_atom ({}) must increase over level {} ({})",
                        entry.level,
                        current.collocation_points_per_atom,
                        previous_level(&sorted, entry.level),
                        previous.collocation_points_per_atom
                    ));
                }
                if previous.de_aliasing && !current.de_aliasing {
                    issues.push(format!(
                        "level {}: de_aliasing must not be turned off again (level {} has it on)",
                        entry.level,
                        previous_level(&sorted, entry.level)
                    ));
                }
            }
            previous = Some(entry);
        }
        issues
    }

    /// 取某一级；不存在时 `None`。
    #[must_use]
    pub fn level(&self, level: u8) -> Option<&AccuracyLevel> {
        self.levels.iter().find(|entry| entry.level == level)
    }

    /// 按路径取等价容差（Hartree）。
    ///
    /// 解析路径没有容差概念（它是基准），返回 `0.0`。
    #[must_use]
    pub fn tolerance_hartree(&self, path: &str) -> f64 {
        match path {
            "ri" => self.equivalence_tolerance.ri,
            "pseudospectral" => self.equivalence_tolerance.pseudospectral,
            _ => 0.0,
        }
    }

    /// 引用了哪些网格等级名（loader 据此做跨文件校验）。
    #[must_use]
    pub fn grid_level_refs(&self) -> Vec<&str> {
        self.levels
            .iter()
            .map(|entry| entry.grid_level.as_str())
            .collect()
    }
}

/// 上一级的等级号（只为错误消息里指认"和谁比"）。
fn previous_level(sorted: &[&AccuracyLevel], level: u8) -> u8 {
    let mut previous = level;
    for entry in sorted {
        if entry.level < level {
            previous = entry.level;
        }
    }
    previous
}

#[cfg(test)]
mod tests {
    use super::{AccuracyLevel, EquivalenceTolerance, NumericsDef, PsSettings};
    use crate::common::{Citation, License};

    pub(crate) fn numerics() -> NumericsDef {
        NumericsDef {
            name: "numerics".into(),
            version: "1.0.0".into(),
            citation: vec![Citation::with_url("fixture", "https://example.org/")],
            license: License::new("Apache-2.0", true, true),
            created: "2026-11-06T00:00:00Z".into(),
            sha256: Some("a".repeat(64)),
            notes: None,
            levels: vec![
                AccuracyLevel::new(1, "grid-coarse").with_ps(PsSettings::new(30, false)),
                AccuracyLevel::new(2, "grid-coarse").with_ps(PsSettings::new(60, false)),
                AccuracyLevel::new(3, "grid-normal").with_ps(PsSettings::new(90, true)),
                AccuracyLevel::new(4, "grid-fine").with_ps(PsSettings::new(120, true)),
                AccuracyLevel::new(5, "grid-veryfine").with_ps(PsSettings::new(180, true)),
            ],
            equivalence_tolerance: EquivalenceTolerance::new(1.0e-6, 5.0e-5),
        }
    }

    #[test]
    fn fixture_is_clean() {
        let def = numerics();
        assert!(def.check().is_empty(), "{:?}", def.check());
        assert_eq!(
            def.level(3).map(|l| l.grid_level.as_str()),
            Some("grid-normal")
        );
        assert!(def.level(6).is_none());
        assert!((def.tolerance_hartree("ri") - 1.0e-6).abs() < 1e-15);
        assert!((def.tolerance_hartree("pseudospectral") - 5.0e-5).abs() < 1e-15);
        assert_eq!(def.tolerance_hartree("analytic"), 0.0);
        assert_eq!(def.grid_level_refs().len(), 5);
    }

    #[test]
    fn duplicate_and_out_of_range_levels_are_rejected() {
        let mut def = numerics();
        def.levels.push(AccuracyLevel::new(3, "grid-fine"));
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("duplicate accuracy level")),
            "{:?}",
            def.check_structure()
        );

        let mut def = numerics();
        def.levels.push(AccuracyLevel::new(6, "grid-fine"));
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("out of range")),
            "{:?}",
            def.check_structure()
        );

        let mut def = numerics();
        def.levels.clear();
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("must not be empty"))
        );

        let mut def = numerics();
        def.levels[0].grid_level = "  ".into();
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("grid_level"))
        );
    }

    #[test]
    fn equivalence_tolerance_must_be_tight_and_positive() {
        let mut def = numerics();
        def.equivalence_tolerance.ri = 0.0;
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("equivalence_tolerance.ri")),
            "{:?}",
            def.check_structure()
        );

        let mut def = numerics();
        def.equivalence_tolerance.pseudospectral = 1.0e-2;
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("too loose")),
            "{:?}",
            def.check_structure()
        );
    }

    #[test]
    fn ps_settings_must_be_monotone_in_the_level() {
        let mut def = numerics();
        // 等级 4 的配点数不该比等级 3 少
        def.levels[3].ps = Some(PsSettings::new(90, true));
        assert!(
            def.check_monotone()
                .iter()
                .any(|m| m.contains("must increase")),
            "{:?}",
            def.check_monotone()
        );

        let mut def = numerics();
        // 去别名一旦打开不许关回去
        def.levels[4].ps = Some(PsSettings::new(180, false));
        assert!(
            def.check_monotone()
                .iter()
                .any(|m| m.contains("de_aliasing must not be turned off")),
            "{:?}",
            def.check_monotone()
        );

        // 没有 ps 的等级不参与单调性比较（只在解析路径上用它）
        let mut def = numerics();
        def.levels[1].ps = None;
        assert!(
            def.check_monotone().is_empty(),
            "{:?}",
            def.check_monotone()
        );
    }

    #[test]
    fn level_with_zero_collocation_points_is_rejected() {
        let mut def = numerics();
        def.levels[0].ps = Some(PsSettings::new(0, false));
        assert!(
            def.check_structure()
                .iter()
                .any(|m| m.contains("collocation_points_per_atom")),
            "{:?}",
            def.check_structure()
        );
    }
}
