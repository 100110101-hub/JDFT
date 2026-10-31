//! 溶剂数据（§4 K1 / §8.5）。
//!
//! 第一阶段只做**接口与数据**，不做完整 PCM：`cavity_model` 必须显式声明，
//! loader 不因为它不认识就报错（§8.5），由 `Engine::run` 返回 `Unsupported`。

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::{Citation, DataCommon, License};
use crate::elements;

/// 空腔模型（§8.5：必须显式声明，禁止隐式默认）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum CavityModel {
    /// 无空腔（真空；理论上不该作为溶剂条目出现）。
    None,
    /// 导体屏蔽模型（C-PCM 一族）。
    Pcm,
    /// SMD（§2.2 非目标；保留数据位以便 loader 通过、引擎返回 Unsupported）。
    Smd,
}

/// 原子半径所依据的半径集（phase-2 §8.2；**不能混用**不同半径集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum RadiiSet {
    /// Bondi van der Waals 半径。
    Bondi,
    /// UFF 力场半径。
    Uff,
    /// SMD 的参数化半径。
    Smd,
}

/// 溶剂（`qwb-data/solvent/*.yaml` 的形态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Solvent {
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
    /// 相对介电常数 `epsilon`。
    pub epsilon: f64,
    /// 温度，单位 K（K1 字段名 `temperature_K`）。
    #[serde(rename = "temperature_K")]
    pub temperature_k: f64,
    /// 各元素原子半径，单位 Å（键：元素符号）。
    pub atomic_radii: BTreeMap<String, f64>,
    /// 空腔模型。
    pub cavity_model: CavityModel,
    /// 半径缩放因子。
    pub scale: f64,
    // ── 阶段二（phase-2 §8.2 / K8）的**可选数据**：旧溶剂文件不受影响。──
    /// 光学介电常数（折射率 `n`；SMD 的非极性项用 `n²`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refractive_index: Option<f64>,
    /// `atomic_radii` 用的是哪一套半径（Bondi / UFF / SMD）。混用会静默错数，所以必须声明。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radii_set: Option<RadiiSet>,
    /// 表面张力，单位 cal/(mol·Å²)；SMD 的空化项用。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_tension_cal_per_mol_angstrom2: Option<f64>,
    /// 离子强度，单位 mol/L（PBF 用）；`None` = 纯溶剂。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ionic_strength_molar: Option<f64>,

    /// SMD 非极性项（可选；缺省 = 该溶剂没有可用的 SMD 参数）。
    #[serde(default)]
    pub smd: Option<SmdParameters>,
}

/// **SMD 非极性项**（DA-05 / D0-02）。加性扩展：缺省表示该溶剂没有可用的 SMD 非极性参数。
///
/// 数据来源是 Minnesota 溶剂化数据库表（Marenich–Cramer–Truhlar, J. Phys. Chem. A 2009,
/// `doi:10.1021/jp810292n`，Table 1）—— 数字是数据事实，不是从任何 GPL 源码文件复制的文本。
///
/// `nonpolar_terms_available = false` 是**有意义的状态**：water 在该表里
/// `gamma/phi²/psi²` 是 `-1` 占位（SMD 对水走单独的 CDS 处理），所以这三个量
/// 不能当 0 用、也不许编造。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SmdParameters {
    /// 表面张力 γ（mN/m，数据库原值）。
    pub gamma_mn_per_m: Option<f64>,
    /// φ² 参数。
    pub phi_sq: Option<f64>,
    /// ψ² 参数。
    pub psi_sq: Option<f64>,
    /// Abraham α（氢键酸性）。
    pub abraham_alpha: Option<f64>,
    /// Abraham β（氢键碱性）。
    pub abraham_beta: Option<f64>,
    /// 20 °C 折射率。
    pub refractive_index_20c: Option<f64>,
    /// 25 °C 折射率。
    pub refractive_index_25c: Option<f64>,
    /// 该溶剂的 SMD 非极性项是否可用（water 为 `false`）。
    pub nonpolar_terms_available: bool,
    /// 数据来源与说明（含 DOI）。
    pub source: Option<String>,
}

impl_data_common!(Solvent);

impl Solvent {
    /// 完整校验（`qwb data verify` 用）：G2 门禁 + 结构检查。
    ///
    /// ε 与文献一致的检查由 D-14 用基准值做，这里只查自洽性。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = self.check_common();
        issues.extend(self.check_structure());
        issues
    }

    /// 结构性检查：**不含** G2 门禁（`Registry::open` 用，见 §8.6）。
    #[must_use]
    pub fn check_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if !self.epsilon.is_finite() || self.epsilon < 1.0 {
            issues.push("epsilon must be finite and >= 1 (vacuum = 1)".into());
        }
        if !self.temperature_k.is_finite() || self.temperature_k <= 0.0 {
            issues.push("temperature_K must be positive and finite".into());
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            issues.push("scale must be positive and finite".into());
        }
        if self.atomic_radii.is_empty() {
            issues.push("atomic_radii must not be empty (§4 K1)".into());
        }
        for (element, radius) in &self.atomic_radii {
            if elements::atomic_number(element).is_none() {
                issues.push(format!("atomic_radii contains unknown symbol `{element}`"));
            }
            if !radius.is_finite() || *radius <= 0.0 {
                issues.push(format!(
                    "atomic_radii.{element} must be positive and finite (Å)"
                ));
            }
        }
        if matches!(self.cavity_model, CavityModel::None) {
            issues.push(
                "cavity_model = none: a solvent entry must declare a real cavity model (§8.5)"
                    .into(),
            );
        }

        // ── 阶段二新增字段的自洽性（缺省时全部跳过）──
        if let Some(n) = self.refractive_index {
            if !n.is_finite() || n <= 0.0 {
                issues.push("refractive_index must be positive and finite".into());
            } else if n * n > self.epsilon + 1e-9 {
                // 物理上 ε_optical = n² ≤ ε_static（phase-2 §8.2）。
                issues.push(format!(
                    "refractive_index² ({:.4}) exceeds epsilon ({:.4}); n² must not exceed ε_static",
                    n * n,
                    self.epsilon
                ));
            }
        }
        if let Some(tension) = self.surface_tension_cal_per_mol_angstrom2
            && (!tension.is_finite() || tension < 0.0)
        {
            issues.push(
                "surface_tension_cal_per_mol_angstrom2 must be non-negative and finite".into(),
            );
        }
        if let Some(ionic) = self.ionic_strength_molar
            && (!ionic.is_finite() || ionic < 0.0)
        {
            issues.push("ionic_strength_molar must be non-negative and finite".into());
        }
        if matches!(self.cavity_model, CavityModel::Smd) {
            // SMD 的非极性项需要 n² 与它自己的半径集；缺了就别假装能算（phase-2 §8.2）。
            if self.refractive_index.is_none() {
                issues.push("cavity_model = smd requires refractive_index (§8.2)".into());
            }
            if !matches!(self.radii_set, Some(RadiiSet::Smd)) {
                issues.push("cavity_model = smd requires radii_set = smd (§8.2)".into());
            }
        }
        issues
    }

    /// 是否为第一阶段引擎能实际处理的空腔模型（PCM 接口尚未实现，§8.5）。
    #[must_use]
    pub fn is_engine_supported(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{CavityModel, RadiiSet, Solvent};
    use crate::common::{Citation, License};

    pub(crate) fn water() -> Solvent {
        let mut radii = BTreeMap::new();
        radii.insert("H".to_string(), 1.2);
        radii.insert("O".to_string(), 1.5);
        Solvent {
            name: "water".into(),
            smd: None,
            version: "1.0.0".into(),
            citation: vec![Citation::with_url(
                "NIST Chemistry WebBook",
                "https://webbook.nist.gov/",
            )],
            license: License::new("Apache-2.0", true, true),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: Some("e".repeat(64)),
            notes: None,
            epsilon: 78.3553,
            temperature_k: 298.15,
            atomic_radii: radii,
            cavity_model: CavityModel::Pcm,
            scale: 1.2,
            refractive_index: None,
            radii_set: None,
            surface_tension_cal_per_mol_angstrom2: None,
            ionic_strength_molar: None,
        }
    }

    #[test]
    fn water_matches_literature_epsilon() {
        let w = water();
        assert!((w.epsilon - 78.3553).abs() < 1e-9, "§8.5 参考值");
        assert!(w.check().is_empty(), "{:?}", w.check());
        assert!(!w.is_engine_supported(), "§8.5: PCM 未实现");
    }

    #[test]
    fn check_catches_broken_solvents() {
        let mut w = water();
        w.epsilon = 0.5;
        assert!(w.check().iter().any(|m| m.contains("epsilon")));

        let mut w = water();
        w.temperature_k = 0.0;
        assert!(w.check().iter().any(|m| m.contains("temperature_K")));

        let mut w = water();
        w.atomic_radii.insert("Xx".into(), 1.0);
        assert!(w.check().iter().any(|m| m.contains("unknown symbol")));

        let mut w = water();
        w.cavity_model = CavityModel::None;
        assert!(w.check().iter().any(|m| m.contains("cavity_model")));

        let mut w = water();
        w.atomic_radii.clear();
        assert!(
            w.check()
                .iter()
                .any(|m| m.contains("atomic_radii must not be empty"))
        );
    }

    // ───────── 阶段二新增字段（phase-2 §8.2）：纯加性 + 自洽性 ─────────

    #[test]
    fn phase1_solvent_files_still_load_without_the_new_fields() {
        // 阶段一的溶剂文件没有 refractive_index/radii_set/...：必须照常通过。
        let w = water();
        assert!(w.refractive_index.is_none());
        assert!(w.radii_set.is_none());
        assert!(w.check().is_empty(), "{:?}", w.check());
    }

    #[test]
    fn refractive_index_must_not_exceed_static_dielectric() {
        let mut w = water();
        w.refractive_index = Some(1.33);
        assert!((1.33f64 * 1.33 - 1.7689).abs() < 1e-12);
        assert!(w.check().is_empty(), "n² = 1.77 < ε = 78.36 合法");

        let mut w = water();
        w.refractive_index = Some(9.0); // n² = 81 > 78.36
        assert!(
            w.check().iter().any(|m| m.contains("must not exceed")),
            "{:?}",
            w.check()
        );

        let mut w = water();
        w.refractive_index = Some(0.0);
        assert!(w.check().iter().any(|m| m.contains("refractive_index")));
    }

    #[test]
    fn smd_requires_its_own_radii_and_refractive_index() {
        let mut w = water();
        w.cavity_model = CavityModel::Smd;
        let issues = w.check();
        assert!(
            issues
                .iter()
                .any(|m| m.contains("requires refractive_index")),
            "{issues:?}"
        );
        assert!(
            issues
                .iter()
                .any(|m| m.contains("requires radii_set = smd")),
            "{issues:?}"
        );

        let mut w = water();
        w.cavity_model = CavityModel::Smd;
        w.refractive_index = Some(1.33);
        w.radii_set = Some(RadiiSet::Smd);
        w.surface_tension_cal_per_mol_angstrom2 = Some(50.0);
        w.ionic_strength_molar = Some(0.0);
        assert!(w.check().is_empty(), "{:?}", w.check());

        let mut w = water();
        w.radii_set = Some(RadiiSet::Bondi);
        w.ionic_strength_molar = Some(-1.0);
        assert!(w.check().iter().any(|m| m.contains("ionic_strength")));
    }
}
