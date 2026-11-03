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

/// NMR 的规范原点选择（`options.nmr.gauge`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum NmrGauge {
    /// 每个核各自为角动量原点（Ramsey 的常规写法）。
    Nucleus,
    /// 核电荷重心为原点（「自由分子」共原规范）。
    Common,
    /// **GIAO / London 轨道**（phase-2 RS-04 的最终形态，v2.2.0 新增）。
    ///
    /// 波函数带 London 相位，所以结果（总和）**没有**「换规范原点」这个自由度：本项目的
    /// 规范检验做成「把整个分子平移到另一个**坐标**原点再算一遍」，总和必须不动，判据 < 1 ppm。
    /// 这与 [`Self::Nucleus`] / [`Self::Common`] 的语义不同（那两种是换 `L` 的原点重算），
    /// 所以在 [`NmrShieldings::method`] 里分别写成 `giao` / `nucleus-origin` / `common-origin`。
    Giao,
}

impl NmrGauge {
    /// 冻结的小写名。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nucleus => "nucleus",
            Self::Common => "common",
            Self::Giao => "giao",
        }
    }

    /// 是否走 GIAO 路径。
    ///
    /// 用方法而不是让调用方写 `== NmrGauge::Giao`：`non_exhaustive` 枚举将来再加变体时，
    /// 这个判据的语义（「带 London 相位 / 规范不变」）不会跟着漂。
    #[must_use]
    pub fn is_giao(self) -> bool {
        matches!(self, Self::Giao)
    }
}

/// NMR 屏蔽的选项（`options.nmr`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NmrOptions {
    /// 用哪个规范原点报屏蔽。
    pub gauge: NmrGauge,
    /// 是否额外报 `length_gauge_spread_ppm`（每个**检验点**各算一次，成本 ×2）。
    ///
    /// phase-2 §4.2 的门禁是「换原点重算的变化 < 1 ppm」—— 那是 GIAO 才有的性质；共原规范下
    /// 实测 1.5 ppm（O）/ 7 ppm（H），所以这个数现在的作用是如实报出差距。
    ///
    /// 对 [`NmrGauge::Giao`]，「换原点」= **把整个分子平移到另一个坐标原点再算一遍**
    /// （GIAO 里没有可以单独挪动的 `L` 原点），比较的是**总和** `σ_iso`；另外两种规范比较的是
    /// 顺磁项（抗磁项与 `L` 的原点无关）。
    pub report_gauge_spread: bool,
}

impl NmrOptions {
    /// 缺省：核中心原点、报规范变化。
    #[must_use]
    pub fn new() -> Self {
        Self {
            gauge: NmrGauge::Nucleus,
            report_gauge_spread: true,
        }
    }

    /// 改规范原点。
    #[must_use]
    pub fn with_gauge(mut self, gauge: NmrGauge) -> Self {
        self.gauge = gauge;
        self
    }

    /// 改「是否报规范变化」。
    #[must_use]
    pub fn with_gauge_spread(mut self, report: bool) -> Self {
        self.report_gauge_spread = report;
        self
    }
}

impl Default for NmrOptions {
    fn default() -> Self {
        Self::new()
    }
}
/// NMR 屏蔽结果块（K7）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct NmrShieldings {
    /// 规范/方法名。取值（冻结，v2.2.0 起三种）：
    ///
    /// * `giao` —— GIAO / London 轨道（[`NmrGauge::Giao`]；phase-2 §13 #30 的目标实现）；
    /// * `nucleus-origin` / `common-origin` —— 非 GIAO 的共原规范（`L` 的原点取该核 /
    ///   核电荷重心）。这两种不是 GIAO，`length_gauge_spread_ppm` 因此会大于 1 ppm。
    ///
    /// 写实方法名是刻意的：读者不能把「共原规范的结果」当成「GIAO 的结果」。
    pub method: String,
    /// 规范原点，单位 Bohr（GIAO 下结果对原点不该敏感——这正是
    /// [`Self::length_gauge_spread_ppm`] 检查的东西）。
    ///
    /// `giao` 时这里是**坐标原点**（London 相位里的那个绝对原点，固定 `[0, 0, 0]`）——
    /// GIAO 没有「选一个规范原点」的自由度，所以它不随 `options.nmr.gauge` 变。
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

    /// 改方法名。
    ///
    /// `new` 里的缺省是 `giao`（契约的目标），而**共原规范不是 GIAO**：非 GIAO 的两条路径报的是
    /// `nucleus-origin` / `common-origin`，`length_gauge_spread_ppm` 因此会大于 1 ppm。把方法名
    /// 写实，读者才不会把「共原规范的结果」当成「GIAO 的结果」（v2.2.0 起 `NmrGauge::Giao`
    /// 走真的 GIAO，方法名就是缺省的 `giao`）。
    #[must_use]
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    /// 报出「换规范原点重算时各向同性屏蔽的最大变化」（ppm）。
    ///
    /// 这个数不是装饰：phase-2 §4.2 的门禁是 `< 1 ppm`，而只有 GIAO 能满足它 —— 共原规范下
    /// 实测 1.5 ppm（O）/ 7 ppm（H）。把差距报出来，比在文档里写「GIAO 还没做」有用。
    #[must_use]
    pub fn with_length_gauge_spread(mut self, spread_ppm: f64) -> Self {
        self.length_gauge_spread_ppm = Some(spread_ppm);
        self
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

/// 一个核的各向同性超精细耦合（Fermi 接触项，phase-2 `ST-05`）。
///
/// 单位：`spin_density_au` 是 `a0^-3`；`a_iso_mhz` / `a_iso_gauss` 是同一个量的两个单位
/// （`1 G = 2.802_49 MHz`，自由电子），两个都落盘是为了省掉读的人自己换算。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct HyperfineCoupling {
    /// 核（原子）下标，与 `molecule.atoms` 同序。
    pub nucleus_index: usize,
    /// 元素符号。
    pub symbol: String,
    /// 核位置上的自旋密度 `Σ (P_α − P_β)_{μν} φ_μ(R_K) φ_ν(R_K)`，单位 `a0^-3`。
    pub spin_density_au: f64,
    /// 各向同性耦合，MHz。
    pub a_iso_mhz: f64,
    /// 各向同性耦合，Gauss。
    pub a_iso_gauss: f64,
}

impl HyperfineCoupling {
    /// 构造一个核的耦合。
    #[must_use]
    pub fn new(
        nucleus_index: usize,
        symbol: impl Into<String>,
        spin_density_au: f64,
        a_iso_mhz: f64,
        a_iso_gauss: f64,
    ) -> Self {
        Self {
            nucleus_index,
            symbol: symbol.into(),
            spin_density_au,
            a_iso_mhz,
            a_iso_gauss,
        }
    }
}

/// EPR 结果块（phase-2 `ST-05`）：开壳层体系在核位置上的自旋密度与各向同性超精细耦合。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EprResult {
    /// 方法说明（`uhf` / `uks`），落盘是为了让读的人知道自旋密度来自哪种参考。
    pub method: String,
    /// 每个核的耦合（顺序 = `molecule.atoms`）。
    pub nuclei: Vec<HyperfineCoupling>,
}

impl EprResult {
    /// 构造 EPR 块。
    #[must_use]
    pub fn new(method: impl Into<String>, nuclei: Vec<HyperfineCoupling>) -> Self {
        Self {
            method: method.into(),
            nuclei,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExcitedStateMethod, ExcitedStates, Multipoles, NmrGauge, NmrOptions, NmrReference,
        NmrShieldings, NucleusShielding, Polarizability, SumRules, Transition,
        TransitionContribution,
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

    /// RS-04 的最后一刀是**加性**的：`NmrGauge::Giao` 只是多一个变体。
    ///
    /// 三条一起钉住：(1) 旧的两个变体的名字/缺省一个字节没变（`NmrOptions::new()` 仍是
    /// `nucleus`）；(2) 新变体序列化成 `giao`（与 `NmrShieldings::method` 的 `giao` 同名，
    /// 但**语义不同**：一个是输入端的规范选择，一个是输出端的方法名）；(3) 旧 `input.yaml`
    /// 里 `gauge: common` 仍然解析成同一个变体（不重命名、不改序）。
    #[test]
    fn nmr_gauge_giao_is_an_additive_variant() {
        // (1) 缺省不变。
        let options = NmrOptions::new();
        assert_eq!(options.gauge, NmrGauge::Nucleus);
        assert_eq!(options.gauge.as_str(), "nucleus");
        assert!(options.report_gauge_spread);

        // 旧的两个变体的冻结小写名。
        assert_eq!(NmrGauge::Common.as_str(), "common");
        assert_eq!(NmrGauge::Giao.as_str(), "giao");
        assert!(!NmrGauge::Nucleus.is_giao());
        assert!(!NmrGauge::Common.is_giao());
        assert!(NmrGauge::Giao.is_giao());

        // (2) 新变体的 serde 形态与 `gauge_origin_bohr` 的读写往返。
        let giao = options.clone().with_gauge(NmrGauge::Giao);
        let json = serde_json::to_string(&giao).expect("serialize");
        assert!(json.contains("\"gauge\":\"giao\""), "{json}");
        let back: NmrOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, giao);
        assert_eq!(back.gauge, NmrGauge::Giao);

        // (3) 旧输入的文本仍然解析成旧变体。
        let old = r#"{"gauge":"common","report_gauge_spread":false}"#;
        let parsed: NmrOptions = serde_json::from_str(old).expect("old input still parses");
        assert_eq!(parsed.gauge, NmrGauge::Common);
        assert!(!parsed.report_gauge_spread);
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
