//! X-01：`FunctionalTerm` 与第一批 LDA 项（Slater 交换 + VWN5 相关）。
//!
//! # 设计（project.md §6.4 / §8.3）
//!
//! - 数据侧只放**参数与组合**，数学形式在代码侧。参数从 `qwb-data/functional/*.yaml`
//!   的 `parameters` 读进来，键名由数据文件声明（`slater_alpha`、`vwn5_p_x0` …）。
//! - 接口是**批量切片**的（`&[f64]` 进、`&mut [f64]` 出），不是逐点标量调用：
//!   网格上有上百万点，逐点虚调用会把开销吃光（§6.4 X-01 的验收要求）。
//! - 提供 `energy_density`（每电子 ε）与 `potential_derivatives`（`dε/dρ`、`dε/dσ`），
//!   而不是直接给势：SCF 需要 `v = ε + ρ·dε/dρ`，把组合留给上层，泛函只负责数学。
//! - **导数必须是解析的**：X-02 用有限差分校验它（`tests/x02_derivatives.rs`），
//!   这是 G3 物理门禁的一部分。
//!
//! # 自旋
//!
//! 第一阶段先做 RKS（闭壳层）。`spin_scaling()` 给出该泛函的自旋标度指数，
//! UKS 在 S 轨落地时按它做 `E[ρ↑,ρ↓]` 的插值（VWN5 的 `α_c` 参数已存进数据，
//! 就是为这一步准备的）。

#![deny(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeMap;

use thiserror::Error;

pub mod gga;
pub mod gga_spin;

/// `(3/π)^{1/3}` —— 用表达式算，不手抄字面量（上一版手抄错成 0.90036，
/// 被 `slater_exchange_matches_the_analytic_formula` 立刻抓到）。
/// 低密度截断阈值（e/Bohr³）：低于它 VWN5 的相关势按平坦处理。
pub const LOW_DENSITY_FLOOR: f64 = 1.0e-12;

pub(crate) fn three_over_pi_cbrt() -> f64 {
    (3.0 / std::f64::consts::PI).cbrt()
}

/// 该泛函项需要哪些密度量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Requirements {
    /// 需要密度 ρ。
    pub rho: bool,
    /// 需要 `σ = |∇ρ|²`。
    pub sigma: bool,
    /// 需要 `∇²ρ`（LYP 一类用；meta-GGA 也常一起给）。
    pub laplacian: bool,
    /// 需要动能密度 `τ = ½ Σ_μν P_μν ∇φ_μ·∇φ_ν`（meta-GGA 用，FX-01）。
    pub kinetic_density: bool,
}

impl Requirements {
    /// 纯 LDA：只要 ρ。
    #[must_use]
    pub fn lda() -> Self {
        Self {
            rho: true,
            sigma: false,
            laplacian: false,
            kinetic_density: false,
        }
    }

    /// GGA：ρ 与 σ。
    #[must_use]
    pub fn gga() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: false,
            kinetic_density: false,
        }
    }

    /// 需要 `∇²ρ` 的项（LYP 一类）：ρ、σ、`∇²ρ` 全要。
    #[must_use]
    pub fn laplacian_gga() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: true,
            kinetic_density: false,
        }
    }

    /// meta-GGA（只依赖 τ，不依赖 `∇²ρ`）：ρ、σ、τ。
    ///
    /// 为什么默认不要 `∇²ρ`：M06-L/M06-2X 这些主流 meta-GGA 的变量是
    /// `(ρ, σ, τ)`，`∇²ρ` 只在 LYP 那样的项里出现。要两样都要，用
    /// [`Self::meta_gga_laplacian`]。
    #[must_use]
    pub fn meta_gga() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: false,
            kinetic_density: true,
        }
    }

    /// `(ρ, σ, ∇²ρ, τ)` 全要的项。
    #[must_use]
    pub fn meta_gga_laplacian() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: true,
            kinetic_density: true,
        }
    }
}

/// 一批网格点的密度量（批量接口，避免逐点虚调用）。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DensityBatch<'a> {
    /// 电子密度 ρ（`e/Bohr³`），长度 = `points`。
    pub rho: &'a [f64],
    /// `σ = |∇ρ|²`；LDA 项可以给空切片。
    pub sigma: &'a [f64],
    /// `∇²ρ`；只有声明 `requires.laplacian` 的项需要（LYP 一类）。
    pub laplacian: &'a [f64],
    /// 动能密度 `τ = ½ Σ_μν P_μν ∇φ_μ·∇φ_ν`（`E_h`/Bohr³）；
    /// 只有声明 `requires.kinetic_density` 的项需要（meta-GGA，FX-01）。
    pub tau: &'a [f64],
}

impl<'a> DensityBatch<'a> {
    /// 用 ρ 构造（LDA 用；其余通道为空）。
    #[must_use]
    pub fn from_rho(rho: &'a [f64]) -> Self {
        Self {
            rho,
            sigma: &[],
            laplacian: &[],
            tau: &[],
        }
    }

    /// 用 ρ 与 σ 构造（GGA 用；不看 `laplacian`/`tau`）。
    #[must_use]
    pub fn new(rho: &'a [f64], sigma: &'a [f64]) -> Self {
        Self {
            rho,
            sigma,
            laplacian: &[],
            tau: &[],
        }
    }

    /// 用 ρ、σ、∇²ρ 构造（LYP 用；不看 `tau`）。
    #[must_use]
    pub fn with_laplacian(rho: &'a [f64], sigma: &'a [f64], laplacian: &'a [f64]) -> Self {
        Self {
            rho,
            sigma,
            laplacian,
            tau: &[],
        }
    }

    /// 用 ρ、σ、τ 构造（meta-GGA 用；`laplacian` 为空）。
    ///
    /// 只要 `kinetic_density` 的项（M06-L/M06-2X 这一族）用这个；
    /// 同时要 `∇²ρ` 的项用 [`Self::with_all`]。
    #[must_use]
    pub fn with_tau(rho: &'a [f64], sigma: &'a [f64], tau: &'a [f64]) -> Self {
        Self {
            rho,
            sigma,
            laplacian: &[],
            tau,
        }
    }

    /// 四个通道都给（元 meta-GGA / LYP + τ）。
    #[must_use]
    pub fn with_all(
        rho: &'a [f64],
        sigma: &'a [f64],
        laplacian: &'a [f64],
        tau: &'a [f64],
    ) -> Self {
        Self {
            rho,
            sigma,
            laplacian,
            tau,
        }
    }

    /// 点数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.rho.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rho.is_empty()
    }

    /// `τ`（空切片时视为 0），单位 `E_h/Bohr³`。
    #[must_use]
    pub fn tau_at(&self, index: usize) -> f64 {
        self.tau.get(index).copied().unwrap_or(0.0)
    }

    /// 是否带 τ 通道。
    #[must_use]
    pub fn has_tau(&self) -> bool {
        !self.tau.is_empty()
    }
}

/// XC 错误。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum XcError {
    /// 数据里出现了本构建不认识的组件 id。
    #[error("unknown functional component `{0}` (first phase: LdaSlaterX, LdaSvwn5)")]
    UnknownTerm(String),
    /// 参数缺失。
    #[error("functional component `{component}` needs parameter `{parameter}`")]
    MissingParameter {
        /// 组件 id。
        component: String,
        /// 参数名。
        parameter: String,
    },
    /// 输入切片长度不一致。
    #[error("density batch shape mismatch: {0}")]
    Shape(String),
    /// 该泛函项没有自旋分辨的实现（缺少铁磁/α_c 参数集）。
    #[error("spin-resolved evaluation is not available: {0}")]
    SpinUnsupported(String),
}

/// 一个交换相关项。
///
/// 实现方必须 `Send + Sync`：SCF 会在网格上并行求值（G7）。
pub trait FunctionalTerm: Send + Sync {
    /// 组件 id（与数据 `functional.yaml` 的 `components[].id` 一致）。
    fn id(&self) -> &'static str;

    /// 需要哪些密度量。
    fn requires(&self) -> Requirements;

    /// 自旋标度指数（`E[ρ↑,ρ↓]` 的插值用）；闭壳层下不影响结果。
    fn spin_scaling(&self) -> f64;

    /// 每电子能量密度 `ε(ρ,σ)`，写进 `out`。
    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]);

    /// 解析导数：`dε/dρ` 与 `dε/dσ`，分别写进 `d_rho` 与 `d_sigma`。
    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    );

    /// 三通道版本：额外给出 `dε/d∇²ρ`。
    ///
    /// 默认实现把 `d_laplacian` 全部置零并转发给 [`Self::potential_derivatives`]，
    /// 所以只有真正依赖 `∇²ρ` 的项（LYP 一类）才需要覆盖它。
    ///
    /// **为什么必须有这个通道**：LYP 的 ε 依赖 `(ρ, σ, ∇²ρ)`，它的变分势里有一项
    /// `∇²(ρ·dε/d∇²ρ)`。若只实现 ρ、σ 两个通道，**能量表达式看着对，但 v_xc 会缺一整项**，
    /// SCF 会收敛到错的密度上——而且这种错误从能量数值上几乎看不出来。
    fn potential_derivatives_laplacian(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
    ) {
        self.potential_derivatives(batch, d_rho, d_sigma);
        d_laplacian.fill(0.0);
    }

    /// 四通道版本：再加 `dε/dτ`（FX-01）。
    ///
    /// 默认实现把 `d_tau` 全部置零并转发给
    /// [`Self::potential_derivatives_laplacian`]，所以只有真正依赖 τ 的项
    /// （meta-GGA）才需要覆盖它 —— 和 `∇²ρ` 通道的做法一致。
    ///
    /// **组装方式不同**：`∂τ/∂P_μν = ½∇φ_μ·∇φ_ν` 是**直接**的，不需要分部积分，
    /// 所以矩阵元里是 `½(dε/dτ)·∇φ_μ·∇φ_ν`（见 `qwb-scf::rks::evaluate_xc_cached`）。
    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        self.potential_derivatives_laplacian(batch, d_rho, d_sigma, d_laplacian);
        d_tau.fill(0.0);
    }

    /// 这一项能否在自旋极化（UKS）路径下求值。
    ///
    /// 默认 `false`：自旋版本必须显式实现 [`Self::energy_density_spin`] 与
    /// [`Self::potential_spin`]，否则 SCF 会拿到一个静默的假势。具体类型上的
    /// `LdaSvwn5::supports_spin` 是"数据里齐不齐参数"，这里是"这个组件有没有自旋实现"，
    /// 两者都要为真才能跑 UKS。
    fn spin_supported(&self) -> bool {
        false
    }
    /// 一次求出自旋分辨 XC 的全部六个量（能量密度、两个势、三个 σ 导数）。
    ///
    /// **写入语义**：被调方把 `out` 的六个切片全部覆盖（不累加），长度必须等于
    /// `batch.len()`；LDA 项要把三个 σ 导数填 0（而不是留着上一次的值）。
    ///
    /// 为什么合成一个方法而不是给六个独立的：能量与势**必须来自同一次求值**，
    /// 分开传切片是"能量用一个公式、势用另一个公式"这类错误最容易藏身的地方。
    ///
    /// 默认实现报错而不是返回零 —— 零势会让 SCF 安静地收敛到错的密度上。
    fn evaluate_spin(
        &self,
        _batch: &SpinDensityBatch<'_>,
        _out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        Err(XcError::SpinUnsupported(format!(
            "{} has no spin-resolved evaluation (UKS needs a spin implementation)",
            self.id()
        )))
    }
}

pub(crate) fn check_shape(
    batch: &DensityBatch<'_>,
    out: &[f64],
    what: &str,
) -> Result<(), XcError> {
    if out.len() != batch.len() {
        return Err(XcError::Shape(format!(
            "{}: output {} vs input {}",
            what,
            out.len(),
            batch.len()
        )));
    }
    for (label, channel) in [
        ("sigma", batch.sigma),
        ("laplacian", batch.laplacian),
        ("tau", batch.tau),
    ] {
        if !channel.is_empty() && channel.len() != batch.len() {
            return Err(XcError::Shape(format!(
                "{}: {label} {} vs rho {}",
                what,
                channel.len(),
                batch.rho.len()
            )));
        }
    }
    Ok(())
}

// ───────────────────────── Slater 交换 ─────────────────────────

/// Slater（LDA）交换：`ε_x = −(9/8)·α·(3/π)^{1/3}·ρ^{1/3}`，`α = 2/3` 时即标准 LDA-X。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LdaSlaterX {
    /// `α`（来自数据的 `slater_alpha`）。
    pub alpha: f64,
}

impl LdaSlaterX {
    /// 由 `α` 构造。
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// `ε_x(ρ)`。
    #[must_use]
    pub fn eps(rho: f64) -> f64 {
        -0.75 * three_over_pi_cbrt() * rho.cbrt()
    }
}

impl FunctionalTerm for LdaSlaterX {
    fn id(&self) -> &'static str {
        "LdaSlaterX"
    }

    fn requires(&self) -> Requirements {
        Requirements::lda()
    }

    fn spin_scaling(&self) -> f64 {
        1.0 / 3.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "LdaSlaterX::energy_density").expect("shape");
        let scale = 1.125 * self.alpha * three_over_pi_cbrt(); // (9/8)·α·(3/π)^{1/3}
        for (slot, rho) in out.iter_mut().zip(batch.rho) {
            *slot = -scale * rho.max(0.0).cbrt();
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "LdaSlaterX::d_rho").expect("shape");
        check_shape(batch, d_sigma, "LdaSlaterX::d_sigma").expect("shape");
        let scale = 1.125 * self.alpha * three_over_pi_cbrt();
        for (slot, rho) in d_rho.iter_mut().zip(batch.rho) {
            let rho = rho.max(1e-300);
            *slot = -scale * rho.cbrt() / (3.0 * rho);
        }
        for slot in d_sigma.iter_mut() {
            *slot = 0.0;
        }
    }

    fn spin_supported(&self) -> bool {
        true
    }

    fn evaluate_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        // 交换的自旋标度是精确的：E_x = ½[E_x(2ρ_α) + E_x(2ρ_β)]。
        exchange_spin_scaling(self, batch, out)
    }
}

// ───────────────────────── VWN5 相关 ─────────────────────────

/// VWN（Vosko–Wilk–Nusair）相关能，顺磁通道。
///
/// **命名注意**：`svwn5.yaml` 里的参数集其实是 libxc 的 **VWN RPA**
/// （`LDA_C_VWN_RPA`），不是社区通常叫的 VWN5（libxc `LDA_C_VWN`）：两者在
/// rs ≈ 0.7 处差 0.02 Ha/e，逐点对照见 `tests/sc01b_spin_xc_pointwise.rs`，
/// 来龙去脉见 `docs/choices.md` 五十。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LdaSvwn5 {
    /// 顺磁参数 `(A, x0, b, c)`。
    pub para: (f64, f64, f64, f64),
    /// 铁磁参数 `(A, x0, b, c)`；`None` = 只支持自旋非极化（闭壳层）。
    ///
    /// VWN5 的自旋插值 `ε_c = ε_c^P + (ε_c^F − ε_c^P) f(ζ)` 需要这一组参数，
    /// 数据里早就有（`vwn5_f_*`），只是闭壳层路径没读它。
    pub ferro: Option<(f64, f64, f64, f64)>,
    /// 自旋插值的 `α_c` 参数集 `(A, x0, b, c)`（`vwn5_a_*`）；VWN5 用它算 `f(ζ)` 的分母。
    pub alpha: Option<(f64, f64, f64, f64)>,
}

/// VWN 的 `G(x)`（`ε_c = G(x)`，`x = √r_s`）。
pub(crate) fn vwn_g(x: f64, a: f64, x0: f64, b: f64, c: f64) -> f64 {
    let big_x = |t: f64| t * t + b * t + c;
    let x0sq = x0 * x0 + b * x0 + c;
    let q = (4.0 * c - b * b).sqrt();
    a * ((x * x / big_x(x)).ln() - (b * x0 / x0sq) * (((x - x0) * (x - x0)) / big_x(x)).ln()
        + (2.0 * b * (c - x0 * x0)) / (x0sq * q) * (q / (2.0 * x + b)).atan())
}

/// `dG/dx`（解析式，X-02 会用有限差分校验它）。
pub(crate) fn vwn_g_prime(x: f64, a: f64, x0: f64, b: f64, c: f64) -> f64 {
    let big_x = |t: f64| t * t + b * t + c;
    let x0sq = x0 * x0 + b * x0 + c;
    // 三项分别求导后合并；最后一个 arctan 项的导数化简为 −Q/(2X)。
    let term1 = 2.0 / x;
    let term2 = -(2.0 * x + b) / big_x(x);
    let term3 = -(b * x0 / x0sq) * (2.0 / (x - x0) - (2.0 * x + b) / big_x(x));
    let term4 = (2.0 * b * (c - x0 * x0)) / (x0sq * (4.0 * c - b * b).sqrt())
        * (-(4.0 * c - b * b).sqrt() / (2.0 * big_x(x)));
    a * (term1 + term2 + term3 + term4)
}

impl LdaSvwn5 {
    /// 由顺磁参数构造（自旋插值不可用，只支持闭壳层）。
    #[must_use]
    pub fn new(para: (f64, f64, f64, f64)) -> Self {
        Self {
            para,
            ferro: None,
            alpha: None,
        }
    }

    /// 补上铁磁与 α_c 参数集（打开自旋插值）。
    #[must_use]
    pub fn with_spin_parameters(
        mut self,
        ferro: (f64, f64, f64, f64),
        alpha: (f64, f64, f64, f64),
    ) -> Self {
        self.ferro = Some(ferro);
        self.alpha = Some(alpha);
        self
    }

    /// 是否支持自旋极化（UKS）。
    #[must_use]
    pub fn supports_spin(&self) -> bool {
        self.ferro.is_some() && self.alpha.is_some()
    }
}

impl FunctionalTerm for LdaSvwn5 {
    fn id(&self) -> &'static str {
        "LdaSvwn5"
    }

    fn requires(&self) -> Requirements {
        Requirements::lda()
    }

    fn spin_scaling(&self) -> f64 {
        // VWN5 用自旋插值（`f(ξ)` + `α_c`），不是简单幂标度；这里给出 1 表示
        // "由实现内部按 α_c 处理"，UKS 落地时读数据里的 `vwn5_a_*` 参数。
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "LdaSvwn5::energy_density").expect("shape");
        let (a, x0, b, c) = self.para;
        for (slot, rho) in out.iter_mut().zip(batch.rho) {
            let rho = rho.max(1e-300);
            let rs = (3.0 / (4.0 * std::f64::consts::PI * rho)).cbrt();
            *slot = vwn_g(rs.sqrt(), a, x0, b, c);
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "LdaSvwn5::d_rho").expect("shape");
        check_shape(batch, d_sigma, "LdaSvwn5::d_sigma").expect("shape");
        let (a, x0, b, c) = self.para;
        for (slot, rho) in d_rho.iter_mut().zip(batch.rho) {
            // 低密度截断（§8.3 的 stability）：dε/dρ 里有 1/ρ，ρ→0 时解析式会溢出到
            // −inf；消费方算 ρ·dε/dρ 就成了 0×(−inf)=NaN。密度低于阈值的区域相关能
            // 已经平坦，导数取 0 既物理又数值安全。
            if *rho <= LOW_DENSITY_FLOOR {
                *slot = 0.0;
                continue;
            }
            let rho = *rho;
            let rs = (3.0 / (4.0 * std::f64::consts::PI * rho)).cbrt();
            let x = rs.sqrt();
            let derivative = vwn_g_prime(x, a, x0, b, c) * (-x / (6.0 * rho));
            *slot = if derivative.is_finite() {
                derivative
            } else {
                0.0
            };
        }
        for slot in d_sigma.iter_mut() {
            *slot = 0.0;
        }
    }

    fn spin_supported(&self) -> bool {
        // 数据里三组参数齐全才算支持（`vwn5_f_*` / `vwn5_a_*`）。
        self.supports_spin()
    }

    fn evaluate_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        check_spin_output(batch.len(), out, "LdaSvwn5::evaluate_spin")?;
        self.require_spin()?;
        for index in 0..batch.len() {
            let alpha = batch.rho_alpha[index].max(0.0);
            let beta = batch.rho_beta[index].max(0.0);
            let (eps, d_alpha, d_beta) = self.spin_energy_at(alpha, beta);
            let total = alpha + beta;
            // 每体积相关能密度与 ∂e_c/∂ρ_σ（VWN 是纯 LDA，σ 通道恒为 0）。
            out.energy[index] = total * eps;
            out.v_alpha[index] = eps + total * d_alpha;
            out.v_beta[index] = eps + total * d_beta;
        }
        out.d_sigma_alpha.fill(0.0);
        out.d_sigma_beta.fill(0.0);
        out.d_sigma_alpha_beta.fill(0.0);
        Ok(())
    }
}

// ─────────────────── 自旋分辨 XC（UKS 的 XC 侧，SC-01b）───────────────────

/// 自旋分辨的密度批次（每个点一对 `ρ_α`、`ρ_β` 与三个 σ）。
///
/// 闭壳层路径用 [`DensityBatch`]（自旋求和）；UKS 需要每个自旋通道各自的密度与梯度，
/// 所以单独一个类型。LDA 项不看 σ，可以给空切片。
///
/// ```text
/// σ_αα = |∇ρ_α|²、σ_ββ = |∇ρ_β|²、σ_αβ = ∇ρ_α·∇ρ_β
/// ```
///
/// 三个 σ 不是独立的：`σ_αβ² ≤ σ_αα σ_ββ`（Cauchy–Schwarz），消费方构造时应当保证。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SpinDensityBatch<'a> {
    /// `ρ_α`（e/Bohr³）。
    pub rho_alpha: &'a [f64],
    /// `ρ_β`（e/Bohr³）。
    pub rho_beta: &'a [f64],
    /// `σ_αα = |∇ρ_α|²`；LDA 项给空切片。
    pub sigma_alpha: &'a [f64],
    /// `σ_ββ = |∇ρ_β|²`；LDA 项给空切片。
    pub sigma_beta: &'a [f64],
    /// `σ_αβ = ∇ρ_α·∇ρ_β`；LDA 项给空切片。
    pub sigma_alpha_beta: &'a [f64],
}

impl<'a> SpinDensityBatch<'a> {
    /// 只要密度（LDA 用；三个 σ 为空）。
    ///
    /// # Panics
    ///
    /// 两个切片长度不同（调用方应先保证网格一致）。
    #[must_use]
    pub fn new(rho_alpha: &'a [f64], rho_beta: &'a [f64]) -> Self {
        assert_eq!(
            rho_alpha.len(),
            rho_beta.len(),
            "spin densities must have the same length"
        );
        Self {
            rho_alpha,
            rho_beta,
            sigma_alpha: &[],
            sigma_beta: &[],
            sigma_alpha_beta: &[],
        }
    }

    /// 密度 + 三个 σ（GGA 用）。
    ///
    /// # Panics
    ///
    /// 五个切片长度不一致。
    #[must_use]
    pub fn with_sigma(
        rho_alpha: &'a [f64],
        rho_beta: &'a [f64],
        sigma_alpha: &'a [f64],
        sigma_beta: &'a [f64],
        sigma_alpha_beta: &'a [f64],
    ) -> Self {
        let points = rho_alpha.len();
        assert_eq!(rho_beta.len(), points, "spin densities must match");
        assert_eq!(sigma_alpha.len(), points, "sigma_alpha must match");
        assert_eq!(sigma_beta.len(), points, "sigma_beta must match");
        assert_eq!(
            sigma_alpha_beta.len(),
            points,
            "sigma_alpha_beta must match"
        );
        Self {
            rho_alpha,
            rho_beta,
            sigma_alpha,
            sigma_beta,
            sigma_alpha_beta,
        }
    }

    /// 点数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.rho_alpha.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rho_alpha.is_empty()
    }

    /// 是否带 σ 通道。
    #[must_use]
    pub fn has_sigma(&self) -> bool {
        !self.sigma_alpha.is_empty()
    }

    /// `σ_αα`（空切片时视为 0）。
    #[must_use]
    pub fn sigma_alpha_at(&self, index: usize) -> f64 {
        self.sigma_alpha.get(index).copied().unwrap_or(0.0)
    }

    /// `σ_ββ`（空切片时视为 0）。
    #[must_use]
    pub fn sigma_beta_at(&self, index: usize) -> f64 {
        self.sigma_beta.get(index).copied().unwrap_or(0.0)
    }

    /// `σ_αβ`（空切片时视为 0）。
    #[must_use]
    pub fn sigma_alpha_beta_at(&self, index: usize) -> f64 {
        self.sigma_alpha_beta.get(index).copied().unwrap_or(0.0)
    }

    /// 自旋极化率 `ζ = (ρ_α − ρ_β)/ρ`（`ρ ≈ 0` 时返回 0）。
    #[must_use]
    pub fn zeta(&self, index: usize) -> f64 {
        let alpha = self.rho_alpha[index].max(0.0);
        let beta = self.rho_beta[index].max(0.0);
        let total = alpha + beta;
        if total <= LOW_DENSITY_FLOOR {
            0.0
        } else {
            ((alpha - beta) / total).clamp(-1.0, 1.0)
        }
    }

    /// `|∇ρ|² = σ_αα + σ_ββ + 2σ_αβ`（总密度的梯度平方）。
    #[must_use]
    pub fn total_sigma(&self, index: usize) -> f64 {
        self.sigma_alpha_at(index)
            + self.sigma_beta_at(index)
            + 2.0 * self.sigma_alpha_beta_at(index)
    }
}

/// 自旋分辨 XC 的六个输出缓冲（全部**写入**，长度都必须等于点数）。
#[derive(Debug)]
#[non_exhaustive]
pub struct SpinXcOutput<'a> {
    /// 每体积能量密度 `e`（`E = ∫ e dV`）。
    pub energy: &'a mut [f64],
    /// `v_α = ∂e/∂ρ_α`。
    pub v_alpha: &'a mut [f64],
    /// `v_β = ∂e/∂ρ_β`。
    pub v_beta: &'a mut [f64],
    /// `∂e/∂σ_αα`。
    pub d_sigma_alpha: &'a mut [f64],
    /// `∂e/∂σ_ββ`。
    pub d_sigma_beta: &'a mut [f64],
    /// `∂e/∂σ_αβ`。
    pub d_sigma_alpha_beta: &'a mut [f64],
}

impl<'a> SpinXcOutput<'a> {
    /// 由六个可变切片构造。
    #[must_use]
    pub fn new(
        energy: &'a mut [f64],
        v_alpha: &'a mut [f64],
        v_beta: &'a mut [f64],
        d_sigma_alpha: &'a mut [f64],
        d_sigma_beta: &'a mut [f64],
        d_sigma_alpha_beta: &'a mut [f64],
    ) -> Self {
        Self {
            energy,
            v_alpha,
            v_beta,
            d_sigma_alpha,
            d_sigma_beta,
            d_sigma_alpha_beta,
        }
    }

    /// 六个切片的长度是否都等于 `points`。
    fn matching(points: usize) -> impl Fn(&Self) -> bool {
        move |out: &Self| {
            out.energy.len() == points
                && out.v_alpha.len() == points
                && out.v_beta.len() == points
                && out.d_sigma_alpha.len() == points
                && out.d_sigma_beta.len() == points
                && out.d_sigma_alpha_beta.len() == points
        }
    }

    /// 全部清零。
    fn clear(&mut self) {
        self.energy.fill(0.0);
        self.v_alpha.fill(0.0);
        self.v_beta.fill(0.0);
        self.d_sigma_alpha.fill(0.0);
        self.d_sigma_beta.fill(0.0);
        self.d_sigma_alpha_beta.fill(0.0);
    }
}

/// 六个切片长度一致性的统一检查。
pub(crate) fn check_spin_output(
    points: usize,
    out: &SpinXcOutput<'_>,
    what: &str,
) -> Result<(), XcError> {
    if SpinXcOutput::matching(points)(out) {
        Ok(())
    } else {
        Err(XcError::Shape(format!(
            "{what}: outputs ({}, {}, {}, {}, {}, {}) vs input {points}",
            out.energy.len(),
            out.v_alpha.len(),
            out.v_beta.len(),
            out.d_sigma_alpha.len(),
            out.d_sigma_beta.len(),
            out.d_sigma_alpha_beta.len()
        )))
    }
}

/// 交换项的自旋标度：`e_x = ρ_α·ε(2ρ_α, 4σ_αα) + ρ_β·ε(2ρ_β, 4σ_ββ)`。
///
/// `E_x = ½[E_x(2ρ_α) + E_x(2ρ_β)]` 对交换泛函是**精确**的（Slater、B88、PBE-X 都适用），
/// 只要把每个通道看成"密度翻倍的全极化子系统"，闭壳层实现就能原样复用。
///
/// **相关项不能用它**（要靠 ζ 插值，见 [`LdaSvwn5::spin_energy_at`] 与
/// `gga::PbeCorrelation` 的自旋实现），因此这不是 trait 默认实现，而是必须显式调用的
/// 辅助函数 —— 选错了在能量上几乎看不出来。
///
/// 六个输出全部按"写入"语义覆盖。
pub(crate) fn exchange_spin_scaling(
    term: &dyn FunctionalTerm,
    batch: &SpinDensityBatch<'_>,
    out: &mut SpinXcOutput<'_>,
) -> Result<(), XcError> {
    let points = batch.len();
    check_spin_output(points, out, "exchange_spin_scaling")?;
    out.clear();

    let needs_sigma = term.requires().sigma;
    let mut doubled = vec![0.0_f64; points];
    let mut quadrupled = vec![0.0_f64; points];
    let mut eps = vec![0.0_f64; points];
    let mut d_eps = vec![0.0_f64; points];
    let mut d_sigma = vec![0.0_f64; points];
    for channel in 0..2 {
        let (source, sigma_source) = if channel == 0 {
            (batch.rho_alpha, batch.sigma_alpha)
        } else {
            (batch.rho_beta, batch.sigma_beta)
        };
        for index in 0..points {
            doubled[index] = 2.0 * source[index].max(0.0);
            quadrupled[index] = 4.0 * sigma_source.get(index).copied().unwrap_or(0.0).max(0.0);
        }
        let closed = if needs_sigma {
            DensityBatch::new(&doubled, &quadrupled)
        } else {
            DensityBatch::from_rho(&doubled)
        };
        term.energy_density(&closed, &mut eps);
        term.potential_derivatives(&closed, &mut d_eps, &mut d_sigma);
        for index in 0..points {
            let rho = source[index].max(0.0);
            // 该通道没有电子：能量、势、σ 导数都必须**严格为 0**。
            // 不能靠"乘 0"来凑：闭壳层实现对 ρ 有下限（`LOW_DENSITY_FLOOR`），
            // ε(下限) ≈ −7e-5，而 v_σ = ε + 2ρ_σ ε_ρ 在 ρ_σ = 0 时会退化成 ε ≠ 0
            // —— 实测就是这里漏了守卫，GGA_X_PBE 的 v_β 在 ρ_β = 0 处偏了 7.4e-5。
            if rho <= 0.0 {
                continue;
            }
            // 每体积能量密度：ρ_σ · ε(2ρ_σ, 4σ_σσ)。
            out.energy[index] += rho * eps[index];
            // v_σ = ε(2ρ_σ) + 2ρ_σ·ε_ρ(2ρ_σ)，即闭壳层 v(2ρ_σ)。
            let potential = eps[index] + 2.0 * rho * d_eps[index];
            if channel == 0 {
                out.v_alpha[index] += potential;
            } else {
                out.v_beta[index] += potential;
            }
            if needs_sigma {
                // ∂e/∂σ_σσ = ρ_σ · ε_σ(2ρ_σ, 4σ_σσ) · 4；σ_αβ 不进交换项。
                let derivative = 4.0 * rho * d_sigma[index];
                if channel == 0 {
                    out.d_sigma_alpha[index] += derivative;
                } else {
                    out.d_sigma_beta[index] += derivative;
                }
            }
        }
    }
    Ok(())
}

impl LdaSvwn5 {
    /// `f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2)`（VWN 的自旋插值因子）。
    ///
    /// 注意：**VWN 用的是两端插值** `ε = ε^P + (ε^F − ε^P)·f`，
    /// 与 PW92 的 `ε^P(1 − f ζ⁴) + ε^F f ζ⁴ + (α_c/f'')(1 − ζ⁴)f` 不是一个式子。
    /// 本项目 `svwn5.yaml` 的参数集是 libxc 的 `lda_c_vwn_rpa`（VWN RPA），
    /// 逐点对照见 `tests/sc01b_spin_xc_pointwise.rs`（差 < 1e-15）。
    #[must_use]
    pub fn interpolation_factor(&self, zeta: f64) -> f64 {
        let zeta = zeta.clamp(-1.0, 1.0);
        let numerator = (1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0;
        numerator / (2.0_f64.powf(4.0 / 3.0) - 2.0)
    }

    /// `f'(ζ)`。
    #[must_use]
    pub fn interpolation_factor_derivative(&self, zeta: f64) -> f64 {
        // 导数在 ζ = ±1 有限：f′(1) = (4/3)·2^{1/3}/(2^{4/3} − 2)。
        let zeta = zeta.clamp(-1.0, 1.0);
        let numerator = (1.0 + zeta).cbrt() - (1.0 - zeta).cbrt();
        (4.0 / 3.0) * numerator / (2.0_f64.powf(4.0 / 3.0) - 2.0)
    }

    /// 有一个点上的 `(ε_c, dε_c/dρ_α, dε_c/dρ_β)`。
    ///
    /// `ε_c = ε_c^P(r_s) + [ε_c^F(r_s) − ε_c^P(r_s)] · f(ζ)`；消费方用
    /// `v_σ = ε_c + ρ · dε_c/dρ_σ`（`ρ = ρ_α + ρ_β`）。
    #[must_use]
    pub fn spin_energy_at(&self, rho_alpha: f64, rho_beta: f64) -> (f64, f64, f64) {
        let (a_p, x0_p, b_p, c_p) = self.para;
        let (a_f, x0_f, b_f, c_f) = self.ferro.unwrap_or(self.para);
        let alpha = rho_alpha.max(0.0);
        let beta = rho_beta.max(0.0);
        let total = (alpha + beta).max(1.0e-300);
        if total <= LOW_DENSITY_FLOOR {
            return (0.0, 0.0, 0.0);
        }
        // VWN 的两端插值在 ζ = ±1 处**光滑**（(1∓ζ)^{4/3} 的一阶导为 0），
        // 所以这里只做区间夹紧、不做 ζ 的数值夹紧 —— 夹紧会让 f′(ζ) 错 2.6e-4，
        // v_β 在 ζ = 1 的点上偏 1e-5（曾经真的这样错过）。
        let zeta = ((alpha - beta) / total).clamp(-1.0, 1.0);
        let rs = (3.0 / (4.0 * std::f64::consts::PI * total)).cbrt();
        let x = rs.sqrt();
        let eps_p = vwn_g(x, a_p, x0_p, b_p, c_p);
        let eps_f = vwn_g(x, a_f, x0_f, b_f, c_f);
        let factor = self.interpolation_factor(zeta);
        let eps = eps_p + (eps_f - eps_p) * factor;

        let gp = vwn_g_prime(x, a_p, x0_p, b_p, c_p);
        let gf = vwn_g_prime(x, a_f, x0_f, b_f, c_f);
        let d_eps_drs = (gp + (gf - gp) * factor) / (2.0 * x);
        let drs_dtotal = -rs / (3.0 * total);
        let df_dzeta = self.interpolation_factor_derivative(zeta);
        // ∂ζ/∂ρ_σ：α 为 (1−ζ)/ρ，β 为 −(1+ζ)/ρ。
        let dzeta_dalpha = (1.0 - zeta) / total;
        let dzeta_dbeta = -(1.0 + zeta) / total;

        let d_alpha = d_eps_drs * drs_dtotal + (eps_f - eps_p) * df_dzeta * dzeta_dalpha;
        let d_beta = d_eps_drs * drs_dtotal + (eps_f - eps_p) * df_dzeta * dzeta_dbeta;
        let finite = |value: f64| if value.is_finite() { value } else { 0.0 };
        (finite(eps), finite(d_alpha), finite(d_beta))
    }

    /// 自旋分辨的**每体积**相关能密度 `e_c = (ρ_α+ρ_β)·ε_c(ρ_α, ρ_β)`。
    ///
    /// # Errors
    ///
    /// 缺铁磁/α_c 参数，或输出长度与输入不一致。
    pub fn energy_density_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut [f64],
    ) -> Result<(), XcError> {
        self.require_spin()?;
        if out.len() != batch.len() {
            return Err(XcError::Shape(format!(
                "LdaSvwn5::energy_density_spin: output {} vs input {}",
                out.len(),
                batch.len()
            )));
        }
        for (index, slot) in out.iter_mut().enumerate() {
            let alpha = batch.rho_alpha[index].max(0.0);
            let beta = batch.rho_beta[index].max(0.0);
            *slot = (alpha + beta) * self.spin_energy_at(alpha, beta).0;
        }
        Ok(())
    }

    /// 自旋分辨的相关势 `v_σ = ∂e_c/∂ρ_σ = ε_c + (ρ_α+ρ_β)·∂ε_c/∂ρ_σ`。
    ///
    /// # Errors
    ///
    /// 缺铁磁/α_c 参数，或输出长度与输入不一致。
    pub fn potential_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        v_alpha: &mut [f64],
        v_beta: &mut [f64],
    ) -> Result<(), XcError> {
        self.require_spin()?;
        if v_alpha.len() != batch.len() || v_beta.len() != batch.len() {
            return Err(XcError::Shape(format!(
                "LdaSvwn5::potential_spin: outputs {} and {} vs input {}",
                v_alpha.len(),
                v_beta.len(),
                batch.len()
            )));
        }
        for (index, (slot_alpha, slot_beta)) in
            v_alpha.iter_mut().zip(v_beta.iter_mut()).enumerate()
        {
            let alpha = batch.rho_alpha[index].max(0.0);
            let beta = batch.rho_beta[index].max(0.0);
            let (eps, d_alpha, d_beta) = self.spin_energy_at(alpha, beta);
            let total = alpha + beta;
            *slot_alpha = eps + total * d_alpha;
            *slot_beta = eps + total * d_beta;
        }
        Ok(())
    }

    fn require_spin(&self) -> Result<(), XcError> {
        if self.supports_spin() {
            Ok(())
        } else {
            Err(XcError::SpinUnsupported(
                "LdaSvwn5 was built without the ferromagnetic/alpha parameter sets".into(),
            ))
        }
    }
}

// ───────────────────────── 工厂 ─────────────────────────

pub(crate) fn parameter(
    id: &str,
    parameters: &BTreeMap<String, f64>,
    name: &str,
) -> Result<f64, XcError> {
    parameters
        .get(name)
        .copied()
        .ok_or_else(|| XcError::MissingParameter {
            component: id.to_string(),
            parameter: name.to_string(),
        })
}

// ───────────────────────── 带权重的组件 ─────────────────────────

/// 给一个泛函项乘上权重（`functional.yaml` 的 `components[].weight`）。
///
/// 复合泛函（如 B3LYP 的 `0.8 B88 + 0.2 HF + 0.81 LYP + 0.19 VWN5`）里每个组件的
/// 能量与导数都要乘自己的系数。把缩放放在这一层，而不是让每个泛函自己记权重，
/// 可以保证 `energy_density` 与 `potential_derivatives` 用的是**同一个**系数
/// （分别传参是这类代码最容易出的错）。
pub struct ScaledTerm {
    /// 被缩放的项。
    pub inner: Box<dyn FunctionalTerm>,
    /// 权重。
    pub weight: f64,
}

impl ScaledTerm {
    /// 包装。
    #[must_use]
    pub fn new(inner: Box<dyn FunctionalTerm>, weight: f64) -> Self {
        Self { inner, weight }
    }
}

impl FunctionalTerm for ScaledTerm {
    fn potential_derivatives_laplacian(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
    ) {
        // 三个导数通道必须用**同一个权重**缩放（与能量一致），否则复合泛函的势会错。
        self.inner
            .potential_derivatives_laplacian(batch, d_rho, d_sigma, d_laplacian);
        for slot in d_rho.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_sigma.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_laplacian.iter_mut() {
            *slot *= self.weight;
        }
    }

    fn potential_derivatives_full(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
        d_tau: &mut [f64],
    ) {
        // 四个通道同一权重（FX-01 的 τ 通道同理：能量与势不能各用各的系数）。
        self.inner
            .potential_derivatives_full(batch, d_rho, d_sigma, d_laplacian, d_tau);
        for slot in d_rho.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_sigma.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_laplacian.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_tau.iter_mut() {
            *slot *= self.weight;
        }
    }

    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn requires(&self) -> Requirements {
        self.inner.requires()
    }

    fn spin_scaling(&self) -> f64 {
        self.inner.spin_scaling()
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        self.inner.energy_density(batch, out);
        for slot in out.iter_mut() {
            *slot *= self.weight;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        self.inner.potential_derivatives(batch, d_rho, d_sigma);
        for slot in d_rho.iter_mut() {
            *slot *= self.weight;
        }
        for slot in d_sigma.iter_mut() {
            *slot *= self.weight;
        }
    }

    fn spin_supported(&self) -> bool {
        self.inner.spin_supported()
    }

    fn evaluate_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        // 能量、势、三个 σ 导数必须用**同一个**权重缩放（与闭壳层路径一致）。
        self.inner.evaluate_spin(batch, out)?;
        for slot in out.energy.iter_mut() {
            *slot *= self.weight;
        }
        for slot in out.v_alpha.iter_mut() {
            *slot *= self.weight;
        }
        for slot in out.v_beta.iter_mut() {
            *slot *= self.weight;
        }
        for slot in out.d_sigma_alpha.iter_mut() {
            *slot *= self.weight;
        }
        for slot in out.d_sigma_beta.iter_mut() {
            *slot *= self.weight;
        }
        for slot in out.d_sigma_alpha_beta.iter_mut() {
            *slot *= self.weight;
        }
        Ok(())
    }
}

/// 这批泛函项里有没有任何一项需要 τ（FX-01 的门禁用）。
///
/// 为什么要一个独立函数：**数据声明 `requires.kinetic_density: true`
/// 与"实现里真的有项在读 τ"是两件事**。两者不一致时必须报错 ——
/// 数据说要 τ 但没有任何项读它，用户会拿到一个 GGA 水平的答案却以为算了 meta-GGA。
#[must_use]
pub fn terms_require_tau(terms: &[Box<dyn FunctionalTerm>]) -> bool {
    terms.iter().any(|term| term.requires().kinetic_density)
}

/// 按权重包装（权重为 1 时原样返回，不引入额外一层虚调用）。
#[must_use]
pub fn scale_term(term: Box<dyn FunctionalTerm>, weight: f64) -> Box<dyn FunctionalTerm> {
    if (weight - 1.0).abs() <= 1e-12 {
        term
    } else {
        Box::new(ScaledTerm::new(term, weight))
    }
}
/// 由数据里的组件 id 与参数构造泛函项。
pub fn build_term(
    id: &str,
    parameters: &BTreeMap<String, f64>,
) -> Result<Box<dyn FunctionalTerm>, XcError> {
    match id {
        "LdaSlaterX" => Ok(Box::new(LdaSlaterX::new(parameter(
            id,
            parameters,
            "slater_alpha",
        )?))),
        "LdaSvwn5" => {
            let mut term = LdaSvwn5::new((
                parameter(id, parameters, "vwn5_p_A")?,
                parameter(id, parameters, "vwn5_p_x0")?,
                parameter(id, parameters, "vwn5_p_b")?,
                parameter(id, parameters, "vwn5_p_c")?,
            ));
            // 三组参数齐全才打开自旋插值（旧的只含顺磁参数的数据仍然可用）。
            if parameters.contains_key("vwn5_f_A") && parameters.contains_key("vwn5_a_A") {
                term = term.with_spin_parameters(
                    (
                        parameter(id, parameters, "vwn5_f_A")?,
                        parameter(id, parameters, "vwn5_f_x0")?,
                        parameter(id, parameters, "vwn5_f_b")?,
                        parameter(id, parameters, "vwn5_f_c")?,
                    ),
                    (
                        parameter(id, parameters, "vwn5_a_A")?,
                        parameter(id, parameters, "vwn5_a_x0")?,
                        parameter(id, parameters, "vwn5_a_b")?,
                        parameter(id, parameters, "vwn5_a_c")?,
                    ),
                );
            }
            Ok(Box::new(term))
        }
        others => crate::gga::build_pbe_term(others, parameters),
    }
}
