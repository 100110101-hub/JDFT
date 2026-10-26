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
    /// 需要 `∇²ρ`（meta-GGA，第一阶段不支持）。
    pub laplacian: bool,
}

impl Requirements {
    /// 纯 LDA：只要 ρ。
    #[must_use]
    pub fn lda() -> Self {
        Self {
            rho: true,
            sigma: false,
            laplacian: false,
        }
    }

    /// GGA：ρ 与 σ。
    #[must_use]
    pub fn gga() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: false,
        }
    }

    /// 需要 `∇²ρ` 的项（LYP 一类）：ρ、σ、`∇²ρ` 全要。
    #[must_use]
    pub fn laplacian_gga() -> Self {
        Self {
            rho: true,
            sigma: true,
            laplacian: true,
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
}

impl<'a> DensityBatch<'a> {
    /// 用 ρ 构造（LDA 用；`sigma` 与 `laplacian` 为空）。
    #[must_use]
    pub fn from_rho(rho: &'a [f64]) -> Self {
        Self {
            rho,
            sigma: &[],
            laplacian: &[],
        }
    }

    /// 用 ρ 与 σ 构造（GGA 用；不看 `laplacian`）。
    #[must_use]
    pub fn new(rho: &'a [f64], sigma: &'a [f64]) -> Self {
        Self {
            rho,
            sigma,
            laplacian: &[],
        }
    }

    /// 用 ρ、σ、∇²ρ 构造（meta-GGA / LYP 用）。
    #[must_use]
    pub fn with_laplacian(rho: &'a [f64], sigma: &'a [f64], laplacian: &'a [f64]) -> Self {
        Self {
            rho,
            sigma,
            laplacian,
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
    if !batch.sigma.is_empty() && batch.sigma.len() != batch.len() {
        return Err(XcError::Shape(format!(
            "{}: sigma {} vs rho {}",
            what,
            batch.sigma.len(),
            batch.rho.len()
        )));
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
}

// ───────────────────────── VWN5 相关 ─────────────────────────

/// VWN5（Vosko–Wilk–Nusair parameterization V）相关能，顺磁通道。
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
}

// ─────────────────── 自旋分辨 LSDA（UKS 的 XC 侧，SC-01b）───────────────────

/// 自旋分辨的密度批次（每个点一对 `ρ_α`、`ρ_β`）。
///
/// 闭壳层路径用 [`DensityBatch`]（自旋求和）；UKS 需要每个自旋通道各自的密度，
/// 所以单独一个类型。两者都在"每个点一个标量"的层面工作，消费方逐点累加权重。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SpinDensityBatch<'a> {
    /// `ρ_α`（e/Bohr³）。
    pub rho_alpha: &'a [f64],
    /// `ρ_β`（e/Bohr³）。
    pub rho_beta: &'a [f64],
}

impl<'a> SpinDensityBatch<'a> {
    /// 由两套密度构造。
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
}

impl LdaSvwn5 {
    /// `f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2)`（VWN5 的自旋插值因子）。
    #[must_use]
    pub fn interpolation_factor(&self, zeta: f64) -> f64 {
        let zeta = zeta.clamp(-1.0, 1.0);
        let numerator = (1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0;
        numerator / (2.0_f64.powf(4.0 / 3.0) - 2.0)
    }

    /// `f'(ζ)`。
    #[must_use]
    pub fn interpolation_factor_derivative(&self, zeta: f64) -> f64 {
        let zeta = zeta.clamp(-1.0 + 1e-12, 1.0 - 1e-12);
        let numerator = (1.0 + zeta).cbrt() - (1.0 - zeta).cbrt();
        (4.0 / 3.0) * numerator / (2.0_f64.powf(4.0 / 3.0) - 2.0)
    }

    /// 自旋分辨的每电子相关能密度 `ε_c(ρ_α, ρ_β)`（写进 `out`）。
    ///
    /// # Errors
    ///
    /// 没有铁磁/α_c 参数集，或输出长度与输入不一致。
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
            *slot = self
                .spin_energy_at(batch.rho_alpha[index], batch.rho_beta[index])
                .0;
        }
        Ok(())
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
