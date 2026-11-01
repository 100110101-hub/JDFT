//! meta-GGA 交换（Minnesota **M06-L** 与 **M06-2X**）。
//!
//! # 变量约定（与 libxc 的 `maple2c/mgga_exc/mgga_x_m06l.c` 逐项对齐）
//!
//! 本项目闭壳层路径只有 `(ρ, σ, τ)`。libxc 的 M06 代码用**每自旋**缩减变量，
//! 对非极化体系等价于
//!
//! ```text
//! x² = 2^{2/3}·σ/ρ^{8/3}     （= (σ/4)/(ρ/2)^{8/3}）
//! t  = 2^{2/3}·τ/ρ^{5/3}     （= (τ/2)/(ρ/2)^{5/3}）
//! K′ = 2^{2/3}·K,  K = (3/10)(3π²)^{2/3}
//! ```
//!
//! **`K′` 是踩过的坑**：把 `t` 换成每自旋量却仍用总密度的 `K`，均匀电子气
//! （`τ = Kρ^{5/3}`）处 `w(t) ≠ 0`，增强因子偏 4%，M06-L 就不再退回 Slater。
//! libxc 的 C 代码里 `t45 = 0.3·6^{2/3}π^{4/3} = 2^{2/3}K`、`t49 = 2^{2/3}τ/ρ^{5/3}`
//! —— 两者同尺度，所以 `w(t) = (K′−t)/(K′+t)`。
//!
//! ```text
//! F_x = pbe_f(x²)·Σ_{i<12} a_i w(t)^i  [+ gtv4(α, d, x², 2(t−K′))]（M06-L 才有）
//! ε   = ε_x^LDA,σ(ρ)·F_x，两个通道非极化下相同，乘 2
//! ```
//!
//! # 与 `energy_density` 的约定
//!
//! 与闭壳层 GGA 一致：`energy_density` 给**每电子** ε（`E = ∫ρ ε dV`）；
//! 导数用三通道对偶数一次算出。（τ 通道的矩阵元另乘 ρ，见 `qwb-scf::rks`。）
#![deny(unsafe_code)]

use crate::gga::Pw92Set;
use crate::{
    DensityBatch, FunctionalTerm, Requirements, SpinDensityBatch, SpinXcOutput, XcError,
    check_shape, check_spin_output,
};

/// `2^{2/3}`：总密度 ↔ 每自旋缩减变量。
fn spin_scale() -> f64 {
    2.0_f64.powf(2.0 / 3.0)
}

/// `K = (3/10)(3π²)^{2/3}`。
fn k_factor_c() -> f64 {
    0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0)
}

/// `X2S = 1/(2(6π²)^{1/3})`。
fn x2s() -> f64 {
    0.5 / (6.0 * std::f64::consts::PI * std::f64::consts::PI).powf(1.0 / 3.0)
}

/// 前向对偶数：`value + Σ_{i<N} d_i·δx_i`，一次求值给出全部一阶偏导。
///
/// `N` 是独立变量个数：闭壳层三元组 `(ρ, σ, τ)` 用 `M = Dual<3>`；自旋相关项的
/// 六元组 `(ρ_α, ρ_β, σ_αα, σ_ββ, τ_α, τ_β)` 用 `S = Dual<6>`。**共用同一个类型**
/// 是为了让 `f_pw` / `b97_g` / `gtv4` / Fermi 衰减这些零件只存在一份 —— 公式抄两遍
/// 就等于多出一处可以悄悄漂的地方。
#[derive(Debug, Clone, Copy)]
struct Dual<const N: usize> {
    value: f64,
    d: [f64; N],
}

/// 闭壳层三元组 `(ρ, σ, τ)`。
type M = Dual<3>;

/// 自旋六元组 `(ρ_α, ρ_β, σ_αα, σ_ββ, τ_α, τ_β)`。
///
/// **为什么没有 `σ_αβ`**：M06 相关项里不存在 `∇ρ_α·∇ρ_β` 的任何组合 —— 同自旋通道只看
/// `σ_σσ`，反自旋通道只看 `x²_α + x²_β` 与 `Z_α + Z_β`。所以 `∂e/∂σ_αβ ≡ 0` 是**结构性**的，
/// 不是漏了一项（libxc 对这三个相关泛函同样返回 vsigma_αβ = 0，见
/// `tests/reference/spin_mgga_pointwise.csv` 的 `d_sigma_alpha_beta` 列）。
type S = Dual<6>;

/// `S` 的通道下标。
const RA: usize = 0;
const RB: usize = 1;
const SAA: usize = 2;
const SBB: usize = 3;
const TA: usize = 4;
const TB: usize = 5;

impl<const N: usize> Dual<N> {
    fn constant(value: f64) -> Self {
        Self { value, d: [0.0; N] }
    }

    /// 第 `index` 个自变量（`d_index = 1`）。
    fn basis(value: f64, index: usize) -> Self {
        let mut d = [0.0; N];
        d[index] = 1.0;
        Self { value, d }
    }

    fn ln(self) -> Self {
        let value = self.value.ln();
        let inv = 1.0 / self.value;
        Self {
            value,
            d: self.d.map(|x| x * inv),
        }
    }

    /// `−expm1(x)`（Fermi_D_corrected 的因子）。
    fn neg_exp_m1(self) -> Self {
        let value = -self.value.exp_m1();
        let scale = -self.value.exp();
        Self {
            value,
            d: self.d.map(|x| scale * x),
        }
    }

    fn recip(self) -> Self {
        let value = 1.0 / self.value;
        let scale = -value * value;
        Self {
            value,
            d: self.d.map(|x| scale * x),
        }
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            d: self.d.map(|x| scale * x),
        }
    }
}

impl<const N: usize> std::ops::Add for Dual<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            d: std::array::from_fn(|i| self.d[i] + other.d[i]),
        }
    }
}

impl<const N: usize> std::ops::Sub for Dual<N> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            d: std::array::from_fn(|i| self.d[i] - other.d[i]),
        }
    }
}

// `Mul` 里出现 `+` 是**乘积法则**，不是打错运算符：`(uv)' = u'v + uv'`。
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> std::ops::Mul for Dual<N> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            d: std::array::from_fn(|i| self.d[i] * other.value + self.value * other.d[i]),
        }
    }
}

impl<const N: usize> std::ops::Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            d: self.d.map(|x| -x),
        }
    }
}

impl<const N: usize> std::ops::Div for Dual<N> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let inverse = 1.0 / other.value;
        Self {
            value,
            d: std::array::from_fn(|i| (self.d[i] - value * other.d[i]) * inverse),
        }
    }
}

/// `pbe_f(x²) = 1 + κμs²/(κ + μs²)`，`s² = X2S²·x²`。
fn pbe_enhancement(w: M, kappa: f64, mu: f64) -> M {
    let one = M::constant(1.0);
    let s2 = M::constant(x2s() * x2s()) * w;
    let kappa_d = M::constant(kappa);
    let mu_d = M::constant(mu);
    one + kappa_d * mu_d * s2 / (kappa_d + mu_d * s2)
}

/// `gtv4(α, d, x², z) = d₁/γ + (d₂x²+d₃z)/γ² + (d₄x⁴+d₅x²z+d₆z²)/γ³`，`γ = 1+α(x²+z)`。
fn gtv4<const N: usize>(alpha: f64, d: &[f64; 6], w: Dual<N>, z: Dual<N>) -> Dual<N> {
    let one = Dual::constant(1.0);
    let gamma = one + Dual::constant(alpha) * (w + z);
    let g2 = gamma * gamma;
    let g3 = g2 * gamma;
    Dual::constant(d[0]) / gamma
        + (Dual::constant(d[1]) * w + Dual::constant(d[2]) * z) / g2
        + (Dual::constant(d[3]) * w * w
            + Dual::constant(d[4]) * w * z
            + Dual::constant(d[5]) * z * z)
            / g3
}

/// `mgga_series_w(a, t) = Σ_{i<12} a_i w^i`，`w = (K′−t)/(K′+t)`。
fn mgga_series_w(a: &[f64; 12], t: M) -> M {
    let k = M::constant(spin_scale() * k_factor_c());
    let w = (k - t) / (k + t);
    let mut polynomial = M::constant(a[11]);
    for &coefficient in a[..11].iter().rev() {
        polynomial = polynomial * w + M::constant(coefficient);
    }
    polynomial
}

/// M06 家族交换参数。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct M06ExchangeParameters {
    /// 动能级数的 12 个系数。
    pub series: [f64; 12],
    /// VS98（`gtv4`）的 6 个系数（M06-2X 全零且 `has_vs98 = false`）。
    pub vs98: [f64; 6],
    /// `gtv4` 的 `α`。
    pub alpha: f64,
    /// 是否带 VS98 项（M06-L 有，M06-2X 没有）。
    pub has_vs98: bool,
}

impl M06ExchangeParameters {
    /// 由系数构造。
    #[must_use]
    pub fn new(series: [f64; 12], vs98: [f64; 6], alpha: f64, has_vs98: bool) -> Self {
        Self {
            series,
            vs98,
            alpha,
            has_vs98,
        }
    }
}

/// M06 家族交换项（`M06-L` 与 `M06-2X` 共用骨架、参数不同）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct M06Exchange {
    /// 组件 id（`MggaXM06L` / `MggaXM062x`）。
    pub id: &'static str,
    /// 参数。
    pub parameters: M06ExchangeParameters,
}

impl M06Exchange {
    /// 由 id 与参数构造。
    #[must_use]
    pub fn new(id: &'static str, parameters: M06ExchangeParameters) -> Self {
        Self { id, parameters }
    }

    /// `F_x(x², t)`。
    fn enhancement(&self, w: M, t: M) -> M {
        let value = pbe_enhancement(w, 0.804, 0.219_514_972_764_517_04)
            * mgga_series_w(&self.parameters.series, t);
        if self.parameters.has_vs98 {
            let k = M::constant(spin_scale() * k_factor_c());
            value
                + gtv4(
                    self.parameters.alpha,
                    &self.parameters.vs98,
                    w,
                    M::constant(2.0) * (t - k),
                )
        } else {
            value
        }
    }

    /// 单点：`(ε, ∂ε/∂ρ, ∂ε/∂σ, ∂ε/∂τ)`（每电子）。
    fn point(&self, rho: f64, sigma: f64, tau: f64) -> (f64, f64, f64, f64) {
        let rho = rho.max(1.0e-12);
        let tau = tau.max(1.0e-20);
        let rho_d = M::basis(rho, 0);
        let scale = M::constant(spin_scale());
        let rs = (M::constant(3.0 / (4.0 * std::f64::consts::PI)) / rho_d).powf(1.0 / 3.0);
        let xs_sq = scale * M::basis(sigma, 1) / rho_d.powf(8.0 / 3.0);
        let t = scale * M::basis(tau, 2) / rho_d.powf(5.0 / 3.0);
        // 每自旋 LDA 交换（每总粒子）：−(3/4)(9/(2π²))^{1/3}·(1/2)^{4/3}/rs
        let c = 0.75 * (9.0 / (2.0 * std::f64::consts::PI * std::f64::consts::PI)).powf(1.0 / 3.0);
        let lda = M::constant(-c * 0.5_f64.powf(4.0 / 3.0)) / rs;
        let value = M::constant(2.0) * lda * self.enhancement(xs_sq, t);
        (value.value, value.d[0], value.d[1], value.d[2])
    }
}

impl FunctionalTerm for M06Exchange {
    fn id(&self) -> &'static str {
        self.id
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    /// 交换项的自旋标度是**精确**的，M06 的交换也不例外：它只依赖同一自旋通道的
    /// `(ρ_σ, σ_σσ, τ_σ)`，所以
    ///
    /// ```text
    /// e = ρ_α·ε(2ρ_α, 4σ_αα, 2τ_α) + ρ_β·ε(2ρ_β, 4σ_ββ, 2τ_β)
    /// ```
    ///
    /// 这一条由 `exchange_spin_scaling` 实现（它已经处理 τ），与「相关项要 ζ 插值」
    /// 的情况不同 —— M06 的相关项**不**适用这个标度。
    fn evaluate_spin(
        &self,
        batch: &crate::SpinDensityBatch<'_>,
        out: &mut crate::SpinXcOutput<'_>,
    ) -> Result<(), crate::XcError> {
        crate::exchange_spin_scaling(self, batch, out)
    }

    /// 自旋**且** τ：能自旋化，但要求调用方同时给出 `τ_α/τ_β`。
    ///
    /// 与 [`FunctionalTerm::spin_supported`] 分开是刻意的：`spin_supported` 表示
    /// 「当前 `uks` 路径已经能喂它」，而 `uks` 还没有积累 τ（那一步是 UKS 的
    /// meta-GGA 接线）。所以这一位只表示**组件本身**已经具备自旋+τ 的求值能力，
    /// 逐点测试直接调 [`FunctionalTerm::evaluate_spin`] 验证它。
    fn spin_meta_supported(&self) -> bool {
        true
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "M06Exchange::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.point(batch.rho[index], sigma, batch.tau_at(index)).0;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "M06Exchange::d_rho").expect("shape");
        check_shape(batch, d_sigma, "M06Exchange::d_sigma").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let (_, dr, ds, _) = self.point(batch.rho[index], sigma, batch.tau_at(index));
            d_rho[index] = dr;
            d_sigma[index] = ds;
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
        check_shape(batch, d_rho, "M06Exchange::d_rho").expect("shape");
        check_shape(batch, d_sigma, "M06Exchange::d_sigma").expect("shape");
        check_shape(batch, d_tau, "M06Exchange::d_tau").expect("shape");
        d_laplacian.fill(0.0);
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let (_, dr, ds, dt) = self.point(batch.rho[index], sigma, batch.tau_at(index));
            d_rho[index] = dr;
            d_sigma[index] = ds;
            d_tau[index] = dt;
        }
    }
}

/// `f''(0) = 8/(9(2^{4/3} − 2))`（PW92 的精确自旋刚度分母）。
fn f_second_derivative_zero() -> f64 {
    8.0 / (9.0 * (2.0_f64.powf(4.0 / 3.0) - 2.0))
}

/// PW92 的 `G(rs)`（顺磁/铁磁/α_c 三组共用一个形状），对偶数版。
fn pw92_form<const N: usize>(rs: Dual<N>, set: &Pw92Set) -> Dual<N> {
    let one = Dual::constant(1.0);
    let root = rs.powf(0.5);
    let beta = Dual::constant(set.betas[0]) * root
        + Dual::constant(set.betas[1]) * rs
        + Dual::constant(set.betas[2]) * rs * root
        + Dual::constant(set.betas[3]) * rs * rs;
    let denominator = Dual::constant(2.0 * set.a) * beta;
    let log = (one + denominator.recip()).ln();
    Dual::constant(-2.0 * set.a) * (one + Dual::constant(set.alpha) * rs) * log
}

/// `f_pw(rs, ζ; 1+ζ, 1−ζ)`：modified PW92 的相关能（每电子），带 ζ⁴ 插值。
///
/// 与 `gga.rs` 的 PBE 相关用同一个 LDA 基（`Pw92Set`，已逐点对照 libxc 到 1e-15）。
/// `opz = 1+ζ` 与 `omz = 1−ζ` 由调用方**分别**给，而不是从 `ζ` 加减：`ζ→±1` 时
/// `1−ζ` 只剩一位有效数字，而 `2ρ_β/ρ` 精确到机器精度（libxc 的 maple 代码同样用
/// `(opz, omz)` 一对量）。
fn f_pw<const N: usize>(rs: Dual<N>, zeta: Dual<N>, opz: Dual<N>, omz: Dual<N>) -> Dual<N> {
    let one = Dual::constant(1.0);
    let ep = pw92_form(rs, &Pw92Set::PARA);
    let ef = pw92_form(rs, &Pw92Set::FERRO);
    let alpha_c = -pw92_form(rs, &Pw92Set::ALPHA);
    let four_thirds = 4.0 / 3.0;
    let f = (opz.powf(four_thirds) + omz.powf(four_thirds) - Dual::constant(2.0))
        / Dual::constant(2.0_f64.powf(four_thirds) - 2.0);
    let z4 = zeta.powf(4.0);
    ep * (one - f * z4)
        + ef * f * z4
        + alpha_c / Dual::constant(f_second_derivative_zero()) * f * (one - z4)
}

/// `b97_g(γ, c, x²) = Σ_i c_i u^i`，`u = γx²/(1 + γx²)`。
fn b97_g<const N: usize>(gamma: f64, c: &[f64; 5], w: Dual<N>) -> Dual<N> {
    let one = Dual::constant(1.0);
    let g = Dual::constant(gamma);
    let u = g * w / (one + g * w);
    let mut polynomial = Dual::constant(0.0);
    for &coefficient in c.iter().rev() {
        polynomial = polynomial * u + Dual::constant(coefficient);
    }
    polynomial
}

/// `Fermi_D(x², t) = (8t − x²)/(8t)`。
fn fermi_d<const N: usize>(w: Dual<N>, t: Dual<N>) -> Dual<N> {
    let eight_t = Dual::constant(8.0) * t;
    (eight_t - w) / eight_t
}

/// `Fermi_D_corrected = Fermi_D·(1 − exp(−4t²/c²))`，`c = 1e-10`。
fn fermi_d_corrected<const N: usize>(w: Dual<N>, t: Dual<N>) -> Dual<N> {
    let cnst = 1.0e-10;
    let arg = Dual::constant(-4.0 / (cnst * cnst)) * t * t;
    fermi_d(w, t) * arg.neg_exp_m1()
}

/// M06 家族相关参数（M06-L 与 M06-2X 共用表达式、参数不同）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct M06CorrelationParameters {
    /// 同自旋 `γ`。
    pub gamma_same: f64,
    /// 反自旋 `γ`。
    pub gamma_opposite: f64,
    /// 同自旋 VS98 `α`。
    pub alpha_same: f64,
    /// 反自旋 VS98 `α`。
    pub alpha_opposite: f64,
    /// 同自旋 `b97` 系数。
    pub c_same: [f64; 5],
    /// 反自旋 `b97` 系数。
    pub c_opposite: [f64; 5],
    /// 同自旋 VS98 系数。
    pub d_same: [f64; 6],
    /// 反自旋 VS98 系数。
    pub d_opposite: [f64; 6],
}

impl M06CorrelationParameters {
    /// 由 8 组系数构造。
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gamma_same: f64,
        gamma_opposite: f64,
        alpha_same: f64,
        alpha_opposite: f64,
        c_same: [f64; 5],
        c_opposite: [f64; 5],
        d_same: [f64; 6],
        d_opposite: [f64; 6],
    ) -> Self {
        Self {
            gamma_same,
            gamma_opposite,
            alpha_same,
            alpha_opposite,
            c_same,
            c_opposite,
            d_same,
            d_opposite,
        }
    }
}

/// M06 家族相关项。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct M06Correlation {
    /// 组件 id。
    pub id: &'static str,
    /// 参数。
    pub parameters: M06CorrelationParameters,
}

impl M06Correlation {
    /// 由 id 与参数构造。
    #[must_use]
    pub fn new(id: &'static str, parameters: M06CorrelationParameters) -> Self {
        Self { id, parameters }
    }

    /// 单点：`(ε, ∂ε/∂ρ, ∂ε/∂σ, ∂ε/∂τ)`（每电子）。
    fn point(&self, rho: f64, sigma: f64, tau: f64) -> (f64, f64, f64, f64) {
        let rho = rho.max(1.0e-12);
        let tau = tau.max(1.0e-20);
        let rho_d = M::basis(rho, 0);
        let scale = M::constant(spin_scale());
        let rs = (M::constant(3.0 / (4.0 * std::f64::consts::PI)) / rho_d).powf(1.0 / 3.0);
        let xs_sq = scale * M::basis(sigma, 1) / rho_d.powf(8.0 / 3.0);
        let t = scale * M::basis(tau, 2) / rho_d.powf(5.0 / 3.0);
        let k = M::constant(spin_scale() * k_factor_c());
        let one = M::constant(1.0);
        let p = &self.parameters;

        // Stoll 分解（非极化：ζ = 0、opz = omz = 1）
        let two = M::constant(2.0);
        let par_same =
            M::constant(0.5) * f_pw(rs * two.powf(1.0 / 3.0), one, two, M::constant(0.0));
        let perp = f_pw(rs, M::constant(0.0), one, one) - par_same - par_same;

        let same_channel = par_same
            * (b97_g(p.gamma_same, &p.c_same, xs_sq) * fermi_d_corrected(xs_sq, t)
                + gtv4(p.alpha_same, &p.d_same, xs_sq, M::constant(2.0) * (t - k))
                    * fermi_d(xs_sq, t));
        let same = M::constant(2.0) * same_channel;

        let w_opposite = xs_sq + xs_sq;
        let z_opposite = M::constant(2.0) * (t + t - M::constant(2.0) * k);
        let opposite = perp
            * (b97_g(p.gamma_opposite, &p.c_opposite, w_opposite)
                + gtv4(p.alpha_opposite, &p.d_opposite, w_opposite, z_opposite));

        let value = same + opposite;
        (value.value, value.d[0], value.d[1], value.d[2])
    }
    /// 自旋分辨单点：每体积能量 `e = ρ·ε` 与
    /// `[∂e/∂ρ_α, ∂e/∂ρ_β, ∂e/∂σ_αα, ∂e/∂σ_ββ, ∂e/∂τ_α, ∂e/∂τ_β]`。
    ///
    /// 逐项对照 libxc 7.0.0 `mgga_c_m06l.c` 与 NWChem `nwxc_c_m06.F` / `nwxc_c_vs98.F`，
    /// 独立数值验证见 `scripts/validate_m06_spin_correlation.py`（能量 3e-16、导数 ~1e-12
    /// 逐点；两个 `ρ_β ≤ 1e-6` 的近空通道单独说明，见下面的下限注释）。
    ///
    /// # 公式（每自旋缩减变量，不做 `2^{2/3}` 缩放）
    ///
    /// ```text
    /// x²_σ = σ_σσ/ρ_σ^{8/3},  t_σ = τ_σ/ρ_σ^{5/3},  Z_σ = 2(t_σ − K′),
    /// K′ = (3/10)(6π²)^{2/3}
    ///
    /// 同自旋 E_σ = ρ_σ·ε_c^LDA(ρ_σ, ζ=1)·[b97_σ(x²_σ)·Fermi_D^corr(x²_σ,t_σ)
    ///                                     + gtv4_σ(x²_σ, Z_σ)·Fermi_D(x²_σ,t_σ)]
    /// 反自旋 X_ab = x²_α + x²_β,  Z_ab = Z_α + Z_β
    ///        E_ab = e_Stoll·[b97_ab(X_ab) + gtv4_ab(X_ab, Z_ab)]
    ///        e_Stoll = ρ·ε_c^LDA(ρ, ζ) − Σ_σ ρ_σ·ε_c^LDA(ρ_σ, ζ=1)
    /// ```
    ///
    /// 注意同自旋通道里 **b97 用 `Fermi_D^corr`、gtv4 用裸 `Fermi_D`**，两者不对称
    /// （照抄上游；把两个衰减因子统一会被逐点测试抓住）。
    ///
    /// 下限只为了不产生 NaN：`ρ_σ ≥ 1e-12`、`τ_σ ≥ 1e-20`，与闭壳层 [`Self::point`]
    /// 同一套。近空通道（`ρ_σ ≤ 1e-6`）的能量可以忽略，而它的势是单侧极限、量级很大，
    /// 逐点测试因此不对这类通道断言（打印出来，不静默）。
    fn spin_point(
        &self,
        rho_alpha: f64,
        rho_beta: f64,
        sigma_alpha: f64,
        sigma_beta: f64,
        tau_alpha: f64,
        tau_beta: f64,
    ) -> (f64, [f64; 6]) {
        let ra_d = S::basis(rho_alpha.max(1.0e-12), RA);
        let rb_d = S::basis(rho_beta.max(1.0e-12), RB);
        let one = S::constant(1.0);
        let two = S::constant(2.0);
        let zero = S::constant(0.0);
        let pi34 = S::constant(3.0 / (4.0 * std::f64::consts::PI));

        let rho_d = ra_d + rb_d;
        let rs = (pi34 / rho_d).powf(1.0 / 3.0);
        // ζ 以及 1±ζ：后者走 2ρ_σ/ρ，`ζ→±1` 时不做抵消。
        let zeta = (ra_d - rb_d) / rho_d;
        let opz = two * ra_d / rho_d;
        let omz = two * rb_d / rho_d;
        let k_prime =
            S::constant(0.3 * (6.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0));

        let saa = S::basis(sigma_alpha.max(0.0), SAA);
        let sbb = S::basis(sigma_beta.max(0.0), SBB);
        let ta = S::basis(tau_alpha.max(1.0e-20), TA);
        let tb = S::basis(tau_beta.max(1.0e-20), TB);
        let x2_alpha = saa / ra_d.powf(8.0 / 3.0);
        let x2_beta = sbb / rb_d.powf(8.0 / 3.0);
        let t_alpha = ta / ra_d.powf(5.0 / 3.0);
        let t_beta = tb / rb_d.powf(5.0 / 3.0);
        let z_alpha = two * (t_alpha - k_prime);
        let z_beta = two * (t_beta - k_prime);

        let rs_alpha = (pi34 / ra_d).powf(1.0 / 3.0);
        let rs_beta = (pi34 / rb_d).powf(1.0 / 3.0);
        let lda_alpha = ra_d * f_pw(rs_alpha, one, two, zero);
        let lda_beta = rb_d * f_pw(rs_beta, one, two, zero);

        let p = &self.parameters;
        let same_alpha = lda_alpha
            * (b97_g(p.gamma_same, &p.c_same, x2_alpha) * fermi_d_corrected(x2_alpha, t_alpha)
                + gtv4(p.alpha_same, &p.d_same, x2_alpha, z_alpha) * fermi_d(x2_alpha, t_alpha));
        let same_beta = lda_beta
            * (b97_g(p.gamma_same, &p.c_same, x2_beta) * fermi_d_corrected(x2_beta, t_beta)
                + gtv4(p.alpha_same, &p.d_same, x2_beta, z_beta) * fermi_d(x2_beta, t_beta));

        // Stoll 分解：ζ 依赖全部留给反自旋通道（ζ=±1 时 e_Stoll 精确为 0）。
        let stoll = rho_d * f_pw(rs, zeta, opz, omz) - lda_alpha - lda_beta;
        let x2_opposite = x2_alpha + x2_beta;
        let z_opposite = z_alpha + z_beta;
        let opposite = stoll
            * (b97_g(p.gamma_opposite, &p.c_opposite, x2_opposite)
                + gtv4(p.alpha_opposite, &p.d_opposite, x2_opposite, z_opposite));
        let value = same_alpha + same_beta + opposite;
        (value.value, value.d)
    }
}

impl FunctionalTerm for M06Correlation {
    fn id(&self) -> &'static str {
        self.id
    }

    fn requires(&self) -> Requirements {
        Requirements::meta_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "M06Correlation::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.point(batch.rho[index], sigma, batch.tau_at(index)).0;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "M06Correlation::d_rho").expect("shape");
        check_shape(batch, d_sigma, "M06Correlation::d_sigma").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let (_, dr, ds, _) = self.point(batch.rho[index], sigma, batch.tau_at(index));
            d_rho[index] = dr;
            d_sigma[index] = ds;
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
        check_shape(batch, d_rho, "M06Correlation::d_rho").expect("shape");
        check_shape(batch, d_sigma, "M06Correlation::d_sigma").expect("shape");
        check_shape(batch, d_tau, "M06Correlation::d_tau").expect("shape");
        d_laplacian.fill(0.0);
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let (_, dr, ds, dt) = self.point(batch.rho[index], sigma, batch.tau_at(index));
            d_rho[index] = dr;
            d_sigma[index] = ds;
            d_tau[index] = dt;
        }
    }

    fn spin_meta_supported(&self) -> bool {
        true
    }

    /// 自旋分辨相关项：走 [`Self::spin_point`]，一次求值给出全部六个偏导。
    ///
    /// 与交换项不同，相关项**不能**用 `exchange_spin_scaling`：ζ 插值让反自旋通道
    /// 依赖两个自旋的和，所以必须整点求值（这也是 `spin_supported` 之前为 false
    /// 的原因之一）。
    fn evaluate_spin(
        &self,
        batch: &SpinDensityBatch<'_>,
        out: &mut SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        let points = batch.len();
        check_spin_output(points, out, "M06Correlation::evaluate_spin")?;
        if out.d_tau_alpha.len() != points || out.d_tau_beta.len() != points {
            return Err(XcError::Shape(format!(
                "{}: spin-resolved correlation needs the τ channels (use SpinXcOutput::with_tau)",
                self.id()
            )));
        }
        out.clear();
        for index in 0..points {
            let (energy, d) = self.spin_point(
                batch.rho_alpha[index],
                batch.rho_beta[index],
                batch.sigma_alpha_at(index),
                batch.sigma_beta_at(index),
                batch.tau_alpha.get(index).copied().unwrap_or(0.0),
                batch.tau_beta.get(index).copied().unwrap_or(0.0),
            );
            out.energy[index] = energy;
            out.v_alpha[index] = d[RA];
            out.v_beta[index] = d[RB];
            out.d_sigma_alpha[index] = d[SAA];
            out.d_sigma_beta[index] = d[SBB];
            // 结构性为零：M06 相关项不依赖 σ_αβ（见 `type S` 的注释）。
            out.d_sigma_alpha_beta[index] = 0.0;
            out.d_tau_alpha[index] = d[TA];
            out.d_tau_beta[index] = d[TB];
        }
        Ok(())
    }
}
