//! GGA（D-10 的 PBE）：交换与相关。
//!
//! # 为什么用「对偶数」（dual number）
//!
//! X-02 要求 `dε/dρ`、`dε/dσ` 与有限差分差 < 1e-9。PBE 的导数手推很容易出错
//! （链式法则套 4 层），所以这里不手推：把能量写成对 `(ρ, σ)` 的对偶数表达式，
//! 一次求值同时得到**值与两个偏导**，数学上是精确的（不是数值近似）。
//!
//! ```text
//! D { v, drho, dsigma }   // v + drho·δρ + dsigma·δσ
//! ```
//!
//! VWN5 的部分没有对偶数实现，但它已有解析导数 `vwn_g_prime`，于是直接构造
//! 「值 + 已知偏导」的对偶数即可（见 [`PbeCorrelation`]）。
//!
//! # 公式（Perdew–Burke–Ernzerhof 1996，闭壳层 ζ = 0）
//!
//! 交换：
//! ```text
//! ε_x = ε_x^LDA(ρ)·F_x(s),  F_x = 1 + κ − κ/(1 + μ s²/κ)
//! s² = σ / (4 k_F² ρ²),      k_F = (3π²ρ)^{1/3}
//! ```
//! 相关：
//! ```text
//! ε_c = ε_c^LDA(ρ) + H(r_s, t)
//! H = γ ln[1 + (β/γ)·t²·(1 + A t²)/(1 + A t² + A² t⁴)]
//! A = (β/γ)/(exp(−ε_c^LDA/γ) − 1),  t² = σ/(4 k_s² ρ²),  k_s² = 4k_F/π
//! ```
//! `σ → 0` 时必须退化为 LDA（`F_x → 1`、`H → 0`）——测试里钉住这一点。

use std::collections::BTreeMap;

use crate::{
    DensityBatch, FunctionalTerm, Requirements, XcError, check_shape, parameter,
    three_over_pi_cbrt, vwn_g, vwn_g_prime,
};

/// 对偶数：`value + ∂ρ·δρ + ∂σ·δσ`。
#[derive(Debug, Clone, Copy)]
struct D {
    value: f64,
    d_rho: f64,
    d_sigma: f64,
}

impl D {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 0.0,
        }
    }

    fn rho(value: f64) -> Self {
        Self {
            value,
            d_rho: 1.0,
            d_sigma: 0.0,
        }
    }

    fn sigma(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 1.0,
        }
    }

    /// 已知值 + 已知偏导（用于 VWN5 这类已有解析导数的子表达式）。
    fn with_partials(value: f64, d_rho: f64, d_sigma: f64) -> Self {
        Self {
            value,
            d_rho,
            d_sigma,
        }
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            d_rho: scale * self.d_rho,
            d_sigma: scale * self.d_sigma,
        }
    }

    fn cbrt(self) -> Self {
        self.powf(1.0 / 3.0)
    }

    fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            d_rho: self.d_rho / self.value,
            d_sigma: self.d_sigma / self.value,
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            d_rho: value * self.d_rho,
            d_sigma: value * self.d_sigma,
        }
    }
}

impl std::ops::Add for D {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            d_rho: self.d_rho + other.d_rho,
            d_sigma: self.d_sigma + other.d_sigma,
        }
    }
}

impl std::ops::Neg for D {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            d_rho: -self.d_rho,
            d_sigma: -self.d_sigma,
        }
    }
}

impl std::ops::Sub for D {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            d_rho: self.d_rho - other.d_rho,
            d_sigma: self.d_sigma - other.d_sigma,
        }
    }
}

impl std::ops::Mul for D {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            d_rho: self.d_rho * other.value + self.value * other.d_rho,
            d_sigma: self.d_sigma * other.value + self.value * other.d_sigma,
        }
    }
}

impl std::ops::Div for D {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let inv = 1.0 / other.value;
        Self {
            value,
            d_rho: (self.d_rho - value * other.d_rho) * inv,
            d_sigma: (self.d_sigma - value * other.d_sigma) * inv,
        }
    }
}

/// 每电子 LDA 交换能 `ε_x^LDA = −(3/4)(3/π)^{1/3}ρ^{1/3}`（对偶数版）。
fn exchange_lda(rho: D) -> D {
    D::constant(-0.75 * three_over_pi_cbrt()) * rho.cbrt()
}

/// B88 交换（`GgaB88X`）：`ε_x = ε_x^LDA·[1 + (β/C_x)·x²/(1 + 6β·x·asinh x)]`，
/// `x = sqrt(σ)/ρ^{4/3}`，`C_x = (3/4)(3/π)^{1/3}`。
///
/// 等价写法（更透明）：`E_x = E_x^LDA − β∫ρ^{4/3}x²/(1+6βx·asinh x)`。
/// `β = 0.0042` 是 Becke 1988 对氢原子交换能的拟合值。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct B88Exchange {
    /// `β = 0.0042`。
    pub beta: f64,
}

impl B88Exchange {
    /// 由数据参数构造。
    #[must_use]
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    /// 在 `(ρ, σ)` 上求「值 + 两个偏导」。
    fn evaluate(&self, rho: f64, sigma: f64) -> D {
        let rho = rho.max(crate::LOW_DENSITY_FLOOR);
        let sigma = sigma.max(0.0);
        let rho_d = D::rho(rho);
        let sigma_d = D::sigma(sigma);

        // x = |∇ρ| / ρ^{4/3}
        let x = sigma_d.powf(0.5) / rho_d.powf(4.0 / 3.0);
        // asinh(x) = ln(x + sqrt(x² + 1))
        let asinh = (x + (x * x + D::constant(1.0)).powf(0.5)).ln();
        let beta = D::constant(self.beta);
        // 修正项：−β·ρ^{1/3}·x²/(1 + 6β·x·asinh x)（每电子）
        let correction =
            beta * rho_d.cbrt() * x * x / (D::constant(1.0) + D::constant(6.0) * beta * x * asinh);
        exchange_lda(rho_d) - correction
    }
}

impl FunctionalTerm for B88Exchange {
    fn id(&self) -> &'static str {
        "GgaB88X"
    }

    fn requires(&self) -> Requirements {
        Requirements::gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "B88Exchange::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.evaluate(batch.rho[index], sigma).value;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "B88Exchange::d_rho").expect("shape");
        check_shape(batch, d_sigma, "B88Exchange::d_sigma").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let value = self.evaluate(batch.rho[index], sigma);
            d_rho[index] = value.d_rho;
            d_sigma[index] = value.d_sigma;
        }
    }
}
/// PBE 交换（`GgaPbeX`）。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PbeExchange {
    /// `κ = 0.804`。
    pub kappa: f64,
    /// `μ = 0.21951`。
    pub mu: f64,
}

impl Default for PbeExchange {
    fn default() -> Self {
        Self {
            kappa: 0.804,
            mu: 0.219_51,
        }
    }
}

impl PbeExchange {
    /// 由数据参数构造。
    #[must_use]
    pub fn new(kappa: f64, mu: f64) -> Self {
        Self { kappa, mu }
    }

    /// 在 `(ρ, σ)` 上求「值 + 两个偏导」。
    fn evaluate(&self, rho: f64, sigma: f64) -> D {
        let rho = rho.max(crate::LOW_DENSITY_FLOOR);
        let sigma = sigma.max(0.0);
        let rho_d = D::rho(rho);
        let sigma_d = D::sigma(sigma);

        // k_F² = (3π²ρ)^{2/3}
        let kf2 = (D::constant(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho_d)
            .powf(2.0 / 3.0);
        // s² = σ / (4 k_F² ρ²)
        let s2 = sigma_d / (D::constant(4.0) * kf2 * rho_d * rho_d);
        // F_x = 1 + κ − κ/(1 + μ s²/κ)
        let kappa = D::constant(self.kappa);
        let mu = D::constant(self.mu);
        let fx = D::constant(1.0) + kappa - kappa / (D::constant(1.0) + mu * s2 / kappa);
        exchange_lda(rho_d) * fx
    }
}

/// PBE 相关（`GgaPbeC`）。`para` 是 VWN5 的顺磁参数 `(A, x0, b, c)`。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PbeCorrelation {
    /// VWN5 顺磁参数（与 `LdaSvwn5` 同一套，保证 `σ→0` 时与 LDA 相关一致）。
    pub para: (f64, f64, f64, f64),
    /// `β = 0.06672455060314922`。
    pub beta: f64,
    /// `γ = (1 − ln2)/π²`。
    pub gamma: f64,
}

impl PbeCorrelation {
    /// 由数据参数构造。
    #[must_use]
    pub fn new(para: (f64, f64, f64, f64), beta: f64, gamma: f64) -> Self {
        Self { para, beta, gamma }
    }

    /// LDA 相关能（值 + 对 ρ 的解析导数）；`σ` 方向为 0。
    fn lda_correlation(&self, rho: f64) -> D {
        let (a, x0, b, c) = self.para;
        let rs = (3.0 / (4.0 * std::f64::consts::PI * rho)).cbrt();
        let x = rs.sqrt();
        let value = vwn_g(x, a, x0, b, c);
        // x = (3/(4πρ))^{1/6} ⇒ dx/dρ = −x/(6ρ)
        let d_x = -x / (6.0 * rho);
        let d_rho = vwn_g_prime(x, a, x0, b, c) * d_x;
        D::with_partials(value, d_rho, 0.0)
    }

    /// 在 `(ρ, σ)` 上求「值 + 两个偏导」。
    fn evaluate(&self, rho: f64, sigma: f64) -> D {
        let rho = rho.max(crate::LOW_DENSITY_FLOOR);
        let sigma = sigma.max(0.0);
        let rho_d = D::rho(rho);
        let sigma_d = D::sigma(sigma);

        let ec_unif = self.lda_correlation(rho);
        let beta = D::constant(self.beta);
        let gamma = D::constant(self.gamma);

        // k_F = (3π²ρ)^{1/3}; k_s² = 4k_F/π; t² = σ/(4 k_s² ρ²)
        let kf = (D::constant(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho_d)
            .powf(1.0 / 3.0);
        let ks2 = D::constant(4.0 / std::f64::consts::PI) * kf;
        let t2 = sigma_d / (D::constant(4.0) * ks2 * rho_d * rho_d);

        // A = (β/γ)/(exp(−ε_c^LDA/γ) − 1)
        let a_param = (beta / gamma) / ((-ec_unif / gamma).exp() - D::constant(1.0));
        let numerator = D::constant(1.0) + a_param * t2;
        let denominator = D::constant(1.0) + a_param * t2 + a_param * a_param * t2 * t2;
        let h = gamma * (D::constant(1.0) + (beta / gamma) * t2 * numerator / denominator).ln();
        ec_unif + h
    }
}

impl FunctionalTerm for PbeExchange {
    fn id(&self) -> &'static str {
        "GgaPbeX"
    }

    fn requires(&self) -> Requirements {
        Requirements::gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "PbeExchange::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.evaluate(batch.rho[index], sigma).value;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "PbeExchange::d_rho").expect("shape");
        check_shape(batch, d_sigma, "PbeExchange::d_sigma").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let value = self.evaluate(batch.rho[index], sigma);
            d_rho[index] = value.d_rho;
            d_sigma[index] = value.d_sigma;
        }
    }
}

impl FunctionalTerm for PbeCorrelation {
    fn id(&self) -> &'static str {
        "GgaPbeC"
    }

    fn requires(&self) -> Requirements {
        Requirements::gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "PbeCorrelation::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            *slot = self.evaluate(batch.rho[index], sigma).value;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        check_shape(batch, d_rho, "PbeCorrelation::d_rho").expect("shape");
        check_shape(batch, d_sigma, "PbeCorrelation::d_sigma").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let value = self.evaluate(batch.rho[index], sigma);
            d_rho[index] = value.d_rho;
            d_sigma[index] = value.d_sigma;
        }
    }
}

/// 由数据参数构造 PBE 的交换/相关项（供 [`crate::build_term`] 调用）。
///
/// # Errors
///
/// 缺参数。
pub fn build_pbe_term(
    id: &str,
    parameters: &BTreeMap<String, f64>,
) -> Result<Box<dyn FunctionalTerm>, XcError> {
    match id {
        "GgaPbeX" => Ok(Box::new(PbeExchange::new(
            parameter(id, parameters, "pbe_kappa")?,
            parameter(id, parameters, "pbe_mu")?,
        ))),
        "GgaB88X" => Ok(Box::new(B88Exchange::new(parameter(
            id, parameters, "b88_beta",
        )?))),
        "GgaPbeC" => Ok(Box::new(PbeCorrelation::new(
            (
                parameter(id, parameters, "vwn5_p_A")?,
                parameter(id, parameters, "vwn5_p_x0")?,
                parameter(id, parameters, "vwn5_p_b")?,
                parameter(id, parameters, "vwn5_p_c")?,
            ),
            parameter(id, parameters, "pbe_beta")?,
            parameter(id, parameters, "pbe_gamma")?,
        ))),
        other => Err(XcError::UnknownTerm(other.to_string())),
    }
}
