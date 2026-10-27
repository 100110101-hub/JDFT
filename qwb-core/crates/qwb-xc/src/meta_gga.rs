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

use crate::{DensityBatch, FunctionalTerm, Requirements, check_shape};

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

/// 三通道对偶数 `value + ∂ρ·δρ + ∂σ·δσ + ∂τ·δτ`。
#[derive(Debug, Clone, Copy)]
struct M {
    value: f64,
    d_rho: f64,
    d_sigma: f64,
    d_tau: f64,
}

impl M {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 0.0,
            d_tau: 0.0,
        }
    }

    fn rho(value: f64) -> Self {
        Self {
            value,
            d_rho: 1.0,
            d_sigma: 0.0,
            d_tau: 0.0,
        }
    }

    fn sigma(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 1.0,
            d_tau: 0.0,
        }
    }

    fn tau(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 0.0,
            d_tau: 1.0,
        }
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            d_rho: scale * self.d_rho,
            d_sigma: scale * self.d_sigma,
            d_tau: scale * self.d_tau,
        }
    }
}

impl std::ops::Add for M {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            d_rho: self.d_rho + other.d_rho,
            d_sigma: self.d_sigma + other.d_sigma,
            d_tau: self.d_tau + other.d_tau,
        }
    }
}

impl std::ops::Sub for M {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            d_rho: self.d_rho - other.d_rho,
            d_sigma: self.d_sigma - other.d_sigma,
            d_tau: self.d_tau - other.d_tau,
        }
    }
}

impl std::ops::Mul for M {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            d_rho: self.d_rho * other.value + self.value * other.d_rho,
            d_sigma: self.d_sigma * other.value + self.value * other.d_sigma,
            d_tau: self.d_tau * other.value + self.value * other.d_tau,
        }
    }
}

impl std::ops::Div for M {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let inverse = 1.0 / other.value;
        Self {
            value,
            d_rho: (self.d_rho - value * other.d_rho) * inverse,
            d_sigma: (self.d_sigma - value * other.d_sigma) * inverse,
            d_tau: (self.d_tau - value * other.d_tau) * inverse,
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
fn gtv4(alpha: f64, d: &[f64; 6], w: M, z: M) -> M {
    let one = M::constant(1.0);
    let gamma = one + M::constant(alpha) * (w + z);
    let g2 = gamma * gamma;
    let g3 = g2 * gamma;
    M::constant(d[0]) / gamma
        + (M::constant(d[1]) * w + M::constant(d[2]) * z) / g2
        + (M::constant(d[3]) * w * w + M::constant(d[4]) * w * z + M::constant(d[5]) * z * z) / g3
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
        let rho_d = M::rho(rho);
        let scale = M::constant(spin_scale());
        let rs = (M::constant(3.0 / (4.0 * std::f64::consts::PI)) / rho_d).powf(1.0 / 3.0);
        let xs_sq = scale * M::sigma(sigma) / rho_d.powf(8.0 / 3.0);
        let t = scale * M::tau(tau) / rho_d.powf(5.0 / 3.0);
        // 每自旋 LDA 交换（每总粒子）：−(3/4)(9/(2π²))^{1/3}·(1/2)^{4/3}/rs
        let c = 0.75 * (9.0 / (2.0 * std::f64::consts::PI * std::f64::consts::PI)).powf(1.0 / 3.0);
        let lda = M::constant(-c * 0.5_f64.powf(4.0 / 3.0)) / rs;
        let value = M::constant(2.0) * lda * self.enhancement(xs_sq, t);
        (value.value, value.d_rho, value.d_sigma, value.d_tau)
    }
}

impl FunctionalTerm for M06Exchange {
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
