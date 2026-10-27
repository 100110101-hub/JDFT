//! 自旋分辨 GGA（UKS 的 XC 侧，SC-01b）：PBE 相关的 ζ 插值与梯度修正。
//!
//! # 为什么换一套参数化
//!
//! 闭壳层 PBE 相关（[`super::gga`]）在 `(ρ, σ)` 上用对偶数求值，一份表达式同时给出
//! 值与两个偏导。自旋版本多出一个 `ζ` 维度，而 `ζ` 只通过
//! `ε_c^unif(ρ,ζ)`、`φ(ζ)` 两处进入 —— 于是把自变量换成
//!
//! ```text
//! (ρ, ζ, g)，  g = |∇ρ|² = σ_αα + σ_ββ + 2σ_αβ
//! ```
//!
//! 再按链式法则落到 `(ρ_α, ρ_β, σ_αα, σ_ββ, σ_αβ)`：
//!
//! ```text
//! ∂/∂ρ_α = ∂/∂ρ + (1−ζ)/ρ · ∂/∂ζ
//! ∂/∂ρ_β = ∂/∂ρ − (1+ζ)/ρ · ∂/∂ζ
//! ∂/∂σ_αα = ∂/∂σ_ββ = ∂/∂g，   ∂/∂σ_αβ = 2·∂/∂g
//! ```
//!
//! 这样只需要三通道的对偶数，公式与闭壳层一一对应，**ζ = 0 时逐位退化**到闭壳层
//! （测试里钉住这一点）。
//!
//! # 公式（Perdew–Burke–Ernzerhof 1996，自旋极化）
//!
//! ```text
//! ε_c^unif(ρ,ζ) = ε_c^P(1 − f ζ⁴) + ε_c^F f ζ⁴ + (α_c/f″(0))·f·(1 − ζ⁴)
//! f(ζ)  = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2)，f″(0) = 8/(9(2^{4/3}−2))
//! φ(ζ)  = [(1+ζ)^{2/3} + (1−ζ)^{2/3}] / 2
//! t²    = g / (4 φ² k_s² ρ²)，k_s² = 4k_F/π
//! A     = (β/γ) / (exp(−ε_c^unif/(γ φ³)) − 1)
//! H     = γ φ³ ln[1 + (β/γ) t² (1 + A t²)/(1 + A t² + A² t⁴)]
//! e     = ρ (ε_c^unif + H)            ← 每体积能量密度
//! ```
//!
//! **φ³ 与 φ² 不能漏**：漏了 `φ³`（`H` 与 `A` 里的那个）在 ζ = 0 时看不出来，
//! 但在 O2 这种强极化体系上能量会偏 1e-2 Ha 量级 —— 逐点对照 libxc 的
//! `tests/reference/spin_xc_pointwise.csv` 就是为了钉住它。
//!
//! # ζ = ±1 的数值处理
//!
//! `φ'(ζ)` 在 ζ → ±1 时以 `(1∓ζ)^{−1/3}` 发散（全极化点上是真实发散）。
//! 网格里 `ρ_β` 下溢成 0 的点会精确落在 ζ = 1 上，那里导数无界，
//! 所以 ζ 被夹到 `[−1+ε, 1−ε]`（`ε = 1e-12`）—— 与 `LOW_DENSITY_FLOOR`
//! 同类的数值保护，记录在 `docs/choices.md`。
#![deny(unsafe_code)]

use crate::gga::PbeCorrelation;
use crate::{LOW_DENSITY_FLOOR, SpinDensityBatch, SpinXcOutput, XcError, check_spin_output};

/// ζ 的夹紧量（见模块文档的"数值处理"）。
const ZETA_EPS: f64 = 1.0e-12;

/// `f″(0) = 8 / (9 (2^{4/3} − 2))`。
fn f_second_derivative_zero() -> f64 {
    8.0 / (9.0 * (2.0_f64.powf(4.0 / 3.0) - 2.0))
}

/// 三通道对偶数：`value + ∂ρ·δρ + ∂ζ·δζ + ∂g·δg`。
#[derive(Debug, Clone, Copy)]
struct J3 {
    value: f64,
    d_rho: f64,
    d_zeta: f64,
    d_g: f64,
}

impl J3 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_zeta: 0.0,
            d_g: 0.0,
        }
    }

    fn rho(value: f64) -> Self {
        Self {
            value,
            d_rho: 1.0,
            d_zeta: 0.0,
            d_g: 0.0,
        }
    }

    fn zeta(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_zeta: 1.0,
            d_g: 0.0,
        }
    }

    fn g(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_zeta: 0.0,
            d_g: 1.0,
        }
    }

    /// 已知值 + 已知对 ρ 的偏导（PW92 的三组量已经解析求过导）。
    fn with_rho_derivative(value: f64, d_rho: f64) -> Self {
        Self {
            value,
            d_rho,
            d_zeta: 0.0,
            d_g: 0.0,
        }
    }

    /// 幂函数（底数必须 > 0：调用方保证 ζ 被夹紧、ρ 有下限）。
    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            d_rho: scale * self.d_rho,
            d_zeta: scale * self.d_zeta,
            d_g: scale * self.d_g,
        }
    }

    fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            d_rho: self.d_rho / self.value,
            d_zeta: self.d_zeta / self.value,
            d_g: self.d_g / self.value,
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            d_rho: value * self.d_rho,
            d_zeta: value * self.d_zeta,
            d_g: value * self.d_g,
        }
    }
}

impl std::ops::Add for J3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            d_rho: self.d_rho + other.d_rho,
            d_zeta: self.d_zeta + other.d_zeta,
            d_g: self.d_g + other.d_g,
        }
    }
}

impl std::ops::Sub for J3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            d_rho: self.d_rho - other.d_rho,
            d_zeta: self.d_zeta - other.d_zeta,
            d_g: self.d_g - other.d_g,
        }
    }
}

impl std::ops::Mul for J3 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            d_rho: self.d_rho * other.value + self.value * other.d_rho,
            d_zeta: self.d_zeta * other.value + self.value * other.d_zeta,
            d_g: self.d_g * other.value + self.value * other.d_g,
        }
    }
}

impl std::ops::Div for J3 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let inverse = 1.0 / other.value;
        Self {
            value,
            d_rho: (self.d_rho - value * other.d_rho) * inverse,
            d_zeta: (self.d_zeta - value * other.d_zeta) * inverse,
            d_g: (self.d_g - value * other.d_g) * inverse,
        }
    }
}

/// 单点的六个结果（每体积能量密度、两个势、三个 σ 导数）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SpinPoint {
    energy: f64,
    v_alpha: f64,
    v_beta: f64,
    d_sigma_alpha: f64,
    d_sigma_beta: f64,
    d_sigma_alpha_beta: f64,
}

impl SpinPoint {
    const ZERO: Self = Self {
        energy: 0.0,
        v_alpha: 0.0,
        v_beta: 0.0,
        d_sigma_alpha: 0.0,
        d_sigma_beta: 0.0,
        d_sigma_alpha_beta: 0.0,
    };
}

/// PBE 相关的单点求值：`(e, v_α, v_β, ∂e/∂σ_αα, ∂e/∂σ_ββ, ∂e/∂σ_αβ)`。
fn pbe_correlation_point(
    term: &PbeCorrelation,
    rho_alpha: f64,
    rho_beta: f64,
    sigma_alpha: f64,
    sigma_beta: f64,
    sigma_alpha_beta: f64,
) -> SpinPoint {
    let alpha = rho_alpha.max(0.0);
    let beta = rho_beta.max(0.0);
    let total = alpha + beta;
    if total <= LOW_DENSITY_FLOOR {
        return SpinPoint::ZERO;
    }
    let zeta = ((alpha - beta) / total).clamp(-1.0 + ZETA_EPS, 1.0 - ZETA_EPS);
    let gradient_squared =
        (sigma_alpha.max(0.0) + sigma_beta.max(0.0) + 2.0 * sigma_alpha_beta).max(0.0);

    let rho = J3::rho(total);
    let zeta_d = J3::zeta(zeta);
    let g = J3::g(gradient_squared);

    // ── ε_c^unif(ρ, ζ) ────────────────────────────────────────────
    let rs = (3.0 / (4.0 * std::f64::consts::PI * total)).cbrt();
    let drs_drho = -rs / (3.0 * total);
    let (eps_para, d_para_drs) = term.pw92_para.energy_and_derivative(rs);
    let (eps_ferro, d_ferro_drs) = term.pw92_ferro.energy_and_derivative(rs);
    let (alpha_c, d_alpha_c_drs) = term.pw92_alpha.stiffness_and_derivative(rs);
    let para = J3::with_rho_derivative(eps_para, d_para_drs * drs_drho);
    let ferro = J3::with_rho_derivative(eps_ferro, d_ferro_drs * drs_drho);
    let stiffness = J3::with_rho_derivative(alpha_c, d_alpha_c_drs * drs_drho);

    // f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2)
    let one = J3::constant(1.0);
    let four_thirds = 4.0 / 3.0;
    let two_to_four_thirds = 2.0_f64.powf(4.0 / 3.0);
    let f_zeta = ((one + zeta_d).powf(four_thirds) + (one - zeta_d).powf(four_thirds)
        - J3::constant(2.0))
        / J3::constant(two_to_four_thirds - 2.0);
    let zeta_fourth = zeta_d.powf(4.0);
    let eps_uniform = para * (one - f_zeta * zeta_fourth)
        + ferro * f_zeta * zeta_fourth
        + stiffness / J3::constant(f_second_derivative_zero()) * f_zeta * (one - zeta_fourth);

    // ── 梯度修正 H ────────────────────────────────────────────────
    // φ(ζ) = [(1+ζ)^{2/3} + (1−ζ)^{2/3}]/2
    let two_thirds = 2.0 / 3.0;
    let phi =
        ((one + zeta_d).powf(two_thirds) + (one - zeta_d).powf(two_thirds)) / J3::constant(2.0);
    let phi_cubed = phi.powf(3.0);
    // k_F = (3π²ρ)^{1/3}，k_s² = 4k_F/π，t² = g/(4 φ² k_s² ρ²)
    let kf =
        (J3::constant(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(1.0 / 3.0);
    let ks2 = J3::constant(4.0 / std::f64::consts::PI) * kf;
    let t_squared = g / (J3::constant(4.0) * phi.powf(2.0) * ks2 * rho * rho);

    let beta_parameter = J3::constant(term.beta);
    let gamma = J3::constant(term.gamma);
    // A = (β/γ) / (exp(−ε_c^unif/(γ φ³)) − 1)
    let a_parameter = (beta_parameter / gamma)
        / ((J3::constant(-1.0) * eps_uniform / (gamma * phi_cubed)).exp() - one);
    let numerator = one + a_parameter * t_squared;
    let denominator =
        one + a_parameter * t_squared + a_parameter * a_parameter * t_squared * t_squared;
    let h = gamma
        * phi_cubed
        * (one + (beta_parameter / gamma) * t_squared * numerator / denominator).ln();

    // 每体积能量密度 e = ρ(ε_c^unif + H)
    let energy = rho * (eps_uniform + h);

    // ── 链式法则：(ρ, ζ, g) → (ρ_α, ρ_β, σ) ──────────────────────
    let d_zeta_d_alpha = (1.0 - zeta) / total;
    let d_zeta_d_beta = -(1.0 + zeta) / total;
    SpinPoint {
        energy: energy.value,
        v_alpha: energy.d_rho + energy.d_zeta * d_zeta_d_alpha,
        v_beta: energy.d_rho + energy.d_zeta * d_zeta_d_beta,
        d_sigma_alpha: energy.d_g,
        d_sigma_beta: energy.d_g,
        d_sigma_alpha_beta: 2.0 * energy.d_g,
    }
}

/// 把 [`pbe_correlation_point`] 铺到整批网格点上。
pub(crate) fn evaluate_pbe_correlation(
    term: &PbeCorrelation,
    batch: &SpinDensityBatch<'_>,
    out: &mut SpinXcOutput<'_>,
) -> Result<(), XcError> {
    let points = batch.len();
    check_spin_output(points, out, "GgaPbeC::evaluate_spin")?;
    for index in 0..points {
        let point = pbe_correlation_point(
            term,
            batch.rho_alpha[index],
            batch.rho_beta[index],
            batch.sigma_alpha_at(index),
            batch.sigma_beta_at(index),
            batch.sigma_alpha_beta_at(index),
        );
        out.energy[index] = point.energy;
        out.v_alpha[index] = point.v_alpha;
        out.v_beta[index] = point.v_beta;
        out.d_sigma_alpha[index] = point.d_sigma_alpha;
        out.d_sigma_beta[index] = point.d_sigma_beta;
        out.d_sigma_alpha_beta[index] = point.d_sigma_alpha_beta;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term() -> PbeCorrelation {
        PbeCorrelation::new(
            (0.031_090_7, -0.409_286, 13.072, 42.719_8),
            0.066_724_550_603_149_22,
            0.031_090_690_869_654_9,
        )
    }

    /// ζ = 0 时自旋版本必须与闭壳层版本逐位一致（含 σ 通道）。
    #[test]
    fn unpolarized_reduces_to_closed_shell() {
        let term = term();
        for (rho, sigma) in [(0.6, 0.02), (0.1, 0.005), (2.0, 0.4)] {
            let point = pbe_correlation_point(
                &term,
                0.5 * rho,
                0.5 * rho,
                0.25 * sigma,
                0.25 * sigma,
                0.25 * sigma,
            );
            let mut epsilon = [0.0_f64];
            let mut d_rho = [0.0_f64];
            let mut d_sigma = [0.0_f64];
            let rho_holder = [rho];
            let sigma_holder = [sigma];
            let batch = crate::DensityBatch::new(&rho_holder, &sigma_holder);
            use crate::FunctionalTerm;
            term.energy_density(&batch, &mut epsilon);
            term.potential_derivatives(&batch, &mut d_rho, &mut d_sigma);
            // e = ρ ε；v_σ = ε + ρ dε/dρ（ζ = 0 时两个通道相同）
            let expected_energy = rho * epsilon[0];
            assert!(
                (point.energy - expected_energy).abs() < 1e-12,
                "ρ = {rho}: 自旋 {:.15} vs 闭壳层 {expected_energy:.15}",
                point.energy
            );
            for (what, got) in [("v_alpha", point.v_alpha), ("v_beta", point.v_beta)] {
                let expected = epsilon[0] + rho * d_rho[0];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "ρ = {rho} σ = {sigma}: {what} {got:.12} vs 闭壳层 {expected:.12}"
                );
            }
            // ∂e/∂σ_αα = ρ dε/dσ · ... 闭壳层里 e 对 σ 的导数就是 ρ dε/dσ
            let expected_sigma = rho * d_sigma[0];
            assert!(
                (point.d_sigma_alpha - expected_sigma).abs() < 1e-10,
                "ρ = {rho} σ = {sigma}: dσ_αα {:.12} vs 闭壳层 {expected_sigma:.12}",
                point.d_sigma_alpha
            );
        }
    }

    /// 导数必须与中心差分一致（X-02 的判据，这里是自旋版本）。
    #[test]
    fn derivatives_match_central_differences() {
        let term = term();
        let step = 1.0e-6;
        for (rho_a, rho_b, sig_a, sig_b, sig_ab) in [
            (0.3, 0.1, 0.05, 0.02, 0.01),
            (0.4, 0.05, 0.2, 0.1, -0.03),
            (0.2, 0.2, 0.01, 0.01, 0.0),
        ] {
            let point = pbe_correlation_point(&term, rho_a, rho_b, sig_a, sig_b, sig_ab);
            let fd = |delta: f64| {
                let plus =
                    pbe_correlation_point(&term, rho_a + delta, rho_b, sig_a, sig_b, sig_ab).energy;
                let minus =
                    pbe_correlation_point(&term, rho_a - delta, rho_b, sig_a, sig_b, sig_ab).energy;
                (plus - minus) / (2.0 * delta)
            };
            assert!(
                (point.v_alpha - fd(step)).abs() < 1e-8,
                "v_alpha {:.12} vs 差分 {:.12}",
                point.v_alpha,
                fd(step)
            );
            let fd_beta = |delta: f64| {
                let plus =
                    pbe_correlation_point(&term, rho_a, rho_b + delta, sig_a, sig_b, sig_ab).energy;
                let minus =
                    pbe_correlation_point(&term, rho_a, rho_b - delta, sig_a, sig_b, sig_ab).energy;
                (plus - minus) / (2.0 * delta)
            };
            assert!(
                (point.v_beta - fd_beta(step)).abs() < 1e-8,
                "v_beta {:.12} vs 差分 {:.12}",
                point.v_beta,
                fd_beta(step)
            );
            let fd_sigma = |delta: f64| {
                let plus =
                    pbe_correlation_point(&term, rho_a, rho_b, sig_a + delta, sig_b, sig_ab).energy;
                let minus =
                    pbe_correlation_point(&term, rho_a, rho_b, sig_a - delta, sig_b, sig_ab).energy;
                (plus - minus) / (2.0 * delta)
            };
            assert!(
                (point.d_sigma_alpha - fd_sigma(step)).abs() < 1e-8,
                "d_sigma_alpha {:.12} vs 差分 {:.12}",
                point.d_sigma_alpha,
                fd_sigma(step)
            );
        }
    }
}
