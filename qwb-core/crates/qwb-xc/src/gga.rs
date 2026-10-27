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
    DensityBatch, FunctionalTerm, Requirements, XcError, check_shape, parameter, three_over_pi_cbrt,
};

/// 对偶数：`value + ∂ρ·δρ + ∂σ·δσ + ∂∇²ρ·δ∇²ρ`。
///
/// 第三个通道是 LYP 一类依赖 `∇²ρ` 的泛函要的。`evaluate_xc` 会把它转成
/// `+ ρ∂ε/∂∇²ρ · ∇²(φ_μ φ_ν)` 的矩阵元。
#[derive(Debug, Clone, Copy)]
struct D {
    value: f64,
    d_rho: f64,
    d_sigma: f64,
    d_laplacian: f64,
}

impl D {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 0.0,
            d_laplacian: 0.0,
        }
    }

    fn rho(value: f64) -> Self {
        Self {
            value,
            d_rho: 1.0,
            d_sigma: 0.0,
            d_laplacian: 0.0,
        }
    }

    fn sigma(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 1.0,
            d_laplacian: 0.0,
        }
    }

    fn laplacian(value: f64) -> Self {
        Self {
            value,
            d_rho: 0.0,
            d_sigma: 0.0,
            d_laplacian: 1.0,
        }
    }

    /// 已知值 + 已知偏导（用于 VWN5 这类已有解析导数的子表达式）。
    fn with_partials(value: f64, d_rho: f64, d_sigma: f64) -> Self {
        Self {
            value,
            d_rho,
            d_sigma,
            d_laplacian: 0.0,
        }
    }

    /// 幂函数。
    ///
    /// **注意**：底数为 0 且 `exponent < 1` 时，导数是无穷（`0^{e-1} = ∞`），
    /// 对偶数会把它变成 NaN。热路径上不要用 `σ.powf(0.5)` 这种写法——改用
    /// `t = σ/ρ^{8/3}` 这类对 `σ` 线性的中间量（见 [`B88Exchange::evaluate`]）。
    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            d_rho: scale * self.d_rho,
            d_sigma: scale * self.d_sigma,
            d_laplacian: scale * self.d_laplacian,
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
            d_laplacian: self.d_laplacian / self.value,
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            d_rho: value * self.d_rho,
            d_sigma: value * self.d_sigma,
            d_laplacian: value * self.d_laplacian,
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
            d_laplacian: self.d_laplacian + other.d_laplacian,
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
            d_laplacian: -self.d_laplacian,
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
            d_laplacian: self.d_laplacian - other.d_laplacian,
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
            d_laplacian: self.d_laplacian * other.value + self.value * other.d_laplacian,
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
            d_laplacian: (self.d_laplacian - value * other.d_laplacian) * inv,
        }
    }
}

/// PW92 的一组参数 `(A, α, β₁..β₄)`（Perdew & Wang, PRB 45, 13244）。
///
/// 同一套函数形状 `ε(rs) = −2A(1+α rs) ln[1 + 1/(2A β(rs))]`，
/// `β(rs) = β₁√rs + β₂ rs + β₃ rs^{3/2} + β₄ rs²`，三组参数分别给出
/// 顺磁（ζ=0）、铁磁（ζ=1）与自旋刚度 `α_c`（PW92 的 ζ⁴ 插值要用它）。
///
/// **为什么单独一个类型**：闭壳层 PBE 相关只需要顺磁那一组（过去是三个 const
/// 写死在 [`PbeCorrelation::lda_correlation`] 里）；自旋情形要三组一起用，
/// 写成类型才能保证"三组用的是同一个公式"这件事只有一处实现。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Pw92Set {
    /// `A`（Hartree）。
    pub a: f64,
    /// `α`。
    pub alpha: f64,
    /// `β₁..β₄`。
    pub betas: [f64; 4],
}

impl Pw92Set {
    /// 顺磁（ζ = 0）参数集。
    pub const PARA: Self = Self {
        a: 0.031_090_7,
        alpha: 0.213_70,
        betas: [7.5957, 3.5876, 1.638_2, 0.492_94],
    };

    /// 铁磁（ζ = 1）参数集。
    pub const FERRO: Self = Self {
        a: 0.015_545_35,
        alpha: 0.205_48,
        betas: [14.1189, 6.1977, 3.366_2, 0.625_17],
    };

    /// `α_c`（自旋刚度）参数集。
    pub const ALPHA: Self = Self {
        a: 0.016_886_9,
        alpha: 0.111_25,
        betas: [10.357, 3.6231, 0.880_26, 0.496_71],
    };

    /// 由数据参数（`{prefix}_A`/`{prefix}_alpha`/`{prefix}_b1..b4`）读一组。
    ///
    /// # Errors
    ///
    /// 缺参数。
    pub fn from_parameters(
        component: &str,
        parameters: &BTreeMap<String, f64>,
        prefix: &str,
    ) -> Result<Self, XcError> {
        let mut betas = [0.0_f64; 4];
        for (index, slot) in betas.iter_mut().enumerate() {
            *slot = parameter(component, parameters, &format!("{prefix}_b{}", index + 1))?;
        }
        Ok(Self {
            a: parameter(component, parameters, &format!("{prefix}_A"))?,
            alpha: parameter(component, parameters, &format!("{prefix}_alpha"))?,
            betas,
        })
    }

    /// `(ε(rs), dε/drs)`。
    #[must_use]
    pub fn energy_and_derivative(&self, rs: f64) -> (f64, f64) {
        let (a, alpha, betas) = (self.a, self.alpha, self.betas);
        let root = rs.sqrt();
        let b = betas[0] * root + betas[1] * rs + betas[2] * rs * root + betas[3] * rs * rs;
        let q1 = 2.0 * a * b;
        let log = (1.0 + 1.0 / q1).ln();
        let value = -2.0 * a * (1.0 + alpha * rs) * log;
        // dε/dr_s = −2Aα·ln(1+1/q1) + (2A(1+α r_s))·q1′/(q1²+q1)
        let db = betas[0] / (2.0 * root) + betas[1] + 1.5 * betas[2] * root + 2.0 * betas[3] * rs;
        let q1_prime = 2.0 * a * db;
        let derivative =
            -2.0 * a * alpha * log + 2.0 * a * (1.0 + alpha * rs) * q1_prime / (q1 * q1 + q1);
        (value, derivative)
    }

    /// `(α_c(rs), dα_c/drs)`：PW92 的 α 参数集用同一个形状，但取**正**。
    #[must_use]
    pub fn stiffness_and_derivative(&self, rs: f64) -> (f64, f64) {
        let (value, derivative) = self.energy_and_derivative(rs);
        (-value, -derivative)
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
    ///
    /// # 为什么用 `t = σ/ρ^{8/3}` 而不是 `x = sqrt(σ)/ρ^{4/3}`
    ///
    /// `x` 对 `σ` 的导数是 `1/(2·sqrt(σ))`，在 `σ = 0` 处发散。对偶数会把这个发散带下去：
    /// `x·x` 的导数变成 `2·0·∞ = NaN`，尽管 `x²` 对 `σ` 是**线性**的、导数有界。
    /// 真实 SCF 里远处网格点的 `σ` 会**精确**等于 0（AO 梯度下溢），于是 B88 的导数是 NaN，
    /// 顺着 v_xc 污染整个 Fock —— 这正是 B3LYP 此前在 grid-coarse 上 NaN 的根因。
    ///
    /// 所以对偶数只承载 `t = x²`（对 `σ` 线性、无奇点），`A(x) = x·asinh(x)` 用解析导数喂进去：
    /// `dA/dt = 1/(2√(1+x²)) + asinh(x)/(2x)`，`x → 0` 时极限为 1。
    /// 单位检查：`t` 是 `σ/ρ^{8/3}`，`dA/dt` 无量纲 ✓。
    fn evaluate(&self, rho: f64, sigma: f64) -> D {
        let rho = rho.max(crate::LOW_DENSITY_FLOOR);
        let sigma = sigma.max(0.0);
        let rho_d = D::rho(rho);
        let sigma_d = D::sigma(sigma);

        // t = x² = σ/ρ^{8/3}（**总密度**的约化梯度平方）
        let t = sigma_d / rho_d.powf(8.0 / 3.0);
        // B88 的原始形式用的是**每个自旋**的约化梯度 x_σ = |∇ρ_σ|/ρ_σ^{4/3}。
        // 闭壳层 ρ_σ = ρ/2 ⇒ x_σ = 2^{1/3}·x_total，即 x_σ² = 2^{2/3}·t。
        // 系数里的 X_FACTOR_C = (3/4)(6/π)^{1/3} = 0.9305257 也是按每自旋定义的；
        // 漏掉这两个 2 的幂会让 B88 的梯度修正弱 1.7 倍（BLYP 在水上偏高 112 mHa 的原因）。
        let c2 = 2.0_f64.cbrt();
        let sqrt_t = t.value.max(0.0).sqrt();
        let x_spin_value = c2 * sqrt_t;
        let asinh_x = x_spin_value.asinh();
        let a_value = x_spin_value * asinh_x;
        let d_a_dt = if x_spin_value > 1.0e-12 {
            (asinh_x + x_spin_value / (1.0 + x_spin_value * x_spin_value).sqrt()) * c2
                / (2.0 * sqrt_t)
        } else {
            c2 * c2
        };
        let a = D::with_partials(a_value, d_a_dt * t.d_rho, d_a_dt * t.d_sigma);
        let beta = D::constant(self.beta);
        let c_x = D::constant(0.75 * three_over_pi_cbrt());
        let x_factor_c = D::constant(0.930_525_736_349_1);
        // ε = −ρ^{1/3}[C_x + (β/X_FACTOR_C)·t_σ/(1 + 6β·A)]，t_σ = 2^{2/3}t
        // 注意 c_x 这一因子：B88 的写法是 ε = −C_xρ^{1/3}[1 + (β/X_FACTOR_C)x_σ²/(…)]，
        // 展开后修正项还要乘 C_x，漏掉它会强 1/0.7386 倍。
        let correction = beta * c_x * D::constant(c2 * c2) / x_factor_c * t
            / (D::constant(1.0) + D::constant(6.0) * beta * a);
        -rho_d.cbrt() * (c_x + correction)
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

/// LYP 相关（`GgaLypC`）——Lee–Yang–Parr 1988。
///
/// 闭壳层形式（`ρ` 为总密度、`σ = |∇ρ|²`、`L = ∇²ρ`）：
///
/// ```text
/// ε_c = −a·[1 + b·ρ^{−5/3}·Br·e^{−c·ρ^{−1/3}}] / (1 + d·ρ^{−1/3})
/// Br  = C_F·ρ^{5/3} + (7/24)·L − (17/72)·σ/ρ,   C_F = (3/10)(3π²)^{2/3}
/// a = 0.04918, b = 0.132, c = 0.2533, d = 0.349
/// ```
///
/// 等价写法（更容易核对来源）：
/// `Br = C_F ρ^{5/3} − 2t_W + (1/9)(t_W + ½L)`，`t_W = (1/8)(σ/ρ − L)`；
/// 展开后 `L` 的系数是 `17/72 + 1/18 = 7/24`、`σ/ρ` 的系数是 `−17/72`。
/// 这两个系数在多个独立来源里以 `+7/24` / `−17/72` 的形式出现，是核对公式的锚点。
///
/// **注意**：LYP 是唯一需要 `∇²ρ` 的项，所以它必须同时覆盖三通道导数
/// （`potential_derivatives_laplacian`），否则 SCF 的 v_xc 会缺一整项。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LypCorrelation {
    /// `a = 0.04918`。
    pub a: f64,
    /// `b = 0.132`。
    pub b: f64,
    /// `c = 0.2533`。
    pub c: f64,
    /// `d = 0.349`。
    pub d: f64,
}

impl LypCorrelation {
    /// 由数据参数构造。
    #[must_use]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// 在 `(ρ, σ, ∇²ρ)` 上求「值 + 三个偏导」。
    fn evaluate(&self, rho: f64, sigma: f64, laplacian: f64) -> D {
        let rho = rho.max(crate::LOW_DENSITY_FLOOR);
        let sigma = sigma.max(0.0);
        let rho_d = D::rho(rho);
        let sigma_d = D::sigma(sigma);
        let laplacian_d = D::laplacian(laplacian);

        // Br = C_F ρ^{5/3} + (7/24)L − (17/72)σ/ρ
        let c_f =
            D::constant(0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0));
        let br = c_f * rho_d.powf(5.0 / 3.0) + D::constant(7.0 / 24.0) * laplacian_d
            - D::constant(17.0 / 72.0) * sigma_d / rho_d;

        let exponential = (D::constant(-self.c) * rho_d.powf(-1.0 / 3.0)).exp();
        let numerator =
            D::constant(1.0) + D::constant(self.b) * rho_d.powf(-5.0 / 3.0) * br * exponential;
        let denominator = D::constant(1.0) + D::constant(self.d) * rho_d.powf(-1.0 / 3.0);
        D::constant(-self.a) * numerator / denominator
    }
}

impl FunctionalTerm for LypCorrelation {
    fn id(&self) -> &'static str {
        "GgaLypC"
    }

    fn requires(&self) -> Requirements {
        Requirements::laplacian_gga()
    }

    fn spin_scaling(&self) -> f64 {
        1.0
    }

    fn energy_density(&self, batch: &DensityBatch<'_>, out: &mut [f64]) {
        check_shape(batch, out, "LypCorrelation::energy_density").expect("shape");
        for (index, slot) in out.iter_mut().enumerate() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let laplacian = batch.laplacian.get(index).copied().unwrap_or(0.0);
            *slot = self.evaluate(batch.rho[index], sigma, laplacian).value;
        }
    }

    fn potential_derivatives(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
    ) {
        // 只填前两通道；第三通道由 potential_derivatives_laplacian 覆盖
        let mut d_laplacian = vec![0.0; batch.len()];
        self.potential_derivatives_laplacian(batch, d_rho, d_sigma, &mut d_laplacian);
    }

    fn potential_derivatives_laplacian(
        &self,
        batch: &DensityBatch<'_>,
        d_rho: &mut [f64],
        d_sigma: &mut [f64],
        d_laplacian: &mut [f64],
    ) {
        check_shape(batch, d_rho, "LypCorrelation::d_rho").expect("shape");
        check_shape(batch, d_sigma, "LypCorrelation::d_sigma").expect("shape");
        check_shape(batch, d_laplacian, "LypCorrelation::d_laplacian").expect("shape");
        for index in 0..batch.len() {
            let sigma = batch.sigma.get(index).copied().unwrap_or(0.0);
            let laplacian = batch.laplacian.get(index).copied().unwrap_or(0.0);
            let value = self.evaluate(batch.rho[index], sigma, laplacian);
            d_rho[index] = value.d_rho;
            d_sigma[index] = value.d_sigma;
            d_laplacian[index] = value.d_laplacian;
        }
    }
}

/// PBE 相关（`GgaPbeC`）。`para` 是 VWN5 的顺磁参数 `(A, x0, b, c)`。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PbeCorrelation {
    /// VWN5 顺磁参数（数据里的 `vwn5_p_*`；闭壳层路径不直接用，留作交叉核对）。
    pub para: (f64, f64, f64, f64),
    /// `β = 0.06672455060314922`。
    pub beta: f64,
    /// `γ = (1 − ln2)/π²`。
    pub gamma: f64,
    /// PW92 顺磁参数集（闭壳层 `ε_c^LDA` 就是这一组）。
    pub pw92_para: Pw92Set,
    /// PW92 铁磁参数集（ζ = 1）。
    pub pw92_ferro: Pw92Set,
    /// PW92 `α_c` 参数集（ζ⁴ 插值里的自旋刚度）。
    pub pw92_alpha: Pw92Set,
}

impl PbeCorrelation {
    /// 由数据参数构造（PW92 三组参数取 PW92 原文的默认值）。
    #[must_use]
    pub fn new(para: (f64, f64, f64, f64), beta: f64, gamma: f64) -> Self {
        Self {
            para,
            beta,
            gamma,
            pw92_para: Pw92Set::PARA,
            pw92_ferro: Pw92Set::FERRO,
            pw92_alpha: Pw92Set::ALPHA,
        }
    }

    /// 换上数据文件给的 PW92 三组参数。
    #[must_use]
    pub fn with_pw92_sets(mut self, para: Pw92Set, ferro: Pw92Set, alpha: Pw92Set) -> Self {
        self.pw92_para = para;
        self.pw92_ferro = ferro;
        self.pw92_alpha = alpha;
        self
    }

    /// LDA 相关能（值 + 对 ρ 的解析导数）；`σ` 方向为 0。
    ///
    /// **PBE 的 LDA 基是 PW92 的 modified 版本**（Perdew & Wang, PRB 45, 13244），
    /// 不是 SVWN5 用的 VWN5：libxc 的 `gga_c_pbe.c` 明确用 `XC_LDA_C_PW_MOD`。
    /// 两者在 r_s ≈ 1 处差约 0.02 Ha/e，水（10 电子）上就是约 0.18 Ha 的总能量差 ——
    /// 这正是 PBE 一开始比 CCCBDB 低 176 mHa 的原因（LSDA 仍用 VWN5，所以 LSDA 一直是对的）。
    fn lda_correlation(&self, rho: f64) -> D {
        let rs = (3.0 / (4.0 * std::f64::consts::PI * rho)).cbrt();
        let (value, d_rs) = self.pw92_para.energy_and_derivative(rs);
        // r_s = (3/(4πρ))^{1/3} ⇒ dr_s/dρ = −r_s/(3ρ)
        D::with_partials(value, d_rs * (-rs / (3.0 * rho)), 0.0)
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
    fn spin_supported(&self) -> bool {
        true
    }

    fn evaluate_spin(
        &self,
        batch: &crate::SpinDensityBatch<'_>,
        out: &mut crate::SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        // 交换的自旋标度是精确的：E_x = ½[E_x(2ρ_α) + E_x(2ρ_β)]。
        crate::exchange_spin_scaling(self, batch, out)
    }

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
    fn spin_supported(&self) -> bool {
        true
    }

    fn evaluate_spin(
        &self,
        batch: &crate::SpinDensityBatch<'_>,
        out: &mut crate::SpinXcOutput<'_>,
    ) -> Result<(), XcError> {
        crate::gga_spin::evaluate_pbe_correlation(self, batch, out)
    }

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
        "GgaLypC" => Ok(Box::new(LypCorrelation::new(
            parameter(id, parameters, "lyp_a")?,
            parameter(id, parameters, "lyp_b")?,
            parameter(id, parameters, "lyp_c")?,
            parameter(id, parameters, "lyp_d")?,
        ))),
        "GgaB88X" => Ok(Box::new(B88Exchange::new(parameter(
            id, parameters, "b88_beta",
        )?))),
        "GgaPbeC" => {
            let term = PbeCorrelation::new(
                (
                    parameter(id, parameters, "vwn5_p_A")?,
                    parameter(id, parameters, "vwn5_p_x0")?,
                    parameter(id, parameters, "vwn5_p_b")?,
                    parameter(id, parameters, "vwn5_p_c")?,
                ),
                parameter(id, parameters, "pbe_beta")?,
                parameter(id, parameters, "pbe_gamma")?,
            )
            .with_pw92_sets(
                Pw92Set::from_parameters(id, parameters, "pw92_p")?,
                Pw92Set::from_parameters(id, parameters, "pw92_f")?,
                Pw92Set::from_parameters(id, parameters, "pw92_a")?,
            );
            Ok(Box::new(term))
        }
        other => Err(XcError::UnknownTerm(other.to_string())),
    }
}
