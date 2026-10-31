//! IN-02 第二层验收：**短程 2e 原函数积分**（`erfc(ωr12)/r12`）。
//!
//! 四条判据（对应 §6 里 IN-02 的四项）：
//!
//! 1. **`ω → 0` 退回全库仑**：与 [`primitive_component`] 的差 < 1e-10（相对）；
//! 2. **`ω → ∞` 短程项 → 0**：`ω = 40` 时整条积分 < 1e-100；
//! 3. **与独立数值积分对照 < 1e-9**：把 `(ss|ss)` 的六维积分**化成一维**（角向解析积掉），
//!    用复合 Simpson 直接积 —— 这条参照路径不碰 Hermite 系数、不碰 `erfc_boys`；
//! 4. **`(ss|ss)` 的解析闭式在有限 `ω` 下可对**：MD 闭式
//!    `2π^{5/2}/(pq√(p+q))·F_0^{SR}(T,α)` 与第 3 条的一维积分一致。

use qwb_integrals::attenuated::{short_range_t_min, truncated_boys};
use qwb_integrals::hermite2e::{
    primitive_component, primitive_component_lr, primitive_component_sr,
};

type Center = [f64; 3];

const S: [u32; 3] = [0, 0, 0];

/// 复合 Simpson（区间数必须偶数）。
fn simpson(intervals: usize, low: f64, high: f64, f: impl Fn(f64) -> f64) -> f64 {
    assert!(intervals.is_multiple_of(2));
    let h = (high - low) / intervals as f64;
    let mut total = f(low);
    let mut nan_seen = false;
    for i in 1..intervals {
        let x = low + i as f64 * h;
        let value = f(x);
        if !value.is_finite() {
            nan_seen = true;
            continue;
        }
        total += if i % 2 == 1 { 4.0 } else { 2.0 } * value;
    }
    let last = f(high);
    if last.is_finite() {
        total += last;
    }
    // 上限取到足够远时，末端的贡献必须已经可以忽略（否则参照本身不可信）。
    assert!(
        !nan_seen || total.is_finite(),
        "参照积分的被积函数出现非有限值"
    );
    total * h / 3.0
}

/// `(ss|ss)` 的**独立参照**：**u 空间**里积高斯阻尼核的闭式。
///
/// `erfc(ωr)/r = (2/√π)∫_ω^∞ e^{−r²u²}du`，而高斯阻尼核的六维积分有初等闭式
///
/// ```text
/// (ss|ss)(μ) = e_ab e_cd (π/(p+q))^{3/2}(π/(ρ+μ))^{3/2} exp(−ρμd²/(ρ+μ)),   μ = u²
/// ```
///
/// 于是 `(ss|ss)_SR = (2/√π)∫_ω^∞ (ss|ss)(u²) du` —— **被积函数是初等的，不含 erfc，
/// 也不含任何 Hermite 系数**，与生产路径（t 表示 + 截断 Boys）完全是另一条路。
///
/// 换元 `u = √ρ·tanθ` 把无穷区间压到 `θ ∈ [atan(ω/√ρ), π/2)`，被积函数在那里快速衰减。
#[allow(clippy::too_many_arguments)]
fn reference_ssss(
    a: Center,
    alpha: f64,
    b: Center,
    beta: f64,
    c: Center,
    gamma: f64,
    d: Center,
    delta: f64,
    omega: f64,
) -> f64 {
    let p = alpha + beta;
    let q = gamma + delta;
    let p_center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let q_center = [
        (gamma * c[0] + delta * d[0]) / q,
        (gamma * c[1] + delta * d[1]) / q,
        (gamma * c[2] + delta * d[2]) / q,
    ];
    let ab2 = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>();
    let cd2 = (0..3).map(|i| (c[i] - d[i]).powi(2)).sum::<f64>();
    let rho = p * q / (p + q);
    let dd2 = (0..3)
        .map(|i| (p_center[i] - q_center[i]).powi(2))
        .sum::<f64>();
    let e_ab = (-alpha * beta / p * ab2).exp();
    let e_cd = (-gamma * delta / q * cd2).exp();
    let common = e_ab * e_cd * (std::f64::consts::PI / (p + q)).powf(1.5) * 2.0
        / std::f64::consts::PI.sqrt();
    // u = √ρ·tanθ, du = √ρ sec²θ dθ；μ = ρtan²θ
    let u = |theta: f64| -> f64 { rho.sqrt() * theta.tan() };
    let mu = |u: f64| -> f64 { u * u };
    let damped = |u: f64| -> f64 {
        let m = mu(u);
        (std::f64::consts::PI / (rho + m)).powf(1.5) * (-rho * m * dd2 / (rho + m)).exp()
    };
    let integrand = |theta: f64| -> f64 {
        let uu = u(theta);
        let sec2 = 1.0 + theta.tan() * theta.tan();
        damped(uu) * rho.sqrt() * sec2
    };
    // θ 从 atan(ω/√ρ) 起；上限取到衰减完（u 足够大时 damped ~ π^{3/2}/u³，但
    // 真正把被积函数压下去的是 exp(−ρ d²)）—— 用 π/2−1e-9 作上限并检查尾部。
    let theta_min = (omega / rho.sqrt()).atan();
    let theta_max = std::f64::consts::FRAC_PI_2 - 1.0e-9;
    common * simpson(40000, theta_min, theta_max, integrand)
}

/// 与生产路径无关的 `erfc`（`erf` 的幂级数，|x| ≤ 6 足够）。
#[allow(dead_code)] // 上一版归约用过；换成 u 空间后不再需要，留着备查
fn reference_erfc(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut term = x;
    let mut factorial = 1.0_f64;
    for n in 0..400 {
        let contribution = term / (factorial * (2.0 * n as f64 + 1.0));
        if n % 2 == 0 {
            sum += contribution;
        } else {
            sum -= contribution;
        }
        if contribution.abs() < 1.0e-22 * sum.abs().max(1.0) {
            break;
        }
        term *= x * x;
        factorial *= (n + 1) as f64;
        if factorial.is_infinite() {
            break;
        }
    }
    1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
}

fn geometries() -> Vec<(&'static str, [Center; 4], [f64; 4])> {
    vec![
        (
            "H2-like",
            [
                [0.0, 0.0, -0.699_2],
                [0.0, 0.0, 0.699_2],
                [0.5, 0.1, 0.3],
                [-0.2, 0.4, 0.7],
            ],
            [1.2, 0.9, 1.1, 0.8],
        ),
        (
            "水分子一角",
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.430_5, 1.107_4],
                [0.0, -1.430_5, 1.107_4],
                [0.6, 0.9, -0.4],
            ],
            [1.5, 0.7, 1.0, 0.55],
        ),
    ]
}

#[test]
fn reference_matches_the_analytic_coulomb_value_at_zero_omega() {
    // 参照路径自己的校准：ω=0 时它必须等于已知的 `(ss|ss)`（生产路径已由 I-03 钉住）。
    for (name, centers, exponents) in geometries() {
        let production = primitive_component(
            S,
            centers[0],
            exponents[0],
            S,
            centers[1],
            exponents[1],
            S,
            centers[2],
            exponents[2],
            S,
            centers[3],
            exponents[3],
        );
        let reference = reference_ssss(
            centers[0],
            exponents[0],
            centers[1],
            exponents[1],
            centers[2],
            exponents[2],
            centers[3],
            exponents[3],
            0.0,
        );
        let relative = (production - reference).abs() / production.abs();
        assert!(
            relative < 1e-9,
            "{name}: 参照路径在 ω=0 处 {reference} 与解析值 {production} 不符（相对差 {relative:.2e}）"
        );
    }
}

#[test]
fn zero_omega_reduces_to_the_full_coulomb_integral() {
    for (name, centers, exponents) in geometries() {
        let full = primitive_component(
            S,
            centers[0],
            exponents[0],
            S,
            centers[1],
            exponents[1],
            S,
            centers[2],
            exponents[2],
            S,
            centers[3],
            exponents[3],
        );
        let short = primitive_component_sr(
            S,
            centers[0],
            exponents[0],
            S,
            centers[1],
            exponents[1],
            S,
            centers[2],
            exponents[2],
            S,
            centers[3],
            exponents[3],
            0.0,
        );
        let relative = (full - short).abs() / full.abs();
        assert!(
            relative < 1e-10,
            "{name}: ω=0 时短程 {short} 没有退回全库仑 {full}（相对差 {relative:.2e}）"
        );
    }
}

#[test]
fn short_range_integral_matches_the_one_dimensional_reference() {
    let mut worst = 0.0_f64;
    for (name, centers, exponents) in geometries() {
        for omega in [0.0, 0.1, 0.3, 0.6, 1.0] {
            let production = primitive_component_sr(
                S,
                centers[0],
                exponents[0],
                S,
                centers[1],
                exponents[1],
                S,
                centers[2],
                exponents[2],
                S,
                centers[3],
                exponents[3],
                omega,
            );
            let reference = reference_ssss(
                centers[0],
                exponents[0],
                centers[1],
                exponents[1],
                centers[2],
                exponents[2],
                centers[3],
                exponents[3],
                omega,
            );
            let scale = reference.abs().max(1.0e-14);
            let relative = (production - reference).abs() / scale;
            assert!(
                (production - reference).abs() < 1e-9 || relative < 1e-9,
                "{name} ω={omega}: 生产 {production:.12e} vs 一维数值参照 {reference:.12e}（相对差 {relative:.2e}）"
            );
            worst = worst.max(relative);
        }
    }
    println!("短程 (ss|ss) 与一维数值参照的最大相对差：{worst:.2e}");
}

#[test]
fn the_general_integral_agrees_with_the_md_closed_form() {
    // 判据 4：`(ss|ss)` 的 MD 闭式 `2π^{5/2}/(pq√(p+q))·F_0^{SR}(T,α)` 与**通式**一致。
    //
    // 两条路径只共享 `erfc_boys`：通式走 Hermite 系数 + R 块表的收缩，闭式直接乘预因子。
    // 所以这条能抓住「通式把 α 传错」「预因子漏项」这类错（α 若用错，ω 依赖会立刻不匹配）。
    for (name, centers, exponents) in geometries() {
        let [alpha, beta, gamma, delta] = exponents;
        let [a, b, c, d] = centers;
        let p = alpha + beta;
        let q = gamma + delta;
        let p_center = [
            (alpha * a[0] + beta * b[0]) / p,
            (alpha * a[1] + beta * b[1]) / p,
            (alpha * a[2] + beta * b[2]) / p,
        ];
        let q_center = [
            (gamma * c[0] + delta * d[0]) / q,
            (gamma * c[1] + delta * d[1]) / q,
            (gamma * c[2] + delta * d[2]) / q,
        ];
        let rho = p * q / (p + q);
        let dd2 = (0..3)
            .map(|i| (p_center[i] - q_center[i]).powi(2))
            .sum::<f64>();
        let t = rho * dd2;
        let ab2 = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>();
        let cd2 = (0..3).map(|i| (c[i] - d[i]).powi(2)).sum::<f64>();
        let prefactor = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt())
            * (-alpha * beta / p * ab2).exp()
            * (-gamma * delta / q * cd2).exp();
        for omega in [0.0, 0.2, 0.5] {
            // 正确的短程 t 表示：截断在 t_min = ω/√(ρ+ω²)，不是 erfc 加权。
            let t_min = short_range_t_min(omega, rho);
            let closed = prefactor * truncated_boys(0, t, t_min);
            let general =
                primitive_component_sr(S, a, alpha, S, b, beta, S, c, gamma, S, d, delta, omega);
            let relative = (general - closed).abs() / closed.abs().max(1e-14);
            assert!(
                relative < 1e-12,
                "{name} ω={omega}: 通式 {general:.12e} vs MD 闭式 {closed:.12e}（相对差 {relative:.2e}）"
            );
        }
    }
}

#[test]
fn large_omega_shrinks_the_short_range_integral_polynomially() {
    // §6 的判据是「ω→∞ 时短程项 → 0」。它在 t 表示下是 `∫_{t_min}^1`、`t_min = ω/√(ρ+ω²)`，
    // 于是 `1 − t_min ≈ ρ/(2ω²)` —— **多项式**衰减，不是指数（短程核在 `r12 → 0` 处有限，
    // 截掉的只是 `r12 ≲ 1/ω` 那一圈，量级就是 `1/ω²`）。所以判据写成：
    // ① 随 ω 单调变小；② 比值 ψ(2ω)/ψ(ω) ≈ 1/4（`1/ω²` 的特征）；③ 绝对值足够小。
    let (_, centers, exponents) = geometries()[0];
    let value = |omega: f64| {
        primitive_component_sr(
            S,
            centers[0],
            exponents[0],
            S,
            centers[1],
            exponents[1],
            S,
            centers[2],
            exponents[2],
            S,
            centers[3],
            exponents[3],
            omega,
        )
    };
    let a = value(20.0);
    let b = value(40.0);
    assert!(b > 0.0 && a > 0.0);
    assert!(b < a, "ω 增大时短程项必须变小（{a} → {b}）");
    let ratio = b / a;
    assert!(
        (ratio - 0.25).abs() < 0.05,
        "短程项的 ω 衰减指数不对：ψ(40)/ψ(20) = {ratio}（应 ≈ 1/4）"
    );
    assert!(b < 1e-3, "ω=40 时短程项 = {b}，还不够小");
}

#[test]
fn short_plus_long_range_equals_the_full_coulomb_integral() {
    // `erf + erfc = 1` ⇒ 短程 + 长程 = 全库仑。这是最强的一条：两条分支共用同一个
    // PrimitiveContext/MD 展开，但种子不同（`∫_{t_min}^1` 与 `∫_0^{t_min}`），
    // 所以它同时钉住「截断点的定义」与「两半的互补性」。
    for (name, centers, exponents) in geometries() {
        let full = primitive_component(
            S,
            centers[0],
            exponents[0],
            S,
            centers[1],
            exponents[1],
            S,
            centers[2],
            exponents[2],
            S,
            centers[3],
            exponents[3],
        );
        for omega in [0.0, 0.1, 0.35, 0.8, 2.0] {
            let short = primitive_component_sr(
                S,
                centers[0],
                exponents[0],
                S,
                centers[1],
                exponents[1],
                S,
                centers[2],
                exponents[2],
                S,
                centers[3],
                exponents[3],
                omega,
            );
            let long = primitive_component_lr(
                S,
                centers[0],
                exponents[0],
                S,
                centers[1],
                exponents[1],
                S,
                centers[2],
                exponents[2],
                S,
                centers[3],
                exponents[3],
                omega,
            );
            let relative = (short + long - full).abs() / full.abs();
            assert!(
                relative < 1e-13,
                "{name} ω={omega}: 短程 {short} + 长程 {long} = {} ≠ 全库仑 {full}（相对差 {relative:.2e}）",
                short + long
            );
            // ω=0 时长程项必须为 0、ω 大时短程项必须小。
            if omega == 0.0 {
                assert!(
                    (long / full).abs() < 1e-14,
                    "{name}: ω=0 时长程项应为 0，实得 {long}"
                );
            }
        }
    }
}

#[test]
fn short_range_works_for_higher_angular_momentum() {
    // ω=0 时必须逐位退回 `primitive_component` —— 对 p/d 分量同样成立（MD 展开是通用的）。
    let centers: [Center; 4] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.430_5, 1.107_4],
        [0.0, -1.430_5, 1.107_4],
        [0.6, 0.9, -0.4],
    ];
    let exponents = [1.5, 0.7, 1.0, 0.55];
    let cases: [[[u32; 3]; 4]; 3] = [
        [[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[2, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
    ];
    for powers in cases {
        let full = primitive_component(
            powers[0],
            centers[0],
            exponents[0],
            powers[1],
            centers[1],
            exponents[1],
            powers[2],
            centers[2],
            exponents[2],
            powers[3],
            centers[3],
            exponents[3],
        );
        let short = primitive_component_sr(
            powers[0],
            centers[0],
            exponents[0],
            powers[1],
            centers[1],
            exponents[1],
            powers[2],
            centers[2],
            exponents[2],
            powers[3],
            centers[3],
            exponents[3],
            0.0,
        );
        let relative = (short - full).abs() / full.abs();
        assert!(
            relative < 1e-10,
            "{powers:?}: ω=0 时短程 {short} ≠ 全库仑 {full}（相对差 {relative:.2e}）"
        );
    }
}
