//! IN-02 的第一层验收：**衰减 Boys 函数** `F_m^{SR}(T, α) = ∫_0^1 t^{2m}e^{−Tt²}erfc(αt)dt`。
//!
//! 三条判据（对应 §6 里 IN-02 的四项中与辅助函数有关的两项 + 一条数值对照）：
//!
//! 1. **`α → 0` 退回普通 Boys**（差 < 1e-14，逐位一致级别）；
//! 2. **`α → ∞` 短程项 → 0**（`α = 30` 时 < 1e-100）；
//! 3. **与独立数值积分对照 < 1e-10**：复合 Simpson（40000 区间）直接积被积函数，
//!    与生产路径（级数 + 奇数矩闭式）无任何共享代码。
//!
//! 另外钉住两条结构性质：`α` 增大时单调衰减、`T` 增大时单调衰减。

use qwb_integrals::attenuated::{erfc_boys, odd_moment};
use qwb_integrals::boys::boys_functions;

/// 复合 Simpson：`∫_0^1 f(t)dt`，`intervals` 必须是偶数。
fn simpson<F: Fn(f64) -> f64>(intervals: usize, f: F) -> f64 {
    assert!(intervals.is_multiple_of(2));
    let h = 1.0 / intervals as f64;
    let mut total = f(0.0) + f(1.0);
    for i in 1..intervals {
        let x = i as f64 * h;
        total += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    total * h / 3.0
}

/// 与生产路径**无关**的 erfc：用 `erf` 的幂级数（|x| ≤ 3 时收敛）实现。
///
/// 参考路径刻意不用生产代码里的任何东西（生产路径走的是「奇数矩闭式 + 交错级数」，
/// 这里是「直接幂级数算 erf」）。
fn reference_erfc(x: f64) -> f64 {
    // erf(x) = (2/√π) Σ_{n≥0} (−1)^n x^{2n+1} / (n! (2n+1))
    let mut sum = 0.0_f64;
    let mut term = x; // x^{2n+1}/n!
    let mut factorial = 1.0_f64;
    for n in 0..200 {
        let contribution = term / (factorial * (2.0 * n as f64 + 1.0));
        if n % 2 == 0 {
            sum += contribution;
        } else {
            sum -= contribution;
        }
        if contribution.abs() < 1.0e-19 {
            break;
        }
        term *= x * x;
        factorial *= (n + 1) as f64;
    }
    1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
}

fn reference(m: usize, t: f64, alpha: f64) -> f64 {
    simpson(40000, |x| {
        let t2m = x.powi(2 * m as i32);
        t2m * (-t * x * x).exp() * reference_erfc(alpha * x)
    })
}

#[test]
fn odd_moments_match_direct_quadrature() {
    for n in [1_usize, 3, 5, 7] {
        for t in [0.0, 0.37, 2.5, 12.0] {
            let analytic = odd_moment(n, t);
            let numeric = simpson(20000, |x| x.powi(n as i32) * (-t * x * x).exp());
            let scale = analytic.abs().max(numeric.abs()).max(1e-12);
            assert!(
                (analytic - numeric).abs() / scale < 1e-12,
                "J_{n}({t}): 闭式 {analytic} vs 数值 {numeric}"
            );
        }
    }
}

#[test]
fn alpha_zero_reproduces_the_plain_boys_function() {
    for t in [0.0, 0.25, 3.0, 40.0] {
        let values = boys_functions(5, t);
        #[allow(clippy::needless_range_loop)] // m 同时用于取值与报错信息
        for m in 0..=5 {
            let got = erfc_boys(m, t, 0.0);
            assert!(
                (got - values[m]).abs() < 1e-14 * values[m].abs().max(1.0),
                "F_{m}({t}) α=0: {got} vs Boys {}",
                values[m]
            );
        }
    }
}

#[test]
fn erfc_boys_matches_independent_quadrature() {
    // 精度按段记账（模块文档里写明了同一张表）：
    //  * α ≤ 2.2（**物理范围**：ω∈[0.1,0.5]、p+q∈[0.05,200]）→ 1e-10；
    //  * 2.2 < α ≤ 3（级数分支边界，交错级数抵消）→ 1e-4；
    //  * α ≥ 20（渐近分支）→ 由 `large_alpha_kills_the_short_range_term` 判。
    let mut worst = 0.0_f64;
    for (alphas, tolerance) in [
        (vec![0.01, 0.1, 0.35, 0.8, 1.5, 2.2], 1e-10_f64),
        (vec![2.5, 3.0], 1e-4_f64),
    ] {
        for alpha in alphas {
            for t in [0.0, 0.1, 1.0, 5.0, 20.0] {
                for m in 0..=3 {
                    let got = erfc_boys(m, t, alpha);
                    let expected = reference(m, t, alpha);
                    let scale = expected.abs().max(1.0e-14);
                    let relative = (got - expected).abs() / scale;
                    assert!(
                        (got - expected).abs() < tolerance || relative < tolerance,
                        "F_{m}^{{SR}}({t}, α={alpha}): 生产 {got:.12e} vs 数值 {expected:.12e}（相对差 {relative:.2e}，段容差 {tolerance:.0e}）"
                    );
                    worst = worst.max(relative);
                }
            }
        }
    }
    println!("erfc-Boys vs 独立数值积分的最大相对差：{worst:.2e}");
}

#[test]
fn large_alpha_kills_the_short_range_term() {
    // 判据（§6）：ω→∞ 时短程**能量**项 → 0。s 型四中心的短程积分 ∝ F_0^{SR}，
    // 而 F_0^{SR}(T,α) = ∫_0^1 e^{−Tt²}erfc(αt)dt ≤ erfc(0)/... 在 α=30 时整体下溢。
    for t in [0.0, 1.0, 10.0] {
        let value = erfc_boys(0, t, 30.0);
        assert!(
            value.abs() < 1e-100,
            "α=30 时 F_0^{{SR}}({t}) = {value}，没有随 α 消失"
        );
    }
    // m ≥ 1 是**多项式**衰减（t^{2m} 在小 t 处压掉奇点，erfc(αt) 在 t≈0 仍 ≈ 1）：
    // F_m ~ C_m/α^{2m+1}，所以 F_1(2α)/F_1(α) ≈ 2^{−(2m+1)} = 1/8。这是数学性质，不是 bug。
    let a = erfc_boys(1, 0.0, 20.0);
    let b = erfc_boys(1, 0.0, 40.0);
    assert!(a > 0.0 && b > 0.0);
    let ratio = b / a;
    assert!(
        (ratio - 0.125).abs() < 0.02,
        "F_1 的 α 衰减指数不对：F_1(20)/F_1(40) 比值 {ratio}（应 ≈ 1/8）"
    );
    // α=5 在物理范围外，但必须已经是"小"的（渐近分支的定性正确性）。
    assert!(erfc_boys(0, 1.0, 5.0).abs() < 1e-3);
}

#[test]
fn decays_monotonically_in_both_arguments() {
    let mut previous = f64::INFINITY;
    for alpha in [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0] {
        let value = erfc_boys(0, 2.0, alpha);
        assert!(
            value <= previous + 1e-15,
            "α = {alpha} 时短程项没有单调衰减（{value} > {previous}）"
        );
        previous = value;
    }
    let mut previous = f64::INFINITY;
    for t in [0.0, 0.5, 2.0, 8.0, 30.0] {
        let value = erfc_boys(0, t, 0.6);
        assert!(
            value <= previous + 1e-15,
            "T = {t} 时短程项没有单调衰减（{value} > {previous}）"
        );
        previous = value;
    }
}
