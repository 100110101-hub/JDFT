//! Boys 函数 `F_m(x) = ∫_0^1 t^{2m} exp(-x t²) dt`。
//!
//! 用**向下递推** `F_m = (2x F_{m+1} + e^{−x}) / (2m+1)`：向下是数值稳定的方向
//! （向上会放大误差）。起点取 `F_{M+1} = 0`，`M = m_max + 40` 足够让误差衰减掉。

/// 小 `x` 的阈值：低于它直接取极限 `F_m(0) = 1/(2m+1)`。
const SMALL_X: f64 = 1.0e-12;

/// 为了启动向下递推，在 `m_max` 之上多算的阶数。
const RECURSION_PADDING: usize = 40;

/// 计算 `F_0(x) .. F_{m_max}(x)`。
///
/// # Panics
///
/// `x` 为负（Boys 函数的定义域是非负）；用断言而不是返回 `NaN`，
/// 因为负值只可能来自调用方的平方距离算错。
#[must_use]
pub fn boys_functions(m_max: usize, x: f64) -> Vec<f64> {
    assert!(x >= 0.0, "Boys function requires x >= 0, got {x}");

    if x < SMALL_X {
        return (0..=m_max).map(|m| 1.0 / (2.0 * m as f64 + 1.0)).collect();
    }

    let top = m_max + RECURSION_PADDING;
    let exp_neg_x = (-x).exp();
    let mut values = vec![0.0_f64; top + 1];

    // F_{top+1} = 0 → 逐级向下得到 F_top .. F_0。
    let mut next = 0.0_f64;
    for m in (0..=top).rev() {
        let current = (2.0 * x * next + exp_neg_x) / (2.0 * m as f64 + 1.0);
        values[m] = current;
        next = current;
    }

    values.truncate(m_max + 1);
    values
}

#[cfg(test)]
mod tests {
    use super::boys_functions;

    fn erf(x: f64) -> f64 {
        // Abramowitz & Stegun 7.1.26 不够精确；这里用高精度级数：
        // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^{2n+1} / (n! (2n+1))
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0_f64;
        loop {
            n += 1.0;
            term *= -x * x / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-18 * sum.abs().max(1e-30) || n > 200.0 {
                break;
            }
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    #[test]
    fn zero_argument_is_the_exact_limit() {
        let values = boys_functions(4, 0.0);
        for (m, value) in values.iter().enumerate() {
            assert!((value - 1.0 / (2.0 * m as f64 + 1.0)).abs() < 1e-16);
        }
    }

    #[test]
    fn f0_matches_the_erf_closed_form() {
        // F_0(x) = (1/2) sqrt(pi/x) erf(sqrt(x))
        for &x in &[1.0e-6, 1.0e-3, 0.5, 1.0, 3.0, 10.0] {
            let expected = 0.5 * (std::f64::consts::PI / x).sqrt() * erf(x.sqrt());
            let actual = boys_functions(0, x)[0];
            assert!(
                (actual - expected).abs() < 1e-12,
                "F0({x}) = {actual} vs {expected}"
            );
        }
        // F_0(1) 的常用参考值。
        assert!((boys_functions(0, 1.0)[0] - 0.746_824_132_812_427).abs() < 1e-13);
    }

    #[test]
    fn downward_recursion_satisfies_the_recurrence() {
        // 2m F_{m-1}(x) ?= 2x F_m(x) + exp(-x) 的等价形式：
        // F_{m-1} = (2x F_m + e^{-x}) / (2m-1)
        let x = 2.5;
        let values = boys_functions(8, x);
        let exp_neg = (-x).exp();
        for m in 1..values.len() {
            let expected = (2.0 * x * values[m] + exp_neg) / (2.0 * m as f64 - 1.0);
            let relative = ((values[m - 1] - expected).abs()) / expected.abs().max(1e-300);
            assert!(relative < 1e-12, "m={m}: relative error {relative:e}");
        }
    }

    #[test]
    fn large_x_stays_positive_and_small() {
        let values = boys_functions(3, 60.0);
        assert!(values[0] > 0.0);
        assert!(values[0] < 0.05);
        assert!(values[3] < values[0]);
    }
}
