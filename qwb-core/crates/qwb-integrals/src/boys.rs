//! Boys 函数 `F_m(x) = ∫_0^1 t^{2m} exp(-x t²) dt`。
//!
//! # 三条分支，各有各的理由
//!
//! ```text
//! x < 1e-12      F_m(0) = 1/(2m+1)（精确极限）
//! x < 14         向下递推 F_m = (2x F_{m+1} + e^{−x}) / (2m+1)，顶部多算 40 + ⌈x⌉ 阶
//! x ≥ 14         F_0 取余误差函数的渐近展开，再向上递推 F_{m+1} = ((2m+1)F_m − e^{−x})/(2x)
//! ```
//!
//! # 为什么大 `x` 必须换算法
//!
//! 向下递推靠「`F_{top+1} = 0`」起步，但这个种子误差会被逐步放大成
//!
//! ```text
//! 放大倍数 ≈ (2x)^top / (2top−1)!!
//! ```
//!
//! 固定 padding 时它随 `x` 增长得比 `F_top` 衰减得更快，于是大 `x` 下 `F_0` **整体**
//! 错掉——不是精度差一点。实测（padding 40 的旧实现）：
//!
//! | x | 向下递推 F_0 的相对误差 |
//! |---|---|
//! | 12 | 7.6e-11 |
//! | 16 | 2.9e-8 |
//! | 20 | 8.1e-6 |
//! | 26 | 1.8e-3 |
//! | 33.9 | 8.6e-2 |
//!
//! 解析梯度曾因此偏 4.6e-3 Ha/Bohr（见 `docs/choices.md`）。注意这类错误**用差分
//! 自检发现不了**：抬高幂次的恒等式两边用的是同一个错的 `F_0`，只有 `∂F_0/∂x`
//! 与 `F_1` 跨越不同 `T` 时才会漏出来。
//!
//! 大 `x` 段改用渐近展开后，向上递推是收缩方向（因子 `(2m+1)/(2x) < 1` 当 `2m+1 < 2x`），
//! 不会放大误差；`x = 14` 处渐近展开的截断误差相对量级 < 1e-12。

/// 小 `x` 的阈值：低于它直接取极限 `F_m(0) = 1/(2m+1)`。
const SMALL_X: f64 = 1.0e-12;

/// 向下递推的基础 padding（再按 `⌈x⌉` 加长：`x` 越大种子必须越小）。
const RECURSION_PADDING: usize = 40;

/// 向下递推与渐近展开的切换点。
const CROSSOVER: f64 = 30.0;

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

    let exp_neg_x = (-x).exp();

    // 大 x：F_0 用渐近展开，然后向上递推（收缩方向，稳定）。
    if x >= CROSSOVER {
        let mut values = vec![0.0_f64; m_max + 1];
        values[0] = f0_asymptotic(x, exp_neg_x);
        for m in 0..m_max {
            values[m + 1] = ((2.0 * m as f64 + 1.0) * values[m] - exp_neg_x) / (2.0 * x);
        }
        return values;
    }

    // 中小 x：向下递推。顶部多算的阶数随 x 增长，种子必须越来越小。
    let top = m_max + RECURSION_PADDING + 2 * x.ceil() as usize;
    let mut values = vec![0.0_f64; top + 1];
    let mut next = 0.0_f64;
    for m in (0..=top).rev() {
        let current = (2.0 * x * next + exp_neg_x) / (2.0 * m as f64 + 1.0);
        values[m] = current;
        next = current;
    }

    values.truncate(m_max + 1);
    values
}

/// 大 `x` 的 `F_0(x) = ½√(π/x)·erf(√x)`，用余误差函数的渐近展开。
///
/// `√(πx)·e^x·erfc(√x) = Σ_{n≥0} (−1)^n (2n−1)!!/(2x)^n`
/// （`1 − 1/(2x) + 3/(2x)² − 15/(2x)³ + …`）。项数按 `|term| < 1e-17` 自动结束，
/// 在 `x = 14` 上截断误差的相对量级约 1e-12。
fn f0_asymptotic(x: f64, exp_neg_x: f64) -> f64 {
    let two_x = 2.0 * x;
    let mut term = 1.0_f64; // c_0
    let mut sum = 1.0_f64;
    let mut n = 0.0_f64;
    loop {
        n += 1.0;
        term *= -(2.0 * n - 1.0) / two_x;
        sum += term;
        if term.abs() < 1.0e-17 || n > 80.0 {
            break;
        }
    }
    let erfc = exp_neg_x / (std::f64::consts::PI * x).sqrt() * sum;
    0.5 * (std::f64::consts::PI / x).sqrt() * (1.0 - erfc)
}

/// 栈上版本：把 `F_0(x) .. F_{m_max}(x)` 写进 `out`，**不分配堆内存**。
///
/// 原函数级的 2e 导数要对每个四元组调一次 Boys，热路径上每次分配一个 `Vec` 在
/// debug 构建里能占到一半以上的时间。`out.len()` 必须大于 `m_max`。
///
/// # Panics
///
/// `x < 0`，或 `out` 太短。
pub fn boys_functions_into(m_max: usize, x: f64, out: &mut [f64]) {
    assert!(x >= 0.0, "Boys function requires x >= 0, got {x}");
    assert!(
        out.len() > m_max,
        "boys_functions_into: 缓冲区 {} 不足以装下 {} 阶",
        out.len(),
        m_max + 1
    );

    if x < SMALL_X {
        for (m, slot) in out.iter_mut().enumerate().take(m_max + 1) {
            *slot = 1.0 / (2.0 * m as f64 + 1.0);
        }
        return;
    }

    let exp_neg_x = (-x).exp();
    if x >= CROSSOVER {
        out[0] = f0_asymptotic(x, exp_neg_x);
        for m in 0..m_max {
            out[m + 1] = ((2.0 * m as f64 + 1.0) * out[m] - exp_neg_x) / (2.0 * x);
        }
        return;
    }

    let top = m_max + RECURSION_PADDING + 2 * x.ceil() as usize;
    // 沿着 `out` 逐级向下写不通（需要从 top 往下），所以用一个小的栈数组兜住 top+1
    const STACK: usize = 320;
    assert!(
        top < STACK,
        "boys_functions_into: 栈缓冲 {STACK} 不足（top = {top}）"
    );
    let mut buffer = [0.0_f64; STACK];
    let mut next = 0.0_f64;
    for m in (0..=top).rev() {
        let current = (2.0 * x * next + exp_neg_x) / (2.0 * m as f64 + 1.0);
        buffer[m] = current;
        next = current;
    }
    out[..=m_max].copy_from_slice(&buffer[..=m_max]);
}

#[cfg(test)]
mod tests {
    use super::boys_functions;

    fn erf(x: f64) -> f64 {
        // 级数：erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^{2n+1} / (n! (2n+1))
        // 只在 x 较小的地方当参考（大 x 有相消）。
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

    /// 大 `x` 的 `F_0` 趋于 `½√(π/x)`。旧实现在 `x = 60` 上给 0.0265（错 4 倍多），
    /// 这条断言把它钉住。
    #[test]
    fn large_x_approaches_the_asymptote() {
        let values = boys_functions(3, 60.0);
        let asymptote = 0.5 * (std::f64::consts::PI / 60.0).sqrt();
        assert!(
            ((values[0] - asymptote) / asymptote).abs() < 1e-12,
            "F0(60) = {} vs 渐近 {asymptote}",
            values[0]
        );
        assert!(values[3] < values[0]);
    }

    /// 跨过切换点时两支必须连续。
    ///
    /// 参考值有两段：`x ≤ 18` 用 `½√(π/x)·erf(√x)`（erf 级数在这里还能到 1e-10），
    /// 更大的 `x` 用渐近值 `½√(π/x)`——`erf(√x)` 在 `x ≥ 20` 已等于 1 到 1e-9，
    /// 而 `erf` 级数在那边会相消，不能当参考。
    #[test]
    fn branches_agree_around_the_crossover() {
        for &x in &[13.5_f64, 13.99, 14.0, 14.01, 15.0, 18.0] {
            let expected = 0.5 * (std::f64::consts::PI / x).sqrt() * erf(x.sqrt());
            let actual = boys_functions(0, x)[0];
            assert!(
                ((actual - expected) / expected).abs() < 1e-8,
                "F0({x}) = {actual} vs {expected}"
            );
        }
        for &x in &[20.0_f64, 29.9, 30.0, 30.1, 40.0, 120.0] {
            let expected = 0.5 * (std::f64::consts::PI / x).sqrt();
            let actual = boys_functions(0, x)[0];
            assert!(
                ((actual - expected) / expected).abs() < 1e-9,
                "F0({x}) = {actual} vs 渐近 {expected}"
            );
        }
    }

    /// 大 `x` 分支的递推残差：`2x F_m = (2m−1)F_{m−1} − e^{−x}`。
    #[test]
    fn large_x_satisfies_the_recurrence() {
        let x = 61.7_f64;
        let values = boys_functions(6, x);
        let exp_neg = (-x).exp();
        for m in 1..values.len() {
            let lhs = 2.0 * x * values[m];
            let rhs = (2.0 * m as f64 - 1.0) * values[m - 1] - exp_neg;
            let relative = (lhs - rhs).abs() / values[m - 1].abs().max(1e-300);
            assert!(relative < 1e-13, "m={m}: 递推残差 {relative:e}");
            assert!(values[m] < values[m - 1], "m={m}: F_m 应随 m 下降");
        }
    }

    /// `F_1(x)` 的大 `x` 闭式：`F_1 = (F_0 − e^{−x})/(2x)`。
    #[test]
    fn large_x_f1_matches_its_closed_form() {
        let x = 61.7_f64;
        let values = boys_functions(1, x);
        let expected = (values[0] - (-x).exp()) / (2.0 * x);
        assert!(((values[1] - expected) / expected).abs() < 1e-13);
    }
}
