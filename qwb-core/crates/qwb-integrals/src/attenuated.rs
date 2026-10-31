//! IN-02：erfc 分域库仑（短程）用的**衰减 Boys 函数**。
//!
//! # 要算的是什么
//!
//! 距离分离杂化把 `1/r12` 拆成 `erfc(ω r12)/r12`（短程）+ `erf(ω r12)/r12`（长程）。
//! 用 Gauss–Laplace 表示 `erfc(ωr)/r = (2/√π)∫_ω^∞ e^{−r²u²}du`，McMurchie–Davidson 的
//! 推导只需要把普通 Boys 函数换成分域版本：
//!
//! ```text
//! F_m^{SR}(T, α) = ∫_0^1 t^{2m} e^{−T t²} erfc(α t) dt,      α = ω/√(p+q)
//! ```
//!
//! 极限行为正好是判据要的两条：`α → 0` 时 `erfc(0) = 1`，退回普通 `F_m(T)`；
//! `α → ∞` 时 `erfc → 0`，短程项消失。
//!
//! # 怎么算（两个分支，都不依赖任何特殊函数库）
//!
//! 记 `J_n(T) = ∫_0^1 t^n e^{−T t²}dt`（`n` 为奇数时是初等函数）：
//!
//! ```text
//! J_{2k+1}(T) = [k! − e^{−T} Σ_{j=0}^{k} (k!/j!) T^j] / (2 T^{k+1})   (T > 0)
//! J_{2k+1}(0) = 1/(2k+2)                                                (T = 0)
//! ```
//!
//! `erf(αt) = (2/√π)Σ_{n≥0} (−1)^n α^{2n+1} t^{2n+1} / (n! (2n+1))`，逐项积分得到
//!
//! ```text
//! F_m^{SR}(T, α) = F_m(T) − (2/√π) Σ_{n≥0} (−1)^n α^{2n+1} / (n! (2n+1)) · J_{2n+2m+1}(T)
//! ```
//!
//! `α` 不大时（SCF 的物理范围：`ω ∈ [0.1, 0.5]`、`p+q ∈ [0.05, 200]` ⇒ `α ≤ 2.2`）这个
//! 交替级数收敛很快。`α` 大时改用 `erfc` 的渐近展开：
//!
//! ```text
//! erfc(x) ≈ e^{−x²}/(x√π) · [1 − 1/(2x²) + 3/(4x⁴) − ...]
//! ⇒ F_m^{SR} ≈ (1/(α√π)) [J_{2m−1}(β) − J_{2m−3}(β)/(2α²) + 3J_{2m−5}(β)/(4α⁴) − ...]
//! ```
//!
//! 渐近展开的误差量级 `(2k−1)!!/(2α²)^k`：`α ≥ 20` 时截到第三项已好于 1e-9，`α ≥ 30` 时
//! 短程项整体小于 1e-100（极限判据要的"消失"）。中间段（`3 < α < 20`）精度只到 1e-3…1e-8，
//! **不在 SCF 的物理范围内**，文档里写明。
//!
//! # 精度分段（实测，不是估计）
//!
//! | `α` 范围 | 精度 | 分支 |
//! | --- | --- | --- |
//! | `α ≤ 2.2`（**物理范围**：`ω ∈ [0.1, 0.5]`、`p+q ∈ [0.05, 200]`） | **< 1e-10** | 级数 |
//! | `2.2 < α ≤ 3` | ~1e-5（交错级数在边界抵消） | 级数 |
//! | `3 < α < 20` | 1e-3…1e-8（渐近展开的误差 `(2k−1)!!/(2α²)^k`） | 渐近 |
//! | `α ≥ 20` | `< 1e-9`；`F_0` 在 `α = 30` 时整体下溢为 0 | 渐近 |
//!
//! **只有物理范围那一段有 1e-10 保证** —— IN-02 的 2e 积分只会走那一段。若将来要支持
//! 更弥散的基组（`p+q` 很小 ⇒ `α` 变大），必须补一个中段的可靠算法，而不是放宽判据。
//!
//! # 验证
//!
//! `tests/in02_erfc_boys.rs`（5 条）：与独立的高阶复合 Simpson 数值积分逐点对照（按上表
//! 分段的容差）、`α = 0` 与 `boys_functions` 一致（< 1e-14）、`α` 增大时单调衰减、
//! `α = 30` 时 `F_0` 归零且 `F_1` 按 `α^{−3}` 衰减（m ≥ 1 是**多项式**衰减，不是指数）。

use crate::boys::boys_functions_into;

/// 低于这个 `α` 用级数分支（幂级数收敛快、精度好）。
pub const SERIES_ALPHA_MAX: f64 = 3.0;
/// 高于这个 `α` 用渐近分支。
pub const ASYMPTOTIC_ALPHA_MIN: f64 = 3.0;
/// 级数最多取多少项（超出即认为收敛失败）。
const SERIES_MAX_TERMS: usize = 400;

/// `J_n(T) = ∫_0^1 t^n e^{−T t²}dt`，只对**奇数** `n` 实现（本模块只用奇数）。
///
/// # 两个分支（都是为了精度，不是为了好看）
///
/// 闭式 `J_{2k+1} = [k! − e^{−T}Σ_{j=0}^k (k!/j!)T^j]/(2T^{k+1})` 在 `T` 小、`k` 大时
/// 是 `k!` 量级减 `k!` 量级、真值只有 `T^{k+1}` 量级 —— 灾难性抵消（实测在
/// `T=0.1, k=8` 处相对误差已经到 1e-7）。所以 `T ≤ 1` 走幂级数
/// `J = Σ_{i≥0} (−T)^i/(i!(n+2i+1))`，`T > 1` 才用闭式（那时 `T^{k+1} ≥ 1`，没有抵消）。
///
/// # Panics
///
/// `n` 为偶数，或 `T < 0`。
#[must_use]
pub fn odd_moment(n: usize, t: f64) -> f64 {
    assert!(n % 2 == 1, "odd_moment 只支持奇数 n（实得 {n}）");
    assert!(t >= 0.0, "T 必须非负（实得 {t}）");
    let k = (n - 1) / 2; // n = 2k+1
    if t <= 1.0 {
        // 幂级数：J = Σ_{i≥0} (−T)^i / (i! (n + 2i + 1))
        let mut total = 0.0_f64;
        let mut power = 1.0_f64; // T^i
        let mut factorial = 1.0_f64; // i!
        for i in 0..400_usize {
            let term = power / (factorial * (n as f64 + 2.0 * i as f64 + 1.0));
            if i % 2 == 0 {
                total += term;
            } else {
                total -= term;
            }
            if term.abs() <= 1.0e-18 * total.abs() {
                break;
            }
            power *= t;
            factorial *= (i + 1) as f64;
        }
        return total;
    }
    // 闭式（T > 1：没有抵消）
    let mut factorial_k = 1.0_f64;
    for value in 1..=k {
        factorial_k *= value as f64;
    }
    let mut sum = 0.0_f64;
    let mut term = factorial_k; // j = 0 项
    for j in 0..=k {
        sum += term;
        if j < k {
            term *= t / (j as f64 + 1.0);
        }
    }
    let numerator = factorial_k - (-t).exp() * sum;
    let denominator = 2.0 * t.powi(k as i32 + 1);
    numerator / denominator
}

/// 普通 Boys 函数的单项查询（`F_m(T)`）。
fn boys_single(m: usize, t: f64) -> f64 {
    let mut values = [0.0_f64; 65];
    let top = m.min(values.len() - 1);
    boys_functions_into(top, t, &mut values[..=top]);
    values[m.min(top)]
}

/// 短程（erfc 分域）Boys 函数 `F_m^{SR}(T, α)`。
///
/// # Panics
///
/// `m` 超过 64（栈缓冲上限），或 `T`/`α` 为负/非有限。
#[must_use]
pub fn erfc_boys(m: usize, t: f64, alpha: f64) -> f64 {
    assert!(m <= 64, "erfc_boys 的 m = {m} 超过 64");
    assert!(t.is_finite() && t >= 0.0, "T 非法：{t}");
    assert!(alpha.is_finite() && alpha >= 0.0, "α 非法：{alpha}");
    if alpha == 0.0 {
        return boys_single(m, t);
    }
    if alpha <= SERIES_ALPHA_MAX {
        series_branch(m, t, alpha)
    } else {
        asymptotic_branch(m, t, alpha)
    }
}

/// 级数分支（`α ≤ 3`）：`F_m − (2/√π)Σ (−1)^n α^{2n+1}/(n!(2n+1)) J_{2n+2m+1}`。
fn series_branch(m: usize, t: f64, alpha: f64) -> f64 {
    let mut result = boys_single(m, t);
    // Kahan 补偿：交错级数在 α 接近 3 时中间项比结果大两个量级。
    let mut compensation = 0.0_f64;
    let mut alpha_power = alpha; // α^{2n+1}
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    for n in 0..SERIES_MAX_TERMS {
        let factorial = {
            let mut value = 1.0_f64;
            for v in 2..=n {
                value *= v as f64;
            }
            value
        };
        let term = alpha_power / (factorial * (2.0 * n as f64 + 1.0));
        let contribution = two_over_sqrt_pi * term * odd_moment(2 * n + 2 * m + 1, t);
        let signed = if n % 2 == 0 {
            -contribution
        } else {
            contribution
        };
        let adjusted = signed - compensation;
        let updated = result + adjusted;
        compensation = (updated - result) - adjusted;
        result = updated;
        // 收敛判定：用「下一项的相对大小」而不是固定项数。
        let next_alpha_power = alpha_power * alpha * alpha;
        let next_factorial = factorial * (n as f64 + 1.0);
        let next_term = next_alpha_power / (next_factorial * (2.0 * (n as f64 + 1.0) + 1.0));
        let next = two_over_sqrt_pi * next_term * odd_moment(2 * (n + 1) + 2 * m + 1, t);
        if next.abs() <= 1.0e-17 * result.abs().max(1.0e-300) {
            break;
        }
        alpha_power = next_alpha_power;
    }
    result.max(0.0)
}

/// 渐近分支（`α > 3`）：`(1/(α√π))[J_{2m−1}(β) − J_{2m−3}(β)/(2α²) + 3J_{2m−5}(β)/(4α⁴)]`。
///
/// `β = T + α²`；`m = 0` 时只有 `J_{−1}` 那类项，直接给 0（`erfc(αt) ≤ e^{−α²t²}/(αt√π)`
/// 在 `α > 3` 时已把 `F_0` 压到 `1e-5` 以下，且随 α 迅速下溢）。
fn asymptotic_branch(m: usize, t: f64, alpha: f64) -> f64 {
    if m == 0 {
        // J_{-1} 不存在：用 J_1 的上界形式 `e^{−β}` 量级即可，α > 3 时 < 1e-4 且迅速衰减。
        let beta = t + alpha * alpha;
        return (-beta).exp() / (alpha * std::f64::consts::PI.sqrt());
    }
    let beta = t + alpha * alpha;
    let inv_two_alpha_sq = 1.0 / (2.0 * alpha * alpha);
    let inv_four_alpha_4 = 1.0 / (4.0 * alpha.powi(4));
    let mut total = odd_moment(2 * m - 1, beta);
    if 2 * m >= 3 {
        total -= inv_two_alpha_sq * odd_moment(2 * m - 3, beta);
    }
    if 2 * m >= 5 {
        total += 3.0 * inv_four_alpha_4 * odd_moment(2 * m - 5, beta);
    }
    total / (alpha * std::f64::consts::PI.sqrt())
}
