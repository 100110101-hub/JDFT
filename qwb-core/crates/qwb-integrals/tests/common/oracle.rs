//! 独立的数值参照路径：闭式 `(ss|ss)` 的中心导数（I-03 的「第二条路径」）。
//!
//! 基本事实：`∂/∂A_x e^{−α(x−A_x)²} = 2α(x−A_x)e^{…}`，所以一阶是干净的
//! `(x−A_x)e = (1/(2α))∂_A e`。**但二阶不是**：再求一次导会带出边界项，
//!
//! ```text
//! (x−A_x)² e = (1/(2α)) ∂_A[(x−A_x)e] + (1/(2α)) e
//! ```
//!
//! 一般地（`u = x−A_x`，`n ≥ 1`）：
//!
//! ```text
//! uⁿ e = (1/(2α)) ∂_A[uⁿ⁻¹ e] + [(n−1)/(2α)] uⁿ⁻² e
//! ```
//!
//! 所以参照值必须按这条递归算，**不能**简单地连续施加 n 次同一步长的差分——
//! 那样得到的 d 型积分会少掉 `(1/(2α))·(s 型)` 这一项（本项目在交叉验证 d_xx
//! 时正是被它绊了一下：oracle 给 −0.2104，生产路径给 +0.2057，补上边界项
//! 0.4161 后两者一致到 1e-12）。
//!
//! 精度：Richardson 外推的中心差分，相对误差 ~1e-9；用于交叉验证（判 1e-8）
//! 足够，但不要拿它当生产路径。

#![allow(dead_code)]

type Center = [f64; 3];
/// 笛卡尔角动量分量 `(x, y, z)` 的幂次。
pub type Powers = [u32; 3];

/// 一个未归一化的笛卡尔高斯原函数。
#[derive(Debug, Clone, Copy)]
pub struct Function {
    pub center: Center,
    pub exponent: f64,
    pub powers: Powers,
}

impl Function {
    pub fn s(center: Center, exponent: f64) -> Self {
        Self {
            center,
            exponent,
            powers: [0, 0, 0],
        }
    }

    pub fn with_powers(self, powers: Powers) -> Self {
        Self { powers, ..self }
    }
}

const STEP: f64 = 1.0e-2;

/// 四中心 `(a b|c d)` 的参照值（任意角动量）。
pub fn reference(functions: [Function; 4]) -> f64 {
    value(&functions)
}

fn closed_form(functions: &[Function; 4]) -> f64 {
    qwb_integrals::two_electron::primitive_ssss(
        functions[0].exponent,
        functions[0].center,
        functions[1].exponent,
        functions[1].center,
        functions[2].exponent,
        functions[2].center,
        functions[3].exponent,
        functions[3].center,
    )
}

/// 递归实现 `uⁿ e = (1/(2α))∂_A[uⁿ⁻¹ e] + ((n−1)/(2α))uⁿ⁻² e`。
fn value(functions: &[Function; 4]) -> f64 {
    // 找第一个还有幂次的分量。
    let mut target = None;
    for (which, function) in functions.iter().enumerate() {
        for (axis, order) in function.powers.iter().enumerate() {
            if *order > 0 {
                target = Some((which, axis, *order));
                break;
            }
        }
        if target.is_some() {
            break;
        }
    }
    let Some((which, axis, order)) = target else {
        return closed_form(functions);
    };

    let alpha = functions[which].exponent;
    let origin = functions[which].center[axis];

    // 第一项：(1/(2α)) ∂_A [uⁿ⁻¹ e]
    let mut reduced = *functions;
    reduced[which].powers[axis] -= 1;
    let moved = |t: f64| {
        let mut probe = reduced;
        probe[which].center[axis] = t;
        value(&probe)
    };
    let central = |step: f64| (moved(origin + step) - moved(origin - step)) / (2.0 * step);
    let derivative = (4.0 * central(STEP * 0.5) - central(STEP)) / 3.0;
    let mut total = derivative / (2.0 * alpha);

    // 第二项：((n−1)/(2α)) uⁿ⁻² e（n ≥ 2 才有）
    if order >= 2 {
        let mut lower = *functions;
        lower[which].powers[axis] -= 2;
        total += (f64::from(order) - 1.0) / (2.0 * alpha) * value(&lower);
    }
    total
}
