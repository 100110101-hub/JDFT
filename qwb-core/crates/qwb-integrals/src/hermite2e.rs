//! 任意角动量的两电子积分（McMurchie–Davidson 的 Hermite 展开）。
//!
//! # 为什么用这种方式组织代码
//!
//! 2e 积分的难点从来不是高斯乘积定理，而是 **Hermite Coulomb 族**
//! `R_{lmn}` 的递推。凭记忆写递推式是这类代码最容易埋错值的地方（本项目
//! 已经因此停过一次），所以这里干脆**不写递推**：直接按定义
//!
//! ```text
//! S_{lmn} = ∂^l_x ∂^m_y ∂^n_z C_0(T),   C_s(T) = K·F_s(T),  T = ρ|P−Q|²
//! ```
//!
//! 做**精确的符号求导**。只要记住两条初等事实：
//!
//! ```text
//! ∂/∂R_axis [R^p]          = p·R^(p−1)
//! ∂/∂R_axis [F_s(ρR²)]     = −2ρ·R_axis·F_{s+1}
//! ```
//!
//! 就能把 `S_{lmn}` 表示成
//!
//! ```text
//! S_{lmn} = Σ_k coef_k · R_x^{px} R_y^{py} R_z^{pz} · F_{s_k}(T)
//! ```
//!
//! 逐次求导只是在维护这张表。这是**恒等式级别的推导**，不依赖任何我记不清的
//! 递推系数；且它的输出会被 `i03_general_l.rs` 里的独立数值路径（闭式
//! `(ss|ss)` 的中心导数）逐分量判定。
//!
//! # 装配
//!
//! ```text
//! (ab|cd) = Σ_{t,u,v} E^{ab}_{tuv} Σ_{τ,υ,φ} E^{cd}_{τυφ} (−1)^{τ+υ+φ} R_{t+τ,u+υ,v+φ}
//! ```
//!
//! `(−1)^{τ+υ+φ}` 与「`S` 用不用 (−1)^{l+m+n}`」两种约定在 l=0 时重合，l>0 时
//! 会差整体符号，因此这条约定由测试对照独立路径来确定，不靠记忆。

use crate::boys::boys_functions;
use crate::one_electron::hermite_1d;

const PI: f64 = std::f64::consts::PI;

/// 展开式里的一项：`coef · R_x^px R_y^py R_z^pz · F_s(T)`。
#[derive(Debug, Clone, Copy)]
struct Term {
    powers: [u32; 3],
    boys_index: u32,
    coefficient: f64,
}

/// `C_0` 的某个导数，按「R 的多项式 × Boys 指标」存表。
#[derive(Debug, Clone)]
struct Kernel {
    terms: Vec<Term>,
}

impl Kernel {
    fn fundamental() -> Self {
        Self {
            terms: vec![Term {
                powers: [0, 0, 0],
                boys_index: 0,
                coefficient: 1.0,
            }],
        }
    }

    /// 对 `R_axis` 求一次导（`rho` = ρ，出现在 `∂F_s/∂R = −2ρR F_{s+1}` 里）。
    fn differentiate(&self, axis: usize, rho: f64) -> Self {
        let mut terms = Vec::with_capacity(self.terms.len() * 2);
        for term in &self.terms {
            // 多项式部分求导。
            if term.powers[axis] > 0 {
                let mut powers = term.powers;
                powers[axis] -= 1;
                terms.push(Term {
                    powers,
                    boys_index: term.boys_index,
                    coefficient: term.coefficient * f64::from(term.powers[axis]),
                });
            }
            // F_s 部分求导：多一个 R_axis，Boys 指标 +1，系数乘 −2ρ。
            let mut powers = term.powers;
            powers[axis] += 1;
            terms.push(Term {
                powers,
                boys_index: term.boys_index + 1,
                coefficient: term.coefficient * (-2.0 * rho),
            });
        }
        Self { terms }
    }

    /// 最高用到的 Boys 指标（调用方据此准备 `F_0..F_M`）。
    fn max_boys_index(&self) -> u32 {
        self.terms
            .iter()
            .map(|term| term.boys_index)
            .max()
            .unwrap_or(0)
    }

    /// 代入 `R = P − Q` 求值（`boys[s]` = `F_s(T)`）。
    fn evaluate(&self, r: [f64; 3], boys: &[f64]) -> f64 {
        let mut total = 0.0;
        for term in &self.terms {
            let mut value = term.coefficient * boys[term.boys_index as usize];
            for (axis, component) in r.iter().enumerate() {
                value *= component.powi(term.powers[axis] as i32);
            }
            total += value;
        }
        total
    }
}

/// `∂^l_x∂^m_y∂^n_z F_0(scale·|R|²)`——核吸引与 2e 积分共用的「Hermite Coulomb」导数核。
///
/// 一电子核吸引用的是 `scale = p`、`R = P − C`；两电子积分用的是 `scale = ρ`、
/// `R = P − Q`。两条路径共享这份**按定义求导**的实现，避免再出现「凭记忆写递推」
/// 的风险。
pub(crate) fn derivative_boys(l: u32, m: u32, n: u32, scale: f64, r: [f64; 3]) -> f64 {
    let mut kernel = Kernel::fundamental();
    for _ in 0..l {
        kernel = kernel.differentiate(0, scale);
    }
    for _ in 0..m {
        kernel = kernel.differentiate(1, scale);
    }
    for _ in 0..n {
        kernel = kernel.differentiate(2, scale);
    }
    let t = scale * (0..3).map(|axis| r[axis] * r[axis]).sum::<f64>();
    let boys = boys_functions(kernel.max_boys_index() as usize, t);
    kernel.evaluate(r, &boys)
}

/// 一个笛卡尔分量 `(a b|c d)` 的原函数积分（未归一化、未收缩）。
///
/// `pa`…`pd` 是四个笛卡尔高斯的幂次 `(x, y, z)`。
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn primitive_component(
    pa: [u32; 3],
    a: [f64; 3],
    alpha: f64,
    pb: [u32; 3],
    b: [f64; 3],
    beta: f64,
    pc: [u32; 3],
    c: [f64; 3],
    gamma: f64,
    pd: [u32; 3],
    d: [f64; 3],
    delta: f64,
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
    let ab2 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>();
    let cd2 = (0..3).map(|axis| (c[axis] - d[axis]).powi(2)).sum::<f64>();
    let e_ab = (-alpha * beta / p * ab2).exp();
    let e_cd = (-gamma * delta / q * cd2).exp();
    let pq = p + q;
    let rho = p * q / pq;
    let r = [
        p_center[0] - q_center[0],
        p_center[1] - q_center[1],
        p_center[2] - q_center[2],
    ];
    let prefactor = 2.0 * PI.powf(2.5) / (p * q * pq.sqrt()) * e_ab * e_cd;

    let e_a: Vec<Vec<f64>> = (0..3)
        .map(|axis| {
            hermite_1d(
                pa[axis],
                pb[axis],
                p,
                p_center[axis] - a[axis],
                p_center[axis] - b[axis],
            )
        })
        .collect();
    let e_c: Vec<Vec<f64>> = (0..3)
        .map(|axis| {
            hermite_1d(
                pc[axis],
                pd[axis],
                q,
                q_center[axis] - c[axis],
                q_center[axis] - d[axis],
            )
        })
        .collect();

    let mut total = 0.0;
    for tx in 0..=pa[0] + pb[0] {
        for ty in 0..=pa[1] + pb[1] {
            for tz in 0..=pa[2] + pb[2] {
                let bra = e_a[0][tx as usize] * e_a[1][ty as usize] * e_a[2][tz as usize];
                if bra == 0.0 {
                    continue;
                }
                for ux in 0..=pc[0] + pd[0] {
                    for uy in 0..=pc[1] + pd[1] {
                        for uz in 0..=pc[2] + pd[2] {
                            let ket =
                                e_c[0][ux as usize] * e_c[1][uy as usize] * e_c[2][uz as usize];
                            if ket == 0.0 {
                                continue;
                            }
                            let sign = if (ux + uy + uz) % 2 == 0 { 1.0 } else { -1.0 };
                            total += bra
                                * ket
                                * sign
                                * derivative_boys(tx + ux, ty + uy, tz + uz, rho, r);
                        }
                    }
                }
            }
        }
    }
    prefactor * total
}
