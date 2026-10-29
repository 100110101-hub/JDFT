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

use crate::boys::boys_functions_into;
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
    // 热路径：栈上缓冲。原函数级 2e 导数每个四元组都要调一次 Boys，
    // 每次分配一个 Vec 在 debug 构建里能占掉一半以上的时间。
    let m_max = kernel.max_boys_index() as usize;
    let mut boys = [0.0_f64; 20];
    boys_functions_into(m_max, t, &mut boys[..=m_max]);
    kernel.evaluate(r, &boys[..=m_max])
}

/// 一个笛卡尔分量 `(a b|c d)` 的原函数积分（未归一化、未收缩）。
///
/// `pa`…`pd` 是四个笛卡尔高斯的幂次 `(x, y, z)`。
/// 一个 primitive 四元组里**与幂次无关**的那部分（中心、指数、预因子）以及
/// Hermite-Coulomb 核的懒缓存 —— 求导数积分时 12 次调用共用同一份。
struct PrimitiveContext {
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
    p: f64,
    q: f64,
    p_center: [f64; 3],
    q_center: [f64; 3],
    rho: f64,
    r: [f64; 3],
    prefactor: f64,
    /// 懒缓存：`derivative_boys(l,m,n)` 只依赖三元组，跨 12 次调用复用。
    kernel: Vec<f64>,
    /// 每个方向的最高 Hermite 指标（缓存的长宽高）。
    ly: u32,
    lz: u32,
}

impl PrimitiveContext {
    fn new(
        a: [f64; 3],
        alpha: f64,
        b: [f64; 3],
        beta: f64,
        c: [f64; 3],
        gamma: f64,
        d: [f64; 3],
        delta: f64,
        max_powers: [u32; 3],
    ) -> Self {
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
        let (lx, ly, lz) = (max_powers[0], max_powers[1], max_powers[2]);
        Self {
            a,
            b,
            c,
            d,
            p,
            q,
            p_center,
            q_center,
            rho,
            r,
            prefactor,
            kernel: vec![f64::NAN; ((lx + 1) * (ly + 1) * (lz + 1)) as usize],
            ly,
            lz,
        }
    }

    /// 给定的四个幂次三元组的原函数积分（未归一化、未收缩）。
    fn integral(&mut self, pa: [u32; 3], pb: [u32; 3], pc: [u32; 3], pd: [u32; 3]) -> f64 {
        let e_a: Vec<Vec<f64>> = (0..3)
            .map(|axis| {
                hermite_1d(
                    pa[axis],
                    pb[axis],
                    self.p,
                    self.p_center[axis] - self.a[axis],
                    self.p_center[axis] - self.b[axis],
                )
            })
            .collect();
        let e_c: Vec<Vec<f64>> = (0..3)
            .map(|axis| {
                hermite_1d(
                    pc[axis],
                    pd[axis],
                    self.q,
                    self.q_center[axis] - self.c[axis],
                    self.q_center[axis] - self.d[axis],
                )
            })
            .collect();
        let stride_y = (self.lz + 1) as usize;
        let stride_x = ((self.ly + 1) * (self.lz + 1)) as usize;
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
                                let index = ((tx + ux) * stride_x as u32
                                    + (ty + uy) * stride_y as u32
                                    + (tz + uz))
                                    as usize;
                                let value = if self.kernel[index].is_nan() {
                                    let computed = derivative_boys(
                                        tx + ux,
                                        ty + uy,
                                        tz + uz,
                                        self.rho,
                                        self.r,
                                    );
                                    self.kernel[index] = computed;
                                    computed
                                } else {
                                    self.kernel[index]
                                };
                                total += bra * ket * sign * value;
                            }
                        }
                    }
                }
            }
        }
        self.prefactor * total
    }
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
    let max_powers = [
        pa[0] + pb[0] + pc[0] + pd[0],
        pa[1] + pb[1] + pc[1] + pd[1],
        pa[2] + pb[2] + pc[2] + pd[2],
    ];
    let mut context = PrimitiveContext::new(a, alpha, b, beta, c, gamma, d, delta, max_powers);
    context.integral(pa, pb, pc, pd)
}

/// 一个 primitive 四元组对**四个中心**的导数 `[[∂/∂A_i]; [∂/∂B_i]; [∂/∂C_i]; [∂/∂D_i]]`。
///
/// 用抬/降幂恒等式 `∂/∂A_k = 2α·(a+e_k) − a_k·(a−e_k)`，而四个中心共享同一份
/// [`PrimitiveContext`]（预因子、指数、Hermite-Coulomb 核缓存）—— 这正是原来那 24 次
/// 独立 `primitive_component` 调用（4 槽 × 3 轴 × 抬/降）里最贵的那部分 setup。
///
/// `∂/∂D` 由平动不变性 `∂/∂A + ∂/∂B + ∂/∂C + ∂/∂D = 0` 得到。
///
/// # Panics
///
/// 不会 panic；幂次都是 `u32`，抬升后仍在缓存范围内（缓存按 `max_powers` 开）。
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn primitive_component_derivatives(
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
) -> [[f64; 3]; 4] {
    // 缓存要覆盖「某一侧抬一阶」后的最高指标。
    let max_powers = [
        pa[0] + pb[0] + pc[0] + pd[0] + 1,
        pa[1] + pb[1] + pc[1] + pd[1] + 1,
        pa[2] + pb[2] + pc[2] + pd[2] + 1,
    ];
    let mut context = PrimitiveContext::new(a, alpha, b, beta, c, gamma, d, delta, max_powers);
    let mut out = [[0.0_f64; 3]; 4];
    let centers = [pa, pb, pc, pd];
    let exponents = [alpha, beta, gamma, delta];
    for slot in 0..3 {
        for axis in 0..3 {
            let mut raised = centers;
            raised[slot][axis] += 1;
            let mut value = 2.0
                * exponents[slot]
                * context.integral(raised[0], raised[1], raised[2], raised[3]);
            if centers[slot][axis] > 0 {
                let mut lowered = centers;
                lowered[slot][axis] -= 1;
                value -= f64::from(centers[slot][axis])
                    * context.integral(lowered[0], lowered[1], lowered[2], lowered[3]);
            }
            out[slot][axis] = value;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for axis in 0..3 {
        out[3][axis] = -(out[0][axis] + out[1][axis] + out[2][axis]);
    }
    out
}
