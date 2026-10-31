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
use crate::one_electron::hermite_1d_into;

const PI: f64 = std::f64::consts::PI;

/// 展开式里的一项，存成**与 ρ 无关的模板**：`sign · (−2ρ)^rho_power · R^p · F_s(T)`。
///
/// `sign_factor` 只由「降幂的整数倍率 × 负号」累积而来 —— 都是精确的整数乘积。
#[derive(Debug, Clone, Copy)]
struct TermTemplate {
    powers: [u32; 3],
    boys_index: u32,
    rho_power: u32,
    sign_factor: f64,
}

/// `C_0` 的某个导数展开成「R 的多项式 × Boys 指标」的项表。
///
/// **只依赖 `(l,m,n)`**：与 `ρ`、与几何都无关。RS-05 的成本归因发现每次调用都重做
/// 一遍「符号求导」占掉了可观的时间，所以按 `(l,m,n)` 缓存（见 [`kernel_template`]）。
#[derive(Debug, Clone)]
struct KernelTemplate {
    terms: Vec<TermTemplate>,
    max_boys: u32,
}

impl KernelTemplate {
    fn fundamental() -> Self {
        Self {
            terms: vec![TermTemplate {
                powers: [0, 0, 0],
                boys_index: 0,
                rho_power: 0,
                sign_factor: 1.0,
            }],
            max_boys: 0,
        }
    }

    /// 对 `R_axis` 求一次导（`∂F_s/∂R = −2ρR F_{s+1}` ⇒ 记一个 ρ 幂与一个负号）。
    fn differentiate(&self, axis: usize) -> Self {
        let mut terms = Vec::with_capacity(self.terms.len() * 2);
        let mut max_boys = 0_u32;
        for term in &self.terms {
            // 多项式部分求导：系数乘上「降幂前的幂次」（整数，精确）。
            if term.powers[axis] > 0 {
                let mut powers = term.powers;
                powers[axis] -= 1;
                terms.push(TermTemplate {
                    powers,
                    boys_index: term.boys_index,
                    rho_power: term.rho_power,
                    sign_factor: term.sign_factor * f64::from(term.powers[axis]),
                });
            }
            // F_s 部分求导：多一个 R_axis，Boys 指标 +1，系数乘 −2ρ（记成 ρ 的一次幂）。
            let mut powers = term.powers;
            powers[axis] += 1;
            terms.push(TermTemplate {
                powers,
                boys_index: term.boys_index + 1,
                rho_power: term.rho_power + 1,
                sign_factor: -term.sign_factor,
            });
            max_boys = max_boys.max(term.boys_index + 1);
        }
        Self { terms, max_boys }
    }
}

/// 每个线程一份的 `(l,m,n) → 模板` 缓存（下标 `l·16² + m·16 + n`，上限 16）。
///
/// 纯函数缓存：只影响速度，不影响任何数值，也不引入跨线程共享状态。
fn kernel_template(l: u32, m: u32, n: u32) -> std::rc::Rc<KernelTemplate> {
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<Option<std::rc::Rc<KernelTemplate>>>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_empty() {
            cache.resize(16 * 16 * 16, None);
        }
        let index = ((l as usize) * 16 + m as usize) * 16 + n as usize;
        if let Some(template) = &cache[index] {
            return std::rc::Rc::clone(template);
        }
        let mut kernel = KernelTemplate::fundamental();
        for _ in 0..l {
            kernel = kernel.differentiate(0);
        }
        for _ in 0..m {
            kernel = kernel.differentiate(1);
        }
        for _ in 0..n {
            kernel = kernel.differentiate(2);
        }
        let template = std::rc::Rc::new(kernel);
        cache[index] = Some(std::rc::Rc::clone(&template));
        template
    })
}

/// `∂^l_x∂^m_y∂^n_z F_0(scale·|R|²)`——核吸引与 2e 积分共用的「Hermite Coulomb」导数核。
///
/// 一电子核吸引用的是 `scale = p`、`R = P − C`；两电子积分用的是 `scale = ρ`、
/// `R = P − Q`。两条路径共享这份**按定义求导**的实现（模板按 `(l,m,n)` 缓存），
/// 避免再出现「凭记忆写递推」的风险。
pub(crate) fn derivative_boys(l: u32, m: u32, n: u32, scale: f64, r: [f64; 3]) -> f64 {
    let kernel = kernel_template(l, m, n);
    let t = scale * (0..3).map(|axis| r[axis] * r[axis]).sum::<f64>();
    // 热路径：栈上缓冲。原函数级 2e 导数每个四元组都要调一次 Boys，
    // 每次分配一个 Vec 在 debug 构建里能占掉一半以上的时间。
    let m_max = kernel.max_boys as usize;
    let mut boys = [0.0_f64; 32];
    assert!(m_max < boys.len(), "Boys 指标 {m_max} 超出栈缓冲");
    boys_functions_into(m_max, t, &mut boys[..=m_max]);
    // 模板里已经记了一个负号（每次 F 求导乘 −2ρ），所以这里只乘 2ρ。
    let two_rho = 2.0 * scale;
    let mut total = 0.0;
    for term in kernel.terms.iter() {
        let mut coefficient = term.sign_factor;
        for _ in 0..term.rho_power {
            coefficient *= two_rho;
        }
        let mut value = coefficient * boys[term.boys_index as usize];
        for (axis, component) in r.iter().enumerate() {
            value *= component.powi(term.powers[axis] as i32);
        }
        total += value;
    }
    total
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
    ///
    /// E 系数用调用方给的栈上缓冲（[`hermite_1d_into`]），不再每次分配 `Vec`：
    /// 二阶导数的一条积分要调几百次，分配开销是它最主要的浪费。
    fn integral(
        &mut self,
        pa: [u32; 3],
        pb: [u32; 3],
        pc: [u32; 3],
        pd: [u32; 3],
        scratch: &mut [f64],
    ) -> f64 {
        let mut e_a = [[0.0_f64; 16]; 3];
        let mut e_c = [[0.0_f64; 16]; 3];
        for axis in 0..3 {
            let t_max = (pa[axis] + pb[axis]) as usize;
            hermite_1d_into(
                pa[axis],
                pb[axis],
                self.p,
                self.p_center[axis] - self.a[axis],
                self.p_center[axis] - self.b[axis],
                &mut e_a[axis][..=t_max],
                scratch,
            );
            let u_max = (pc[axis] + pd[axis]) as usize;
            hermite_1d_into(
                pc[axis],
                pd[axis],
                self.q,
                self.q_center[axis] - self.c[axis],
                self.q_center[axis] - self.d[axis],
                &mut e_c[axis][..=u_max],
                scratch,
            );
        }
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
    let mut scratch = [0.0_f64; 16 * 16 * 16];
    context.integral(pa, pb, pc, pd, &mut scratch)
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
    let mut scratch = [0.0_f64; 16 * 16 * 16];
    let mut out = [[0.0_f64; 3]; 4];
    let centers = [pa, pb, pc, pd];
    let exponents = [alpha, beta, gamma, delta];
    for slot in 0..3 {
        for axis in 0..3 {
            let mut raised = centers;
            raised[slot][axis] += 1;
            let mut value = 2.0
                * exponents[slot]
                * context.integral(raised[0], raised[1], raised[2], raised[3], &mut scratch);
            if centers[slot][axis] > 0 {
                let mut lowered = centers;
                lowered[slot][axis] -= 1;
                value -= f64::from(centers[slot][axis])
                    * context.integral(
                        lowered[0],
                        lowered[1],
                        lowered[2],
                        lowered[3],
                        &mut scratch,
                    );
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

/// 四中心的幂次组（每个中心一个 `(x, y, z)`）。
pub type Power4 = [[u32; 3]; 4];

/// 把「抬/降幂」线性算子作用在展开表的每一项上（RS-05 的二阶导数基元）。
///
/// ```text
/// ∂/∂A_{s,i} f[幂次] = 2·ζ_s·f[+e_i] − 幂次_i·f[−e_i]
/// ```
///
/// `系数`乘的是**降幂项**里的「当前幂次」——注意它可能已经被更早的算子抬过一次，
/// 所以必须在**逐项**的幂次上取，不能在外层预先取原幂次。
fn apply_shifting_operator(terms: &mut Vec<(Power4, f64)>, slot: usize, axis: usize, zeta: f64) {
    let mut next = Vec::with_capacity(terms.len() * 2);
    for (powers, coefficient) in terms.iter() {
        let mut raised = *powers;
        raised[slot][axis] += 1;
        next.push((raised, coefficient * 2.0 * zeta));
        let order = powers[slot][axis];
        if order > 0 {
            let mut lowered = *powers;
            lowered[slot][axis] -= 1;
            next.push((lowered, coefficient * -f64::from(order)));
        }
    }
    *terms = next;
}

/// 查出（或算出并缓存）某个幂次组的原函数积分。
///
/// 缓存是线性扫描（幂次组只有几百个）；实测把它换成 `HashMap` 没有改善，说明
/// 真正的成本在 `PrimitiveContext::integral` 本身（每次要重建 Hermite 系数表）。
fn cached_integral(
    context: &mut PrimitiveContext,
    cache: &mut Vec<(Power4, f64)>,
    powers: Power4,
    scratch: &mut [f64],
) -> f64 {
    if let Some((_, value)) = cache.iter().find(|(key, _)| *key == powers) {
        return *value;
    }
    let value = context.integral(powers[0], powers[1], powers[2], powers[3], scratch);
    cache.push((powers, value));
    value
}

/// 一个 primitive 四元组对**四个中心两两**的二阶导数。
///
/// `out[s][t][i][j] = ∂²/∂A_{s,i}∂A_{t,j}(ab|cd)`（`s,t` ∈ `0..4`，`i,j` ∈ `0..3` 是 x/y/z）。
///
/// # 为什么复用一阶导数的恒等式
///
/// 二阶导数的传统写法是「把 Hermite 展开推进到 L+2 阶」——那要再写一套递推，而本项目
/// 在 2e 积分上已经因为「凭记忆写递推」出过错。这里换一条等价的路：把一阶导数已经
/// 用了很多轮、并被独立数值路径验证过的抬/降幂恒等式
///
/// ```text
/// ∂/∂A_{s,i} f = 2·ζ_s·f[+e_i] − 幂次_i·f[−e_i]
/// ```
///
/// 当成**线性算子连续作用两次**，得到「若干移位幂次的积分 × 系数」的展开，再用同一份
/// [`PrimitiveContext`]（预因子、指数、Hermite–Coulomb 核缓存）把展开里出现的所有移位
/// 幂次各算一次。于是二阶导数没有引入任何新的求积公式，只有代数展开。
///
/// # 成本
///
/// 一次一阶导数要 12 个分量、约 24 次移位积分；一次二阶导数要 144 个分量，展开里出现的
/// **不同**幂次组大约 200–300 个。所以单位原函数四元组的成本大约是 8–12 倍——这个倍数是
/// RS-05 判「CPHF Hessian 能不能同时满足 20×」的输入，实测数字记在 `docs/choices.md`。
///
/// # Panics
///
/// 不 panic（幂次是 `u32`，抬升两次后仍在按 `max_powers + 2` 开的缓存范围内）。
#[allow(clippy::too_many_arguments)]
// 四个中心 × 三个轴的 4 重下标循环是这段数学最直白的写法（144 个分量一一对应），
// 改成迭代器只会把人绕晕。
#[allow(clippy::needless_range_loop)]
#[must_use]
pub fn primitive_component_second_derivatives(
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
) -> [[[[f64; 3]; 3]; 4]; 4] {
    // 连续两次抬幂 → 每个方向最多 +2。
    let max_powers = [
        pa[0] + pb[0] + pc[0] + pd[0] + 2,
        pa[1] + pb[1] + pc[1] + pd[1] + 2,
        pa[2] + pb[2] + pc[2] + pd[2] + 2,
    ];
    let mut context = PrimitiveContext::new(a, alpha, b, beta, c, gamma, d, delta, max_powers);
    let base: Power4 = [pa, pb, pc, pd];
    let exponents = [alpha, beta, gamma, delta];
    // 展开表按 (s,t,i,j) 现场构造——表很小（≤ 8 项），比预先存 144 份更省内存。
    let mut cache: Vec<(Power4, f64)> = Vec::new();
    let mut scratch = [0.0_f64; 16 * 16 * 16];
    let mut out = [[[[0.0_f64; 3]; 3]; 4]; 4];
    // 只算前三个中心的两两导数（9 个中心对），第四个中心由**平动不变性**
    // `∂²/∂A_3∂A_t = −Σ_{s<3} ∂²/∂A_s∂A_t` 得到 —— 省掉 7/16 的展开量。
    for s in 0..3 {
        for t in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    let mut terms: Vec<(Power4, f64)> = vec![(base, 1.0)];
                    // 先作用在 t（外侧），再作用在 s —— 同一中心两次作用时顺序也不能乱。
                    apply_shifting_operator(&mut terms, t, j, exponents[t]);
                    apply_shifting_operator(&mut terms, s, i, exponents[s]);
                    let mut total = 0.0;
                    for (powers, coefficient) in terms {
                        total += coefficient
                            * cached_integral(&mut context, &mut cache, powers, &mut scratch);
                    }
                    out[s][t][i][j] = total;
                }
            }
        }
    }
    for t in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                // ∂²/∂A_3,i ∂A_t,j = −Σ_{s<3} ∂²/∂A_s,i ∂A_t,j
                let column: f64 = (0..3).map(|s| out[s][t][i][j]).sum();
                out[3][t][i][j] = -column;
                // ∂²/∂A_t,i ∂A_3,j = −Σ_{b<3} ∂²/∂A_t,i ∂A_b,j
                let row: f64 = (0..3).map(|b| out[t][b][i][j]).sum();
                out[t][3][i][j] = -row;
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let sum: f64 = (0..3)
                .map(|s| (0..3).map(|t| out[s][t][i][j]).sum::<f64>())
                .sum();
            out[3][3][i][j] = sum;
        }
    }
    out
}
