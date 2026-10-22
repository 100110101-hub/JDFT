//! I-03：**两电子积分** `(ab|cd)`。
//!
//! # 现状（诚实边界）
//!
//! - **s 型（`l = 0`）已实现并交叉验证**：`(ss|ss)` 用闭合式
//!   `2π^{5/2}/(p q √(p+q))·E_AB·E_CD·F_0(α)`，`α = pq/(p+q)|P−Q|²`。
//!   T1（H2/STO-3G）只需要这一条。
//! - **l>0 未实现**：Obara–Saika 的垂直递推要建立在正确的 Hermite Coulomb 族上，
//!   而后者目前只对 `l = 0` 可信（见 `one_electron::primitive_nuclear` 的说明）。
//!   因此 [`two_electron`] 遇到 l>0 壳层显式返回 `Unsupported`，**不会**返回错值。
//!
//! # 独立交叉验证
//!
//! 生产路径走 **Boys 函数**；`tests/d03_two_electron.rs` 另走一条 **erf 卷积**路径：
//! 把 6 维积分先做掉一个中心（"高斯电荷的势 = erf(√W r)/r"），再对差坐标积分，
//! 得到
//!
//! ```text
//! (ss|ss) = E_AB E_CD (π/(p+q))^{3/2} (π/W)^{3/2} erf(√W |d|)/|d|,  W = pq/(p+q), d = P−Q
//! ```
//!
//! 两条路在数学上等价但计算路径完全不同（一条用 Boys 递推、一条用 erf），
//! 所以匹配到 1e-12 才算数。

#![allow(clippy::too_many_arguments)] // 4 个原函数 = 4×(指数, 中心)，签名由数学决定

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};

use crate::boys::boys_functions;

const PI: f64 = std::f64::consts::PI;

/// 由两个 s 型（未归一化）原函数构成的高斯乘积：`p`、`P`、`E_AB`。
fn s_pair(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> (f64, [f64; 3], f64) {
    let p = alpha + beta;
    let center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let r2 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>();
    (p, center, (-alpha * beta / p * r2).exp())
}

/// 四个（未归一化）s 型原函数的 `(ss|ss)`。
///
/// 闭合式：`2π^{5/2}/(p q √(p+q)) E_AB E_CD F_0(α)`。
#[must_use]
pub fn primitive_ssss(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
) -> f64 {
    let (p, p_center, e_ab) = s_pair(alpha, a, beta, b);
    let (q, q_center, e_cd) = s_pair(gamma, c, delta, d);
    let pq = p + q;
    let r2 = (0..3)
        .map(|axis| (p_center[axis] - q_center[axis]).powi(2))
        .sum::<f64>();
    let alpha_boys = p * q / pq * r2;
    let f0 = boys_functions(0, alpha_boys)[0];

    2.0 * PI.powf(2.5) / (p * q * pq.sqrt()) * e_ab * e_cd * f0
}

/// 两电子积分张量（化学家记号 `(ij|kl)`，扁平 `i*n³ + j*n² + k*n + l`）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TwoElectronIntegrals {
    /// 基函数个数。
    pub n_functions: usize,
    /// 扁平存储的 `(ij|kl)`。
    pub values: Vec<f64>,
}

impl TwoElectronIntegrals {
    /// 取 `(i j | k l)`。
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.values[self.index(i, j, k, l)]
    }

    fn index(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        let n = self.n_functions;
        ((i * n + j) * n + k) * n + l
    }
}

/// 计算全部分子轨道的两电子积分（当前只支持全 s 基组）。
///
/// # Errors
///
/// 基组里存在 `l > 0` 的壳层 → [`EngineError::Unsupported`]（见模块文档）。
pub fn two_electron(
    basis: &InternalBasis,
    _molecule: &Molecule,
) -> Result<TwoElectronIntegrals, EngineError> {
    if let Some(shell) = basis.shells().iter().find(|shell| shell.l > 0) {
        return Err(EngineError::Unsupported(format!(
            "two-electron integrals for l>0 shells are not implemented yet (I-03: \
             Obara-Saika vertical recursion needs the corrected Hermite Coulomb family): \
             the basis has an l={} shell",
            shell.l
        )));
    }

    let n = basis.n_functions();
    let mut values = vec![0.0; n * n * n * n];

    // 每个壳层贡献 1 个 s 型 AO（因为只支持 l=0），壳层序号即 AO 序号。
    let shells: Vec<&qwb_basis::InternalShell> = basis.shells().iter().collect();
    debug_assert_eq!(shells.len(), n);

    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            for (ic, shell_c) in shells.iter().enumerate() {
                for (id, shell_d) in shells.iter().enumerate() {
                    let mut sum = 0.0;
                    for (alpha, ca) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                        let norm_a = crate::one_electron::primitive_norm((0, 0, 0), *alpha);
                        for (beta, cb) in shell_b.exponents.iter().zip(&shell_b.coefficients) {
                            let norm_b = crate::one_electron::primitive_norm((0, 0, 0), *beta);
                            for (gamma, cc) in shell_c.exponents.iter().zip(&shell_c.coefficients) {
                                let norm_c = crate::one_electron::primitive_norm((0, 0, 0), *gamma);
                                for (delta, cd) in
                                    shell_d.exponents.iter().zip(&shell_d.coefficients)
                                {
                                    let norm_d =
                                        crate::one_electron::primitive_norm((0, 0, 0), *delta);
                                    let weight =
                                        ca * cb * cc * cd * norm_a * norm_b * norm_c * norm_d;
                                    sum += weight
                                        * primitive_ssss(
                                            *alpha,
                                            shell_a.center_bohr,
                                            *beta,
                                            shell_b.center_bohr,
                                            *gamma,
                                            shell_c.center_bohr,
                                            *delta,
                                            shell_d.center_bohr,
                                        );
                                }
                            }
                        }
                    }
                    // bra 与 ket 各自的对称性：填 8 个等价位置。
                    let element = sum;
                    let index = ((ia * n + ib) * n + ic) * n + id;
                    values[index] = element;
                    let mirror =
                        |i: usize, j: usize, k: usize, l: usize| ((i * n + j) * n + k) * n + l;
                    values[mirror(ib, ia, ic, id)] = element;
                    values[mirror(ia, ib, id, ic)] = element;
                    values[mirror(ic, id, ia, ib)] = element;
                    values[mirror(id, ic, ia, ib)] = element;
                    values[mirror(ia, ib, id, ic)] = element;
                    values[mirror(ib, ia, id, ic)] = element;
                    values[mirror(ic, id, ib, ia)] = element;
                    values[mirror(id, ic, ib, ia)] = element;
                }
            }
        }
    }

    Ok(TwoElectronIntegrals {
        n_functions: n,
        values,
    })
}
