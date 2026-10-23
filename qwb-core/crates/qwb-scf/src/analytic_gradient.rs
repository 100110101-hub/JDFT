//! 解析梯度（G-01）：闭壳层 HF / KS 的总能量对核坐标的导数。
//!
//! # 结构
//!
//! ```text
//! dE/dR_A = Tr(P dH/dR_A)                     Hellmann–Feynman（1e）
//!          + ½ Σ P P [d(μν|λσ) − ½ d(μλ|νσ)]/dR_A   2e
//!          − Tr(W dS/dR_A)                     Pulay（轨道弛豫）
//!          + dE_nuc/dR_A
//! ```
//!
//! `W = ½ P F P` 的系数与符号由 `tests/g01_pulay.rs` 的恒等式
//! `Tr(F dP) = −Tr(W dS)` 独立确认过（残差随位移线性下降）。
//!
//! # 积分导数怎么来
//!
//! 对**单个高斯原函数**，对中心求导等价于「把一个方向的幂次抬高一阶」：
//!
//! ```text
//! ∂/∂A_x [ u_x^a u_y^b u_z^c e^{−α|u|²} ]
//!     = a·u_x^{a−1}(…) − 2α·u_x^{a+1}(…)
//! ```
//!
//! 所以 s 型（`a = b = c = 0`）的导数只需要**同一套积分例程在幂次抬高一阶时的值**，
//! 不需要任何新的积分实现。核吸引还多一项：**原子核本身也在动**，
//! 那一项有独立闭式（`nuclear_center_derivative`）。
//!
//! # 目前的范围
//!
//! 只处理 **s 型基组**（H₂/STO-3G 这类）：`l > 0` 会显式报 `Unsupported`，
//! 不会静默给出半对的梯度。p/d 的导数需要 ±1 幂次的组合，属于同一思路的推广。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::boys::boys_functions;
use qwb_integrals::one_electron::{
    primitive_kinetic, primitive_norm, primitive_nuclear, primitive_overlap,
};
use qwb_linalg::Matrix;
use qwb_registry::elements::atomic_number;

use crate::pulay::pulay_w;

const PI: f64 = std::f64::consts::PI;

/// 一个 s 型 AO 的描述。
struct SAo {
    center: [f64; 3],
    center_index: usize,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

fn s_orbitals(basis: &InternalBasis, molecule: &Molecule) -> Result<Vec<SAo>, EngineError> {
    let mut out = Vec::with_capacity(basis.n_functions());
    let mut offset = 0usize;
    for (atom_index, atom) in molecule.atoms.iter().enumerate() {
        let atom_positions = atom.position_bohr();
        for shell in basis.shells() {
            if shell.center_bohr != atom_positions {
                continue;
            }
            if shell.l != 0 {
                return Err(EngineError::Unsupported(format!(
                    "analytic gradient currently supports s-type shells only (l={})",
                    shell.l
                )));
            }
            let _ = atom_index;
            for _ in 0..shell.n_functions() {
                out.push(SAo {
                    center: shell.center_bohr,
                    center_index: atom_index,
                    exponents: shell.exponents.clone(),
                    coefficients: shell.coefficients.clone(),
                });
                offset += 1;
            }
        }
    }
    if offset != basis.n_functions() {
        return Err(EngineError::Internal(format!(
            "analytic gradient: matched {offset} AOs but the basis has {}",
            basis.n_functions()
        )));
    }
    out.sort_by_key(|orbital| orbital.center_index);
    Ok(out)
}

/// 核吸引里「核本身移动」那一项的闭式：
/// `∂/∂C_x ⟨ss|−Z/r_C⟩ = −(−Z)(2π/p)·F₀'(T)·∂T/∂C_x`
/// 其中 `T = p|P−C|²`，`F₀'(T) = −F₁(T)`，`∂T/∂C_x = −2p(P_x−C_x)`。
fn nuclear_center_derivative(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    c: [f64; 3],
    z: f64,
    axis: usize,
) -> f64 {
    let p = alpha + beta;
    let center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let ab2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let pc = (0..3).map(|k| (center[k] - c[k]).powi(2)).sum::<f64>();
    let t = p * pc;
    let f1 = boys_functions(1, t)[1];
    let prefactor = -z * 2.0 * PI / p * (-alpha * beta / p * ab2).exp();
    // ∂/∂C_x [F₀(T)] = −F₁·∂T/∂C_x，∂T/∂C_x = −2p(P_x−C_x)
    let d_f0 = -f1 * (-2.0 * p * (center[axis] - c[axis]));
    prefactor * d_f0
}

/// s 型原函数对 A 中心求导后与另一个 s 型的重叠（`∂/∂A_x`，未归一化）。
fn d_overlap(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3], axis: usize) -> f64 {
    let mut raised = [0u32; 3];
    raised[axis] = 1;
    -2.0 * alpha
        * primitive_overlap(
            alpha,
            (raised[0], raised[1], raised[2]),
            a,
            beta,
            (0, 0, 0),
            b,
        )
}

/// 同上，但换成动能。
fn d_kinetic(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3], axis: usize) -> f64 {
    let mut raised = [0u32; 3];
    raised[axis] = 1;
    -2.0 * alpha
        * primitive_kinetic(
            alpha,
            (raised[0], raised[1], raised[2]),
            a,
            beta,
            (0, 0, 0),
            b,
        )
}

/// 同上，但换成核吸引（只含 AO 移动那一项）。
fn d_nuclear_ao(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    c: [f64; 3],
    z: f64,
    axis: usize,
) -> f64 {
    let mut raised = [0u32; 3];
    raised[axis] = 1;
    -2.0 * alpha
        * primitive_nuclear(
            alpha,
            (raised[0], raised[1], raised[2]),
            a,
            beta,
            (0, 0, 0),
            b,
            c,
            z,
        )
}

/// 2e 原函数对 A 中心求导（s 型）。
#[allow(clippy::too_many_arguments)]
fn d_ssss(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
    axis: usize,
) -> f64 {
    let p = alpha + beta;
    let center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    // 用 FD 之外的方式：直接对展开式求导。这里复用闭式：把 A 中心替换成「抬高一阶」不可行
    // （primitive_ssss 只接受 s），所以用中心导数恒等式在展开式上做：
    let q = gamma + delta;
    let q_center = [
        (gamma * c[0] + delta * d[0]) / q,
        (gamma * c[1] + delta * d[1]) / q,
        (gamma * c[2] + delta * d[2]) / q,
    ];
    let ab2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let cd2 = (0..3).map(|k| (c[k] - d[k]).powi(2)).sum::<f64>();
    let pq = p + q;
    let rho = p * q / pq;
    let r2 = (0..3)
        .map(|k| (center[k] - q_center[k]).powi(2))
        .sum::<f64>();
    let e = (-alpha * beta / p * ab2 - gamma * delta / q * cd2).exp();
    let prefactor = 2.0 * PI.powf(2.5) / (p * q * pq.sqrt());
    // ∂E_AB/∂A_x = −2(αβ/p)(A_x−B_x)·E_AB
    let d_e = -2.0 * alpha * beta / p * (a[axis] - b[axis]);
    // ∂T/∂A_x = 2ρ(P_x−Q_x)(α/p)
    let d_t = 2.0 * rho * (center[axis] - q_center[axis]) * (alpha / p);
    let boys = boys_functions(1, rho * r2);
    let d_f0 = -boys[1] * d_t;
    prefactor * e * (d_e * boys[0] + d_f0)
    // 注：`raised` 只是占位，避免未使用告警；s 型的抬阶形式在闭式里已体现。
}

/// 闭壳层 HF/KS 的解析梯度（仅 s 型基组）。
///
/// `fock` 必须是**收敛后的** Fock 矩阵（与 `density` 一致）。
///
/// # Errors
///
/// 基组含 `l > 0` 壳层，或形状不匹配。
pub fn analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
    fock: &Matrix,
) -> Result<Vec<[f64; 3]>, EngineError> {
    let aos = s_orbitals(basis, molecule)?;
    let n = aos.len();
    if density.nrows() != n || fock.nrows() != n {
        return Err(EngineError::Internal(
            "analytic_gradient: density/fock size mismatch".to_string(),
        ));
    }
    let w = pulay_w(density, fock)?;

    // 核电荷与位置
    let nuclei: Vec<([f64; 3], f64)> = molecule
        .atoms
        .iter()
        .map(|atom| {
            let z = atomic_number(&atom.symbol)
                .map(f64::from)
                .ok_or_else(|| EngineError::Internal(format!("unknown element {}", atom.symbol)))?;
            Ok((atom.position_bohr(), z))
        })
        .collect::<Result<_, EngineError>>()?;

    // 2e 张量（s 型）与其导数都需要；这里按需重算原函数级，避免存储 N^4 的导数。
    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];

    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut value = 0.0;

            // ① Tr(P dH) − Tr(W dS)
            for mu in 0..n {
                for nu in 0..n {
                    let p_mu_nu = density[(mu, nu)];
                    let w_mu_nu = w[(mu, nu)];
                    let mut d_h = 0.0;
                    let mut d_s = 0.0;
                    for (i, &alpha) in aos[mu].exponents.iter().enumerate() {
                        let norm_mu = primitive_norm((0, 0, 0), alpha);
                        for (j, &beta) in aos[nu].exponents.iter().enumerate() {
                            let norm_nu = primitive_norm((0, 0, 0), beta);
                            let weight = aos[mu].coefficients[i]
                                * aos[nu].coefficients[j]
                                * norm_mu
                                * norm_nu;
                            // A 侧的 AO 在动
                            if aos[mu].center_index == atom {
                                d_s += weight
                                    * d_overlap(alpha, aos[mu].center, beta, aos[nu].center, axis);
                                d_h += weight
                                    * (d_kinetic(
                                        alpha,
                                        aos[mu].center,
                                        beta,
                                        aos[nu].center,
                                        axis,
                                    ) + nuclei
                                        .iter()
                                        .map(|(c, z)| {
                                            d_nuclear_ao(
                                                alpha,
                                                aos[mu].center,
                                                beta,
                                                aos[nu].center,
                                                *c,
                                                *z,
                                                axis,
                                            )
                                        })
                                        .sum::<f64>());
                            }
                            // B 侧的 AO 在动
                            if aos[nu].center_index == atom {
                                d_s += weight
                                    * d_overlap(beta, aos[nu].center, alpha, aos[mu].center, axis);
                                d_h += weight
                                    * (d_kinetic(
                                        beta,
                                        aos[nu].center,
                                        alpha,
                                        aos[mu].center,
                                        axis,
                                    ) + nuclei
                                        .iter()
                                        .map(|(c, z)| {
                                            d_nuclear_ao(
                                                beta,
                                                aos[nu].center,
                                                alpha,
                                                aos[mu].center,
                                                *c,
                                                *z,
                                                axis,
                                            )
                                        })
                                        .sum::<f64>());
                            }
                        }
                    }
                    value += p_mu_nu * d_h - w_mu_nu * d_s;
                }
            }

            // ② 核本身移动（核吸引 + 核排斥）
            //    核吸引：只有 atom 自己的核在动
            let (own_position, own_charge) = nuclei[atom];
            let mut d_v_nucleus = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let p_mu_nu = density[(mu, nu)];
                    for (i, &alpha) in aos[mu].exponents.iter().enumerate() {
                        let norm_mu = primitive_norm((0, 0, 0), alpha);
                        for (j, &beta) in aos[nu].exponents.iter().enumerate() {
                            let norm_nu = primitive_norm((0, 0, 0), beta);
                            let weight = aos[mu].coefficients[i]
                                * aos[nu].coefficients[j]
                                * norm_mu
                                * norm_nu;
                            d_v_nucleus += weight
                                * p_mu_nu
                                * nuclear_center_derivative(
                                    alpha,
                                    aos[mu].center,
                                    beta,
                                    aos[nu].center,
                                    own_position,
                                    own_charge,
                                    axis,
                                );
                        }
                    }
                }
            }
            value += d_v_nucleus;

            // 核排斥导数：Σ_B Z_A Z_B (R_A − R_B)_axis / |R_A − R_B|³
            let mut d_nuc = 0.0;
            for (other, (other_position, other_charge)) in nuclei.iter().enumerate() {
                if other == atom {
                    continue;
                }
                let delta = [
                    own_position[0] - other_position[0],
                    own_position[1] - other_position[1],
                    own_position[2] - other_position[2],
                ];
                let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                d_nuc += own_charge * other_charge * delta[axis] / (r * r * r);
            }
            value += d_nuc;

            // ③ 2e 项：½ Σ P P [d(μν|λσ) − ½ d(μλ|νσ)]
            let mut d_2e = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let p_mu_nu = density[(mu, nu)];
                    if p_mu_nu == 0.0 {
                        continue;
                    }
                    for lambda in 0..n {
                        for sigma in 0..n {
                            let p_lambda_sigma = density[(lambda, sigma)];
                            if p_lambda_sigma == 0.0 {
                                continue;
                            }
                            // 只有落在 atom 上的 AO 才会动
                            let mut d_coulomb = 0.0;
                            let mut d_exchange = 0.0;
                            for (i, &alpha) in aos[mu].exponents.iter().enumerate() {
                                let norm_mu = primitive_norm((0, 0, 0), alpha);
                                for (j, &beta) in aos[nu].exponents.iter().enumerate() {
                                    let norm_nu = primitive_norm((0, 0, 0), beta);
                                    for (k, &gamma) in aos[lambda].exponents.iter().enumerate() {
                                        let norm_lambda = primitive_norm((0, 0, 0), gamma);
                                        for (l, &delta) in aos[sigma].exponents.iter().enumerate() {
                                            let norm_sigma = primitive_norm((0, 0, 0), delta);
                                            let weight = aos[mu].coefficients[i]
                                                * aos[nu].coefficients[j]
                                                * aos[lambda].coefficients[k]
                                                * aos[sigma].coefficients[l]
                                                * norm_mu
                                                * norm_nu
                                                * norm_lambda
                                                * norm_sigma;
                                            if aos[mu].center_index == atom {
                                                d_coulomb += weight
                                                    * d_ssss(
                                                        alpha,
                                                        aos[mu].center,
                                                        beta,
                                                        aos[nu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        axis,
                                                    );
                                                d_exchange += weight
                                                    * d_ssss(
                                                        alpha,
                                                        aos[mu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        beta,
                                                        aos[nu].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        axis,
                                                    );
                                            }
                                            if aos[nu].center_index == atom {
                                                d_coulomb += weight
                                                    * d_ssss(
                                                        beta,
                                                        aos[nu].center,
                                                        alpha,
                                                        aos[mu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        axis,
                                                    );
                                                d_exchange += weight
                                                    * d_ssss(
                                                        beta,
                                                        aos[nu].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        alpha,
                                                        aos[mu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        axis,
                                                    );
                                            }
                                            if aos[lambda].center_index == atom {
                                                d_coulomb += weight
                                                    * d_ssss(
                                                        gamma,
                                                        aos[lambda].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        alpha,
                                                        aos[mu].center,
                                                        beta,
                                                        aos[nu].center,
                                                        axis,
                                                    );
                                                d_exchange += weight
                                                    * d_ssss(
                                                        alpha,
                                                        aos[mu].center,
                                                        beta,
                                                        aos[nu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        delta,
                                                        aos[sigma].center,
                                                        axis,
                                                    );
                                            }
                                            if aos[sigma].center_index == atom {
                                                d_coulomb += weight
                                                    * d_ssss(
                                                        delta,
                                                        aos[sigma].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        alpha,
                                                        aos[mu].center,
                                                        beta,
                                                        aos[nu].center,
                                                        axis,
                                                    );
                                                d_exchange += weight
                                                    * d_ssss(
                                                        delta,
                                                        aos[sigma].center,
                                                        beta,
                                                        aos[nu].center,
                                                        alpha,
                                                        aos[mu].center,
                                                        gamma,
                                                        aos[lambda].center,
                                                        axis,
                                                    );
                                            }
                                        }
                                    }
                                }
                            }
                            d_2e += 0.5 * p_mu_nu * p_lambda_sigma * (d_coulomb - 0.5 * d_exchange);
                        }
                    }
                }
            }
            value += d_2e;
            gradient[atom][axis] = value;
        }
    }
    Ok(gradient)
}
