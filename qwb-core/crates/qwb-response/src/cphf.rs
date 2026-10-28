//! CPHF：闭壳层 RHF 的静态线性响应（迭代、矩阵自由）。
//!
//! # 方程的来源（完整推导）
//!
//! 闭壳层 RHF 能量（空间轨道，`i,j` 占据、`a,b` 虚）：
//!
//! ```text
//! E = 2Σ_i h_ii + Σ_ij [2(ii|jj) − (ij|ji)]
//! ```
//!
//! 一个均匀电场 `F` 给体系加上单电子扰动 `V = −F·D`，其中
//! `D = Σ_N Z_N R_N − Σ_e r_e`；对**电子**哈密顿量来说矩阵元是
//! `v_{ai} = +F·⟨a|r|i⟩`（核的部分是常数，对二阶导没有贡献）。于是
//! `v = F·d`，`d^{ai} = ⟨a|r|i⟩`。
//!
//! 一阶轨道响应：`C_i⁽¹⁾ = Σ_a U_ai C_a`（`U` 反对称）。一阶密度
//! `P⁽¹⁾_{ai} = P⁽¹⁾_{ia} = 2U_ai`（其余块为零），一阶 Fock
//!
//! ```text
//! F⁽¹⁾_{ai} = v_{ai} + Σ_{rs} P⁽¹⁾_{rs} [ (ai|rs) − ½(ar|is) ]
//! ```
//!
//! 代入 `P⁽¹⁾` 只有占据-虚块非零、且 `(pq|rs)=(pq|sr)`：
//!
//! ```text
//! Coulomb 项：Σ_bj 2U_bj[(ai|bj) + (ai|jb)] = 4Σ_bj U_bj (ai|bj)
//! 交换项  ：−½ Σ_bj 2U_bj[(ab|ij) + (aj|ib)] = −Σ_bj U_bj[(ab|ij) + (aj|ib)]
//! ```
//!
//! 微扰后的 Fock 在**微扰基**里必须仍然对角，展开到一阶给出
//! `(ε_a − ε_i)U_ai + F⁽¹⁾_{ai} = 0`，即
//!
//! ```text
//! (A + B)_{ai,bj} U_bj = −v_{ai},
//! (A + B)_{ai,bj} = (ε_a − ε_i) δ_ij δ_ab + 4(ai|bj) − (ab|ij) − (aj|ib)
//! ```
//!
//! # 偶极响应与极化率
//!
//! `dμ_i/dF_j = −tr(P⁽¹⁾ d_i) = −4Σ_{ai} U^{F_j}_{ai} d_i^{ai}`，再用
//! `U^{F} = −(A+B)⁻¹ (F d)` 得
//!
//! ```text
//! α_ij = 4 d_iᵀ (A+B)⁻¹ d_j
//! ```
//!
//! 这个形式是 `(A+B)⁻¹`（SPD）上的二次型，所以主对角必然为正 —— 可以当自检，
//! 也可以解释为什么"极化率算出来是负的"一定是符号/因子写错了。
//!
//! # 求解：矩阵自由的预条件共轭梯度
//!
//! `(A+B)` 只以算子形式出现（[`orbital_hessian_apply`]），不稠密化。预条件取
//! Jacobi（对角 = `ε_a − ε_i`，即"未耦合"的近似），对分子体系足够：水/def2-SVP
//! 的 (A+B) 条件数在 1e3 量级，几十步到 1e-10。

use qwb_engine_api::EngineError;
use qwb_linalg::Vector;

use crate::mo_integrals::{CphfIntegrals, pair_index};

/// CPHF 的迭代选项。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CphfOptions {
    /// 最大迭代步数。
    pub max_iter: u32,
    /// 残差 2-范数的收敛阈值（`||r||₂ < tolerance`）。
    pub tolerance: f64,
}

impl Default for CphfOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tolerance: 1.0e-8,
        }
    }
}

/// CPHF 的解与诊断。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CphfResult {
    /// 解向量（每个 RHS 一个），按 `p = i·n_vir + a` 展平。
    pub solutions: Vec<Vec<f64>>,
    /// 每个 RHS 的实际迭代步数。
    pub iterations: Vec<u32>,
    /// 每个 RHS 的最终残差 2-范数。
    pub residuals: Vec<f64>,
    /// 是否全部收敛。
    pub converged: bool,
}

/// `(A+B)` 作用在一个展平向量上（矩阵自由）。
///
/// `x[p]` 里 `p = pair_index(p) = i·n_vir + a`；`orbital_energies` 是升序 MO 能。
///
/// 用到的两块 MO 积分与 `(aj|ib) = T1[(j,a)][(i,b)]` 的关系见
/// [`crate::mo_integrals`]。
#[must_use]
pub fn orbital_hessian_apply(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let n_occ = integrals.n_occupied;
    let n_vir = integrals.n_virtual;
    let n_pairs = n_occ * n_vir;
    let mut out = vec![0.0_f64; n_pairs];
    for i in 0..n_occ {
        for a in 0..n_vir {
            let p = pair_index(n_vir, i, a);
            let mut sum = 0.0;
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let q = pair_index(n_vir, j, b);
                    let xq = x[q];
                    if xq == 0.0 {
                        continue;
                    }
                    // Coulomb：4(ai|bj)
                    let coulomb = 4.0 * integrals.t1(p, q);
                    // 交换 1：(ab|ij)  → T2[[i,a]][[j,b]]
                    let exch1 = integrals.t2(p, q);
                    // 交换 2：(aj|ib) = T1[[j,a]][[i,b]]
                    let exch2 = integrals.t1(pair_index(n_vir, j, a), pair_index(n_vir, i, b));
                    sum += xq * (coulomb - exch1 - exch2);
                }
            }
            out[p] = (orbital_energies[n_occ + a] - orbital_energies[i]) * x[p] + sum;
        }
    }
    out
}

/// 解 `(A+B) X = rhs`（多个右端，Jacobi 预条件共轭梯度）。
///
/// # Errors
///
/// 维度不匹配、迭代未收敛、出现非有限值 → [`EngineError::Internal`] /
/// [`EngineError::ScfNotConverged`]。
pub fn solve_cphf(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    right_hand_sides: &[Vec<f64>],
    options: CphfOptions,
) -> Result<CphfResult, EngineError> {
    let n_pairs = integrals.n_occupied * integrals.n_virtual;
    if orbital_energies.len() < integrals.n_functions {
        return Err(EngineError::Internal(format!(
            "CPHF: only {} orbital energies for {} functions",
            orbital_energies.len(),
            integrals.n_functions
        )));
    }
    if right_hand_sides.is_empty() {
        return Err(EngineError::InvalidInput(
            "CPHF: at least one right-hand side is required".into(),
        ));
    }
    // Jacobi 预条件：对角元 = ε_a − ε_i。
    let mut diagonal = vec![0.0_f64; n_pairs];
    for i in 0..integrals.n_occupied {
        for a in 0..integrals.n_virtual {
            let p = pair_index(integrals.n_virtual, i, a);
            diagonal[p] = orbital_energies[integrals.n_occupied + a] - orbital_energies[i];
            if diagonal[p] <= 0.0 {
                return Err(EngineError::Internal(format!(
                    "CPHF: 非正的能量差 ε_a − ε_i = {} (i={i}, a={a})；占据/虚轨道分界有误？",
                    diagonal[p]
                )));
            }
        }
    }

    let mut solutions = Vec::with_capacity(right_hand_sides.len());
    let mut iterations = Vec::with_capacity(right_hand_sides.len());
    let mut residuals = Vec::with_capacity(right_hand_sides.len());
    let mut converged = true;
    for rhs in right_hand_sides {
        if rhs.len() != n_pairs {
            return Err(EngineError::Internal(format!(
                "CPHF: 右端长度 {} 与 (n_occ·n_vir) = {n_pairs} 不符",
                rhs.len()
            )));
        }
        // x = 0 起，r = rhs，z = M⁻¹ r，p = z
        let mut x = vec![0.0_f64; n_pairs];
        let mut r = rhs.clone();
        let rhs_norm = dot(&r, &r).sqrt();
        if rhs_norm == 0.0 {
            solutions.push(x);
            iterations.push(0);
            residuals.push(0.0);
            continue;
        }
        let mut steps = 0;
        let mut residual_norm = rhs_norm;
        if rhs_norm < options.tolerance {
            solutions.push(x);
            iterations.push(0);
            residuals.push(rhs_norm);
            continue;
        }
        let mut z: Vec<f64> = (0..n_pairs).map(|p| r[p] / diagonal[p]).collect();
        let mut direction = z.clone();
        let mut rz = dot(&r, &z);
        for step in 0..options.max_iter {
            steps = step + 1;
            let ap = apply_as_vector(integrals, orbital_energies, &direction);
            let curvature = dot(&direction, &ap);
            if curvature <= 0.0 || !curvature.is_finite() {
                return Err(EngineError::Internal(format!(
                    "CPHF: 方向曲率 pᵀ(A+B)p = {curvature} 非正；(A+B) 应当正定"
                )));
            }
            let alpha = rz / curvature;
            for p in 0..n_pairs {
                x[p] += alpha * direction[p];
                r[p] -= alpha * ap[p];
            }
            residual_norm = dot(&r, &r).sqrt();
            if residual_norm < options.tolerance {
                break;
            }
            z = (0..n_pairs).map(|p| r[p] / diagonal[p]).collect();
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for p in 0..n_pairs {
                direction[p] = z[p] + beta * direction[p];
            }
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(EngineError::Internal("CPHF: 解出现非有限值".into()));
        }
        if residual_norm >= options.tolerance {
            converged = false;
        }
        solutions.push(x);
        iterations.push(steps);
        residuals.push(residual_norm);
    }

    Ok(CphfResult {
        solutions,
        iterations,
        residuals,
        converged,
    })
}

fn apply_as_vector(integrals: &CphfIntegrals, orbital_energies: &[f64], x: &[f64]) -> Vec<f64> {
    orbital_hessian_apply(integrals, orbital_energies, x)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 把一个 `Vector` 换成 `Vec<f64>`（`qwb-linalg` 的 `Vector` 只在求解器接口上出现）。
#[must_use]
pub fn vector_to_vec(v: &Vector) -> Vec<f64> {
    (0..v.len()).map(|i| v.get(i)).collect()
}
