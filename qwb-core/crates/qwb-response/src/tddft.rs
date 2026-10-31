//! ST-01：**完整 TDDFT（RPA / TDHF）**——把 TDA 丢掉的 B 块（去激发耦合）加回来。
//!
//! # 方程
//!
//! 闭壳层、单重态、空间轨道，`p = (i, a)`（占据在外层）：
//!
//! ```text
//! A_{p q} = δ_ij δ_ab (ε_a − ε_i) + 2 (ia|jb) − (ij|ab)
//! B_{p q} = 2 (ia|jb) − (ib|ja)
//!
//! [ A  B ] [ X ]      [ 1  0 ] [ X ]
//! [ B  A ] [ Y ]  = ω [ 0 -1 ] [ Y ]
//! ```
//!
//! TDA 就是扔掉 `B` 与 `Y`（`A X = ω X`）。`B` 不是"另一套积分"：用
//! `T1[p][q] = (a_p i_p | a_q i_q)` 与八重对称 `(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)` 就有
//!
//! ```text
//! B_{p q} = 2 T1[p][q] − T1[(j,a)][(i,b)]        （第二项把两侧的占/虚角色对调）
//! ```
//!
//! 所以本模块只多一次索引置换，不需要新的积分路径 —— "耦合项不是新物理，只是新组合"。
//!
//! # 为什么不能直接对 `A` 做本征分解再加 B
//!
//! 这个方程**不是**厄米的（度量是 `diag(1, −1)`），`A` 与 `B` 也不可交换。标准做法是靠
//! `S = A − B` 的**正定性**（RPA 稳定性条件）：
//!
//! ```text
//! Z = X + Y,  W = X − Y
//! (A+B) Z = ω W,  (A−B) W = ω Z
//! ⇒  S^{1/2}(A+B)S^{1/2} U = ω² U,   Z = S^{1/2}U/ω,  W = S^{-1/2}U,  |U|² = ω
//! ```
//!
//! 于是问题变成对一个**对称**矩阵做本征分解，`ω = √λ`。`A − B` 非正定就没有稳定的 RPA 解
//! （基态不是极小），这里明确报错而不是取绝对值硬算 —— TDA 仍然可用。
//!
//! # 振子强度
//!
//! 与 TDA 同一个公式，只是把 `X` 换成 `Z = X + Y`：
//! `μ_n = √2 Σ_p Z_p ⟨a_p|r|i_p⟩`、`f_n = (2/3) ω_n |μ_n|²`。归一化 `Z·W = 1` 与 TDA 的
//! `ΣX² = 1` 是同一个约定（TDA 里 `W = Z`，所以 `X·X = 1`）。代码对每个态真的算一遍
//! `Z·W` 并要求它等于 1（1e-6）—— 这是本模块唯一不依赖外部参考的自检。
use qwb_engine_api::EngineError;
use qwb_linalg::{Matrix, Vector, matmul, symmetric_eigen};

use crate::mo_integrals::{CphfIntegrals, pair_index};
use crate::tda::Excitation;

/// 完整 TDDFT（RPA）的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TddftResult {
    /// 按能量升序的激发态（只含能量为正的）。
    pub excitations: Vec<Excitation>,
    /// 振子强度之和（全空间）。
    pub oscillator_sum: f64,
    /// 参与展开的 (占据, 虚) 对数（= 本征问题的维数）。
    pub dimension: usize,
    /// `A − B` 的最小本征值（RPA 稳定性；必须为正）。
    pub lowest_ab_eigenvalue: f64,
    /// 出现的最小 `ω²`（诊断用：若接近 0 说明参考态不稳）。
    pub lowest_omega_squared: f64,
}

/// `S = A − B` 的 `S^{1/2}`、`S^{-1/2}` 与最小本征值。
fn matrix_powers(s: &Matrix) -> Result<(Matrix, Matrix, f64), EngineError> {
    let (values, vectors) = symmetric_eigen(s)
        .map_err(|err| EngineError::Internal(format!("TDDFT: (A−B) 本征分解失败：{err}")))?;
    let n = s.nrows();
    let lowest = values.get(0);
    if !(lowest.is_finite() && lowest > 0.0) {
        return Err(EngineError::Internal(format!(
            "TDDFT: (A−B) 不是正定（最小本征值 {lowest:.3e}）；RPA 参考态不稳定，\
             不做本征分解（TDA 仍然可用）"
        )));
    }
    let mut root = Matrix::zeros(n, n);
    let mut inverse_root = Matrix::zeros(n, n);
    for column in 0..n {
        let value = values.get(column);
        let r = value.sqrt();
        for row in 0..n {
            root.set(row, column, vectors[(row, column)] * r);
            inverse_root.set(row, column, vectors[(row, column)] / r);
        }
    }
    let transposed = vectors.transpose();
    let root = matmul(&root, &transposed)
        .map_err(|err| EngineError::Internal(format!("TDDFT: S^(1/2) 组装失败：{err}")))?;
    let inverse_root = matmul(&inverse_root, &transposed)
        .map_err(|err| EngineError::Internal(format!("TDDFT: S^(-1/2) 组装失败：{err}")))?;
    Ok((root, inverse_root, lowest))
}

fn column(matrix: &Matrix, index: usize) -> Vector {
    let mut out = Vector::zeros(matrix.nrows());
    for row in 0..matrix.nrows() {
        out.set(row, matrix[(row, index)]);
    }
    out
}

fn as_column(vector: &Vector) -> Matrix {
    let mut out = Matrix::zeros(vector.len(), 1);
    for row in 0..vector.len() {
        out.set(row, 0, vector.get(row));
    }
    out
}

/// 单重态完整 TDDFT（HF 参考下即 TDHF / RPA）。
///
/// `n_states` 是返回的最低激发态个数；本征问题在全 `n_occ·n_vir` 空间里解。
///
/// # Errors
///
/// 维度不匹配、`A − B` 非正定、本征分解失败、`Z·W ≠ 1`、出现非有限值 →
/// [`EngineError::Internal`]；没有 (占, 虚) 对 → [`EngineError::InvalidInput`]。
pub fn tddft_singlet(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    n_states: usize,
) -> Result<TddftResult, EngineError> {
    let n_occ = integrals.n_occupied;
    let n_vir = integrals.n_virtual;
    let n_pairs = n_occ * n_vir;
    if orbital_energies.len() < integrals.n_functions {
        return Err(EngineError::Internal(format!(
            "TDDFT: only {} orbital energies for {} functions",
            orbital_energies.len(),
            integrals.n_functions
        )));
    }
    if n_pairs == 0 {
        return Err(EngineError::InvalidInput(
            "TDDFT: 没有 (占据, 虚) 对可用于展开".into(),
        ));
    }

    // A 与 B：两条都是 T1/T2 的组合，B 的第二项是 T1 的索引置换。
    let mut a = Matrix::zeros(n_pairs, n_pairs);
    let mut b = Matrix::zeros(n_pairs, n_pairs);
    for i in 0..n_occ {
        for va in 0..n_vir {
            let p = pair_index(n_vir, i, va);
            for j in 0..n_occ {
                for vb in 0..n_vir {
                    let q = pair_index(n_vir, j, vb);
                    let mut value = 2.0 * integrals.t1(p, q) - integrals.t2(p, q);
                    if p == q {
                        value += orbital_energies[n_occ + va] - orbital_energies[i];
                    }
                    a[(p, q)] = value;
                    // (ib|ja) = T1[(j,a)][(i,b)]
                    let swapped = integrals.t1(pair_index(n_vir, j, va), pair_index(n_vir, i, vb));
                    b[(p, q)] = 2.0 * integrals.t1(p, q) - swapped;
                }
            }
        }
    }

    let mut s = Matrix::zeros(n_pairs, n_pairs);
    let mut sum = Matrix::zeros(n_pairs, n_pairs);
    for row in 0..n_pairs {
        for column_index in 0..n_pairs {
            s.set(
                row,
                column_index,
                a[(row, column_index)] - b[(row, column_index)],
            );
            sum.set(
                row,
                column_index,
                a[(row, column_index)] + b[(row, column_index)],
            );
        }
    }
    let (root, inverse_root, lowest_ab) = matrix_powers(&s)?;

    let left = matmul(&root, &sum)
        .map_err(|err| EngineError::Internal(format!("TDDFT: (A+B)S^(1/2) 失败：{err}")))?;
    let mut m = matmul(&left, &root)
        .map_err(|err| EngineError::Internal(format!("TDDFT: S^(1/2)(A+B)S^(1/2) 失败：{err}")))?;
    for row in 0..n_pairs {
        for column_index in row + 1..n_pairs {
            let average = 0.5 * (m[(row, column_index)] + m[(column_index, row)]);
            m.set(row, column_index, average);
            m.set(column_index, row, average);
        }
    }
    let (squared, vectors) = symmetric_eigen(&m)
        .map_err(|err| EngineError::Internal(format!("TDDFT: ω² 本征分解失败：{err}")))?;

    let mut excitations = Vec::new();
    let mut oscillator_sum = 0.0;
    let mut lowest_omega_squared = f64::INFINITY;
    for state in 0..n_pairs {
        let lambda = squared.get(state);
        if !lambda.is_finite() {
            return Err(EngineError::Internal("TDDFT: ω² 出现非有限值".into()));
        }
        lowest_omega_squared = lowest_omega_squared.min(lambda);
        if lambda <= 0.0 {
            // 虚频/零频：RPA 在这个方向上没有实的激发能。跳过，但最小值仍如实汇报。
            continue;
        }
        let omega = lambda.sqrt();
        // |U'|² = ω ⇒ W = S^(-1/2)U'、Z = S^(1/2)U'/ω，于是 Z·W = |U'|²/ω = 1。
        let scale = omega.sqrt();
        let mut u = column(&vectors, state);
        for row in 0..u.len() {
            u.set(row, u.get(row) * scale);
        }
        let w = matmul(&inverse_root, &as_column(&u))
            .map_err(|err| EngineError::Internal(format!("TDDFT: W 计算失败：{err}")))?;
        let z = matmul(&root, &as_column(&u))
            .map_err(|err| EngineError::Internal(format!("TDDFT: Z 计算失败：{err}")))?;
        let mut overlap = 0.0;
        for row in 0..n_pairs {
            overlap += (z[(row, 0)] / omega) * w[(row, 0)];
        }
        if (overlap - 1.0).abs() > 1.0e-6 {
            return Err(EngineError::Internal(format!(
                "TDDFT: 归一化 Z·W = {overlap:.6e}（应为 1）—— S^(1/2) 与 S^(-1/2) 不一致"
            )));
        }
        let mut dipole = [0.0_f64; 3];
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            let mut total = 0.0;
            for p in 0..n_pairs {
                // Z 已按 Z·W = 1 定标；这里再除 ω 得到 Z_p。
                total += (z[(p, 0)] / omega) * integrals.dipole[axis][p];
            }
            dipole[axis] = std::f64::consts::SQRT_2 * total;
        }
        let norm2 = dipole.iter().map(|value| value * value).sum::<f64>();
        let oscillator = (2.0 / 3.0) * omega * norm2;
        oscillator_sum += oscillator;
        if omega > 1.0e-10 && excitations.len() < n_states {
            excitations.push(Excitation {
                energy_hartree: omega,
                transition_dipole_au: dipole,
                oscillator_strength: oscillator,
            });
        }
    }
    Ok(TddftResult {
        excitations,
        oscillator_sum,
        dimension: n_pairs,
        lowest_ab_eigenvalue: lowest_ab,
        lowest_omega_squared,
    })
}
