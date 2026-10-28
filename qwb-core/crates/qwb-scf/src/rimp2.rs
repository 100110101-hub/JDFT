//! M3 / SC-05：**RI-MP2**（闭壳层，自旋匹配，无冻结核）。
//!
//! # 公式
//!
//! 与 canonical MP2 同一个能量表达式，只把 4 中心积分换成密度拟合：
//!
//! ```text
//! (ia|jb) ≈ Σ_{P,Q} (ia|P) [V⁻¹]_{PQ} (Q|jb),   V_{PQ} = (P|Q)
//! E2 = Σ_{i≤j} w_ij Σ_{ab} (ia|jb)[2(ia|jb) − (ib|ja)] / (ε_i+ε_j−ε_a−ε_b)
//! ```
//!
//! 实现上先做**白化**：`B_P,ia = Σ_Q (V^{-1/2})_{PQ} (Q|ia)`，于是
//! `(ia|jb) = Σ_P B_P,ia B_P,jb`，(ia|jb) 与 (ib|ja) 在固定 (i,j) 下互为转置
//! （与 canonical 版的结构完全一样，见 [`crate::mp2::mp2_correlation`]）。
//!
//! # 代价
//!
//! * 3 中心变换：`n_aux` 次 `n³`（`Cᵀ (μν|P) C`）；
//! * 白化：`n_aux² · n_occ · n_vir`；
//! * 收缩：`n_occ² · n_vir² · n_aux`（canonical 版是 `n_occ² · n_vir² · n_aux²` 量级的
//!   4 指标求和，所以 RI 的收益随 `n_aux` 变大而变小 —— 优点在于**不需要 n⁴ 的 4 中心张量**）。
//!
//! # 拟合误差必须落盘
//!
//! RI 把 4 中心积分换成拟合值，所以相关能有系统偏差。K9 的
//! `CorrelationResult.ri_error_hartree` 就是给这个偏差的：引擎会**同时**算
//! canonical 的 `E2`（用同一份 4 中心张量）并落差值，所以 `kind: ri-mp2` 的
//! 一次作业比纯 RI 慢 —— 这是"近似必须留痕"（K10 第 ③ 条）的价格，不是缺陷。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::EngineError;
use qwb_integrals::ri::{ThreeCenterIntegrals, inverse_sqrt};
use qwb_linalg::{Matrix, matmul};

/// `V^{-1/2}` 的截断容差：特征值小于 `1e-10·λ_max` 的方向视为线性相关。
pub const V_TOLERANCE: f64 = 1.0e-10;

/// RI-MP2 相关能的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RiMp2Result {
    /// RI 的二阶相关能（Hartree，负值）。
    pub e2_hartree: f64,
    /// 占据轨道数。
    pub n_occupied: usize,
    /// 虚轨道数。
    pub n_virtual: usize,
    /// 辅助函数个数（含被 `V^{-1/2}` 截断掉的线性相关方向）。
    pub n_auxiliary: usize,
    /// `V` 的秩（未被截断的方向数）。小于 `n_auxiliary` 说明辅助基里线性相关。
    pub v_rank: usize,
    /// 白化后 `(ia|jb) = Σ_P B_P,ia B_P,jb` 的最大单对贡献（诊断用）。
    pub dominant_pair_energy: f64,
}

/// RI-MP2 相关能。
///
/// `three` 是 3 中心 `(ij|P)`，`v` 是 2 中心库仑 `(P|Q)`。
///
/// # Errors
///
/// 轨道系数/能量维度不匹配、`(P|Q)` 不正定、或出现非有限值 →
/// [`EngineError::Internal`] / [`EngineError::InvalidInput`]。
pub fn ri_mp2_correlation(
    three: &ThreeCenterIntegrals,
    v: &Matrix,
    coefficients: &Matrix,
    orbital_energies: &[f64],
    n_occupied: usize,
) -> Result<RiMp2Result, EngineError> {
    let n = three.n_orbital;
    let n_aux = three.n_auxiliary;
    if coefficients.nrows() != n || coefficients.ncols() != n {
        return Err(EngineError::Internal(format!(
            "RI-MP2: MO coefficient matrix is {}x{}, expected {n}x{n}",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if orbital_energies.len() < n {
        return Err(EngineError::Internal(format!(
            "RI-MP2: only {} orbital energies for {n} functions",
            orbital_energies.len()
        )));
    }
    if n_occupied == 0 || n_occupied >= n {
        return Err(EngineError::InvalidInput(format!(
            "RI-MP2 needs at least one occupied and one virtual orbital (n_occ = {n_occupied}, n = {n})"
        )));
    }
    if v.nrows() != n_aux || v.ncols() != n_aux {
        return Err(EngineError::Internal(format!(
            "RI-MP2: (P|Q) is {}x{}, expected {n_aux}x{n_aux}",
            v.nrows(),
            v.ncols()
        )));
    }
    let n_vir = n - n_occupied;
    let w = inverse_sqrt(v, V_TOLERANCE)?;

    // ── 3 中心 MO 变换：(ia|P) = Σ_μν C_μi C_νa (μν|P) ─────────────
    // 每个辅助函数一次：先 Y = Cᵀ M，再 X = Y C。
    let mut a = vec![0.0_f64; n_aux * n_occupied * n_vir];
    let mut mo_3c = vec![0.0_f64; n_aux * n_occupied * n_vir];
    for p in 0..n_aux {
        let block = three.matrix_for(p);
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = block[i * n + j];
            }
        }
        let ct = coefficients.transpose();
        let y = matmul(&ct, &m)
            .map_err(|err| EngineError::Internal(format!("RI-MP2: Cᵀ M 失败：{err}")))?;
        let x = matmul(&y, coefficients)
            .map_err(|err| EngineError::Internal(format!("RI-MP2: (Cᵀ M) C 失败：{err}")))?;
        for i in 0..n_occupied {
            for virt in 0..n_vir {
                mo_3c[(p * n_occupied + i) * n_vir + virt] = x[(i, n_occupied + virt)];
            }
        }
    }
    // ── 白化：B_P,ia = Σ_Q W[P][Q] (Q|ia) ──────────────────────────
    for p in 0..n_aux {
        for i in 0..n_occupied {
            for virt in 0..n_vir {
                let mut sum = 0.0;
                for q in 0..n_aux {
                    sum += w[(p, q)] * mo_3c[(q * n_occupied + i) * n_vir + virt];
                }
                a[(p * n_occupied + i) * n_vir + virt] = sum;
            }
        }
    }

    // ── 对 (i,j) 收缩 ─────────────────────────────────────────────
    let mut e2 = 0.0_f64;
    let mut dominant = 0.0_f64;
    let mut matrix = vec![0.0_f64; n_vir * n_vir];
    for i in 0..n_occupied {
        for j in i..n_occupied {
            for va in 0..n_vir {
                for vb in 0..n_vir {
                    let mut sum = 0.0;
                    for p in 0..n_aux {
                        sum += a[(p * n_occupied + i) * n_vir + va]
                            * a[(p * n_occupied + j) * n_vir + vb];
                    }
                    matrix[va * n_vir + vb] = sum;
                }
            }
            let weight = if i == j { 1.0 } else { 2.0 };
            let mut pair = 0.0;
            for va in 0..n_vir {
                let energy_a = orbital_energies[n_occupied + va];
                for vb in 0..n_vir {
                    let iab = matrix[va * n_vir + vb];
                    let iba = matrix[vb * n_vir + va];
                    let denominator = orbital_energies[i] + orbital_energies[j]
                        - energy_a
                        - orbital_energies[n_occupied + vb];
                    pair += iab * (2.0 * iab - iba) / denominator;
                }
            }
            if !pair.is_finite() {
                return Err(EngineError::Internal(format!(
                    "RI-MP2: non-finite pair energy for occupied orbitals {i},{j}"
                )));
            }
            let contribution = weight * pair;
            dominant = dominant.max(contribution.abs());
            e2 += contribution;
        }
    }
    if !e2.is_finite() {
        return Err(EngineError::Internal(
            "RI-MP2: correlation energy is not finite".into(),
        ));
    }
    // V 的秩：W 的列范数为 0 的方向就是被截断掉的。
    let mut v_rank = 0;
    for q in 0..n_aux {
        let mut norm = 0.0;
        for p in 0..n_aux {
            norm += w[(p, q)] * w[(p, q)];
        }
        if norm > 0.0 {
            v_rank += 1;
        }
    }

    Ok(RiMp2Result {
        e2_hartree: e2,
        n_occupied,
        n_virtual: n_vir,
        n_auxiliary: n_aux,
        v_rank,
        dominant_pair_energy: dominant,
    })
}
