//! AO → MO 的两电子块变换与偶极矩阵元（CPHF 的输入）。
//!
//! 只变换响应方程真正用到的两块，不做完整的 `n⁴` MO 张量（苯/def2-SVP 的 `n⁴`
//! 是 1.7e8 个双精度 = 1.3 GB，那样 RS-02 在苯上就跑不动了）。
//!
//! # 两块是什么
//!
//! 用 `p = (i, a)`（占据在外层）展平占据-虚对，则
//!
//! ```text
//! T1[p][q] = (a_p i_p | a_q i_q)      ← Coulomb 与 (aj|ib) 项都用它
//! T2[p][q] = (a_p b_q | i_p j_q)      ← 交换项 (ab|ij) 用它
//! ```
//!
//! 其中 `(aj|ib)` 可以从 `T1` 直接取：`(aj|ib) = T1[(j,a)][(i,b)]`
//! （用 `(pq|rs) = (rs|pq)` 交换两侧电子）。
//!
//! # 代价（四步半变换只做两步）
//!
//! ```text
//! T1: 先对 λσ 收缩出 X[μ][ν][bj]（n²·n_occ·n_vir 个数），再对 μν 收缩
//! T2: 先对 λσ 收缩出 Y[μ][ν][ij]（n²·n_occ²），再对 μν 收缩
//! ```
//!
//! 两次都是 `O(n⁴·n_occ·n_vir)`，比"整张 n⁴ MO 张量"省内存得多。

use qwb_engine_api::EngineError;
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::Matrix;

/// `(ai)` 对的展平：`p = i * n_virtual + a`。
#[must_use]
pub fn pair_index(n_virtual: usize, i: usize, a: usize) -> usize {
    i * n_virtual + a
}

/// CPHF 需要的两块 MO 两电子积分 + 偶极矩阵元。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CphfIntegrals {
    /// 占据轨道数。
    pub n_occupied: usize,
    /// 虚轨道数。
    pub n_virtual: usize,
    /// 基函数数（`n_occupied + n_virtual`）。
    pub n_functions: usize,
    /// `T1[p][q] = (a_p i_p | a_q i_q)`，`p` 行优先展平。
    pub t1: Vec<f64>,
    /// `T2[p][q] = (a_p b_q | i_p j_q)`，`p` 行优先展平。
    pub t2: Vec<f64>,
    /// 偶极矩阵元 `d[axis][p] = ⟨a_p|r_axis|i_p⟩`（原子单位）。
    pub dipole: [Vec<f64>; 3],
}

impl CphfIntegrals {
    /// `T1[p][q]`。
    #[must_use]
    pub fn t1(&self, p: usize, q: usize) -> f64 {
        self.t1[p * self.n_occupied * self.n_virtual + q]
    }

    /// `T2[p][q]`。
    #[must_use]
    pub fn t2(&self, p: usize, q: usize) -> f64 {
        self.t2[p * self.n_occupied * self.n_virtual + q]
    }
}

/// 由 AO 2e 张量、MO 系数、偶极积分构造 [`CphfIntegrals`]。
///
/// # Errors
///
/// 维度不匹配 → [`EngineError::Internal`]；出现非有限值 → [`EngineError::Internal`]。
pub fn build_cphf_integrals(
    tensor: &TwoElectronIntegrals,
    coefficients: &Matrix,
    dipole_ao: &[Matrix; 3],
    n_occupied: usize,
) -> Result<CphfIntegrals, EngineError> {
    let n = tensor.n_functions;
    if coefficients.nrows() != n || coefficients.ncols() != n {
        return Err(EngineError::Internal(format!(
            "CPHF: MO coefficient matrix is {}x{}, expected {n}x{n}",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if n_occupied == 0 || n_occupied >= n {
        return Err(EngineError::InvalidInput(format!(
            "CPHF needs at least one occupied and one virtual orbital (n_occ = {n_occupied}, n = {n})"
        )));
    }
    let n_vir = n - n_occupied;
    let n_pairs = n_occupied * n_vir;
    let c = |mu: usize, p: usize| coefficients[(mu, p)];

    // ── T1: 先对 λσ 收缩（得到 X[μ][ν][bj]），再对 μν 收缩 ──────────
    let mut x = vec![0.0_f64; n * n * n_pairs];
    for mu in 0..n {
        for nu in 0..n {
            for j in 0..n_occupied {
                for b in 0..n_vir {
                    let mut sum = 0.0;
                    for lambda in 0..n {
                        let cl = c(lambda, n_occupied + b);
                        if cl == 0.0 {
                            continue;
                        }
                        for sigma in 0..n {
                            sum += cl * c(sigma, j) * tensor.get(mu, nu, lambda, sigma);
                        }
                    }
                    x[(mu * n + nu) * n_pairs + pair_index(n_vir, j, b)] = sum;
                }
            }
        }
    }
    let mut t1 = vec![0.0_f64; n_pairs * n_pairs];
    for i in 0..n_occupied {
        for a in 0..n_vir {
            let p = pair_index(n_vir, i, a);
            for q in 0..n_pairs {
                let mut sum = 0.0;
                for mu in 0..n {
                    let ca = c(mu, n_occupied + a);
                    if ca == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        sum += ca * c(nu, i) * x[(mu * n + nu) * n_pairs + q];
                    }
                }
                t1[p * n_pairs + q] = sum;
            }
        }
    }
    drop(x);

    // ── T2: (a_p b_q | i_p j_q)，先对 λσ 收缩出 Y[μ][ν][ij] ─────────
    let n_occ_pairs = n_occupied * n_occupied;
    let mut y = vec![0.0_f64; n * n * n_occ_pairs];
    for mu in 0..n {
        for nu in 0..n {
            for i in 0..n_occupied {
                for j in 0..n_occupied {
                    let mut sum = 0.0;
                    for lambda in 0..n {
                        let cl = c(lambda, i);
                        if cl == 0.0 {
                            continue;
                        }
                        for sigma in 0..n {
                            sum += cl * c(sigma, j) * tensor.get(mu, nu, lambda, sigma);
                        }
                    }
                    y[(mu * n + nu) * n_occ_pairs + i * n_occupied + j] = sum;
                }
            }
        }
    }
    let mut t2 = vec![0.0_f64; n_pairs * n_pairs];
    for i in 0..n_occupied {
        for a in 0..n_vir {
            let p = pair_index(n_vir, i, a);
            for j in 0..n_occupied {
                for b in 0..n_vir {
                    let q = pair_index(n_vir, j, b);
                    let mut sum = 0.0;
                    for mu in 0..n {
                        let ca = c(mu, n_occupied + a);
                        if ca == 0.0 {
                            continue;
                        }
                        for nu in 0..n {
                            sum += ca
                                * c(nu, n_occupied + b)
                                * y[(mu * n + nu) * n_occ_pairs + i * n_occupied + j];
                        }
                    }
                    t2[p * n_pairs + q] = sum;
                }
            }
        }
    }
    drop(y);

    // ── 偶极矩阵元 ⟨a|r|i⟩（电子坐标，正号）──────────────────────────
    let mut dipole = [
        vec![0.0_f64; n_pairs],
        vec![0.0_f64; n_pairs],
        vec![0.0_f64; n_pairs],
    ];
    for (axis, d_ao) in dipole_ao.iter().enumerate() {
        for i in 0..n_occupied {
            for a in 0..n_vir {
                let mut sum = 0.0;
                for mu in 0..n {
                    let ca = c(mu, n_occupied + a);
                    if ca == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        sum += ca * c(nu, i) * d_ao[(mu, nu)];
                    }
                }
                dipole[axis][pair_index(n_vir, i, a)] = sum;
            }
        }
    }
    for value in t1.iter().chain(t2.iter()).chain(dipole.iter().flatten()) {
        if !value.is_finite() {
            return Err(EngineError::Internal(
                "CPHF: MO-transformed integral is not finite".into(),
            ));
        }
    }

    Ok(CphfIntegrals {
        n_occupied,
        n_virtual: n_vir,
        n_functions: n,
        t1,
        t2,
        dipole,
    })
}
