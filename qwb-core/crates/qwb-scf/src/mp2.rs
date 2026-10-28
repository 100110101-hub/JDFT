//! M3 / SC-04：**canonical MP2**（闭壳层，自旋匹配，无冻结核）。
//!
//! # 公式
//!
//! ```text
//! E2 = Σ_{ij}^{occ} Σ_{ab}^{vir} (ia|jb)·[2(ia|jb) − (ib|ja)] / (ε_i + ε_j − ε_a − ε_b)
//! ```
//!
//! `(pq|rs)` 是化学家记号 `∫∫φ_p(1)φ_q(1) r₁₂⁻¹ φ_r(2)φ_s(2)`（与
//! [`qwb_integrals::two_electron`] 的约定一致：`J_ij = Σ_kl (ij|kl) P_kl`）。
//! 指标 `i,j` 只跑占据、`a,b` 只跑虚轨道；`[2(ia|jb) − (ib|ja)]` 已经把两个
//! 自旋通道都算进去了，所以这是「空间轨道」上的闭壳层公式，**不要再乘 2**。
//!
//! # 变换顺序：`μ → i`、`ν → a`、`λ → j`、`σ → b`
//!
//! 想直接 4 重求和是 `n⁴·n_occ²·n_vir²`：水/def2-TZVP（n = 41）约 1e10，
//! 太慢。逐指标半变换是 `O(n⁵)`（≈1e8），中间张量只有 `n_vir·n²`。
//!
//! 关键的一步是**配对**：`(ia|jb)` 里电子 1 上的两个指标是 `(i,a)`、电子 2 上
//! 是 `(j,b)`，所以必须先 `μ→i`、`ν→a`，再 `λ→j`、`σ→b`。第一版按
//! `μ→i, ν→j, λ→b, σ→a` 走，得到的是 `(ij|ba)` —— 那是库仑型积分，和
//! `(ia|jb)` 完全不是一回事。实测代价：水/STO-3G 的相关能给成 −1.1641 Ha
//! （PySCF −0.0356，放大了 33 倍），H2/STO-3G 给成 −0.1762（−0.0131，13 倍）。
//! 相关能**全部**来自这些"小"积分，所以配对一错，误差是量级级的、不是百分比级的。
//!
//! # 对称性：只算 `i ≤ j`
//!
//! `(pq|rs) = (rs|pq)` 给出 `(jb|ia) = (ia|jb)` 与 `(ja|ib) = (ib|ja)`，于是
//! `f_{ji}[a][b] = f_{ij}[b][a]`，把 `(i,j)` 与 `(j,i)` 两项加起来正好是同一个
//! 求和的 2 倍。所以外层只跑 `i ≤ j`：对角取权重 1、非对角取 2；`(ib|ja)` 就是
//! 同一个 `f` 矩阵的转置元素。
//!
//! # 与 RI-MP2（SC-05）的关系
//!
//! RI-MP2 复用这里的 `(ia|jb)` 组装与分母，只把四中心积分换成三中心拟合；
//! 它自己的验收是「与 canonical 差 < 0.1 mHa」+ `ri_error_hartree` 落盘。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::EngineError;
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::Matrix;

/// MP2 相关能的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Mp2Result {
    /// 二阶相关能（Hartree，负值）。
    pub e2_hartree: f64,
    /// 占据轨道数（= 电子数/2）。
    pub n_occupied: usize,
    /// 虚轨道数。
    pub n_virtual: usize,
    /// 最大的单对贡献（诊断用：哪一对占据轨道主导相关能）。
    pub dominant_pair_energy: f64,
}

/// 闭壳层 canonical MP2 相关能。
///
/// # Errors
///
/// - 轨道系数/能量的维度与基组、占据数不匹配；
/// - 出现非有限值。
pub fn mp2_correlation(
    tensor: &TwoElectronIntegrals,
    coefficients: &Matrix,
    orbital_energies: &[f64],
    n_occupied: usize,
) -> Result<Mp2Result, EngineError> {
    let n = tensor.n_functions;
    if coefficients.nrows() != n || coefficients.ncols() != n {
        return Err(EngineError::Internal(format!(
            "MP2: MO coefficient matrix is {}x{}, expected {n}x{n}",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if orbital_energies.len() < n {
        return Err(EngineError::Internal(format!(
            "MP2: only {} orbital energies for {n} functions",
            orbital_energies.len()
        )));
    }
    if n_occupied == 0 || n_occupied >= n {
        return Err(EngineError::InvalidInput(format!(
            "MP2 needs at least one occupied and one virtual orbital (n_occ = {n_occupied}, n = {n})"
        )));
    }

    let n_vir = n - n_occupied;
    let coefficient = |mu: usize, p: usize| coefficients[(mu, p)];

    // t1[ν][λ][σ] = Σ_μ C_μi (μν|λσ)          —— 与 i 有关
    // t2[a][λ][σ] = Σ_ν C_ν,a t1[ν][λ][σ]      —— 与 i 有关，(ia|λσ)
    let mut t1 = vec![0.0_f64; n * n * n];
    let mut t2 = vec![0.0_f64; n_vir * n * n];
    // t3[a][σ] = Σ_λ C_λj t2[a][λ][σ]          —— 与 (i,j) 有关，(ia|jσ)
    let mut t3 = vec![0.0_f64; n_vir * n];
    // f[a][b] = Σ_σ C_σ,b t3[a][σ]             —— (ia|jb)
    let mut f = vec![0.0_f64; n_vir * n_vir];

    let mut e2 = 0.0_f64;
    let mut dominant = 0.0_f64;
    for i in 0..n_occupied {
        // ── 第一步：μ → i ──────────────────────────────────────
        for nu in 0..n {
            for lambda in 0..n {
                for sigma in 0..n {
                    let mut sum = 0.0;
                    for mu in 0..n {
                        sum += coefficient(mu, i) * tensor.get(mu, nu, lambda, sigma);
                    }
                    t1[(nu * n + lambda) * n + sigma] = sum;
                }
            }
        }
        // ── 第二步：ν → a（虚轨道），得到 (ia|λσ) ───────────────
        for a in 0..n_vir {
            let orbital = n_occupied + a;
            for lambda in 0..n {
                for sigma in 0..n {
                    let mut sum = 0.0;
                    for nu in 0..n {
                        sum += coefficient(nu, orbital) * t1[(nu * n + lambda) * n + sigma];
                    }
                    t2[(a * n + lambda) * n + sigma] = sum;
                }
            }
        }
        for j in i..n_occupied {
            // ── 第三步：λ → j（占据），得到 (ia|jσ) ─────────────
            for a in 0..n_vir {
                for sigma in 0..n {
                    let mut sum = 0.0;
                    for lambda in 0..n {
                        sum += coefficient(lambda, j) * t2[(a * n + lambda) * n + sigma];
                    }
                    t3[a * n + sigma] = sum;
                }
            }
            // ── 第四步：σ → b（虚轨道），得到 (ia|jb) ───────────
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let orbital_b = n_occupied + b;
                    let mut sum = 0.0;
                    for sigma in 0..n {
                        sum += coefficient(sigma, orbital_b) * t3[a * n + sigma];
                    }
                    f[a * n_vir + b] = sum;
                }
            }

            let weight = if i == j { 1.0 } else { 2.0 };
            let mut pair = 0.0;
            for a in 0..n_vir {
                let energy_a = orbital_energies[n_occupied + a];
                for b in 0..n_vir {
                    let iab = f[a * n_vir + b]; // (ia|jb)
                    let iba = f[b * n_vir + a]; // (ib|ja)
                    let denominator = orbital_energies[i] + orbital_energies[j]
                        - energy_a
                        - orbital_energies[n_occupied + b];
                    pair += iab * (2.0 * iab - iba) / denominator;
                }
            }
            if !pair.is_finite() {
                return Err(EngineError::Internal(format!(
                    "MP2: non-finite pair energy for occupied orbitals {i},{j}"
                )));
            }
            let contribution = weight * pair;
            dominant = dominant.max(contribution.abs());
            e2 += contribution;
        }
    }
    if !e2.is_finite() {
        return Err(EngineError::Internal(
            "MP2: correlation energy is not finite".into(),
        ));
    }
    Ok(Mp2Result {
        e2_hartree: e2,
        n_occupied,
        n_virtual: n_vir,
        dominant_pair_energy: dominant,
    })
}
