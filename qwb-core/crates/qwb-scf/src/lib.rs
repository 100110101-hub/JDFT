//! S-01：**Fock 矩阵构建**（不含 XC）。
//!
//! ```text
//! J_ij = Σ_kl (ij|kl) P_kl        库仑
//! K_ij = Σ_kl (ik|jl) P_kl        交换
//! F    = H_core + J − ½K          （闭壳层 HF：P 是自旋求和的密度）
//! ```
//!
//! # 为什么 S-01 能独立于"2e 积分支持到什么 l"
//!
//! S-01 只做**张量收缩**：输入是任意 `(ij|kl)` 张量 + 密度矩阵，输出 J/K/F。
//! 它不关心张量是怎么算出来的（那是 I-03），所以只要传入合法张量就能验收。
//! 上游 `two_electron` 目前只支持 s 型，这是 I-03 的事，不是 S-01 的。
//!
//! # 验收怎么做到"与直接乘对照 < 1e-12"
//!
//! 生产路径用**四重循环**（最直白、最容易读），测试路径把张量**展成矩阵**
//! 再用 `qwb_linalg::matmul` 乘一次 —— 两条完全不同的实现，差值必须在 1e-12 内。
//! 这就是 §6.5 S-01 的"与 nalgebra/faer 直接乘对照"。

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod analytic_gradient;
pub mod checkpoint;
pub mod diis;
pub mod ecp;
pub mod guess;
pub mod mp2;
pub mod pulay;
pub mod rimp2;
pub mod rks;
pub mod uhf;
pub mod uks;

use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::{LinalgError, Matrix};

/// Fock 构建结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FockMatrices {
    /// 库仑矩阵 `J`。
    pub coulomb: Matrix,
    /// 交换矩阵 `K`。
    pub exchange: Matrix,
    /// `F = H_core + J − ½K`。
    pub fock: Matrix,
}

/// 由 2e 张量与密度矩阵构建 J、K、F（闭壳层 HF 形式）。`P` 是自旋求和密度
///
/// # Errors
///
/// 尺寸不匹配（`density` 或 `hcore` 与张量的 `n_functions` 不一致）。
pub fn build_fock(
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density: &Matrix,
) -> Result<FockMatrices, LinalgError> {
    let n = tensor.n_functions;
    for (label, matrix) in [("hcore", hcore), ("density", density)] {
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(LinalgError::Dimension {
                context: if label == "hcore" {
                    "build_fock: hcore"
                } else {
                    "build_fock: density"
                },
                left_rows: matrix.nrows(),
                left_cols: matrix.ncols(),
                right_rows: n,
                right_cols: n,
            });
        }
    }

    // **IN-05：按 i 行并行，但保持逐位确定性。**
    //
    // 为什么这样并行不会改变结果：每个输出元素 `(i,j)` 仍由**同一个线程**按**同一个 k,l
    // 顺序**累加，所以浮点加法序列完全不变 → 与串行结果逐位相同（不是「容差内」，是 `==`）。
    // 反例（不能这么做）：多线程往同一个 `coulomb[(i,j)]` 上 `+=`，求和顺序随调度变化，
    // 结果会随运行漂移 —— 那就违反了确定性约束，也破坏了 K10 的可复现性要求。
    // 用项目的 `DeterministicReduce::map_ordered`（G7/R-13 认可的姿势）：结果按下标顺序
    // 返回，所以「谁算哪一行」不影响任何元素的加法顺序。
    let indices: Vec<usize> = (0..n).collect();
    let rows: Vec<(Vec<f64>, Vec<f64>)> =
        qwb_integrals::deterministic::DeterministicReduce::map_ordered(&indices, |&i| {
            let mut j_row = vec![0.0_f64; n];
            let mut k_row = vec![0.0_f64; n];
            for (j, slot) in j_row.iter_mut().enumerate() {
                let mut j_sum = 0.0_f64;
                let mut k_sum = 0.0_f64;
                for k in 0..n {
                    for l in 0..n {
                        let p = density[(k, l)];
                        if p == 0.0 {
                            continue;
                        }
                        j_sum += tensor.get(i, j, k, l) * p;
                        k_sum += tensor.get(i, k, j, l) * p;
                    }
                }
                *slot = j_sum;
                k_row[j] = k_sum;
            }
            (j_row, k_row)
        });

    let mut coulomb = Matrix::zeros(n, n);
    let mut exchange = Matrix::zeros(n, n);
    for (i, (j_row, k_row)) in rows.into_iter().enumerate() {
        for j in 0..n {
            coulomb[(i, j)] = j_row[j];
            exchange[(i, j)] = k_row[j];
        }
    }

    let mut fock = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            fock[(i, j)] = hcore[(i, j)] + coulomb[(i, j)] - 0.5 * exchange[(i, j)];
        }
    }

    Ok(FockMatrices {
        coulomb,
        exchange,
        fock,
    })
}

/// **串行参考实现**：与并行版逐位相同的对照基准（IN-05）。
///
/// 它就是并行化之前那段循环的逐字副本，保留在源码里是为了让「逐位确定性」成为
/// **可执行的性质**而不是口头承诺：测试对同一输入分别调用两者，要求所有元素 `==`。
///
/// # Errors
///
/// 维度不匹配 → [`LinalgError::Dimension`]。
pub fn build_fock_serial(
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density: &Matrix,
) -> Result<FockMatrices, LinalgError> {
    let n = tensor.n_functions;
    for (label, matrix) in [("hcore", hcore), ("density", density)] {
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(LinalgError::Dimension {
                context: if label == "hcore" {
                    "build_fock_serial: hcore"
                } else {
                    "build_fock_serial: density"
                },
                left_rows: matrix.nrows(),
                left_cols: matrix.ncols(),
                right_rows: n,
                right_cols: n,
            });
        }
    }
    let mut coulomb = Matrix::zeros(n, n);
    let mut exchange = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut j_sum = 0.0;
            let mut k_sum = 0.0;
            for k in 0..n {
                for l in 0..n {
                    let p = density[(k, l)];
                    if p == 0.0 {
                        continue;
                    }
                    j_sum += tensor.get(i, j, k, l) * p;
                    k_sum += tensor.get(i, k, j, l) * p;
                }
            }
            coulomb[(i, j)] = j_sum;
            exchange[(i, j)] = k_sum;
        }
    }
    let mut fock = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            fock[(i, j)] = hcore[(i, j)] + coulomb[(i, j)] - 0.5 * exchange[(i, j)];
        }
    }
    Ok(FockMatrices {
        coulomb,
        exchange,
        fock,
    })
}

/// 双电子能量贡献 `E_2e = ½ Σ_ij P_ij (J_ij − ½K_ij)`（闭壳层 HF）。
///
/// # Panics
///
/// 形状不一致（调用方应先用 [`build_fock`] 的校验路径）。
#[must_use]
pub fn two_electron_energy(density: &Matrix, fock: &FockMatrices) -> f64 {
    let n = density.nrows();
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total += density[(i, j)] * (fock.coulomb[(i, j)] - 0.5 * fock.exchange[(i, j)]);
        }
    }
    0.5 * total
}
