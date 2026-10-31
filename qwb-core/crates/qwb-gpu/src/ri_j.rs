//! EX-03 第三块：**RI-J** 的核（B[(μν),P] 两次收缩），CPU gemm 与 cuBLAS 双路径。
//!
//! 为什么不是稠密 4 索引：见 `docs/choices.md` 九十一 —— `n=200` 时 `(μν|λσ)` 需要 12.2 GB，
//! 超过 8 GiB 显存；RI 的 3 索引只需 `n²·n_aux`（n=300 时 618 MB）。
//!
//! ```text
//! B[(μν),P] = (μν|P)                      # 3 索引，行主序 (μν) × P
//! t[P]      = Σ_λσ B[(λσ),P] P_λσ           # gemv
//! J[μν]     = Σ_P B[(μν),P] t[P]            # gemv
//! ```
//!
//! 这里把两步都写成 gemm 形状（`t` 当 1 列矩阵），以便和 T1 共用同一条 gemm 路径；
//! 没有 GPU 时退回本文件的朴素 CPU 乘法，**绝不 panic**。

use crate::GpuError;
use crate::gemm_f64;

/// RI-J 核的结果。
#[derive(Debug, Clone)]
pub struct RiJResult {
    /// `J[μν]`，行主序 `n × n`。
    pub j: Vec<f64>,
    /// 是否真的走了 GPU。
    pub used_gpu: bool,
    /// 退回 CPU 的原因（GPU 不可用时有内容）。
    pub fallback_reason: Option<String>,
}

fn naive_matmul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// 一次 gemm：GPU 优先，不可用则 CPU（返回 `(结果, 是否 GPU)`）。
fn gemm_or_cpu(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<(Vec<f64>, bool, Option<String>), String> {
    match gemm_f64(m, k, n, a, b) {
        Ok(values) => Ok((values, true, None)),
        Err(GpuError::Unavailable(why)) => Ok((naive_matmul(m, k, n, a, b), false, Some(why))),
        Err(other) => Err(format!("gemm 失败：{other}")),
    }
}

/// 朴素参考实现（独立于核）：直接四重循环算 RI-J。
#[must_use]
pub fn ri_j_naive(n: usize, n_aux: usize, b: &[f64], density: &[f64]) -> Vec<f64> {
    let mut j = vec![0.0_f64; n * n];
    let mut t = vec![0.0_f64; n_aux];
    for p in 0..n_aux {
        let mut sum = 0.0_f64;
        for lambda in 0..n {
            for sigma in 0..n {
                sum += b[(lambda * n + sigma) * n_aux + p] * density[lambda * n + sigma];
            }
        }
        t[p] = sum;
    }
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..n_aux {
                sum += b[(mu * n + nu) * n_aux + p] * t[p];
            }
            j[mu * n + nu] = sum;
        }
    }
    j
}

/// 用 gemm 实现的 RI-J（`B` 行主序 `(μν) × P`，`density` 行主序 `n × n`）。
///
/// # Errors
///
/// 形状不匹配或 GPU 调用硬失败。
pub fn ri_j_gemm(n: usize, n_aux: usize, b: &[f64], density: &[f64]) -> Result<RiJResult, String> {
    if b.len() != n * n * n_aux || density.len() != n * n {
        return Err(format!(
            "形状不匹配：n={n}, n_aux={n_aux}, b.len={}（应 {}）, density.len={}（应 {}）",
            b.len(),
            n * n * n_aux,
            density.len(),
            n * n
        ));
    }
    // 第一步：t[P] = Σ_(λσ) B[(λσ),P] · P_λσ —— 收缩的是 **B 的行**，所以需要 Bᵀ·P。
    // 这里显式转置一次（n²·n_aux 次拷贝）；后续可以给 gemm 加 transpose 标志省掉这次拷贝，
    // 但那要先证明它真的是瓶颈，现在不做。
    let mut b_t = vec![0.0_f64; n_aux * n * n];
    for row in 0..n * n {
        for p in 0..n_aux {
            b_t[p * n * n + row] = b[row * n_aux + p];
        }
    }
    let (t, gpu_t, why_t) = gemm_or_cpu(n_aux, n * n, 1, &b_t, density)?;
    // 第二步：J[μν] = Σ_P B[(μν),P] t[P]。
    let (j, gpu_j, why_j) = gemm_or_cpu(n * n, n_aux, 1, b, &t)?;
    Ok(RiJResult {
        j,
        used_gpu: gpu_t && gpu_j,
        fallback_reason: why_t.or(why_j),
    })
}
