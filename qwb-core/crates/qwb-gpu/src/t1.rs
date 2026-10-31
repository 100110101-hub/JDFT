//! EX-03 第二块：把 **AO→MO 的 T1 变换**做成可 GPU 的核，并用独立朴素实现对照。
//!
//! 目标形态与 `qwb_response::mo_integrals::build_cphf_integrals` 的 `T1` 一致：
//!
//! ```text
//! T1[(i,a),(j,b)] = Σ_μνλσ C_μa C_νi (μν|λσ) C_λb C_σj
//! ```
//!
//! 这里不碰产品路径（CPU 基准的位模式一点不动），只交付一个**可被接线**的核 +
//! 它自己的验收：与朴素四重循环在同一输入上 **< 1e-12**，并实测 GPU 加速比。
//! 没有 GPU / 显存不足 → 自动退回 CPU（`Fallback` 计数可见）。

/// 这个核走的是哪条路径（不在产品里，但验收与记账都要看得见）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// cuBLAS
    Gpu,
    /// CPU（无 GPU、或显存不足时）
    Cpu,
}

/// `T1` 核的结果：展平的 `n_pairs × n_pairs`（`p = i * n_virtual + a`）与所用路径。
#[derive(Debug, Clone)]
pub struct T1Result {
    /// `T1[p*n_pairs + q]`。
    pub values: Vec<f64>,
    /// 实际走的路径。
    pub path: Path,
    /// 退回 CPU 的原因（`Path::Cpu` 且曾有 GPU 可用时才有内容）。
    pub fallback_reason: Option<String>,
}

/// 朴素四重循环参考实现（**独立于核**，写在同一文件里但算法不同：先缩 λσ 再缩 μν）。
#[must_use]
pub fn t1_naive(n: usize, n_occ: usize, c: &[f64], eri: &[f64]) -> Vec<f64> {
    let n_vir = n - n_occ;
    let n_pairs = n_occ * n_vir;
    let mut out = vec![0.0_f64; n_pairs * n_pairs];
    for i in 0..n_occ {
        for a in 0..n_vir {
            let p = i * n_vir + a;
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let q = j * n_vir + b;
                    let mut sum = 0.0_f64;
                    for mu in 0..n {
                        let ca = c[mu * n + n_occ + a];
                        if ca == 0.0 {
                            continue;
                        }
                        for nu in 0..n {
                            let ci = c[nu * n + i];
                            if ci == 0.0 {
                                continue;
                            }
                            let mut inner = 0.0_f64;
                            for lambda in 0..n {
                                let cb = c[lambda * n + n_occ + b];
                                if cb == 0.0 {
                                    continue;
                                }
                                for sigma in 0..n {
                                    let cj = c[sigma * n + j];
                                    if cj == 0.0 {
                                        continue;
                                    }
                                    inner +=
                                        cb * cj * eri[((mu * n + nu) * n + lambda) * n + sigma];
                                }
                            }
                            sum += ca * ci * inner;
                        }
                    }
                    out[p * n_pairs + q] = sum;
                }
            }
        }
    }
    out
}

/// 用 gemm 做的 `T1`（CPU 走 `qwb-gpu` 自带的 gemm，有 GPU 则走 cuBLAS）。
///
/// 两步变换（与朴素实现完全等价，只是把内层收缩换成矩阵乘）：
///
/// ```text
/// X[(μν),(λ),j] = Σ_σ (μν|λσ) C_σj
/// T1[(i,a),(j,b)] = Σ_λ C_λb Σ_μν C_μa C_νi X[(μν),λ,j]
/// ```
///
/// # Errors
///
/// 形状不匹配 → `Err`。
pub fn t1_gemm(n: usize, n_occ: usize, c: &[f64], eri: &[f64]) -> Result<T1Result, String> {
    if c.len() != n * n || eri.len() != n * n * n * n {
        return Err(format!(
            "形状不匹配：n={n}, c.len={}, eri.len={}（应为 {} 与 {}）",
            c.len(),
            eri.len(),
            n * n,
            n * n * n * n
        ));
    }
    if n_occ == 0 || n_occ >= n {
        return Err(format!("n_occ={n_occ} 必须在 1..n({n}) 之间"));
    }
    let n_vir = n - n_occ;
    let n_pairs = n_occ * n_vir;
    // X 的形状：(μν) × (λ) × (j)，即 [n*n][n][n_occ]
    let mut x = vec![0.0_f64; n * n * n * n_occ];

    // 第一步：对每个 (μ,ν) 与每个 λ，用 C[:, j] 缩 σ。
    // 这里按 (μνλ) 行、j 列做一次 gemm，是最省事也最像「真实核」的形状。
    let rows = n * n * n;
    let mut a_matrix = vec![0.0_f64; rows * n];
    for mu in 0..n {
        for nu in 0..n {
            for lambda in 0..n {
                let row = (mu * n + nu) * n + lambda;
                for sigma in 0..n {
                    a_matrix[row * n + sigma] = eri[((mu * n + nu) * n + lambda) * n + sigma];
                }
            }
        }
    }
    let mut c_occ = vec![0.0_f64; n * n_occ];
    for mu in 0..n {
        for i in 0..n_occ {
            c_occ[mu * n_occ + i] = c[mu * n + i];
        }
    }
    let (x1, mut used_gpu) = gemm_or_cpu(rows, n, n_occ, &a_matrix, &c_occ)?;
    for row in 0..rows {
        for j in 0..n_occ {
            x[row * n_occ + j] = x1[row * n_occ + j];
        }
    }

    // 第二步：对每个 (i,a) 先缩 (μ,ν)，再缩 λ。
    // Y[(μν),(λ),j] 已在上一步；这里 T1[p][q] = Σ_μν C_μa C_νi ( Σ_λ X[(μν)λj] C_λb )
    let mut out = vec![0.0_f64; n_pairs * n_pairs];
    let mut c_virt = vec![0.0_f64; n * n_vir];
    for mu in 0..n {
        for a in 0..n_vir {
            c_virt[mu * n_vir + a] = c[mu * n + n_occ + a];
        }
    }
    for i in 0..n_occ {
        for j in 0..n_occ {
            // Z[μν][b] = Σ_λ X[μνλj] C_λb  → 行 = μν，列 = b
            let mut z_rows = vec![0.0_f64; n * n * n];
            for mu in 0..n {
                for nu in 0..n {
                    for lambda in 0..n {
                        let value = x[((mu * n + nu) * n + lambda) * n_occ + j];
                        z_rows[(mu * n + nu) * n + lambda] = value;
                    }
                }
            }
            let (z, gpu_here) = gemm_or_cpu(n * n, n, n_vir, &z_rows, &c_virt)?;
            used_gpu = used_gpu && gpu_here;
            // T1[(i,a),(j,b)] = Σ_μν C_μa C_νi Z[μν][b]
            for a in 0..n_vir {
                let p = i * n_vir + a;
                for b in 0..n_vir {
                    let q = j * n_vir + b;
                    let mut sum = 0.0_f64;
                    for mu in 0..n {
                        let ca = c_virt[mu * n_vir + a];
                        if ca == 0.0 {
                            continue;
                        }
                        for nu in 0..n {
                            sum += ca * c_occ[nu * n_occ + i] * z[(mu * n + nu) * n_vir + b];
                        }
                    }
                    out[p * n_pairs + q] = sum;
                }
            }
        }
    }

    let fallback_reason = LAST_FALLBACK.with(|slot| slot.borrow_mut().take());
    Ok(T1Result {
        values: out,
        path: if used_gpu { Path::Gpu } else { Path::Cpu },
        fallback_reason,
    })
}

/// `gemm_f64`，不可用则退回本文件的朴素 CPU 乘法（保证「没 GPU 也能跑」）。
/// 返回 `(结果, 是否真的走了 GPU)`，好让调用方如实报告路径。
fn gemm_or_cpu(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<(Vec<f64>, bool), String> {
    match crate::gemm_f64(m, k, n, a, b) {
        Ok(values) => Ok((values, true)),
        Err(crate::GpuError::Unavailable(why)) => {
            LAST_FALLBACK.with(|slot| *slot.borrow_mut() = Some(why));
            Ok((naive_matmul(m, k, n, a, b), false))
        }
        Err(other) => Err(format!("gemm 失败：{other}")),
    }
}

thread_local! {
    /// 最近一次退回 CPU 的原因（`Path::Cpu` 时用于如实说明）。
    static LAST_FALLBACK: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
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
