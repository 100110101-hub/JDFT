#![deny(unsafe_code)]
//! 可选的 **cuBLAS + NVRTC** 后端（EX-03）。
//!
//! 设计约束（见 project-phase2.md 附录 A.3）：
//!
//! 1. **CPU 永远是基准路径**：这个 crate 只是「同一运算的另一个实现」，
//!    任何调用方都必须能在 `Err` 时退回 CPU，且不得改变 CPU 的位模式。
//! 2. **没 GPU / 没 CUDA 必须优雅退化**：动态加载失败 → `GpuError::Unavailable`，
//!    不是 panic、不是构建失败（所以用 `libloading` 而不是链接期 `-lcublas`）。
//! 3. **不猜版本**：CUDA 13 的库名是 `cublas64_13.dll`；这里按顺序试一串候选名，
//!    并允许用 `QWB_CUDA_LIB_DIR` 显式指定目录。
//!
//! 交付三块核：
//!
//! * [`gemm`]（`C = A·B`，行主序 `f64`，cuBLAS）与 [`available`] / [`memory_info`] 的探测；
//! * `t1` / `ri_j`：AO→MO 变换与 RI-J；
//! * [`xc`]：**DFT 格点 XC 的收缩段**（feature `gpu-kernel`）—— 行点积核 + `v_xc` 装配，
//!   缺 GPU / 缺 NVRTC / 显存不够时逐位退回到与产品 CPU 路径相同的收缩。
//!
//! # 不安全面的位置
//!
//! 整个 crate 只有 `src/jit.rs` 开着 `unsafe_code`（JIT / 驱动绑定层），其余每个模块都
//! `deny(unsafe_code)`；这条规则由 `tests/unsafe_boundary.rs` 逐字扫描源码钉住（不是注释
//! 约定）—— 所以本文件里连那个关键字都不出现。

pub(crate) mod jit;
pub mod ri_j;
pub mod t1;
pub mod xc;
#[cfg(feature = "gpu-kernel")]
pub mod xc_device;

/// GPU 后端错误。**没有**任何变体会让调用方 panic —— 全部可回退到 CPU。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// 找不到 CUDA 运行库（没装 CUDA / 没设 PATH）。
    Unavailable(String),
    /// CUDA/cuBLAS 调用返回非零状态码。
    Call { what: &'static str, code: i32 },
    /// 形状不合法（`A` 是 `m×k`、`B` 是 `k×n`）。
    Dimension {
        m: usize,
        k: usize,
        n: usize,
        len_a: usize,
        len_b: usize,
    },
    /// 尺寸超出 `c_int`。
    TooLarge(usize),
    /// 显存预算不够：**调用方据此退回 CPU**，不重试、不缩小问题。
    InsufficientMemory { required: u64, available: u64 },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "CUDA 不可用：{why}"),
            Self::Call { what, code } => write!(f, "{what} 返回错误码 {code}"),
            Self::Dimension {
                m,
                k,
                n,
                len_a,
                len_b,
            } => write!(
                f,
                "形状不合法：A 应为 {m}×{k}（实际 {}）、B 应为 {k}×{n}（实际 {}）",
                len_a, len_b
            ),
            Self::TooLarge(size) => write!(f, "尺寸 {size} 超过 c_int 上限"),
            Self::InsufficientMemory {
                required,
                available,
            } => write!(
                f,
                "显存预算不够：需要 {required} B，预算 {available} B（请退回 CPU）"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<std::num::TryFromIntError> for GpuError {
    fn from(_: std::num::TryFromIntError) -> Self {
        GpuError::TooLarge(usize::MAX)
    }
}

/// 显存快照（字节）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    /// 当前空闲显存。
    pub free_bytes: u64,
    /// 总显存。
    pub total_bytes: u64,
}

/// CUDA 后端是否可用（加载 + 建一次 handle 并销毁）。
///
/// # Errors
///
/// 找不到运行库、或建 handle 失败 → [`GpuError::Unavailable`] / [`GpuError::Call`]。
pub fn available() -> Result<(), GpuError> {
    jit::available()
}

/// 查询当前设备的显存（`cudaMemGetInfo`）。
///
/// # Errors
///
/// 找不到运行库或调用失败。
pub fn memory_info() -> Result<MemoryInfo, GpuError> {
    jit::memory_info()
}

/// 一次 `gemm` 的显存需求（字节）：`A`、`B`、`C` 三块 `f64`。
#[must_use]
pub fn gemm_bytes(m: usize, k: usize, n: usize) -> u64 {
    (m * k + k * n + m * n) as u64 * 8
}

/// 默认显存预算：**空闲显存的 75 %**（给 cuBLAS 的 workspace 与驱动留余量）。
///
/// 75 % 不是调出来的"魔法系数"，而是一个**预算**：算力路径要么在预算内走 GPU，要么立刻
/// 退回 CPU —— 不允许"试着分配、失败了再说"，因为失败点可能在第三次 `cudaMalloc` 上，
/// 那时前两块已经分出去了（现在有清理路径，但预算检查更早、更好测）。
///
/// # Errors
///
/// 取不到显存信息（没有 CUDA）。
pub fn memory_budget_bytes() -> Result<u64, GpuError> {
    let info = memory_info()?;
    Ok(info.free_bytes / 4 * 3)
}

/// 一次 `gemm` 走了哪条路。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmRoute {
    /// 走 GPU（cuBLAS）。
    Gpu,
    /// 退回 CPU，附带 GPU 不可用的原因（无运行库 / 预算不够 / 调用失败）。
    Cpu(GpuError),
}

/// 行主序 `f64` 的 `gemm`，**用默认预算**（空闲显存的 75 %）。
///
/// # Errors
///
/// 形状不合法、CUDA 不可用、显存预算不够、cuBLAS 调用失败 —— **全部可回退**。
pub fn gemm_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, GpuError> {
    if a.len() != m * k || b.len() != k * n {
        return Err(GpuError::Dimension {
            m,
            k,
            n,
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    gemm_f64_budgeted(m, k, n, a, b, memory_budget_bytes()?)
}

/// 带**显式显存预算**的 `gemm`：`gemm_bytes(m, k, n) > budget_bytes` 时直接返回
/// [`GpuError::InsufficientMemory`]，一次显存都不碰（所以这条路径可以脱离真实 OOM 测试）。
///
/// # Errors
///
/// 同 [`gemm_f64`]。
pub fn gemm_f64_budgeted(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    budget_bytes: u64,
) -> Result<Vec<f64>, GpuError> {
    jit::gemm_f64_budgeted(m, k, n, a, b, budget_bytes)
}

/// 行主序 `f64` 的 `gemm`，**任何 GPU 失败都自动退回 CPU**，返回结果与走的路。
///
/// CPU 侧用与测试参考实现同款的朴素三重循环（`c = a·b`，按 `p` 累加），所以退回时的数值
/// 与"一开始就走 CPU"逐位相同 —— 这正是附录 A.3 要的"CPU 永远是基准"。
#[must_use]
pub fn gemm_f64_with_fallback(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> (Vec<f64>, GemmRoute) {
    match gemm_f64(m, k, n, a, b) {
        Ok(result) => (result, GemmRoute::Gpu),
        Err(reason) => (gemm_cpu(m, k, n, a, b), GemmRoute::Cpu(reason)),
    }
}

/// 朴素三重循环 `gemm`（行主序）：GPU 不可用时的基准路径。
#[must_use]
pub fn gemm_cpu(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
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
