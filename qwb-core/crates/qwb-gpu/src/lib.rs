#![deny(unsafe_code)]
//! 可选的 **cuBLAS** 后端（EX-03）。
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
//! 只做两件事：`available()` 探测，和行主序 `f64` 的 `gemm`（`C = A·B`）。

pub mod ri_j;
pub mod t1;

use std::ffi::c_void;
use std::sync::OnceLock;

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
        }
    }
}

impl std::error::Error for GpuError {}

impl From<std::num::TryFromIntError> for GpuError {
    fn from(_: std::num::TryFromIntError) -> Self {
        GpuError::TooLarge(usize::MAX)
    }
}

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type CublasDgemm = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    *const f64,
    i32,
    *const f64,
    i32,
    *const f64,
    *mut f64,
    i32,
) -> i32;

/// 动态加载进来的 CUDA/cuBLAS 函数表。
struct Cuda {
    _cudart: libloading::Library,
    _cublas_lib: libloading::Library,
    cuda_malloc: CudaMalloc,
    cuda_free: CudaFree,
    cuda_memcpy: CudaMemcpy,
    cublas_create: CublasCreate,
    cublas_destroy: CublasDestroy,
    cublas_dgemm: CublasDgemm,
}

// 函数指针取自常驻的动态库，调用本身是线程安全的（cuBLAS handle 每次调用新建）。
#[allow(unsafe_code)]
unsafe impl Send for Cuda {}
#[allow(unsafe_code)]
unsafe impl Sync for Cuda {}

const CUBLAS_OP_N: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

/// 候选库名：先试 CUDA 13 / 12 的常见命名，再试不带版本号的名字。
fn library_candidates(stem: &str, majors: &[&str]) -> Vec<String> {
    let mut names = Vec::new();
    for major in majors {
        names.push(format!("{stem}{major}.dll"));
    }
    names.push(format!("{stem}.dll"));
    names
}

fn search_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(explicit) = std::env::var("QWB_CUDA_LIB_DIR") {
        dirs.push(std::path::PathBuf::from(explicit));
    }
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = std::env::var(var) {
            let base = std::path::PathBuf::from(base);
            dirs.push(base.join("bin").join("x64"));
            dirs.push(base.join("bin"));
            dirs.push(base.join("lib").join("x64"));
        }
    }
    dirs
}

#[allow(unsafe_code)]
fn load_first(stem: &str, majors: &[&str]) -> Result<libloading::Library, String> {
    let names = library_candidates(stem, majors);
    let mut tried = Vec::new();
    for dir in search_dirs() {
        for name in &names {
            let path = dir.join(name);
            tried.push(path.display().to_string());
            // SAFETY: 只加载库并取符号，符号类型下方逐一断言。
            if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                return Ok(lib);
            }
        }
    }
    for name in &names {
        tried.push(name.clone());
        // SAFETY: 同上；这里走系统库搜索路径。
        if let Ok(lib) = unsafe { libloading::Library::new(name) } {
            return Ok(lib);
        }
    }
    Err(format!(
        "找不到 {stem}（试过 {}）；设置 QWB_CUDA_LIB_DIR 指向 CUDA 的 bin 目录",
        tried.join(", ")
    ))
}

#[allow(unsafe_code)]
fn init() -> Result<&'static Cuda, GpuError> {
    static CUDA: OnceLock<Result<Cuda, String>> = OnceLock::new();
    match CUDA.get_or_init(|| {
        // 顺序有意义：cublas 依赖 cublasLt，先把它加载好。
        let cudart = load_first("cudart64_", &["13", "12", "11"])?;
        let _lt = load_first("cublasLt64_", &["13", "12", "11"])?;
        let cublas_lib = load_first("cublas64_", &["13", "12", "11"])?;
        // SAFETY: 符号名与 cudart / cuBLAS 的公开 ABI 一致；取不到就报错，不猜。
        unsafe {
            let cuda_malloc = *cudart
                .get::<CudaMalloc>(b"cudaMalloc\0")
                .map_err(|e| format!("cudaMalloc: {e}"))?;
            let cuda_free = *cudart
                .get::<CudaFree>(b"cudaFree\0")
                .map_err(|e| format!("cudaFree: {e}"))?;
            let cuda_memcpy = *cudart
                .get::<CudaMemcpy>(b"cudaMemcpy\0")
                .map_err(|e| format!("cudaMemcpy: {e}"))?;
            let cublas_create = *cublas_lib
                .get::<CublasCreate>(b"cublasCreate_v2\0")
                .map_err(|e| format!("cublasCreate_v2: {e}"))?;
            let cublas_destroy = *cublas_lib
                .get::<CublasDestroy>(b"cublasDestroy_v2\0")
                .map_err(|e| format!("cublasDestroy_v2: {e}"))?;
            let cublas_dgemm = *cublas_lib
                .get::<CublasDgemm>(b"cublasDgemm_v2\0")
                .map_err(|e| format!("cublasDgemm_v2: {e}"))?;
            Ok(Cuda {
                _cudart: cudart,
                _cublas_lib: cublas_lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cublas_create,
                cublas_destroy,
                cublas_dgemm,
            })
        }
    }) {
        Ok(cuda) => Ok(cuda),
        Err(why) => Err(GpuError::Unavailable(why.clone())),
    }
}

/// CUDA 后端是否可用（加载 + 建一次 handle 并销毁）。
///
/// # Errors
///
/// 找不到运行库、或建 handle 失败 → [`GpuError::Unavailable`] / [`GpuError::Call`]。
#[allow(unsafe_code)]
pub fn available() -> Result<(), GpuError> {
    let cuda = init()?;
    let mut handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: handle 是本函数内的局部变量，建完立刻销毁。
    let status = unsafe { (cuda.cublas_create)(&mut handle) };
    if status != 0 {
        return Err(GpuError::Call {
            what: "cublasCreate_v2",
            code: status,
        });
    }
    // SAFETY: handle 由 cublasCreate_v2 建、只有这里销毁一次。
    unsafe { (cuda.cublas_destroy)(handle) };
    Ok(())
}

/// 行主序 `f64` 矩阵乘 `C = A·B`（`A: m×k`、`B: k×n`、`C: m×n`）。
///
/// 实现细节：cuBLAS 是**列主序**。行主序的 `A(m×k)` 在列主序里就是 `Aᵀ(k×m)`，
/// 所以 `C = A·B` 等价于列主序的 `Cᵀ = Bᵀ·Aᵀ`，即用 `(n, m, k)` 调一次 `dgemm`
/// 并交换两个指针（`lda = n`、`ldb = k`、`ldc = n`）。这样不需要转置拷贝。
///
/// # Errors
///
/// 形状不合法 / 尺寸超 `c_int` / CUDA 不可用 / 调用失败。
#[allow(unsafe_code)]
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
    let (m_i, k_i, n_i) = (i32::try_from(m)?, i32::try_from(k)?, i32::try_from(n)?);
    let cuda = init()?;
    let mut handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: 下面每个 CUDA 调用都检查状态码，失败路径统一释放后返回。
    unsafe {
        let mut status = (cuda.cublas_create)(&mut handle);
        if status != 0 {
            return Err(GpuError::Call {
                what: "cublasCreate_v2",
                code: status,
            });
        }
        let mut device_a: *mut c_void = std::ptr::null_mut();
        let mut device_b: *mut c_void = std::ptr::null_mut();
        let mut device_c: *mut c_void = std::ptr::null_mut();
        let mut result: Result<Vec<f64>, GpuError> = Ok(Vec::new());
        let cleanup = |status: i32, what: &'static str| -> Result<Vec<f64>, GpuError> {
            Err(GpuError::Call { what, code: status })
        };
        status = (cuda.cuda_malloc)(&mut device_a, std::mem::size_of_val(a));
        if status != 0 {
            result = cleanup(status, "cudaMalloc(A)");
        }
        if result.is_ok() {
            status = (cuda.cuda_malloc)(&mut device_b, std::mem::size_of_val(b));
            if status != 0 {
                result = cleanup(status, "cudaMalloc(B)");
            }
        }
        if result.is_ok() {
            status = (cuda.cuda_malloc)(&mut device_c, m * n * std::mem::size_of::<f64>());
            if status != 0 {
                result = cleanup(status, "cudaMalloc(C)");
            }
        }
        if result.is_ok() {
            status = (cuda.cuda_memcpy)(
                device_a,
                a.as_ptr().cast(),
                std::mem::size_of_val(a),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if status != 0 {
                result = cleanup(status, "cudaMemcpy(A)");
            }
        }
        if result.is_ok() {
            status = (cuda.cuda_memcpy)(
                device_b,
                b.as_ptr().cast(),
                std::mem::size_of_val(b),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if status != 0 {
                result = cleanup(status, "cudaMemcpy(B)");
            }
        }
        if result.is_ok() {
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            status = (cuda.cublas_dgemm)(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n_i,
                m_i,
                k_i,
                &alpha,
                device_b.cast::<f64>(),
                n_i,
                device_a.cast::<f64>(),
                k_i,
                &beta,
                device_c.cast::<f64>(),
                n_i,
            );
            if status != 0 {
                result = cleanup(status, "cublasDgemm_v2");
            }
        }
        if result.is_ok() {
            let mut out = vec![0.0_f64; m * n];
            status = (cuda.cuda_memcpy)(
                out.as_mut_ptr().cast(),
                device_c,
                m * n * std::mem::size_of::<f64>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            if status != 0 {
                result = cleanup(status, "cudaMemcpy(C)");
            } else {
                result = Ok(out);
            }
        }
        // 释放（顺序无关，任何一个为 null 时 cudaFree 也接受 null）。
        (cuda.cuda_free)(device_a);
        (cuda.cuda_free)(device_b);
        (cuda.cuda_free)(device_c);
        (cuda.cublas_destroy)(handle);
        result
    }
}
