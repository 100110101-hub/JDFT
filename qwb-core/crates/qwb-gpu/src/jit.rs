#![allow(unsafe_code)]
//! EX-03 的 **JIT / 驱动绑定层**，也是本 crate 里**唯一**允许 `unsafe` 的模块。
//!
//! 规则由 `crates/qwb-gpu/tests/unsafe_boundary.rs` 逐字扫描源码钉住：
//! `crates/qwb-gpu/src/` 下的 `unsafe` 关键字只出现在本文件（`lib.rs`、`t1.rs`、`ri_j.rs`、
//! `xc.rs` 一律零出现）。
//!
//! 这个文件装两块东西：
//!
//! 1. **cudart / cuBLAS 的动态绑定**（`libloading`，从不链接期依赖）。这一段原本写在
//!    `lib.rs` 里，EX-03 的 XC 核引入 NVRTC 之后按 `docs/choices.md` 二百的决定把全部
//!    `unsafe` 收进这一层 —— 纯代码搬移：公开 API、调用次序、错误码一字未动。
//! 2. **NVRTC + CUDA 驱动 API**（feature `gpu-kernel`）：把**常量** CUDA C 源码编译成 PTX、
//!    装进 module、launch 内核。源码是 `const XC_SOURCE`，不接受任何用户输入。
//!
//! # `unsafe` 靠什么不变量成立
//!
//! 两条外部 ABI 是 FFI：函数指针签名必须与 CUDA 头文件逐项一致（类型别名就是声明），
//! 缓冲指针必须指向真实设备内存、长度必须是元素数×8。因此每个 `unsafe` 块旁边都有一条
//! `// SAFETY:` 说明它依赖的不变量，而不是一句"调用 FFI"。

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::{GpuError, MemoryInfo};

// ---------------------------------------------------------------------------------------
// 1. cudart / cuBLAS 函数指针类型
// ---------------------------------------------------------------------------------------

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type CudaMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
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
pub(crate) struct Cuda {
    _cudart: libloading::Library,
    _cublas_lib: libloading::Library,
    pub(crate) cuda_malloc: CudaMalloc,
    pub(crate) cuda_free: CudaFree,
    pub(crate) cuda_memcpy: CudaMemcpy,
    cuda_mem_get_info: CudaMemGetInfo,
    pub(crate) cublas_create: CublasCreate,
    pub(crate) cublas_destroy: CublasDestroy,
    pub(crate) cublas_dgemm: CublasDgemm,
}

// 函数指针取自常驻的动态库，调用本身是线程安全的（cuBLAS handle 每次调用新建）。
// SAFETY: 这里只把一个**不可变**的函数表标记为可跨线程共享；表里每一项都是进程生命期内
// 一直有效的动态库符号（`_cudart` / `_cublas_lib` 由本结构体持有，永不 drop），且 CUDA
// 的 runtime / cuBLAS 调用自身是线程安全的。
unsafe impl Send for Cuda {}
// SAFETY: 同上：`&Cuda` 只被读取（取函数指针），不存在数据竞争。
unsafe impl Sync for Cuda {}

pub(crate) const CUBLAS_OP_N: i32 = 0;
/// 只在 `gpu-kernel` 的 `v_xc` 装配（`Φᵀ·scaled`）里用得到。
#[cfg(feature = "gpu-kernel")]
pub(crate) const CUBLAS_OP_T: i32 = 1;
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

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
            dirs.push(base.join("lib"));
        }
    }
    dirs
}

/// 在候选目录 / 系统搜索路径里找第一个能加载的库。
///
/// 刻意不 `unsafe` 包装成"安全函数"：符号类型的正确性由调用方在这里一次性断言，
/// 所以它保持私有、只被本文件调用。
fn load_first(stem: &str, majors: &[&str]) -> Result<libloading::Library, String> {
    let names = library_candidates(stem, majors);
    let mut tried = Vec::new();
    for dir in search_dirs() {
        for name in &names {
            let path = dir.join(name);
            tried.push(path.display().to_string());
            // SAFETY: 只加载库并取符号；符号类型由 `init()` 里逐项 `get::<T>` 断言，
            // 签名与 CUDA 公开 ABI 一致。
            if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                return Ok(lib);
            }
        }
    }
    for name in &names {
        tried.push(name.clone());
        // SAFETY: 同上；这里走系统库搜索路径（`nvcuda.dll` 就只在那里）。
        if let Ok(lib) = unsafe { libloading::Library::new(name) } {
            return Ok(lib);
        }
    }
    Err(format!(
        "找不到 {stem}（试过 {}）；设置 QWB_CUDA_LIB_DIR 指向 CUDA 的 bin 目录",
        tried.join(", ")
    ))
}

/// 进程内一次性初始化：加载 cudart + cuBLAS，取出全部符号。
pub(crate) fn init() -> Result<&'static Cuda, GpuError> {
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
            let cuda_mem_get_info = *cudart
                .get::<CudaMemGetInfo>(b"cudaMemGetInfo\0")
                .map_err(|e| format!("cudaMemGetInfo: {e}"))?;
            let cublas_create = *cublas_lib
                .get::<CublasCreate>(b"cublasCreate_v2\0")
                .map_err(|e| format!("cublasCreate_v2: {e}"))?;
            let cublas_destroy = *cublas_lib
                .get::<CublasDestroy>(b"cublasDestroy_v2\0")
                .map_err(|e| format!("cublasDestroy_v2: {e}"))?;
            let cublas_dgemm = *cublas_lib
                .get::<CublasDgemm>(b"cublasDgemm_v2\0")
                .map_err(|e| format!("cublasDgemm_v2: {e}"))?;
            // CUDA 的 runtime 是**惰性初始化**的：没有任何 context 时，
            // `cublasCreate_v2` 会返回 1（`CUBLAS_STATUS_NOT_INITIALIZED`），
            // `cudaMemGetInfo` 会返回 801（`cudaErrorNotSupported`）。两个测试并发跑时
            // 这个竞态真的会踩到（第 376 轮实测：`availability_probe_never_panics` 单独跑绿、
            // 与 `gemm_matches...` 并行跑红）。`cudaFree(0)` 是官方文档里
            // "初始化 runtime" 的无操作调用 —— 在**一次性**的 `init()` 里踢一脚，
            // 之后所有入口（gemm / memory_info / available）都不必各自处理。
            // SAFETY: 这里已经在上面那个 `unsafe { ... }` 块里；传空指针给 cudaFree 是
            // 文档允许的初始化用法，不释放任何东西。
            cuda_free(std::ptr::null_mut());
            Ok(Cuda {
                _cudart: cudart,
                _cublas_lib: cublas_lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cuda_mem_get_info,
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
pub(crate) fn available() -> Result<(), GpuError> {
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

/// 查询当前设备的显存（`cudaMemGetInfo`）。
pub(crate) fn memory_info() -> Result<MemoryInfo, GpuError> {
    let cuda = init()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: 两个指针都指向本函数内的局部变量，cudaMemGetInfo 只写它们。
    let status = unsafe { (cuda.cuda_mem_get_info)(&mut free, &mut total) };
    if status != 0 {
        return Err(GpuError::Call {
            what: "cudaMemGetInfo",
            code: status,
        });
    }
    Ok(MemoryInfo {
        free_bytes: free as u64,
        total_bytes: total as u64,
    })
}

/// 带显式显存预算的 `gemm`（形状检查与预算判定在 `lib.rs` 的调用方里已经做过一遍，
/// 这里再做一遍是为了让本函数单独被调用时也安全）。
pub(crate) fn gemm_f64_budgeted(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    budget_bytes: u64,
) -> Result<Vec<f64>, GpuError> {
    if a.len() != m * k || b.len() != k * n {
        return Err(GpuError::Dimension {
            m,
            k,
            n,
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let required = crate::gemm_bytes(m, k, n);
    if required > budget_bytes {
        return Err(GpuError::InsufficientMemory {
            required,
            available: budget_bytes,
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

// ---------------------------------------------------------------------------------------
// 2. NVRTC + CUDA driver API（feature `gpu-kernel`）
// ---------------------------------------------------------------------------------------

#[cfg(feature = "gpu-kernel")]
pub(crate) mod kernel {
    //! XC 核的 JIT 层：常量 CUDA C 源码 → NVRTC → PTX → `CUmodule` → `cuLaunchKernel`。
    //!
    //! 只暴露**按名字命名的一次性操作**（行点积、按行缩比、两个固定形状的 gemm），
    //! 调度层（`xc.rs`）看到的是安全的 Rust 函数，看不到裸指针。

    use super::{CUBLAS_OP_N, CUBLAS_OP_T, Cuda, init, load_first};
    use crate::GpuError;
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// 内核源码。**常量字符串**，不接受任何用户输入（附录 A · EX-03 约束 ①）。
    ///
    /// 两个核，都是逐元素形状：
    ///
    /// * `qwb_xc_row_dot`：`out[p] = Σ_μ left[p,μ]·right[p,μ]` —— **本次交付的主角**。
    ///   BLAS 给不出它：`gemm` 只能算出 `ΦP`，而 `ρ_p = Σ_μ Φ_pμ (ΦP)_pμ` 是再把两张
    ///   同形矩阵**按行点积**压成一个向量。每个线程吃一整行，读 2·n 个 double、写 1 个。
    /// * `qwb_xc_scale_rows`：`out[p,μ] = scale[p]·phi[p,μ]` —— 把逐点系数摊到 AO 行上，
    ///   之后 `v_xc = Φᵀ·(a·Φ)` 就又变回一次标准 gemm（`cuBLAS` 直接吃）。
    ///
    /// 求和顺序里没有原子操作、没有跨线程归约，所以结果是**确定**的。
    const XC_SOURCE: &str = r#"
extern "C" __global__ void qwb_xc_row_dot(const double* __restrict__ left,
                                          const double* __restrict__ right,
                                          double* __restrict__ out,
                                          int n, long long points) {
    long long p = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= points) { return; }
    const double* x = left + p * (long long)n;
    const double* y = right + p * (long long)n;
    double sum = 0.0;
    for (int mu = 0; mu < n; ++mu) { sum += x[mu] * y[mu]; }
    out[p] = sum;
}

extern "C" __global__ void qwb_xc_scale_rows(const double* __restrict__ phi,
                                             const double* __restrict__ scale,
                                             double* __restrict__ out,
                                             int n, long long points) {
    long long p = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= points) { return; }
    const double s = scale[p];
    const long long base = p * (long long)n;
    for (int mu = 0; mu < n; ++mu) { out[base + mu] = s * phi[base + mu]; }
}
"#;

    /// 每个 block 的线程数（行点积是访存密集型，256 足够打满 3070 的 40 个 SM）。
    const BLOCK: u32 = 256;

    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
    type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
    type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    #[allow(clippy::type_complexity)]
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuGetErrorName = unsafe extern "C" fn(i32, *mut *const i8) -> i32;

    type NvrtcCreateProgram = unsafe extern "C" fn(
        *mut *mut c_void,
        *const i8,
        *const i8,
        i32,
        *const *const i8,
        *const *const i8,
    ) -> i32;
    type NvrtcCompileProgram = unsafe extern "C" fn(*mut c_void, i32, *const *const i8) -> i32;
    type NvrtcGetProgramLogSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;
    type NvrtcGetProgramLog = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;
    type NvrtcGetPtxSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;
    type NvrtcGetPtx = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;
    type NvrtcDestroyProgram = unsafe extern "C" fn(*mut *mut c_void) -> i32;

    /// 驱动 API 函数表（只在 `XcSession` 里用）。
    struct DriverApi {
        cu_init: CuInit,
        cu_device_get: CuDeviceGet,
        cu_device_get_attribute: CuDeviceGetAttribute,
        cu_primary_ctx_retain: CuDevicePrimaryCtxRetain,
        cu_ctx_set_current: CuCtxSetCurrent,
        cu_module_load_data: CuModuleLoadData,
        cu_module_get_function: CuModuleGetFunction,
        cu_mem_alloc: CuMemAlloc,
        cu_mem_free: CuMemFree,
        cu_memcpy_htod: CuMemcpyHtoD,
        cu_memcpy_dtoh: CuMemcpyDtoH,
        cu_launch_kernel: CuLaunchKernel,
        cu_ctx_synchronize: CuCtxSynchronize,
        cu_get_error_name: CuGetErrorName,
    }

    /// 一块设备内存。`Drop` 释放；`ptr == 0` 表示空缓冲（不释放）。
    pub(crate) struct DeviceBuffer {
        ptr: u64,
        free: CuMemFree,
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if self.ptr != 0 {
                // SAFETY: `ptr` 只可能由本模块的 `cuMemAlloc_v2` 产生（见 `XcSession::alloc`），
                // 每个 `DeviceBuffer` 恰好持有一块，`Drop` 只跑一次，所以这里是这块内存的
                // 唯一一次释放。`free` 是 `nvcuda.dll` 里的 `cuMemFree_v2`，库本身由
                // 进程级 `OnceLock` 持有、不会先于缓冲卸载。
                unsafe { (self.free)(self.ptr) };
            }
        }
    }

    /// 一次 JIT + module 装载的成果，进程内只做一次。
    pub(crate) struct XcSession {
        _nvrtc: libloading::Library,
        _driver: libloading::Library,
        api: DriverApi,
        ctx: *mut c_void,
        _module: *mut c_void,
        row_dot_fn: *mut c_void,
        scale_rows_fn: *mut c_void,
        /// NVRTC 报出的版本号（诊断用：`13.0`）。
        pub(crate) nvrtc_version: (i32, i32),
        /// 编译期用的 `--gpu-architecture=compute_XY`。
        pub(crate) gpu_arch: i32,
    }

    // SAFETY: 结构体里只有动态库句柄、CUDA 驱动的不透明句柄和函数指针。它们都是进程生命期
    // 内常驻的对象；驱动 API 本身线程安全（context 是按线程 current 的，每次进入操作前
    // `ensure_current` 都会重新绑定本线程）。`&XcSession` 的两个方法对共享状态只读。
    unsafe impl Send for XcSession {}
    // SAFETY: 同上。
    unsafe impl Sync for XcSession {}

    impl XcSession {
        /// 取进程内唯一的会话；第一次调用会编译内核（约几十毫秒）。
        ///
        /// # Errors
        ///
        /// 没装 NVRTC / 没有驱动 / 没有设备 / 编译失败 —— 全部是 [`GpuError::Unavailable`]，
        /// 调用方据此退回 CPU，**绝不 panic**。
        pub(crate) fn get() -> Result<&'static XcSession, GpuError> {
            static SESSION: OnceLock<Result<XcSession, String>> = OnceLock::new();
            match SESSION.get_or_init(XcSession::create) {
                Ok(session) => Ok(session),
                Err(why) => Err(GpuError::Unavailable(why.clone())),
            }
        }

        fn create() -> Result<XcSession, String> {
            // ---- 驱动 ----
            let driver = load_first("nvcuda", &[])
                .map_err(|why| format!("{why}（CUDA 驱动库不可加载：没有 GPU / 没装驱动）"))?;
            // SAFETY: 下面的符号名都是 `nvcuda.dll` 的公开导出；取不到就报错，不猜。
            // 每个 `get::<T>` 的类型 `T` 就是 CUDA 头文件里的签名。
            let api = unsafe {
                let get = |name: &[u8]| -> Result<*mut c_void, String> {
                    driver
                        .get::<*mut c_void>(name)
                        .map(|symbol| *symbol)
                        .map_err(|e| {
                            format!(
                                "驱动符号 {} 取不到：{e}",
                                String::from_utf8_lossy(&name[..name.len() - 1])
                            )
                        })
                };
                let cu_init: CuInit = std::mem::transmute(get(b"cuInit\0")?);
                let cu_device_get: CuDeviceGet = std::mem::transmute(get(b"cuDeviceGet\0")?);
                let cu_device_get_attribute: CuDeviceGetAttribute =
                    std::mem::transmute(get(b"cuDeviceGetAttribute\0")?);
                let cu_primary_ctx_retain: CuDevicePrimaryCtxRetain =
                    std::mem::transmute(get(b"cuDevicePrimaryCtxRetain\0")?);
                let cu_ctx_set_current: CuCtxSetCurrent =
                    std::mem::transmute(get(b"cuCtxSetCurrent\0")?);
                let cu_module_load_data: CuModuleLoadData =
                    std::mem::transmute(get(b"cuModuleLoadData\0")?);
                let cu_module_get_function: CuModuleGetFunction =
                    std::mem::transmute(get(b"cuModuleGetFunction\0")?);
                let cu_mem_alloc: CuMemAlloc = std::mem::transmute(get(b"cuMemAlloc_v2\0")?);
                let cu_mem_free: CuMemFree = std::mem::transmute(get(b"cuMemFree_v2\0")?);
                let cu_memcpy_htod: CuMemcpyHtoD = std::mem::transmute(get(b"cuMemcpyHtoD_v2\0")?);
                let cu_memcpy_dtoh: CuMemcpyDtoH = std::mem::transmute(get(b"cuMemcpyDtoH_v2\0")?);
                let cu_launch_kernel: CuLaunchKernel =
                    std::mem::transmute(get(b"cuLaunchKernel\0")?);
                let cu_ctx_synchronize: CuCtxSynchronize =
                    std::mem::transmute(get(b"cuCtxSynchronize\0")?);
                let cu_get_error_name: CuGetErrorName =
                    std::mem::transmute(get(b"cuGetErrorName\0")?);
                DriverApi {
                    cu_init,
                    cu_device_get,
                    cu_device_get_attribute,
                    cu_primary_ctx_retain,
                    cu_ctx_set_current,
                    cu_module_load_data,
                    cu_module_get_function,
                    cu_mem_alloc,
                    cu_mem_free,
                    cu_memcpy_htod,
                    cu_memcpy_dtoh,
                    cu_launch_kernel,
                    cu_ctx_synchronize,
                    cu_get_error_name,
                }
            };

            // SAFETY: 每个调用都检查返回码；`ctx` 由 `cuDevicePrimaryCtxRetain` 写出，
            // 之后只作为不透明句柄传给驱动。primary context 与 cudart 自己用的是同一个，
            // 所以 cuBLAS（经 cudart 建）和 CUmodule 落在同一个上下文里。
            let (ctx, gpu_arch) = unsafe {
                let status = (api.cu_init)(0);
                if status != 0 {
                    return Err(call_error(&api, "cuInit", status));
                }
                let mut device: i32 = 0;
                let status = (api.cu_device_get)(&mut device, 0);
                if status != 0 {
                    return Err(call_error(&api, "cuDeviceGet", status));
                }
                let mut major: i32 = 0;
                let mut minor: i32 = 0;
                // 75 / 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}
                let status = (api.cu_device_get_attribute)(&mut major, 75, device);
                if status != 0 {
                    return Err(call_error(&api, "cuDeviceGetAttribute(major)", status));
                }
                let status = (api.cu_device_get_attribute)(&mut minor, 76, device);
                if status != 0 {
                    return Err(call_error(&api, "cuDeviceGetAttribute(minor)", status));
                }
                let mut ctx: *mut c_void = std::ptr::null_mut();
                let status = (api.cu_primary_ctx_retain)(&mut ctx, device);
                if status != 0 {
                    return Err(call_error(&api, "cuDevicePrimaryCtxRetain", status));
                }
                let status = (api.cu_ctx_set_current)(ctx);
                if status != 0 {
                    return Err(call_error(&api, "cuCtxSetCurrent", status));
                }
                (ctx, major * 10 + minor)
            };

            // ---- NVRTC ----
            let nvrtc = load_nvrtc()?;
            let (nvrtc_version, image) = compile_ptx(&nvrtc, gpu_arch)?;

            // SAFETY: `image` 是 NVRTC 刚写满的 PTX 文本（已含 NUL）；`module` 由驱动写出。
            let (module, row_dot_fn, scale_rows_fn) = unsafe {
                let mut module: *mut c_void = std::ptr::null_mut();
                let status = (api.cu_module_load_data)(&mut module, image.as_ptr().cast());
                if status != 0 {
                    return Err(call_error(&api, "cuModuleLoadData", status));
                }
                let mut row_dot_fn: *mut c_void = std::ptr::null_mut();
                let status = (api.cu_module_get_function)(
                    &mut row_dot_fn,
                    module,
                    c"qwb_xc_row_dot".as_ptr(),
                );
                if status != 0 {
                    return Err(call_error(&api, "cuModuleGetFunction(row_dot)", status));
                }
                let mut scale_rows_fn: *mut c_void = std::ptr::null_mut();
                let status = (api.cu_module_get_function)(
                    &mut scale_rows_fn,
                    module,
                    c"qwb_xc_scale_rows".as_ptr(),
                );
                if status != 0 {
                    return Err(call_error(&api, "cuModuleGetFunction(scale_rows)", status));
                }
                (module, row_dot_fn, scale_rows_fn)
            };

            Ok(XcSession {
                _nvrtc: nvrtc,
                _driver: driver,
                api,
                ctx,
                _module: module,
                row_dot_fn,
                scale_rows_fn,
                nvrtc_version,
                gpu_arch,
            })
        }

        /// 把本线程的 current context 绑到这个会话的 primary context 上。
        ///
        /// context 是**按线程** current 的：SCF 在别的线程上调用时（或第一次调用发生在
        /// 别的线程）必须重新绑。这一步很便宜，所以每次进操作都做。
        fn ensure_current(&self) -> Result<(), GpuError> {
            // SAFETY: `ctx` 是 `cuDevicePrimaryCtxRetain` 给出的有效句柄，进程生命期内不变。
            let status = unsafe { (self.api.cu_ctx_set_current)(self.ctx) };
            if status != 0 {
                return Err(GpuError::Call {
                    what: "cuCtxSetCurrent",
                    code: status,
                });
            }
            Ok(())
        }

        /// 分配 `len` 个 `f64`。
        pub(crate) fn alloc(&self, len: usize) -> Result<DeviceBuffer, GpuError> {
            if len == 0 {
                return Err(GpuError::Dimension {
                    m: 0,
                    k: 0,
                    n: 0,
                    len_a: 0,
                    len_b: 0,
                });
            }
            self.ensure_current()?;
            let bytes = len
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or(GpuError::TooLarge(len))?;
            let mut ptr: u64 = 0;
            // SAFETY: `ptr` 是局部变量，`cuMemAlloc_v2` 只写它；`bytes` 已做过溢出检查。
            let status = unsafe { (self.api.cu_mem_alloc)(&mut ptr, bytes) };
            if status != 0 {
                return Err(GpuError::Call {
                    what: "cuMemAlloc",
                    code: status,
                });
            }
            Ok(DeviceBuffer {
                ptr,
                free: self.api.cu_mem_free,
            })
        }

        /// 上传一块 host `f64` 切片。
        pub(crate) fn upload(&self, data: &[f64]) -> Result<DeviceBuffer, GpuError> {
            let buffer = self.alloc(data.len())?;
            if !data.is_empty() {
                self.ensure_current()?;
                // SAFETY: 源是 `data.as_ptr()`（有效、只读、长度 `size_of_val(data)`），
                // 目标是 `buffer.ptr`（本会话刚分配的 ≥ 同样大小的设备内存）。
                let status = unsafe {
                    (self.api.cu_memcpy_htod)(
                        buffer.ptr,
                        data.as_ptr().cast(),
                        std::mem::size_of_val(data),
                    )
                };
                if status != 0 {
                    return Err(GpuError::Call {
                        what: "cuMemcpyHtoD",
                        code: status,
                    });
                }
            }
            Ok(buffer)
        }

        /// 下载 `len` 个 `f64`。
        pub(crate) fn download(
            &self,
            src: &DeviceBuffer,
            len: usize,
        ) -> Result<Vec<f64>, GpuError> {
            let mut out = vec![0.0_f64; len];
            if len > 0 {
                self.ensure_current()?;
                // SAFETY: 目标是 `out`（长度 `len`），源是 `src.ptr`；调用方保证 `src`
                // 至少有 `len` 个 f64（本模块的 `alloc` 按元素数分配）。
                let status = unsafe {
                    (self.api.cu_memcpy_dtoh)(
                        out.as_mut_ptr().cast(),
                        src.ptr,
                        std::mem::size_of_val(&out[..]),
                    )
                };
                if status != 0 {
                    return Err(GpuError::Call {
                        what: "cuMemcpyDtoH",
                        code: status,
                    });
                }
            }
            Ok(out)
        }

        /// 等所有已提交的内核/gemm 做完。
        pub(crate) fn synchronize(&self) -> Result<(), GpuError> {
            // SAFETY: 无参数，只阻塞当前线程直到设备队列排空。
            let status = unsafe { (self.api.cu_ctx_synchronize)() };
            if status != 0 {
                return Err(GpuError::Call {
                    what: "cuCtxSynchronize",
                    code: status,
                });
            }
            Ok(())
        }

        /// `out[p] = Σ_μ left[p,μ]·right[p,μ]`，三个缓冲都是 `points×n` 行主序（`out` 是 `points`）。
        pub(crate) fn row_dot(
            &self,
            left: &DeviceBuffer,
            right: &DeviceBuffer,
            out: &DeviceBuffer,
            n: usize,
            points: usize,
        ) -> Result<(), GpuError> {
            let (n_i, points_i) = (i32::try_from(n)?, i64::try_from(points)?);
            let (mut a, mut b, mut c) = (left.ptr, right.ptr, out.ptr);
            let (mut n_arg, mut points_arg) = (n_i, points_i);
            self.launch(
                self.row_dot_fn,
                points,
                &mut [
                    (&raw mut a).cast::<c_void>(),
                    (&raw mut b).cast::<c_void>(),
                    (&raw mut c).cast::<c_void>(),
                    (&raw mut n_arg).cast::<c_void>(),
                    (&raw mut points_arg).cast::<c_void>(),
                ],
                "qwb_xc_row_dot",
            )
        }

        /// `out[p,μ] = scale[p]·phi[p,μ]`（`phi`/`out` 是 `points×n`，`scale` 是 `points`）。
        pub(crate) fn scale_rows(
            &self,
            phi: &DeviceBuffer,
            scale: &DeviceBuffer,
            out: &DeviceBuffer,
            n: usize,
            points: usize,
        ) -> Result<(), GpuError> {
            let (n_i, points_i) = (i32::try_from(n)?, i64::try_from(points)?);
            let (mut a, mut b, mut c) = (phi.ptr, scale.ptr, out.ptr);
            let (mut n_arg, mut points_arg) = (n_i, points_i);
            self.launch(
                self.scale_rows_fn,
                points,
                &mut [
                    (&raw mut a).cast::<c_void>(),
                    (&raw mut b).cast::<c_void>(),
                    (&raw mut c).cast::<c_void>(),
                    (&raw mut n_arg).cast::<c_void>(),
                    (&raw mut points_arg).cast::<c_void>(),
                ],
                "qwb_xc_scale_rows",
            )
        }

        /// `xm = phi·p`：`phi` 是 `points×n`、`p` 是 `n×n`、`xm` 是 `points×n`（全行主序）。
        ///
        /// cuBLAS 是列主序，行主序的 `Φ` 在列主序里就是 `Φᵀ`，于是
        /// `xmᵀ = pᵀ·Φᵀ` 正好是一次 `(m=n, n=points, k=n)` 的无转置 `dgemm`。
        pub(crate) fn gemm_xm(
            &self,
            xm: &DeviceBuffer,
            phi: &DeviceBuffer,
            p: &DeviceBuffer,
            n: usize,
            points: usize,
        ) -> Result<(), GpuError> {
            self.dgemm(
                i32::try_from(n)?,
                i32::try_from(points)?,
                i32::try_from(n)?,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                p.ptr,
                n,
                phi.ptr,
                n,
                xm.ptr,
                n,
                "cublasDgemm(xm = phi·p)",
            )
        }

        /// `v = phiᵀ·scaled`：`phi`/`scaled` 是 `points×n`，`v` 是 `n×n` 行主序。
        ///
        /// 列主序视角：`vᵀ = Φᵀ·scaled`，即 `dgemm(m=n, n=n, k=points)`，
        /// 第一操作数是 `phi` 的列主序视图、第二操作数取转置。
        pub(crate) fn gemm_v(
            &self,
            v: &DeviceBuffer,
            phi: &DeviceBuffer,
            scaled: &DeviceBuffer,
            n: usize,
            points: usize,
        ) -> Result<(), GpuError> {
            self.dgemm(
                i32::try_from(n)?,
                i32::try_from(n)?,
                i32::try_from(points)?,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                phi.ptr,
                n,
                scaled.ptr,
                n,
                v.ptr,
                n,
                "cublasDgemm(v = phiᵀ·scaled)",
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn dgemm(
            &self,
            m: i32,
            n: i32,
            k: i32,
            op_a: i32,
            op_b: i32,
            a: u64,
            lda: usize,
            b: u64,
            ldb: usize,
            c: u64,
            ldc: usize,
            what: &'static str,
        ) -> Result<(), GpuError> {
            let cuda: &Cuda = init()?;
            self.ensure_current()?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            // SAFETY: handle 是局部变量；下面成功/失败两条路径都恰好销毁一次。
            let status = unsafe { (cuda.cublas_create)(&mut handle) };
            if status != 0 {
                return Err(GpuError::Call {
                    what: "cublasCreate_v2",
                    code: status,
                });
            }
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            // SAFETY: 三个指针都是本会话分配的设备内存，形状/主维由调用方按上面的文档
            // 逐项对齐；`cublasDgemm_v2` 只读 `a`/`b`、只写 `c`。
            let status = unsafe {
                (cuda.cublas_dgemm)(
                    handle,
                    op_a,
                    op_b,
                    m,
                    n,
                    k,
                    &alpha,
                    a as *const f64,
                    i32::try_from(lda)?,
                    b as *const f64,
                    i32::try_from(ldb)?,
                    &beta,
                    c as *mut f64,
                    i32::try_from(ldc)?,
                )
            };
            // SAFETY: handle 由上面那个 `cublasCreate_v2` 建，这里销毁一次。
            unsafe { (cuda.cublas_destroy)(handle) };
            if status != 0 {
                return Err(GpuError::Call { what, code: status });
            }
            Ok(())
        }

        /// 提交一个内核。`args` 里每一项都指向**调用方栈上**的参数副本（`cuLaunchKernel`
        /// 在返回前就把参数表读走了，所以只要它们活到本调用结束即可）。
        fn launch(
            &self,
            function: *mut c_void,
            points: usize,
            args: &mut [*mut c_void],
            what: &'static str,
        ) -> Result<(), GpuError> {
            self.ensure_current()?;
            let grid = u32::try_from(points.div_ceil(BLOCK as usize))
                .map_err(|_| GpuError::TooLarge(points))?;
            // SAFETY: `function` 来自本会话的 `cuModuleGetFunction`；`args` 的每一项都是
            // 上面文档说的栈上参数地址，类型与常量源码里的形参表逐项一致
            // （double*, double*, double*, int, long long）。
            let status = unsafe {
                (self.api.cu_launch_kernel)(
                    function,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if status != 0 {
                return Err(GpuError::Call { what, code: status });
            }
            Ok(())
        }
    }

    fn error_name(api: &DriverApi, code: i32) -> String {
        let mut name: *const i8 = std::ptr::null();
        // SAFETY: `name` 是局部变量，`cuGetErrorName` 只写它、并返回一个静态字符串指针。
        let status = unsafe { (api.cu_get_error_name)(code, &mut name) };
        if status != 0 || name.is_null() {
            return format!("未知错误码 {code}");
        }
        // SAFETY: 成功时 `cuGetErrorName` 保证写入一个以 NUL 结尾的静态字符串。
        unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    }

    fn call_error(api: &DriverApi, what: &'static str, code: i32) -> String {
        format!("{what} 返回 {code}（{}）", error_name(api, code))
    }

    /// 找 `nvrtc64_*.dll`：先把它的 `nvrtc-builtins64_*.dll` 按全路径加载好，
    /// 再加载主库 —— Windows 的依赖查找按**模块基名**命中已加载的模块，这样就不必改动
    /// 进程的 DLL 搜索目录（那需要额外的全局副作用）。
    fn load_nvrtc() -> Result<libloading::Library, String> {
        let majors = ["13", "12", "11"];
        // 候选：`nvrtc64_130_0.dll` 这种「大版本 + 小版本」命名，以及 `nvrtc64_13.dll`。
        let names: Vec<String> = majors
            .iter()
            .flat_map(|major| {
                [
                    format!("nvrtc64_{major}0_0.dll"),
                    format!("nvrtc64_{major}_0.dll"),
                    format!("nvrtc64_{major}.dll"),
                ]
            })
            .chain(std::iter::once("nvrtc64.dll".to_string()))
            .collect();
        let mut tried = Vec::new();
        for dir in super::search_dirs() {
            for name in &names {
                let path = dir.join(name);
                tried.push(path.display().to_string());
                // builtins 必须先加载（见函数文档）。
                for builtins in [
                    name.replace("nvrtc64_", "nvrtc-builtins64_"),
                    "nvrtc-builtins64_13.dll".to_string(),
                    "nvrtc-builtins64_130.dll".to_string(),
                ] {
                    let builtins_path = dir.join(&builtins);
                    if builtins_path.is_file() {
                        // SAFETY: 只加载库；失败无所谓（主库也可能不依赖它）。
                        let _ = unsafe { libloading::Library::new(&builtins_path) };
                        break;
                    }
                }
                // SAFETY: 只加载库；符号在 `compile_ptx` 里按 CUDA 公开 ABI 逐项取。
                if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                    return Ok(lib);
                }
            }
        }
        Err(format!(
            "NVRTC 不可用：找不到 nvrtc64_*.dll（试过 {}）。装 CUDA Toolkit（含 NVRTC）或设 \
             QWB_CUDA_LIB_DIR 指向它的 bin\\x64；这条路径可以整体退回 CPU",
            tried.join(", ")
        ))
    }

    /// 用 NVRTC 编译常量源码，返回 PTX 文本与 NVRTC 版本。
    fn compile_ptx(
        nvrtc: &libloading::Library,
        gpu_arch: i32,
    ) -> Result<((i32, i32), Vec<u8>), String> {
        // SAFETY: 下面的符号名都是 `nvrtc64_*.dll` 的公开导出；取不到就报错。
        let (create, compile, log_size, log, ptx_size, ptx, destroy, nvrtc_version_fn): (
            NvrtcCreateProgram,
            NvrtcCompileProgram,
            NvrtcGetProgramLogSize,
            NvrtcGetProgramLog,
            NvrtcGetPtxSize,
            NvrtcGetPtx,
            NvrtcDestroyProgram,
            unsafe extern "C" fn(*mut i32, *mut i32) -> i32,
        ) = unsafe {
            let missing = |name: &str| format!("NVRTC 符号 {name} 取不到");
            (
                *nvrtc
                    .get::<NvrtcCreateProgram>(b"nvrtcCreateProgram\0")
                    .map_err(|_| missing("nvrtcCreateProgram"))?,
                *nvrtc
                    .get::<NvrtcCompileProgram>(b"nvrtcCompileProgram\0")
                    .map_err(|_| missing("nvrtcCompileProgram"))?,
                *nvrtc
                    .get::<NvrtcGetProgramLogSize>(b"nvrtcGetProgramLogSize\0")
                    .map_err(|_| missing("nvrtcGetProgramLogSize"))?,
                *nvrtc
                    .get::<NvrtcGetProgramLog>(b"nvrtcGetProgramLog\0")
                    .map_err(|_| missing("nvrtcGetProgramLog"))?,
                *nvrtc
                    .get::<NvrtcGetPtxSize>(b"nvrtcGetPTXSize\0")
                    .map_err(|_| missing("nvrtcGetPTXSize"))?,
                *nvrtc
                    .get::<NvrtcGetPtx>(b"nvrtcGetPTX\0")
                    .map_err(|_| missing("nvrtcGetPTX"))?,
                *nvrtc
                    .get::<NvrtcDestroyProgram>(b"nvrtcDestroyProgram\0")
                    .map_err(|_| missing("nvrtcDestroyProgram"))?,
                *nvrtc
                    .get::<unsafe extern "C" fn(*mut i32, *mut i32) -> i32>(b"nvrtcVersion\0")
                    .map_err(|_| missing("nvrtcVersion"))?,
            )
        };

        let (mut major, mut minor) = (0_i32, 0_i32);
        // SAFETY: 两个指针都是本函数内的局部变量，`nvrtcVersion` 只写它们。
        let status = unsafe { nvrtc_version_fn(&mut major, &mut minor) };
        if status != 0 {
            return Err(format!("nvrtcVersion 返回 {status}"));
        }

        let arch = format!("--gpu-architecture=compute_{gpu_arch}");
        let source =
            std::ffi::CString::new(XC_SOURCE).map_err(|e| format!("内核源码里有 NUL：{e}"))?;
        let fname = c"qwb_xc.cu";
        let arch_c = std::ffi::CString::new(arch.clone()).map_err(|e| format!("{e}"))?;
        let std_c = c"--std=c++14";
        let options = [arch_c.as_ptr(), std_c.as_ptr()];

        let mut program: *mut c_void = std::ptr::null_mut();
        // SAFETY: 源码/文件名/选项都是本函数内有效的 NUL 结尾 C 串，生命周期覆盖下面所有调用；
        // `program` 由 NVRTC 写出，并在函数返回前被 `nvrtcDestroyProgram` 销毁一次。
        let image = unsafe {
            let status = create(
                &mut program,
                source.as_ptr(),
                fname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if status != 0 {
                return Err(format!("nvrtcCreateProgram 返回 {status}"));
            }
            let cleanup = |program: &mut *mut c_void, message: String| -> String {
                destroy(program);
                *program = std::ptr::null_mut();
                message
            };
            let status = compile(program, options.len() as i32, options.as_ptr());
            if status != 0 {
                let mut length = 0_usize;
                log_size(program, &mut length);
                let mut buffer = vec![0_i8; length.max(1)];
                log(program, buffer.as_mut_ptr());
                let text = std::ffi::CStr::from_ptr(buffer.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(cleanup(
                    &mut program,
                    format!(
                        "nvrtcCompileProgram 失败（{arch}，返回 {status}）：{}",
                        text.trim()
                    ),
                ));
            }
            let mut length = 0_usize;
            let status = ptx_size(program, &mut length);
            if status != 0 {
                return Err(cleanup(
                    &mut program,
                    format!("nvrtcGetPTXSize 返回 {status}"),
                ));
            }
            // PTX 是文本，NVRTC 保证末尾有 NUL；多留一个字节给驱动读取。
            let mut buffer = vec![0_i8; length + 1];
            let status = ptx(program, buffer.as_mut_ptr());
            if status != 0 {
                return Err(cleanup(&mut program, format!("nvrtcGetPTX 返回 {status}")));
            }
            destroy(&mut program);
            let bytes: Vec<u8> = buffer.into_iter().map(|byte| byte as u8).collect();
            bytes
        };
        Ok(((major, minor), image))
    }
}
