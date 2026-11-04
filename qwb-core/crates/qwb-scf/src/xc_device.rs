//! EX-03：网格 XC 收缩的**设备后端注入点**（`qwb-scf` 侧）。
//!
//! # 为什么不直接 `qwb-scf → qwb-gpu`
//!
//! `qwb-gpu` 的 `[dev-dependencies]` 里有 `qwb-scf`（它的 `tests/ex03_*.rs` 拿产品 CPU 路径
//! 当唯一基准）。再让 `qwb-scf` 可选依赖 `qwb-gpu` 就在包图上形成
//! `qwb-scf → qwb-gpu → qwb-scf` 的环 —— cargo 能接受（环经过 dev 边），但 `project.md` §3.7
//! 写的是「**workspace 层级不得引入循环依赖**」，那条规矩说的是图，不是 cargo 的容忍度。
//!
//! 所以方向反过来：**高层知道接口，低层知道实现**。
//!
//! * `qwb-scf` 只定义这里的 trait 与两个纯数据类型（`XcGridRequest` / `XcGridDeviceOutcome`），
//!   全程**不引用** `qwb-gpu`，任何构建（含 `--all-features`）都没有 `qwb-gpu` 这个依赖边；
//! * `qwb-gpu` 在 `gpu-kernel` feature 下依赖 `qwb-scf` 并实现 [`XcGridAccelerator`]；
//! * `qwb-engine`（产品集成层，本来就依赖 `qwb-scf`）在 `gpu-kernel` feature 下把实现
//!   **注入** 进 [`crate::rks::RksOptions`]。
//!
//! 层序因此是 DAG：`qwb-scf ← qwb-gpu ← {qwb-response, qwb-engine}`，没有环。
//!
//! # 这个接口的契约
//!
//! 数据是**行主序扁平切片**（与 `qwb-gpu::xc` 的核契约同形），刻意不传 `Grid`/`Matrix`：
//! 设备侧只需要 `Φ`、`∇Φ`、`P`、`w` 四张数，传领域类型只会把「谁的求和顺序」这件事弄模糊。
//!
//! 设备不可用（没 GPU / 没 NVRTC / 显存不够 / feature 没开）时实现必须返回
//! [`XcGridDeviceOutcome::Unavailable`]，**不许 panic、不许静默给出少一项的数**：
//! 调用方据此退回产品 CPU 路径，并把这个原因记下来。

use std::sync::Arc;

/// 网格 XC 收缩走哪条路。
///
/// 默认 [`XcGridPath::Cpu`] —— 产品路径、位级锚点（`tests/xc_grid_bit_pattern.rs`）钉住的
/// 那条。选 [`XcGridPath::Gpu`] 但没有注入后端时，SCF 会退回 CPU 并把原因写进
/// `RksResult::xc_fallback_reason`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XcGridPath {
    /// CPU 逐点收缩（产品默认）。
    #[default]
    Cpu,
    /// 设备收缩（需要注入 [`XcGridAccelerator`]）。
    Gpu,
}

/// 一次网格 XC 收缩的请求。
#[derive(Debug, Clone, Copy)]
pub struct XcGridRequest<'a> {
    /// **驻留键**：同一份 `Φ`/`∇Φ`（同一网格 + 同一基组）在 SCF 各步之间保持不变。
    ///
    /// 后端用它判断「设备上的那几张矩阵还是不是这一批数」——键不同就重传，
    /// 所以键的语义只要「变了就得重传」即可，不需要全局唯一（本仓库用单调计数器，
    /// 见 `rks::XcGridFlatAo::build`）。
    pub residency: u64,
    /// 基函数个数。
    pub n: usize,
    /// 网格点数。
    pub n_points: usize,
    /// `Φ[p·n + μ]`（`n_points × n`，行主序）。
    pub ao_values: &'a [f64],
    /// `∂Φ/∂x`、`∂Φ/∂y`、`∂Φ/∂z`（每个 `n_points × n` 行主序）。LDA 通道为 `None`。
    pub ao_gradients: Option<[&'a [f64]; 3]>,
    /// `P[μ·n + ν]`（`n × n`，行主序）。**每步都变**，所以每步都要上传（`n²` 个双精度）。
    pub density: &'a [f64],
    /// 每个点的求积权重 `w_p`。
    pub weights: &'a [f64],
}

/// 设备路径的分项记账（逐次累加；`RksResult::xc_device_report` 是整个 SCF 的总和）。
///
/// 四项时间是**互相排斥**的阶段计时，加起来约等于设备路径的墙钟（不含 host 上的泛函求值
/// 与 `E_xc` 累加，那两段与 CPU 路径相同，不属于设备开销）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct XcGridDeviceReport {
    /// 收缩次数。
    pub calls: u32,
    /// 其中发生了 `Φ`/`∇Φ` 上传的次数（驻留生效时只有第一次）。
    pub uploads: u32,
    /// 上传字节数（`Φ`/`∇Φ`/`P`/逐点系数）。
    pub uploaded_bytes: u64,
    /// 下载字节数（`ρ`/`∇ρ`/`σ`/`v_xc`）。
    pub downloaded_bytes: u64,
    /// 上传阶段（含设备缓冲的分配/释放）。
    pub upload_seconds: f64,
    /// 手写核的发射 + 同步阶段。
    pub kernel_seconds: f64,
    /// cuBLAS gemm 阶段。
    pub gemm_seconds: f64,
    /// 下载阶段。
    pub download_seconds: f64,
}

impl XcGridDeviceReport {
    /// 把一次调用的记账并进来。
    pub fn accumulate(&mut self, other: &Self) {
        self.calls = self.calls.saturating_add(other.calls);
        self.uploads = self.uploads.saturating_add(other.uploads);
        self.uploaded_bytes = self.uploaded_bytes.saturating_add(other.uploaded_bytes);
        self.downloaded_bytes = self.downloaded_bytes.saturating_add(other.downloaded_bytes);
        self.upload_seconds += other.upload_seconds;
        self.kernel_seconds += other.kernel_seconds;
        self.gemm_seconds += other.gemm_seconds;
        self.download_seconds += other.download_seconds;
    }

    /// 设备路径的总墙钟（四个阶段之和），秒。
    #[must_use]
    pub fn device_seconds(&self) -> f64 {
        self.upload_seconds + self.kernel_seconds + self.gemm_seconds + self.download_seconds
    }
}

/// 一次设备收缩的结果。
#[derive(Debug, Clone)]
pub enum XcGridDeviceOutcome {
    /// 设备路径完成。
    Done {
        /// `E_xc = Σ_p w_p ρ_p ε_p`（Hartree）。
        energy: f64,
        /// `v_xc[μ·n + ν]`（`n × n`，行主序）。
        v_xc: Vec<f64>,
        /// 这一次的分项记账。
        report: XcGridDeviceReport,
    },
    /// 设备不可用（含原因）；调用方退回 CPU 逐点路径。
    Unavailable {
        /// 为什么退回（要能读、能进日志）。
        reason: String,
    },
}

/// LDA 通道的泛函求值回调：`ρ_p` → `(ε_p, ε + ρ·dε/dρ)`。
///
/// 第二个返回值是**合成后**的势（不是 `dε/dρ`），与 `qwb-gpu::xc::xc_grid_lda` 的契约一致。
pub type XcLdaEvaluate<'a> = dyn FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String> + 'a;

/// GGA 通道的泛函求值回调：`(ρ_p, σ_p)` → `(ε_p, dε/dρ, dε/dσ)`。
pub type XcGgaEvaluate<'a> =
    dyn FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> + 'a;

/// 网格 XC 收缩的设备后端。
///
/// 两条方法对应两条通道：只要 `ρ` 的走 [`Self::contract_lda`]，要 `ρ` 与 `σ` 的走
/// [`Self::contract_gga`]。需要 `∇²ρ` 或 `τ` 的泛函**不要**送进来（由 `qwb-scf` 侧先拒）。
pub trait XcGridAccelerator: std::fmt::Debug + Send + Sync {
    /// LDA 通道：`evaluate(ρ) → (ε, ε + ρ·dε/dρ)`。
    fn contract_lda(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcLdaEvaluate<'_>,
    ) -> XcGridDeviceOutcome;

    /// GGA 通道：`evaluate(ρ, σ) → (ε, dε/dρ, dε/dσ)`。
    fn contract_gga(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcGgaEvaluate<'_>,
    ) -> XcGridDeviceOutcome;

    /// 供日志与错误消息用的名字。
    fn name(&self) -> &'static str;
}

/// [`XcGridAccelerator`] 的句柄（`RksOptions` 需要 `Clone + PartialEq + Debug`）。
///
/// `PartialEq` 比的是句柄本身（`Arc` 指针），与 [`crate::rks::ExternalFieldHandle`] 同一套
/// 理由：后端没有可比较的语义。
#[derive(Clone)]
pub struct XcGridBackendHandle(Arc<dyn XcGridAccelerator>);

impl XcGridBackendHandle {
    /// 包一个后端实现。
    #[must_use]
    pub fn new(backend: Arc<dyn XcGridAccelerator>) -> Self {
        Self(backend)
    }

    /// 取出内部实现。
    #[must_use]
    pub fn backend(&self) -> &dyn XcGridAccelerator {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for XcGridBackendHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "XcGridBackend({})", self.0.name())
    }
}

impl PartialEq for XcGridBackendHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
