//! EX-03 第四块：**DFT 格点 XC 的收缩段**（feature `gpu-kernel` 才有 GPU 路径）。
//!
//! # 这一段为什么值得单独做
//!
//! `qwb-scf` 的 `evaluate_xc_cached` 把网格 XC 拆成三段，实测收缩段占每步 85–96 %：
//!
//! ```text
//! ρ_p      = Σ_μν P_μν Φ_pμ Φ_pν           # (ΦP) 一次 gemm + 一次**行点积**
//! ε_p, v_p = 泛函(ρ_p)                      # 逐点超越函数，留在 host
//! v_xc[μν] = Σ_p w_p v_p Φ_pμ Φ_pν          # = Φᵀ·(w·v·Φ)，又是一次 gemm
//! E_xc     = Σ_p w_p ρ_p ε_p
//! ```
//!
//! 中间那一步 `ρ_p = Σ_μ Φ_pμ (ΦP)_pμ` 是**两张同形矩阵按行点积压成一个向量**：BLAS 给不出
//! 这个形状（`gemm` 只能算 `ΦP`），稠密收缩的算术强度 ≈ `n/4` flop/byte（n=24 时 6 flop/byte）
//! 即**访存受限**。所以它是本次交付里唯一手写的 GPU 核（见 `jit.rs` 的 `XC_SOURCE`）。
//!
//! # 分界线：什么在设备上、什么在 host 上
//!
//! * 设备上：`(ΦP)` 的 gemm、行点积、按行缩比、`v_xc` 的 gemm（都是 `O(n_points·n²)`）；
//! * host 上：泛函求值（`FunctionalTerm` 是 Rust trait object，不能搬进 CUDA 源）与
//!   `E_xc` 的 `O(n_points)` 累加。所以每个 SCF 步有一次 `ρ` 上行、一次系数下行 ——
//!   3 万点的 `ρ` 是 240 KB，量级上可以忽略。
//!
//! # 两条通道
//!
//! * [`xc_grid_lda`]：只要 `ρ`（`requires.sigma == false` 的泛函）。`ρ` 走 gemm + 行点积。
//! * [`xc_grid_gga`]：要 `ρ` 与 `σ = |∇ρ|²`。多三张 `∇Φ` 的收缩（`∇ρ_p` 的三个分量），
//!   并且 `v_xc` 的装配多一项 `σ` 通道（`2ρ·dε/dσ` 与 `∇φ_μ∇φ_ν` 的收缩）。
//!
//! 两条**共用**同一套 JIT/launch 机器与同一次 NVRTC 编译（见 `jit.rs` 的 `XC_SOURCE`）；
//! GGA 没有另起炉灶。`∇²ρ` 与 `τ` 通道**没有**实现：[`xc_grid_gga`] 的文档里写明调用方
//! 必须自己保证泛函项不要那两个量。
//!
//! # 回退
//!
//! 没有 GPU / 没有 NVRTC / 驱动报错 / 显存预算不够 / `gpu-kernel` 没开 / 调用方要求走 CPU：
//! 一律走 [`cpu_contract`]（LDA）或 [`cpu_contract_gga`]（GGA），它们的求和顺序与
//! `qwb-scf` 的 `evaluate_xc_cached` **逐句一致**，所以退回后的 `E_xc` 与 `v_xc` 与产品
//! CPU 路径**逐位相同**（`tests/ex03_xc_fallback.rs` 钉住，LDA 与 GGA 各一条）。

#[cfg(feature = "gpu-kernel")]
use std::sync::OnceLock;

use crate::GpuError;

/// 一次收缩实际走了哪条路。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcPath {
    /// 设备路径（NVRTC 编译的核 + cuBLAS）。
    Gpu,
    /// CPU 基准路径（与产品路径逐位相同）。
    Cpu,
}

/// 网格 XC 收缩的输入。全部是**行主序扁平切片**，刻意不依赖 `qwb-grid` / `qwb-xc` 的类型，
/// 这样核本身与网格/泛函实现解耦。
#[derive(Debug, Clone, Copy)]
pub struct XcGridInputs<'a> {
    /// 基函数个数。
    pub n: usize,
    /// 网格点数。
    pub n_points: usize,
    /// AO 值 `Φ[p·n + μ]`（`n_points × n`，行主序）。
    pub ao_values: &'a [f64],
    /// 密度矩阵 `P[μ·n + ν]`（`n × n`，行主序）。
    pub density: &'a [f64],
    /// 每个点的求积权重 `w_p`。
    pub weights: &'a [f64],
}

/// 跑法选项（**注入式**开关：A3 的"禁用 GPU 路径"不靠改环境变量，靠这里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct XcGridOptions {
    /// 强制走 CPU（与 `QWB_GPU_XC_DISABLE=1` 等价，但可复现、可并行测试）。
    pub force_cpu: bool,
    /// 显式显存预算（`None` = 空闲显存的 75 %）。
    pub budget_bytes: Option<u64>,
}

/// 一次收缩的结果。
#[derive(Debug, Clone)]
pub struct XcGridOutput {
    /// `E_xc = Σ_p w_p ρ_p ε_p`，Hartree。
    pub energy: f64,
    /// `v_xc[μ·n + ν]`（`n × n`，行主序）。
    pub v_xc: Vec<f64>,
    /// 实际走的路径。
    pub path: XcPath,
    /// 退回 CPU 的原因（`path == Cpu` 时总是有内容）。
    pub fallback_reason: Option<String>,
}

/// 设备路径的显存需求（字节）：`Φ`、`P`、`(ΦP)`、`ρ`、系数、`v_xc` 六块。
///
/// 与 `gemm_bytes` 一样，**先按这个数判预算、不试分配** —— 试分配的失败点可能在第四块，
/// 那时前三块已经分出去了。
#[must_use]
pub fn xc_bytes(n_points: usize, n: usize) -> u64 {
    (2 * n_points * n + 2 * n * n + 2 * n_points) as u64 * 8
}

/// 是否被运维开关关掉了（`QWB_GPU_XC_DISABLE` 非空即关）。
fn disabled_by_env() -> bool {
    std::env::var_os("QWB_GPU_XC_DISABLE").is_some_and(|value| !value.is_empty())
}

#[cfg(feature = "gpu-kernel")]
fn default_budget() -> Option<u64> {
    static BUDGET: OnceLock<Option<u64>> = OnceLock::new();
    *BUDGET.get_or_init(|| crate::memory_budget_bytes().ok())
}

fn validate(inputs: &XcGridInputs<'_>) -> Result<(), String> {
    let XcGridInputs {
        n,
        n_points,
        ao_values,
        density,
        weights,
    } = *inputs;
    if n == 0 || n_points == 0 {
        return Err(format!("形状不合法：n={n}, n_points={n_points}"));
    }
    if ao_values.len() != n_points * n {
        return Err(format!(
            "形状不合法：ao_values.len()={}，应为 n_points·n = {}",
            ao_values.len(),
            n_points * n
        ));
    }
    if density.len() != n * n {
        return Err(format!(
            "形状不合法：density.len()={}，应为 n² = {}",
            density.len(),
            n * n
        ));
    }
    if weights.len() != n_points {
        return Err(format!(
            "形状不合法：weights.len()={}，应为 n_points = {n_points}",
            weights.len()
        ));
    }
    Ok(())
}

/// **CPU 基准收缩**：与 `qwb-scf` 的 `evaluate_xc_cached` 的 LDA 分支逐句同序。
///
/// 这里的求和顺序是契约（"CPU 位模式不得改变"）：外层 `i`、内层 `j`，`v_i · P_ij · v_j`；
/// `v_xc` 外层 `i`、内层 `j`，`w_p · (v_i · v_j · pot)`；`E_xc` 是 `w_p · ρ_p · ε_p`。
/// 改成 gemm / 改成累加别的顺序都能"看起来等价"，但会改变最后几位 —— 所以这里不优化。
///
/// # Errors
///
/// `ρ` 非有限（网格坐标或 AO 值坏了）、泛函求值失败、`v_xc` 值非有限。
#[allow(clippy::needless_range_loop)] // 两个下标各有含义（AO 对 × 点），range 最接近被复刻的那段代码
pub fn cpu_contract<F>(inputs: &XcGridInputs<'_>, evaluate: &mut F) -> Result<XcGridOutput, String>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String>,
{
    validate(inputs)?;
    let XcGridInputs {
        n,
        n_points,
        ao_values,
        density,
        weights,
    } = *inputs;

    let mut rho_values = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let row = &ao_values[p * n..(p + 1) * n];
        let mut rho = 0.0_f64;
        for i in 0..n {
            let v_i = row[i];
            let base = i * n;
            for j in 0..n {
                rho += v_i * density[base + j] * row[j];
            }
        }
        if !rho.is_finite() {
            return Err(format!(
                "density is not finite ({rho}) at grid point {p}; grid coordinates or AO \
                 values are corrupted"
            ));
        }
        rho_values.push(rho.max(0.0));
    }

    let (epsilon, potential) = evaluate(&rho_values)?;
    if epsilon.len() != n_points {
        return Err(format!(
            "泛函返回 {} 个 ε，应为 n_points = {n_points}",
            epsilon.len()
        ));
    }
    if potential.len() != n_points {
        return Err(format!(
            "泛函返回 {} 个 v，应为 n_points = {n_points}",
            potential.len()
        ));
    }

    let mut energy = 0.0_f64;
    let mut v_xc = vec![0.0_f64; n * n];
    for p in 0..n_points {
        let rho = rho_values[p];
        energy += weights[p] * rho * epsilon[p];
        let potential_here = potential[p];
        if !potential_here.is_finite() {
            return Err(format!(
                "XC 势在网格点 {p} 非有限（{potential_here:.3e}），ρ = {rho:.3e}"
            ));
        }
        let row = &ao_values[p * n..(p + 1) * n];
        for i in 0..n {
            if row[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                let element = row[i] * row[j] * potential_here;
                v_xc[i * n + j] += weights[p] * element;
            }
        }
    }
    Ok(XcGridOutput {
        energy,
        v_xc,
        path: XcPath::Cpu,
        fallback_reason: None,
    })
}

/// 设备路径的失败。分成两类，因为**只有第一类能退回 CPU**：
/// 第二类是泛函/输入本身坏了，退回 CPU 会得到同一个错误，所以直接上报。
enum Attempt {
    /// GPU 侧不可用（无库 / 无 NVRTC / 预算不够 / 驱动报错）。
    Device(GpuError),
    /// 输入或泛函出错。
    #[cfg_attr(not(feature = "gpu-kernel"), allow(dead_code))]
    // 无 feature 时设备路径整段不编译
    Input(String),
}

/// 网格 XC 收缩：能上 GPU 就上，任何设备侧问题都退回 [`cpu_contract`]。
///
/// `evaluate` 把 `ρ_p` 列表变成 `(ε_p, v_p)`：`v_p` 是 `ε + ρ·dε/dρ`（ρ=0 时取 `ε`），
/// 也就是 CPU 路径里那个 `potential`。泛函留在 host 上跑，所以核不需要知道泛函长什么样。
///
/// 只覆盖 **LDA 通道**：需要 `σ` / `∇²ρ` / `τ` 的泛函（GGA、meta-GGA）请直接走
/// [`cpu_contract`] —— 那三个通道的 `v_xc` 项要多几张 `∇Φ`/`∇²Φ` 的收缩，目前没有实现，
/// 也不假装实现。
///
/// # Errors
///
/// 形状不合法、`ρ` 或 `v_xc` 非有限、泛函求值失败 —— 与 [`cpu_contract`] 相同。
pub fn xc_grid_lda<F>(
    inputs: &XcGridInputs<'_>,
    options: XcGridOptions,
    mut evaluate: F,
) -> Result<XcGridOutput, String>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String>,
{
    validate(inputs)?;
    let mut reason = if options.force_cpu {
        Some("调用方要求走 CPU（XcGridOptions::force_cpu）".to_string())
    } else if disabled_by_env() {
        Some("QWB_GPU_XC_DISABLE 已设置".to_string())
    } else {
        match crate::xc::gpu_contract(inputs, options, &mut evaluate) {
            Ok(output) => return Ok(output),
            Err(Attempt::Device(error)) => Some(format!("{error}")),
            Err(Attempt::Input(why)) => return Err(why),
        }
    };
    let mut output = cpu_contract(inputs, &mut evaluate)?;
    output.fallback_reason = reason.take();
    Ok(output)
}

#[cfg(feature = "gpu-kernel")]
fn gpu_contract<F>(
    inputs: &XcGridInputs<'_>,
    options: XcGridOptions,
    evaluate: &mut F,
) -> Result<XcGridOutput, Attempt>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String>,
{
    use crate::jit::kernel::XcSession;

    let XcGridInputs {
        n,
        n_points,
        ao_values,
        density,
        weights,
    } = *inputs;

    let budget = match options.budget_bytes {
        Some(bytes) => bytes,
        None => default_budget().ok_or_else(|| {
            Attempt::Device(GpuError::Unavailable(
                "取不到显存预算（没有 CUDA 设备）".to_string(),
            ))
        })?,
    };
    let required = xc_bytes(n_points, n);
    if required > budget {
        return Err(Attempt::Device(GpuError::InsufficientMemory {
            required,
            available: budget,
        }));
    }

    let session = XcSession::get().map_err(Attempt::Device)?;
    let phi = session.upload(ao_values).map_err(Attempt::Device)?;
    let p_matrix = session.upload(density).map_err(Attempt::Device)?;
    let xm = session.alloc(n_points * n).map_err(Attempt::Device)?;
    let rho_device = session.alloc(n_points).map_err(Attempt::Device)?;
    let v_device = session.alloc(n * n).map_err(Attempt::Device)?;
    session
        .gemm_xm(&xm, &phi, &p_matrix, n, n_points)
        .map_err(Attempt::Device)?;
    session
        .row_dot(&xm, &phi, &rho_device, n, n_points)
        .map_err(Attempt::Device)?;
    session.synchronize().map_err(Attempt::Device)?;
    let raw_rho = session
        .download(&rho_device, n_points)
        .map_err(Attempt::Device)?;

    let mut rho_values = Vec::with_capacity(n_points);
    for (p, rho) in raw_rho.iter().enumerate() {
        if !rho.is_finite() {
            return Err(Attempt::Input(format!(
                "density is not finite ({rho}) at grid point {p}; grid coordinates or AO \
                 values are corrupted"
            )));
        }
        rho_values.push(rho.max(0.0));
    }

    let (epsilon, potential) = evaluate(&rho_values).map_err(Attempt::Input)?;
    if epsilon.len() != n_points || potential.len() != n_points {
        return Err(Attempt::Input(format!(
            "泛函返回 {} 个 ε / {} 个 v，应为 n_points = {n_points}",
            epsilon.len(),
            potential.len()
        )));
    }

    // 装配：a_p = w_p · v_p，然后 v_xc = Φᵀ·(a·Φ)。
    let mut scale = Vec::with_capacity(n_points);
    let mut energy = 0.0_f64;
    for p in 0..n_points {
        let potential_here = potential[p];
        if !potential_here.is_finite() {
            return Err(Attempt::Input(format!(
                "XC 势在网格点 {p} 非有限（{potential_here:.3e}），ρ = {:.3e}",
                rho_values[p]
            )));
        }
        scale.push(weights[p] * potential_here);
        // 与 CPU 路径同一个求和顺序，所以 E_xc 的差只来自 ρ 的结合顺序。
        energy += weights[p] * rho_values[p] * epsilon[p];
    }
    let scale_device = session.upload(&scale).map_err(Attempt::Device)?;
    // `xm` 已经用完（ρ 出来了），按行缩比直接写回它，省一块 `n_points·n` 缓冲。
    session
        .scale_rows(&phi, &scale_device, &xm, n, n_points)
        .map_err(Attempt::Device)?;
    session
        .gemm_v(&v_device, &phi, &xm, n, n_points)
        .map_err(Attempt::Device)?;
    session.synchronize().map_err(Attempt::Device)?;
    let v_xc = session
        .download(&v_device, n * n)
        .map_err(Attempt::Device)?;
    if !v_xc.iter().all(|value| value.is_finite()) {
        return Err(Attempt::Input(
            "v_xc 里出现非有限值（设备内核的结果不可信）".to_string(),
        ));
    }

    Ok(XcGridOutput {
        energy,
        v_xc,
        path: XcPath::Gpu,
        fallback_reason: None,
    })
}

#[cfg(not(feature = "gpu-kernel"))]
fn gpu_contract<F>(
    _inputs: &XcGridInputs<'_>,
    _options: XcGridOptions,
    _evaluate: &mut F,
) -> Result<XcGridOutput, Attempt>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String>,
{
    Err(Attempt::Device(GpuError::Unavailable(
        "feature `gpu-kernel` 未启用（编译期就没有 NVRTC/驱动层）".to_string(),
    )))
}

// ---------------------------------------------------------------------------------------
// GGA 通道：ρ、∇ρ、σ 三样一起，`v_xc` 多一项 σ 通道
// ---------------------------------------------------------------------------------------

/// GGA 收缩的输入：在 [`XcGridInputs`] 之上多三张 `∇Φ`（笛卡尔分量 x、y、z 各一张）。
#[derive(Debug, Clone, Copy)]
pub struct XcGridGgaInputs<'a> {
    /// 基函数个数。
    pub n: usize,
    /// 网格点数。
    pub n_points: usize,
    /// AO 值 `Φ[p·n + μ]`（`n_points × n`，行主序）。
    pub ao_values: &'a [f64],
    /// AO 梯度 `∂Φ/∂x`、`∂Φ/∂y`、`∂Φ/∂z`，每个都是 `n_points × n` 行主序。
    ///
    /// 三个分量分开传（而不是交错成 `n_points × 3n`）：设备核按轴取行、`gemm` 按轴吃掉
    /// 一整张矩阵，交错布局只会让两边都多一层跨步。
    pub ao_gradients: [&'a [f64]; 3],
    /// 密度矩阵 `P[μ·n + ν]`（`n × n`，行主序）。
    pub density: &'a [f64],
    /// 每个点的求积权重 `w_p`。
    pub weights: &'a [f64],
}

/// GGA 收缩的结果。
#[derive(Debug, Clone)]
pub struct XcGridGgaOutput {
    /// `E_xc = Σ_p w_p ρ_p ε_p`，Hartree。
    pub energy: f64,
    /// `v_xc[μ·n + ν]`（`n × n`，行主序）。
    pub v_xc: Vec<f64>,
    /// `∇ρ_p = (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)`，按点排列（诊断/验收用：轴序错了 `σ` 看不出来，
    /// 但 `v_xc` 会错，所以把它交出来让测试逐轴比）。
    pub gradient_rho: Vec<[f64; 3]>,
    /// `σ_p = |∇ρ_p|²`，与 `qwb-scf` 的 `sigma_values` 同一个定义（不做截断）。
    pub sigma: Vec<f64>,
    /// 截断到非负之后的 `ρ_p = max(Σ_μν P_μν Φ_μ Φ_ν, 0)`。
    ///
    /// 交出来是为了让验收能**直接数**设备与 CPU 的 `ρ` 有几位不同（LDA 那条路径把 `ρ`
    /// 留在核里、没有出口，所以那时只能看能量）。
    pub rho: Vec<f64>,
    /// 实际走的路径。
    pub path: XcPath,
    /// 退回 CPU 的原因（`path == Cpu` 时总是有内容）。
    pub fallback_reason: Option<String>,
}

/// 设备路径的显存需求（字节）。
///
/// 逐项：`Φ`、`∇Φ`×3、按行缩比的临时块 = `5·n_points·n`；`P`、三个 `∇Φ` 通道的
/// `M_a = ∇Φᵀ_a·(s_a∘Φ)`、`v_xc` = `5·n²`；`ρ`、`σ`、`∇ρ`×3、`w·v`、`w·σ 系数` = `7·n_points`。
///
/// 与 [`xc_bytes`] 同样**先按这个数判预算、不试分配**。
#[must_use]
pub fn xc_gga_bytes(n_points: usize, n: usize) -> u64 {
    (5 * n_points * n + 5 * n * n + 7 * n_points) as u64 * 8
}

fn validate_gga(inputs: &XcGridGgaInputs<'_>) -> Result<(), String> {
    let XcGridGgaInputs {
        n,
        n_points,
        ao_values,
        ao_gradients,
        density,
        weights,
    } = *inputs;
    if n == 0 || n_points == 0 {
        return Err(format!("形状不合法：n={n}, n_points={n_points}"));
    }
    if ao_values.len() != n_points * n {
        return Err(format!(
            "形状不合法：ao_values.len()={}，应为 n_points·n = {}",
            ao_values.len(),
            n_points * n
        ));
    }
    for (axis, gradient) in ["x", "y", "z"].iter().zip(ao_gradients.iter()) {
        if gradient.len() != n_points * n {
            return Err(format!(
                "形状不合法：ao_gradients[{axis}].len()={}，应为 n_points·n = {}",
                gradient.len(),
                n_points * n
            ));
        }
    }
    if density.len() != n * n {
        return Err(format!(
            "形状不合法：density.len()={}，应为 n² = {}",
            density.len(),
            n * n
        ));
    }
    if weights.len() != n_points {
        return Err(format!(
            "形状不合法：weights.len()={}，应为 n_points = {n_points}",
            weights.len()
        ));
    }
    Ok(())
}

/// **GGA 的 CPU 基准收缩**：与 `qwb-scf` 的 `evaluate_xc_cached` 的 GGA 分支逐句同序。
///
/// 复刻的三段（每一段的求和次序都是契约，不是一个实现细节）：
///
/// 1. `ρ_p = Σ_i Σ_j Φ_i P_ij Φ_j`（`i` 外层、`j` 内层，**不**跳过零 `P`）；
/// 2. `∇ρ_p = Σ_i Σ_j P_ij (∇Φ_i Φ_j + Φ_i ∇Φ_j)`（同上，但**跳过** `P_ij == 0`），
///    `σ_p = ∇ρ_x² + ∇ρ_y² + ∇ρ_z²`（按 x、y、z 的次序加）；
/// 3. 装配 `v_xc`：`w_p·[Φ_i Φ_j (ε + ρ dε/dρ) + 2ρ(dε/dσ)·∇ρ·∇(Φ_i Φ_j)]`，
///    外层跳过 `Φ_i == 0` 的点。
///
/// `evaluate` 收 `(ρ_p, σ_p)`，给 `(ε_p, dε/dρ, dε/dσ)`：把 `ε + ρ dε/dρ` 与 `2ρ dε/dσ`
/// 这两种组合留在这里算（而不是留给泛函），是因为**它们的组装方式**也是要复刻的东西。
///
/// # Errors
///
/// `ρ`/`∇ρ` 非有限、泛函求值失败、`v_xc` 系数非有限 —— 与 `qwb-scf` 的报错一一对应。
#[allow(clippy::needless_range_loop)] // 两个下标各有含义（AO 对 × 点），range 最接近被复刻的那段代码
pub fn cpu_contract_gga<F>(
    inputs: &XcGridGgaInputs<'_>,
    evaluate: &mut F,
) -> Result<XcGridGgaOutput, String>
where
    F: FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
{
    validate_gga(inputs)?;
    let XcGridGgaInputs {
        n,
        n_points,
        ao_values,
        ao_gradients,
        density,
        weights,
    } = *inputs;
    let [gx, gy, gz] = ao_gradients;

    let mut rho_values = Vec::with_capacity(n_points);
    let mut sigma_values = Vec::with_capacity(n_points);
    let mut gradient_rho: Vec<[f64; 3]> = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let row = &ao_values[p * n..(p + 1) * n];
        let gx_row = &gx[p * n..(p + 1) * n];
        let gy_row = &gy[p * n..(p + 1) * n];
        let gz_row = &gz[p * n..(p + 1) * n];
        let mut rho = 0.0_f64;
        for i in 0..n {
            let v_i = row[i];
            let base = i * n;
            for j in 0..n {
                rho += v_i * density[base + j] * row[j];
            }
        }
        if !rho.is_finite() {
            return Err(format!(
                "density is not finite ({rho}) at grid point {p}; grid coordinates or AO \
                 values are corrupted"
            ));
        }
        let mut gradient_here = [0.0_f64; 3];
        for i in 0..n {
            let g_i = [gx_row[i], gy_row[i], gz_row[i]];
            let v_i = row[i];
            let base = i * n;
            for j in 0..n {
                let p_ij = density[base + j];
                if p_ij == 0.0 {
                    continue;
                }
                let v_j = row[j];
                let g_j = [gx_row[j], gy_row[j], gz_row[j]];
                for axis in 0..3 {
                    gradient_here[axis] += p_ij * (g_i[axis] * v_j + v_i * g_j[axis]);
                }
            }
        }
        if !gradient_here.iter().all(|value| value.is_finite()) {
            return Err("density gradient is not finite at a grid point".to_string());
        }
        rho_values.push(rho.max(0.0));
        sigma_values.push(
            gradient_here[0] * gradient_here[0]
                + gradient_here[1] * gradient_here[1]
                + gradient_here[2] * gradient_here[2],
        );
        gradient_rho.push(gradient_here);
    }

    let (energy, potential, sigma_coefficient) =
        gga_evaluate_and_coefficients(&rho_values, &sigma_values, weights, evaluate)?;
    let v_xc = gga_assemble_vxc(inputs, &gradient_rho, &potential, &sigma_coefficient);
    Ok(XcGridGgaOutput {
        energy,
        v_xc,
        gradient_rho,
        sigma: sigma_values,
        rho: rho_values,
        path: XcPath::Cpu,
        fallback_reason: None,
    })
}

/// 泛函求值 + `E_xc` 累加 + 逐点系数：CPU 与设备两条路径**共用**这一段。
///
/// 共用是刻意的：`energy` 的累加序、`potential = ε + ρ dε/dρ`、`2ρ dε/dσ` 这几条组合只有
/// 一份代码，两条路径不可能在这上面走偏（`ρ`/`σ` 从哪来才是路径之间的差别）。
///
/// 返回 `(E_xc, potential_p, sigma_coefficient_p)`；后两个就是 `v_xc` 装配要的两个逐点系数。
#[allow(clippy::type_complexity)]
fn gga_evaluate_and_coefficients<F>(
    rho_values: &[f64],
    sigma_values: &[f64],
    weights: &[f64],
    evaluate: &mut F,
) -> Result<(f64, Vec<f64>, Vec<f64>), String>
where
    F: FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
{
    let n_points = rho_values.len();
    let (epsilon, d_rho, d_sigma) = evaluate(rho_values, sigma_values)?;
    for (got, what) in [
        (epsilon.len(), "ε"),
        (d_rho.len(), "dε/dρ"),
        (d_sigma.len(), "dε/dσ"),
    ] {
        if got != n_points {
            return Err(format!(
                "泛函返回 {got} 个 {what}，应为 n_points = {n_points}"
            ));
        }
    }

    let mut energy = 0.0_f64;
    let mut potential = Vec::with_capacity(n_points);
    let mut sigma_coefficient = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let rho = rho_values[p];
        energy += weights[p] * rho * epsilon[p];
        // 与 `qwb-scf` 一模一样的两条组合（ρ=0 时不乘 ρ，0×∞ 会变 NaN）。
        let potential_here = if rho > 0.0 {
            epsilon[p] + rho * d_rho[p]
        } else {
            epsilon[p]
        };
        let sigma_coefficient_here = if rho > 0.0 {
            2.0 * rho * d_sigma[p]
        } else {
            0.0
        };
        if !(potential_here.is_finite() && sigma_coefficient_here.is_finite()) {
            return Err(format!(
                "XC 系数在网格点 {p} 非有限（potential {potential_here:.3e}, 2ρdε/dσ \
                 {sigma_coefficient_here:.3e}, ρ = {rho:.3e}, dε/dρ = {:.3e}, dε/dσ = {:.3e}）",
                d_rho[p], d_sigma[p]
            ));
        }
        potential.push(potential_here);
        sigma_coefficient.push(sigma_coefficient_here);
    }
    Ok((energy, potential, sigma_coefficient))
}

/// CPU 的 `v_xc` 装配（`O(n_points·n²)`）：与 `qwb-scf` 逐句同序。
#[allow(clippy::needless_range_loop)]
fn gga_assemble_vxc(
    inputs: &XcGridGgaInputs<'_>,
    gradient_rho: &[[f64; 3]],
    potential: &[f64],
    sigma_coefficient: &[f64],
) -> Vec<f64> {
    let XcGridGgaInputs {
        n,
        n_points,
        ao_values,
        ao_gradients,
        weights,
        ..
    } = *inputs;
    let [gx, gy, gz] = ao_gradients;
    let mut v_xc = vec![0.0_f64; n * n];
    for p in 0..n_points {
        let row = &ao_values[p * n..(p + 1) * n];
        let gx_row = &gx[p * n..(p + 1) * n];
        let gy_row = &gy[p * n..(p + 1) * n];
        let gz_row = &gz[p * n..(p + 1) * n];
        let gradient_here = gradient_rho[p];
        for i in 0..n {
            if row[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                let mut element = row[i] * row[j] * potential[p];
                if sigma_coefficient[p] != 0.0 {
                    // ∇ρ · ∇(Φ_i Φ_j)
                    let g_i = [gx_row[i], gy_row[i], gz_row[i]];
                    let g_j = [gx_row[j], gy_row[j], gz_row[j]];
                    let dot = (0..3)
                        .map(|axis| gradient_here[axis] * (g_i[axis] * row[j] + row[i] * g_j[axis]))
                        .sum::<f64>();
                    element += sigma_coefficient[p] * dot;
                }
                v_xc[i * n + j] += weights[p] * element;
            }
        }
    }
    v_xc
}

/// 网格 XC 收缩（**GGA 通道**）：算 `∇ρ`、`σ`，装配 `v_xc` 的 `σ` 项。
///
/// `evaluate` 收 `(ρ_p, σ_p)`，返回 `(ε_p, dε/dρ, dε/dσ)` —— 就是
/// `qwb-scf::rks::evaluate_xc_cached` 从 `FunctionalTerm` 里取的那三个量，所以
/// `tests/ex03_xc_grid.rs` 里两侧共用同一段泛函求和代码。
///
/// 只覆盖 **GGA**：需要 `∇²ρ`（LYP 一类）或 `τ`（meta-GGA）的项请直接走
/// [`cpu_contract_gga`] —— 那两个通道的 `v_xc` 项要多几张 `∇²Φ`/`τ` 的收缩，目前没有实现，
/// 也不假装实现。调用方应当自己按 `term.requires()` 拒绝这类项。
///
/// # Errors
///
/// 形状不合法、`ρ`/`∇ρ`/系数非有限、泛函求值失败 —— 与 [`cpu_contract_gga`] 相同。
pub fn xc_grid_gga<F>(
    inputs: &XcGridGgaInputs<'_>,
    options: XcGridOptions,
    mut evaluate: F,
) -> Result<XcGridGgaOutput, String>
where
    F: FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
{
    // 先验形状：设备核按 `n`/`n_points` 算偏移，长度对不上就是**越界读设备缓冲**。
    // （`tests/ex03_xc_fallback.rs::gga_shape_errors_are_reported_not_panicked` 抓到过一次
    // 漏掉这一行的版本：它没报错，直接拿错长度的 `∇Φ` 跑出了结果。）
    validate_gga(inputs)?;
    let mut reason = if options.force_cpu {
        Some("调用方要求走 CPU（XcGridOptions::force_cpu）".to_string())
    } else if disabled_by_env() {
        Some("QWB_GPU_XC_DISABLE 已设置".to_string())
    } else {
        match gpu_contract_gga(inputs, options, &mut evaluate) {
            Ok(output) => return Ok(output),
            Err(Attempt::Device(error)) => Some(format!("{error}")),
            Err(Attempt::Input(why)) => return Err(why),
        }
    };
    let mut output = cpu_contract_gga(inputs, &mut evaluate)?;
    output.fallback_reason = reason.take();
    Ok(output)
}

#[cfg(feature = "gpu-kernel")]
fn gpu_contract_gga<F>(
    inputs: &XcGridGgaInputs<'_>,
    options: XcGridOptions,
    evaluate: &mut F,
) -> Result<XcGridGgaOutput, Attempt>
where
    F: FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
{
    use crate::jit::kernel::XcSession;

    let XcGridGgaInputs {
        n,
        n_points,
        ao_values,
        ao_gradients,
        density,
        weights,
    } = *inputs;
    let [gx, gy, gz] = ao_gradients;

    let budget = match options.budget_bytes {
        Some(bytes) => bytes,
        None => default_budget().ok_or_else(|| {
            Attempt::Device(GpuError::Unavailable(
                "取不到显存预算（没有 CUDA 设备）".to_string(),
            ))
        })?,
    };
    let required = xc_gga_bytes(n_points, n);
    if required > budget {
        return Err(Attempt::Device(GpuError::InsufficientMemory {
            required,
            available: budget,
        }));
    }

    let session = XcSession::get().map_err(Attempt::Device)?;
    let phi_d = session.upload(ao_values).map_err(Attempt::Device)?;
    // 三张 `∇Φ` 各自一整块：设备核按轴取行，σ 通道的三个 gemm 也按轴各吃一张。
    let gx_d = session.upload(gx).map_err(Attempt::Device)?;
    let gy_d = session.upload(gy).map_err(Attempt::Device)?;
    let gz_d = session.upload(gz).map_err(Attempt::Device)?;
    let p_d = session.upload(density).map_err(Attempt::Device)?;
    let rho_d = session.alloc(n_points).map_err(Attempt::Device)?;
    let grad_d = session.alloc(3 * n_points).map_err(Attempt::Device)?;
    let sigma_d = session.alloc(n_points).map_err(Attempt::Device)?;
    // 一次算齐 ρ、∇ρ、σ（同一个核，同一个线程内串行累加）。
    session
        .gga_rho_grad(
            &phi_d, &gx_d, &gy_d, &gz_d, &p_d, &rho_d, &grad_d, &sigma_d, n, n_points,
        )
        .map_err(Attempt::Device)?;
    session.synchronize().map_err(Attempt::Device)?;
    let raw_rho = session
        .download(&rho_d, n_points)
        .map_err(Attempt::Device)?;
    let raw_grad = session
        .download(&grad_d, 3 * n_points)
        .map_err(Attempt::Device)?;
    let sigma_values = session
        .download(&sigma_d, n_points)
        .map_err(Attempt::Device)?;

    let mut rho_values = Vec::with_capacity(n_points);
    for (p, rho) in raw_rho.iter().enumerate() {
        if !rho.is_finite() {
            return Err(Attempt::Input(format!(
                "density is not finite ({rho}) at grid point {p}; grid coordinates or AO \
                 values are corrupted"
            )));
        }
        rho_values.push(rho.max(0.0));
    }
    // 设备上的 `grad_d` 是**轴优先**布局（`[axis·n_points + p]`），这里转回按点。
    let mut gradient_rho: Vec<[f64; 3]> = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let here = [
            raw_grad[p],
            raw_grad[n_points + p],
            raw_grad[2 * n_points + p],
        ];
        if !here.iter().all(|value| value.is_finite()) {
            return Err(Attempt::Input(
                "density gradient is not finite at a grid point".to_string(),
            ));
        }
        gradient_rho.push(here);
    }

    // 系数（host，O(n_points)）与 CPU 路径**同一个函数**，所以 `potential` / `2ρ dε/dσ`
    // 与 `E_xc` 的累加序一条不差；差别只在 `ρ`/`σ` 是设备算的。
    let (energy, potential, sigma_coefficient) =
        gga_evaluate_and_coefficients(&rho_values, &sigma_values, weights, evaluate)
            .map_err(Attempt::Input)?;
    let scale: Vec<f64> = (0..n_points).map(|p| weights[p] * potential[p]).collect();
    let sigma_scale: Vec<f64> = (0..n_points)
        .map(|p| weights[p] * sigma_coefficient[p])
        .collect();

    let scale_d = session.upload(&scale).map_err(Attempt::Device)?;
    let temp_d = session.alloc(n_points * n).map_err(Attempt::Device)?;
    let v_d = session.alloc(n * n).map_err(Attempt::Device)?;
    let m_d = session.alloc(n * n).map_err(Attempt::Device)?;
    // 四个 gemm 共用**一个** cuBLAS handle：`cublasCreate` 实测 0.83 ms/次（`available()`
    // 十次平均），逐个建/销毁就是白付 3.2 ms —— 而 n=7 时整个设备侧才十几毫秒。
    let mut contributions = Vec::with_capacity(3);
    session
        .with_gemm_handle(|handle| {
            // LDA 那半边：v_xc ← Φᵀ·(a∘Φ)。
            session.scale_rows(&phi_d, &scale_d, &temp_d, n, n_points)?;
            handle.gemm_v(&v_d, &phi_d, &temp_d, n, n_points)?;
            // σ 那半边：一个轴向一张 `M_a = ∇Φᵀ_a·(s∘Φ)`，贡献是 `M_a + M_aᵀ`
            // （转置在 host 上加，n² 只有几百个元素）。
            for (axis, gradient_d) in [&gx_d, &gy_d, &gz_d].into_iter().enumerate() {
                let mut s_axis = Vec::with_capacity(n_points);
                for p in 0..n_points {
                    s_axis.push(sigma_scale[p] * gradient_rho[p][axis]);
                }
                let s_axis_d = session.upload(&s_axis)?;
                session.scale_rows(&phi_d, &s_axis_d, &temp_d, n, n_points)?;
                handle.gemm_v(&m_d, gradient_d, &temp_d, n, n_points)?;
                session.synchronize()?;
                contributions.push(session.download(&m_d, n * n)?);
            }
            Ok(())
        })
        .map_err(Attempt::Device)?;
    session.synchronize().map_err(Attempt::Device)?;
    let mut v_xc = session.download(&v_d, n * n).map_err(Attempt::Device)?;
    for contribution in &contributions {
        for i in 0..n {
            for j in 0..n {
                v_xc[i * n + j] += contribution[i * n + j] + contribution[j * n + i];
            }
        }
    }
    if !v_xc.iter().all(|value| value.is_finite()) {
        return Err(Attempt::Input(
            "v_xc 里出现非有限值（设备内核的结果不可信）".to_string(),
        ));
    }

    Ok(XcGridGgaOutput {
        energy,
        v_xc,
        gradient_rho,
        sigma: sigma_values,
        rho: rho_values,
        path: XcPath::Gpu,
        fallback_reason: None,
    })
}

#[cfg(not(feature = "gpu-kernel"))]
fn gpu_contract_gga<F>(
    _inputs: &XcGridGgaInputs<'_>,
    _options: XcGridOptions,
    _evaluate: &mut F,
) -> Result<XcGridGgaOutput, Attempt>
where
    F: FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
{
    Err(Attempt::Device(GpuError::Unavailable(
        "feature `gpu-kernel` 未启用（编译期就没有 NVRTC/驱动层）".to_string(),
    )))
}

/// NVRTC 是否真的可用、用的什么架构 —— 给验收报告与诊断用。
///
/// # Errors
///
/// feature 没开、或 NVRTC/驱动不可用（消息里带原因）。
#[cfg(feature = "gpu-kernel")]
pub fn nvrtc_status() -> Result<String, String> {
    let session = crate::jit::kernel::XcSession::get().map_err(|error| format!("{error}"))?;
    Ok(format!(
        "NVRTC {}.{}, --gpu-architecture=compute_{}",
        session.nvrtc_version.0, session.nvrtc_version.1, session.gpu_arch
    ))
}
/// NVRTC 是否真的可用 —— feature 没开时恒为 `Err`。
///
/// # Errors
///
/// feature `gpu-kernel` 未启用。
#[cfg(not(feature = "gpu-kernel"))]
pub fn nvrtc_status() -> Result<String, String> {
    Err("feature `gpu-kernel` 未启用".to_string())
}
