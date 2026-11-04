//! EX-03 第五块：把网格 XC 的收缩段**接进产品 SCF**，并让 `Φ`/`∇Φ` 常驻设备。
//!
//! # 这一段补的是上一块留下的两个洞
//!
//! 上一块（`xc.rs`）交付了两条核（LDA/GGA）与逐位一致的 CPU 回退，但有两个问题让它
//! 上不了产品路径：
//!
//! 1. **没有产品接缝** —— `qwb-scf` 不能依赖 `qwb-gpu`（会成环，见
//!    `qwb_scf::xc_device` 的模块文档）。这里实现 `qwb-scf` 定义的
//!    [`XcGridAccelerator`]，由 `qwb-engine` 在 `gpu-kernel` feature 下注入。
//! 2. **每次收缩都重传 `Φ`/`∇Φ`** —— 实测占设备时间 ~28 %。这几张矩阵在整次 SCF 里
//!    **一个字都不会变**，所以它们应该留在设备上，每次只上传 `P`（`n²` 个双精度）。
//!
//! # 驻留怎么成立、为什么不会算错
//!
//! [`Bound`] 按 [`XcGridRequest::residency`] 缓存设备缓冲。那个键由 `qwb-scf` 的
//! `XcGridFlatAo::build` 用单调计数器分配：同一份 `Φ`/`∇Φ` 一个键，换一份就换键。
//! 键不同 → 丢掉旧缓冲、重传；键相同 → 一次上传都不做。键只影响"要不要重传"，不参与
//! 任何数值，所以**键错了只有性能后果、没有正确性后果**（最坏是每次重传）。
//!
//! # 与上一块的关系
//!
//! `xc.rs` 的两个入口（`xc_grid_lda` / `xc_grid_gga`）**一个字节没改**：它们还是
//! "每次收缩都 upload + alloc"的形状。本文件的**
//! [`XcDeviceAccelerator::with_residency`]`(false)` 走同一段收缩代码、只是每次把缓冲
//! 丢掉重来**，所以下面 before/after 的两列是同一段代码的两个开关位置，不是两个版本的
//! 函数各自计时。

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use qwb_scf::xc_device::{
    XcGgaEvaluate, XcGridAccelerator, XcGridDeviceOutcome, XcGridDeviceReport, XcGridRequest,
    XcLdaEvaluate,
};

use crate::GpuError;
use crate::jit::kernel::{DeviceBuffer, XcSession};
use crate::xc::{gga_evaluate_and_coefficients, xc_bytes, xc_gga_bytes};

/// 网格 XC 的设备收缩器（`qwb_scf::xc_device::XcGridAccelerator` 的实现）。
///
/// 一个实例对应一份驻留缓存：SCF 期间每个迭代步调一次
/// [`XcGridAccelerator::contract_lda`] / [`XcGridAccelerator::contract_gga`]。
pub struct XcDeviceAccelerator {
    resident: bool,
    budget_bytes: Option<u64>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    bound: Option<Bound>,
}

/// 一次「绑定」的成果：与密度无关的输入缓冲 + 复用的输出/临时缓冲。
struct Bound {
    session: &'static XcSession,
    residency: u64,
    n: usize,
    n_points: usize,
    /// `Φ`（`n_points × n`）。
    phi: DeviceBuffer,
    /// `P`（`n × n`）—— 每步重写，缓冲常驻。
    density: DeviceBuffer,
    /// `ρ_p`（`n_points`）—— 手写核的输出。
    rho: DeviceBuffer,
    /// 逐点系数 `w·v`（`n_points`）—— 每步重写；GGA 的 σ 通道也复用它。
    scale: DeviceBuffer,
    /// `Φ` 的按行缩比结果（`n_points × n`）。
    temp: DeviceBuffer,
    /// `v_xc` 的 LDA 半边（`n × n`）。
    vxc: DeviceBuffer,
    gga: Option<GgaBound>,
}

/// GGA 通道多出来的那几张。
struct GgaBound {
    /// `∂Φ/∂x`、`∂Φ/∂y`、`∂Φ/∂z`。
    ao_gradients: [DeviceBuffer; 3],
    /// `∇ρ_p`（轴优先 `[axis·n_points + p]`）。
    gradient_rho: DeviceBuffer,
    /// `σ_p = |∇ρ|²`。
    sigma: DeviceBuffer,
    /// `w·2ρ dε/dσ`（`n_points`）。
    sigma_scale: DeviceBuffer,
    /// 三个轴向的 `M_a = ∇Φᵀ_a·(s_a∘Φ)`（各 `n × n`）。
    contributions: [DeviceBuffer; 3],
}

impl Bound {
    fn create(
        session: &'static XcSession,
        request: &XcGridRequest<'_>,
        with_gradients: bool,
    ) -> Result<Self, GpuError> {
        let n = request.n;
        let n_points = request.n_points;
        let phi = session.upload(request.ao_values)?;
        let gga = if with_gradients {
            let [gx, gy, gz] = request
                .ao_gradients
                .ok_or_else(|| GpuError::Unavailable("GGA 通道缺少 ∇Φ".to_string()))?;
            Some(GgaBound {
                ao_gradients: [
                    session.upload(gx)?,
                    session.upload(gy)?,
                    session.upload(gz)?,
                ],
                gradient_rho: session.alloc(3 * n_points)?,
                sigma: session.alloc(n_points)?,
                sigma_scale: session.alloc(n_points)?,
                contributions: [
                    session.alloc(n * n)?,
                    session.alloc(n * n)?,
                    session.alloc(n * n)?,
                ],
            })
        } else {
            None
        };
        Ok(Self {
            session,
            residency: request.residency,
            n,
            n_points,
            phi,
            density: session.alloc(n * n)?,
            rho: session.alloc(n_points)?,
            scale: session.alloc(n_points)?,
            temp: session.alloc(n_points * n)?,
            vxc: session.alloc(n * n)?,
            gga,
        })
    }

    fn matches(&self, request: &XcGridRequest<'_>, with_gradients: bool) -> bool {
        self.residency == request.residency
            && self.n == request.n
            && self.n_points == request.n_points
            && self.gga.is_some() == with_gradients
    }

    /// 这块绑定在设备上占多少字节（含 `Φ`/`∇Φ` 与临时块）—— 驻留的代价，报告里写清楚。
    fn device_bytes(&self) -> u64 {
        let n = self.n;
        let n_points = self.n_points;
        let mut doubles = n_points * n + n * n + n_points + n_points + n_points * n + n * n;
        if self.gga.is_some() {
            doubles += 3 * n_points * n + 3 * n_points + n_points + n_points + 3 * n * n;
        }
        (doubles as u64) * 8
    }
}

impl XcDeviceAccelerator {
    /// 默认：`Φ`/`∇Φ` **常驻**（SCF 各步之间不重传）。
    #[must_use]
    pub fn new() -> Self {
        Self {
            resident: true,
            budget_bytes: None,
            state: Mutex::new(State::default()),
        }
    }

    /// 驻留开关。`false` 复刻**改之前**的行为（每次收缩都重新 upload + alloc + free），
    /// 用来给「加速了多少」量一个同代码的对照。
    #[must_use]
    pub fn with_residency(resident: bool) -> Self {
        Self {
            resident,
            budget_bytes: None,
            state: Mutex::new(State::default()),
        }
    }

    /// 显式显存预算（`None` = 空闲显存的 75 %）。测「显存不够就退回 CPU」用。
    #[must_use]
    pub fn with_budget(mut self, bytes: u64) -> Self {
        self.budget_bytes = Some(bytes);
        self
    }

    /// 当前驻留了多少设备字节（没绑定时 0）。
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.lock().bound.as_ref().map_or(0, Bound::device_bytes)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn check_budget(
        &self,
        request: &XcGridRequest<'_>,
        with_gradients: bool,
    ) -> Result<(), String> {
        let required = if with_gradients {
            xc_gga_bytes(request.n_points, request.n)
        } else {
            xc_bytes(request.n_points, request.n)
        };
        let budget = match self.budget_bytes {
            Some(bytes) => bytes,
            None => crate::memory_budget_bytes().map_err(|error| error.to_string())?,
        };
        if required > budget {
            return Err(format!(
                "显存预算不够：需要 {required} B，预算 {budget} B（请退回 CPU）"
            ));
        }
        Ok(())
    }

    /// 取（或重建）本请求对应的绑定，并把上传阶段记进 `report`。
    ///
    /// 驻留命中时**一个字节都不上传**。
    fn bind(
        &self,
        request: &XcGridRequest<'_>,
        with_gradients: bool,
        report: &mut XcGridDeviceReport,
    ) -> Result<Bound, String> {
        let mut state = self.lock();
        let reusable = self.resident
            && state
                .bound
                .as_ref()
                .is_some_and(|bound| bound.matches(request, with_gradients));
        if reusable {
            // `take` 出来用、用完放回去：绕开"持有 `&mut state` 又想改 `state`"的借用冲突。
            return Ok(state.bound.take().expect("reusable 已检查过 Some"));
        }
        let started = Instant::now();
        state.bound = None;
        let session = XcSession::get().map_err(|error| error.to_string())?;
        let bound =
            Bound::create(session, request, with_gradients).map_err(|error| error.to_string())?;
        // 上传的字节：`Φ`（GGA 再加 `∇Φ`×3）。这就是驻留省下来的那一笔。
        let mut doubles = request.n_points * request.n;
        if with_gradients {
            doubles += 3 * request.n_points * request.n;
        }
        report.uploads = report.uploads.saturating_add(1);
        report.uploaded_bytes = report.uploaded_bytes.saturating_add((doubles as u64) * 8);
        report.upload_seconds += started.elapsed().as_secs_f64();
        Ok(bound)
    }

    /// 用完把绑定放回去（驻留）或释放（`with_residency(false)`）。
    fn release(&self, bound: Bound, report: &mut XcGridDeviceReport) {
        if self.resident {
            self.lock().bound = Some(bound);
        } else {
            let started = Instant::now();
            drop(bound);
            // 释放也记在"上传"阶段：那一列的名字就是「每步重新分配/传输」的代价。
            report.upload_seconds += started.elapsed().as_secs_f64();
        }
    }

    /// LDA 通道：`ρ` 走 gemm + 行点积，`v_xc` 走一次 gemm。
    fn run_lda(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcLdaEvaluate<'_>,
    ) -> Result<(f64, Vec<f64>, XcGridDeviceReport), String> {
        validate(request, false)?;
        self.check_budget(request, false)?;
        let n = request.n;
        let n_points = request.n_points;
        let mut report = XcGridDeviceReport {
            calls: 1,
            ..XcGridDeviceReport::default()
        };

        let bound = self.bind(request, false, &mut report)?;
        let session = bound.session;

        // ── 上传：`P`（`Φ` 已经在设备上） ──
        let started = Instant::now();
        session
            .copy_htod(&bound.density, request.density)
            .map_err(|error| error.to_string())?;
        report.uploaded_bytes = report.uploaded_bytes.saturating_add(((n * n) as u64) * 8);
        report.upload_seconds += started.elapsed().as_secs_f64();

        // ── gemm：`(ΦP)` ──
        let started = Instant::now();
        session
            .gemm_xm(&bound.temp, &bound.phi, &bound.density, n, n_points)
            .map_err(|error| error.to_string())?;
        report.gemm_seconds += started.elapsed().as_secs_f64();

        // ── 手写核：`ρ_p = Σ_μ Φ_pμ (ΦP)_pμ` ──
        let started = Instant::now();
        session
            .row_dot(&bound.temp, &bound.phi, &bound.rho, n, n_points)
            .map_err(|error| error.to_string())?;
        session.synchronize().map_err(|error| error.to_string())?;
        report.kernel_seconds += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let raw_rho = session
            .download(&bound.rho, n_points)
            .map_err(|error| error.to_string())?;
        report.downloaded_bytes = report
            .downloaded_bytes
            .saturating_add((n_points as u64) * 8);
        report.download_seconds += started.elapsed().as_secs_f64();

        let mut rho_values = Vec::with_capacity(n_points);
        for (point, rho) in raw_rho.iter().enumerate() {
            if !rho.is_finite() {
                return Err(format!(
                    "density is not finite ({rho}) at grid point {point}; grid coordinates or AO \
                     values are corrupted"
                ));
            }
            rho_values.push(rho.max(0.0));
        }

        let (epsilon, potential) = evaluate(&rho_values)?;
        if epsilon.len() != n_points || potential.len() != n_points {
            return Err(format!(
                "泛函返回 {} 个 ε / {} 个 v，应为 n_points = {n_points}",
                epsilon.len(),
                potential.len()
            ));
        }
        // `E_xc` 与 `w·v` 的求和序与 `qwb-scf::rks::evaluate_xc_cached` 逐句一致，
        // 所以 LDA 通道的 `E_xc` 与产品 CPU 路径**逐位相同**（不是"在容差内"）。
        let mut energy = 0.0_f64;
        let mut scale = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let potential_here = potential[point];
            if !potential_here.is_finite() {
                return Err(format!(
                    "XC 势在网格点 {point} 非有限（{potential_here:.3e}），ρ = {:.3e}",
                    rho_values[point]
                ));
            }
            scale.push(request.weights[point] * potential_here);
            energy += request.weights[point] * rho_values[point] * epsilon[point];
        }

        // ── 上传系数，`v_xc` 的 gemm，下载 ──
        let started = Instant::now();
        session
            .copy_htod(&bound.scale, &scale)
            .map_err(|error| error.to_string())?;
        report.uploaded_bytes = report.uploaded_bytes.saturating_add((n_points as u64) * 8);
        report.upload_seconds += started.elapsed().as_secs_f64();

        let started = Instant::now();
        session
            .scale_rows(&bound.phi, &bound.scale, &bound.temp, n, n_points)
            .map_err(|error| error.to_string())?;
        session
            .gemm_v(&bound.vxc, &bound.phi, &bound.temp, n, n_points)
            .map_err(|error| error.to_string())?;
        session.synchronize().map_err(|error| error.to_string())?;
        report.gemm_seconds += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let v_xc = session
            .download(&bound.vxc, n * n)
            .map_err(|error| error.to_string())?;
        report.downloaded_bytes = report.downloaded_bytes.saturating_add(((n * n) as u64) * 8);
        report.download_seconds += started.elapsed().as_secs_f64();
        if !v_xc.iter().all(|value| value.is_finite()) {
            return Err("v_xc 里出现非有限值（设备内核的结果不可信）".to_string());
        }

        self.release(bound, &mut report);
        Ok((energy, v_xc, report))
    }

    /// GGA 通道：`ρ`、`∇ρ`、`σ` 一次算齐（一个核），`v_xc` 装配要四个 gemm。
    fn run_gga(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcGgaEvaluate<'_>,
    ) -> Result<(f64, Vec<f64>, XcGridDeviceReport), String> {
        validate(request, true)?;
        self.check_budget(request, true)?;
        let n = request.n;
        let n_points = request.n_points;
        let mut report = XcGridDeviceReport {
            calls: 1,
            ..XcGridDeviceReport::default()
        };

        let bound = self.bind(request, true, &mut report)?;
        let session = bound.session;
        let gga = bound.gga.as_ref().expect("GGA 通道一定绑定了 ∇Φ");

        // ── 上传：`P` ──
        let started = Instant::now();
        session
            .copy_htod(&bound.density, request.density)
            .map_err(|error| error.to_string())?;
        report.uploaded_bytes = report.uploaded_bytes.saturating_add(((n * n) as u64) * 8);
        report.upload_seconds += started.elapsed().as_secs_f64();

        // ── 手写核：一次算齐 ρ、∇ρ、σ ──
        let started = Instant::now();
        session
            .gga_rho_grad(
                &bound.phi,
                &gga.ao_gradients[0],
                &gga.ao_gradients[1],
                &gga.ao_gradients[2],
                &bound.density,
                &bound.rho,
                &gga.gradient_rho,
                &gga.sigma,
                n,
                n_points,
            )
            .map_err(|error| error.to_string())?;
        session.synchronize().map_err(|error| error.to_string())?;
        report.kernel_seconds += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let raw_rho = session
            .download(&bound.rho, n_points)
            .map_err(|error| error.to_string())?;
        // 设备上的 `∇ρ` 是**轴优先**布局（`[axis·n_points + p]`），这里转回按点。
        let raw_gradient = session
            .download(&gga.gradient_rho, 3 * n_points)
            .map_err(|error| error.to_string())?;
        let sigma_values = session
            .download(&gga.sigma, n_points)
            .map_err(|error| error.to_string())?;
        report.downloaded_bytes = report
            .downloaded_bytes
            .saturating_add(((5 * n_points) as u64) * 8);
        report.download_seconds += started.elapsed().as_secs_f64();

        let mut rho_values = Vec::with_capacity(n_points);
        for (point, rho) in raw_rho.iter().enumerate() {
            if !rho.is_finite() {
                return Err(format!(
                    "density is not finite ({rho}) at grid point {point}; grid coordinates or AO \
                     values are corrupted"
                ));
            }
            rho_values.push(rho.max(0.0));
        }
        let mut gradient_rho: Vec<[f64; 3]> = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let here = [
                raw_gradient[point],
                raw_gradient[n_points + point],
                raw_gradient[2 * n_points + point],
            ];
            if !here.iter().all(|value| value.is_finite()) {
                return Err("density gradient is not finite at a grid point".to_string());
            }
            gradient_rho.push(here);
        }

        // 系数与 `E_xc` 走**和上一块完全同一个函数**（`gga_evaluate_and_coefficients`）：
        // 所以 `E_xc` 的累加序、`ε + ρ dε/dρ`、`2ρ dε/dσ` 这两条组合只有一份代码。
        let (energy, potential, sigma_coefficient) =
            gga_evaluate_and_coefficients(&rho_values, &sigma_values, request.weights, &mut {
                let evaluate = &mut *evaluate;
                move |rho: &[f64], sigma: &[f64]| evaluate(rho, sigma)
            })?;

        let scale: Vec<f64> = (0..n_points)
            .map(|point| request.weights[point] * potential[point])
            .collect();
        let sigma_scale: Vec<f64> = (0..n_points)
            .map(|point| request.weights[point] * sigma_coefficient[point])
            .collect();

        // ── 上传逐点系数 ──
        let started = Instant::now();
        session
            .copy_htod(&bound.scale, &scale)
            .map_err(|error| error.to_string())?;
        session
            .copy_htod(&gga.sigma_scale, &sigma_scale)
            .map_err(|error| error.to_string())?;
        report.uploaded_bytes = report
            .uploaded_bytes
            .saturating_add((2 * n_points as u64) * 8);
        report.upload_seconds += started.elapsed().as_secs_f64();

        // ── 四个 gemm：LDA 半边 + σ 的三个轴 ──
        //
        // 四个 gemm 都在同一个 cuBLAS handle 上（`with_gemm_handle`）：`cublasCreate`/
        // `Destroy` 实测 0.83 ms/次，逐个建就是白付 2.5 ms。同步点只有一个（在下载之前），
        // 三个 σ 轴向的中间结果各占一块常驻缓冲，所以不需要"gemm → 同步 → 下载"三次往返。
        //
        // 记账口径：三个 `s_a` 的上传（`3·n_points` 个双精度）落在这一段的窗口里 ——
        // 它们与 gemm 交替进行，拆不成"纯 gemm"与"纯上传"两段；字节数照实记在
        // `uploaded_bytes` 里，时间算在 `gemm_seconds`。
        let started = Instant::now();
        session
            .with_gemm_handle(|handle| {
                session.scale_rows(&bound.phi, &bound.scale, &bound.temp, n, n_points)?;
                handle.gemm_v(&bound.vxc, &bound.phi, &bound.temp, n, n_points)?;
                for (axis, gradient) in gga.ao_gradients.iter().enumerate() {
                    let mut s_axis = Vec::with_capacity(n_points);
                    for point in 0..n_points {
                        s_axis.push(sigma_scale[point] * gradient_rho[point][axis]);
                    }
                    // `scale` 的 LDA 半边已经用完，这里复用它装 `s_a`。
                    session.copy_htod(&bound.scale, &s_axis)?;
                    session.scale_rows(&bound.phi, &bound.scale, &bound.temp, n, n_points)?;
                    handle.gemm_v(&gga.contributions[axis], gradient, &bound.temp, n, n_points)?;
                }
                Ok(())
            })
            .map_err(|error| error.to_string())?;
        report.uploaded_bytes = report
            .uploaded_bytes
            .saturating_add((3 * n_points as u64) * 8);
        session.synchronize().map_err(|error| error.to_string())?;
        report.gemm_seconds += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let mut v_xc = session
            .download(&bound.vxc, n * n)
            .map_err(|error| error.to_string())?;
        let mut contributions = Vec::with_capacity(3);
        for buffer in &gga.contributions {
            contributions.push(
                session
                    .download(buffer, n * n)
                    .map_err(|error| error.to_string())?,
            );
        }
        report.downloaded_bytes = report
            .downloaded_bytes
            .saturating_add(((4 * n * n) as u64) * 8);
        report.download_seconds += started.elapsed().as_secs_f64();
        // σ 通道的贡献是 `M_a + M_aᵀ`（转置在 host 上加：`n²` 只有几百个元素）。
        for contribution in &contributions {
            for i in 0..n {
                for j in 0..n {
                    v_xc[i * n + j] += contribution[i * n + j] + contribution[j * n + i];
                }
            }
        }
        if !v_xc.iter().all(|value| value.is_finite()) {
            return Err("v_xc 里出现非有限值（设备内核的结果不可信）".to_string());
        }

        self.release(bound, &mut report);
        Ok((energy, v_xc, report))
    }
}

impl Default for XcDeviceAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for XcDeviceAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("XcDeviceAccelerator")
            .field("resident", &self.resident)
            .field("resident_bytes", &self.resident_bytes())
            .finish()
    }
}

fn finish(result: Result<(f64, Vec<f64>, XcGridDeviceReport), String>) -> XcGridDeviceOutcome {
    match result {
        Ok((energy, v_xc, report)) => XcGridDeviceOutcome::Done {
            energy,
            v_xc,
            report,
        },
        Err(reason) => XcGridDeviceOutcome::Unavailable { reason },
    }
}

impl XcGridAccelerator for XcDeviceAccelerator {
    fn contract_lda(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcLdaEvaluate<'_>,
    ) -> XcGridDeviceOutcome {
        finish(self.run_lda(request, evaluate))
    }

    fn contract_gga(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcGgaEvaluate<'_>,
    ) -> XcGridDeviceOutcome {
        finish(self.run_gga(request, evaluate))
    }

    fn name(&self) -> &'static str {
        "qwb-gpu xc_device"
    }
}

fn validate(request: &XcGridRequest<'_>, with_gradients: bool) -> Result<(), String> {
    let n = request.n;
    let n_points = request.n_points;
    if n == 0 || n_points == 0 {
        return Err(format!("形状不合法：n={n}, n_points={n_points}"));
    }
    if request.ao_values.len() != n * n_points {
        return Err(format!(
            "形状不合法：ao_values.len()={}，应为 n_points·n = {}",
            request.ao_values.len(),
            n * n_points
        ));
    }
    if request.density.len() != n * n {
        return Err(format!(
            "形状不合法：density.len()={}，应为 n² = {}",
            request.density.len(),
            n * n
        ));
    }
    if request.weights.len() != n_points {
        return Err(format!(
            "形状不合法：weights.len()={}，应为 n_points = {n_points}",
            request.weights.len()
        ));
    }
    if with_gradients {
        let gradients = request
            .ao_gradients
            .ok_or_else(|| "GGA 通道缺少 ∇Φ".to_string())?;
        for (axis, gradient) in ["x", "y", "z"].iter().zip(gradients.iter()) {
            if gradient.len() != n * n_points {
                return Err(format!(
                    "形状不合法：ao_gradients[{axis}].len()={}，应为 n_points·n = {}",
                    gradient.len(),
                    n * n_points
                ));
            }
        }
    }
    Ok(())
}
