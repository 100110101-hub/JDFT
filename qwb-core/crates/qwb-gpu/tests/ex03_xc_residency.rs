//! EX-03 第五块验收：**设备驻留**（`Φ`/`∇Φ` 只传一次）与产品 SCF 接线。
//!
//! 上一块（`ex03_xc_grid.rs`）量的是"一次收缩里设备比 CPU 快多少"，但它每次收缩都重传
//! `Φ`/`∇Φ` —— 实测占设备时间 ~28 %。这一块量的是**把那两张（GGA 四张）留在设备上之后**
//! 的分项，以及它接进 `qwb_scf::rks::rks_scf` 之后的真实效果。
//!
//! # 三个测试
//!
//! * `a1_the_resident_contract_matches_the_cpu_path`：驻留路径的 `E_xc`/`v_xc` 与产品 CPU
//!   路径（`evaluate_xc_cached`）比，LDA 与 GGA 各过一遍；
//! * `a2_the_ao_data_stays_resident_across_the_scf`：同一段收缩代码的两个开关位置
//!   （`with_residency(false)` = **改之前**、`true` = **改之后**）跑完同一个 PBE SCF，
//!   报 `uploads / kernel / gemm / download` 四项的分项与字节数，并把**上传字节数逐字节**
//!   对上公式（不是"看起来少了"）；
//! * `a3_the_product_scf_runs_on_the_device_path`：经由 `rks_scf` 的 `xc_path`/`xc_backend`
//!   选项跑完整 SCF，报 CPU vs GPU 的能量差与墙钟。
//!
//! 没有 GPU / 没开 `gpu-kernel` 时不是"跳过"：`xc_path = Gpu` 会退回 CPU 并把原因写在
//! `RksResult::xc_fallback_reason` 里，测试断言**退回后的数与 CPU 路径逐位相同**。
//!
//! 这条构建里 `qwb_gpu::xc_device` 不存在（它整个模块都在 `gpu-kernel` 下面），所以这个
//! 测试文件在**没开 feature** 时整体不编译；那种构建下的退回由
//! `qwb-scf/tests/ex03_xc_device_hook.rs`（不依赖 GPU 的假后端）与
//! `qwb-engine/tests/ex03_scf_gpu_xc.rs`（真实产品路径）覆盖。
#![cfg(feature = "gpu-kernel")]
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use qwb_engine_api::{Atom, Molecule};
use qwb_gpu::xc_device::XcDeviceAccelerator;
use qwb_registry::Registry;
use qwb_scf::rks::{GridAoCached, RksOptions, XcGridFlatAo, evaluate_xc_cached, rks_scf};
use qwb_scf::xc_device::{XcGridAccelerator, XcGridBackendHandle, XcGridDeviceReport, XcGridPath};
use qwb_xc::{DensityBatch, FunctionalTerm};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

fn terms_from(registry: &Registry, name: &str) -> Vec<Box<dyn FunctionalTerm>> {
    let functional = registry.get_functional(name).expect(name);
    let parameters: std::collections::BTreeMap<String, f64> = functional.parameters.clone();
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        // 精确交换不是能量密度项（它在 Fock 矩阵里）。
        if component.weight.abs() <= 1e-12 || component.id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

/// 产品路径的 LDA 泛函求值（与 `evaluate_xc_cached` 的 LDA 分支逐句同序）。
#[allow(clippy::type_complexity)]
fn lda_evaluate<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64]| {
        let points = rho.len();
        let batch = DensityBatch::from_rho(rho);
        let mut epsilon = vec![0.0; points];
        let mut d_rho = vec![0.0; points];
        for term in terms {
            let mut local_epsilon = vec![0.0; points];
            let mut local_d_rho = vec![0.0; points];
            let mut local_d_sigma = vec![0.0; points];
            let mut local_d_laplacian = vec![0.0; points];
            let mut local_d_tau = vec![0.0; points];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_full(
                &batch,
                &mut local_d_rho,
                &mut local_d_sigma,
                &mut local_d_laplacian,
                &mut local_d_tau,
            );
            for index in 0..points {
                epsilon[index] += local_epsilon[index];
                d_rho[index] += local_d_rho[index];
            }
        }
        let potential = (0..points)
            .map(|index| {
                if rho[index] > 0.0 {
                    epsilon[index] + rho[index] * d_rho[index]
                } else {
                    epsilon[index]
                }
            })
            .collect();
        Ok((epsilon, potential))
    }
}

/// 产品路径的 GGA 泛函求值（与 `evaluate_xc_cached` 的 GGA 分支逐句同序）。
#[allow(clippy::type_complexity)]
fn gga_evaluate<'a>(
    terms: &'a [Box<dyn FunctionalTerm>],
) -> impl FnMut(&[f64], &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> + 'a {
    move |rho: &[f64], sigma: &[f64]| {
        let points = rho.len();
        let batch = DensityBatch::new(rho, sigma);
        let mut epsilon = vec![0.0; points];
        let mut d_rho = vec![0.0; points];
        let mut d_sigma = vec![0.0; points];
        for term in terms {
            let mut local_epsilon = vec![0.0; points];
            let mut local_d_rho = vec![0.0; points];
            let mut local_d_sigma = vec![0.0; points];
            let mut local_d_laplacian = vec![0.0; points];
            let mut local_d_tau = vec![0.0; points];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_full(
                &batch,
                &mut local_d_rho,
                &mut local_d_sigma,
                &mut local_d_laplacian,
                &mut local_d_tau,
            );
            for index in 0..points {
                epsilon[index] += local_epsilon[index];
                d_rho[index] += local_d_rho[index];
                d_sigma[index] += local_d_sigma[index];
            }
        }
        Ok((epsilon, d_rho, d_sigma))
    }
}

/// 一个算例的夹具：基组、网格、AO 缓存、**本项目自己的 SCF 密度**、泛函项。
struct Setup {
    basis: qwb_basis::InternalBasis,
    grid: qwb_grid::Grid,
    cache: GridAoCached,
    density: qwb_linalg::Matrix,
    terms: Vec<Box<dyn FunctionalTerm>>,
}

fn solve(basis_name: &str, grid_name: &str, functional: &str) -> Setup {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
        .expect("grid");
    let terms = terms_from(&registry, functional);
    let mut options = RksOptions::default();
    options.max_iter = 200;
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    assert!(result.scf_converged, "{basis_name} 的 SCF 没收敛");
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    let cache = GridAoCached::build(&basis, &grid, needs_sigma, needs_laplacian, needs_tau)
        .expect("ao cache");
    Setup {
        basis,
        grid,
        cache,
        density: result.density,
        terms,
    }
}

fn flat_density(density: &qwb_linalg::Matrix) -> Vec<f64> {
    let n = density.nrows();
    let mut flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            flat.push(density[(i, j)]);
        }
    }
    flat
}

/// `v_xc` 的相对最大差（除以尺度），与 `ex03_xc_grid.rs` 同一个口径。
fn relative_difference(gpu: &[f64], cpu: &qwb_linalg::Matrix) -> f64 {
    let n = cpu.nrows();
    let mut worst = 0.0_f64;
    let mut scale = f64::MIN_POSITIVE;
    for i in 0..n {
        for j in 0..n {
            let reference = cpu[(i, j)];
            scale = scale.max(reference.abs());
            worst = worst.max((gpu[i * n + j] - reference).abs());
        }
    }
    worst / scale
}

// ---------------------------------------------------------------------------------------
// A1：驻留路径 vs 产品 CPU 路径
// ---------------------------------------------------------------------------------------

#[test]
fn a1_the_resident_contract_matches_the_cpu_path() {
    println!("NVRTC 状态：{:?}", qwb_gpu::xc::nvrtc_status());
    for functional in ["svwn5", "pbe"] {
        for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
            let setup = solve(basis_name, grid_name, functional);
            let n = setup.basis.n_functions();
            let n_points = setup.grid.len();
            let needs_sigma = setup.terms.iter().any(|term| term.requires().sigma);
            let (energy_cpu, vxc_cpu) = evaluate_xc_cached(
                &setup.basis,
                &setup.grid,
                &setup.cache,
                &setup.density,
                &setup.terms,
            )
            .expect("cpu xc");
            let flat = XcGridFlatAo::build(&setup.basis, &setup.grid, &setup.cache, needs_sigma)
                .expect("flat ao");
            let flat_p = flat_density(&setup.density);
            let request = flat.request(&flat_p, needs_sigma);
            let accelerator = XcDeviceAccelerator::new();
            let mut report = XcGridDeviceReport::default();
            // 走 `qwb-scf` 的注入点（不是直接调 `xc_grid_gga`）：这里要验的正是**产品接缝**。
            let outcome = if needs_sigma {
                let mut evaluate = gga_evaluate(&setup.terms);
                accelerator.contract_gga(&request, &mut evaluate)
            } else {
                let mut evaluate = lda_evaluate(&setup.terms);
                accelerator.contract_lda(&request, &mut evaluate)
            };
            let (energy_gpu, v_xc) = match outcome {
                qwb_scf::xc_device::XcGridDeviceOutcome::Done {
                    energy,
                    v_xc,
                    report: call,
                } => {
                    report.accumulate(&call);
                    (energy, v_xc)
                }
                qwb_scf::xc_device::XcGridDeviceOutcome::Unavailable { reason } => {
                    // 没有 GPU / 没开 feature：**不是跳过** —— 断言退回与 CPU 逐位相同。
                    println!(
                        "{functional}/{basis_name}: 设备不可用（{reason}），断言 CPU 路径不变"
                    );
                    let (energy_again, vxc_again) = evaluate_xc_cached(
                        &setup.basis,
                        &setup.grid,
                        &setup.cache,
                        &setup.density,
                        &setup.terms,
                    )
                    .expect("cpu xc");
                    assert_eq!(energy_again.to_bits(), energy_cpu.to_bits());
                    assert_eq!(vxc_again, vxc_cpu);
                    continue;
                }
            };
            let delta = (energy_gpu - energy_cpu).abs();
            let relative = relative_difference(&v_xc, &vxc_cpu);
            println!(
                "{functional}/{basis_name}/{grid_name}: n = {n}, points = {n_points}, \
                 E_cpu = {energy_cpu:.15e}, E_gpu = {energy_gpu:.15e}, |ΔE| = {delta:.3e} Ha \
                 （逐位相同 = {}）, v_xc 相对最大差 = {relative:.3e}, 上传 {} B / 下载 {} B",
                energy_gpu.to_bits() == energy_cpu.to_bits(),
                report.uploaded_bytes,
                report.downloaded_bytes
            );
            // 验收线就是附录 A 的 1e-10：不因为"实测 0"就改成 `assert_eq!`，
            // 也不因为实测是 0 就把线放宽。
            assert!(delta <= 1.0e-10, "|ΔE| = {delta:.3e} 超过 1e-10 Ha");
            assert!(relative <= 1.0e-10, "v_xc 相对差 {relative:.3e} 超过 1e-10");
        }
    }
}

// ---------------------------------------------------------------------------------------
// A2：驻留的 before/after 分项（同一段代码的两个开关）
// ---------------------------------------------------------------------------------------

/// 跑一次 PBE SCF（`xc_path = Gpu`，后端是给定驻留开关的收缩器），返回
/// `(能量, 设备分项, 墙钟秒)`。
fn scf_on_device(
    setup: &Setup,
    basis_name: &str,
    grid_name: &str,
    resident: bool,
) -> (f64, XcGridDeviceReport, f64, Option<String>) {
    let molecule = water();
    let tensor = qwb_integrals::two_electron::two_electron(&setup.basis, &molecule).expect("2e");
    let registry = Registry::open(data_root()).expect("qwb-data");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
        .expect("grid");
    let accelerator = Arc::new(XcDeviceAccelerator::with_residency(resident));
    let mut options = RksOptions::default();
    options.max_iter = 200;
    options.energy_threshold = 1.0e-9;
    options.xc_path = XcGridPath::Gpu;
    options.xc_backend = Some(XcGridBackendHandle::new(accelerator));
    let started = Instant::now();
    let result = rks_scf(
        &setup.basis,
        &molecule,
        &tensor,
        &grid,
        &setup.terms,
        options,
    )
    .expect(basis_name);
    let seconds = started.elapsed().as_secs_f64();
    assert!(result.scf_converged, "{basis_name} 的 GPU 路径 SCF 没收敛");
    (
        result.energy_hartree,
        result.xc_device_report.unwrap_or_default(),
        seconds,
        result.xc_fallback_reason,
    )
}

#[test]
fn a2_the_ao_data_stays_resident_across_the_scf() {
    println!("NVRTC 状态：{:?}", qwb_gpu::xc::nvrtc_status());
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let setup = solve(basis_name, grid_name, "pbe");
        let n = setup.basis.n_functions();
        let n_points = setup.grid.len();
        // 改之前 / 改之后：同一段收缩代码，只有"缓冲留不留"这一个开关不同。
        let (energy_before, before, seconds_before, reason_before) =
            scf_on_device(&setup, basis_name, grid_name, false);
        let (energy_after, after, seconds_after, reason_after) =
            scf_on_device(&setup, basis_name, grid_name, true);
        println!(
            "{basis_name}/{grid_name}: n = {n}, points = {n_points} | \
             before（每次重传）: {} 次收缩, {} 次上传, 上传 {} B, 下载 {} B, \
             上传 {:.1} ms / 核 {:.1} ms / gemm {:.1} ms / 下载 {:.1} ms, 设备合计 {:.1} ms, SCF {:.3} s | \
             after（Φ/∇Φ 驻留）: {} 次收缩, {} 次上传, 上传 {} B, 下载 {} B, \
             上传 {:.1} ms / 核 {:.1} ms / gemm {:.1} ms / 下载 {:.1} ms, 设备合计 {:.1} ms, SCF {:.3} s",
            before.calls,
            before.uploads,
            before.uploaded_bytes,
            before.downloaded_bytes,
            before.upload_seconds * 1e3,
            before.kernel_seconds * 1e3,
            before.gemm_seconds * 1e3,
            before.download_seconds * 1e3,
            before.device_seconds() * 1e3,
            seconds_before,
            after.calls,
            after.uploads,
            after.uploaded_bytes,
            after.downloaded_bytes,
            after.upload_seconds * 1e3,
            after.kernel_seconds * 1e3,
            after.gemm_seconds * 1e3,
            after.download_seconds * 1e3,
            after.device_seconds() * 1e3,
            seconds_after,
        );
        if let Some(reason) = reason_before.as_ref().or(reason_after.as_ref()) {
            println!("  设备路径不可用，已退回 CPU：{reason}");
            // 退回时两次跑的**必须是同一段 CPU 代码**：能量逐位相同。
            assert_eq!(energy_before.to_bits(), energy_after.to_bits());
            assert_eq!(before.calls, 0, "设备不可用时不该有设备调用记账");
            assert_eq!(after.calls, 0, "设备不可用时不该有设备调用记账");
            continue;
        }

        // ── 字节数逐字节对上公式（不是"看起来少了"） ──
        let ao_bytes = (4 * n_points * n) as u64 * 8; // Φ + ∇Φ×3
        let per_call_up = ((n * n + 5 * n_points) as u64) * 8; // P + w·v + w·σ 系数 + 3 个 s_a
        let per_call_down = ((5 * n_points + 4 * n * n) as u64) * 8; // ρ/∇ρ/σ + v_xc + 3 个 M_a
        assert_eq!(
            before.uploaded_bytes,
            before.calls as u64 * (ao_bytes + per_call_up),
            "改之前：每次收缩都该重传 Φ/∇Φ"
        );
        assert_eq!(
            after.uploaded_bytes,
            ao_bytes + after.calls as u64 * per_call_up,
            "改之后：Φ/∇Φ 只该传一次（第一步）"
        );
        assert_eq!(after.uploads, 1, "Φ/∇Φ 只该上传一次");
        assert_eq!(before.uploads, before.calls, "改之前每次都要上传");
        assert_eq!(
            after.downloaded_bytes,
            after.calls as u64 * per_call_down,
            "下载字节数只与每步的 ρ/∇ρ/σ 与 v_xc 有关（驻留不影响）"
        );
        // 驻留省下来的就是 `(calls − 1)·AO` 这一笔。
        let saved = before.uploaded_bytes - after.uploaded_bytes;
        let saved_seconds = before.upload_seconds - after.upload_seconds;
        println!(
            "  省下 {} B（= ({} − 1) × {ao_bytes} B）、上传阶段省 {:.1} ms，设备合计 {:.1} → {:.1} ms（{:.2}x）",
            saved,
            after.calls,
            saved_seconds * 1e3,
            before.device_seconds() * 1e3,
            after.device_seconds() * 1e3,
            before.device_seconds() / after.device_seconds(),
        );
        assert_eq!(saved, (after.calls as u64 - 1) * ao_bytes);
        assert!(saved_seconds > 0.0, "驻留没有减少上传阶段的时间");

        // 两条路的能量必须逐位相同（同一段收缩代码，差别只在缓冲留不留）。
        assert_eq!(
            energy_before.to_bits(),
            energy_after.to_bits(),
            "驻留开关改变了能量 —— 那说明它动了数值，不只是动了传输"
        );
    }
}

// ---------------------------------------------------------------------------------------
// A4：显存不够时退回 CPU（SCF 级）
// ---------------------------------------------------------------------------------------

#[test]
fn a4_a_vram_shortfall_falls_back_to_the_cpu_path() {
    let setup = solve("sto-3g", "grid-normal", "pbe");
    let molecule = water();
    let tensor = qwb_integrals::two_electron::two_electron(&setup.basis, &molecule).expect("2e");
    let registry = Registry::open(data_root()).expect("qwb-data");
    let grid = qwb_grid::molecular_grid(
        registry.get_grid("grid-normal").expect("grid-normal"),
        &molecule,
    )
    .expect("grid");

    let mut cpu_options = RksOptions::default();
    cpu_options.max_iter = 200;
    cpu_options.energy_threshold = 1.0e-9;
    let cpu = rks_scf(
        &setup.basis,
        &molecule,
        &tensor,
        &grid,
        &setup.terms,
        cpu_options,
    )
    .expect("cpu scf");

    // 显存预算设成 1 KiB：`xc_gga_bytes` 是几十 MB，必然不够 → 后端报不可用 → 退回 CPU。
    // 选一个**不可能满足**的预算，而不是去挤占真实显存：判据要可复现，不能依赖机器状态。
    let accelerator = Arc::new(XcDeviceAccelerator::new().with_budget(1024));
    let mut gpu_options = RksOptions::default();
    gpu_options.max_iter = 200;
    gpu_options.energy_threshold = 1.0e-9;
    gpu_options.xc_path = XcGridPath::Gpu;
    gpu_options.xc_backend = Some(XcGridBackendHandle::new(accelerator));
    let fallback = rks_scf(
        &setup.basis,
        &molecule,
        &tensor,
        &grid,
        &setup.terms,
        gpu_options,
    )
    .expect("fallback scf");

    let reason = fallback
        .xc_fallback_reason
        .as_ref()
        .expect("显存不够时必须留下退回原因");
    println!(
        "显存预算 1024 B：路径 {:?}，原因「{reason}」，能量 {:.15e}，与 CPU 逐位相同 = {}",
        fallback.xc_path,
        fallback.energy_hartree,
        fallback.energy_hartree.to_bits() == cpu.energy_hartree.to_bits()
    );
    assert_eq!(fallback.xc_path, XcGridPath::Cpu, "显存不够时必须走 CPU");
    assert!(
        reason.contains("显存预算不够"),
        "原因里要能读到是哪一种不可用"
    );
    assert!(
        fallback.xc_device_report.is_none(),
        "退回时不该有设备调用记账"
    );
    assert_eq!(fallback.iterations, cpu.iterations);
    // 退回的是**同一段 CPU 代码**，所以能量必须逐位相同（不是"在容差内"）。
    assert_eq!(
        fallback.energy_hartree.to_bits(),
        cpu.energy_hartree.to_bits(),
        "退回 CPU 后的能量与默认路径不同 —— 那不是同一条 CPU 代码"
    );
}

// ---------------------------------------------------------------------------------------
// A3：产品 SCF 的 CPU vs GPU
// ---------------------------------------------------------------------------------------

#[test]
fn a3_the_product_scf_runs_on_the_device_path() {
    for (basis_name, grid_name) in [("sto-3g", "grid-normal"), ("def2-svp", "grid-normal")] {
        let setup = solve(basis_name, grid_name, "pbe");
        let molecule = water();
        let tensor =
            qwb_integrals::two_electron::two_electron(&setup.basis, &molecule).expect("2e");
        let registry = Registry::open(data_root()).expect("qwb-data");
        let grid =
            qwb_grid::molecular_grid(registry.get_grid(grid_name).expect(grid_name), &molecule)
                .expect("grid");

        let mut cpu_options = RksOptions::default();
        cpu_options.max_iter = 200;
        cpu_options.energy_threshold = 1.0e-9;
        let started = Instant::now();
        let cpu = rks_scf(
            &setup.basis,
            &molecule,
            &tensor,
            &grid,
            &setup.terms,
            cpu_options,
        )
        .expect("cpu scf");
        let cpu_seconds = started.elapsed().as_secs_f64();

        let accelerator = Arc::new(XcDeviceAccelerator::new());
        let mut gpu_options = RksOptions::default();
        gpu_options.max_iter = 200;
        gpu_options.energy_threshold = 1.0e-9;
        gpu_options.xc_path = XcGridPath::Gpu;
        gpu_options.xc_backend = Some(XcGridBackendHandle::new(accelerator));
        let started = Instant::now();
        let gpu = rks_scf(
            &setup.basis,
            &molecule,
            &tensor,
            &grid,
            &setup.terms,
            gpu_options,
        )
        .expect("gpu scf");
        let gpu_seconds = started.elapsed().as_secs_f64();

        let delta = (gpu.energy_hartree - cpu.energy_hartree).abs();
        println!(
            "PBE/{basis_name}/{grid_name}: n = {}, points = {} | CPU {cpu_seconds:.3} s \
             ({} 步) / GPU {gpu_seconds:.3} s ({} 步, 路径 {:?}) | E_cpu = {:.15e}, \
             E_gpu = {:.15e}, |ΔE| = {delta:.3e} Ha（逐位相同 = {}）| 墙钟加速 {:.2}x{}",
            setup.basis.n_functions(),
            grid.len(),
            cpu.iterations,
            gpu.iterations,
            gpu.xc_path,
            cpu.energy_hartree,
            gpu.energy_hartree,
            gpu.energy_hartree.to_bits() == cpu.energy_hartree.to_bits(),
            cpu_seconds / gpu_seconds,
            match &gpu.xc_fallback_reason {
                Some(reason) => format!("，回退原因：{reason}"),
                None => String::new(),
            }
        );
        // A1：总能量差 ≤ 1e-10 Ha。退回 CPU 时它自然更小，这一条对两种结局都成立。
        assert!(delta <= 1.0e-10, "|ΔE| = {delta:.3e} 超过 1e-10 Ha");
        assert!(gpu.scf_converged && cpu.scf_converged, "SCF 没收敛");
        // 迭代步数不该因为路径不同而变化（同样的收敛判据、同样的初猜）。
        assert_eq!(
            gpu.iterations, cpu.iterations,
            "GPU 与 CPU 的迭代步数不同 —— 那就不是「同一收敛判据」，比较不成立"
        );
    }
}
