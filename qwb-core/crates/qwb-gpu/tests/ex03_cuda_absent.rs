//! EX-03 A4：**这台机器没有 CUDA 运行库时**，产品 SCF 必须退回 CPU 且不 panic。
//!
//! # 为什么用子进程
//!
//! 「没有 NVIDIA 驱动 / 没有 NVRTC」是**环境事实**，不是代码开关：`XcSession` 是进程级
//! `OnceLock`，第一次探测失败之后整个进程都失败；而 `std::env::set_var` 在 Rust 2024 里是
//! `unsafe`（本仓库 `deny(unsafe_code)`，白名单只有 `qwb-gpu/src/jit.rs`）。
//!
//! 所以这个测试**重新执行自己**：子进程的 `PATH` 里只有 `C:\Windows\System32`（`nvcuda.dll`
//! 在那里，所以驱动还能加载），CUDA toolkit 的 `bin` 被移出 `PATH`、`CUDA_PATH` /
//! `QWB_CUDA_LIB_DIR` 被清空 —— 于是 `cudart64_*.dll` / `cublas64_*.dll` /
//! `nvrtc64_*.dll` 全都找不到，正是**没装 CUDA 的机器**的样子。
//!
//! 父进程只负责：跑子进程、断言它退出码为 0。子进程里的断言（退回、能量逐位相同、
//! 原因可读）由子进程自己执行，任何一条不满足都会让退出码非零。
//!
//! 这台机器上有 CUDA 13.0，所以「有驱动、缺 NVRTC」这一种**拆不开**（`load_nvrtc` 与
//! `init` 都走同一份搜索目录列表）；能拆开的是"整套 CUDA 运行库都不在"。两者进的是同一个
//! 漏斗：`XcSession::get()` / `memory_budget_bytes()` → `GpuError::Unavailable` →
//! `XcGridDeviceOutcome::Unavailable` → 产品 CPU 路径。
//!
//! 没开 `gpu-kernel` 时 `qwb_gpu::xc_device` 不存在，所以这个文件整体不编译；那种构建下
//! 的退回由 `qwb-engine/tests/ex03_scf_gpu_xc.rs` 覆盖。
#![cfg(feature = "gpu-kernel")]
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use qwb_engine_api::{Atom, Molecule};
use qwb_gpu::xc_device::XcDeviceAccelerator;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};
use qwb_scf::xc_device::{XcGridBackendHandle, XcGridPath};
use qwb_xc::FunctionalTerm;

/// 子进程的标记：有了它就（在"没有 CUDA"的环境里）跑真正的场景。
const CHILD_MARKER: &str = "QWB_EX03_NO_CUDA_CHILD";
/// 子进程要跑的测试名（父进程用 `--exact` 指它）。
const CHILD_TEST: &str = "the_child_runs_the_scf_without_cuda";

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
        if component.weight.abs() <= 1.0e-12 || component.id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

#[test]
fn the_child_runs_the_scf_without_cuda() {
    if std::env::var_os(CHILD_MARKER).is_none() {
        // 父进程：去掉 CUDA toolkit 之后重新跑自己。
        let exe = std::env::current_exe().expect("current_exe");
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let status = Command::new(&exe)
            .args(["--exact", CHILD_TEST, "--nocapture"])
            .env(CHILD_MARKER, "1")
            .env(
                "PATH",
                format!("{system_root}\\System32;{system_root}\\System32\\Wbem"),
            )
            .env_remove("CUDA_PATH")
            .env_remove("CUDA_HOME")
            .env_remove("QWB_CUDA_LIB_DIR")
            .status()
            .expect("spawn the child test process");
        println!("子进程（PATH 里没有 CUDA toolkit）退出码：{status}");
        assert!(
            status.success(),
            "没有 CUDA 的子进程没有正常退回 CPU（退出码 {status}）"
        );
        return;
    }

    // ── 子进程：这里的环境里**没有 CUDA 运行库** ──
    println!(
        "子进程环境：CUDA_PATH = {:?}，PATH 条目数 = {}",
        std::env::var("CUDA_PATH").ok(),
        std::env::var("PATH").map_or(0, |path| path.split(';').count())
    );
    assert!(
        qwb_gpu::available().is_err(),
        "子进程里 CUDA 竟然可用 —— 这个测试的前提（PATH 里没有 toolkit）不成立"
    );

    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let terms = terms_from(&registry, "pbe");

    let mut cpu_options = RksOptions::default();
    cpu_options.max_iter = 200;
    cpu_options.energy_threshold = 1.0e-9;
    let cpu = rks_scf(&basis, &molecule, &tensor, &grid, &terms, cpu_options).expect("cpu scf");

    let mut gpu_options = RksOptions::default();
    gpu_options.max_iter = 200;
    gpu_options.energy_threshold = 1.0e-9;
    gpu_options.xc_path = XcGridPath::Gpu;
    gpu_options.xc_backend = Some(XcGridBackendHandle::new(Arc::new(
        XcDeviceAccelerator::new(),
    )));
    let fallback =
        rks_scf(&basis, &molecule, &tensor, &grid, &terms, gpu_options).expect("fallback scf");

    let reason = fallback
        .xc_fallback_reason
        .as_ref()
        .expect("没有 CUDA 时必须留下退回原因");
    println!(
        "没有 CUDA：路径 {:?}，原因「{reason}」，能量 {:.15e}，与 CPU 逐位相同 = {}",
        fallback.xc_path,
        fallback.energy_hartree,
        fallback.energy_hartree.to_bits() == cpu.energy_hartree.to_bits()
    );
    assert!(fallback.scf_converged, "退回 CPU 后 SCF 必须仍然收敛");
    assert_eq!(fallback.xc_path, XcGridPath::Cpu);
    assert!(!reason.is_empty(), "退回原因不能是空字符串");
    assert!(
        fallback.xc_device_report.is_none(),
        "退回时不该有设备调用记账"
    );
    assert_eq!(
        fallback.energy_hartree.to_bits(),
        cpu.energy_hartree.to_bits(),
        "退回 CPU 后的能量与默认路径不同 —— 那不是同一条 CPU 代码"
    );
}
