//! EX-03 的**产品接缝**测试（`qwb-scf` 侧，**不需要 GPU**）。
//!
//! 这一条钉三件事：
//!
//! 1. `RksOptions` 的默认值不碰设备通道：`xc_path == Cpu`、没有退回原因、没有设备记账，
//!    能量与"没改之前"逐位相同（位级锚点另有一条 `xc_grid_bit_pattern.rs`）。
//! 2. 选了 `Gpu` 但**没注入后端**（= 没开 `gpu-kernel` 的构建、或调用方忘了注入）时，
//!    SCF 退回 CPU、留下可读的原因，而且**能量逐位不变** —— 退回的是同一段 CPU 代码。
//! 3. 注入的后端**真的**驱动数值：一个故意把 `E_xc` 抬高 `1e-6 Ha` 的假后端，必须让
//!    总能量跟着抬高同样的量。没有这一条，"注入点"可能是一段谁也没读的死代码而测试全绿。
//!
//! 假后端只用安全 Rust 写（`qwb-scf` 里连 `unsafe` 这个词都不许出现），所以它同时证明了
//! `XcGridAccelerator` 这个 trait 在 `qwb-gpu` 之外也能实现。
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚

use std::path::PathBuf;
use std::sync::Arc;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, RksResult, rks_scf};
use qwb_scf::xc_device::{
    XcGgaEvaluate, XcGridAccelerator, XcGridBackendHandle, XcGridDeviceOutcome, XcGridDeviceReport,
    XcGridPath, XcGridRequest, XcLdaEvaluate,
};
use qwb_xc::FunctionalTerm;

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

/// 在一个小算例上跑 SCF（H2O/STO-3G、grid-coarse、svwn5），返回结果。
fn run(options: RksOptions) -> RksResult {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let terms = terms_from(&registry, "svwn5");
    rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf")
}

fn base_options() -> RksOptions {
    let mut options = RksOptions::default();
    options.max_iter = 200;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-8;
    options
}

/// 一个**假**的 LDA 后端：`v_xc` 算对，但把 `E_xc` 抬高 `1e-6 Ha`。
///
/// 抬高的量是刻意选的：比 A1 的 `1e-10 Ha` 一致线大四个数量级，所以"注入的数值没进 SCF"
/// 会被这条测试当场抓住；而 `v_xc` 仍然正确，SCF 才会照常收敛 —— 否则它可能因为不收敛而
/// 报错，看起来像"接线成功了"。
#[derive(Debug)]
struct BiasedFake;

const BIAS: f64 = 1.0e-6;

impl XcGridAccelerator for BiasedFake {
    fn contract_lda(
        &self,
        request: &XcGridRequest<'_>,
        evaluate: &mut XcLdaEvaluate<'_>,
    ) -> XcGridDeviceOutcome {
        let n = request.n;
        let n_points = request.n_points;
        if request.ao_values.len() != n_points * n
            || request.density.len() != n * n
            || request.weights.len() != n_points
        {
            return XcGridDeviceOutcome::Unavailable {
                reason: "假后端：形状不合法".to_string(),
            };
        }
        let mut rho_values = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let row = &request.ao_values[point * n..(point + 1) * n];
            let mut rho = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    rho += row[i] * request.density[i * n + j] * row[j];
                }
            }
            rho_values.push(rho.max(0.0));
        }
        let (epsilon, potential) = match evaluate(&rho_values) {
            Ok(pair) => pair,
            Err(reason) => return XcGridDeviceOutcome::Unavailable { reason },
        };
        let mut energy = BIAS;
        for point in 0..n_points {
            energy += request.weights[point] * rho_values[point] * epsilon[point];
        }
        let mut v_xc = vec![0.0_f64; n * n];
        for (point, potential_here) in potential.iter().enumerate() {
            let row = &request.ao_values[point * n..(point + 1) * n];
            for i in 0..n {
                if row[i] == 0.0 {
                    continue;
                }
                for j in 0..n {
                    v_xc[i * n + j] += request.weights[point] * row[i] * row[j] * potential_here;
                }
            }
        }
        XcGridDeviceOutcome::Done {
            energy,
            v_xc,
            report: XcGridDeviceReport {
                calls: 1,
                uploads: 1,
                uploaded_bytes: 1,
                ..XcGridDeviceReport::default()
            },
        }
    }

    fn contract_gga(
        &self,
        _request: &XcGridRequest<'_>,
        _evaluate: &mut XcGgaEvaluate<'_>,
    ) -> XcGridDeviceOutcome {
        XcGridDeviceOutcome::Unavailable {
            reason: "假后端只实现 LDA 通道".to_string(),
        }
    }

    fn name(&self) -> &'static str {
        "biased fake"
    }
}

#[test]
fn a1_the_default_options_never_touch_the_device_path() {
    let result = run(base_options());
    println!(
        "默认选项：路径 {:?}，退回原因 {:?}，设备记账 {:?}，E = {:.15e}，{} 步",
        result.xc_path,
        result.xc_fallback_reason,
        result.xc_device_report,
        result.energy_hartree,
        result.iterations
    );
    assert_eq!(result.xc_path, XcGridPath::Cpu);
    assert!(result.xc_fallback_reason.is_none());
    assert!(result.xc_device_report.is_none());
    assert!(result.scf_converged);
}

#[test]
fn a2_a_gpu_path_without_a_backend_falls_back_bit_identically() {
    let cpu = run(base_options());
    let mut options = base_options();
    options.xc_path = XcGridPath::Gpu;
    // `xc_backend` 保持 `None` —— 这就是「没开 gpu-kernel 的构建」的样子。
    let fallback = run(options);
    let reason = fallback
        .xc_fallback_reason
        .as_ref()
        .expect("选了 Gpu 却没有后端时必须留下原因");
    println!(
        "选了 Gpu、没注入后端：路径 {:?}，原因「{reason}」，E = {:.15e}，与默认逐位相同 = {}",
        fallback.xc_path,
        fallback.energy_hartree,
        fallback.energy_hartree.to_bits() == cpu.energy_hartree.to_bits()
    );
    assert_eq!(fallback.xc_path, XcGridPath::Cpu);
    assert!(fallback.xc_device_report.is_none());
    assert_eq!(fallback.iterations, cpu.iterations);
    assert_eq!(
        fallback.energy_hartree.to_bits(),
        cpu.energy_hartree.to_bits(),
        "退回 CPU 后的能量与默认路径不同 —— 那不可能是同一段 CPU 代码"
    );
}

#[test]
fn a3_an_injected_backend_really_drives_the_numbers() {
    let cpu = run(base_options());
    let mut options = base_options();
    options.xc_path = XcGridPath::Gpu;
    options.xc_backend = Some(XcGridBackendHandle::new(Arc::new(BiasedFake)));
    let biased = run(options);
    let report = biased
        .xc_device_report
        .as_ref()
        .expect("注入的后端被调用过才有记账");
    let shift = biased.energy_hartree - cpu.energy_hartree;
    println!(
        "注入假后端（E_xc +{BIAS:.1e} Ha）：路径 {:?}，{} 次收缩 / {} 次上传，\
         E_gpu − E_cpu = {shift:.6e} Ha，步数 {} vs {}",
        biased.xc_path, report.calls, report.uploads, biased.iterations, cpu.iterations
    );
    assert_eq!(biased.xc_path, XcGridPath::Gpu, "注入的后端没有被用上");
    assert_eq!(
        report.calls as usize, biased.iterations as usize,
        "每一步 SCF 都该恰好调一次设备收缩"
    );
    // 抬高量与 `1e-6` 同量级：差值必须远大于 A1 的 1e-10 一致线。
    assert!(
        (shift - BIAS).abs() < 1.0e-8,
        "总能量只挪了 {shift:.3e} Ha，与注入的 +{BIAS:.1e} 不符 —— 注入点可能没接上"
    );
    assert!(biased.scf_converged, "假后端的 v_xc 正确，SCF 应该仍然收敛");
}
