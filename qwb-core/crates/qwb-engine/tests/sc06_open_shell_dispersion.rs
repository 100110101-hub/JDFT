//! SC-06（第一半）：**开壳层 × D3** 的最小可用组合。
//!
//! SC-06 的两条判据是：「至少 1 个开壳层体系在 M06-2X/D3 下跑通并给出梯度」与
//! 「`--threads` 归约仍确定性」。这一轮只能交付**第二半的前半**：
//!
//! * `scf_type = uks` + `options.dispersion` 在开壳层体系上跑通，色散块进结果，
//!   并且线程数不同也逐位相同（色散是纯函数、逐对归约，没有原子化的并行写）；
//! * `scf_type = uhf` + 色散同理（自旋无关的项，两条自旋通道都只影响 SCF 那半）。
//!
//! 第一半（**M06-2X + 开壳层**）仍未做，原因是实现缺口而不是配置：`uks.rs` 明确拒绝
//! 自旋分辨的 meta-GGA（`spin-resolved meta-GGA (τ_α/τ_β) is not implemented yet`），
//! 而形如 M06 的交换项 τ 依赖无法用闭壳层的标度拆成 α/β 通道。这条写在
//! `docs/status-v2.yaml` 的 SC-06 里，不在这里假装跑通。

use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, DispersionOptions, EngineBackend, GridLevel, JobInput, JobType, Molecule, ScfType,
    TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data must load"))
}

/// O2 三重态（STO-3G）：开壳层的标准小体系。
fn dioxygen() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.6035]),
            Atom::new("O", [0.0, 0.0, -0.6035]),
        ],
        0,
        3,
    )
}

fn request(scf_type: ScfType, threads: u32) -> JobInput {
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    options.threads = threads;
    options.scf_type = Some(scf_type);
    options.dispersion = Some(DispersionOptions::default());
    // `uhf` 只接受 HF（配 DFT 泛函会被显式拒绝），`uks` 要走 DFT 泛函。
    let method = match scf_type {
        ScfType::Uhf => "hf",
        _ => "pbe",
    };
    JobInput::new(
        dioxygen(),
        "sto-3g",
        method,
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        options,
    )
}

#[test]
fn sc06_open_shell_with_dispersion_runs_and_is_thread_deterministic() {
    let backend = backend();
    for scf_type in [ScfType::Uks, ScfType::Uhf] {
        let single = backend
            .run(&request(scf_type, 1), &NullReporter)
            .unwrap_or_else(|err| panic!("{scf_type:?} + D3 must run: {err}"));
        let dispersion = single
            .dispersion_energy_hartree
            .unwrap_or_else(|| panic!("{scf_type:?}: the dispersion block must be present"));
        let four = backend
            .run(&request(scf_type, 4), &NullReporter)
            .unwrap_or_else(|err| panic!("{scf_type:?} + D3 (4 threads): {err}"));
        println!(
            "{scf_type:?}: E = {:.10} Ha, E_disp = {dispersion:.6e} Ha, \
             threads 1 vs 4 identical = {}",
            single.energy_hartree,
            single.energy_hartree == four.energy_hartree
        );
        // ① 色散是负的、量级合理（O2 的分子内 D3(BJ) 在 −1e-3 Ha 量级）。
        assert!(
            dispersion < 0.0 && dispersion > -5.0e-3,
            "{scf_type:?}: E_disp = {dispersion}"
        );
        // ② 线程数不同 → 总能量逐位相同（色散逐对归约、SCF 走确定性归约）。
        assert_eq!(
            single.energy_hartree, four.energy_hartree,
            "{scf_type:?}: energy must not depend on the thread count"
        );
        assert_eq!(
            single.dispersion_energy_hartree,
            four.dispersion_energy_hartree
        );
        // ③ 总能量 = SCF + 色散（可加项的不变量）。
        let total = single.energy_hartree;
        let electronic = total - dispersion;
        assert!(
            electronic < 0.0 && electronic > total,
            "{scf_type:?}: {electronic}"
        );
    }
}

#[test]
fn sc06_meta_gga_with_an_open_shell_is_rejected_with_a_reason() {
    let backend = backend();
    let mut options = TaskOptions::default();
    options.scf_max_iter = 50;
    options.scf_type = Some(ScfType::Uks);
    let input = JobInput::new(
        dioxygen(),
        "sto-3g",
        "m06-2x",
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        options,
    );
    let error = backend
        .run(&input, &NullReporter)
        .expect_err("spin-resolved meta-GGA is not implemented");
    let message = format!("{error}");
    println!("uks + m06-2x: {message}");
    // 实际报的是「这个组件没有自旋分辨实现」——`MggaXM062x` 连 spin_supported 都没过，
    // 所以还轮不到那条更具体的 τ 报错。两条都属于「开壳层 meta-GGA 不存在」。
    assert!(
        message.contains("no spin-resolved implementation") && message.contains("uks"),
        "the rejection must explain why open-shell meta-GGA is unavailable: {message}"
    );
}
