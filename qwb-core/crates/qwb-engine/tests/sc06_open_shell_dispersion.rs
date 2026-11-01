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
fn sc06_open_shell_meta_gga_with_dispersion_runs_end_to_end() {
    // 判据 ①「至少 1 个开壳层体系在 M06-2X/D3 下跑通」：`uks + m06-2x + d3zero` 在 O2 三重态
    // 上跑通，色散块进结果、总能量 = SCF + 色散。参考值来自 PySCF 2.14.0（同一几何/基组，
    // `qwb-scf/tests/reference/uks_mgga.csv`，O2/STO-3G 的 M06-2X = −148.274954768331 Ha）；
    // 这里只断言「跑通 + 可加性」这条判据本身，能量对表在
    // `qwb-scf/tests/sc06_uks_mgga.rs` 里做（0.5 mHa 判据，实测 4.5e-5）。
    let backend = backend();
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    options.scf_type = Some(ScfType::Uks);
    // M06-L / M06-2X **没有发表的 D3(BJ) 参数**，只有零阻尼（`d3zero`）那套
    // （见 `qwb-data/dispersion/d3zero.yaml` 的 `zero_damping_parameters`）；
    // 所以 SC-06 这条判据在 M06-2X 上只能配 D3(0)，不是配 D3(BJ)。
    options.dispersion = Some({
        let mut dispersion = DispersionOptions::default();
        dispersion.data = "d3zero".to_string();
        dispersion
    });
    let input = JobInput::new(
        dioxygen(),
        "sto-3g",
        "m06-2x",
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        options,
    );
    let result = backend
        .run(&input, &NullReporter)
        .expect("uks + m06-2x + d3zero must run (SC-06 τ wiring)");
    let dispersion = result
        .dispersion_energy_hartree
        .expect("the dispersion block must be present");
    println!(
        "uks + m06-2x + d3zero on O2: E_total = {:.10} Ha, E_disp = {:+.6e} Ha, \
         E_scf = {:.10} Ha",
        result.energy_hartree,
        dispersion,
        result.energy_hartree - dispersion
    );
    assert!(
        result.energy_hartree.is_finite(),
        "the total energy must be finite"
    );
    // 色散是负的吸引项，但在 O2 的键长（1.207 Å）上 **D3(0) 几乎为零**：
    // m06-2x 的参数集是 s6 = 1、s8 = 0、rs6 = 1.619、alp = 14（Grimme 2010 的零阻尼表），
    // 而零阻尼的 `f = 1/(1 + 6(R/(s_r6·R0))^{−14})` 在 R ≈ 2.28 Bohr、R0(O,O) ≈ 6.4 Bohr 处
    // 已经被压到 ~1e-9。所以这条断言量的是**通路**（色散块进结果、总能量可加），不是量级；
    // 量级由 `sc06_open_shell_with_dispersion_runs_and_is_thread_deterministic`
    // （PBE/HF + D3(BJ)：−2.67e-4 / −3.39e-3 Ha）负责。
    assert!(
        dispersion <= 0.0,
        "D3(0) cannot raise the energy: {dispersion:e}"
    );
    // K9 的可加性：引擎不覆盖 SCF 能量，只在总能量上加一项。
    let scf_only = result.energy_hartree - dispersion;
    assert!(scf_only < 0.0 && scf_only > result.energy_hartree);
}
