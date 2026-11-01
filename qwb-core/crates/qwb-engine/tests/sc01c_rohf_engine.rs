//! SC-01c 的**引擎级**验收：`scf_type: rohf` 真的能跑完并且数值对得上 PySCF。
//!
//! SCF 内核的验收在 `qwb-scf/tests/sc01c_rohf.rs`（4/5 体系对 PySCF 到 1e-8；NH2 是已知开放项）。
//! 这条测试守的是**接线**：选项被接受、能量/自旋块/偶极矩都从 ROHF 路径出来，
//! 以及"ROHF 只做 Hartree-Fock"这条限制在引擎层生效。
use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, ScfType, TaskOptions,
};
use qwb_registry::Registry;
use std::path::PathBuf;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn input(method: &str, multiplicity: u32, options: TaskOptions) -> JobInput {
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.207_5]),
        ],
        0,
        multiplicity,
    );
    JobInput::new(
        molecule,
        "sto-3g",
        method,
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    )
}

#[test]
fn the_engine_runs_rohf_and_reports_the_pure_spin() {
    let options = TaskOptions::default().with_scf_type(ScfType::Rohf);
    let result = backend()
        .run(&input("hf", 3, options), &NullReporter)
        .expect("ROHF 单点");
    // PySCF 的 ROHF/STO-3G 参考值（见 `sc01c_rohf.rs`）。
    let reference = -147.632_166_990_7;
    println!(
        "engine rohf: E = {:.10} Ha (PySCF {reference:.10}), converged = {}",
        result.energy_hartree, result.scf_converged
    );
    assert!(
        (result.energy_hartree - reference).abs() < 1.0e-5,
        "ROHF 引擎能量 {:.10} 与 PySCF {reference:.10} 差得太多",
        result.energy_hartree
    );
    assert!(result.scf_converged);
    let spin = result.spin.expect("ROHF 结果必须带 spin 块");
    assert_eq!(spin.multiplicity, 3);
    assert_eq!(spin.n_alpha, 9, "O2 三重态：9 个 α");
    assert_eq!(spin.n_beta, 7, "O2 三重态：7 个 β");
    // 纯自旋：`⟨S²⟩` 必须是精确的 S(S+1) = 2（不是 UHF 的 2.0034 之类）。
    assert!(
        (spin.s_squared - 2.0).abs() < 1.0e-12,
        "⟨S²⟩ = {}",
        spin.s_squared
    );
    assert!(result.dipole_debye.is_some(), "单点结果应带偶极矩");
    assert!(result.log.contains("scf_type = rohf"), "{}", result.log);
}

#[test]
fn a_closed_shell_rohf_request_is_refused() {
    // 多重度 1 没有开壳层 —— 必须显式拒绝，而不是悄悄当成 RHF 跑（那会静默改变方法）。
    let options = TaskOptions::default().with_scf_type(ScfType::Rohf);
    let error = backend()
        .run(&input("hf", 1, options), &NullReporter)
        .expect_err("mult 1 的 rohf 必须被拒");
    println!("rohf mult 1 -> {error}");
    assert!(
        matches!(error, qwb_engine_api::EngineError::InvalidInput(_)),
        "{error}"
    );
}
