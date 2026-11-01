//! ST-05 的引擎级验收（`epr` 块进结果）：开壳层作业 + `options.epr: true` 时，
//! `result.json` 里必须有 `epr` 块，且数值与 crate 级实现一致；闭壳层打开它必须被拒绝。
//!
//! 三条：
//! 1. OH 自由基（UHF/STO-3G）→ `epr` 块出现，`method = uhf`，两个核都在，
//!    H 的自旋密度为负（自旋极化）、O 的为正；
//! 2. 同一作业不打开 `options.epr` → 块不出现（加性扩展：旧输入行为不变）；
//! 3. 闭壳层体系（水，multiplicity = 1）打开它 → `InvalidInput`（K5 退出码 1），
//!    因为闭壳层的自旋密度恒为零，报出来全是 0 等于静默失败。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn radical() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.980]),
        ],
        0,
        2,
    )
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

fn input(molecule: Molecule, epr: bool) -> JobInput {
    let mut options = TaskOptions::default().with_scf_type(qwb_engine_api::ScfType::Uhf);
    options.epr = epr;
    let mut job = JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    );
    job.options.scf_max_iter = 400;
    job.options.scf_energy_threshold_hartree = 1.0e-10;
    job.options.scf_density_threshold = 1.0e-8;
    job
}

#[test]
fn the_epr_block_lands_in_the_result() {
    let result = backend()
        .run(&input(radical(), true), &NullReporter)
        .expect("OH radical with options.epr should run");
    let block = result.epr.as_ref().expect("epr block must be present");
    assert_eq!(block.method, "uhf");
    assert_eq!(block.nuclei.len(), 2, "OH has two nuclei");
    for coupling in &block.nuclei {
        println!(
            "  {}{}: rho_s = {:+.6} a0^-3, a_iso = {:+.4} MHz = {:+.4} G",
            coupling.symbol,
            coupling.nucleus_index,
            coupling.spin_density_au,
            coupling.a_iso_mhz,
            coupling.a_iso_gauss
        );
        assert!(
            (coupling.a_iso_mhz / qwb_props::epr::MHZ_PER_GAUSS - coupling.a_iso_gauss).abs()
                < 1e-9
        );
    }
    let oxygen = &block.nuclei[0];
    let hydrogen = &block.nuclei[1];
    assert_eq!(oxygen.symbol, "O");
    assert_eq!(hydrogen.symbol, "H");
    assert!(
        oxygen.spin_density_au > 0.0,
        "the unpaired electron lives on O"
    );
    assert!(
        hydrogen.spin_density_au < 0.0,
        "the proton is spin-polarised (negative)"
    );
    // 序列化后 result.json 里看得到（K7 的加性块）。
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("\"epr\""), "missing epr block: {json}");
    assert!(json.contains("\"a_iso_mhz\""), "missing a_iso_mhz");
}

#[test]
fn the_epr_block_is_absent_unless_requested() {
    let result = backend()
        .run(&input(radical(), false), &NullReporter)
        .expect("OH radical without options.epr should run");
    assert!(result.epr.is_none());
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(!json.contains("\"epr\""), "epr must stay out of the JSON");
}

#[test]
fn a_closed_shell_epr_request_is_an_input_error() {
    let error = backend()
        .run(&input(water(), true), &NullReporter)
        .expect_err("closed-shell EPR must be rejected");
    match error {
        EngineError::InvalidInput(message) => {
            assert!(message.contains("multiplicity"), "{message}");
        }
        other => panic!("expected InvalidInput (exit code 1), got {other:?}"),
    }
}
