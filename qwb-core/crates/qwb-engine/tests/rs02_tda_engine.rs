//! M4 / RS-02..03 的引擎级验收：`options.excited_states` 真的进 `result.json`。
//!
//! 三条：
//! 1. `method: tda`（hf 参考）跑通，`excited_states` 块带 `transitions` 与 `sum_rules`；
//! 2. `method: cis` 与 `tda` 在同一参考下给同样的数（RS-03 的 1e-6 a.u. 判据）；
//! 3. `method: tddft` 与非 `hf` 参考显式拒绝（K5 退出码 4），不静默降级。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, ExcitedStateMethod, ExcitedStatesOptions, GridLevel,
    JobInput, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
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

fn water_input(method: &str, options: TaskOptions) -> JobInput {
    let mut input = JobInput::new(
        water(),
        "sto-3g",
        method,
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    );
    input.options.scf_energy_threshold_hartree = 1.0e-11;
    input.options.scf_density_threshold = 1.0e-9;
    input
}

#[test]
fn tda_lands_excited_states_in_the_result() {
    let options = TaskOptions::default()
        .with_excited_states(ExcitedStatesOptions::new(3, ExcitedStateMethod::Tda));
    let result = backend()
        .run(&water_input("hf", options), &NullReporter)
        .expect("TDA should be wired");
    let block = result
        .excited_states
        .as_ref()
        .expect("excited_states block must be in the result");
    assert_eq!(block.method, ExcitedStateMethod::Tda);
    assert_eq!(block.n_states, 3);
    assert_eq!(block.transitions.len(), 3);
    println!(
        "sum_rules: trk = {:.6}, n_electrons = {:.1}, deficit = {:.6}",
        block.sum_rules.trk_oscillator,
        block.sum_rules.n_electrons,
        block.sum_rules.deficit()
    );
    for transition in &block.transitions {
        println!(
            "  state {}: {:.6} Ha = {:.4} eV, f = {:.6}",
            transition.index,
            transition.energy_hartree,
            transition.wavelength_nm,
            transition.oscillator_strength
        );
        assert!(transition.energy_hartree > 0.0);
        assert!(transition.oscillator_strength >= 0.0);
    }
    // 前 3 态与 RS-02 的 crate 级参考一致（13.1998 / 15.1244 / 16.8053 eV）。
    let expected_ev = [13.1998_f64, 15.1244, 16.8053];
    for (transition, expected) in block.transitions.iter().zip(expected_ev.iter()) {
        let ev = transition.energy_hartree * 27.211_386_245_988;
        assert!(
            (ev - expected).abs() < 0.01,
            "state {}: {ev:.4} eV vs expected {expected:.4} eV",
            transition.index
        );
    }
    // result.json 里要看得到。
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(
        json.contains("\"excited_states\""),
        "missing excited_states block"
    );
    assert!(json.contains("\"method\":\"tda\""), "method spelling");
    assert!(json.contains("\"sum_rules\""), "missing sum_rules");
}

#[test]
fn cis_and_tda_agree_at_the_engine_level() {
    let mut values = Vec::new();
    for method in [ExcitedStateMethod::Cis, ExcitedStateMethod::Tda] {
        let options =
            TaskOptions::default().with_excited_states(ExcitedStatesOptions::new(3, method));
        let result = backend()
            .run(&water_input("hf", options), &NullReporter)
            .expect("cis/tda should run");
        let block = result.excited_states.expect("block");
        values.push(
            block
                .transitions
                .iter()
                .map(|t| t.energy_hartree)
                .collect::<Vec<_>>(),
        );
    }
    for (index, (a, b)) in values[0].iter().zip(values[1].iter()).enumerate() {
        let delta = (a - b).abs();
        assert!(
            delta < 1.0e-6,
            "state {index}: cis {a} vs tda {b}, delta {delta} (criterion 1e-6 a.u.)"
        );
    }
    println!("cis == tda at the engine level (criterion 1e-6 a.u.)");
}

#[test]
fn tddft_and_dft_references_are_rejected() {
    let options = TaskOptions::default()
        .with_excited_states(ExcitedStatesOptions::new(3, ExcitedStateMethod::Tddft));
    let error = backend()
        .run(&water_input("hf", options), &NullReporter)
        .expect_err("full TDDFT must be rejected with its WP");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("ST-01"), "{message}");
        }
        other => panic!("expected Unsupported (exit code 4), got {other:?}"),
    }

    let options = TaskOptions::default()
        .with_excited_states(ExcitedStatesOptions::new(3, ExcitedStateMethod::Tda));
    let error = backend()
        .run(&water_input("pbe", options), &NullReporter)
        .expect_err("TDA with a DFT reference must be rejected");
    match error {
        EngineError::Unsupported(message) => {
            assert!(
                message.contains("closed-shell")
                    || message.contains("XC kernel")
                    || message.contains("hf"),
                "{message}"
            );
        }
        other => panic!("expected Unsupported (exit code 4), got {other:?}"),
    }
}
