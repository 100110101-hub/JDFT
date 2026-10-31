//! M4 / RS-02..03 的引擎级验收：`options.excited_states` 真的进 `result.json`。
//!
//! 三条：
//! 1. `method: tda`（hf 参考）跑通，`excited_states` 块带 `transitions` 与 `sum_rules`；
//! 2. `method: cis` 与 `tda` 在同一参考下给同样的数（RS-03 的 1e-6 a.u. 判据）；
//! 3. 非 `hf` 参考显式拒绝（K5 退出码 4），不静默降级；`method: tddft` 自 ST-01 起**能跑**，
//!    所以这里改成断言它跑通并且逐态低于 `tda`（引擎级的那条 ST-01 验收）。
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
fn tddft_runs_and_sits_below_tda() {
    // ST-01 之前这里是"tddft 被拒并点名 ST-01"；现在 ST-01 已交付，断言反过来：
    // tddft 跑通、报告自己的 method、且每个态都低于 tda（B 块的物理方向）。
    let mut energies = Vec::new();
    for method in [ExcitedStateMethod::Tda, ExcitedStateMethod::Tddft] {
        let options =
            TaskOptions::default().with_excited_states(ExcitedStatesOptions::new(3, method));
        let result = backend()
            .run(&water_input("hf", options), &NullReporter)
            .expect("tda/tddft should run");
        let block = result.excited_states.as_ref().expect("block");
        assert_eq!(block.method, method);
        energies.push(
            block
                .transitions
                .iter()
                .map(|t| t.energy_hartree)
                .collect::<Vec<_>>(),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            json.contains(&format!("\"method\":\"{}\"", method.as_str())),
            "method must be reported as itself: {json}"
        );
    }
    for (index, (full, tda)) in energies[1].iter().zip(energies[0].iter()).enumerate() {
        println!(
            "state {}: tddft {:.6} Ha vs tda {:.6} Ha",
            index + 1,
            full,
            tda
        );
        assert!(
            full < tda,
            "state {index}: tddft {full} should be below tda {tda}"
        );
    }
}

#[test]
fn a_dft_reference_is_rejected_and_names_the_xc_kernel() {
    let options = TaskOptions::default()
        .with_excited_states(ExcitedStatesOptions::new(3, ExcitedStateMethod::Tda));
    let error = backend()
        .run(&water_input("pbe", options), &NullReporter)
        .expect_err("TDA with a DFT reference must be rejected");
    match error {
        EngineError::Unsupported(message) => {
            assert!(
                message.contains("XC kernel") && message.contains("FX-03"),
                "a DFT reference lacks the XC kernel, and the message must say so: {message}"
            );
        }
        other => panic!("expected Unsupported (exit code 4), got {other:?}"),
    }
}
