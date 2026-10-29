//! SC-02 验收（引擎侧）：开壳层 `opt` 跑通，且自旋不漂移。
//!
//! 判据（project-phase2.md §6.7 SC-02）：「`opt` 在开壳层跑通（符号与自旋不变）」。
//! 梯度本身与数值参照的一致性由 `qwb-scf` 的 `sc02_uhf_gradient` 钉住（OH 双重态
//! 3.63e-7、O2 三重态 7.29e-7 Ha/Bohr，判据 1e-6）。
//!
//! 这里再做三件事：
//! 1. OH（双重态）用 `scf_type: uhf` + `jobtype: opt` 真的收敛；
//! 2. 收尾的 `spin` 块多重度仍是 2、污染 ≈ 0（自旋没被优化器带跑）；
//! 3. UKS + `opt` 仍然显式拒绝（SC-02 的剩余一半），不静默降级到闭壳层。
use std::path::PathBuf;

use qwb_engine::{OptSink, RksBackend};
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, ScfType, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn hydroxyl() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.020]),
        ],
        0,
        2,
    )
}

fn oh_input(jobtype: JobType, scf_type: ScfType) -> JobInput {
    let mut input = JobInput::new(
        hydroxyl(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        jobtype,
        TaskOptions::default().with_scf_type(scf_type),
    );
    input.options.scf_energy_threshold_hartree = 1.0e-10;
    input.options.scf_density_threshold = 1.0e-8;
    input.options.opt_gradient_threshold_hartree_per_bohr = 4.5e-4;
    input.options.max_opt_steps = 30;
    input
}

#[test]
fn sc02_open_shell_optimisation_runs_and_keeps_the_spin() {
    let sink = OptSink::new();
    let backend = backend().with_opt_sink(sink.clone());
    let result = backend
        .run(&oh_input(JobType::Opt, ScfType::Uhf), &NullReporter)
        .expect("open-shell optimisation must run (SC-02)");
    let report = sink.take().expect("the opt report must be available");
    println!(
        "opt report: converged = {}, iterations = {}, used_analytic = {}, fallback = {:?}",
        report.converged,
        report.iterations,
        report.used_analytic_gradient,
        report.analytic_fallback
    );
    assert!(
        report.used_analytic_gradient,
        "the open-shell optimisation must use the analytic gradient, fell back: {:?}",
        report.analytic_fallback
    );
    let gradient = result
        .gradient_hartree_per_bohr
        .clone()
        .expect("optimisation must report the final gradient");
    let max_force = gradient
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let spin = result.spin.expect("the spin block must be emitted (K9)");
    println!(
        "OH opt: E = {:.9} Ha, max|grad| = {max_force:.3e} Ha/Bohr, multiplicity = {}, \
         <S^2> = {:.6}, contamination = {:.2e}",
        result.energy_hartree, spin.multiplicity, spin.s_squared, spin.spin_contamination
    );
    println!("final geometry:");
    for atom in &report.molecule.atoms {
        println!("  {} {:?}", atom.symbol, atom.position_angstrom);
    }
    assert!(
        max_force < 4.5e-4,
        "optimisation must converge below the force threshold, got {max_force:.3e}"
    );
    assert_eq!(
        spin.multiplicity, 2,
        "the spin must not drift during optimisation"
    );
    assert_eq!(
        spin.n_alpha - spin.n_beta,
        1,
        "one unpaired electron for a doublet"
    );
    // UHF 的污染是方法固有的：OH/STO-3G 实测 <S^2> = 0.7543（精确 0.75），污染 4.3e-3。
    // 这里钉的是「优化过程没把自旋带跑」（量级不变），不是「零污染」。
    assert!(
        spin.spin_contamination.abs() < 1.0e-2,
        "spin contamination {:.2e} for a UHF doublet",
        spin.spin_contamination
    );
    // 优化后的 O–H 键长：STO-3G/UHF 下应当在 0.90–1.05 Å 之间
    let atoms = &report.molecule.atoms;
    let dx = atoms[1].position_angstrom[0] - atoms[0].position_angstrom[0];
    let dy = atoms[1].position_angstrom[1] - atoms[0].position_angstrom[1];
    let dz = atoms[1].position_angstrom[2] - atoms[0].position_angstrom[2];
    let bond = (dx * dx + dy * dy + dz * dz).sqrt();
    println!("optimised O–H = {bond:.4} A");
    assert!(
        (0.90..1.05).contains(&bond),
        "optimised O–H bond {bond:.4} A is not physical"
    );
}

#[test]
fn sc02_uks_optimisation_is_still_rejected() {
    let error = backend()
        .run(&oh_input(JobType::Opt, ScfType::Uks), &NullReporter)
        .expect_err("UKS + opt must stay rejected until the XC spin split lands");
    let message = error.to_string();
    println!("rejected: {message}");
    assert!(message.contains("SC-02"), "{message}");
    assert!(message.contains("uks"), "{message}");
}
