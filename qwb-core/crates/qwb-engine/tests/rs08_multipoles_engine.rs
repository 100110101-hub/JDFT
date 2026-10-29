//! RS-08 的引擎侧验收：`result.json` 的 `multipoles` 块带无迹四极矩与原點。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, TaskOptions};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn rs08_engine_emits_the_quadrupole() {
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    );
    let mut input = JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default(),
    );
    input.options.scf_energy_threshold_hartree = 1.0e-10;
    let job = backend.run(&input, &NullReporter).expect("run");
    let multipoles = job.multipoles.as_ref().expect("K7 multipoles block");
    let theta = multipoles
        .quadrupole_traceless_debye_angstrom
        .expect("quadrupole");
    let trace = theta[0][0] + theta[1][1] + theta[2][2];
    println!(
        "Θ_zz = {:+.9} D·Å, trace = {trace:+.2e}, origin = {:?} Å",
        theta[2][2], multipoles.origin_angstrom
    );
    assert!(trace.abs() < 1.0e-9, "traceless, got trace {trace:.2e}");
    // 与 `rs08_quadrupole` 的独立计算一致（同一密度、同一原点）
    assert!(
        (theta[2][2] + 0.155_7).abs() < 0.05,
        "Θ_zz = {:.4} 与解析量级不符",
        theta[2][2]
    );
}
