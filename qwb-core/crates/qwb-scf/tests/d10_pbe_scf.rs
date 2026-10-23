//! D-10 端到端：PBE 必须真的能跑起来，且 LDA 结果不受影响（回归）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

#[test]
fn pbe_runs_end_to_end_and_lda_is_unchanged() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();

    let mut options = RksOptions::default();
    options.max_iter = 400;
    let lda = rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "svwn5",
        "grid-normal",
        options.clone(),
    )
    .expect("svwn5");
    println!(
        "SVWN5: E = {:.10}  E_xc = {:.10}  iters = {}",
        lda.energy_hartree, lda.exchange_correlation_hartree, lda.iterations
    );
    assert!(
        (lda.energy_hartree + 1.157_000_499_1).abs() < 1.0e-8,
        "LDA 结果变了：{}",
        lda.energy_hartree
    );

    let pbe = rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "pbe",
        "grid-normal",
        options,
    )
    .expect("pbe");
    println!(
        "PBE:   E = {:.10}  E_xc = {:.10}  iters = {}  converged = {}",
        pbe.energy_hartree, pbe.exchange_correlation_hartree, pbe.iterations, pbe.scf_converged
    );
    assert!(pbe.scf_converged, "PBE 必须收敛");
    assert!(pbe.energy_hartree.is_finite());
    let difference = (pbe.energy_hartree - lda.energy_hartree).abs();
    assert!(
        difference < 0.1,
        "PBE 与 LDA 的总能量差 {difference:.4} Ha 太大，可能有实现错误"
    );
    println!(
        "E_xc 差 = {:.6} Ha",
        pbe.exchange_correlation_hartree - lda.exchange_correlation_hartree
    );
}
