//! 外部参考：H2O/STO-3G 的 Hartree–Fock 总能量。
//!
//! 这是一个含 p 函数的分子，走的是「核吸引 + 2e 张量 + SCF」全链路，所以它同时
//! 校验 I-02/I-03 的 p 型结果与 SCF 装配。文献值 HF/STO-3G H2O = −74.963 Ha。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

#[test]
fn h2o_sto3g_hf_matches_the_external_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(basis_k1, &molecule).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");

    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");

    let mo: Vec<f64> = (0..result.mo_energies_hartree.len())
        .map(|i| result.mo_energies_hartree.get(i))
        .collect();
    println!(
        "H2O/STO-3G HF: E = {:.8} Ha, iters = {}",
        result.energy_hartree, result.iterations
    );
    println!("MO: {mo:?}");

    // 文献 HF/STO-3G H2O = −74.963 Ha（实验几何）。同一文献的 MO 能级约
    // [−20.20, −1.20, −0.55, −0.42, −0.35]；我们实测
    // [−20.24, −1.27, −0.62, −0.45, −0.39]。
    // NIST CCCBDB（Release 22）：HF/STO-3G 水 = −75.204892 Ha（STO-3G 优化几何）。
    let reference = -74.963_f64;
    let difference = (result.energy_hartree - reference).abs();
    assert!(
        difference < 1.0e-3,
        "HF 能量 {:.6} 与文献 {reference} 差 {difference:.4} Ha",
        result.energy_hartree
    );
}
