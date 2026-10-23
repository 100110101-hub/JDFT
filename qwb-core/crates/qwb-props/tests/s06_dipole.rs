//! S-06 验收（偶极部分）：H2 偶极为零、中性分子偶极与原点无关、量级合理。
#![allow(clippy::needless_range_loop)] // 三个分量的断言用下标更清楚

use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_props::dipole_debye;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn solve(molecule: &Molecule) -> (InternalBasis, Matrix) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = InternalBasis::build(basis_k1, molecule).expect("build");
    let _one = one_electron(&basis, molecule).expect("1e");
    let tensor = two_electron(&basis, molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("scf");
    (basis, result.density)
}

#[test]
fn symmetric_molecules_have_zero_dipole() {
    let h2 = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let (basis, density) = solve(&h2);
    let mu = dipole_debye(&basis, &h2, &density).expect("dipole");
    println!("H2: μ = {mu:?}");
    for axis in 0..3 {
        assert!(
            mu[axis].abs() < 1.0e-9,
            "H2 的 μ[{axis}] = {} 应为 0",
            mu[axis]
        );
    }
}

#[test]
fn dipoles_are_independent_of_the_origin_for_a_neutral_molecule() {
    // 同一個分子平移到别处：中性分子的偶极矩是平移不变的物理量。
    let origin = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    let shifted = Molecule::new(
        vec![
            Atom::new("C", [1.0, 2.0, 3.0]),
            Atom::new("H", [1.0, 2.757, 3.586]),
            Atom::new("H", [1.0, 1.243, 3.586]),
        ],
        0,
        1,
    );
    let (basis_a, density_a) = solve(&origin);
    let (basis_b, density_b) = solve(&shifted);
    let mu_a = dipole_debye(&basis_a, &origin, &density_a).expect("a");
    let mu_b = dipole_debye(&basis_b, &shifted, &density_b).expect("b");
    println!("原点处 μ = {mu_a:?}\n平移后 μ = {mu_b:?}");
    for (axis, (a, b)) in mu_a.iter().zip(&mu_b).enumerate() {
        assert!((a - b).abs() < 1.0e-8, "μ[{axis}] 随原点变了：{a} vs {b}");
    }
}

#[test]
fn dipole_magnitude_is_physically_reasonable() {
    // CH2（水几何）：STO-3G 下应是非零的几十分之一到几个 Debye。
    let ch2 = Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    let (basis, density) = solve(&ch2);
    let mu = dipole_debye(&basis, &ch2, &density).expect("dipole");
    let magnitude = (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt();
    println!("CH2: μ = {mu:?}  |μ| = {magnitude:.4} D");
    assert!(
        (0.1..5.0).contains(&magnitude),
        "CH2 的 |μ| = {magnitude} D 量级可疑"
    );
    // 分子在 yz 平面内 ⇒ μ_x 必须为 0（对称性）
    assert!(mu[0].abs() < 1.0e-9, "μ_x = {} 应为 0", mu[0]);
}
