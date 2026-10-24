//! f 变换落地后的冒烟验收：H2O 在三个基组上的 HF 能量必须彼此接近（基组收敛）。
//!
//! 这条测试的价值不在"某个数对不对"，而在于：**def2-TZVP 现在能算了**。
//! 之前它直接报 `spherical transformation for l=3 is not implemented`。
//!
//! 判据：三个基组（def2-SVP ⊂ 6-311G(d,p) ⊂ def2-TZVP 大致同档）的 HF 总能量
//! 彼此相差 < 0.2 Ha，且 def2-TZVP 是最低（基组越大能量越低）。这是一个与
//! 外部参考无关、但足以发现「f 变换把能量搞飞」的量级检查。
#![allow(clippy::field_reassign_with_default)]
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

fn hf_energy(basis_name: &str) -> (f64, usize) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
    (result.energy_hartree, basis.n_functions())
}

#[test]
fn hf_energy_is_consistent_across_basis_sets() {
    let mut energies = Vec::new();
    for name in ["def2-svp", "6-311g(d,p)", "def2-tzvp"] {
        let (energy, functions) = hf_energy(name);
        println!("{name:<12} {functions:>3} 个基函数   HF = {energy:.8} Ha");
        energies.push((name, energy));
    }
    // 两两相差 < 0.2 Ha（它们都是同档基组）
    for window in energies.windows(2) {
        let difference = (window[0].1 - window[1].1).abs();
        assert!(
            difference < 0.2,
            "{} 与 {} 的 HF 能量差 {difference:.3} Ha，太大（f 变换可能有问题）",
            window[0].0,
            window[1].0
        );
    }
    // 基组更大 → 能量更低（变分原理）
    let smallest = energies
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!("最低的是 {}（{}）", smallest.0, smallest.1);
}
