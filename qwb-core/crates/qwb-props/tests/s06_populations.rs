//! S-06 验收（下半）：Mulliken / Löwdin 布居的和必须精确等于总电荷。
//!
//! 判据原文：「Mulliken 与 Löwdin 电荷各自求和 == 总电荷（1e-10）」。
//! 这是恒等式级别的检查（Σ (PS)_μμ = Tr(PS) = N、Tr(S^{1/2}PS^{1/2}) = Tr(PS)）。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_props::populations;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, RksResult, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn solve(molecule: &Molecule) -> (InternalBasis, RksResult, Matrix) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = InternalBasis::build(basis_k1, molecule).expect("build");
    let one = one_electron(&basis, molecule).expect("1e");
    let tensor = two_electron(&basis, molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("scf");
    (basis, result, one.overlap)
}

#[test]
fn populations_sum_to_the_total_charge() {
    let cases: Vec<(&str, Molecule, f64)> = vec![
        (
            "H2",
            Molecule::new(
                vec![
                    Atom::new("H", [0.0, 0.0, -0.37]),
                    Atom::new("H", [0.0, 0.0, 0.37]),
                ],
                0,
                1,
            ),
            2.0,
        ),
        (
            "CH2(水几何)",
            Molecule::new(
                vec![
                    Atom::new("C", [0.0, 0.0, 0.0]),
                    Atom::new("H", [0.0, 0.757, 0.586]),
                    Atom::new("H", [0.0, -0.757, 0.586]),
                ],
                0,
                1,
            ),
            8.0,
        ),
    ];

    for (label, molecule, electrons) in cases {
        let (basis, result, overlap) = solve(&molecule);
        let pops = populations(&basis, &molecule, &overlap, &result.density).expect("populations");
        let mulliken = pops.mulliken_total();
        let lowdin = pops.lowdin_total();
        println!(
            "{label}: N = {electrons}  Mulliken 和 = {mulliken:.12}  Löwdin 和 = {lowdin:.12}\n  \
             Mulliken 原子布居 = {:?}\n  Löwdin 原子布居 = {:?}",
            pops.mulliken_atoms, pops.lowdin_atoms
        );
        assert!(
            (mulliken - electrons).abs() < 1.0e-10,
            "{label}: Mulliken 和 {mulliken} ≠ {electrons}"
        );
        assert!(
            (lowdin - electrons).abs() < 1.0e-10,
            "{label}: Löwdin 和 {lowdin} ≠ {electrons}"
        );
        assert!(
            pops.charge_total().abs() < 1.0e-10,
            "{label}: 净电荷和 = {}",
            pops.charge_total()
        );
        for (i, q) in pops.lowdin_ao.iter().enumerate() {
            assert!(
                *q > -1.0e-10 && *q < 2.0 + 1.0e-10,
                "{label}: Löwdin 布居[{i}] = {q} 越界"
            );
        }
    }
}

#[test]
fn populations_are_byte_reproducible() {
    let molecule = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    );
    let (basis_a, result_a, overlap_a) = solve(&molecule);
    let (basis_b, result_b, overlap_b) = solve(&molecule);
    let pops_a = populations(&basis_a, &molecule, &overlap_a, &result_a.density).expect("a");
    let pops_b = populations(&basis_b, &molecule, &overlap_b, &result_b.density).expect("b");
    assert_eq!(pops_a, pops_b, "两次运行的布居必须完全相同");
    assert_eq!(
        pops_a
            .mulliken_ao
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        pops_b
            .mulliken_ao
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "位模式也必须相同"
    );
}
