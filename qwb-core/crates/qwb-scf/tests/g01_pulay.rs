//! G-01 前置：Pulay W 矩阵的恒等式 `Tr(F dP) = −Tr(W dS)`。
//!
//! 用两个**各自收敛**的相邻几何做差分——不依赖任何外部参照，也不依赖梯度实现本身。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_scf::build_fock;
use qwb_scf::pulay::{pulay_residual, pulay_w};
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

struct Geometry {
    basis: InternalBasis,
    overlap: qwb_linalg::Matrix,
    fock: qwb_linalg::Matrix,
    density: qwb_linalg::Matrix,
}

fn solve(registry: &Registry, molecule: &Molecule) -> Geometry {
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = InternalBasis::build(basis_k1, molecule).expect("build");
    let one = one_electron(&basis, molecule).expect("1e");
    let tensor = two_electron(&basis, molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("scf");
    let fock = build_fock(&one.core_hamiltonian, &tensor, &result.density).expect("fock");
    Geometry {
        basis,
        overlap: one.overlap,
        fock: fock.fock,
        density: result.density,
    }
}

fn h2(bond: f64) -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -bond / 2.0]),
            Atom::new("H", [0.0, 0.0, bond / 2.0]),
        ],
        0,
        1,
    )
}

#[test]
fn pulay_identity_holds_for_a_finite_displacement() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let reference = solve(&registry, &h2(0.74));
    let n = reference.basis.n_functions();

    for delta in [1.0e-3_f64, 1.0e-4] {
        let displaced = solve(&registry, &h2(0.74 + delta));
        let w = pulay_w(&reference.density, &reference.fock).expect("w");

        // dP、dS
        let mut d_density = qwb_linalg::Matrix::zeros(n, n);
        let mut d_overlap = qwb_linalg::Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                d_density[(i, j)] = displaced.density[(i, j)] - reference.density[(i, j)];
                d_overlap[(i, j)] = displaced.overlap[(i, j)] - reference.overlap[(i, j)];
            }
        }
        let residual = pulay_residual(&reference.fock, &d_density, &w, &d_overlap);

        // 量级参考：Tr(F dP) 与 Tr(W dS) 各自的大小
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale += (reference.fock[(i, j)] * d_density[(i, j)]).abs()
                    + (w[(i, j)] * d_overlap[(i, j)]).abs();
            }
        }
        let relative = residual.abs() / scale.max(1.0e-30);
        println!(
            "δ = {delta:.0e}：Tr(F dP) + Tr(W dS) = {residual:.3e}（两项量级 {scale:.3e}，相对 {relative:.2e}）"
        );
        // 一阶恒等式 ⇒ 残差相对量级应与 δ 同阶（二阶小量 / 一阶量）
        assert!(
            relative < 20.0 * delta,
            "δ={delta:.0e} 时相对残差 {relative:.2e} 过大，W 的系数或符号可能不对"
        );
    }
}
