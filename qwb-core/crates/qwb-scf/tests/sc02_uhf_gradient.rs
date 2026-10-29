//! SC-02 验收：UHF（开壳层）解析梯度 vs 数值梯度（OH 双重态、O2 三重态）。
//!
//! 判据（project-phase2.md §6.7 SC-02）：与数值梯度的最大分量差 < **1e-6 Ha/Bohr**。
//! 数值参照自身也有截断误差，所以同时给出两个步长（1e-3 / 4e-3 bohr）之间的散布，
//! 用来看"解析与数值的差"是不是被参照的散布解释掉了（与 `g01_analytic_water` 同一套做法）。
//!
//! 另外钉一条**结构判据**：闭壳层极限下（`P^α = P^β = P/2`、`W` 同样平分）
//! `uhf_analytic_gradient` 必须与 `analytic_gradient` 逐位一致 —— 这条不依赖数值梯度，
//! 能立刻抓到"交换项忘了按自旋收缩"或"系数 2 写错"这类错误。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::{
    analytic_gradient, energy_weighted_density, uhf_analytic_gradient,
};
use qwb_scf::build_fock;
use qwb_scf::rks::{RksOptions, rks_scf};
use qwb_scf::uhf::{UhfOptions, uhf_scf};

const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn hydroxyl() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.969_66]),
        ],
        0,
        2,
    )
}

fn dioxygen() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, -0.603_5]),
            Atom::new("O", [0.0, 0.0, 0.603_5]),
        ],
        0,
        3,
    )
}

fn displaced(molecule: &Molecule, atom: usize, axis: usize, delta_bohr: f64) -> Molecule {
    let mut atoms = Vec::with_capacity(molecule.atoms.len());
    for (index, item) in molecule.atoms.iter().enumerate() {
        let mut position = item.position_angstrom;
        if index == atom {
            position[axis] += delta_bohr / BOHR_PER_ANGSTROM;
        }
        atoms.push(Atom::new(item.symbol.clone(), position));
    }
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

fn uhf_energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
) -> (f64, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let mut options = UhfOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-9;
    let result = uhf_scf(&basis, molecule, &tensor, options).unwrap();
    assert!(result.scf_converged, "UHF did not converge");
    let w_alpha = energy_weighted_density(
        &result.coefficients_alpha,
        &result.mo_energies_alpha,
        result.n_alpha,
    )
    .unwrap();
    let w_beta = energy_weighted_density(
        &result.coefficients_beta,
        &result.mo_energies_beta,
        result.n_beta,
    )
    .unwrap();
    let analytic = uhf_analytic_gradient(
        &basis,
        molecule,
        &result.density_alpha,
        &result.density_beta,
        &w_alpha,
        &w_beta,
    )
    .unwrap();
    (result.energy_hartree, analytic, Vec::new())
}

fn uhf_energy(registry: &Registry, molecule: &Molecule) -> f64 {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let mut options = UhfOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-9;
    uhf_scf(&basis, molecule, &tensor, options)
        .unwrap()
        .energy_hartree
}

#[allow(clippy::needless_range_loop)]
fn numerical_gradient(registry: &Registry, molecule: &Molecule, delta_bohr: f64) -> Vec<[f64; 3]> {
    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let plus = uhf_energy(registry, &displaced(molecule, atom, axis, delta_bohr));
            let minus = uhf_energy(registry, &displaced(molecule, atom, axis, -delta_bohr));
            gradient[atom][axis] = (plus - minus) / (2.0 * delta_bohr);
        }
    }
    gradient
}

fn check(name: &str, molecule: &Molecule) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let (energy, analytic, _) = uhf_energy_and_gradient(&registry, molecule);
    let numeric = numerical_gradient(&registry, molecule, 1.0e-3);
    let coarse = numerical_gradient(&registry, molecule, 4.0e-3);
    let mut scatter = 0.0_f64;
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize);
    println!("{name}: E = {energy:.9} Ha");
    for atom in 0..analytic.len() {
        for axis in 0..3 {
            scatter = scatter.max((numeric[atom][axis] - coarse[atom][axis]).abs());
            let difference = (analytic[atom][axis] - numeric[atom][axis]).abs();
            if difference > worst {
                worst = difference;
                worst_at = (atom, axis);
            }
            println!(
                "  atom {atom} axis {axis}: 解析 {:>13.9}  数值 {:>13.9}  差 {difference:.2e}",
                analytic[atom][axis], numeric[atom][axis]
            );
        }
    }
    println!("  数值参照自身的散布（δ=1e-3 vs 4e-3）= {scatter:.2e}");
    println!(
        "  最大差 {worst:.3e} Ha/Bohr（atom {} axis {}）",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < 1.0e-6,
        "{name}: analytic vs numerical gradient differs by {worst:.3e} Ha/Bohr (criterion 1e-6)"
    );
}

#[test]
fn sc02_uhf_gradient_matches_numerical_for_oh() {
    check("OH (doublet)", &hydroxyl());
}

#[test]
fn sc02_uhf_gradient_matches_numerical_for_o2() {
    check("O2 (triplet)", &dioxygen());
}

#[test]
fn sc02_uhf_gradient_reduces_to_the_closed_shell_formula() {
    // 闭壳层极限：P^α = P^β = P/2、W 同样平分 → 两条路径必须逐位一致。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.757_0]),
            Atom::new("H", [0.0, 0.587_0, -0.468_0]),
        ],
        0,
        1,
    );
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("rks");
    let one = one_electron(&basis, &molecule).expect("1e");
    let fock = build_fock(&one.core_hamiltonian, &tensor, &result.density)
        .expect("fock")
        .fock;
    let closed = analytic_gradient(&basis, &molecule, &result.density, &fock).expect("closed");
    let w = {
        // pulay_w 用的约定是 W = ½·P F P；UHF 的两条自旋各贡献 P^σ F^σ P^σ，
        // 在闭壳层极限（P^σ = P/2）下合计 = 2·(P/2)F(P/2) = ½·P F P —— 所以每条自旋
        // 拿 **¼**·PFP。
        let p_f = qwb_linalg::matmul(&result.density, &fock).unwrap();
        qwb_linalg::matmul(&p_f, &result.density).unwrap()
    };
    let nf = basis.n_functions();
    let mut half_p = Matrix::zeros(nf, nf);
    let mut half_w = Matrix::zeros(nf, nf);
    for mu in 0..nf {
        for nu in 0..nf {
            half_p[(mu, nu)] = 0.5 * result.density[(mu, nu)];
            half_w[(mu, nu)] = 0.25 * w[(mu, nu)];
        }
    }
    let uhf =
        uhf_analytic_gradient(&basis, &molecule, &half_p, &half_p, &half_w, &half_w).expect("uhf");
    let mut worst = 0.0_f64;
    for atom in 0..closed.len() {
        for axis in 0..3 {
            worst = worst.max((closed[atom][axis] - uhf[atom][axis]).abs());
        }
    }
    println!("闭壳层极限：两条路径最大差 {worst:.3e} Ha/Bohr");
    assert!(
        worst < 1.0e-10,
        "UHF gradient must reduce to the closed-shell formula, worst {worst:.3e}"
    );
}
