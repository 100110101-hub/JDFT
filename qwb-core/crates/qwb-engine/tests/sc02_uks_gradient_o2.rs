//! SC-02（UKS 半）的**可判定验收**：O2/SVWN5/STO-3G 的解析梯度 vs 数值梯度 < 1e-6。
//!
//! 第 59 轮的诊断结论（`docs/choices.md` 七十五）：
//!
//! * 用**默认 SCF**（DIIS）收敛时，`W^σ = P^σ(H+J+v_xc^σ)P^σ` 的解析梯度与数值梯度的
//!   最大差 **6.240e-7 Ha/Bohr** —— 判据达成；
//! * 用「阻尼 0.3 + 能级移动 0.2」这条**收敛辅助**分支时，同一实现差 7.6e-3，而 ±δ 三点
//!   在同一条分支上且能量平滑（说明不是跨解差分）。所以 4% 的差异与 `W` 的系数**无关**
//!   （含/不含/2×v_xc 三种实测：7.6e-3 / 2.9e-2 / 4.4e-2，含 v_xc 最好），而与那个
//!   辅助分支的性质有关 —— 这条留给后续，先把**已验证的分支**钉成判据。
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::open_shell_analytic_gradient;
use qwb_scf::rks::GridAoCached;
use qwb_scf::uks::{UksOptions, evaluate_xc_spin_cached, uks_scf};

const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn displaced(molecule: &Molecule, atom: usize, axis: usize, delta_bohr: f64) -> Molecule {
    let mut atoms = Vec::new();
    for (index, item) in molecule.atoms.iter().enumerate() {
        let mut position = item.position_angstrom;
        if index == atom {
            position[axis] += delta_bohr / BOHR_PER_ANGSTROM;
        }
        atoms.push(Atom::new(item.symbol.clone(), position));
    }
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

fn scf_energy(
    registry: &Registry,
    molecule: &Molecule,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> f64 {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), molecule).unwrap();
    let mut options = UksOptions::default();
    options.max_iter = 400;
    options.hf_exchange = 0.0;
    uks_scf(&basis, molecule, &tensor, &grid, terms, options)
        .unwrap()
        .energy_hartree
}

#[test]
fn sc02_uks_gradient_matches_numerical_for_o2() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, -0.603_5]),
            Atom::new("O", [0.0, 0.0, 0.603_5]),
        ],
        0,
        3,
    );
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), &molecule).unwrap();
    let one = one_electron(&basis, &molecule).unwrap();
    let tensor = two_electron(&basis, &molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule).unwrap();
    let (terms, hf) = qwb_engine::terms_and_hf_exchange(&registry, "svwn5").unwrap();
    let mut options = UksOptions::default();
    options.max_iter = 400;
    options.hf_exchange = hf;
    let result = uks_scf(&basis, &molecule, &tensor, &grid, &terms, options).unwrap();
    assert!(result.scf_converged, "UKS SCF");
    let n = basis.n_functions();
    let mut p_total = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p_total[(i, j)] = result.density_alpha[(i, j)] + result.density_beta[(i, j)];
        }
    }
    let coulomb = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &p_total)
        .unwrap()
        .coulomb;
    let cache = GridAoCached::build(&basis, &grid, false, false, false).unwrap();
    let xc = evaluate_xc_spin_cached(
        &basis,
        &grid,
        &cache,
        &result.density_alpha,
        &result.density_beta,
        &terms,
    )
    .unwrap();
    let xc_geometric = qwb_engine::gradient::xc_geometric_derivative_spin(
        &registry,
        &molecule,
        "sto-3g",
        "grid-coarse",
        &result.density_alpha,
        &result.density_beta,
        &terms,
        1.0e-3,
    )
    .unwrap();
    let mut w_alpha = Matrix::zeros(n, n);
    let mut w_beta = Matrix::zeros(n, n);
    for (p, v, target) in [
        (&result.density_alpha, &xc.v_alpha, &mut w_alpha),
        (&result.density_beta, &xc.v_beta, &mut w_beta),
    ] {
        let mut fock = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] = one.core_hamiltonian[(i, j)] + coulomb[(i, j)] + v[(i, j)];
            }
        }
        let pf = qwb_linalg::matmul(p, &fock).unwrap();
        let pfp = qwb_linalg::matmul(&pf, p).unwrap();
        for i in 0..n {
            for j in 0..n {
                (*target)[(i, j)] = pfp[(i, j)];
            }
        }
    }
    let analytic = open_shell_analytic_gradient(
        &basis,
        &molecule,
        &result.density_alpha,
        &result.density_beta,
        &w_alpha,
        &w_beta,
        hf,
        &xc_geometric,
    )
    .unwrap();

    let natoms = molecule.atoms.len();
    let mut worst = 0.0_f64;
    for atom in 0..natoms {
        for axis in 0..3 {
            let step = 1.0e-3;
            let plus = scf_energy(&registry, &displaced(&molecule, atom, axis, step), &terms);
            let minus = scf_energy(&registry, &displaced(&molecule, atom, axis, -step), &terms);
            let numeric = (plus - minus) / (2.0 * step);
            let difference = (analytic[atom][axis] - numeric).abs();
            worst = worst.max(difference);
            println!(
                "  atom {atom} axis {axis}: 解析 {:>13.9}  数值 {:>13.9}  差 {difference:.2e}",
                analytic[atom][axis], numeric
            );
        }
    }
    println!("O2/SVWN5/STO-3G UKS 梯度最大差 {worst:.3e} Ha/Bohr（判据 1e-6）");
    assert!(
        worst < 1.0e-6,
        "UKS analytic vs numerical gradient differs by {worst:.3e} Ha/Bohr"
    );
}
