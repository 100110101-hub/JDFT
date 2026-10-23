//! 诊断：把 `gradient_components` 的每一项与「固定密度」的中心差分逐项对照。
//!
//! 这是 G-01 唯一可靠的定位手段：总分差一点点时，先确认是哪一项。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::gradient_components;
use qwb_scf::build_fock;
use qwb_scf::rks::{RksOptions, rks_scf};

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

fn displaced(molecule: &Molecule, atom: usize, axis: usize, delta_bohr: f64) -> Molecule {
    let mut out = molecule.clone();
    out.atoms[atom].position_angstrom[axis] += delta_bohr * ANGSTROM_PER_BOHR;
    out
}

fn build(registry: &Registry, molecule: &Molecule) -> InternalBasis {
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    InternalBasis::build(basis_k1, molecule).expect("build")
}

/// 固定密度的 `Tr(P T)` 与 `Tr(P V_nuc)`。
fn trace_split(registry: &Registry, molecule: &Molecule, density: &Matrix) -> (f64, f64) {
    let basis = build(registry, molecule);
    let one = one_electron(&basis, molecule).expect("1e");
    let n = density.nrows();
    let mut kinetic = 0.0;
    let mut nuclear = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            kinetic += density[(mu, nu)] * one.kinetic[(mu, nu)];
            nuclear += density[(mu, nu)] * one.nuclear[(mu, nu)];
        }
    }
    (kinetic, nuclear)
}

/// 固定密度的 2e 能量 `½ Σ P P [(μν|λσ) − ½(μλ|νσ)]`。
fn energy_2e(registry: &Registry, molecule: &Molecule, density: &Matrix) -> f64 {
    let basis = build(registry, molecule);
    let tensor = two_electron(&basis, molecule).expect("2e");
    let n = density.nrows();
    let mut value = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            for lambda in 0..n {
                for sigma in 0..n {
                    let coulomb = tensor.get(mu, nu, lambda, sigma);
                    let exchange = tensor.get(mu, lambda, nu, sigma);
                    value += 0.5
                        * density[(mu, nu)]
                        * density[(lambda, sigma)]
                        * (coulomb - 0.5 * exchange);
                }
            }
        }
    }
    value
}

#[test]
fn diag_gradient_components() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    for bond in [0.74_f64, 1.1] { let molecule = h2(bond); println!("=== R = {bond} ===");

    let basis = build(&registry, &molecule);
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let one = one_electron(&basis, &molecule).expect("1e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
    let fock = build_fock(&one.core_hamiltonian, &tensor, &result.density)
        .expect("fock")
        .fock;

    let components = gradient_components(&basis, &molecule, &result.density, &fock).expect("comp");
    let mut worst_1e = 0.0_f64;
    let mut worst_2e = 0.0_f64;

    let delta = 1.0e-4_f64;
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let plus = displaced(&molecule, atom, axis, delta);
            let minus = displaced(&molecule, atom, axis, -delta);
            let (kin_plus, nuc_plus) = trace_split(&registry, &plus, &result.density);
            let (kin_minus, nuc_minus) = trace_split(&registry, &minus, &result.density);
            let fd_kin = (kin_plus - kin_minus) / (2.0 * delta);
            let fd_nuc = (nuc_plus - nuc_minus) / (2.0 * delta);
            let fd_2e = (energy_2e(&registry, &plus, &result.density)
                - energy_2e(&registry, &minus, &result.density))
                / (2.0 * delta);

            worst_1e = worst_1e
                .max((components.kinetic[atom][axis] - fd_kin).abs())
                .max((components.nuclear_ao[atom][axis] + components.nuclear_center[atom][axis] - fd_nuc).abs());
            worst_2e = worst_2e.max((components.two_electron[atom][axis] - fd_2e).abs());
            let a_kin = components.kinetic[atom][axis];
            let a_vao = components.nuclear_ao[atom][axis];
            let a_vc = components.nuclear_center[atom][axis];
            println!("atom {atom} axis {axis}");
            println!("  动能:   解析 {a_kin:>14.9}  差分 {fd_kin:>14.9}  差 {:>10.3e}", (a_kin - fd_kin).abs());
            println!(
                "  核吸引: 解析 {a_vao:>14.9}（AO 移动）/ {a_vc:>14.9}（核移动）  差分 {fd_nuc:>14.9}  差 {:>10.3e}",
                (a_vao + a_vc - fd_nuc).abs()
            );
            println!(
                "  2e:     解析 {:>14.9}  差分 {fd_2e:>14.9}  差 {:>10.3e}",
                components.two_electron[atom][axis],
                (components.two_electron[atom][axis] - fd_2e).abs()
            );
            println!(
                "  Pulay {:>14.9}  核排斥 {:>14.9}",
                components.pulay[atom][axis], components.nuclear_repulsion[atom][axis]
            );
        }
    }
    println!("逐项最大偏差：1e {:>9.2e}，2e {:>9.2e}", worst_1e, worst_2e);
    assert!(worst_1e < 1e-6, "1e 分项与差分不符：{worst_1e:.2e}");
    assert!(worst_2e < 1e-6, "2e 分项与差分不符：{worst_2e:.2e}");
    }
}
