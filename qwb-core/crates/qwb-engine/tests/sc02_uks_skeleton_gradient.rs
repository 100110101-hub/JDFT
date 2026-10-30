//! 临时诊断（SC-02 UKS 半）：把「骨架梯度」（固定密度、无 Pulay/CPHF 响应）单独对数值差分，
//! 判断那 4% 的差在**骨架**里还是在**响应**里。
//!
//! 骨架能量（固定 P、随几何变积分与 XC 网格项）：
//!
//! ```text
//! E_fixed(R) = Tr(P_tot H) + ½ Tr(P_tot J(P_tot)) − ½α Σ_σ Tr(P^σ K(P^σ)) + E_xc[P_α,P_β;R] + E_nuc
//! ```
//!
//! 它的导数**没有** Pulay 项（系数不响应），正好对上 `gradient_components` 的
//! `kinetic + nuclear_ao + nuclear_center + two_electron_* + nuclear_repulsion`
//! 再加上 XC 的**几何**导数。
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::one_electron::one_electron;
use qwb_integrals::two_electron::two_electron;
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::analytic_gradient::gradient_components;
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

fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.nrows();
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total += a.get(i, j) * b.get(i, j);
        }
    }
    total
}

/// 固定两条自旋密度、给定几何下的骨架能量。
fn fixed_energy(
    registry: &Registry,
    molecule: &Molecule,
    p_alpha: &Matrix,
    p_beta: &Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> f64 {
    let basis = InternalBasis::build(registry.get_basis("sto-3g").unwrap(), molecule).unwrap();
    let one = one_electron(&basis, molecule).unwrap();
    let tensor = two_electron(&basis, molecule).unwrap();
    let n = basis.n_functions();
    let mut p_total = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p_total[(i, j)] = p_alpha[(i, j)] + p_beta[(i, j)];
        }
    }
    let fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &p_total).unwrap();
    let coulomb = trace_product(&p_total, &fock.coulomb);
    let kinetic_nuclear = trace_product(&p_total, &one.core_hamiltonian);
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), molecule).unwrap();
    let cache = GridAoCached::build(&basis, &grid, false, false, false).unwrap();
    let xc = evaluate_xc_spin_cached(&basis, &grid, &cache, p_alpha, p_beta, terms).unwrap();
    let mut e_nuc = 0.0;
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let z = f64::from(qwb_registry::elements::atomic_number(&atom.symbol).unwrap_or(0));
        for other in molecule.atoms.iter().skip(index + 1) {
            let z_other =
                f64::from(qwb_registry::elements::atomic_number(&other.symbol).unwrap_or(0));
            let dx = atom.position_bohr()[0] - other.position_bohr()[0];
            let dy = atom.position_bohr()[1] - other.position_bohr()[1];
            let dz = atom.position_bohr()[2] - other.position_bohr()[2];
            e_nuc += z * z_other / (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    kinetic_nuclear + 0.5 * coulomb + xc.energy_hartree + e_nuc
}

#[test]
fn bisect_uks_skeleton() {
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
    let tensor = two_electron(&basis, &molecule).unwrap();
    let grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-coarse").unwrap(), &molecule).unwrap();
    let (terms, hf) = qwb_engine::terms_and_hf_exchange(&registry, "svwn5").unwrap();
    let mut options = UksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-9;
    options.hf_exchange = hf;
    let result = uks_scf(&basis, &molecule, &tensor, &grid, &terms, options).unwrap();
    assert!(result.scf_converged);

    // 解析骨架（固定密度）：gradient_components 的 non-Pulay 项 + XC 几何导数
    let one = one_electron(&basis, &molecule).unwrap();
    let n = basis.n_functions();
    let mut p_total = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p_total[(i, j)] = result.density_alpha[(i, j)] + result.density_beta[(i, j)];
        }
    }
    let fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &p_total).unwrap();
    let mut ks_fock = Matrix::zeros(n, n);
    let cache = GridAoCached::build(&basis, &grid, false, false, false).unwrap();
    let xc_at = evaluate_xc_spin_cached(
        &basis,
        &grid,
        &cache,
        &result.density_alpha,
        &result.density_beta,
        &terms,
    )
    .unwrap();
    // 骨架只用到总密度的 Fock（XC 的势在 Pulay 项里，骨架不含它）
    for i in 0..n {
        for j in 0..n {
            ks_fock[(i, j)] = one.core_hamiltonian[(i, j)] + fock.coulomb[(i, j)];
        }
    }
    let components = gradient_components(&basis, &molecule, &p_total, &ks_fock).unwrap();
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
    let natoms = molecule.atoms.len();
    let mut worst = 0.0_f64;
    for atom in 0..natoms {
        for axis in 0..3 {
            let analytic = components.kinetic[atom][axis]
                + components.nuclear_ao[atom][axis]
                + components.nuclear_center[atom][axis]
                + components.two_electron_coulomb[atom][axis]
                + components.two_electron_exchange[atom][axis] * hf
                + components.nuclear_repulsion[atom][axis]
                + xc_geometric[atom][axis];
            let step = 1.0e-3;
            let plus = fixed_energy(
                &registry,
                &displaced(&molecule, atom, axis, step),
                &result.density_alpha,
                &result.density_beta,
                &terms,
            );
            let minus = fixed_energy(
                &registry,
                &displaced(&molecule, atom, axis, -step),
                &result.density_alpha,
                &result.density_beta,
                &terms,
            );
            let numeric = (plus - minus) / (2.0 * step);
            let difference = (analytic - numeric).abs();
            worst = worst.max(difference);
            println!(
                "  atom {atom} axis {axis}: 解析骨架 {analytic:>13.9}  数值 {numeric:>13.9}  差 {difference:.2e}"
            );
        }
    }
    let _ = xc_at;
    println!("骨架最大差 {worst:.3e} Ha/Bohr");
    // 判据：骨架梯度（固定密度、无 Pulay）必须与固定密度能量差分的差 < 1e-6。
    // 这条钉住 kinetic / nuclear / coulomb / exchange / XC 几何 / E_nuc 这一半；
    // 它一旦绿，**全**梯度剩下的差异就只可能在响应（Pulay / W）那一半。
    assert!(
        worst < 1.0e-6,
        "UKS skeleton gradient differs from the fixed-density FD by {worst:.3e}"
    );
}
