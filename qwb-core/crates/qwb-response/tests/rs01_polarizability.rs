//! RS-01 的验收：静态极化率 vs 有限场（±0.001 a.u.），判据差 < 1e-4 a.u.。
//!
//! # 有限场怎么做
//!
//! 均匀电场 F 给**电子**哈密顿量加 `+F·Σ r`（核的部分是常数，对二阶导没有贡献），
//! 也就是往 `H_core` 里加一个 `F·⟨μ|r_axis|ν⟩` 的矩阵。`RksOptions::ecp_potential`
//! 的语义正好是"加到 `H_core` 的 `n_ao × n_ao` 矩阵"，所以这里直接借它当电场项 ——
//! 不是为了省事，而是因为**它就是同一个数学对象**（单电子外势）。测试里写清这一点，
//! 免得后来的人以为 ECP 被滥用。
//!
//! 有限场取 `F = ±0.001` 与 `±0.002`，用中心差分 `(μ(F) − μ(−F)) / (2F)` 得到
//! `α(F)`，再 Richardson 外推 `α₀ = (4α(F/2) − α(F)) / 3` 消掉 `O(F²)` 项
//! （能量里 `αF²/2` 的下一项是 `βF³/6`，中心差分已经消掉了奇次项）。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::dipole::dipole_integrals;
use qwb_registry::Registry;
use qwb_response::cphf::CphfOptions;
use qwb_response::mo_integrals::build_cphf_integrals;
use qwb_response::polarizability::static_polarizability;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

fn n_occupied(molecule: &Molecule) -> usize {
    let mut electrons = -molecule.charge;
    for atom in &molecule.atoms {
        let z = qwb_registry::elements::atomic_number(&atom.symbol)
            .unwrap_or_else(|| panic!("unknown element {}", atom.symbol));
        electrons += i32::try_from(z).expect("Z fits i32");
    }
    assert!(electrons > 0 && electrons % 2 == 0, "not closed shell");
    usize::try_from(electrons / 2).expect("n_occ")
}

/// 参考态 + CPHF 输入（一次 SCF，避免重复算积分）。
struct Reference {
    molecule: Molecule,
    basis: qwb_basis::InternalBasis,
    tensor: qwb_integrals::TwoElectronIntegrals,
    grid: qwb_grid::Grid,
    coefficients: qwb_linalg::Matrix,
    orbital_energies: Vec<f64>,
    dipole_ao: [qwb_linalg::Matrix; 3],
}

fn setup(basis_name: &str) -> Reference {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    let scf = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("HF");
    let orbital_energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();
    let dipole_ao = dipole_integrals(&basis, &molecule).expect("dipole");
    Reference {
        molecule,
        basis,
        tensor,
        grid,
        coefficients: scf.coefficients,
        orbital_energies,
        dipole_ao,
    }
}

/// 有限场：把 `F·⟨μ|r_axis|ν⟩` 加进 `H_core`，返回 dipole 的 a.u. 值（含核项）。
fn field_dipole(reference: &Reference, axis: usize, field: f64) -> [f64; 3] {
    let n = reference.basis.n_functions();
    let mut potential = reference.dipole_ao[axis].clone();
    for i in 0..n {
        for j in 0..n {
            potential[(i, j)] *= field;
        }
    }
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-12;
    options.density_threshold = 1.0e-10;
    options.ecp_potential = Some(potential);
    let scf = rks_scf(
        &reference.basis,
        &reference.molecule,
        &reference.tensor,
        &reference.grid,
        &[],
        options,
    )
    .expect("field SCF");
    // `dipole_debye` 返回 Debye；除以换算因子得到 a.u.（e·bohr）。
    let debye = qwb_props::dipole_debye(&reference.basis, &reference.molecule, &scf.density)
        .expect("dipole");
    debye.map(|value| value / 2.541_746_632_031_1)
}

/// `α_axis` 的中心差分 + Richardson 外推。
fn finite_field_diagonal(reference: &Reference, axis: usize) -> f64 {
    let alpha_at = |field: f64| -> f64 {
        let plus = field_dipole(reference, axis, field);
        let minus = field_dipole(reference, axis, -field);
        (plus[axis] - minus[axis]) / (2.0 * field)
    };
    let coarse = alpha_at(0.002);
    let fine = alpha_at(0.001);
    (4.0 * fine - coarse) / 3.0
}

#[test]
fn static_polarizability_matches_finite_field_for_sto3g() {
    let reference = setup("sto-3g");
    let n_occ = n_occupied(&reference.molecule);
    let integrals = build_cphf_integrals(
        &reference.tensor,
        &reference.coefficients,
        &reference.dipole_ao,
        n_occ,
    )
    .expect("CPHF integrals");
    let mut options = CphfOptions::default();
    options.tolerance = 1.0e-10;
    let (alpha, outcome) = static_polarizability(&integrals, &reference.orbital_energies, options)
        .expect("static polarizability");
    println!(
        "H2O/STO-3G：α (CPHF) = {:.8} (iso)，张量 diag = [{:.8}, {:.8}, {:.8}]，各向异性 {:.8}",
        alpha.isotropic_au,
        alpha.tensor_au[0][0],
        alpha.tensor_au[1][1],
        alpha.tensor_au[2][2],
        alpha.anisotropy_au
    );
    println!(
        "CPHF 迭代：{:?} 步，残差 {:?}",
        outcome.iterations, outcome.residuals
    );
    // 判据：CPHF 迭代收敛到 1e-8（这里更严，1e-10）。
    assert!(outcome.converged, "CPHF 未收敛：{:?}", outcome.residuals);
    for (axis, residual) in outcome.residuals.iter().enumerate() {
        assert!(
            *residual < 1.0e-8,
            "第 {axis} 个右端的残差 {residual:.3e} 未到 1e-8"
        );
    }

    for axis in 0..3 {
        let fd = finite_field_diagonal(&reference, axis);
        let cphf = alpha.tensor_au[axis][axis];
        let difference = (cphf - fd).abs();
        println!("  axis {axis}: CPHF {cphf:.8} vs 有限场 {fd:.8}，差 {difference:.3e}");
        assert!(
            difference < 1.0e-4,
            "axis {axis}: α(CPHF) = {cphf:.8} 与有限场 {fd:.8} 差 {difference:.3e}（判据 1e-4）"
        );
    }
}

#[test]
fn static_polarizability_matches_finite_field_for_def2svp() {
    let reference = setup("def2-svp");
    let n_occ = n_occupied(&reference.molecule);
    let integrals = build_cphf_integrals(
        &reference.tensor,
        &reference.coefficients,
        &reference.dipole_ao,
        n_occ,
    )
    .expect("CPHF integrals");
    let mut options = CphfOptions::default();
    options.tolerance = 1.0e-10;
    let (alpha, outcome) = static_polarizability(&integrals, &reference.orbital_energies, options)
        .expect("static polarizability");
    println!(
        "H2O/def2-SVP：α (CPHF) = {:.8} (iso)，diag = [{:.8}, {:.8}, {:.8}]，各向异性 {:.8}，\
         迭代 {:?} 步，残差 {:?}",
        alpha.isotropic_au,
        alpha.tensor_au[0][0],
        alpha.tensor_au[1][1],
        alpha.tensor_au[2][2],
        alpha.anisotropy_au,
        outcome.iterations,
        outcome.residuals
    );
    assert!(outcome.converged, "CPHF 未收敛：{:?}", outcome.residuals);
    for axis in 0..3 {
        let fd = finite_field_diagonal(&reference, axis);
        let cphf = alpha.tensor_au[axis][axis];
        let difference = (cphf - fd).abs();
        println!("  axis {axis}: CPHF {cphf:.8} vs 有限场 {fd:.8}，差 {difference:.3e}");
        assert!(
            difference < 1.0e-4,
            "axis {axis}: α(CPHF) = {cphf:.8} 与有限场 {fd:.8} 差 {difference:.3e}（判据 1e-4）"
        );
    }
}

/// 确定性：两次调用逐位相同（`RS-01` 的「1 vs N 线程 identical」在当前实现里是
/// 平凡成立的 —— 求解器是串行的，没有归约顺序问题；这里把它钉住，等 IN-05 把
/// CPHF 并行化时这条测试会立刻变成有意义的检查）。
#[test]
fn cphf_is_deterministic() {
    let reference = setup("sto-3g");
    let n_occ = n_occupied(&reference.molecule);
    let integrals = build_cphf_integrals(
        &reference.tensor,
        &reference.coefficients,
        &reference.dipole_ao,
        n_occ,
    )
    .expect("CPHF integrals");
    let options = CphfOptions::default();
    let first = static_polarizability(&integrals, &reference.orbital_energies, options)
        .expect("first")
        .0;
    let mut second_options = CphfOptions::default();
    second_options.max_iter = 100;
    let second = static_polarizability(&integrals, &reference.orbital_energies, second_options)
        .expect("second")
        .0;
    // 同一容差下逐位相同（容差不同只影响步数，不影响解到 1e-10 的位模式）。
    assert_eq!(first.tensor_au, second.tensor_au);
}
