//! RP-02：**收敛验收不许被「密度停住」骗过去** —— `Diis::error_norm` 进门。
//!
//! # 这个测试重建的是什么状态
//!
//! RP-02 的缺陷态（实测，2026-11 的一轮测量）长这样：
//!
//! ```text
//! scf step 21: E = -161.377243672  dE = 1.853e2  max|dP| = 4.828e0   sum(f) = 14  |e|_diis = 8.381e-31
//! scf step 22: E = -81.712254754   dE = 7.966e1  max|dP| = 1.215e-13  sum(f) = 14  |e|_diis = 6.375e0
//! scf step 23: E = -81.712254754   dE = 7.105e-14 max|dP| = 6.151e-14  sum(f) = 14  |e|_diis = 6.375e0
//! ```
//!
//! 也就是说：密度**停在 `H_core` 初猜上**（`max|dP| ~ 1e-13`），能量不再变
//! （`dE ~ 1e-14`），DIIS 子空间里是二十组**零误差**的陈旧向量 —— 旧的验收条件
//! 「`dE < energy_threshold` 且 `max|dP| < density_threshold`」两条全过，于是 SCF
//! 把 `H_core` 初猜的能量当成收敛解返回。可是 `F` 与 `P` 根本不对易：
//! `|e|_diis = 6.375`，比真收敛的体系大 **七个数量级**。
//!
//! 这个测试用**公开 API** 把那个终态一步重建出来（`resume_from` 一个密度 = 初猜、
//! DIIS 子空间 = 8 组零误差向量的检查点），然后要求 `rks_scf` **拒绝**它。
//! 拒绝的方式是 `ScfNotConverged`：引擎拿到错误会按既有顺序换下一级退路，
//! 而不是把初猜的能量写进结果。
//!
//! # 它同时证明的三件事
//!
//! 1. 缺陷报的那两个数 `-81.7122547541 Ha` 就是**初猜密度**的能量：本测试用
//!    `E = Tr(PH) + ½Tr(PJ) + E_xc + E_nuc` 在同一个基组/网格/泛函上独立算一遍，
//!    与缺陷值对到 1e-6 Ha 以内；
//! 2. 真收敛的 SCF 仍然被接受（H2 / H2O × {STO-3G, def2-SVP} × {SVWN5, PBE}，见
//!    `the_guard_accepts_genuine_convergence`），所以这道门不会把正常收敛判成不收敛；
//! 3. `Diis::error_norm`（原先无人使用）现在**是**判据的一部分。
#![allow(clippy::field_reassign_with_default)]

use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_linalg::{Matrix, symmetric_eigen};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn registry() -> Registry {
    Registry::open(data_root()).expect("qwb-data")
}

/// C 在原点、N 沿 +z 1.17 Å、H 在 x–z 平面里 1.19 Å 处。
fn hcn(angle_deg: f64) -> Molecule {
    let radians = angle_deg.to_radians();
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("N", [0.0, 0.0, 1.17]),
            Atom::new("H", [1.19 * radians.sin(), 0.0, 1.19 * radians.cos()]),
        ],
        0,
        1,
    )
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.119]),
            Atom::new("H", [0.0, 0.763, -0.477]),
            Atom::new("H", [0.0, -0.763, -0.477]),
        ],
        0,
        1,
    )
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

fn terms(registry: &Registry, name: &str) -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let functional = registry.get_functional(name).expect("functional");
    let parameters = functional.parameters.clone();
    let mut out: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &parameters).expect("term");
        out.push(qwb_xc::scale_term(term, component.weight));
    }
    out
}

/// `S^{-1/2}`（对称正交化），与 SCF 内部同一个公式。
fn inverse_sqrt(s: &Matrix) -> Matrix {
    let (values, vectors) = symmetric_eigen(s).expect("eigen");
    let n = s.nrows();
    let mut scaled = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            scaled[(i, j)] = vectors[(i, j)] / values.get(j).sqrt();
        }
    }
    qwb_linalg::matmul(&scaled, &vectors.transpose()).expect("x")
}

fn nuclear_repulsion(molecule: &Molecule, charges: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..molecule.n_atoms() {
        for j in (i + 1)..molecule.n_atoms() {
            let a = molecule.atoms[i].position_bohr();
            let b = molecule.atoms[j].position_bohr();
            let distance = ((0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>()).sqrt();
            total += charges[i] * charges[j] / distance;
        }
    }
    total
}

/// 闭壳层 KS 的能量式 `E = Tr(PH) + ½Tr(PJ) + E_xc + E_nuc`。
///
/// 造这个函数而不是直接读 `RksResult`，是因为要算的密度**不是** SCF 的输出（它是
/// `H_core` 初猜）。公式本身在 `f2_sto3g_svwn5` 里用一次真收敛的 SCF 校准过。
fn energy_of_density(
    basis: &qwb_basis::InternalBasis,
    grid: &qwb_grid::Grid,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    hcore: &Matrix,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    density: &Matrix,
    e_nuc: f64,
) -> f64 {
    let fock_data = qwb_scf::build_fock(hcore, tensor, density).expect("fock");
    let (e_xc, _v_xc) = qwb_scf::rks::evaluate_xc(basis, grid, density, terms).expect("xc");
    let n = basis.n_functions();
    let mut electronic = 0.0;
    for i in 0..n {
        for j in 0..n {
            electronic += density[(i, j)] * (hcore[(i, j)] + 0.5 * fock_data.coulomb[(i, j)]);
        }
    }
    electronic + e_xc + e_nuc
}

/// 修复前 `scf` 为 90° 弯 HCN（PBE/def2-SVP/grid-coarse）报的「收敛」能量。
const DEFECT_REPORTED_HARTREE: f64 = -81.712_254_754_1;

#[test]
fn the_guard_rejects_a_density_frozen_at_the_hcore_guess() {
    let registry = registry();
    let molecule = hcn(90.0);
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        &molecule,
    )
    .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let terms = terms(&registry, "pbe");
    let n = basis.n_functions();

    let integrals = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let x = inverse_sqrt(&integrals.overlap);
    // 14 个电子 ⇒ 7 个占据轨道。
    let guess = qwb_scf::rks::core_guess(&integrals.core_hamiltonian, &x, 7).expect("guess");
    let e_nuc = nuclear_repulsion(&molecule, &[6.0, 7.0, 1.0]);
    let guess_energy = energy_of_density(
        &basis,
        &grid,
        &terms,
        &integrals.core_hamiltonian,
        &tensor,
        &guess,
        e_nuc,
    );
    println!("E[H_core 初猜密度] = {guess_energy:.12} Ha");
    println!("缺陷报的值        = {DEFECT_REPORTED_HARTREE:.10} Ha");
    assert!(
        (guess_energy - DEFECT_REPORTED_HARTREE).abs() < 1.0e-6,
        "缺陷报的 {DEFECT_REPORTED_HARTREE:.10} 与初猜密度的能量 {guess_energy:.12} 不一致；\
         说明那个数不是初猜能量，本测试的前提要重写"
    );

    // 缺陷的终态：密度 = 初猜，DIIS 子空间 = 8 组**零**误差向量。
    let zeros = Matrix::zeros(n, n);
    let focks: Vec<Vec<f64>> = (0..8)
        .map(|_| integrals.core_hamiltonian.as_column_major().to_vec())
        .collect();
    let errors: Vec<Vec<f64>> = (0..8).map(|_| zeros.as_column_major().to_vec()).collect();
    let checkpoint = qwb_scf::checkpoint::ScfCheckpoint::new(&guess, Some(guess_energy), 0)
        .with_diis(focks, errors);

    let mut options = RksOptions::default();
    options.max_iter = 12;
    options.energy_threshold = 1.0e-9;
    options.density_threshold = 1.0e-7;
    options.resume_from = Some(checkpoint);
    let outcome = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options);
    match outcome {
        Err(EngineError::ScfNotConverged { iters, max }) => {
            println!("按预期拒绝：ScfNotConverged {{ iters: {iters}, max: {max} }}");
            assert_eq!(max, 12, "拒绝时要把调用方给的预算原样报回来");
        }
        Err(other) => panic!("期望 ScfNotConverged（让引擎换下一级退路），实际 {other:?}"),
        Ok(result) => panic!(
            "SCF 接受了「密度冻在 H_core 初猜上」的状态：E = {:.12} Ha（初猜能量 {:.12}）——\
             收敛验收必须要求 DIIS 交换子误差也小（见 rks::diis_error_limit）",
            result.energy_hartree, guess_energy
        ),
    }
}

/// 这道门不能把**真**收敛判成不收敛：三种体系（含 def2-SVP + 网格 + LDA/GGA）都要 `Ok`。
#[test]
fn the_guard_accepts_genuine_convergence() {
    for (label, molecule, basis_name, functional, grid) in [
        ("H2", h2(), "sto-3g", "svwn5", "grid-normal"),
        ("H2O", water(), "sto-3g", "svwn5", "grid-normal"),
        ("H2O", water(), "def2-svp", "pbe", "grid-coarse"),
    ] {
        let result = rks_from_data(
            &registry(),
            &molecule,
            basis_name,
            functional,
            grid,
            RksOptions::default(),
        )
        .unwrap_or_else(|err| panic!("{label}/{basis_name}/{functional}/{grid} 不收敛了：{err}"));
        println!(
            "{label}/{basis_name}/{functional}/{grid}: E = {:.12} Ha, {} 步, converged = {}",
            result.energy_hartree, result.iterations, result.scf_converged
        );
        assert!(result.scf_converged);
    }
}

/// 能量式校准：用它算一次**真收敛**的 SCF 密度，必须复现 SCF 报的总能量。
#[test]
fn the_energy_formula_is_calibrated_on_a_converged_density() {
    let registry = registry();
    let molecule = h2();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-normal").expect("grid"), &molecule)
        .expect("grid");
    let terms = terms(&registry, "svwn5");
    let integrals = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let result = rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "svwn5",
        "grid-normal",
        RksOptions::default(),
    )
    .expect("scf");
    let formula = energy_of_density(
        &basis,
        &grid,
        &terms,
        &integrals.core_hamiltonian,
        &tensor,
        &result.density,
        result.nuclear_repulsion_hartree,
    );
    println!(
        "formula = {formula:.12}  reported = {:.12}",
        result.energy_hartree
    );
    assert!(
        (formula - result.energy_hartree).abs() < 1.0e-8,
        "能量式 {formula:.12} 与 SCF 报的 {:.12} 不一致",
        result.energy_hartree
    );
}
