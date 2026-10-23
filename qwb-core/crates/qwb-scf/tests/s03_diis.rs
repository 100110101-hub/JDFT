//! S-03：DIIS 加速 + 收敛判据。
//!
//! `project.md` §6.2 的验收是「H2O/PBE 收敛到 1e-8；不收敛时返回
//! `ScfNotConverged{iters,max}`」。H2O 需要 p 函数（I-03 未完成）、PBE 需要
//! GGA（F 轨未开工），所以这里用当前能算的体系把**判据本身**测死：
//!
//! * DIIS 必须真的起作用（不是安静地什么都不做——第一版就踩过这个坑：
//!   加边系统不定，Cholesky 求解失败后被 `Ok(None)` 吞掉，步数与不用 DIIS
//!   一模一样）；
//! * 不收敛必须返回契约错误，且 `exit_code()` 是 K5 冻结的 `2`。
//!
//! 用低对称的 H4 丛做测试体系：全 s 函数的对称体系（H2、正方形 H4）初猜即近
//! 不动点，两步入收敛，根本测不出加速器好坏。

use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_registry::Registry;
use qwb_scf::diis::Diis;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn cluster(points: &[[f64; 3]]) -> Molecule {
    Molecule::new(points.iter().map(|p| Atom::new("H", *p)).collect(), 0, 1)
}

/// 正四面体形状的 H4（低对称，普通迭代要 31 步）。
fn tetrahedron() -> Molecule {
    cluster(&[
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.45, 0.85, 0.0],
        [0.45, 0.28, 0.8],
    ])
}

/// 松散的 H4 丛：普通迭代在 400 步内**根本不收敛**，DIIS 6 步收敛。
fn loose_cluster() -> Molecule {
    cluster(&[
        [0.0, 0.0, 0.0],
        [2.2, 0.0, 0.0],
        [1.1, 2.0, 0.1],
        [1.1, 0.7, 1.9],
    ])
}

#[derive(Debug)]
struct Run {
    energy: f64,
    iterations: u32,
}

fn run_scf(molecule: &Molecule, use_diis: bool, max_iter: u32) -> Result<Run, EngineError> {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = qwb_basis::InternalBasis::build(basis_k1, molecule).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, molecule).expect("grid");

    let mut options = RksOptions::default();
    options.max_iter = max_iter;
    options.energy_threshold = 1.0e-8;
    options.use_diis = use_diis;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options)?;
    Ok(Run {
        energy: result.energy_hartree,
        iterations: result.iterations,
    })
}

#[test]
fn diis_cuts_the_iteration_count_on_a_stiff_case() {
    let molecule = tetrahedron();
    let plain = run_scf(&molecule, false, 200).expect("普通迭代应能收敛");
    let accelerated = run_scf(&molecule, true, 200).expect("DIIS 应能收敛");

    println!(
        "H4/tetra: plain {} iters (E = {:.10}), DIIS {} iters (E = {:.10})",
        plain.iterations, plain.energy, accelerated.iterations, accelerated.energy
    );

    assert!(
        accelerated.iterations * 3 < plain.iterations,
        "DIIS 没起作用：plain {} 步 vs DIIS {} 步（应至少快 3 倍）",
        plain.iterations,
        accelerated.iterations
    );
    assert!(
        (plain.energy - accelerated.energy).abs() < 1.0e-7,
        "两条路径收敛到不同的能量：{} vs {}",
        plain.energy,
        accelerated.energy
    );
}

#[test]
fn diis_converges_a_case_where_plain_iteration_fails() {
    let molecule = loose_cluster();
    let plain = run_scf(&molecule, false, 60);
    match plain {
        Err(EngineError::ScfNotConverged { iters, max }) => {
            assert_eq!((iters, max), (60, 60), "步数上报不对");
        }
        Ok(run) => panic!(
            "这个体系本应是硬骨头，普通迭代却在 {} 步内收敛了——测试体系要换",
            run.iterations
        ),
        Err(other) => panic!("预期 ScfNotConverged，实得 {other}"),
    }

    let accelerated = run_scf(&molecule, true, 60).expect("DIIS 必须能收敛这个体系");
    println!(
        "H4/loose: DIIS {} iters (E = {:.10})；普通迭代 60 步未收敛",
        accelerated.iterations, accelerated.energy
    );
    assert!(
        accelerated.iterations <= 20,
        "DIIS 收敛太慢：{} 步",
        accelerated.iterations
    );
}

#[test]
fn non_convergence_returns_the_frozen_contract_error() {
    let molecule = tetrahedron();
    let error = run_scf(&molecule, true, 1).expect_err("1 步不可能收敛");
    match &error {
        EngineError::ScfNotConverged { iters, max } => {
            assert_eq!(*iters, 1, "实际步数应为 1");
            assert_eq!(*max, 1, "允许上限应为 1");
        }
        other => panic!("预期 ScfNotConverged，实得 {other}"),
    }
    // K5 冻结：ScfNotConverged → 退出码 2。
    assert_eq!(error.exit_code(), 2);
}

#[test]
fn diis_commutator_vanishes_for_commuting_matrices() {
    // F = I、P = I、S = I ⇒ F P S − S P F = 0，这是收敛点的判据。
    let identity = qwb_linalg::Matrix::identity(3);
    let error = Diis::commutator(&identity, &identity, &identity).expect("commutator");
    for value in error.as_column_major() {
        assert!(value.abs() < 1.0e-15, "交换子应归零，实得 {value}");
    }
}

#[test]
fn diis_extrapolation_preserves_an_unchanged_fock() {
    // 子空间里所有 Fock 都相同 ⇒ 权重和为 1 ⇒ 外推结果必须还是它自己。
    let mut diis = Diis::new(4);
    let mut fock = qwb_linalg::Matrix::zeros(2, 2);
    fock.set(0, 0, -1.5);
    fock.set(0, 1, 0.25);
    fock.set(1, 0, 0.25);
    fock.set(1, 1, -0.75);

    let error =
        Diis::commutator(&fock, &fock, &qwb_linalg::Matrix::identity(2)).expect("commutator");
    diis.push(fock.clone(), error.clone());
    diis.push(fock.clone(), error);
    assert_eq!(diis.len(), 2);

    let extrapolated = diis
        .extrapolate()
        .expect("solve")
        .expect("两组向量应能外推");
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (extrapolated.get(i, j) - fock.get(i, j)).abs() < 1.0e-10,
                "外推改变了 Fock({i},{j})：{} vs {}",
                extrapolated.get(i, j),
                fock.get(i, j)
            );
        }
    }
}

#[test]
fn diis_needs_two_vectors_before_extrapolating() {
    let mut diis = Diis::new(4);
    assert!(
        diis.extrapolate().expect("solve").is_none(),
        "空子空间不该外推"
    );
    let fock = qwb_linalg::Matrix::identity(2);
    diis.push(fock.clone(), fock.clone());
    assert!(
        diis.extrapolate().expect("solve").is_none(),
        "一组向量时不该外推"
    );
    diis.push(fock.clone(), fock);
    assert!(diis.extrapolate().expect("solve").is_some(), "两组时应外推");
}
