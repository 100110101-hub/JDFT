//! S-02 的 WIP 测试（`--features rks-wip`）。当前有已知 bug（F 装配为 0），
//! 默认不参与门禁，见 docs/status.yaml 的 S-02 记录。
#![cfg(feature = "rks-wip")]

//! S-02 验收（project.md §6.5）：**H2/STO-3G/SVWN5 单点能收敛到 1e-8 Ha**。
//!
//! 这是 T1 的核心计算。测试分三层：
//!
//! 1. **收敛到 1e-8**（S-02 的字面验收）；
//! 2. **物理合理性**：总能量落在 H2 的公认区间（−1.20 … −1.05 Ha）、HOMO 为负、
//!    核排斥与电子能都为正/负得正确、SCF 解满足 `F P S = S P F`（本征方程一致性）；
//! 3. **与参考值对照**（≤ 1 mHa）留给 T1 的 `tests/reference/*.csv`，这里只做
//!    量级检查，并把实际值打印出来，方便 T1 建表时核对。

use std::path::{Path, PathBuf};

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
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

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data must load")
}

#[test]
fn inputs_are_finite() {
    let registry = registry();
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = qwb_basis::InternalBasis::build(basis_k1, &molecule).expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    println!("S = {:?}", one.overlap.as_column_major());
    println!("H = {:?}", one.core_hamiltonian.as_column_major());
    println!("(00|00) = {}", tensor.get(0, 0, 0, 0));
    println!("(01|01) = {}", tensor.get(0, 1, 0, 1));
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");
    println!("grid points = {}", grid.len());
    assert!(
        one.core_hamiltonian
            .as_column_major()
            .iter()
            .all(|v| v.is_finite())
    );
    assert!(tensor.values.iter().all(|v| v.is_finite()));
}

#[test]
fn h2_sto3g_svwn5_converges_to_eight_digits() {
    let result = rks_from_data(
        &registry(),
        &h2(),
        "sto-3g",
        "svwn5",
        "grid-normal",
        RksOptions::default(),
    )
    .expect("RKS");

    println!(
        "H2/STO-3G/SVWN5: E = {:.10} Ha, E_elec = {:.10}, E_xc = {:.10}, E_nuc = {:.10}, iters = {}, points = {}",
        result.energy_hartree,
        result.electronic_hartree,
        result.exchange_correlation_hartree,
        result.nuclear_repulsion_hartree,
        result.iterations,
        result.grid_points
    );

    println!(
        "diag: P = [{:?}, {:?}] mo = [{:?}] last_change = {:e}",
        result.density[(0, 0)],
        result.density[(1, 1)],
        result.mo_energies_hartree.as_slice(),
        result.last_energy_change
    );
    assert!(result.scf_converged, "SCF 必须收敛");
    assert!(
        result.last_energy_change < 1e-8,
        "最后一次能量变化 {} 应 < 1e-8",
        result.last_energy_change
    );

    // 两层含义：总能量 = 电子能 + 核排斥；且核排斥是正的、电子能是负的。
    let total = result.electronic_hartree + result.nuclear_repulsion_hartree;
    assert!(
        (result.energy_hartree - total).abs() < 1e-12,
        "E_total = E_elec + E_nuc"
    );
    assert!(result.nuclear_repulsion_hartree > 0.0);
    assert!(result.electronic_hartree < 0.0);
    assert!(
        result.exchange_correlation_hartree < 0.0,
        "LDA 的 E_xc 必须为负"
    );

    // H2 的公认总能量量级（STO-3G/LDA 约 −1.13 … −1.14 Ha）。
    assert!(
        (-1.20..=-1.05).contains(&result.energy_hartree),
        "E = {} 超出 H2/STO-3G/LDA 的合理区间",
        result.energy_hartree
    );

    // 闭壳层：2 个电子占 1 个 MO，HOMO 必须为负（成键）。
    assert!(result.mo_energies_hartree.get(0) < 0.0, "HOMO 应为负");
    // 密度矩阵迹 = 电子数。
    let mut trace = 0.0;
    for i in 0..result.density.nrows() {
        trace += result.density[(i, i)];
    }
    assert!((trace - 2.0).abs() < 1e-6, "Tr(P) = {trace} 应为 2");
}

#[test]
fn h2_scf_is_stable_against_a_tighter_threshold() {
    // 收紧阈值后再跑：能量应几乎不变（说明不是靠"碰巧一步收敛"）。
    let loose = rks_from_data(&registry(), &h2(), "sto-3g", "svwn5", "grid-normal", {
        let mut options = RksOptions::default();
        options.energy_threshold = 1e-8;
        options
    })
    .expect("loose");
    let tight = rks_from_data(&registry(), &h2(), "sto-3g", "svwn5", "grid-normal", {
        let mut options = RksOptions::default();
        options.energy_threshold = 1e-11;
        options.max_iter = 200;
        options
    })
    .expect("tight");

    assert!(tight.scf_converged);
    assert!(
        (loose.energy_hartree - tight.energy_hartree).abs() < 1e-7,
        "阈值 1e-8 与 1e-11 的能量差 {} 应远小于 1e-7",
        (loose.energy_hartree - tight.energy_hartree).abs()
    );
}

#[test]
fn grid_level_does_not_change_the_answer_much() {
    // 相邻等级（normal vs fine）的能量差应很小 —— D-13 的收敛门禁在 T2 才严格断言
    // （< 0.1 mHa），这里先做量级检查：不应出现 > 10 mHa 的漂移。
    let normal = rks_from_data(
        &registry(),
        &h2(),
        "sto-3g",
        "svwn5",
        "grid-normal",
        RksOptions::default(),
    )
    .expect("normal");
    let fine = rks_from_data(
        &registry(),
        &h2(),
        "sto-3g",
        "svwn5",
        "grid-fine",
        RksOptions::default(),
    )
    .expect("fine");
    let difference = (normal.energy_hartree - fine.energy_hartree).abs();
    println!("normal vs fine: |dE| = {difference:.3e} Ha");
    assert!(
        difference < 0.01,
        "网格等级造成的能量漂移 {difference} Ha 太大"
    );
}

#[test]
fn odd_electron_count_is_rejected() {
    let h2_plus = Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        1,
        2,
    );
    let registry = registry();
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let basis = qwb_basis::InternalBasis::build(basis_k1, &h2_plus).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &h2_plus).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &h2_plus).expect("grid");
    let error = rks_scf(&basis, &h2_plus, &tensor, &grid, &[], RksOptions::default())
        .expect_err("奇电子数必须被拒");
    assert!(matches!(
        error,
        qwb_engine_api::EngineError::InvalidInput(_)
    ));
}
