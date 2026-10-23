//! S-03 验收：Pulay DIIS 在**低对称、含 p 函数、GGA** 的体系上确实起作用。
//!
//! 这条测试补的是 S-03 缺的那一半：DIIS 的机制与契约此前已在 H4 丛（`s03_diis.rs`）上测过，
//! 但「加速器真的有用」当时只在 H/C 的 s/p 体系上验证过，把 H2O/STO-3G/PBE 记成了
//! in_progress（H2O 要 p 函数、PBE 要 GGA，两者都还没落地）。
//!
//! 三条断言：
//!
//! 1. 开 DIIS：H2O/STO-3G/PBE **收敛**，且步数远小于不开的预算；
//! 2. 关 DIIS：同样预算内**不收敛**（返回 `ScfNotConverged`）——这才是「加速器真的有用」
//!    的硬证据；为了控制耗时，关 DIIS 这边用有限预算（见 `NO_DIIS_BUDGET`）；
//! 3. 收紧能量阈值后能量不变（< 1e-7），即收敛到的是同一驻点。
//!
//! 数值（grid-coarse）：E = −74.2760 Ha、E_xc = −8.3891 Ha。注意这条**不能用来看绝对能量**：
//! 同一体系在 grid-normal 上是 −77.4363 Ha，网格等级差 3.16 Ha（见 docs/error-budget.md §2）。
//! PBE/STO-3G 的水没有可引用的
//! 外部表值，所以这里**不把总能量当外部参照**（§7.0）：本测试判的是加速器与自洽性。
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚
use std::path::PathBuf;

use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, RksResult, rks_scf};

/// 关 DIIS 时的迭代预算：既够证明「不收敛」，又不至于让 gate 变慢。
const NO_DIIS_BUDGET: u32 = 30;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

fn solve(use_diis: bool, energy_threshold: f64, max_iter: u32) -> Result<RksResult, EngineError> {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(basis_k1, &molecule).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");
    let functional = registry.get_functional("pbe").expect("pbe");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    let mut options = RksOptions::default();
    options.max_iter = max_iter;
    options.energy_threshold = energy_threshold;
    options.use_diis = use_diis;
    rks_scf(&basis, &molecule, &tensor, &grid, &terms, options)
}

#[test]
fn diis_makes_water_pbe_converge() {
    let result = solve(true, 1.0e-10, 400).expect("开 DIIS 应当收敛");
    println!(
        "H2O/STO-3G/PBE（DIIS 开）: E = {:.8} Ha, E_xc = {:.8} Ha, iters = {}",
        result.energy_hartree, result.exchange_correlation_hartree, result.iterations
    );
    assert!(result.scf_converged, "开 DIIS 却没收敛");
    assert!(
        result.exchange_correlation_hartree < 0.0,
        "E_xc = {} 应为负（PBE 交换+相关都在起作用）",
        result.exchange_correlation_hartree
    );
    assert!(
        result.iterations < NO_DIIS_BUDGET,
        "开 DIIS 也要 {} 步，与关 DIIS 的 {} 步预算比不出快慢",
        result.iterations,
        NO_DIIS_BUDGET
    );

    // 关 DIIS：同一预算内不收敛。这条比「步数更多」有信息量，而且成本有界。
    match solve(false, 1.0e-10, NO_DIIS_BUDGET) {
        Err(EngineError::ScfNotConverged { iters, max }) => {
            println!("H2O/STO-3G/PBE（DIIS 关）: {iters}/{max} 步未收敛");
            assert_eq!(iters, max);
        }
        Err(other) => panic!("关 DIIS 期望 ScfNotConverged，实得 {other:?}"),
        Ok(plain) => panic!(
            "关 DIIS 竟然在 {} 步内收敛了——与开 DIIS 的 {} 步相比，加速器的作用需要重新论证",
            plain.iterations, result.iterations
        ),
    }

    // 收紧阈值：同一驻点。
    let tight = solve(true, 1.0e-11, 400).expect("收紧阈值后应当收敛");
    println!(
        "阈值 1e-11: E = {:.8} Ha, iters = {}",
        tight.energy_hartree, tight.iterations
    );
    assert!(tight.scf_converged, "收紧阈值后不收敛");
    assert!(
        (tight.energy_hartree - result.energy_hartree).abs() < 1.0e-7,
        "收紧阈值后能量变了：{} vs {}",
        result.energy_hartree,
        tight.energy_hartree
    );
}
