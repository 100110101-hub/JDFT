//! KS（PBE / LSDA）的解析梯度：`rks_analytic_gradient` 的完整公式。
//!
//! 之前解析梯度只在 HF 分支可用，KS 上差 1.4e-1。补完之后发现是**两个**缺口：
//!
//! 1. 缺 `∂E_xc/∂R|_P`（XC 能量的显式基组运动项；σ 通道需要 AO 二阶导数）；
//! 2. `gradient_components` 的 2e 骨架项与 Pulay 用的 `F` 都是按 **HF** 写死的
//!    （库仑 + 交换，`F = H + J − ½K`）。纯泛函根本没有 HF 交换，杂化泛函要乘
//!    `α_hf` —— 直接塞一个 `H + J − ½K + v_xc` 进去会差 8.8e-2 Ha/Bohr。
//!
//! **这个测试验的是"部分解析"那一半。** 2026-10-26 发现它还缺**第三个**缺口：
//! `xc_gradient_terms` 没算网格自己随核移动（Becke 权重导数），在 H2 上被对称性抹平
//! （所以这里一直是 8.4e-7），在 H2O 上是 3.30e-4。生产路径改用
//! `rks_analytic_gradient_with_xc` + 固定密度的 XC 几何差分；对称破缺体系上的验收在
//! `tests/g01_gradient_localize.rs`（H2O/STO-3G/PBE 1.48e-7）与
//! `tests/g01_analytic_def2svp.rs`（H2O/def2-SVP/PBE 1.59e-7）。
//!
//! 判据（§6 G-01）：PBE/STO-3G 的 H2（拉长到 1.1 Å）上，解析 vs 数值 < 5e-6。
//! LSDA 用同一条代码路径作对照，把「v_xc 接线」与「σ 通道二阶导数」分开。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_analytic_gradient, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// H2 拉长到 1.1 Å：梯度大、SCF 便宜（2 个基函数）。
///
/// 用 H2 而不是 H2O：KS 的数值梯度要 12~36 次 SCF，H2O/PBE 在 debug 构建里是十几分钟量级。
/// 换 H2 后同样的**代码路径**（KS Fock + Pulay + XC 显式项）只要几秒，能进 gate。
fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.55]),
            Atom::new("H", [0.0, 0.0, 0.55]),
        ],
        0,
        1,
    )
}

fn terms_for(registry: &Registry, name: &str) -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let functional = registry.get_functional(name).expect("functional");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

/// 跑一次 SCF，返回解析梯度（`rks_analytic_gradient`）与数值梯度的最大差。
fn compare(functional: &str, grid_level: &str, hf_exchange: f64, delta: f64) -> (f64, f64, f64) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_level).expect("grid"), &molecule)
        .expect("grid");
    let terms = terms_for(&registry, functional);
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.hf_exchange = hf_exchange;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");

    let analytic = rks_analytic_gradient(
        &basis,
        &molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &result.density,
        &terms,
        hf_exchange,
    )
    .expect("analytic");

    let mut options = GradientOptions::new("sto-3g", functional, grid_level);
    options.delta_bohr = delta;
    let numeric = numerical_gradient(&registry, &molecule, &options).expect("numeric");

    let mut worst = 0.0_f64;
    for atom in 0..analytic.len() {
        for axis in 0..3 {
            let error = (analytic[atom][axis] - numeric[atom][axis]).abs();
            worst = worst.max(error);
        }
    }
    (worst, analytic[0][2], numeric[0][2])
}

#[test]
fn lsda_analytic_gradient_matches_numerical() {
    let (worst, analytic, numeric) = compare("svwn5", "grid-coarse", 0.0, 1.0e-3);
    println!("LSDA：解析 z = {analytic:>13.9}  数值 z = {numeric:>13.9}  最大差 {worst:.2e}");
    assert!(
        worst < 5.0e-6,
        "LSDA 解析与数值梯度最大差 {worst:.2e}（要求 < 5e-6）"
    );
}

#[test]
fn pbe_analytic_gradient_matches_numerical() {
    let (worst, analytic, numeric) = compare("pbe", "grid-coarse", 0.0, 1.0e-3);
    println!("PBE：解析 z = {analytic:>13.9}  数值 z = {numeric:>13.9}  最大差 {worst:.2e}");
    assert!(
        worst < 5.0e-6,
        "PBE 解析与数值梯度最大差 {worst:.2e}（要求 < 5e-6）"
    );
}

/// 数值参照自身的散布必须比解析–数值的差大，否则"接近"可能只是两边都在同一个
/// 差分误差里 —— 这条对照把"碰巧对上"排除掉。
#[test]
fn analytic_gradient_is_closer_than_the_numerical_scatter() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let mut scatter = 0.0_f64;
    for delta in [1.0e-3, 4.0e-3] {
        let mut options = GradientOptions::new("sto-3g", "pbe", "grid-coarse");
        options.delta_bohr = delta;
        let gradient = numerical_gradient(&registry, &molecule, &options).expect("numeric");
        if delta == 1.0e-3 {
            scatter = gradient[0][2];
        } else {
            scatter = (scatter - gradient[0][2]).abs();
        }
    }
    let (worst, _, _) = compare("pbe", "grid-coarse", 0.0, 1.0e-3);
    println!("数值参照散布 {scatter:.2e}，解析–数值差 {worst:.2e}");
    assert!(
        worst < scatter.max(1.0e-7) * 10.0,
        "解析–数值差 {worst:.2e} 超过数值参照散布 {scatter:.2e} 的 10 倍"
    );
}
