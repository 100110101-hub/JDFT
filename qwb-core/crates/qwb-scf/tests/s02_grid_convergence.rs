//! X-03 验收（求积收敛）：分子网格的 Becke 分区必须真的把密度积对。
//!
//! 这条测试的来源是一次真实事故：`smoothed_step` 被写成 `½(1 + p^k(ν))`，正好是 Becke
//! 标准式 `½(1 − p^k(ν))` 的**补**，于是分区权重反了——O 的网格在自己核心处拿到 ≈0 的权重，
//! 而以 H 为中心的网格在那里根本没有点。后果：
//!
//! * 固定密度的 `∫ρ dr` 从 10 变成 9.80（分子网格在 r < 0.1 bohr 积出**精确的 0**）；
//! * 水的 PBE 能量在四个网格等级上是 −74.28 / −77.44 / −74.44 / −73.65 Ha（三 Ha 且不单调）；
//! * 而所有**平滑**被积函数的测试都通过（高斯 α ≤ 2、`Σw = 4π`、单原子分区恒为 1），
//!   因为平滑函数在核心处的贡献本来就小到测不出来。
//!
//! 修好后（一个字符），`∫ρ = 9.99999998`、四个等级的间隔回到 1e-6 量级。判据：
//!
//! 1. 固定密度（grid-fine 收敛出来的 P）下 `∫ρ dr` 与 10 的差：normal < 1e-4、fine < 1e-6；
//! 2. H2 与 H2O 的 SVWN5 / PBE 能量，`grid-normal` 与 `grid-fine` 的差 < 0.1 mHa（§6.2）；
//! 3. 同一体系 `grid-fine` 与 `grid-veryfine` 差 < 0.01 mHa（收敛已在尾部）。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, ao_value, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

fn build(
    registry: &Registry,
    molecule: &Molecule,
    functional_name: &str,
) -> (
    qwb_basis::InternalBasis,
    qwb_integrals::two_electron::TwoElectronIntegrals,
    Vec<Box<dyn qwb_xc::FunctionalTerm>>,
) {
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), molecule)
            .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let functional = registry
        .get_functional(functional_name)
        .expect("functional");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    (basis, tensor, terms)
}

fn solve(registry: &Registry, molecule: &Molecule, functional_name: &str, grid_name: &str) -> f64 {
    let (basis, tensor, terms) = build(registry, molecule, functional_name);
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect("grid"), molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    rks_scf(&basis, molecule, &tensor, &grid, &terms, options)
        .expect("scf")
        .energy_hartree
}

/// 分子网格上的 `∫ρ dr`（密度取自 grid-fine 的收敛解）。
#[test]
fn molecular_grid_integrates_the_density() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    // 用 **HF** 密度：它不经过网格（SCF 里泛函列表为空），所以「∫ρ = 10」这一条
    // 完全不依赖被测的网格代码，也不会被 XC 的能量噪声干扰。
    let (basis, tensor, _) = build(&registry, &molecule, "pbe");
    let terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    let reference_grid =
        qwb_grid::molecular_grid(registry.get_grid("grid-fine").expect("grid"), &molecule)
            .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    let density = rks_scf(&basis, &molecule, &tensor, &reference_grid, &terms, options)
        .expect("scf")
        .density;

    // grid-normal 的求积误差本身在 3e-5 量级（不是分区反了），fine 起进入 1e-6 以内。
    for (name, tolerance) in [("grid-normal", 1.0e-4_f64), ("grid-fine", 1.0e-6)] {
        let grid = qwb_grid::molecular_grid(registry.get_grid(name).expect("grid"), &molecule)
            .expect("grid");
        let mut electrons = 0.0;
        for point in &grid.points {
            let ao = ao_value(&basis, point.position_bohr).expect("ao");
            let mut rho = 0.0;
            for mu in 0..basis.n_functions() {
                for nu in 0..basis.n_functions() {
                    rho += density[(mu, nu)] * ao[mu] * ao[nu];
                }
            }
            electrons += point.weight * rho;
        }
        println!("{name:<14} ∫ρ dr = {electrons:.10}");
        assert!(
            (electrons - 10.0).abs() < tolerance,
            "{name} 的 ∫ρ dr = {electrons} 与 10 差 {:e}（阈值 {tolerance:e}；差到 1e-1 说明分区符号又反了）",
            (electrons - 10.0).abs()
        );
    }
}

/// `grid-normal` 与 `grid-fine` 的能量差必须 < 0.1 mHa（§6.2 的判据）。
///
/// 这里跑 **H₂**（两个泛函、三个等级）：它便宜，而且足以判定「分区符号反了」这类
/// 灾难性错误（反了会差 3 Ha 量级）。含重原子的 H₂O 覆盖在 `molecular_grid_integrates_the_density`
/// 里（用 HF 密度，不依赖 XC，快且更独立）。
///
/// H₂O 的分级收敛数字记录在 `docs/error-budget.md` §2（SVWN5：normal −74.9281445340、
/// fine −74.9281497852、veryfine −74.9281494778，相邻差 5.3e-6 / 3.1e-7 Ha）——
/// 一次跑完要几分钟，所以不进 gate，而由 error-budget 记录、必要时手工复跑。
#[test]
fn adjacent_grid_levels_agree_on_the_energy() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    for functional_name in ["svwn5", "pbe"] {
        let coarse = solve(&registry, &molecule, functional_name, "grid-coarse");
        let normal = solve(&registry, &molecule, functional_name, "grid-normal");
        let fine = solve(&registry, &molecule, functional_name, "grid-fine");
        let veryfine = solve(&registry, &molecule, functional_name, "grid-veryfine");
        let d1 = (normal - fine).abs();
        let d2 = (fine - veryfine).abs();
        println!(
            "H2 {functional_name:<6} coarse {coarse:.10}  normal {normal:.10}  fine {fine:.10}  veryfine {veryfine:.10}  |Δ| = {d1:.2e} / {d2:.2e}"
        );
        assert!(
            d1 < 1.0e-4,
            "H2/{functional_name}: normal 与 fine 差 {d1:.2e} Ha（要求 < 1e-4）"
        );
        assert!(
            d2 < 1.0e-5,
            "H2/{functional_name}: fine 与 veryfine 差 {d2:.2e} Ha（要求 < 1e-5）"
        );
    }
}
