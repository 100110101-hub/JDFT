//! F-01/F-02 的基础验收：数值 Hessian 与简正模的**内部一致性**。
//!
//! F-01 的判据是「H2O/def2-SVP/PBE Hessian 与文献差 < 5%（相对 Frobenius 范数）」——那需要
//! 外部参考；F-02 的「IR 峰位与实验差 < 30 cm⁻¹」同样。所以这里判的是**不依赖外部参照**
//! 的那几条，它们也足以抓住实现错误：
//!
//! 1. **对称性**：Hessian 必须对称（中心差分 + 并行各自算行，最容易在这里出错）；
//! 2. **曲率一致性**：H2 沿键方向的二阶导必须等于「能量三点拟合」的曲率（同一条物理量、
//!    两条完全独立的路：Hessian 走梯度差分，拟合走能量差分）；
//! 3. **简正模**：H2 只有 1 个振动模（其余 5 个是平移/转动，波数应接近 0），
//!    且该模的频率对应上游离 H2 的伸缩；非线性分子的自由度是 3N。
//!
//! 用 H2/STO-3G/HF 是因为它便宜（6 个位移 × 2 次解析梯度），能进 gate。
//! H2O 的三模检查在 `hessian_water.rs`（`#[ignore]`，几分钟）。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::hessian::{HESSIAN_DELTA_BOHR, numerical_hessian};
use qwb_engine::opt::OptOptions;
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// H2，平衡键长 0.7125 Å（HF/STO-3G）。
fn h2(shift_first_atom: f64) -> Molecule {
    let half = 0.7125 / 2.0;
    Molecule::new(
        vec![
            // 只移动**第一个原子**：这样键长变化量就等于位移量，
            // 三点拟合出来的曲率才直接等于 Hessian 元 H[z0,z0]。
            Atom::new(
                "H",
                [0.0, 0.0, -half + shift_first_atom * ANGSTROM_PER_BOHR],
            ),
            Atom::new("H", [0.0, 0.0, half]),
        ],
        0,
        1,
    )
}

fn energy(molecule: &Molecule) -> f64 {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), molecule)
            .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    rks_scf(&basis, molecule, &tensor, &grid, &[], options)
        .expect("scf")
        .energy_hartree
}

#[test]
fn h2_hessian_has_one_vibrational_mode_and_matches_the_curvature() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 先优化到驻点再算 Hessian：在非驻点几何上，转动模式会被残余力污染成
    // 一百多 cm⁻¹ 的假模（实测 110.5）。驻点处它们应接近 0。
    let options = OptOptions::new("sto-3g", "", "grid-coarse");
    let optimized = qwb_engine::opt::optimize(&registry, &h2(0.0), &options).expect("optimize");
    assert!(optimized.converged, "优化没收敛，Hessian 没有意义");
    let molecule = optimized.molecule;
    println!("驻点能量 {:.10} Ha", optimized.energy_hartree);
    let result = numerical_hessian(&registry, &molecule, &options).expect("hessian");
    let dof = result.degrees_of_freedom;
    assert_eq!(dof, 6, "H2 有 6 个自由度");

    // ① 对称性
    let mut asymmetry = 0.0_f64;
    for i in 0..dof {
        for j in 0..dof {
            asymmetry =
                asymmetry.max((result.hessian[i * dof + j] - result.hessian[j * dof + i]).abs());
        }
    }
    println!("Hessian 不对称度 {asymmetry:.2e}");
    assert!(asymmetry < 1.0e-12, "Hessian 不对称：{asymmetry:.2e}");

    // ② 曲率 vs 能量三点拟合（沿 z 伸缩）
    let shift = HESSIAN_DELTA_BOHR;
    let mut plus = molecule.clone();
    let mut minus = molecule.clone();
    const A2B: f64 = ANGSTROM_PER_BOHR;
    plus.atoms[0].position_angstrom[2] += shift * A2B;
    minus.atoms[0].position_angstrom[2] -= shift * A2B;
    let e_plus = energy(&plus);
    let e_zero = energy(&molecule);
    let e_minus = energy(&minus);
    let curvature = (e_plus - 2.0 * e_zero + e_minus) / (shift * shift);
    // 只移动原子 0 ⇒ 键长变化 = 位移量 ⇒ 能量三点曲率**就是** H[z0,z0]
    let analytic = result.hessian[2 * dof + 2];
    println!("能量三点曲率 {curvature:.8} Ha/Bohr²，Hessian H[z0,z0] = {analytic:.8}");
    assert!(
        (analytic - curvature).abs() < 5.0e-3,
        "Hessian 的 H[z0,z0] = {analytic:.8} 与能量拟合的曲率 {curvature:.8} 不符"
    );

    // ③ 简正模：1 个振动模 + 5 个平移/转动
    let nonzero: Vec<f64> = result
        .modes
        .iter()
        .map(|mode| mode.wavenumber)
        .filter(|value| value.abs() > 500.0)
        .collect();
    println!(
        "简正模波数（cm⁻¹）：{:?}",
        result
            .modes
            .iter()
            .map(|m| (m.wavenumber * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    assert_eq!(nonzero.len(), 1, "H2 应只有一个振动模，实得 {nonzero:?}");
    assert!(nonzero[0] > 0.0, "H2 的伸缩模应为正（极小点）");
    assert!(
        result.zero_point_cm > 0.0,
        "ZPE 应为正，实得 {}",
        result.zero_point_cm
    );
    println!("ZPE = {:.1} cm⁻¹", result.zero_point_cm);
}
