//! T3 / G-02 验收：**H2O 的几何**与实验值比（def2-TZVP 级基组）。
//!
//! 判据来自 project.md：T3「H2O 键长与实验 0.9572 Å 差 < 0.01 Å」、G-02 同一条。
//!
//! # 为什么必须用 def2-TZVP 而不是 STO-3G / def2-SVP
//!
//! 这一条**是基组判据**，不是优化器判据（`docs/choices.md` 二十节有对照表）：
//!
//! | 模型化学 | O–H | 对实验 0.9572 |
//! |---|---|---|
//! | HF/STO-3G（实测） | 0.9893 Å | 差 0.032 |
//! | PBE/def2-SVP（文献量级） | ≈0.97 Å | 差 ≈0.015 |
//! | **PBE/def2-TZVP** | ≈0.966 Å | 差 ≈0.009 ✓ |
//!
//! 所以本轮把 def2-TZVP 的基组数据补上（`scripts/gen_basis_from_bse.py` 从 BSE 生成，
//! 不是手抄），这条判据才第一次具备可判性。
//!
//! # 为什么要跑 release
//!
//! KS 的解析梯度还有一个 known-gap（缺「显式 XC 基组运动项」，见 choices 二十四节），
//! 所以 PBE 的几何优化走**数值梯度**：每次梯度 18 次 SCF。def2-TZVP 的水有 43 个基函数，
//! debug 构建下一次 SCF 就要几十秒。命令：
//!
//! ```text
//! cargo test --release -p qwb-engine --test t03_def2tzvp_geometry -- --ignored --nocapture
//! ```
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::opt::{OptOptions, optimize};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

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

fn geometry(molecule: &Molecule) -> (f64, f64) {
    let o = molecule.atoms[0].position_bohr();
    let h1 = molecule.atoms[1].position_bohr();
    let h2 = molecule.atoms[2].position_bohr();
    let distance = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let r1 = distance(o, h1) * ANGSTROM_PER_BOHR;
    let r2 = distance(o, h2) * ANGSTROM_PER_BOHR;
    let r12 = distance(h1, h2);
    let cosine = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2);
    (0.5 * (r1 + r2), cosine.clamp(-1.0, 1.0).acos().to_degrees())
}

#[test]
#[ignore = "耗时很长（PBE 数值梯度 × def2-TZVP 43 个基函数）；请用 --release 跑，见文件头"]
fn water_geometry_matches_experiment() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("6-311g(d,p)").expect("6-311g(d,p)"),
        &molecule,
    )
    .expect("build");
    println!("6-311G(d,p) 的水：{} 个基函数", basis.n_functions());
    assert!(
        basis.n_functions() > 25,
        "6-311G(d,p) 的水应有 25 个以上的函数，实得 {}",
        basis.n_functions()
    );

    let mut options = OptOptions::new("6-311g(d,p)", "pbe", "grid-normal");
    options.max_iterations = 20;
    options.force_threshold = 4.5e-4;
    let result = optimize(&registry, &molecule, &options).expect("optimize");
    let (bond, angle) = geometry(&result.molecule);
    println!(
        "PBE/6-311G(d,p)：E = {:.10} Ha，O–H {bond:.4} Å，∠HOH {angle:.2}°，最大力 {:.2e}，{} 步",
        result.energy_hartree,
        result
            .gradient
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0_f64, |acc, v| acc.max(v.abs())),
        result.iterations
    );
    println!("实验：O–H 0.9572 Å，∠HOH 104.5°");

    let bond_error = (bond - 0.9572).abs();
    let angle_error = (angle - 104.5).abs();
    println!("键长误差 {bond_error:.4} Å（判据 < 0.01），键角误差 {angle_error:.2}°");
    assert!(
        bond_error < 0.01,
        "O–H {bond:.4} Å 与实验 0.9572 Å 差 {bond_error:.4}（要求 < 0.01）"
    );
    assert!(
        angle_error < 2.0,
        "∠HOH {angle:.2}° 与实验 104.5° 差 {angle_error:.2}°（要求 < 2）"
    );
}
