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
    // 单位必须一致：先在 Bohr 下算余弦（角度是纯几何量，与单位无关），
    // 再单独把键长换成 Å。早期版本把 r1/r2 换成 Å 而 r12 留在 Bohr，
    // 于是余弦恒接近 −1、角度恒报 180°——差点据此把「势能面坏了」写进账本。
    let r1 = distance(o, h1);
    let r2 = distance(o, h2);
    let r12 = distance(h1, h2);
    let cosine = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2);
    (
        0.5 * (r1 + r2) * ANGSTROM_PER_BOHR,
        cosine.clamp(-1.0, 1.0).acos().to_degrees(),
    )
}

#[test]
/// 实测（PBE/6-311G(d,p)，release，3 步上限，约 28 分钟）：
/// **O–H 0.9682 Å、∠HOH 103.34°**、最大力 6.58e-4 Ha/Bohr、E = −76.5379429467 Ha。
///
/// 对照实验（0.9572 Å / 104.5°）：键角误差 1.16° ✓，**键长误差 0.0110 Å ✗**（判据 < 0.01）——
/// 差 0.001 Å 未达标。这不是实现错误：PBE 在这个基组上系统性偏长 ~0.01 Å，
/// 而真正能把误差压到 0.01 以内的 def2-TZVP 目前被「f 函数（l=3）球谐变换未实现」挡住
/// （见 status.yaml 的 D-06 与 T3）。所以 T3 保持 in_progress，缺口写清楚而不是改判据。
#[ignore = "PBE 数值梯度 × 30 个基函数、约 28 分钟；实测 0.9682 Å（差 0.0110，未达 0.01）。请用 --release 跑"]
fn water_geometry_matches_experiment() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    // 出发点就是弯曲的实验几何；判据是「优化后仍然弯」且键长误差 < 0.01。
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
    options.max_iterations = 3;
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
    println!("实验：O–H 0.9572 Å（r_e 0.9578），∠HOH 104.5°");
    for (index, atom) in result.molecule.atoms.iter().enumerate() {
        println!("  {} {:?}", index, atom.position_angstrom);
    }

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
