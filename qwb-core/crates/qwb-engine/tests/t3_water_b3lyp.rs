//! T3 的键长判据（**hybrid = PBE0** 版本）：H2O 的 O–H 与实验 0.9572 Å 差 < 0.01 Å。
//!
//! # 为什么换成 hybrid
//!
//! `t3_water_geometry.rs` 已经量过：**PBE** 在 def2-SVP / 6-311G(d,p) / def2-TZVP 上
//! 收敛到 0.9682 / 0.9682 / 0.9689 Å —— 基组改进只挪 0.0007，说明 0.0117 的偏差来自**泛函**
//! （PBE 的 O–H 系统性偏长）。要压进 0.01 Å 必须换 hybrid；D-11/D-12 收口之后 B3LYP 可用，
//! 这个测试就是把那条补齐。用 **PBE0**（数据层可以注册的 hybrid）：B3LYP 的
//! requires.laplacian 为 true 被 K1 的 phase-1 语义门禁拒绝（冻结契约），所以只能选 PBE0。
//!
//! 已知的 hybrid 键长参考：B3LYP/6-31G(d) 0.9646 Å、PBE0/def2-SVP 约 0.963 Å，
//! 都在 0.01 Å 判据内。本测试**不预设**参考值，直接跑优化再看结果。
//!
//! 用数值梯度（KS 解析梯度还缺 v_xc 的基组移动项，见 docs/choices.md 二十四节），
//! grid-coarse（网格等级对平衡键长的影响远小于泛函/基组效应）。AO 三元组缓存之后
//! 单次 SCF 明显变快，所以这个测试比 PBE/def2-TZVP 那个版本更容易跑完。
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

/// `(O–H1, O–H2, ∠HOH)`，长度单位 Å。
fn geometry(molecule: &Molecule) -> (f64, f64, f64) {
    let o = molecule.atoms[0].position_bohr();
    let h1 = molecule.atoms[1].position_bohr();
    let h2 = molecule.atoms[2].position_bohr();
    let distance = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let r1 = distance(o, h1) * ANGSTROM_PER_BOHR;
    let r2 = distance(o, h2) * ANGSTROM_PER_BOHR;
    let r12 = distance(h1, h2) * ANGSTROM_PER_BOHR;
    let cos = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2);
    let angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
    (r1, r2, angle)
}

fn max_force(gradient: &[[f64; 3]]) -> f64 {
    gradient
        .iter()
        .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt())
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "耗时（数值梯度的 PBE0/def2-SVP 优化）；数字记在 status.yaml 的 T3"]
fn pbe0_def2svp_water_geometry_matches_experiment() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let mut options = OptOptions::new("def2-svp", "pbe0", "grid-coarse");
    options.max_iterations = 12;
    options.force_threshold = 4.5e-4;
    let result = optimize(&registry, &water(), &options).expect("optimize");
    let (r1, r2, angle) = geometry(&result.molecule);
    println!(
        "PBE0/def2-SVP: E = {:.8} Ha，O–H {r1:.4} / {r2:.4} Å，∠HOH {angle:.2}°，\
         最大力 {:.2e}，{} 步",
        result.energy_hartree,
        max_force(&result.gradient),
        result.iterations
    );
    for (step, (energy, force)) in result.trajectory.iter().enumerate() {
        println!("  步 {step}: E = {energy:.8}，最大力 {force:.2e}");
    }
    assert!(result.converged, "优化没有收敛");
    let bond_error = (r1 - 0.9572).abs().max((r2 - 0.9572).abs());
    println!("O–H 与实验 0.9572 Å 差 {bond_error:.4} Å（T3 判据 < 0.01）");
    assert!(
        bond_error < 0.01,
        "O–H {r1:.4} Å 与实验 0.9572 差 {bond_error:.4}，超出 T3 的 0.01 Å"
    );
}
