//! G-02 验收（部分）：BFGS 在**多原子**体系上的收敛性。
//!
//! G-02 的完整验收是「H2O 优化收敛：最大力 < 4.5e-4 Ha/Bohr；键长误差 < 0.01 Å（实验 0.9572 Å）」。
//! 本轮只能判前一半，原因是**基组**而不是优化器：
//!
//! * `analytic_gradient` 目前只支持 s 型基组（O 有 2p），所以 H2O 走数值梯度；
//! * 数值梯度能跑的模型化学里，STO-3G 是最便宜的，而 STO-3G 的水键长约 0.99 Å，
//!   天生就比实验 0.9572 Å 长 0.03 Å（最小基组过结合）。def2-SVP/PBE 约 0.97 Å（差 0.015），
//!   仍 > 0.01 Å；要达到 0.01 Å 需要 def2-TZVP/cc-pVTZ 级别，而 D-05/D-06 的数据还没有。
//!
//! 所以这条测试判的是「优化器能在 H2O 上收敛到驻点」，并把实测几何打印出来；
//! 键长判据的缺口记在 `docs/status.yaml` 的 G-02 里，不在这里假装通过。
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

fn max_force(gradient: &[[f64; 3]]) -> f64 {
    gradient
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// 键长（Å）与键角（度）。
fn geometry(molecule: &Molecule) -> (f64, f64, f64) {
    let o = molecule.atoms[0].position_bohr();
    let h1 = molecule.atoms[1].position_bohr();
    let h2 = molecule.atoms[2].position_bohr();
    let distance = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let r1 = distance(o, h1);
    let r2 = distance(o, h2);
    let r12 = distance(h1, h2);
    let cosine = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2);
    (
        r1 * ANGSTROM_PER_BOHR,
        r2 * ANGSTROM_PER_BOHR,
        cosine.clamp(-1.0, 1.0).acos().to_degrees(),
    )
}

/// 实测（HF/STO-3G，数值梯度）：E = −74.9659011964 Ha，O–H 0.9893 / 0.9893 Å，
/// ∠HOH 100.04°，最大力 1.75e-4 Ha/Bohr，5 步收敛。这就是 STO-3G 的模型化学几何
/// （实验 0.9572 Å / 104.5°），所以「键长误差 < 0.01 Å」这条 G-02 判据**没有满足**——
/// 差的是基组，不是优化器。
///
/// 耗时约 8 分钟（18 次位移 × 5 步 × 每次 SCF），所以不进 gate。
#[test]
#[ignore = "耗时约 8 分钟（数值梯度的 H2O 优化）；数字已实测并写在文档注释与 status.yaml 的 G-02 里"]
fn h2o_optimises_to_a_stationary_point() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let mut options = OptOptions::new("sto-3g", "", "grid-coarse");
    options.max_iterations = 25;
    options.force_threshold = 4.5e-4;
    let result = optimize(&registry, &water(), &options).expect("optimize");
    let (r1, r2, angle) = geometry(&result.molecule);
    println!(
        "H2O/STO-3G/HF：E = {:.10} Ha，O–H {r1:.4} / {r2:.4} Å，∠HOH {angle:.2}°，\
         最大力 {:.2e}，{} 步，解析梯度 = {}",
        result.energy_hartree,
        max_force(&result.gradient),
        result.iterations,
        result.used_analytic_gradient
    );
    println!("实验：O–H 0.9572 Å，∠HOH 104.5°（STO-3G 的模型化学值约 0.99 Å / 100°）");
    assert!(
        !result.used_analytic_gradient,
        "H2O 有 p 函数，解析梯度还不支持，这里应当走数值梯度"
    );
    assert!(result.converged, "H2O 优化没有收敛");
    assert!(
        max_force(&result.gradient) < 4.5e-4,
        "最大力 {:.2e} 超过 4.5e-4",
        max_force(&result.gradient)
    );
    // 对称性必须保持：两个 O–H 键长相等
    assert!(
        (r1 - r2).abs() < 1.0e-3,
        "两个 O–H 键长 {r1:.4} / {r2:.4} 应相等（C2v 对称）"
    );
    // 模型化学的量级检查（不是实验判据）
    assert!(
        (0.95..1.02).contains(&r1),
        "STO-3G 的 O–H 键长 {r1:.4} Å 超出合理范围"
    );
    assert!(
        (97.0..105.0).contains(&angle),
        "∠HOH {angle:.2}° 超出合理范围"
    );
}
