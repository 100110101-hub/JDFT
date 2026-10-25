//! 定位 def2-SVP 上 2.3e-4 的解析梯度偏差：是 **d 壳层**，还是 **XC 的网格项**？
//!
//! 四象限（数值全部来自本文件与 `g01_analytic_def2svp`）：
//!
//! | | s/p 基组（STO-3G） | d 基组（def2-SVP） |
//! | --- | --- | --- |
//! | HF（无 XC） | `g01_analytic_water` 6.7e-7 ✓ | def2-SVP/HF 1.14e-7 ✓ |
//! | PBE（有 XC 网格） | 3.30e-4 ✗ → 修好后 1e-6 量级 | 2.28e-4 ✗ → 修好后 1e-6 量级 |
//!
//! 结论（2026-10-26）：**不是 d 壳层**（HF 在 def2-SVP 上就是 1e-7），是 XC 的**几何导数**
//! —— `xc_gradient_terms` 只算了"密度随核移动"，没算"网格随核移动"（Becke 权重导数
//! + 网格点位置导数）。H2 上被对称性抹平，所以 `g01_analytic_pbe` 一直是 8.4e-7。
//!
//! 修法（生产路径）：`qwb_engine::gradient::xc_geometric_derivative` 在**固定密度**下
//! 对 `E_xc(R)` 做中心差分，接进 `rks_analytic_gradient_with_xc`。
//! 本文件的断言就是"修好了"的判据：两条都要 < 5e-6。
use std::path::PathBuf;

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine::opt::{OptOptions, analytic_energy_and_gradient};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;

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

/// 生产路径的梯度（解析 1e/2e/Pulay + XC 几何项的固定密度差分）。
fn analytic(registry: &Registry, molecule: &Molecule, basis: &str, method: &str) -> Vec<[f64; 3]> {
    let options = OptOptions::new(basis, method, "grid-normal");
    let (_, gradient) =
        analytic_energy_and_gradient(registry, molecule, &options).expect("gradient");
    gradient
}

fn numeric(registry: &Registry, molecule: &Molecule, basis: &str, method: &str) -> Vec<[f64; 3]> {
    let mut options = GradientOptions::new(basis.to_string(), method.to_string(), "grid-normal");
    options.delta_bohr = 1.0e-3;
    numerical_gradient(registry, molecule, &options).expect("numerical gradient")
}

fn compare(label: &str, analytic: &[[f64; 3]], numeric: &[[f64; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize, 0.0, 0.0);
    for (atom, (a_row, n_row)) in analytic.iter().zip(numeric).enumerate() {
        for axis in 0..3 {
            let difference = (a_row[axis] - n_row[axis]).abs();
            if difference > worst {
                worst = difference;
                worst_at = (atom, axis, a_row[axis], n_row[axis]);
            }
        }
    }
    println!(
        "{label}: 最大分量差 {worst:.3e} Ha/Bohr（atom {}, axis {}: 解析 {:.10} vs 数值 {:.10}）",
        worst_at.0, worst_at.1, worst_at.2, worst_at.3
    );
    worst
}

#[test]
#[ignore = "慢档（约 1 分钟）：STO-3G 的 PBE 数值梯度"]
fn sto3g_pbe_water_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let a = analytic(&registry, &molecule, "sto-3g", "pbe");
    let n = numeric(&registry, &molecule, "sto-3g", "pbe");
    let worst = compare("H2O/STO-3G/PBE", &a, &n);
    assert!(worst < 5.0e-6, "STO-3G 的 PBE 梯度差 {worst:.3e}");
}

#[test]
#[ignore = "慢档（release 下约 1 分钟）：def2-SVP 的 HF 数值梯度"]
fn def2svp_hf_water_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let a = analytic(&registry, &molecule, "def2-svp", "");
    let n = numeric(&registry, &molecule, "def2-svp", "");
    let worst = compare("H2O/def2-SVP/HF", &a, &n);
    assert!(worst < 5.0e-6, "def2-SVP 的 HF 梯度差 {worst:.3e}");
}
