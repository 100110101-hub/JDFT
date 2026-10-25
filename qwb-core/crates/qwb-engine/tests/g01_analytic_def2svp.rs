//! G-01 的缺口补测：**含 d 函数**的基组上，KS 梯度 vs 数值梯度。
//!
//! # 为什么非补不可
//!
//! 在 T8 之前，所有"梯度对不对"的证据都建在 STO-3G 上（H2 只有 s，H2O 只有 s/p）：
//! `g01_analytic`、`g01_analytic_water`、`g01_analytic_pbe`、`t6_b3lyp_gradient` 全是 STO-3G，
//! 而 T8 判据点名的是 **def2-SVP**（O 上有 d 壳层）。
//!
//! 触发这次补测的是一个具体的对不上：H2O/PBE/def2-SVP/opt 用梯度优化得到
//! O–H = 0.9745 Å、∠HOH = 102.08°，而 T3 用**数值**梯度在同一模型化学上记的是 0.9682 Å。
//! 同一模型化学的两个梯度路径不该给出不同的驻点 —— 先怀疑梯度，再怀疑优化器。
//! 结果抓到一个真 bug（XC 的网格几何导数没算，见 `g01_gradient_localize.rs` 的四象限），
//! 修好后这两条判据都成立。
//!
//! # 判据
//!
//! 生产梯度（解析 1e/2e/Pulay + XC 几何项的固定密度差分）与中心差分
//! （`δ = 1e-3` Bohr，每次位移重跑 SCF）的**最大分量差** < 5e-6 Ha/Bohr。
//!
//! 慢档：def2-SVP 水的数值梯度要 18 次 SCF（`#[ignore]`，release 下 1–3 分钟）。
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

fn analytic(registry: &Registry, molecule: &Molecule, basis: &str, method: &str) -> Vec<[f64; 3]> {
    let options = OptOptions::new(basis, method, "grid-normal");
    let (_, gradient) =
        analytic_energy_and_gradient(registry, molecule, &options).expect("analytic gradient");
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
#[ignore = "慢档（release 下 1–3 分钟）：def2-SVP 水的 PBE 数值梯度（18 次 SCF）"]
fn def2svp_pbe_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let analytic = analytic(&registry, &molecule, "def2-svp", "pbe");
    let numeric = numeric(&registry, &molecule, "def2-svp", "pbe");
    let worst = compare("H2O/def2-SVP/PBE", &analytic, &numeric);
    assert!(
        worst < 5.0e-6,
        "def2-SVP（含 d 壳层）的 PBE 梯度与中心差分差 {worst:.3e} Ha/Bohr，超过 5e-6"
    );
}

#[test]
#[ignore = "慢档（release 下 1–3 分钟）：def2-SVP 水的 HF 数值梯度（18 次 SCF）"]
fn def2svp_hf_analytic_gradient_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let analytic = analytic(&registry, &molecule, "def2-svp", "");
    let numeric = numeric(&registry, &molecule, "def2-svp", "");
    let worst = compare("H2O/def2-SVP/HF", &analytic, &numeric);
    assert!(
        worst < 5.0e-6,
        "def2-SVP 的 HF 梯度与中心差分差 {worst:.3e} Ha/Bohr，超过 5e-6"
    );
}
