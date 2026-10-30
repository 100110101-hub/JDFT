//! SC-02（UKS 半）的引擎级验收：O2/SVWN5/STO-3G 的解析梯度 vs 数值梯度 < 1e-6。
//!
//! 两个诊断结论（`docs/choices.md` 七十五）决定了本测试的形状：
//!
//! * `W` 里 `v_xc` 的系数是对的（含/不含/2× 三种实测：7.6e-3 / 2.9e-2 / 4.4e-2）；
//! * 4% 的差来自**收敛辅助分支**：能级移动的不动点交换子残差 1.1e-2，不是 Brillouin 点。
//!   正文算法分支（DIIS）上残差 6.7e-11，解析梯度与数值梯度一致到 6.0e-7。
//!
//! OH（双重态）在这个基组/泛函上**达不到真驻点**（DIIS 400 步不收敛；辅助分支收敛但残差
//! 1.1e-2），所以 `uks` + `opt` 仍保持显式拒绝，OH 的梯度不对外提供 —— 宁缺勿错。
use std::path::PathBuf;

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine::opt::{OptOptions, analytic_energy_and_gradient};
use qwb_engine_api::{Atom, Molecule, ScfType};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn dioxygen() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, -0.603_5]),
            Atom::new("O", [0.0, 0.0, 0.603_5]),
        ],
        0,
        3,
    )
}

fn hydroxyl() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.969_66]),
        ],
        0,
        2,
    )
}

/// 与 [`sc02_uks_gradient_matches_numerical_for_o2`] 同一条判据，换 OH（双重态）。
#[test]
fn sc02_uks_gradient_matches_numerical_for_oh() {
    check_gradient("OH", &hydroxyl());
}

fn check_gradient(name: &str, molecule: &Molecule) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let mut options = OptOptions::new("sto-3g", "svwn5", "grid-coarse");
    options.scf_type = Some(ScfType::Uks);
    options.scf_energy_threshold = 1.0e-9;
    options.scf_density_threshold = 1.0e-7;
    let (energy, analytic) =
        analytic_energy_and_gradient(&registry, molecule, &options).expect("uks analytic");
    let mut gradient_options = GradientOptions::new("sto-3g", "svwn5", "grid-coarse");
    gradient_options.scf_type = Some(ScfType::Uks);
    gradient_options.delta_bohr = 1.0e-3;
    let numeric = numerical_gradient(&registry, molecule, &gradient_options).expect("numeric");
    let mut worst = 0.0_f64;
    println!("{name}/SVWN5/STO-3G (UKS): E = {energy:.9} Ha");
    for atom in 0..analytic.len() {
        for axis in 0..3 {
            let difference = (analytic[atom][axis] - numeric[atom][axis]).abs();
            worst = worst.max(difference);
            println!(
                "  atom {atom} axis {axis}: 解析 {:>13.9}  数值 {:>13.9}  差 {difference:.2e}",
                analytic[atom][axis], numeric[atom][axis]
            );
        }
    }
    println!("  {name} 最大差 {worst:.3e} Ha/Bohr（判据 1e-6）");
    assert!(
        worst < 1.0e-6,
        "{name}: UKS analytic vs numerical gradient differs by {worst:.3e} Ha/Bohr"
    );
}

#[test]
fn sc02_uks_gradient_matches_numerical_for_o2() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = dioxygen();
    let mut options = OptOptions::new("sto-3g", "svwn5", "grid-coarse");
    options.scf_type = Some(ScfType::Uks);
    options.scf_energy_threshold = 1.0e-10;
    options.scf_density_threshold = 1.0e-9;
    let (energy, analytic) =
        analytic_energy_and_gradient(&registry, &molecule, &options).expect("uks analytic");
    let mut gradient_options = GradientOptions::new("sto-3g", "svwn5", "grid-coarse");
    gradient_options.scf_type = Some(ScfType::Uks);
    gradient_options.delta_bohr = 1.0e-3;
    let numeric = numerical_gradient(&registry, &molecule, &gradient_options).expect("numeric");
    let mut worst = 0.0_f64;
    println!("O2/SVWN5/STO-3G (UKS): E = {energy:.9} Ha");
    for atom in 0..analytic.len() {
        for axis in 0..3 {
            let difference = (analytic[atom][axis] - numeric[atom][axis]).abs();
            worst = worst.max(difference);
            println!(
                "  atom {atom} axis {axis}: 解析 {:>13.9}  数值 {:>13.9}  差 {difference:.2e}",
                analytic[atom][axis], numeric[atom][axis]
            );
        }
    }
    println!("O2 最大差 {worst:.3e} Ha/Bohr（判据 1e-6）");
    assert!(
        worst < 1.0e-6,
        "UKS analytic vs numerical gradient differs by {worst:.3e} Ha/Bohr"
    );
}
