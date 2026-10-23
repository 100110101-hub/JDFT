//! G-03 验收：**三个分子**的几何优化全部收敛，并给出多步轨迹与收敛报告。
//!
//! G-03 的判据是「H2O、CH4、NH3 三个分子优化全部收敛」。收敛的定义与 T3 一致：
//! 力的最大分量 < 4.5e-4 Ha/Bohr。三个分子都从**故意偏离平衡**的初值出发，
//! 所以轨迹至少几步，能看出 BFGS 是否真的在收敛（而不是「一步到位碰巧对了」）。
//!
//! 模型化学统一用 HF/STO-3G：STO-3G 覆盖 H/C/N/O，而且 HF 不走 XC 网格的那套
//! 开销最大路径，所以这是本仓库唯一能在可接受时间内跑完三分子优化的组合。
//! 键长本身对该模型化学是「已知偏短」（见 `docs/choices.md` 二十节），
//! 本测试判的是**收敛性**，不是与实验比几何。
//!
//! 耗时：三个分子各 5-7 步、每步一次 SCF + 一次解析梯度，实测约 20 分钟，
//! 所以标 `#[ignore]`；判据与数字都记在 `docs/status.yaml` 的 G-03。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::opt::{OptOptions, OptResult, optimize};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// 一个稍微拉长的水（平衡附近但不满）。
fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.80, 0.62]),
            Atom::new("H", [0.0, -0.80, 0.62]),
        ],
        0,
        1,
    )
}

/// 甲烷：四面体，C–H 先取 1.15 Å（平衡约 1.09）。
fn methane() -> Molecule {
    let r = 1.15 / 3.0_f64.sqrt();
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [r, r, r]),
            Atom::new("H", [r, -r, -r]),
            Atom::new("H", [-r, r, -r]),
            Atom::new("H", [-r, -r, r]),
        ],
        0,
        1,
    )
}

/// 氨：锥形，N–H 先取 1.05 Å（平衡约 1.01），HNH 约 107°。
fn ammonia() -> Molecule {
    let bond = 1.05_f64;
    let angle = 107.0_f64.to_radians() / 2.0;
    let y = bond * angle.sin();
    let z = -bond * angle.cos();
    Molecule::new(
        vec![
            Atom::new("N", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, y, z]),
            Atom::new("H", [y * 0.866, -y * 0.5, z]),
            Atom::new("H", [-y * 0.866, -y * 0.5, z]),
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

fn report(label: &str, result: &OptResult) -> f64 {
    println!(
        "{label}：{} 步，E = {:.10} Ha，解析梯度 = {}",
        result.iterations, result.energy_hartree, result.used_analytic_gradient
    );
    println!("  步 | 能量 (Ha)        | 最大力 (Ha/Bohr)");
    for (step, (energy, force)) in result.trajectory.iter().enumerate() {
        println!("  {step:>2} | {energy:>16.10} | {force:.3e}");
    }
    max_force(&result.gradient)
}

#[test]
#[ignore = "耗时约 20 分钟（三个 STO-3G 分子的解析梯度优化）；判据与数字记在 status.yaml 的 G-03"]
fn three_molecules_converge() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let mut options = OptOptions::new("sto-3g", "", "grid-coarse");
    options.max_iterations = 40;
    let cases: [(&str, Molecule); 3] = [("H2O", water()), ("CH4", methane()), ("NH3", ammonia())];
    for (label, molecule) in cases {
        let result = optimize(&registry, &molecule, &options).expect("optimize");
        let force = report(label, &result);
        assert!(result.used_analytic_gradient, "{label} 应当走解析梯度");
        assert!(result.converged, "{label} 没有收敛（最大力 {force:.2e}）");
        assert!(force < 4.5e-4, "{label} 的最大力 {force:.2e} 超过 4.5e-4");
        // 轨迹必须是单调下降的（BFGS + Armijo 的必然结果），否则说明线搜索有问题
        for window in result.trajectory.windows(2) {
            assert!(
                window[1].0 <= window[0].0 + 1.0e-10,
                "{label} 的轨迹出现能量上升：{:?}",
                window
            );
        }
        // 至少两步，否则「多步轨迹」名不副实
        assert!(
            result.trajectory.len() >= 2,
            "{label} 一步就收敛了，轨迹没有说服力"
        );
    }
}
