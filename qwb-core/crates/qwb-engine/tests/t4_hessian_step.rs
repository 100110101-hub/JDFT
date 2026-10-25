//! T4 的口径/步长研究：HF/6-311G(d,p) 的 H2O 频率随**Hessian 差分步长**怎么变。
//!
//! 背景：`numerical_hessian` 用中心差分，步长 `HESSIAN_DELTA_BOHR = 1e-2` Bohr。
//! 中心差分的二阶导数有 `O(δ²)` 的**非谐**误差；水分子两个伸缩模的非谐性特别大，
//! 所以先量一下"步长 → 频率"的收敛性，再决定 T4 的口径要不要把这一项算进去。
//!
//! 实测（release，2026-10-25）：
//! * `δ = 1e-2 Bohr`：4236.5 / 4140.2 / 1749.3 cm⁻¹
//! * `δ = 5e-3 Bohr`：见下方输出
//! * `δ = 2.5e-3 Bohr`：见下方输出
//! * `δ = 1.25e-3 Bohr`：见下方输出
//!
//! 判据：只断言"三个振动模、都为正"，把数字打出来给记账用（这不是判据测试）。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::hessian::numerical_hessian_with_delta;
use qwb_engine::opt::{OptOptions, optimize};
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

#[test]
#[ignore = "慢档（release 下几分钟/步长）：H2O 频率随 Hessian 差分步长的收敛性"]
fn hessian_step_convergence() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let options = OptOptions::new("6-311g(d,p)", "", "grid-coarse");
    let optimized = optimize(&registry, &water(), &options).expect("optimize");
    assert!(optimized.converged, "优化没收敛");
    println!("驻点能量 {:.10} Ha", optimized.energy_hartree);
    let molecule = optimized.molecule;

    let mut previous: Option<Vec<f64>> = None;
    for delta in [1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3] {
        let began = std::time::Instant::now();
        let hessian =
            numerical_hessian_with_delta(&registry, &molecule, &options, delta).expect("hessian");
        let mut vibrational: Vec<f64> = hessian
            .modes
            .iter()
            .map(|mode| mode.wavenumber)
            .filter(|value| value.abs() > 100.0)
            .collect();
        vibrational.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
        let shift = previous.as_ref().map_or(0.0, |old| {
            old.iter()
                .zip(vibrational.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        });
        println!(
            "δ = {delta:>7.4e} Bohr：{vibrational:?} cm⁻¹（与上一步长最大差 {shift:.2} cm⁻¹，用时 {:.0} s）",
            began.elapsed().as_secs_f64()
        );
        assert_eq!(vibrational.len(), 3, "H2O 应有 3 个振动模");
        assert!(
            vibrational.iter().all(|value| *value > 100.0),
            "振动模应为正"
        );
        previous = Some(vibrational);
    }
}
