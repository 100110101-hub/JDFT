//! T4 的**实测**（不是判据测试）：HF/6-311G(d,p) 的 H2O 标度谐和频率 vs 实验。
//!
//! 为什么单独一个文件：`f02_frequencies` 用的是 STO-3G（147 s 能跑完），而 T4 的
//! "与实验差 < 30 cm⁻¹"需要**像样的模型化学**。6-311G(d,p) 上 31 个笛卡尔基函数，
//! 18 个位移各要一次 SCF + 解析梯度，debug 构建里是几十分钟量级 —— 所以钉成
//! `#[ignore]`，用 `cargo test --release ... -- --ignored --nocapture` 跑，
//! 数字写进 `docs/status.yaml` 的 T4。
//!
//! 口径（与 `f02_frequencies` 一致）：**谐和频率 × 0.909**（CCCBDB 的 HF/6-311G** 标度因子，
//! NIST CCCBDB Release 22）再与实验基频（1595 / 3657 / 3756 cm⁻¹）比。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::hessian::numerical_hessian;
use qwb_engine::opt::{OptOptions, optimize};
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;

const CCCBDB_SCALE_FACTOR_HF_6_311GSS: f64 = 0.909;

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
#[ignore = "慢档（release 下分钟级到十分钟级）：HF/6-311G(d,p) 的 H2O 优化 + Hessian，T4 的实测"]
fn h2o_hf_6311gdp_scaled_frequencies() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let options = OptOptions::new("6-311g(d,p)", "", "grid-coarse");
    let optimized = optimize(&registry, &water(), &options).expect("optimize");
    assert!(optimized.converged, "优化没收敛，频率没有意义");
    println!(
        "驻点能量 {:.10} Ha（HF/6-311G(d,p)）",
        optimized.energy_hartree
    );
    let molecule = optimized.molecule;

    let began = std::time::Instant::now();
    let hessian = numerical_hessian(&registry, &molecule, &options).expect("hessian");
    println!("Hessian 用时 {:.1} s", began.elapsed().as_secs_f64());

    let mut vibrational: Vec<f64> = hessian
        .modes
        .iter()
        .map(|mode| mode.wavenumber)
        .filter(|value| value.abs() > 100.0)
        .collect();
    vibrational.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let experiment = [3756.0_f64, 3657.0, 1595.0];
    println!("谐和 / 标度后 / 实验 / 差（cm⁻¹）");
    let mut worst = 0.0_f64;
    for (index, harmonic) in vibrational.iter().enumerate() {
        let scaled = harmonic * CCCBDB_SCALE_FACTOR_HF_6_311GSS;
        let reference = experiment.get(index).copied().unwrap_or(f64::NAN);
        let error = (scaled - reference).abs();
        worst = worst.max(error);
        println!(
            "{harmonic:>9.1} {scaled:>9.1} {reference:>9.1} {:>+9.1}",
            scaled - reference
        );
    }
    println!("振动模 {} 个，最差偏差 {worst:.1} cm⁻¹", vibrational.len());
    assert_eq!(vibrational.len(), 3, "H2O 应有 3 个振动模");
    // 这条测试是**测量**而不是判据：T4 的 "差 < 30 cm⁻¹" 由 docs/status.yaml 记账决定，
    // 因为本环境的模型化学（HF/6-311G(d,p) + 0.909 标度）够不够，要用实测数字说话，
    // 不能写一个先验的断言再让它去"碰巧通过"。
    assert!(worst.is_finite(), "偏差必须是有限值");
}
