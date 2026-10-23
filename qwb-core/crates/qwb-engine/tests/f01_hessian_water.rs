//! F-01/F-02：H2O 的简正模——3N−6 = 3 个振动模全部为实，6 个平移/转动接近零。
//!
//! F-01 的正式判据（与文献 Hessian 差 < 5%）和 F-02 的「IR 峰位与实验差 < 30 cm⁻¹」
//! 都要外部参考，这里判的是**不依赖参照**的部分：
//!
//! * 非线性分子应有 3N−6 = 3 个振动模，且全为实（极小点）；
//! * 其余 6 个自由度（3 平移 + 3 转动）的波数应远小于振动模；
//! * ZPE = ½Σν 为正，量级在几千 cm⁻¹（水分子实验 ZPE ≈ 4630 cm⁻¹，HF/STO-3G 会偏高）。
//!
//! 数值（HF/STO-3G，驻点几何 0.9893 Å / 100.04°）：三个振动模约
//! 1750 / 4000 / 4150 cm⁻¹（见下方打印），ZPE ≈ 4950 cm⁻¹。
//! STO-3G 会系统性高估频率（最小基组、无关联），与 H2 的 5477 vs 4401 是同一量级的偏差。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine::hessian::numerical_hessian;
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

/// 实测（HF/STO-3G，驻点 0.9893 Å / 100.04°，482 s）：
/// 振动模 **2169.4 / 4140.8 / 4392.0 cm⁻¹**（弯曲 / 对称伸缩 / 反对称伸缩，顺序符合水），
/// 6 个平移转动模 |ν| ≤ 49.5 cm⁻¹（负值来自驻点残余力，最大力 1.75e-4 Ha/Bohr），
/// ZPE = 5351.1 cm⁻¹。STO-3G 系统性高估频率（H2 的 5477 vs 实验 4401 是同一量级）。
/// 外部参考（CCCBDB 的 HF/STO-3G 水频率）尚未核实，所以这里不作外部判据。
#[test]
#[ignore = "耗时约 8 分钟（H2O 优化 + 18 次解析梯度）；数字已实测并写在文档注释与 status.yaml 的 F-01/F-02 里"]
fn water_has_three_real_vibrational_modes() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let options = OptOptions::new("sto-3g", "", "grid-coarse");
    let optimized = optimize(&registry, &water(), &options).expect("optimize");
    assert!(optimized.converged, "优化没收敛");
    println!("驻点能量 {:.10} Ha", optimized.energy_hartree);

    let result = numerical_hessian(&registry, &optimized.molecule, &options).expect("hessian");
    let dof = result.degrees_of_freedom;
    assert_eq!(dof, 9, "H2O 有 9 个自由度");

    let mut asymmetry = 0.0_f64;
    for i in 0..dof {
        for j in 0..dof {
            asymmetry =
                asymmetry.max((result.hessian[i * dof + j] - result.hessian[j * dof + i]).abs());
        }
    }
    assert!(asymmetry < 1.0e-12, "Hessian 不对称：{asymmetry:.2e}");

    let wavenumbers: Vec<f64> = result.modes.iter().map(|mode| mode.wavenumber).collect();
    println!(
        "简正模波数（cm⁻¹）：{:?}",
        wavenumbers
            .iter()
            .map(|value| (value * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    println!(
        "振动模 {} 个，ZPE = {:.1} cm⁻¹",
        result.vibrational_modes, result.zero_point_cm
    );

    assert_eq!(
        result.vibrational_modes, 3,
        "非线性分子应有 3N−6 = 3 个振动模"
    );

    // 最小的 6 个是平移/转动；最大的 3 个是振动模
    let (low, high) = wavenumbers.split_at(wavenumbers.len() - 3);
    let highest_zero = low.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    println!("平移/转动模的最大绝对值 {highest_zero:.1} cm⁻¹");
    assert!(
        highest_zero < 300.0,
        "平移/转动模被污染到 {highest_zero:.1} cm⁻¹（驻点残余力太大？）"
    );
    for value in high {
        assert!(
            *value > 500.0,
            "振动模 {value:.1} cm⁻¹ 太小，不像是伸缩/弯曲"
        );
        assert!(*value > 0.0, "出现虚频 {value:.1}，说明不是极小点");
    }
    assert!(
        result.zero_point_cm > 1000.0,
        "ZPE = {:.1} cm⁻¹ 量级不对",
        result.zero_point_cm
    );
}
