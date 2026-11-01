//! SV-03：CDS 能量的**表达式层**验收（σ 查表/拼装 + 面积约定），用单原子极限。
//!
//! 单原子时表面就是一个球，面积是闭式 `4π(R + 0.4)²`，所以这一层可以**独立于**表面分解算法
//! 验证。σ 表本身已由 `scripts/validate_smd_cds.py` 对 PySCF 2.14.0 的 `libsolvent` 验到
//! 水相 1.6e-5 cal/(mol·Å²)、五个非水溶剂 3.8e-7 kcal/mol；这里守的是 Rust 侧的转写：
//! 12 个元素的水相 σ 从数据里读出来之后，`σ·A/1000` 必须复现那些值。
//!
//! 非水那半需要溶剂的六个 Abraham 描述符（`SOLA/SOLB/SOLC/SOLG/SOLH/SOLN`），而本项目的
//! `solvent/*.yaml` 目前只有 `abraham_alpha/beta`、`gamma` 与折射率 —— **缺 SOLC 与 SOLH**。
//! 所以这条测试只钉"非水路径在缺描述符时**显式报错**"，不编造溶剂参数。

use std::path::PathBuf;

use qwb_props::smd::{SmdDescriptors, cds_energy_kcal, single_atom_area_angstrom2};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

#[test]
fn the_water_cds_sigma_reproduces_the_validated_values() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    // 这些是 `cds_atomic.json` 里的表值，也等价于 PySCF 反推出来的 σ（1.6e-5 以内）。
    let expected = [
        ("H", 48.69),
        ("C", 129.74),
        ("N", 0.0),
        ("O", 0.0),
        ("F", 38.18),
        ("S", -9.10),
        ("Cl", 9.82),
        ("Br", -8.72),
    ];
    let mut worst = 0.0_f64;
    for (symbol, sigma) in expected {
        let area = single_atom_area_angstrom2(data, symbol).expect("area");
        let energy_kcal = cds_energy_kcal(data, &[symbol], &[area], true, None).expect("water cds");
        // 水相没有常数项，所以 `E = σ·A/1000`，反解回 σ 应当逐位相同。
        let derived = energy_kcal * 1000.0 / area;
        worst = worst.max((derived - sigma).abs());
        println!("{symbol:3} area={area:9.4} E={energy_kcal:12.7} sigma={derived:11.6}");
    }
    println!("worst |Δσ| = {worst:.3e} cal/(mol·Å²)");
    assert!(worst < 1.0e-9, "sigma round-trip drifted by {worst:.3e}");
}

#[test]
fn the_area_convention_is_the_validated_one() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    assert!((data.cds.area_offset_angstrom - 0.4).abs() < 1.0e-12);
    // H: 4π(1.2+0.4)² = 32.169909；C: 4π(1.7+0.4)² = 55.417694
    // （这两个数来自 `validate_smd_cds.py` 打印的面积列，不是从公式里抄回来的）。
    let h = single_atom_area_angstrom2(data, "H").expect("H");
    let c = single_atom_area_angstrom2(data, "C").expect("C");
    assert!(
        (h - 32.169_909).abs() < 1.0e-4,
        "H area = {h} (expected 4*pi*(1.2+0.4)^2 = 32.169909)"
    );
    assert!(
        (c - 55.417_694).abs() < 1.0e-4,
        "C area = {c} (expected 4*pi*(1.7+0.4)^2 = 55.417694)"
    );
}

#[test]
fn the_nonaqueous_path_requires_the_descriptors() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let area = single_atom_area_angstrom2(data, "C").expect("area");
    let error = cds_energy_kcal(data, &["C"], &[area], false, None)
        .expect_err("nonaqueous CDS must refuse to run without the descriptors");
    println!("{error}");
    assert!(error.contains("descriptors"), "{error}");

    // 给了描述符就能算（数值本身由 validate_smd_cds.py 对官方验证；这里只钉"能算、形状对"）。
    // toluene 的 alpha/beta/gamma/n（本仓库 `solvent/toluene.yaml` 里有）；
    // SOLC/SOLH 还没有进数据，这里取 0 并**明确说明**这是占位，不是溶剂参数。
    let descriptors = SmdDescriptors::new(0.0, 0.14, 0.0, 40.2, 0.0, 1.4961);
    let energy = cds_energy_kcal(data, &["C"], &[area], false, Some(&descriptors)).expect("cds");
    println!("toluene-like descriptors: E_CDS(C) = {energy:+.6} kcal/mol");
    assert!(energy != 0.0 && energy.is_finite());
}

#[test]
fn a_symbol_area_count_mismatch_is_rejected() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let error = cds_energy_kcal(data, &["H", "C"], &[1.0], true, None).expect_err("mismatch");
    assert!(error.contains("surface areas"), "{error}");
}
