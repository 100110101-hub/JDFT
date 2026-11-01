//! DA-06 的 D4 **能量层**验收：两体项 + ATM 三体项。
//!
//! 对照物 `tests/reference/d4_dispersion.csv`：官方 dftd4 4.3.1 的 `get_dispersion()`
//! （默认变体 `bj-eeq-atm`，`s9 = 1` 即 **ATM 开**），9 个二聚体 × 5 个泛函 + `water_dimer` 的 q = ±1。
//!
//! # 实测
//!
//! 60 行的最差 |ΔE| = **6.435e-13 Ha**（m06-l 单独 4.8e-13），判据 1e-9。
//!
//! 达到这个精度的关键是 **ATM 的 C6 用"原子部分电荷 = 0"重建** —— 注意这里的"0"是每个原子
//! 的分电荷为零，而不是"把体系总电荷设成 0 再解一遍 EEQ"（后者仍给出 ±0.6 的分电荷，实测让
//! 水二聚体的 ATM 从 −3.2866e-7 变成 −9.6076e-8，差 3.42 倍；见 `docs/choices.md` 一百五十七、
//! 一百五十八）。

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_registry::Registry;
use qwb_scf::d4;

/// K2 的 Å→Bohr（与参考值生成脚本一致）。
const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

type Row = BTreeMap<String, String>;

fn read_csv(path: &std::path::Path) -> Vec<Row> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split(',')
        .map(str::trim)
        .map(String::from)
        .collect();
    lines
        .map(|line| {
            line.split(',')
                .map(str::trim)
                .zip(header.iter())
                .map(|(value, key)| (key.clone(), value.to_string()))
                .collect()
        })
        .collect()
}

fn field<'a>(row: &'a Row, key: &str) -> &'a str {
    row.get(key)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("missing column `{key}`"))
}

/// 上游（dftd4）的泛函键名 → 本项目数据文件的键名。
fn canonical_method(name: &str) -> &str {
    match name {
        "m06l" => "m06-l",
        other => other,
    }
}

fn geometries() -> BTreeMap<String, (Vec<String>, Vec<[f64; 3]>)> {
    let rows = read_csv(&reference_dir().join("d3_dimers.csv"));
    let mut molecules: BTreeMap<String, (Vec<String>, Vec<[f64; 3]>)> = BTreeMap::new();
    for row in &rows {
        let entry = molecules
            .entry(field(row, "dimer").to_string())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(field(row, "element").to_string());
        entry.1.push([
            field(row, "x_angstrom").parse::<f64>().expect("x") * BOHR_PER_ANGSTROM,
            field(row, "y_angstrom").parse::<f64>().expect("y") * BOHR_PER_ANGSTROM,
            field(row, "z_angstrom").parse::<f64>().expect("z") * BOHR_PER_ANGSTROM,
        ]);
    }
    molecules
}

#[test]
fn d4_energies_against_the_official_reference() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = geometries();
    let rows = read_csv(&reference_dir().join("d4_dispersion.csv"));
    assert!(rows.len() >= 50, "expected >= 50 rows, got {}", rows.len());

    let mut worst = 0.0_f64;
    let mut worst_row = String::new();
    let mut worst_atm_share = 0.0_f64;
    for row in &rows {
        let dimer = field(row, "dimer");
        let method = canonical_method(field(row, "method"));
        let charge = field(row, "charge").parse::<f64>().expect("charge");
        let expected = field(row, "energy_hartree").parse::<f64>().expect("energy");
        let (symbols, positions) = molecules
            .get(dimer)
            .unwrap_or_else(|| panic!("{dimer}: geometry missing"));
        let refs: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let result = d4::dispersion_energy(data, method, &refs, positions, charge)
            .unwrap_or_else(|error| panic!("{dimer}/{method}: {error}"));
        let delta = (result.energy_hartree - expected).abs();
        if delta > worst {
            worst = delta;
            worst_row = format!("{dimer}/{method}/q={charge:+}");
        }
        if result.energy_hartree != 0.0 {
            worst_atm_share =
                worst_atm_share.max((result.atm_hartree / result.energy_hartree).abs());
        }
        println!(
            "{dimer:24} {method:6} q={charge:+} E = {:+.12} (ref {:+.12}) Δ = {delta:.2e} \
             [2-body {:+.3e}, ATM {:+.3e}]",
            result.energy_hartree, expected, result.two_body_hartree, result.atm_hartree
        );
    }
    println!("worst |ΔE| = {worst:.3e} Ha ({worst_row}); max |ATM/E| = {worst_atm_share:.4}");

    assert!(
        worst < 1.0e-9,
        "the D4 energy differs from the oracle by {worst:.3e} Ha at {worst_row}"
    );
    assert!(
        worst_atm_share > 0.01,
        "the ATM contribution looks unplugged (max |ATM/E| = {worst_atm_share:.3e})"
    );
}

#[test]
fn m06_2x_has_no_d4_parameters_and_says_why() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let molecules = geometries();
    let (symbols, positions) = molecules.get("water_dimer").expect("water_dimer");
    let refs: Vec<&str> = symbols.iter().map(String::as_str).collect();
    let error = d4::dispersion_energy(data, "m06-2x", &refs, positions, 0.0)
        .expect_err("M06-2X has no D4 parameters upstream");
    println!("m06-2x: {error}");
    assert!(
        error.contains("m06-2x") && error.contains("available"),
        "{error}"
    );
    assert!(error.contains("no damping parameter"), "{error}");
}
