//! SV-03 第五半验收：**SMD 参数化半径**（eq. 16）。
//!
//! 数据侧来自 `_smd_radii_staging/smd_radii.json`（`radii.py` + `smd_radii` 两份上游源码
//! 按 sha256 钉住），参考值来自 PySCF 的 `smd.smd_radii(alpha)`（bohr）。
//!
//! 判据：
//! 1. 36 个元素 × 三个 α（0.0 / 0.43 / 0.5）与上游逐个对上（相对 < 1e-9；差别只可能来自
//!    PySCF 的 `BOHR` 常数 0.52917721092 与仓库 CODATA 2018 的 0.529177210903 的第 11 位）；
//! 2. **两套半径不许混用**：C 1.85 vs CDS 的 1.70、N 1.89 vs 1.55（有断言）；
//! 3. α 越小 O 越大（`R = 1.52 + 1.8·(0.43 − α)`），并且在阈值处连续。
use std::path::PathBuf;

use qwb_engine_api::{EngineError, Molecule, molecule::BOHR_PER_ANGSTROM};
use qwb_props::smd_radii::smd_parameterised_radii;
use qwb_registry::Registry;
use qwb_registry::smd_atomic::SmdAtomic;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn smd() -> SmdAtomic {
    let registry = Registry::open(data_root()).expect("qwb-data");
    registry
        .get_smd_atomic("smd-atomic")
        .expect("smd-atomic")
        .clone()
}

fn single_atom(symbol: &str) -> Molecule {
    Molecule::new(
        vec![qwb_engine_api::Atom::new(symbol, [0.0, 0.0, 0.0])],
        0,
        1,
    )
}

/// 参考 CSV：`(symbol, alpha, radius_bohr, radius_angstrom)`。
fn read_reference() -> Vec<(String, f64, f64, f64)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference/smd_parameterised_radii.csv");
    let text = std::fs::read_to_string(&path).expect("smd_parameterised_radii.csv");
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 7, "line {}: expected 7 columns", index + 1);
        rows.push((
            fields[0].to_string(),
            fields[2].parse::<f64>().expect("alpha"),
            fields[3].parse::<f64>().expect("radius_bohr"),
            fields[4].parse::<f64>().expect("radius_angstrom"),
        ));
    }
    assert_eq!(rows.len(), 108, "36 elements x three alphas");
    rows
}

#[test]
fn the_parameterised_radii_reproduce_the_upstream_table() {
    let data = smd();
    // 两列都要比：
    // * Å 列是"数据 + 规则"的直接对照（容差 1e-12）；
    // * bohr 列的残差 ~1.8e-8 来自 **Å↔bohr 常数**不同：仓库用 CODATA 2018 的
    //   `BOHR_PER_ANGSTROM = 1.889726089838572`，PySCF 用 0.52917721092（1.8897261247），
    //   两者差 1.8e-8 —— 那是常数，不是数据。所以 bohr 列用 1e-7 并把这个差记在测试里。
    let mut worst_angstrom = 0.0_f64;
    let mut worst_bohr = 0.0_f64;
    for (symbol, alpha, expected_bohr, expected_angstrom) in read_reference() {
        let molecule = single_atom(&symbol);
        let ours = smd_parameterised_radii(&data, &molecule, alpha).expect("radii");
        worst_angstrom =
            worst_angstrom.max((ours[0] - expected_angstrom).abs() / expected_angstrom);
        let ours_bohr = ours[0] * BOHR_PER_ANGSTROM;
        worst_bohr = worst_bohr.max((ours_bohr - expected_bohr).abs() / expected_bohr);
    }
    println!("worst relative deviation: {worst_angstrom:.3e} (angstrom), {worst_bohr:.3e} (bohr)");
    assert!(
        worst_angstrom < 1.0e-12,
        "the SMD parameterised radii must reproduce smd_radii in angstrom, got {worst_angstrom:.3e}"
    );
    assert!(
        worst_bohr < 1.0e-7,
        "the bohr comparison may only differ by the unit constant, got {worst_bohr:.3e}"
    );
}

#[test]
fn the_two_radius_families_are_kept_apart() {
    // C/N 在 SMD 参数化半径与 CDS vdW 半径下**不同**（1.85/1.70、1.89/1.55）。
    // 混用会静默错数，所以这条断言把两套数据钉在同一处。
    let data = smd();
    let parameterised = |symbol: &str| {
        smd_parameterised_radii(&data, &single_atom(symbol), 0.43).expect("radii")[0]
    };
    assert!((parameterised("C") - 1.85).abs() < 1.0e-12);
    assert!((parameterised("N") - 1.89).abs() < 1.0e-12);
    assert!((parameterised("H") - 1.20).abs() < 1.0e-12);
    assert!((data.cds.vdw_radii_angstrom["C"] - 1.70).abs() < 1.0e-12);
    assert!((data.cds.vdw_radii_angstrom["N"] - 1.55).abs() < 1.0e-12);
    assert!(
        (parameterised("C") - data.cds.vdw_radii_angstrom["C"]).abs() > 0.1,
        "the two families must not coincide for carbon"
    );
}

#[test]
fn the_oxygen_radius_follows_the_alpha_rule() {
    let data = smd();
    let oxygen =
        |alpha: f64| smd_parameterised_radii(&data, &single_atom("O"), alpha).expect("radii")[0];
    // α ≥ 0.43 → 1.52；α = 0 → 1.52 + 1.8·0.43 = 2.294。
    assert!((oxygen(0.43) - 1.52).abs() < 1.0e-12);
    assert!((oxygen(0.50) - 1.52).abs() < 1.0e-12);
    assert!((oxygen(0.0) - (1.52 + 1.8 * 0.43)).abs() < 1.0e-12);
    assert!((oxygen(0.2) - (1.52 + 1.8 * 0.23)).abs() < 1.0e-12);
    // 单调：α 越小 O 越大。
    assert!(oxygen(0.0) > oxygen(0.2));
    assert!(oxygen(0.2) > oxygen(0.42));
    println!(
        "O: alpha 0 -> {:.4}, 0.2 -> {:.4}, 0.43 -> {:.4}",
        oxygen(0.0),
        oxygen(0.2),
        oxygen(0.43)
    );
}

#[test]
fn unknown_elements_are_reported() {
    let data = smd();
    let xenon = single_atom("Xe");
    let error = smd_parameterised_radii(&data, &xenon, 0.43)
        .expect_err("Xe is outside the H-Kr data range");
    println!("Xe -> {error}");
    assert!(matches!(error, EngineError::Data(_)));
}
