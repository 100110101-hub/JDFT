//! SC-01b / M1b：**UKS**（自旋极化 Kohn–Sham）的验收。
//!
//! 判据（project-phase2.md §7 M1b）：
//!
//! 1. O2/OH 的 **UKS** 与 PySCF 同值（< 1e-6）：O2 用 `svwn5` 与 `pbe`，OH 用 `pbe`
//!    （OH + VWN-RPA 在 PySCF 里三种策略都不收敛，CSV 里如实记的是 `nan`，这里跳过）；
//! 2. 闭壳层体系 `uks` ≡ `rks`（< 1e-8）—— 退化一致性；
//! 3. ⟨S²⟩ 与 PySCF 差 < 1e-4，且**必须报出来**（`spin_contamination`）；
//! 4. BLYP 的 UKS 必须**显式报错**（B88/LYP 没有自旋实现），不能静默用闭壳层公式。

use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data};
use qwb_scf::uks::{UksOptions, uks_from_data};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn o2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            // `Atom::new` 的坐标单位是 **Å**（K2 的边界单位）；不要再除 Bohr 换算因子，
            // 否则距离被放大 1.89 倍（这条曾经让 O2/OH 的 UKS 全部对不上 PySCF）。
            Atom::new("O", [0.0, 0.0, 1.208]),
        ],
        0,
        3,
    )
}

fn oh() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 0.980]),
        ],
        0,
        2,
    )
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

/// PySCF 参考（`tests/reference/uks_open_shell.csv`）。
struct ReferenceRow {
    species: String,
    functional: String,
    energy: f64,
    s_squared: f64,
    n_alpha: usize,
    n_beta: usize,
    /// 与 PySCF 的容差（CSV 的 `tolerance_hartree`）：残差主要来自网格。
    tolerance: f64,
}

fn reference_rows() -> Vec<ReferenceRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/uks_open_shell.csv");
    let text = std::fs::read_to_string(path).expect("read uks_open_shell.csv");
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        rows.push(ReferenceRow {
            species: fields[0].to_string(),
            functional: fields[1].to_string(),
            energy: fields[4].parse().expect("energy"),
            s_squared: fields[5].parse().expect("s_squared"),
            n_alpha: fields[6].parse::<f64>().expect("n_alpha") as usize,
            n_beta: fields[7].parse::<f64>().expect("n_beta") as usize,
            tolerance: fields[8].parse().expect("tolerance_hartree"),
        });
    }
    rows
}

fn uks_options() -> UksOptions {
    let mut options = UksOptions::default();
    options.max_iter = 400;
    // 比默认更紧：闭壳层退化判据是 1e-8，两侧各自的收敛误差不能吃掉它。
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-8;
    options
}

/// 跑一行参考数据（逐行断言）。**拆成三个 `#[test]` 只是为了并行**：
/// 这里每个体系的 UKS 各自要几十秒到几分钟，串在一个测试里会把门禁尾巴拖到
/// 10 分钟以上；断言集合与容差一字未改（`choices` 八十三）。
fn check_reference_row(row: &ReferenceRow) {
    if row.energy.is_nan() {
        println!(
            "{}/{}: PySCF 参考是 nan（未收敛），跳过",
            row.species, row.functional
        );
        return;
    }
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = match row.species.as_str() {
        "O2" => o2(),
        "OH" => oh(),
        other => panic!("未知体系 {other}"),
    };
    let result = uks_from_data(
        &registry,
        &molecule,
        &row.basis_placeholder(),
        &row.functional,
        "grid-fine",
        uks_options(),
    )
    .unwrap_or_else(|err| panic!("{}/{}: {err}", row.species, row.functional));
    println!(
        "{}/{}: E = {:.10} (PySCF {:.10}, 差 {:+.2e})  <S^2> = {:.6} (PySCF {:.6})  {} iters",
        row.species,
        row.functional,
        result.energy_hartree,
        row.energy,
        result.energy_hartree - row.energy,
        result.s_squared,
        row.s_squared,
        result.iterations
    );
    assert!(
        (result.energy_hartree - row.energy).abs() < row.tolerance,
        "{}/{}: E = {:.10} vs PySCF {:.10}（容差 {:.1e}）",
        row.species,
        row.functional,
        result.energy_hartree,
        row.energy,
        row.tolerance
    );
    assert!(
        (result.s_squared - row.s_squared).abs() < 1.0e-4,
        "{}/{}: <S^2> = {:.6} vs PySCF {:.6}",
        row.species,
        row.functional,
        result.s_squared,
        row.s_squared
    );
    assert_eq!(result.n_alpha, row.n_alpha, "{}: n_alpha", row.species);
    assert_eq!(result.n_beta, row.n_beta, "{}: n_beta", row.species);
    // 污染必须能被读者看到（不是只写在日志里）
    assert!(
        result.spin_contamination.is_finite(),
        "{}: spin_contamination 必须是有限数",
        row.species
    );
}

/// 拆测试不许顺手丢掉覆盖：表是 4 行（O2/svwn5、O2/pbe、OH/svwn5=[nan]、OH/pbe），
/// 其中至少 3 行有可用参考值 —— 这正是拆之前那条 `checked >= 3` 断言。
#[test]
fn uks_reference_rows_cover_three_systems() {
    let rows = reference_rows();
    let names: Vec<String> = rows
        .iter()
        .map(|row| format!("{}/{}", row.species, row.functional))
        .collect();
    println!("参考行: {names:?}");
    let usable = rows.iter().filter(|row| !row.energy.is_nan()).count();
    assert_eq!(rows.len(), 4, "参考表行数变了（现在 {names:?}）");
    assert!(
        usable >= 3,
        "至少要有 3 行可用参考值（现在只有 {usable} 行）"
    );
}

fn run_reference_row(index: usize) {
    let rows = reference_rows();
    assert!(
        index < rows.len(),
        "参考表只有 {} 行，取不到第 {index} 行",
        rows.len()
    );
    check_reference_row(&rows[index]);
}

#[test]
fn uks_matches_pyscf_for_o2_and_oh_row0() {
    run_reference_row(0);
}

#[test]
fn uks_matches_pyscf_for_o2_and_oh_row1() {
    run_reference_row(1);
}

#[test]
fn uks_matches_pyscf_for_o2_and_oh_row2() {
    // OH/svwn5：PySCF 参考是 nan，这里只确认「跳过」这条路仍然成立。
    run_reference_row(2);
}

#[test]
fn uks_matches_pyscf_for_o2_and_oh_row3() {
    run_reference_row(3);
}

#[test]
fn closed_shell_uks_matches_rks() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let mut rks_options = RksOptions::default();
    rks_options.max_iter = 400;
    rks_options.energy_threshold = 1.0e-10;
    rks_options.density_threshold = 1.0e-8;
    let rks = rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "pbe",
        "grid-normal",
        rks_options.clone(),
    )
    .expect("rks");
    let uks = uks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "pbe",
        "grid-normal",
        uks_options(),
    )
    .expect("uks");
    println!(
        "H2/PBE: RKS = {:.10}  UKS = {:.10}  (差 {:+.2e}), UKS <S^2> = {:.3e}",
        rks.energy_hartree,
        uks.energy_hartree,
        uks.energy_hartree - rks.energy_hartree,
        uks.s_squared
    );
    assert!(
        (uks.energy_hartree - rks.energy_hartree).abs() < 1.0e-8,
        "闭壳层 UKS 应当与 RKS 一致：{} vs {}",
        uks.energy_hartree,
        rks.energy_hartree
    );
    assert!(
        uks.s_squared.abs() < 1.0e-8,
        "闭壳层 <S^2> 必须是 0，实测 {}",
        uks.s_squared
    );
}

#[test]
fn unsupported_functionals_are_rejected_not_silently_wrong() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = o2();
    // BLYP 需要 B88 + LYP，两者都没有自旋实现。
    let error = uks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "b88",
        "grid-normal",
        uks_options(),
    )
    .expect_err("B88 单独也不能跑 UKS");
    let message = error.to_string();
    println!("B88 的 UKS 被拒绝：{message}");
    assert!(
        message.contains("spin") || message.contains("Unsupported"),
        "错误信息要点明自旋实现缺失：{message}"
    );
}

impl ReferenceRow {
    fn basis_placeholder(&self) -> String {
        // CSV 的 basis 列都是 sto-3g（生成脚本里写死的），这里保持单一来源。
        "sto-3g".to_string()
    }
}
