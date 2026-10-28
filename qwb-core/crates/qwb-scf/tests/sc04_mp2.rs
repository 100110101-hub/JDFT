//! M3 / SC-04：**canonical MP2** 的验收测试。
//!
//! # 验收口径（`project-phase2.md` §6，SC-04）
//!
//! - MP2 相关能与参考差 < 0.1 mHa；
//! - 基组单调性（DZ → TZ）方向正确（`|E_corr|` 变大，即相关能更负）；
//! - `correlation` 块进 `result.json`（在 `qwb-engine` 的测试里覆盖）。
//!
//! # 参考的选取
//!
//! 规格写的是 H2O/cc-pVDZ，但 `qwb-data` 里没有 cc-pVDZ（D-16 节流：只带
//! 用得到的基组）。这里改用 def2-SVP（DZ 级）与 def2-TZVP（TZ 级）做**同源**
//! 对照：参考值由 PySCF 2.14.0 的 `scf.RHF` + `mp.MP2` 在同一几何、同一基组、
//! 同样无冻结核的条件下算出，落在 `tests/reference/mp2_correlation.csv`。
//! 我们用「与 PySCF 差 < 1e-4 Ha」代替「与文献差 < 0.1 mHa」——两者是同一个
//! 数量级门槛，而 PySCF 对照能把「实现错」和「基组不同」彻底分开。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::mp2::mp2_correlation;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.117_790]),
            Atom::new("H", [0.0, 0.755_450, -0.471_160]),
            Atom::new("H", [0.0, -0.755_450, -0.471_160]),
        ],
        0,
        1,
    )
}

/// 与 `scripts/pyscf_mp2_reference.py` 对齐的参考表。
struct Reference {
    species: String,
    basis: String,
    e_scf: f64,
    e_corr: f64,
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("mp2_correlation.csv")
}

fn load_reference() -> Vec<Reference> {
    let text = std::fs::read_to_string(reference_path()).expect("mp2 reference csv");
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 5,
            "reference row {lineno} has {} fields",
            fields.len()
        );
        out.push(Reference {
            species: fields[0].to_string(),
            basis: fields[1].to_string(),
            e_scf: fields[2].parse().expect("e_scf"),
            e_corr: fields[3].parse().expect("e_corr"),
        });
    }
    assert!(!out.is_empty(), "empty MP2 reference table");
    out
}

fn lookup(reference: &[Reference], species: &str, basis: &str) -> (f64, f64) {
    reference
        .iter()
        .find(|row| row.species == species && row.basis == basis)
        .map(|row| (row.e_scf, row.e_corr))
        .unwrap_or_else(|| panic!("no MP2 reference for {species}/{basis}"))
}

fn n_occupied(molecule: &Molecule) -> usize {
    let electrons: i32 = molecule
        .atoms
        .iter()
        .map(|atom| {
            qwb_registry::elements::atomic_number(&atom.symbol)
                .unwrap_or_else(|| panic!("unknown element {}", atom.symbol)) as i32
        })
        .sum::<i32>()
        - molecule.charge;
    assert!(electrons > 0 && electrons % 2 == 0, "not closed shell");
    usize::try_from(electrons / 2).expect("n_occ")
}

/// 跑一遍闭壳层 HF（`terms` 为空即纯 HF），再算 MP2 相关能。
fn mp2_for(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
) -> (f64, f64, f64, usize, usize) {
    let basis_k1 = registry.get_basis(basis_name).expect(basis_name);
    let basis = qwb_basis::InternalBasis::build(basis_k1, molecule).expect("build basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, molecule).expect("grid");

    let mut options = RksOptions::default();
    options.max_iter = 500;
    options.energy_threshold = 1.0e-11;
    // 默认 `density_threshold = 1e-6` 太松：def2-TZVP 的水会停在密度残差 1e-6
    // 附近，总能量差 PySCF 2.7e-6 Ha。MP2 的分母对轨道能很敏感，这里一起收紧。
    options.density_threshold = 1.0e-10;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("HF");

    let n_occ = n_occupied(molecule);
    let orbital_energies: Vec<f64> = (0..result.mo_energies_hartree.len())
        .map(|i| result.mo_energies_hartree.get(i))
        .collect();
    let mp2 =
        mp2_correlation(&tensor, &result.coefficients, &orbital_energies, n_occ).expect("mp2");
    (
        result.energy_hartree,
        mp2.e2_hartree,
        mp2.dominant_pair_energy,
        mp2.n_occupied,
        mp2.n_virtual,
    )
}

#[test]
fn mp2_correlation_matches_pyscf_for_h2() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let reference = load_reference();
    let molecule = h2();

    for basis in ["sto-3g", "def2-svp", "def2-tzvp"] {
        let (e_scf, e_corr_ref) = lookup(&reference, "H2", basis);
        let (e_scf_qwb, e2, _dominant, n_occ, n_vir) = mp2_for(&registry, &molecule, basis);
        println!(
            "H2/{basis}: E_SCF = {e_scf_qwb:.10} (ref {e_scf:.10}), \
             E_corr = {e2:.10} (ref {e_corr_ref:.10}), n_occ = {n_occ}, n_vir = {n_vir}"
        );

        assert!(
            (e_scf_qwb - e_scf).abs() < 1.0e-6,
            "H2/{basis}: SCF {e_scf_qwb:.10} vs PySCF {e_scf:.10}"
        );
        let difference = (e2 - e_corr_ref).abs();
        assert!(
            difference < 1.0e-4,
            "H2/{basis}: MP2 相关能 {e2:.10} 与 PySCF {e_corr_ref:.10} 差 {difference:.2e} Ha"
        );
        assert!(e2 < 0.0, "H2/{basis}: 相关能应为负，实测 {e2:.6}");
    }
}

#[test]
fn mp2_correlation_matches_pyscf_for_water() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let reference = load_reference();
    let molecule = water();

    for basis in ["sto-3g", "def2-svp", "def2-tzvp"] {
        let (e_scf, e_corr_ref) = lookup(&reference, "H2O", basis);
        let (e_scf_qwb, e2, dominant, n_occ, n_vir) = mp2_for(&registry, &molecule, basis);
        println!(
            "H2O/{basis}: E_SCF = {e_scf_qwb:.10} (ref {e_scf:.10}), \
             E_corr = {e2:.10} (ref {e_corr_ref:.10}), dominant pair = {dominant:.6}, \
             n_occ = {n_occ}, n_vir = {n_vir}"
        );

        assert!(
            (e_scf_qwb - e_scf).abs() < 1.0e-6,
            "H2O/{basis}: SCF {e_scf_qwb:.10} vs PySCF {e_scf:.10}"
        );
        let difference = (e2 - e_corr_ref).abs();
        assert!(
            difference < 1.0e-4,
            "H2O/{basis}: MP2 相关能 {e2:.10} 与 PySCF {e_corr_ref:.10} 差 {difference:.2e} Ha"
        );
        assert!(e2 < 0.0, "H2O/{basis}: 相关能应为负，实测 {e2:.6}");
    }
}

/// 基组单调性：DZ → TZ 时相关能必须更负（`|E_corr|` 变大）。
///
/// 这是 MP2 的物理性质（相关能被基组系统性低估），也是识别「变换写错但在
/// 单个基组上凑巧接近」的关键旁证。
#[test]
fn mp2_correlation_is_monotone_along_the_basis_ladder() {
    let registry = Registry::open(data_root()).expect("qwb-data");

    for (name, molecule) in [("H2", h2()), ("H2O", water())] {
        let (_, e_corr_dz, _, _, _) = mp2_for(&registry, &molecule, "def2-svp");
        let (_, e_corr_tz, _, _, _) = mp2_for(&registry, &molecule, "def2-tzvp");
        println!("{name}: E_corr(def2-SVP) = {e_corr_dz:.10}, E_corr(def2-TZVP) = {e_corr_tz:.10}");
        assert!(
            e_corr_tz < e_corr_dz,
            "{name}: DZ → TZ 相关能应变负，实测 {e_corr_dz:.10} → {e_corr_tz:.10}"
        );
    }
}
