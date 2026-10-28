//! M3 / SC-05 的验收测试：RI-MP2。
//!
//! # 判据（`project-phase2.md` §6.5，SC-05）
//!
//! 1. 与 SC-04（canonical MP2）的差 < **0.1 mHa**（H2O/def2-SVP）；
//! 2. 墙钟优势 ≥ **3×**（否则记录实测并降级 —— 本轮实测见
//!    [`ri_path_is_faster_on_the_larger_system`]）；
//! 3. `ri_error_hartree` 进报告（引擎侧，`qwb-engine/tests/sc05_rimp2_engine.rs`）。
use std::path::PathBuf;
use std::time::Instant;

use qwb_engine_api::{Atom, Molecule};
use qwb_integrals::ri::{three_center, two_center_coulomb};
use qwb_registry::Registry;
use qwb_scf::mp2::mp2_correlation;
use qwb_scf::rimp2::ri_mp2_correlation;
use qwb_scf::rks::{RksOptions, RksResult, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("rimp2_correlation.csv")
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

fn benzene() -> Molecule {
    let ring = [
        [0.0, 1.3941],
        [1.2074, 0.69705],
        [1.2074, -0.69705],
        [0.0, -1.3941],
        [-1.2074, -0.69705],
        [-1.2074, 0.69705],
    ];
    let mut atoms = Vec::new();
    for [x, y] in ring {
        atoms.push(Atom::new("C", [x, y, 0.0]));
    }
    let hydrogen = [
        [0.0, 2.4815],
        [2.149, 1.24075],
        [2.149, -1.24075],
        [0.0, -2.4815],
        [-2.149, -1.24075],
        [-2.149, 1.24075],
    ];
    for [x, y] in hydrogen {
        atoms.push(Atom::new("H", [x, y, 0.0]));
    }
    Molecule::new(atoms, 0, 1)
}

/// PySCF 2.14.0 的参考（`scripts/pyscf_rimp2_reference.py`）。
struct Reference {
    e_corr_canonical: f64,
    e_corr_ri: f64,
}

fn load_reference(species: &str) -> Reference {
    let text = std::fs::read_to_string(reference_path()).expect("rimp2_correlation.csv");
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 6,
            "row {lineno} has {} fields",
            fields.len()
        );
        if fields[0] == species {
            return Reference {
                e_corr_canonical: fields[4].parse().expect("e_corr_canonical"),
                e_corr_ri: fields[5].parse().expect("e_corr_ri"),
            };
        }
    }
    panic!("no RI-MP2 reference for {species}");
}

fn run_scf(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
) -> (
    RksResult,
    qwb_integrals::TwoElectronIntegrals,
    qwb_basis::InternalBasis,
) {
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis(basis_name).expect(basis_name),
        molecule,
    )
    .expect("orbital basis");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), molecule)
        .expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.density_threshold = 1.0e-9;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options).expect("HF");
    (result, tensor, basis)
}

fn n_occupied(molecule: &Molecule) -> usize {
    let mut electrons = -molecule.charge;
    for atom in &molecule.atoms {
        let z = qwb_registry::elements::atomic_number(&atom.symbol)
            .unwrap_or_else(|| panic!("unknown element {}", atom.symbol));
        electrons += i32::try_from(z).expect("atomic number fits i32");
    }
    assert!(electrons > 0 && electrons % 2 == 0, "not closed shell");
    usize::try_from(electrons / 2).expect("n_occ")
}

#[test]
fn ri_mp2_matches_canonical_and_pyscf_for_water() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let reference = load_reference("H2O");
    let (scf, tensor, basis) = run_scf(&registry, &molecule, "def2-svp");
    let n_occ = n_occupied(&molecule);
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();

    let canonical =
        mp2_correlation(&tensor, &scf.coefficients, &energies, n_occ).expect("canonical");
    let aux = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp-ri").expect("def2-svp-ri"),
        &molecule,
    )
    .expect("aux basis");
    let three = three_center(&basis, &aux, &molecule).expect("3-center");
    let v = two_center_coulomb(&aux, &molecule).expect("(P|Q)");
    let ri = ri_mp2_correlation(&three, &v, &scf.coefficients, &energies, n_occ).expect("RI-MP2");

    println!(
        "H2O/def2-SVP：E2(canonical) = {:.10}，E2(RI) = {:.10}，差 {:.3e} Ha；\
         n_aux = {}，V 的秩 = {}",
        canonical.e2_hartree,
        ri.e2_hartree,
        (ri.e2_hartree - canonical.e2_hartree).abs(),
        ri.n_auxiliary,
        ri.v_rank
    );

    // 判据 1：与 canonical 差 < 0.1 mHa。
    let difference = (ri.e2_hartree - canonical.e2_hartree).abs();
    assert!(
        difference < 1.0e-4,
        "RI-MP2 与 canonical 差 {difference:.3e} Ha（判据 1e-4）"
    );
    // 外部参考：canonical 与 PySCF 一致（SC-04 已钉），RI 与 PySCF 的 RI-MP2 一致。
    println!(
        "PySCF：E2(canonical) = {:.10}，E2(RI) = {:.10}",
        reference.e_corr_canonical, reference.e_corr_ri
    );
    assert!(
        (canonical.e2_hartree - reference.e_corr_canonical).abs() < 1.0e-6,
        "canonical 与 PySCF 不一致：{:.10} vs {:.10}",
        canonical.e2_hartree,
        reference.e_corr_canonical
    );
    assert!(
        (ri.e2_hartree - reference.e_corr_ri).abs() < 1.0e-6,
        "RI-MP2 与 PySCF 的 RI-MP2 不一致：{:.10} vs {:.10}",
        ri.e2_hartree,
        reference.e_corr_ri
    );
    // 辅助基里不该有被截断的方向（def2-SVP-RI 对水是满秩的）。
    assert_eq!(ri.v_rank, ri.n_auxiliary, "V 有被截断的方向？");
}

/// # 判据 2 的实测口径
///
/// 判据是「≥ 3× 否则记录并降级」。所以这里**不**断言 3×，只打印实测值。
/// 这条测试在 `cargo test` 的默认档里被忽略（苯/def2-SVP 的 canonical 半变换在
/// debug 下要几分钟），用 `cargo test --release -- --ignored sc05` 跑。
#[test]
#[ignore = "慢档（debug 下几分钟）：苯/def2-SVP 的 canonical MP2 半变换；用 --release 跑"]
fn ri_path_is_faster_on_the_larger_system() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = benzene();
    let (scf, tensor, basis) = run_scf(&registry, &molecule, "def2-svp");
    let n_occ = n_occupied(&molecule);
    let energies: Vec<f64> = (0..scf.mo_energies_hartree.len())
        .map(|i| scf.mo_energies_hartree.get(i))
        .collect();

    let started = Instant::now();
    let canonical =
        mp2_correlation(&tensor, &scf.coefficients, &energies, n_occ).expect("canonical");
    let canonical_seconds = started.elapsed().as_secs_f64();

    let aux = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp-ri").expect("def2-svp-ri"),
        &molecule,
    )
    .expect("aux basis");
    let started = Instant::now();
    let three = three_center(&basis, &aux, &molecule).expect("3-center");
    let v = two_center_coulomb(&aux, &molecule).expect("(P|Q)");
    let ri = ri_mp2_correlation(&three, &v, &scf.coefficients, &energies, n_occ).expect("RI-MP2");
    let ri_seconds = started.elapsed().as_secs_f64();

    println!(
        "C6H6/def2-SVP：n_orb = {}，n_aux = {}；canonical {canonical_seconds:.2} s \
         （E2 = {:.8}），RI {ri_seconds:.2} s（E2 = {:.8}），比值 {:.2}×",
        tensor.n_functions,
        ri.n_auxiliary,
        canonical.e2_hartree,
        ri.e2_hartree,
        canonical_seconds / ri_seconds
    );
    // 精度：与 PySCF 的两条路径各对到 1e-6 Ha。**不**在这里断言 0.1 mHa ——
    // 那是 SC-05 对 H2O/def2-SVP 的判据（0.085 mHa），苯上同一个辅助基的拟合误差
    // 是 0.47 mHa（PySCF 也是这个数），属于辅助基性质而非实现问题。
    let difference = (ri.e2_hartree - canonical.e2_hartree).abs();
    println!("苯上 |E2(RI) − E2(canonical)| = {difference:.3e} Ha（PySCF 同值）");
    let reference = load_reference("C6H6");
    assert!(
        (canonical.e2_hartree - reference.e_corr_canonical).abs() < 1.0e-6,
        "苯 canonical 与 PySCF 不一致：{:.10} vs {:.10}",
        canonical.e2_hartree,
        reference.e_corr_canonical
    );
    assert!(
        (ri.e2_hartree - reference.e_corr_ri).abs() < 1.0e-6,
        "苯 RI-MP2 与 PySCF 不一致：{:.10} vs {:.10}",
        ri.e2_hartree,
        reference.e_corr_ri
    );
}
