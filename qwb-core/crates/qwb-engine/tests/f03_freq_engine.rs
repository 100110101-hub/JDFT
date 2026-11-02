//! `jobtype: freq` 的引擎级验收：数值 Hessian → 简正模 → 谐振频率 + 热化学。
//!
//! 三条外部/独立判据：
//!
//! 1. **频率对 PySCF 的解析 Hessian**（`tests/reference/hessian_water.csv`，生成脚本
//!    `scripts/pyscf_hessian_reference.py --check`）：同一个几何、同一基组，两条独立算法
//!    （数值 ∂²E/∂x² vs 解析导数）差应在 1 cm⁻¹ 量级；
//! 2. **零点能自洽**：`ZPE = ½Σν`，与块里报的 `zero_point_cm` 一致（< 1e-6 cm⁻¹）；
//! 3. **标准熵对实验**：H2O 的 298 K 标准熵是 188.83 J/(mol·K)（NIST/CCCBDB 族）。
//!    STO-3G + 谐振近似会偏高几个 J/(mol·K)，所以判据写成"落在这个窗口里"，并把
//!    **对称数 σ 的效果**单独钉住（σ=2 与 σ=1 的差必须正好是 `R ln 2`）—— 这一条是本轮
//!    修掉的真缺陷：原来的热化学只对同核双原子判 σ，水/氨/甲烷都按 σ=1 算。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/hessian_water.csv")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

/// H2O/STO-3G/HF 的**优化几何**（与 `hessian_water.csv` 的生成脚本逐字一致）。
fn optimized_water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.148_940_223_535]),
            Atom::new("H", [0.0, 0.758_058_893_909, -0.486_735_104_178]),
            Atom::new("H", [0.0, -0.758_058_893_909, -0.486_735_104_178]),
        ],
        0,
        1,
    )
}

/// 拉长一根 O–H 键到 1.45 Å：这个几何在 STO-3G 上有一个**虚频**（伸缩模）。
fn distorted_water() -> Molecule {
    let mut molecule = optimized_water();
    molecule.atoms[1].position_angstrom = [0.0, 0.758_058_893_909, 0.0];
    molecule
}

fn input(molecule: Molecule, jobtype: JobType, symmetry_number: u32) -> JobInput {
    JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        jobtype,
        TaskOptions::default().with_symmetry_number(symmetry_number),
    )
}

fn pyscf_wavenumbers() -> Vec<f64> {
    let text = std::fs::read_to_string(reference_path()).expect("hessian_water.csv");
    let mut values: Vec<(usize, f64)> = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 || fields[3] != "wavenumber" {
            continue;
        }
        let index: usize = fields[4].parse().expect("index");
        let value: f64 = fields[5].parse().expect("value");
        assert_eq!(fields[0], "water");
        assert_eq!(fields[1], "hf");
        assert_eq!(fields[2], "sto-3g");
        values.push((index, value));
    }
    values.sort_by_key(|(index, _)| *index);
    values.into_iter().map(|(_, value)| value).collect()
}

/// 1 Hartree 对应的波数（CODATA 2018，cm⁻¹）。
const CM_PER_HARTREE: f64 = 219_474.631_370_5;
const GAS_CONSTANT: f64 = 8.314_462_618;

#[test]
fn frequencies_match_the_analytic_pyscf_hessian() {
    let result = backend()
        .run(&input(optimized_water(), JobType::Freq, 1), &NullReporter)
        .expect("freq 应跑通");
    let block = result
        .vibrational_frequencies
        .expect("freq 必须带 vibrational_frequencies");
    assert_eq!(block.method, "numerical-hessian");
    assert_eq!(block.vibrational_modes, 3, "H2O 有 3N−6 = 3 个振动模");
    assert_eq!(block.degrees_of_freedom, 9);
    assert_eq!(block.modes.len(), 3);
    assert_eq!(block.imaginary_count(), 0, "优化几何上没有虚频");

    let reference = pyscf_wavenumbers();
    assert_eq!(reference.len(), 3, "CSV 里应有 3 个频率");
    let mut worst = 0.0_f64;
    for (mode, expected) in block.modes.iter().zip(&reference) {
        let deviation = (mode.wavenumber_cm - expected).abs();
        worst = worst.max(deviation);
        assert!(
            deviation < 1.0,
            "模 {:.4} cm⁻¹ 与 PySCF 解析 Hessian 的 {expected:.4} 差 {deviation:.4}",
            mode.wavenumber_cm
        );
        // 三支都是红外活性的（水有 C2v 对称性，弯曲 + 两个伸缩都允许）。
        let intensity = mode.ir_intensity_km_per_mol.expect("应报 IR 强度");
        assert!(intensity >= 0.0 && intensity.is_finite());
    }
    println!(
        "H2O/STO-3G 频率：{:?} cm⁻¹（对 PySCF 最差 {worst:.4} cm⁻¹）",
        block
            .modes
            .iter()
            .map(|mode| (mode.wavenumber_cm * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );

    // ZPE = ½Σν（只含振动模）；块里报的两个数必须自洽。
    let half_sum: f64 = block
        .modes
        .iter()
        .map(|mode| mode.wavenumber_cm / 2.0)
        .sum();
    assert!(
        (block.zero_point_cm - half_sum).abs() < 1.0e-6,
        "zero_point_cm = {:.6} 与 ½Σν = {half_sum:.6} 不一致",
        block.zero_point_cm
    );
    block.validate().expect("块自身的自洽性");
}

#[test]
fn thermochemistry_reports_the_symmetry_number_and_it_shifts_the_entropy() {
    // σ = 2：水的 C2 轴。这是本轮修掉的那半个缺陷 —— 原来只对同核双原子判 σ。
    let with_symmetry = backend()
        .run(&input(optimized_water(), JobType::Freq, 2), &NullReporter)
        .expect("freq 应跑通");
    let without = backend()
        .run(&input(optimized_water(), JobType::Freq, 1), &NullReporter)
        .expect("freq 应跑通");
    let thermo = with_symmetry
        .thermochemistry
        .expect("freq 必须带 thermochemistry");
    let thermo_no_symmetry = without.thermochemistry.expect("thermochemistry");
    assert_eq!(thermo.symmetry_number, 2);
    assert_eq!(thermo_no_symmetry.symmetry_number, 1);
    assert!((thermo.temperature_k - 298.15).abs() < 1e-12);
    assert!((thermo.pressure_pa - 101_325.0).abs() < 1e-9);

    // 转动熵对 σ 是 R ln σ 的差：σ=2 与 σ=1 的 S 必须正好差 R ln 2。
    let expected_shift = GAS_CONSTANT * 2.0_f64.ln();
    let shift = thermo_no_symmetry.entropy_j_per_mol_k - thermo.entropy_j_per_mol_k;
    assert!(
        (shift - expected_shift).abs() < 1e-9,
        "R ln 2 = {expected_shift:.6} J/(mol·K)，实测 {shift:.6}"
    );
    assert!(thermo.entropy_j_per_mol_k > 0.0);
    // 水的 G 在 298 K 上**高于** E_elec（ZPE + 热修正 > T·S），所以这里不钉方向；
    // 要钉的是 `validate` 里的 G = E + H_corr − T·S（两处数必须自洽）。
    thermo
        .validate(with_symmetry.energy_hartree)
        .expect("热化学块自身一致（G = E + H_corr − T·S）");

    // 标准熵对实验：H2O 的 298 K 标准熵 188.83 J/(mol·K)（NIST/CCCBDB 族）。
    // STO-3G + 谐振近似的系统性偏差是几个 J/(mol·K) 量级，所以判据是窗口而不是等号。
    println!(
        "H2O/STO-3G 热化学：S = {:.2} J/(mol·K)（实验 188.83），H_corr = {:.6} Ha，G = {:.6} Ha",
        thermo.entropy_j_per_mol_k, thermo.enthalpy_hartree, thermo.gibbs_hartree
    );
    assert!(
        (170.0..=205.0).contains(&thermo.entropy_j_per_mol_k),
        "S = {:.2} J/(mol·K) 不在 170…205 的窗口里",
        thermo.entropy_j_per_mol_k
    );

    // ZPE 与频率块一致（同一个 Hessian，不许两处各算一套）。
    let frequencies = with_symmetry.vibrational_frequencies.expect("frequencies");
    // ZPE 块里报的是 cm⁻¹（½Σν），热化学块里报的是 Hartree：两者必须互为换算。
    let zero_point_from_modes = frequencies.zero_point_cm / CM_PER_HARTREE;
    assert!(
        (zero_point_from_modes - thermo.zero_point_hartree).abs() < 1e-9,
        "ZPE {:.9} Ha（cm⁻¹ 块换算）与热化学块的 {:.9} Ha 不一致",
        zero_point_from_modes,
        thermo.zero_point_hartree
    );
}

#[test]
fn imaginary_modes_are_refused_instead_of_misclassified() {
    // 拉长的几何有一个虚频。3N−6 的振动集合是按波数取的，虚频会排在平移/转动模之下，
    // 所以这里**拒绝**而不是悄悄把它当成非振动模（RP-01 要先做子空间投影）。
    let error = backend()
        .run(&input(distorted_water(), JobType::Freq, 2), &NullReporter)
        .expect_err("有虚频的几何必须拒绝");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("imaginary"), "{message}");
            assert!(message.contains("RP-01"), "{message}");
        }
        other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
    }
}

#[test]
fn open_shell_frequencies_are_refused_with_a_reason() {
    let mut job = input(optimized_water(), JobType::Freq, 2);
    job.options.scf_type = Some(qwb_engine_api::ScfType::Uks);
    job.molecule.multiplicity = 2;
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("开壳层 freq 未验收");
    match error {
        EngineError::Unsupported(message) => {
            // 拒绝可能来自 freq 的闭壳层前置，也可能来自 uks 的 "sp only" 守卫 —— 两者都
            // 说明开壳层频率没有验收。
            assert!(
                message.contains("closed-shell") || message.contains("only"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn freq_on_a_solvated_system_is_refused() {
    let mut job = input(optimized_water(), JobType::Freq, 2);
    job.solvent = "water".into();
    job.options.solvent_model = Some(qwb_engine_api::SolventModel::Cosmo);
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("溶剂 + freq 未实现（SV-02）");
    match error {
        EngineError::Unsupported(message) => assert!(message.contains("SV-02"), "{message}"),
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}
