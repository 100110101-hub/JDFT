//! RS-06 的引擎接线：`jobtype: freq` + `options.raman` → `result.json` 的 `raman` 块。
//!
//! 这个文件的重点不是物理（那个在 `tests/rs06_raman.rs` 里对 PySCF 逐模对照），而是
//! **三条接线纪律**：
//!
//! 1. `options.raman` 只在 `freq` 上有意义 —— 别的 jobtype 必须**显式拒绝**，不许静默忽略；
//! 2. 块里的数与模块级 API（[`qwb_engine::raman`]）**逐位相同**（不许两处各算一套）；
//! 3. 块里的元数据（激发线波数、步长、单位、活性来源）必须与算它时用的参数一致 ——
//!    「相对强度」离开激发线就没有意义，这条是它可核对的前提。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine::opt::OptOptions;
use qwb_engine::raman::{polarizability_derivatives, raman_activities};
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, RamanOptions,
    ScfType, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

/// H2O/STO-3G 的 HF 优化几何（与 `rs06_raman.rs`、PySCF 生成脚本逐字相同）。
fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.150_168_163_058]),
            Atom::new("H", [0.0, 0.758_080_671_347, -0.485_634_081_529]),
            Atom::new("H", [0.0, -0.758_080_671_347, -0.485_634_081_529]),
        ],
        0,
        1,
    )
}

fn input(jobtype: JobType, raman: Option<RamanOptions>) -> JobInput {
    let mut options = TaskOptions::default().with_symmetry_number(2);
    if let Some(raman) = raman {
        options = options.with_raman(raman);
    }
    JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        jobtype,
        options,
    )
}

#[test]
fn freq_with_raman_reports_the_block_and_matches_the_module_api() {
    let result = backend()
        .run(
            &input(JobType::Freq, Some(RamanOptions::new(532.0, 298.15))),
            &NullReporter,
        )
        .expect("freq + raman 应跑通");
    let frequencies = result
        .vibrational_frequencies
        .expect("freq 必须带 vibrational_frequencies");
    let raman = frequencies.raman.clone().expect("options.raman 必须落盘");
    assert_eq!(raman.method, "finite-difference-polarizability");
    assert_eq!(raman.activity_unit, "angstrom^4/amu");
    assert!(
        (raman.delta_bohr - 1.0e-2).abs() < 1e-12,
        "步长应与 Hessian 同一个约定"
    );
    assert!(
        (raman.excitation_wavenumber_cm - 1.0e7 / 532.0).abs() < 1e-9,
        "激发波数 = 1e7/532 = {:.6}，实得 {:.6}",
        1.0e7 / 532.0,
        raman.excitation_wavenumber_cm
    );
    assert!((raman.temperature_k - 298.15).abs() < 1e-12);
    assert_eq!(raman.activities.len(), 3);
    assert_eq!(raman.intensities.len(), 3);
    let strongest = raman.intensities.iter().cloned().fold(0.0_f64, f64::max);
    assert!((strongest - 100.0).abs() < 1e-12, "相对强度归一化到 100");
    assert!(raman.activities.iter().all(|value| *value > 0.0));
    frequencies.validate().expect("块的自洽性");

    // 与模块级 API **逐位相同**（引擎没有第二条公式）。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let options = OptOptions::new("sto-3g", "", "grid-coarse");
    let hessian =
        qwb_engine::hessian::numerical_hessian(&registry, &water(), &options).expect("hessian");
    let derivatives = polarizability_derivatives(&registry, &water(), "sto-3g", "", "grid-coarse")
        .expect("dα/dR");
    let direct = raman_activities(&hessian, &derivatives);
    let start = hessian
        .modes
        .len()
        .saturating_sub(hessian.vibrational_modes);
    for (index, activity) in raman.activities.iter().enumerate() {
        assert_eq!(
            *activity,
            direct[start + index].activity_a4_per_amu,
            "结果的第 {index} 个活性与模块级 API 不一致"
        );
    }
    // 频率与只跑 IR 的那条路（没有 raman）必须一致 —— raman 是加性块。
    let plain = backend()
        .run(&input(JobType::Freq, None), &NullReporter)
        .expect("freq 应跑通");
    let plain_frequencies = plain.vibrational_frequencies.expect("frequencies");
    assert!(
        plain_frequencies.raman.is_none(),
        "没给 options.raman 就不该算"
    );
    for (with, without) in frequencies.modes.iter().zip(&plain_frequencies.modes) {
        assert_eq!(with.wavenumber_cm, without.wavenumber_cm);
        assert_eq!(
            with.ir_intensity_km_per_mol,
            without.ir_intensity_km_per_mol
        );
    }
    println!("H2O/STO-3G 的 Raman 活性（Å⁴/amu）：{:?}", raman.activities);
    println!("相对强度（532 nm）：{:?}", raman.intensities);
}

#[test]
fn raman_outside_a_freq_job_is_refused_with_a_reason() {
    // 别的 jobtype 走不到频率那一段，放过就是「静默忽略了一个用户明确要求的性质」。
    for jobtype in [JobType::Sp, JobType::Opt] {
        let error = backend()
            .run(
                &input(jobtype, Some(RamanOptions::default())),
                &NullReporter,
            )
            .expect_err("非 freq 的 raman 必须拒绝");
        match error {
            EngineError::Unsupported(message) => {
                assert!(message.contains("freq"), "{message}");
                assert!(message.contains("jobtype"), "{message}");
            }
            other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
        }
    }
}

#[test]
fn raman_with_an_open_shell_reference_is_refused() {
    let mut job = input(JobType::Freq, Some(RamanOptions::default()));
    job.options.scf_type = Some(ScfType::Uks);
    job.molecule.multiplicity = 2;
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("开壳层 Raman 未实现");
    match error {
        EngineError::Unsupported(message) => {
            assert!(
                message.contains("closed-shell") || message.contains("open-shell"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn a_non_positive_excitation_wavelength_is_invalid_input() {
    let job = input(JobType::Freq, Some(RamanOptions::new(0.0, 298.15)));
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("波长 0 必须拒绝");
    match error {
        EngineError::InvalidInput(message) => {
            assert!(message.contains("excitation_wavelength_nm"), "{message}")
        }
        other => panic!("期望 InvalidInput，得到 {other:?}"),
    }
}
