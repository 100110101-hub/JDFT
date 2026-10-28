//! M3 / SC-05（第三条验收）：**`ri_error_hartree` 必须进报告**，`kind: ri-mp2` 真的算。
//!
//! 三条要求（`project-phase2.md` §6.5）：
//! 1. 与 SC-04（canonical）差 < 0.1 mHa（H2O/def2-SVP）—— 引擎里同一份参考态上直接比；
//! 2. `aux_basis_name` 落盘（知道用的哪套辅助基）；
//! 3. `ri_error_hartree` 落盘（近似必须留痕，K10 第 ③ 条）。
//!
//! 另外钉住两条"不许静默降级"：RI-MP2 缺 `aux_basis_name` → 退出码 1（输入错误，
//! K2 的 `validate` 就拦）；辅助基不存在 → `Data`（退出码 3）。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, Mp2Kind, Mp2Options,
    TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
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

fn water_input(options: TaskOptions) -> JobInput {
    let mut input = JobInput::new(
        water(),
        "def2-svp",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    );
    input.options.scf_energy_threshold_hartree = 1.0e-10;
    input.options.scf_density_threshold = 1.0e-9;
    input
}

#[test]
fn ri_mp2_lands_energy_aux_basis_and_fitting_error() {
    // PySCF 2.14.0：H2O/def2-SVP 的 E_corr(MP2) = −0.2036434634、
    // E_corr(RI-MP2, def2-svp-ri) = −0.2035582026（`scripts/pyscf_rimp2_reference.py`）。
    const E_CORR_RI_REFERENCE: f64 = -0.203_558_202_6;
    let options = TaskOptions::default()
        .with_mp2(Mp2Options::new(Mp2Kind::RiMp2).with_aux_basis("def2-svp-ri"));
    let result = backend()
        .run(&water_input(options), &NullReporter)
        .expect("RI-MP2 应已接线");

    let correlation = result
        .correlation
        .as_ref()
        .expect("SC-05：correlation 块必须进 JobResult");
    println!(
        "method = {}, aux = {:?}, E_corr = {:.10}, E_total = {:.10}, ri_error = {:?}",
        correlation.method.as_str(),
        correlation.aux_basis_name,
        correlation.e2_hartree,
        correlation.e_total_hartree,
        correlation.ri_error_hartree
    );
    assert_eq!(correlation.method, Mp2Kind::RiMp2);
    // ① 与 SC-04（canonical）差 < 0.1 mHa：canonical 的参考值来自 SC-04 的测试。
    let difference = (correlation.e2_hartree - (-0.203_643_463_4_f64)).abs();
    assert!(
        difference < 1.0e-4,
        "RI-MP2 {:.10} 与 canonical −0.2036434634 差 {difference:.3e} Ha（判据 1e-4）",
        correlation.e2_hartree
    );
    // ② 外部参考：与 PySCF 的 RI-MP2 差 < 1e-6。
    assert!(
        (correlation.e2_hartree - E_CORR_RI_REFERENCE).abs() < 1.0e-6,
        "RI-MP2 {:.10} 与 PySCF {E_CORR_RI_REFERENCE:.10} 不一致",
        correlation.e2_hartree
    );
    // ③ aux_basis_name 与 ri_error_hartree 都落盘。
    assert_eq!(correlation.aux_basis_name.as_deref(), Some("def2-svp-ri"));
    let ri_error = correlation
        .ri_error_hartree
        .expect("SC-05：ri_error_hartree 必须落盘");
    println!("ri_error = {ri_error:.3e} Ha");
    assert!(
        ri_error > 0.0 && ri_error < 1.0e-3,
        "ri_error {ri_error:.3e} 超出合理范围（def2-SVP-RI 实测 8.5e-5）"
    );
    // `e_total_hartree` 仍是 SCF + E_corr。
    let expected_total = result.energy_hartree + correlation.e2_hartree;
    assert!(
        (correlation.e_total_hartree - expected_total).abs() < 1.0e-12,
        "e_total = {} 应为 {expected_total}",
        correlation.e_total_hartree
    );
    // `result.json` 里要看得到这三个键。
    let json = serde_json::to_string(&result).expect("serialize JobResult");
    for key in [
        "\"correlation\"",
        "\"method\":\"ri-mp2\"",
        "\"aux_basis_name\"",
        "\"ri_error_hartree\"",
    ] {
        assert!(json.contains(key), "result.json 缺 {key}");
    }
}

#[test]
fn ri_mp2_without_aux_basis_is_an_input_error() {
    // K2 的 `Mp2Options::validate` 要求 RI 必须给 aux_basis_name；`JobInput::validate`
    // 会在进入引擎前拦下（退出码 1，不是 4 —— 这是输入写错，不是"没实现"）。
    let options = TaskOptions::default().with_mp2(Mp2Options::new(Mp2Kind::RiMp2));
    let input = water_input(options);
    let error = input.validate().expect_err("RI-MP2 缺辅助基必须报错");
    match error {
        EngineError::InvalidInput(message) => {
            assert!(message.contains("aux_basis_name"), "{message}");
        }
        other => panic!("期望 InvalidInput（退出码 1），得到 {other:?}"),
    }
}

#[test]
fn ri_mp2_with_a_missing_aux_basis_is_a_data_error() {
    let options = TaskOptions::default()
        .with_mp2(Mp2Options::new(Mp2Kind::RiMp2).with_aux_basis("def2-svp-ri-nope"));
    let error = backend()
        .run(&water_input(options), &NullReporter)
        .expect_err("不存在的辅助基必须报错");
    match error {
        EngineError::Data(message) => {
            assert!(message.contains("def2-svp-ri-nope"), "{message}");
        }
        other => panic!("期望 Data（退出码 3），得到 {other:?}"),
    }
}
