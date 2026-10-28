//! M3 / SC-04（第三条验收）：**`correlation` 块必须真的进 `JobResult`（也就进 `result.json`）**。
//!
//! 前面的 `qwb-scf` 测试（`tests/sc04_mp2.rs`）证明 `mp2_correlation` 的**数值**与
//! PySCF 一致；这条测试证明**接线**对：引擎在同一个基组/同一份 2e 张量上跑完 SCF，
//! 再把相关能挂进 K2 的 `CorrelationResult`，且 `e_total_hartree = E_SCF + E_corr`。
//!
//! 另外钉住三件"不能静默降级"的事：
//! - `kind: mp2` 会真的算，不再以"未接线"拒绝；
//! - `kind: ri-mp2` 仍以 `Unsupported` + WP 编号拒绝（SC-05 没做）；
//! - MP2 只接 `jobtype: sp`（MP2 梯度没实现），`opt` 显式拒绝。
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

/// 与 `qwb-scf/tests/reference/mp2_correlation.csv` 同一个几何。
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
    // PySCF 对照用的是 1e-12 级收敛；引擎默认阈值不够紧，这里按 K2 的字段收紧。
    input.options.scf_energy_threshold_hartree = 1.0e-10;
    input.options.scf_density_threshold = 1.0e-9;
    input
}

#[test]
fn mp2_correlation_is_attached_to_the_job_result() {
    // PySCF 2.14.0：H2O/def2-SVP 的 RHF = −75.9609699902，MP2 相关能 = −0.2036434634。
    const E_CORR_REFERENCE: f64 = -0.203_643_463_4;
    let options = TaskOptions::default().with_mp2(Mp2Options::new(Mp2Kind::Mp2));
    let result = backend()
        .run(&water_input(options), &NullReporter)
        .expect("MP2 应已接线");

    let correlation = result
        .correlation
        .as_ref()
        .expect("SC-04：correlation 块必须进 JobResult");
    println!(
        "method = {}, E_corr = {:.10}, E_total = {:.10}, E_SCF = {:.10}",
        correlation.method.as_str(),
        correlation.e2_hartree,
        correlation.e_total_hartree,
        result.energy_hartree
    );
    assert_eq!(correlation.method, Mp2Kind::Mp2);
    let difference = (correlation.e2_hartree - E_CORR_REFERENCE).abs();
    assert!(
        difference < 1.0e-4,
        "MP2 相关能 {:.10} 与 PySCF {E_CORR_REFERENCE:.10} 差 {difference:.2e} Ha",
        correlation.e2_hartree
    );
    // `e_total_hartree` 的定义就是 SCF + E_corr，且**不能**覆盖掉 SCF 能量本身。
    let expected_total = result.energy_hartree + correlation.e2_hartree;
    assert!(
        (correlation.e_total_hartree - expected_total).abs() < 1.0e-12,
        "e_total = {} 应为 E_SCF + E_corr = {expected_total}",
        correlation.e_total_hartree
    );
    // canonical MP2 没有拟合误差，也就不用辅助基。
    assert!(correlation.aux_basis_name.is_none());
    assert!(correlation.ri_error_hartree.is_none());

    // `result.json` 里要看得到：JobResult 的 serde 表示必须带 correlation 键。
    let json = serde_json::to_string(&result).expect("serialize JobResult");
    assert!(
        json.contains("\"correlation\""),
        "result.json 缺 correlation 块"
    );
    assert!(
        json.contains("\"method\":\"mp2\""),
        "method 拼写应为 mp2：{json}"
    );
}

#[test]
fn mp2_is_absent_when_not_requested() {
    let result = backend()
        .run(&water_input(TaskOptions::default()), &NullReporter)
        .expect("普通 HF 单点");
    assert!(
        result.correlation.is_none(),
        "没要求 MP2 就不该有 correlation 块"
    );
}

// `kind: ri-mp2` 的验收在 SC-05 的测试文件里（`sc05_rimp2_engine.rs`）：它要钉
// `aux_basis_name` 与 `ri_error_hartree` 落盘，而这两条属于 SC-05，不属于 SC-04。

#[test]
fn mp2_with_opt_is_rejected() {
    let options = TaskOptions::default().with_mp2(Mp2Options::new(Mp2Kind::Mp2));
    let mut input = water_input(options);
    input.jobtype = JobType::Opt;
    let error = backend()
        .run(&input, &NullReporter)
        .expect_err("MP2 梯度没实现，`opt` 必须拒绝");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("MP2 gradient"), "{message}");
        }
        other => panic!("期望 Unsupported（K5 退出码 4），得到 {other:?}"),
    }
}
