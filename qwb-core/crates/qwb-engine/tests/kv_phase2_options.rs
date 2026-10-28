//! KV-01..KV-03（phase-2 §4.1 / §6.1）：阶段二的选项**契约已就位、引擎未实现**。
//!
//! 这条边界很重要：契约 crate 现在能解析 `scf_type` / `mp2` / `excited_states` /
//! `solvent_model`，但引擎还没实现它们。如果引擎静默忽略这些字段，用户会拿到一个
//! "看起来成功"的闭壳层气相结果，却以为那是 UKS/MP2/溶剂化的结果 —— 这正是
//! `project-phase2.md` §10.3 明令禁止的"静默降级"。
//!
//! 所以本文件钉住三件事：
//! 1. 每个未实现的选项都以 `Unsupported`（K5 退出码 4）**显式拒绝**，消息里带 WP 编号；
//! 2. 与阶段一行为等价的写法（`scf_type: rks` / `rhf`、`solvent_model: vacuum`）**放过**，
//!    否则旧输入会被无故拒绝；
//! 3. 拒绝发生在任何数值计算之前（返回得快、不烧 CPU）。

use std::path::PathBuf;
use std::time::Instant;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, CorrelationResult, EngineBackend, EngineError, ExcitedStateMethod, ExcitedStatesOptions,
    GridLevel, JobInput, JobType, Molecule, Mp2Kind, Mp2Options, NumericsOptions, NumericsPath,
    ScfType, SolventModel, SpinResult, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data 必须能加载"))
}

fn molecule(multiplicity: u32) -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        multiplicity,
    )
}

fn input(method: &str, multiplicity: u32, options: TaskOptions) -> JobInput {
    JobInput::new(
        molecule(multiplicity),
        "sto-3g",
        method,
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        options,
    )
}

fn unsupported(input: &JobInput) -> String {
    // 阶段二的 Redis 守卫在 SCF 之前：用 `NullReporter` 直接调后端。
    let error = backend()
        .run(input, &NullReporter)
        .err()
        .unwrap_or_else(|| panic!("{} 应当被显式拒绝", error_context(input)));
    match error {
        EngineError::Unsupported(message) => message,
        other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
    }
}

fn error_context(input: &JobInput) -> String {
    format!(
        "scf_type={:?} mp2={:?} excited_states={:?} solvent_model={:?}",
        input.options.scf_type,
        input.options.mp2,
        input.options.excited_states,
        input.options.solvent_model
    )
}

#[test]
fn open_shell_scf_type_is_rejected_with_its_wp() {
    // UHF（SC-01）与 UKS（SC-01b）都已实装，只剩 ROHF（SC-01c，blocked）会被拒。
    // UHF 的验收在 `sc01_uhf_engine.rs`，UKS 的在 `sc01b_uks_engine.rs`。
    let options = TaskOptions::default().with_scf_type(ScfType::Rohf);
    let message = unsupported(&input("svwn5", 3, options));
    assert!(message.contains("SC-01c"), "{message}");
    assert!(message.contains("ROHF"), "{message}");
    assert!(message.contains("blocked"), "{message}");
}

#[test]
fn ri_mp2_is_rejected_with_its_wp() {
    // canonical MP2 在 M3 已接（SC-04）；RI-MP2（SC-05）还没做，必须显式拒绝。
    let options = TaskOptions::default()
        .with_mp2(Mp2Options::new(Mp2Kind::RiMp2).with_aux_basis("def2-svp-ri"));
    let message = unsupported(&input("svwn5", 1, options));
    assert!(message.contains("SC-05"), "{message}");
    assert!(message.contains("RI-MP2"), "{message}");
}

#[test]
fn excited_states_are_rejected_with_their_wp() {
    let options = TaskOptions::default()
        .with_excited_states(ExcitedStatesOptions::new(5, ExcitedStateMethod::Tda));
    let message = unsupported(&input("svwn5", 1, options));
    assert!(message.contains("RS-02"), "{message}");
    assert!(message.contains("CIS/TDA"), "{message}");
}

#[test]
fn implicit_solvent_models_are_rejected_with_their_wp() {
    let mut job = input("svwn5", 1, TaskOptions::default());
    job.solvent = "water".into();
    job.options.solvent_model = Some(SolventModel::Cosmo);
    let message = unsupported(&job);
    assert!(message.contains("SV-01"), "{message}");
    assert!(message.contains("cosmo"), "{message}");
}

#[test]
fn numerics_path_selection_is_rejected_with_its_wp() {
    // 精度等级本身没有任何代码在读：放过就是静默忽略（K10 要求近似必须留痕）。
    for path in [NumericsPath::Analytic, NumericsPath::Ri] {
        let options = TaskOptions::default().with_numerics(NumericsOptions::new(path, 3));
        let message = unsupported(&input("svwn5", 1, options));
        assert!(message.contains("IN-03"), "{message}");
        assert!(message.contains("K10"), "{message}");
    }
}

#[test]
fn phase1_equivalent_options_are_not_rejected() {
    // 闭壳层 RKS / RHF 与阶段一的默认路径等价；显式 vacuum 也一样。
    for (method, scf_type) in [("svwn5", ScfType::Rks), ("hf", ScfType::Rhf)] {
        let options = TaskOptions::default()
            .with_scf_type(scf_type)
            .with_solvent_model(SolventModel::Vacuum);
        let result = backend().run(&input(method, 1, options), &NullReporter);
        let result = result.unwrap_or_else(|e| panic!("{method}/{scf_type:?} 不该被拒：{e}"));
        assert!(result.scf_converged, "H2/STO-3G/{method} 应当收敛");
        assert!(result.energy_hartree.is_finite());
        // 阶段一的结果块仍然缺省（新字段只在实现后才出现）。
        assert!(result.spin.is_none());
        assert!(result.correlation.is_none());
    }
}

#[test]
fn rejection_happens_before_any_numerics() {
    // canonical MP2 已经实装，所以这里用**还没做**的路径（RI-MP2 / SC-05）钉"立刻拒绝"。
    let options = TaskOptions::default()
        .with_mp2(Mp2Options::new(Mp2Kind::RiMp2).with_aux_basis("def2-svp-ri"));
    let started = Instant::now();
    let message = unsupported(&input("svwn5", 1, options));
    let elapsed = started.elapsed();
    assert!(message.contains("SC-05"), "{message}");
    // 一次 STO-3G/H2 的 SCF 在本机也要几十毫秒；"立刻拒绝"应当是毫秒级。
    assert!(
        elapsed.as_millis() < 50,
        "拒绝发生在数值计算之后？耗时 {elapsed:?}"
    );
}

#[test]
fn result_blocks_round_trip_through_the_k2_types() {
    // KV-01/KV-03 的产物要能被"拼进"结果并原样读回（引擎实现前的最小可用性检查）。
    let spin = SpinResult::new(5, 5, 3, 2.0);
    let correlation =
        CorrelationResult::new(Mp2Kind::RiMp2, -0.31, -76.33).with_aux_basis("def2-svp-ri");
    let json = serde_json::to_string(&(spin, &correlation)).expect("serialize");
    let (spin2, correlation2): (SpinResult, CorrelationResult) =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(spin2, spin);
    assert_eq!(correlation2, correlation);
    assert!(correlation2.validate().is_ok());
}
