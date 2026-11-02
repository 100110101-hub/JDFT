//! RP-01 的引擎级验收：Hessian 引导的局部过渡态搜索。
//!
//! 判据是**两条一起**（phase-2 §6 RP-01）：`max|grad| < 4.5e-4 Ha/Bohr` **且**收敛后的
//! Hessian 恰好**一个**虚频且形状正确。这里用 NH3 的伞形反转（D3h 平面鞍点）当尺子：
//! 从略微锥形的初猜出发，本征向量跟随应当爬上平面鞍点。
//!
//! `n_imaginary` **只计真虚频**（波数 < −100 cm⁻¹）：平面 NH3 的零模噪声实测有 −16.4 与
//! −0.016 cm⁻¹ 两个点，把它们算进去会让"恰好一个虚频"永远不成立 —— 这条正是第一版踩的坑。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
    TsOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

/// 略微锥形的 NH3（与 `examples/nh3_ts.yaml` 逐字一致）。
fn pyramidal_ammonia() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("N", [0.0, 0.0, 0.06]),
            Atom::new("H", [0.94, 0.0, -0.20]),
            Atom::new("H", [-0.47, 0.814, -0.20]),
            Atom::new("H", [-0.47, -0.814, -0.20]),
        ],
        0,
        1,
    )
}

fn input(options: TsOptions) -> JobInput {
    JobInput::new(
        pyramidal_ammonia(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Ts,
        TaskOptions::default().with_ts(options),
    )
}

#[test]
fn the_ammonia_umbrella_saddle_is_a_first_order_saddle() {
    let started = std::time::Instant::now();
    let result = backend()
        .run(&input(TsOptions::new().with_max_steps(20)), &NullReporter)
        .expect("NH3 的伞形鞍点应当找到");
    let block = result.ts_result.expect("ts 作业必须带 ts_result");
    block.validate().expect("块自洽");

    // ① 梯度判据
    assert!(
        block.max_gradient_hartree_per_bohr < 4.5e-4,
        "max|g| = {:.3e} Ha/Bohr 没到判据",
        block.max_gradient_hartree_per_bohr
    );
    // ② 恰好一个虚频，且量级对（伞形振动是 −1000 cm⁻¹ 量级）
    assert_eq!(block.n_imaginary, 1, "一阶鞍点必须恰好一个虚频");
    assert!(
        (-1300.0..=-800.0).contains(&block.imaginary_wavenumber_cm),
        "虚频 {:.2} cm⁻¹ 不在伞形振动的量级里",
        block.imaginary_wavenumber_cm
    );
    assert!(block.is_first_order_saddle(4.5e-4));
    assert!(block.iterations > 0);
    // ③ 能量落在 STO-3G/HF 平面 NH3 的量级上（−55.4 Ha 附近）
    assert!(
        (block.energy_hartree + 55.437_665).abs() < 1e-2,
        "能量 {:.10} Ha",
        block.energy_hartree
    );
    // 鞍点几何：`examples/nh3_irc.yaml` 的起点取自本测试跑出的 `artifacts/optimized.xyz`
    // （手工填一个"大概平面"的几何会得到 4 个虚频 —— 非驻点的 Hessian 没有意义）。
    println!(
        "NH3 伞形鞍点：E = {:.10} Ha，max|g| = {:.3e} Ha/Bohr，虚频 {:.2} cm⁻¹，{} 步（{:.1} s）",
        block.energy_hartree,
        block.max_gradient_hartree_per_bohr,
        block.imaginary_wavenumber_cm,
        block.iterations,
        started.elapsed().as_secs_f64()
    );
}

#[test]
fn open_shell_ts_searches_are_refused_with_a_reason() {
    let mut job = input(TsOptions::new().with_max_steps(2));
    job.options.scf_type = Some(qwb_engine_api::ScfType::Uks);
    job.molecule.multiplicity = 2;
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("开壳层 TS 未验收");
    match error {
        EngineError::Unsupported(message) => {
            // 拒绝可能来自 TS 的闭壳层前置，也可能来自 `uks` 的 "sp only" 守卫
            // （注意后者的反引号形式是 "sp` only"）。
            assert!(
                message.contains("closed-shell") || message.contains("only"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported，得到 {other:?}"),
    }
}

#[test]
fn a_ts_search_that_runs_out_of_steps_says_so() {
    // 1 步远不够：必须报"没收敛"，而不是返回一个看起来正常的几何。
    let error = backend()
        .run(&input(TsOptions::new().with_max_steps(1)), &NullReporter)
        .expect_err("1 步不可能收敛");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("did not converge"), "{message}");
        }
        other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
    }
}
