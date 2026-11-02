//! RP-02 的引擎级验收：IRC（质量加权最陡下降，双向）。
//!
//! 判据（phase-2 §6 RP-02）三条：
//!
//! 1. **两端优化后的几何与已知极小差 < 0.02 Å**：用**内坐标指纹**比（键长 Å + 键角 °）——
//!    两个极小点是镜像，逐原子笛卡尔比较要先解决对齐与置换。"已知极小"由**独立的一条路**
//!    给出：从锥形初猜直接做自由优化（与 IRC 无关），再与 IRC 两端比；
//! 2. **能量单调下降**（允许 ≤ 1 个平台步）：`IrcEnd::monotone` 由点集派生；
//! 3. **`irc.json` 落盘**（CLI 侧，`t21_irc.rs`）。
//!
//! 起点是 `qwb ts` 在 NH3 上收敛出来的真鞍点（平面 D3h，伞形虚频 −1081 cm⁻¹）。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine::irc;
use qwb_engine::opt::{OptOptions, optimize};
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, IrcDirection, IrcOptions, JobInput, JobType, Molecule,
    TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

fn registry() -> Registry {
    Registry::open(data_root()).expect("qwb-data")
}

/// NH3 伞形鞍点（`qwb ts` 收敛出的几何，与 `examples/nh3_irc.yaml` 逐字一致）。
fn saddle() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("N", [-0.000_000_988_907, 0.0, 0.013_835_288_012]),
            Atom::new("H", [1.005_477_268_447, 0.0, 0.014_273_161_513]),
            Atom::new(
                "H",
                [-0.502_740_138_030, 0.870_769_888_810, 0.013_642_939_891],
            ),
            Atom::new(
                "H",
                [-0.502_740_138_029, -0.870_769_888_811, 0.013_642_939_886],
            ),
        ],
        0,
        1,
    )
}

/// 锥形初猜（`examples/nh3_ts.yaml` 的起点）：自由优化它会走到**锥形极小点**。
fn pyramidal_start() -> Molecule {
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

fn input(options: IrcOptions) -> JobInput {
    JobInput::new(
        saddle(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Irc,
        TaskOptions::default().with_irc(options),
    )
}

fn opt_options() -> OptOptions {
    let mut options = OptOptions::new("sto-3g", "", "grid-coarse");
    options.force_threshold = 4.5e-4;
    options
}

#[test]
fn an_irc_from_the_ammonia_saddle_reaches_both_minima() {
    let started = std::time::Instant::now();
    let result = backend()
        .run(
            &input(
                IrcOptions::new()
                    .with_max_steps(40)
                    .with_step_size(0.10)
                    .with_direction(IrcDirection::Both),
            ),
            &NullReporter,
        )
        .expect("NH3 的 IRC 应当跑通");
    let block = result.irc_result.expect("irc 作业必须带 irc_result");
    block.validate().expect("块自洽");

    assert!(
        (-1300.0..=-800.0).contains(&block.imaginary_wavenumber_cm),
        "虚频 {:.2} cm⁻¹",
        block.imaginary_wavenumber_cm
    );

    let forward = block.forward.expect("双向扫描应有 forward 端");
    let reverse = block.reverse.expect("双向扫描应有 reverse 端");
    assert!(forward.points.len() >= 5 && reverse.points.len() >= 5);

    // ② 单调性（`IrcEnd::new` 从点集派生）
    assert!(forward.monotone, "forward 端能量不单调");
    assert!(reverse.monotone, "reverse 端能量不单调");

    // ① 已知极小：独立地自由优化锥形初猜（与 IRC 无关）
    let minimum = optimize(&registry(), &pyramidal_start(), &opt_options()).expect("optimise");
    assert!(minimum.converged, "锥形初猜的优化没收敛");
    let reference = irc::fingerprint(&minimum.molecule);
    println!(
        "已知极小 E = {:.10} Ha，指纹 = {reference:?}",
        minimum.energy_hartree
    );

    // 指纹的前 `n(n-1)/2` 项是**键长（Å）**，之后是**键角（度）**：判据「< 0.02 Å」只对键长
    // 有意义，角度得用另一个尺度（1°）—— 把 0.02 同时套在角度上是第一版的做法，它把一个
    // 0.05° 的正常差异判成失败。
    let n_atoms = 4_usize;
    let n_distances = n_atoms * (n_atoms - 1) / 2;
    for (label, end) in [("forward", &forward), ("reverse", &reverse)] {
        assert_eq!(end.fingerprint.len(), reference.len());
        for (index, (value, expected)) in end.fingerprint.iter().zip(&reference).enumerate() {
            let tolerance = if index < n_distances { 0.02 } else { 1.0 };
            let unit = if index < n_distances { "Å" } else { "°" };
            assert!(
                (value - expected).abs() < tolerance,
                "{label} 端指纹第 {index} 项 {value:.6} {unit} 与已知极小 {expected:.6} 差 > {tolerance}"
            );
        }
        assert!(
            (end.minimum_energy_hartree - minimum.energy_hartree).abs() < 1.0e-6,
            "{label} 端端点能量 {:.10} 与已知极小 {:.10} 差得太多",
            end.minimum_energy_hartree,
            minimum.energy_hartree
        );
    }
    assert!((forward.minimum_energy_hartree - reverse.minimum_energy_hartree).abs() < 1.0e-6);
    println!(
        "IRC：{} + {} 点，虚频 {:.2} cm⁻¹，端点 E = {:.10} / {:.10} Ha（{:.1} s）",
        forward.points.len(),
        reverse.points.len(),
        block.imaginary_wavenumber_cm,
        forward.minimum_energy_hartree,
        reverse.minimum_energy_hartree,
        started.elapsed().as_secs_f64()
    );
}

#[test]
fn a_non_saddle_start_is_refused_with_a_clear_reason() {
    let registry = registry();
    let minimum = optimize(&registry, &pyramidal_start(), &opt_options()).expect("optimise");
    let error =
        irc::saddle_mode(&registry, &minimum.molecule, &opt_options()).expect_err("极小点不是鞍点");
    let message = error.to_string();
    assert!(
        message.contains("exactly one real imaginary mode"),
        "{message}"
    );
}

#[test]
fn a_single_direction_only_walks_that_side() {
    let result = backend()
        .run(
            &input(
                IrcOptions::new()
                    .with_max_steps(40)
                    .with_step_size(0.10)
                    .with_direction(IrcDirection::Forward),
            ),
            &NullReporter,
        )
        .expect("单向 IRC 应跑通");
    let block = result.irc_result.expect("irc_result");
    assert!(block.forward.is_some());
    assert!(
        block.reverse.is_none(),
        "只走 forward 时 reverse 端应为 None"
    );
    block.validate().expect("单向块也自洽");
}
