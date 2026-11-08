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
mod common;

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
fn the_gradient_is_consistent_along_the_irc() {
    // RP-04 对 RP-02 的要求：解析 vs 数值梯度 < 1e-5，**不允许只有能量对**。取两处：
    // ① 离开鞍点之后的路径起点（沿虚频本征向量挪 0.15）；② 端点盆地里的一个非驻点几何
    // （把某个 N–H 拉长 0.05 Å）—— 两个驻点（鞍点、极小点）上两个梯度都是零，0 vs 0 没信息量。
    use qwb_engine::hessian::{atomic_mass, numerical_hessian};
    use qwb_engine::ts::analyse_hessian;

    let registry = registry();
    let options = opt_options();
    let (mode, _) = irc::saddle_mode(&registry, &saddle(), &options).expect("mode");
    let mut displaced = saddle();
    for atom in 0..displaced.atoms.len() {
        let mass = atomic_mass(&displaced.atoms[atom].symbol).expect("mass");
        for axis in 0..3 {
            let coordinate = 3 * atom + axis;
            displaced.atoms[atom].position_angstrom[axis] +=
                0.15 * mode[coordinate] / mass.sqrt() * 0.529_177_210_903;
        }
    }
    common::assert_gradient_matches(
        "IRC 起点（离开鞍点 0.15）",
        &registry,
        &displaced,
        &options,
        1.0e-3,
        1.0e-5,
        1.0e-3,
    );

    // 端点盆地里：自由优化后的极小点本身梯度为零，所以往外拉 0.05 Å 再比。
    let minimum = optimize(&registry, &pyramidal_start(), &options).expect("optimise");
    let mut nudged = minimum.molecule.clone();
    nudged.atoms[1].position_angstrom[0] += 0.05;
    common::assert_gradient_matches(
        "IRC 端点盆地（拉长 0.05 Å 的 N–H）",
        &registry,
        &nudged,
        &options,
        1.0e-3,
        1.0e-5,
        1.0e-2,
    );

    // 顺带核对：`saddle_mode` 与直接对角化给的是同一个模式（两条路同一份数据）。
    let hessian = numerical_hessian(&registry, &saddle(), &options).expect("hessian");
    let (eigenvalues, _) = analyse_hessian(&saddle(), &hessian.hessian).expect("eigen");
    assert!(eigenvalues[0] < 0.0, "最低本征值必须是负的（鞍点）");
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

/// **两端必须走到两个不同的极小点** —— HCN ↔ HNC 是这条判据的尺子。
///
/// 这个用例是 RP-01 的实测**抓出来的缺陷**的回归钉（第 397 轮）：`irc::walk` 原来用
/// 「沿 ±v 各探一步、取能量更低的那边」来决定起始方向，而**一阶鞍点的两侧都降能**（这就是
/// 虚频的定义）—— 于是 forward 与 reverse 无论传进来的 `sign` 是什么，都走同一条路，
/// 两端落在同一个极小点上。NH3 的伞形反转两端是**镜像**（同一个能量、同一套指纹），
/// 那条验收看不出任何异常；HCN ↔ HNC 两端差 19.3 kcal/mol，一眼就露。
///
/// 几何是 `qwb ts` 在 HF/STO-3G 上从文献鞍点几何收敛出来的（`E = -91.5648510258 Ha`、
/// 虚频 −1248.5 cm⁻¹、2 步收敛、max|g| = 1.12e-4）。两个极小点是本项目优化器在同层级
/// 上的输出：HCN `E = -91.6752089899 Ha`（2 步）、HNC `E = -91.6444372254 Ha`（4 步）。
/// 独立实现（PySCF 2.14.0）在同一层级给出同样的鞍点：`E = -91.5648510208 Ha`、
/// 虚频 −1248.37 cm⁻¹ —— 两套实现互差 5e-9 Ha / 0.17 cm⁻¹。
#[test]
#[ignore = "slow: HCN↔HNC 的双向 IRC（HF/STO-3G / grid-coarse，本机实测约 150 s）；run with -Slow"]
fn the_two_ends_of_an_unsymmetrical_irc_reach_different_minima() {
    // `qwb ts` 的鞍点输出（`artifacts/optimized.xyz` 逐字抄下来）。
    let saddle = Molecule::new(
        vec![
            Atom::new("C", [0.002_070_926_774, 0.0, -0.017_611_583_955]),
            Atom::new("N", [-0.003_712_698_394, 0.0, 1.203_676_105_275]),
            Atom::new("H", [1.148_414_368_756, 0.0, 0.343_830_922_593]),
        ],
        0,
        1,
    );
    let input = JobInput::new(
        saddle,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Irc,
        TaskOptions::default().with_irc(
            IrcOptions::new()
                .with_max_steps(80)
                .with_step_size(0.20)
                .with_direction(IrcDirection::Both),
        ),
    );
    let started = std::time::Instant::now();
    let block = backend()
        .run(&input, &NullReporter)
        .expect("HCN→HNC 鞍点的双向 IRC 应跑通")
        .irc_result
        .expect("irc_result");
    block.validate().expect("块自洽");

    let forward = block.forward.expect("forward 端");
    let reverse = block.reverse.expect("reverse 端");
    for (label, end) in [("forward", &forward), ("reverse", &reverse)] {
        assert!(end.monotone, "{label} 端能量不单调");
        assert!(
            end.points.len() >= 5,
            "{label} 端只有 {} 点",
            end.points.len()
        );
    }

    // 两个极小点在同一层级上的能量（上面的文档注释里写明了它们怎么来的）。
    let hcn = -91.675_208_989_9_f64;
    let hnc = -91.644_437_225_4_f64;
    let forward_energy = forward.minimum_energy_hartree;
    let reverse_energy = reverse.minimum_energy_hartree;

    // ① 两端**不是**同一个极小点：HNC − HCN = 19.31 kcal/mol（0.03077 Ha）。
    let gap = (forward_energy - reverse_energy).abs();
    assert!(
        gap > 0.02,
        "两端能量差 {gap:.6} Ha 太小 —— forward 与 reverse 大概走了同一条路（这个用例正是\
         为了钉住那条缺陷）"
    );
    // ② 一端是 HCN、另一端是 HNC（各 1 mHa 之内）。
    let mut ends = [forward_energy, reverse_energy];
    ends.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert!(
        (ends[0] - hcn).abs() < 1.0e-3,
        "低的一端 {:.10} Ha 不是 HCN 的 {hcn:.10}",
        ends[0]
    );
    assert!(
        (ends[1] - hnc).abs() < 1.0e-3,
        "高的一端 {:.10} Ha 不是 HNC 的 {hnc:.10}",
        ends[1]
    );
    println!(
        "HCN↔HNC 双向 IRC：{} + {} 点，两端 E = {forward_energy:.10} / {reverse_energy:.10} Ha\
         （差 {:.4} kcal/mol），{:.1} s",
        forward.points.len(),
        reverse.points.len(),
        gap * 627.5094740631,
        started.elapsed().as_secs_f64()
    );
}
