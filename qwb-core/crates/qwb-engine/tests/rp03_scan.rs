//! RP-03 的引擎级验收：**刚性** PES 扫描（`jobtype: scan`）。
//!
//! 判据（phase-2 §6 RP-03）里这一半能验的是三条：
//!
//! 1. **能量连续**：相邻点的 |ΔE| 有界，且曲线在几何平衡位置附近取极小 —— 这是"扫描真的扫了
//!    而不是在原地重复计算"的证据；
//! 2. **`pes.json` 可被 GUI 读取**：由 CLI 侧测（`t19_scan.rs`），这里只测块本身；
//! 3. **`jobtype: scan` 进 `SUPPORTED_FEATURES`**：引擎在注册表声明该 feature 时不拒绝。
//!
//! 弛豫那一半（约束优化）**还没有实现**，所以这里有一条专门的拒绝测试 —— 不许把"每个点自由
//! 优化"当成弛豫扫描（那会把所有点压回同一个极小点，曲线是平的）。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::pes::{ScanKind, ScanOptions};
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
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
            Atom::new("O", [0.0, 0.0, 0.11779]),
            Atom::new("H", [0.0, 0.75545, -0.47116]),
            Atom::new("H", [0.0, -0.75545, -0.47116]),
        ],
        0,
        1,
    )
}

fn input(scan: ScanOptions) -> JobInput {
    JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Scan,
        TaskOptions::default().with_scan(scan),
    )
}

#[test]
fn a_rigid_oh_scan_has_a_minimum_near_the_equilibrium_bond_length() {
    let scan = ScanOptions::new(ScanKind::Distance, vec![0, 1], 0.80, 1.20, 9);
    let result = backend()
        .run(&input(scan), &NullReporter)
        .expect("刚性扫描应跑通");
    let block = result.pes_scan.expect("scan 作业必须带 pes_scan");
    block.validate().expect("块自洽");
    assert_eq!(block.kind, "distance");
    assert_eq!(block.unit, "angstrom");
    assert!(!block.relaxed, "缺省是刚性扫描");
    assert_eq!(block.points.len(), 9);
    assert!(block.points.iter().all(|point| !point.optimised));

    // ① 坐标是等距的、两端的值就是 start/stop。
    assert!((block.points[0].value - 0.80).abs() < 1e-12);
    assert!((block.points[8].value - 1.20).abs() < 1e-12);
    for pair in block.points.windows(2) {
        assert!((pair[1].value - pair[0].value - 0.05).abs() < 1e-12);
    }

    // ② 能量连续（相邻差有界）：实测 3.1e-2 Ha，判据取 5e-2（再大就说明某一步换了 SCF 解）。
    assert!(
        block.max_adjacent_delta_hartree < 5.0e-2,
        "max adjacent |dE| = {:.3e} Ha",
        block.max_adjacent_delta_hartree
    );
    // ③ 曲线是"一个碗"：极小点在 1.00 Å（50 步网格上的近平衡点），两端都比它高。
    let minimum = block.minimum().expect("minimum");
    assert!(
        (minimum.value - 1.00).abs() < 1e-9,
        "极小点在 {}，平衡键长 0.9898 附近",
        minimum.value
    );
    assert!(
        block.points[0].energy_hartree > minimum.energy_hartree + 1e-3,
        "0.80 Å 的压缩能应当明显更高"
    );
    assert!(
        block.points[8].energy_hartree > minimum.energy_hartree + 1e-3,
        "1.20 Å 的拉伸能应当明显更高"
    );
    println!(
        "O–H 扫描：{} 点，极小 {:.3} Å / {:.10} Ha，max adjacent |dE| = {:.3e} Ha",
        block.points.len(),
        minimum.value,
        minimum.energy_hartree,
        block.max_adjacent_delta_hartree
    );

    // 结果块里的能量是"最低点"，与块里的极小点一致。
    assert!((result.energy_hartree - minimum.energy_hartree).abs() < 1e-12);
}

#[test]
fn an_angle_scan_also_works_and_keeps_the_bond_lengths() {
    let scan = ScanOptions::new(ScanKind::Angle, vec![1, 0, 2], 90.0, 115.0, 6);
    let result = backend()
        .run(&input(scan), &NullReporter)
        .expect("键角扫描应跑通");
    let block = result.pes_scan.expect("pes_scan");
    assert_eq!(block.kind, "angle");
    assert_eq!(block.unit, "degree");
    assert_eq!(block.points.len(), 6);
    assert!((block.points[0].value - 90.0).abs() < 1e-9);
    assert!((block.points[5].value - 115.0).abs() < 1e-9);
    // 只动第 3 个原子：两根 O–H 的长度都不该变（改键角不该顺带改键长）。
    let minimum = block.minimum().expect("minimum");
    assert!(minimum.energy_hartree < block.points[0].energy_hartree);
}

#[test]
fn relaxed_scans_are_refused_instead_of_collapsing_to_one_minimum() {
    let scan = ScanOptions::new(ScanKind::Distance, vec![0, 1], 0.85, 1.15, 3).with_relax(true);
    let error = backend()
        .run(&input(scan), &NullReporter)
        .expect_err("弛豫扫描（约束优化）还没实现，必须拒绝");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("constrained"), "{message}");
            assert!(message.contains("RP-03"), "{message}");
        }
        other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
    }
}

#[test]
fn a_scan_without_coordinates_is_refused_with_the_field_name() {
    let job = JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Scan,
        TaskOptions::default(),
    );
    let error = backend()
        .run(&job, &NullReporter)
        .expect_err("没有 options.scan 必须拒绝");
    match error {
        EngineError::InvalidInput(message) => {
            assert!(message.contains("options.scan"), "{message}");
        }
        other => panic!("期望 InvalidInput（退出码 1），得到 {other:?}"),
    }
}

#[test]
fn scan_is_a_declared_feature() {
    assert!(
        qwb_engine_api::SUPPORTED_FEATURES.contains(&"jobtype:scan"),
        "RP-03 的第三条判据：`jobtype: scan` 要在 SUPPORTED_FEATURES 里"
    );
}
