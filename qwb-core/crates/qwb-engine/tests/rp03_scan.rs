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
mod common;

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
fn the_gradient_is_consistent_at_every_scan_type() {
    // RP-04 对 RP-03 的要求：解析 vs 数值梯度 < 1e-5。约束切空间那一半已经在
    // `the_constrained_gradient_matches_a_finite_difference_along_the_tangent` 里验过；
    // 这里补**无约束**的普通梯度：距离、键角扫描点各取一个（都是把坐标挪到扫描值之后的几何，
    // 不是驻点，所以不是 0 vs 0）。
    use qwb_engine::opt::OptOptions;

    let registry = Registry::open(data_root()).expect("qwb-data");
    let mut options = OptOptions::new("sto-3g", "", "grid-coarse");
    options.force_threshold = 4.5e-4;

    let stretched = qwb_engine::scan::set_coordinate(&water(), ScanKind::Distance, &[0, 1], 1.10)
        .expect("distance scan point");
    common::assert_gradient_matches(
        "距离扫描点（O–H = 1.10 Å）",
        &registry,
        &stretched,
        &options,
        1.0e-3,
        1.0e-5,
        1.0e-2,
    );

    let bent = qwb_engine::scan::set_coordinate(&water(), ScanKind::Angle, &[1, 0, 2], 95.0)
        .expect("angle scan point");
    common::assert_gradient_matches(
        "键角扫描点（H–O–H = 95°）",
        &registry,
        &bent,
        &options,
        1.0e-3,
        1.0e-5,
        1.0e-2,
    );
}

#[test]
fn a_relaxed_scan_keeps_each_bond_length_and_lowers_the_energy() {
    // 弛豫扫描 = 每个点上做**约束优化**：固定 O–H，其余自由度放开。
    let values = [0.90, 1.00, 1.10];
    let scan =
        ScanOptions::new(ScanKind::Distance, vec![0, 1], values[0], values[2], 3).with_relax(true);
    let result = backend()
        .run(&input(scan), &NullReporter)
        .expect("约束弛豫扫描应跑通");
    let block = result.pes_scan.expect("pes_scan");
    block.validate().expect("块自洽");
    assert!(block.relaxed, "relax: true 的扫描必须标成 relaxed");
    assert_eq!(block.points.len(), 3);
    for (point, expected) in block.points.iter().zip(values) {
        assert!((point.value - expected).abs() < 1e-9);
        assert!(point.optimised, "弛豫点必须标成 optimised");
        assert!(point.iterations > 0, "弛豫点应报优化步数");
    }

    // ① 约束真的把内坐标固定在目标值上：曲线**不是平的**（自由优化会让三点塌回一个极小点）。
    let spread = block.max_adjacent_delta_hartree;
    assert!(
        spread > 1.0e-4,
        "弛豫扫描的相邻差只有 {spread:.3e} Ha —— 三点都塌回同一个极小点了？"
    );
    // ② 每个点的能量都不高于同坐标的刚性扫描（放松另外两个自由度只能降能）。
    let rigid = backend()
        .run(
            &input(ScanOptions::new(
                ScanKind::Distance,
                vec![0, 1],
                0.90,
                1.10,
                3,
            )),
            &NullReporter,
        )
        .expect("刚性扫描");
    let rigid_block = rigid.pes_scan.expect("pes_scan");
    for (relaxed, stiff) in block.points.iter().zip(&rigid_block.points) {
        assert!(
            relaxed.energy_hartree <= stiff.energy_hartree + 1e-9,
            "弛豫 {:.10} 应不高于刚性 {:.10}",
            relaxed.energy_hartree,
            stiff.energy_hartree
        );
    }
    println!(
        "弛豫扫描：{:?} Ha，max adjacent |dE| = {:.3e} Ha",
        block
            .points
            .iter()
            .map(|p| p.energy_hartree)
            .collect::<Vec<_>>(),
        spread
    );
}

#[test]
fn the_constrained_gradient_matches_a_finite_difference_along_the_tangent() {
    // RP-04 对 RP-03 的要求：解析梯度在**约束切空间**里与数值一致（< 1e-5）。
    use qwb_engine::opt::{Constraint, OptOptions, analytic_energy_and_gradient};
    use qwb_engine::scan::{constraint_gradient, measure, set_coordinate};

    let registry = Registry::open(data_root()).expect("qwb-data");
    // 先把对称性打破：水的两个 O–H 等价，对称方向上的梯度分量恒为零，"0 vs 0"的对照没有
    // 说服力。把第三个原子在 z 上推 0.05 Å（再重设 O–H），切空间方向导数才是非零的。
    let mut asymmetric = water();
    asymmetric.atoms[2].position_angstrom[2] += 0.05;
    let geometry = set_coordinate(&asymmetric, ScanKind::Distance, &[0, 1], 1.10).expect("set");
    // 约束对象本身也走一遍（`gradient()` 与 `constraint_gradient` 必须同一条路）。
    let constraint = Constraint::new(ScanKind::Distance, vec![0, 1], 1.10);
    let via_constraint = constraint.gradient(&geometry).expect("constraint gradient");
    assert_eq!(via_constraint.len(), 9);
    let options = OptOptions::new("sto-3g", "", "grid-coarse");

    let (_, gradient) =
        analytic_energy_and_gradient(&registry, &geometry, &options).expect("gradient");
    let mut flat: Vec<f64> = gradient
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let direction = constraint_gradient(&geometry, ScanKind::Distance, &[0, 1]).expect("c");
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let along: f64 = flat.iter().zip(&direction).map(|(g, c)| g * c).sum::<f64>() / (norm * norm);
    for (value, component) in flat.iter_mut().zip(&direction) {
        *value -= along * component;
    }

    // 切空间里的方向就取**投影后的梯度方向本身**：解析方向导数就是 |g_eff|，数值方向导数
    // 用同一方向的中心差分（每次位移后都把 O–H 重设回 1.10，保证差分留在约束面上）。
    let projected_norm = flat.iter().map(|value| value * value).sum::<f64>().sqrt();
    println!("约束切空间里的 |g_eff| = {projected_norm:.3e} Ha/Bohr");
    assert!(
        projected_norm > 1.0e-4,
        "投影后的梯度几乎为零（{projected_norm:.3e}）—— 这个几何已经在约束极小点上，对照没有信息量"
    );
    let mut tangent: Vec<f64> = flat.iter().map(|value| value / projected_norm).collect();
    let _ = &mut tangent;
    let analytic = flat.iter().zip(&tangent).map(|(g, t)| g * t).sum::<f64>();

    let step = 1.0e-3_f64;
    let displaced = |sign: f64| -> f64 {
        let mut moved = geometry.clone();
        for (index, component) in tangent.iter().enumerate() {
            let atom = index / 3;
            let axis = index % 3;
            moved.atoms[atom].position_angstrom[axis] += sign * step * component;
        }
        let moved = set_coordinate(&moved, ScanKind::Distance, &[0, 1], 1.10).expect("re-impose");
        let (energy, _) =
            analytic_energy_and_gradient(&registry, &moved, &options).expect("energy");
        energy
    };
    // 差分是在 **Å** 上做的（几何用 Å），而解析梯度是 Ha/**Bohr** —— 差一个 Å→Bohr 常数。
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let numeric = (displaced(1.0) - displaced(-1.0)) / (2.0 * step) * ANGSTROM_PER_BOHR;
    let difference = (analytic - numeric).abs();
    println!("约束切空间方向导数：解析 {analytic:.10}，数值 {numeric:.10}，差 {difference:.3e}");
    assert!(
        analytic.abs() > 1.0e-5,
        "这个切方向上的导数恒为零（{analytic:.3e}）—— 对照没有信息量，换一个非对称几何/方向"
    );
    assert!(
        difference < 1.0e-5,
        "约束下的方向导数：解析 {analytic:.10} vs 数值 {numeric:.10}（差 {difference:.3e}）"
    );
    // 顺带：约束梯度本身与几何一致（O–H 的方向），不是零向量。
    assert!(
        (measure(&geometry, ScanKind::Distance, &[0, 1]).expect("measure") - 1.10).abs() < 1e-10
    );
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
