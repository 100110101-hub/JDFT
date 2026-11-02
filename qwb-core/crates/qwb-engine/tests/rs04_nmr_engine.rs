//! RS-04 的引擎级验收：NMR 屏蔽接进结果块（抗磁 + 顺磁）。
//!
//! 判据（phase-2 §6 RS-04）要的是 **GIAO**：H2O 的 ¹H 化学位移差 < 0.5 ppm、CH4/CH3OH 的
//! ¹H 与 ¹³C < 1.0/5.0 ppm、`length_gauge_spread_ppm < 1`。本项目的屏蔽是**共原规范**的
//! （核中心 / 核电荷重心），所以第三条不满足 —— 这个测试的作用是把它**量出来并报进结果块**，
//! 而不是假装通过。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, NmrGauge, NmrOptions, TaskOptions,
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
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

fn input(options: NmrOptions) -> JobInput {
    JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default().with_nmr(options),
    )
}

#[test]
fn nmr_shieldings_land_in_the_result_block_with_their_gauge_spread() {
    let result = backend()
        .run(&input(NmrOptions::new()), &NullReporter)
        .expect("NMR 屏蔽应跑通");
    let block = result.nmr_shieldings.expect("nmr_shieldings 块");
    assert_eq!(
        block.method, "nucleus-origin",
        "共原规范不是 GIAO，不许写 giao"
    );
    assert_eq!(block.nuclei.len(), 3);
    for (index, nucleus) in block.nuclei.iter().enumerate() {
        assert_eq!(nucleus.index, index);
        assert!(nucleus.isotropic_ppm.is_finite());
    }
    let oxygen = &block.nuclei[0];
    let hydrogen = &block.nuclei[1];
    assert_eq!(oxygen.element, "O");
    assert_eq!(hydrogen.element, "H");
    // 与 PySCF 的独立实现同量级（`rs04_paramagnetic` 里钉的是 171.779 / 99.569，这里的几何与
    // 那一份略有不同，所以判据给 1 ppm 的窗口而不是逐位比）。
    println!(
        "水/STO-3G：O = {:.4} ppm，H = {:.4} ppm，规范变化 = {:?} ppm（方法 {}）",
        oxygen.isotropic_ppm, hydrogen.isotropic_ppm, block.length_gauge_spread_ppm, block.method
    );
    assert!(
        (oxygen.isotropic_ppm - 171.8).abs() < 1.5,
        "O 屏蔽 {:.4} ppm 不在 171.8±1.5",
        oxygen.isotropic_ppm
    );
    assert!(
        (hydrogen.isotropic_ppm - 99.6).abs() < 1.5,
        "H 屏蔽 {:.4} ppm 不在 99.6±1.5",
        hydrogen.isotropic_ppm
    );
    // 判据的第三条：共原规范下 > 1 ppm —— 正是"还没有 GIAO"的量化证据。
    let spread = block.length_gauge_spread_ppm.expect("缺省要报规范变化");
    assert!(
        spread > 1.0,
        "共原规范的规范变化 {spread:.4} ppm 竟然 < 1 ppm？那要么是 GIAO（还没做），要么是常数没接对"
    );
}

#[test]
fn the_common_gauge_choice_changes_the_numbers_but_not_by_orders_of_magnitude() {
    let nucleus = backend()
        .run(
            &input(NmrOptions::new().with_gauge(NmrGauge::Nucleus)),
            &NullReporter,
        )
        .expect("nucleus gauge");
    let common = backend()
        .run(
            &input(NmrOptions::new().with_gauge(NmrGauge::Common)),
            &NullReporter,
        )
        .expect("common gauge");
    let left = nucleus.nmr_shieldings.expect("nucleus block");
    let right = common.nmr_shieldings.expect("common block");
    assert_eq!(left.method, "nucleus-origin");
    assert_eq!(right.method, "common-origin");
    let difference = (left.nuclei[0].isotropic_ppm - right.nuclei[0].isotropic_ppm).abs();
    println!(
        "O 屏蔽：核中心 {:.4} vs 共原 {:.4} ppm（差 {difference:.4}）",
        left.nuclei[0].isotropic_ppm, right.nuclei[0].isotropic_ppm
    );
    // 两种原点的差就是"规范变化"，量级在 ppm 上；不许是 0（那说明原点参数没被用上）。
    assert!(difference > 0.1, "两种原点给出同一个数：原点参数没接上");
    assert!(difference < 20.0, "两种原点差得离谱：{difference:.4} ppm");
}
