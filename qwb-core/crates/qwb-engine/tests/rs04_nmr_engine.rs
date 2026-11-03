//! RS-04 的引擎级验收：NMR 屏蔽接进结果块（GIAO / 非 GIAO 两种规范）。
//!
//! 判据（phase-2 §6 RS-04）要的是 **GIAO**：H2O 的 ¹H 化学位移差 < 0.5 ppm、CH4/CH3OH 的
//! ¹H 与 ¹³C < 1.0/5.0 ppm、`length_gauge_spread_ppm < 1`。GIAO 那一半现在**满足**（本文件
//! 末尾两条测试，方法名 `giao`、规范变化 < 1 ppm）；`nucleus-origin` / `common-origin`
//! 那两条**不是** GIAO，所以第三条对它们不满足 —— 那两条测试的作用是把它**量出来并报进
//! 结果块**，而不是假装通过（历史断言一行没改）。
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

/// GIAO 验收用的水几何（Å）：与 `qwb-response/tests/rs04_giao_nmr.rs::water()` 以及
/// `scripts/pyscf_giao_nmr_reference.py` 的 `SYSTEMS["water"]` **逐字相同**。
///
/// 必须用它，而不是本文件上面的 `water()`：365.672798 / 33.682096 这两个验收值是在**这份**
/// 几何上钉的，换一份几何差好几 ppm（所以上面那两条历史断言继续用它们自己的 `water()`）。
fn validated_water() -> Molecule {
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

fn giao_input() -> JobInput {
    JobInput::new(
        validated_water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default().with_nmr(NmrOptions::new().with_gauge(NmrGauge::Giao)),
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

/// **A1（GIAO 绝对值）**：引擎报 `method = "giao"`，逐核各向同性进 0.1 ppm，规范变化 < 1 ppm。
///
/// 参考值是 `qwb-response/tests/rs04_giao_nmr.rs` 里逐核钉过的数字（PySCF 2.14 + libcint
/// GIAO + 转写的 `prop/nmr/rhf.py`）：水/STO-3G O **365.672764**、H **33.682096**。本实现实测
/// 365.672798 / 33.682096（见 `giao_nmr.rs` 模块文档），所以 0.1 ppm 的窗口两头都够。
#[test]
fn the_giao_shielding_lands_in_the_result_block_with_a_sub_ppm_gauge_spread() {
    let result = backend()
        .run(&giao_input(), &NullReporter)
        .expect("GIAO 屏蔽应跑通");
    let block = result.nmr_shieldings.expect("nmr_shieldings 块");
    assert_eq!(block.method, "giao", "GIAO 模式的方法名必须是 giao");
    // GIAO 没有"选一个规范原点"的自由度，落盘的是坐标原点。
    assert_eq!(block.gauge_origin_bohr, [0.0, 0.0, 0.0]);
    assert_eq!(block.nuclei.len(), 3);
    for (index, nucleus) in block.nuclei.iter().enumerate() {
        assert_eq!(nucleus.index, index);
        assert!(nucleus.isotropic_ppm.is_finite());
    }
    let oxygen = &block.nuclei[0];
    let hydrogen = &block.nuclei[1];
    let hydrogen2 = &block.nuclei[2];
    assert_eq!(oxygen.element, "O");
    assert_eq!(hydrogen.element, "H");
    println!(
        "水/STO-3G（GIAO，验收几何）：O = {:.6} ppm、H = {:.6}/{:.6} ppm、\
         规范变化 = {:?} ppm（方法 {}）",
        oxygen.isotropic_ppm,
        hydrogen.isotropic_ppm,
        hydrogen2.isotropic_ppm,
        block.length_gauge_spread_ppm,
        block.method
    );
    // A1：0.1 ppm 窗口（与 `rs04_giao_nmr.rs` 的 ABSOLUTE_TOLERANCE_PPM 同一个判据）。
    for (label, computed, expected) in [
        ("O", oxygen.isotropic_ppm, 365.672_798_f64),
        ("H#1", hydrogen.isotropic_ppm, 33.682_096_f64),
        ("H#2", hydrogen2.isotropic_ppm, 33.682_096_f64),
    ] {
        assert!(
            (computed - expected).abs() < 0.1,
            "{label} 的 GIAO 屏蔽 {computed:.6} ppm 不在 {expected:.6}±0.1 内"
        );
    }
    // GIAO 给的是完整张量：不是共原规范那样填零。
    let tensor_norm: f64 = oxygen.tensor_ppm.iter().flatten().map(|v| v.abs()).sum();
    assert!(
        tensor_norm > 100.0,
        "O 的屏蔽张量几乎全是 0（|Σ| = {tensor_norm}）：GIAO 的完整张量没落盘"
    );
    // 第三条判据（phase-2 §4.2）：GIAO 的规范变化必须真的 < 1 ppm。
    let spread = block
        .length_gauge_spread_ppm
        .expect("GIAO 缺省要报规范变化");
    assert!(
        spread < 1.0,
        "GIAO 的规范变化 {spread:.3e} ppm 不小于 1 ppm"
    );
}

/// **A2（非 GIAO 路径没被改动）**：`nucleus-origin` / `common-origin` 的数与加 GIAO 之前一样。
///
/// 判据直接对着**既有报告值**（4 位小数逐字比）：O 172.2631、H 99.6291、
/// `length_gauge_spread_ppm` 7.1041。这条不是"再测一遍"，而是"加性变更不许动老数字"。
#[test]
fn the_non_giao_modes_keep_their_pre_giao_numbers() {
    let nucleus = backend()
        .run(&input(NmrOptions::new()), &NullReporter)
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
    println!(
        "核中心：O = {:.9}、H = {:.9}、规范变化 = {:.9} ppm",
        left.nuclei[0].isotropic_ppm,
        left.nuclei[1].isotropic_ppm,
        left.length_gauge_spread_ppm.expect("spread")
    );
    println!(
        "共原：O = {:.9}、H = {:.9}、规范变化 = {:.9} ppm",
        right.nuclei[0].isotropic_ppm,
        right.nuclei[1].isotropic_ppm,
        right.length_gauge_spread_ppm.expect("spread")
    );
    // 冻结值（4 位小数，逐字）：与加 GIAO 之前**完全一样**。
    assert_eq!(format!("{:.4}", left.nuclei[0].isotropic_ppm), "172.2631");
    assert_eq!(format!("{:.4}", left.nuclei[1].isotropic_ppm), "99.6291");
    assert_eq!(
        format!(
            "{:.4}",
            left.length_gauge_spread_ppm.expect("nucleus spread")
        ),
        "7.1041"
    );
    // 共原模式的老数字同样钉住（同一次运行实测：173.733816210 / 106.733270077）。
    assert_eq!(format!("{:.4}", right.nuclei[0].isotropic_ppm), "173.7338");
    assert_eq!(format!("{:.4}", right.nuclei[1].isotropic_ppm), "106.7333");
}

/// **拒绝的组合**：开壳层、DFT 参考、ECP —— 三条都必须是显式错误，不是静默忽略。
///
/// 开壳层那一条尤其重要：`scf_type: uhf` 在 `run_single_point` 开头就分派到自旋通道，
/// 守卫如果只写在 NMR 块里，`options.nmr` 就会被**静默丢掉**（`nmr_shieldings` 直接不出现）。
#[test]
fn giao_and_nmr_reject_combinations_they_cannot_serve() {
    // ① 开壳层（UHF，三重态 O2）。
    let open_shell = JobInput::new(
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.0]),
                Atom::new("O", [0.0, 0.0, 1.208]),
            ],
            0,
            3,
        ),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        JobType::Sp,
        TaskOptions::default()
            .with_nmr(NmrOptions::new().with_gauge(NmrGauge::Giao))
            .with_scf_type(qwb_engine_api::ScfType::Uhf),
    );
    let message = backend()
        .run(&open_shell, &NullReporter)
        .expect_err("开壳层 + NMR 必须拒绝")
        .to_string();
    println!("开壳层拒绝：{message}");
    assert!(
        message.contains("closed-shell") && message.contains("nmr"),
        "错误里应点名 closed-shell / nmr：{message}"
    );

    // ② DFT 参考（GIAO 的响应算符没有 XC kernel）。
    let mut dft = giao_input();
    dft.method_name = "pbe".into();
    let message = backend()
        .run(&dft, &NullReporter)
        .expect_err("DFT + GIAO 必须拒绝")
        .to_string();
    println!("DFT 拒绝：{message}");
    assert!(
        message.contains("hf") && message.contains("giao"),
        "错误里应点名 hf / giao：{message}"
    );

    // ③ ECP（占据数必须按 z_eff 数）。
    let mut ecp = giao_input();
    ecp.options.ecp_name = Some("lanl2dz-cu".into());
    let message = backend()
        .run(&ecp, &NullReporter)
        .expect_err("ECP + GIAO 必须拒绝")
        .to_string();
    println!("ECP 拒绝：{message}");
    assert!(
        message.contains("ECP") && message.contains("z_eff"),
        "错误里应点名 ECP / z_eff：{message}"
    );
}
