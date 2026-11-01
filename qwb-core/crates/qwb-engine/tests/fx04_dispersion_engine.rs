//! FX-04 验收（引擎侧）：色散作为**可加项**进 `result.json`。
//!
//! 判据（§6, FX-04）的第三条：「`dispersion_energy_hartree` 进 `result.json`」。
//! 数值核心的两条判据（对官方实现 < 1e-3 kcal/mol、梯度 vs 有限差分 < 1e-6 Ha/Bohr）
//! 由 `qwb-scf` 的 `fx04_dispersion.rs` 覆盖，这里只管引擎接线：
//!
//! 1. **不给选项时逐位不变**：同一份输入加/不加 `options.dispersion`，SCF 能量完全相同，
//!    且结果里的 `dispersion_energy_hartree` 是 `None`（阶段一/既有里程碑的回归不受影响）；
//! 2. **给了选项时**：`energy_hartree` = SCF 能量 + 色散能，`dispersion_energy_hartree`
//!    就是这个差值（落盘的不是混合总数），并且 log 里两部分都写出来；
//! 3. **opt 显式拒绝**：优化器的能量/梯度不含色散，事后加会自相矛盾，所以报
//!    `Unsupported` 而不是给半个答案；
//! 4. **未知数据名/泛函名报错**，消息里给出可用的键名；
//! 5. M06-2X（没有 BJ 参数）带色散时也必须显式报错。

use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, DispersionOptions, EngineBackend, GridLevel, JobInput, JobType, Molecule, TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data must load"))
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.757, 0.586, 0.0]),
            Atom::new("H", [-0.757, 0.586, 0.0]),
        ],
        0,
        1,
    )
}

fn input(jobtype: JobType, method: &str) -> JobInput {
    let mut options = TaskOptions::default();
    options.scf_max_iter = 100;
    JobInput::new(
        water(),
        "sto-3g",
        method,
        GridLevel::Normal,
        "vacuum",
        jobtype,
        options,
    )
}

#[test]
fn fx04_dispersion_lands_in_the_result_and_adds_up() {
    let backend = backend();
    let base = backend
        .run(&input(JobType::Sp, "pbe0"), &NullReporter)
        .expect("baseline sp");
    assert!(
        base.dispersion_energy_hartree.is_none(),
        "without the option the block must stay absent"
    );

    let mut with_dispersion = input(JobType::Sp, "pbe0");
    with_dispersion.options.dispersion = Some(DispersionOptions::default());
    let result = backend
        .run(&with_dispersion, &NullReporter)
        .expect("sp with dispersion");
    let dispersion = result
        .dispersion_energy_hartree
        .expect("dispersion_energy_hartree must be set");
    println!(
        "scf = {:.12} Ha, dispersion = {:.12} Ha ({:.4} kcal/mol), total = {:.12} Ha",
        base.energy_hartree,
        dispersion,
        dispersion * 627.509_474_063_1,
        result.energy_hartree
    );

    // ① SCF 那一半逐位不变。
    assert_eq!(
        base.mo_energies_hartree, result.mo_energies_hartree,
        "the SCF part must be bit-identical"
    );
    // ② 总能量 = SCF + 色散，误差在浮点加法量级。
    assert!(
        (result.energy_hartree - (base.energy_hartree + dispersion)).abs() < 1e-14,
        "total {} != scf {} + dispersion {dispersion}",
        result.energy_hartree,
        base.energy_hartree
    );
    // ③ 水分子 + STO-3G + PBE0 的 D3(BJ) 在 −1e-3 Ha 量级（水二聚体是 −2.19e-3）；
    //    这里只断言量级，绝对值由 qwb-scf 的官方对照钉住。
    assert!(
        dispersion < 0.0 && dispersion > -1.0e-3,
        "dispersion = {dispersion} Ha out of the expected range"
    );
    // ④ log 里两部分都写出来（可核对，不是一个混合总数）。
    assert!(
        result.log.contains("E_dispersion"),
        "log should report the dispersion term: {}",
        result.log
    );
}

#[test]
fn fx04_without_the_option_is_the_old_behaviour() {
    let backend = backend();
    // 阶段一的回归靠这一条：默认输入不产生色散块，能量与 mo 能级不带任何新项。
    for method in ["pbe0", "m06-2x", "hf"] {
        let result = backend
            .run(&input(JobType::Sp, method), &NullReporter)
            .expect("baseline sp");
        assert!(
            result.dispersion_energy_hartree.is_none(),
            "{method}: dispersion block must be absent by default"
        );
    }
}

#[test]
fn fx04_optimisation_with_dispersion_is_rejected() {
    let backend = backend();
    let mut request = input(JobType::Opt, "pbe0");
    request.options.dispersion = Some(DispersionOptions::default());
    let error = backend
        .run(&request, &NullReporter)
        .expect_err("opt + dispersion must be rejected");
    let message = format!("{error}");
    println!("opt + dispersion: {message}");
    assert!(message.contains("supports `sp` only"), "{message}");
}

#[test]
fn fx04_unknown_data_and_functionals_are_reported() {
    let backend = backend();

    let mut request = input(JobType::Sp, "pbe0");
    // `DispersionOptions` 是 non_exhaustive：先 Default 再覆盖字段。
    let mut options = DispersionOptions::default();
    options.data = "d4".into();
    request.options.dispersion = Some(options);
    let error = backend
        .run(&request, &NullReporter)
        .expect_err("unknown data must be rejected");
    let message = format!("{error}");
    println!("unknown data: {message}");
    assert!(message.contains("d4"), "{message}");

    // M06-2X 没有 BJ 参数：必须报错并指出可用的四个泛函。
    let mut request = input(JobType::Sp, "m06-2x");
    request.options.dispersion = Some(DispersionOptions::default());
    let error = backend
        .run(&request, &NullReporter)
        .expect_err("a functional without BJ parameters must be rejected");
    let message = format!("{error}");
    println!("m06-2x + dispersion: {message}");
    assert!(message.contains("no D3(BJ) parameters"), "{message}");
    assert!(message.contains("b3lyp"), "{message}");
}
