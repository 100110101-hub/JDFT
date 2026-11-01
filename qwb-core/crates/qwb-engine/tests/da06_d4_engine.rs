//! DA-06：**D4 已经能从引擎里选**（`options.dispersion.data = "d4"`）。
//!
//! 这一条守的是**接线**而不是物理：D4 的能量/梯度精度由 `qwb-scf` 的四条测试分别钉住
//! （中间量 1e-9/1e-10、能量 6.4e-13 Ha、梯度 5.6e-14 Ha/Bohr）。这里要确认的是引擎把
//! 「体系总电荷、泛函名、Bohr 坐标」正确地交给了 D4 那条路径，而不是还在用 D3 —— 两者的数据
//! 是**两套类型**（`kind: d4` 与 `kind: d3bj/d3zero`），所以必须分派到不同的加载器与内核。
//!
//! 判据：
//! 1. `result.dispersion_energy_hartree` 出现，且 `total = SCF + E_disp`；
//! 2. 引擎给的 E_disp 与**直接调库**（同一几何、同一电荷）逐位一致（差 < 1e-14 Ha）——
//!    这条抓的是"引擎传错了电荷/坐标/泛函名"；
//! 3. 同一体系下 D4 与 D3(0) 的数值不同（否则说明 `data: d4` 被静默当成 D3 处理）。

use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, DispersionOptions, EngineBackend, GridLevel, JobInput, JobType, Molecule, ScfType,
    TaskOptions,
};
use qwb_registry::Registry;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn dioxygen() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.208]),
        ],
        0,
        3,
    )
}

fn run_with(data: &str) -> (f64, f64) {
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let mut options = TaskOptions::default();
    options.scf_max_iter = 200;
    options.scf_type = Some(ScfType::Uks);
    options.dispersion = Some({
        let mut dispersion = DispersionOptions::default();
        dispersion.data = data.to_string();
        dispersion
    });
    let input = JobInput::new(
        dioxygen(),
        "sto-3g",
        "m06-l",
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        options,
    );
    let result = backend
        .run(&input, &NullReporter)
        .unwrap_or_else(|error| panic!("uks + m06-l + {data}: {error}"));
    (
        result.energy_hartree,
        result
            .dispersion_energy_hartree
            .expect("the dispersion block must be present"),
    )
}

#[test]
fn the_engine_can_select_d4_and_reports_it_additively() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let (total_d4, dispersion_d4) = run_with("d4");
    let (total_d3, dispersion_d3) = run_with("d3zero");
    println!("D4: total = {total_d4:.12} Ha, E_disp = {dispersion_d4:+.12} Ha");
    println!("D3(0): total = {total_d3:.12} Ha, E_disp = {dispersion_d3:+.12} Ha");

    // ① 可加性：total = SCF + E_disp，且 SCF 那半两次相同（同一体系、同一泛函）。
    let scf_d4 = total_d4 - dispersion_d4;
    let scf_d3 = total_d3 - dispersion_d3;
    assert!(
        (scf_d4 - scf_d3).abs() < 1.0e-9,
        "the SCF energy must not depend on the dispersion data set ({scf_d4} vs {scf_d3})"
    );
    assert!(dispersion_d4.is_finite() && dispersion_d4 < 0.0);

    // ② 与直接调库逐位一致（同一几何、同一总电荷、同一泛函名）。
    let molecules = dioxygen();
    let symbols: Vec<&str> = molecules
        .atoms
        .iter()
        .map(|atom| atom.symbol.as_str())
        .collect();
    let positions: Vec<[f64; 3]> = molecules
        .atoms
        .iter()
        .map(|atom| atom.position_bohr())
        .collect();
    let data = registry.get_d4_dispersion("d4").expect("d4 data");
    let direct = qwb_scf::d4::dispersion_energy(
        data,
        "m06-l",
        &symbols,
        &positions,
        f64::from(molecules.charge),
    )
    .expect("direct d4")
    .energy_hartree;
    println!("direct library call: {direct:+.12} Ha");
    assert!(
        (dispersion_d4 - direct).abs() < 1.0e-14,
        "the engine must pass charge/functional/positions through unchanged \
         ({dispersion_d4} vs {direct})"
    );

    // ③ D4 与 D3(0) 在这条短键（1.208 Å）上数值不同 —— 证明 `data: d4` 没被当成 D3。
    assert!(
        (dispersion_d4 - dispersion_d3).abs() > 1.0e-6,
        "D4 and D3(0) must differ here ({dispersion_d4} vs {dispersion_d3})"
    );
}

#[test]
fn an_unknown_dispersion_data_key_is_rejected() {
    // 拼错的数据键名要**显式报错**，而不是静默不算色散。
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let mut options = TaskOptions::default();
    options.scf_type = Some(ScfType::Uks);
    options.scf_max_iter = 200;
    options.dispersion = Some({
        let mut dispersion = DispersionOptions::default();
        dispersion.data = "d4-typo".to_string();
        dispersion
    });
    let input = JobInput::new(
        dioxygen(),
        "sto-3g",
        "m06-l",
        GridLevel::Normal,
        "vacuum",
        JobType::Sp,
        options,
    );
    let error = backend
        .run(&input, &NullReporter)
        .expect_err("an unknown dispersion key must be rejected");
    println!("d4-typo: {error}");
    assert!(
        format!("{error}").contains("d4-typo"),
        "the error must name the unknown key: {error}"
    );
}
