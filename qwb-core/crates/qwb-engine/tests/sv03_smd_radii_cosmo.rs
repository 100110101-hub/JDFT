//! SV-03 的接线验收：`radii_set: smd` 时 COSMO/C-PCM 的空腔半径改由 **SMD 参数化半径**（eq. 16）
//! 提供，并且**不再乘** Bondi 那套的 `scale`。
//!
//! 为什么值得单测：`cosmo_radii` 是 `Engine::run` 里选半径的唯一入口，而当前**没有**任何溶剂
//! 文件声明 `radii_set: smd`（`cavity_model` 还是 `pcm`）—— 所以这条分支不会被端到端跑到。
//! 把它抽成纯函数、直接对着真实的 `smd-atomic` 数据测，比"等哪天有数据再说"更可靠：
//! 分支一旦被激活（把某个溶剂改成 `radii_set: smd`），这里的断言就是它的规格说明。
use std::collections::BTreeMap;

use qwb_engine::cosmo_radii;
use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_registry::{RadiiSet, Registry};
use std::path::PathBuf;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn registry() -> Registry {
    Registry::open(data_root()).expect("qwb-data")
}

fn water() -> Molecule {
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

fn bondi() -> BTreeMap<String, f64> {
    let mut radii = BTreeMap::new();
    radii.insert("H".to_string(), 1.20);
    radii.insert("O".to_string(), 1.52);
    radii
}

#[test]
fn the_bondi_path_is_returned_unchanged() {
    let molecule = water();
    let registry = registry();
    let smd = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    for set in [None, Some(RadiiSet::Bondi), Some(RadiiSet::Uff)] {
        let (radii, scale) =
            cosmo_radii(&molecule, set, &bondi(), 1.2, Some(0.43), Some(smd)).expect("radii");
        assert_eq!(
            radii,
            bondi(),
            "{set:?}: the file's radii must pass through"
        );
        assert_eq!(scale, 1.2, "{set:?}: the file's scale must pass through");
    }
}

#[test]
fn the_smd_path_uses_the_parameterised_radii_and_scale_one() {
    let molecule = water();
    let registry = registry();
    let smd = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    // α = 0.43（水的 Abraham 酸性）→ O 取 1.52 Å、H 取 1.20 Å；缩放必须变成 1.0，
    // 否则 Bondi 那套的 1.2 会把 eq. 16 的半径再放大 20%。
    let (radii, scale) = cosmo_radii(
        &molecule,
        Some(RadiiSet::Smd),
        &bondi(),
        1.2,
        Some(0.43),
        Some(smd),
    )
    .expect("radii");
    assert_eq!(radii["O"], 1.52);
    assert_eq!(radii["H"], 1.20);
    assert_eq!(scale, 1.0, "the SMD radii must not be scaled again");

    // α 变小 → O 变大（eq. 16 的斜率 1.8 Å/α）。
    let (low_alpha, _) = cosmo_radii(
        &molecule,
        Some(RadiiSet::Smd),
        &bondi(),
        1.2,
        Some(0.0),
        Some(smd),
    )
    .expect("radii");
    println!(
        "O radius: alpha 0.43 -> 1.52, alpha 0.0 -> {:.4}",
        low_alpha["O"]
    );
    assert!((low_alpha["O"] - (1.52 + 1.8 * 0.43)).abs() < 1.0e-12);
}

#[test]
fn the_smd_path_refuses_missing_data_or_alpha() {
    let molecule = water();
    let registry = registry();
    let smd = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let error = cosmo_radii(
        &molecule,
        Some(RadiiSet::Smd),
        &bondi(),
        1.2,
        None,
        Some(smd),
    )
    .expect_err("alpha is required for the oxygen radius");
    println!("missing alpha -> {error}");
    assert!(matches!(error, EngineError::Data(_)));

    let error = cosmo_radii(
        &molecule,
        Some(RadiiSet::Smd),
        &bondi(),
        1.2,
        Some(0.43),
        None,
    )
    .expect_err("the smd-atomic data is required");
    println!("missing data -> {error}");
    assert!(matches!(error, EngineError::Data(_)));
}

#[test]
fn the_cavity_area_changes_when_the_radius_set_changes() {
    // 半径换一套，面积与 Born 型能量都会变 —— 这里只量化面积，避免把"换半径"当成无操作。
    // 水：bondi（O 1.52 + 1.2 缩放 = 1.824 有效）vs SMD（O 1.52、scale 1.0）。
    let molecule = water();
    let registry = registry();
    let smd = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let (bondi_radii, bondi_scale) = cosmo_radii(
        &molecule,
        Some(RadiiSet::Bondi),
        &bondi(),
        1.2,
        Some(0.43),
        Some(smd),
    )
    .expect("bondi");
    let (smd_radii, smd_scale) = cosmo_radii(
        &molecule,
        Some(RadiiSet::Smd),
        &bondi(),
        1.2,
        Some(0.43),
        Some(smd),
    )
    .expect("smd");
    let with_bondi =
        qwb_props::cavity::cavity(&molecule, &bondi_radii, bondi_scale, 250).expect("cavity");
    let with_smd =
        qwb_props::cavity::cavity(&molecule, &smd_radii, smd_scale, 250).expect("cavity");
    println!(
        "cavity area: bondi x1.2 = {:.3} A^2, smd = {:.3} A^2",
        with_bondi.surface_area_angstrom2, with_smd.surface_area_angstrom2
    );
    assert!(
        with_smd.surface_area_angstrom2 < with_bondi.surface_area_angstrom2,
        "SMD radii (no extra 1.2 scaling) must give the smaller cavity"
    );
}
