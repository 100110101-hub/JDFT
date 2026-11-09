//! SV-01 的转动不变性在**溶剂化能**上的验收：`ΔG_solv` 不许随刚性转动漂移。
//!
//! 上一轮审计量到的事实（本轮把它钉住，不重新推导）：把水分子刚性转动一下，
//! 同一份输入的 `ΔG_solv` 移动 **+0.2410 / +1.2762 kcal/mol**（M 410→413、
//! A 36.4582→36.5878、`E_vac` 逐位相同）。来源是空腔的 Fibonacci 点钉在全局坐标轴上
//! （见 `qwb_props::cavity` 的模块文档），不是 SCF 也不是 COSMO 求解器的问题。
//!
//! 第 434 轮加的 [`qwb_props::cavity::CavityFrame::Covariant`] 让坐标架随分子一起转，于是
//! 空腔逐位不变、`ΔG_solv` 只剩 SCF 的舍入；第 435 轮把它**翻成默认**（`cavity()` 与
//! `smd_surface()` 都用它）。这个文件在**真实 COSMO-SCF** 上量：
//!
//! * 协变模式下 COSMO SCF 是否还收敛、`ΔG_solv` 与实验值的差（水/甲醇）；
//! * `ΔG_solv` 在三次刚性转动下的**离散度**（被换掉的全局模式实测 1.276 kcal/mol（水）/
//!   0.121（甲醇）；现在默认的协变模式是 1.78e-11 / 1.78e-10 kcal/mol，只剩 SCF 舍入）；
//! * tessera 数 M 与表面积 A 在转动下是否还跳（协变 408/660 逐个不差，全局 [410,413,413,409]）；
//! * `CavityFrame::default()` 确实是协变（`[Global]` 那一列只剩历史对照的作用）。
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use qwb_basis::InternalBasis;
use qwb_engine::cosmo_scf::CosmoScfField;
use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_props::cavity::{CavityFrame, cavity_with_frame, rotate_molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// Hartree → kcal/mol（CODATA 2018，与 `sv01_cosmo_scf.rs` 同一个常数）。
fn hartree_to_kcal() -> f64 {
    627.509_474_063_6
}

fn bondi_radii() -> BTreeMap<String, f64> {
    let mut radii = BTreeMap::new();
    for (symbol, radius) in [
        ("H", 1.20),
        ("C", 1.70),
        ("N", 1.55),
        ("O", 1.52),
        ("F", 1.47),
        ("S", 1.80),
        ("Cl", 1.75),
    ] {
        radii.insert(symbol.to_string(), radius);
    }
    radii
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

/// 甲醇：与 `sv01_cosmo_scf.rs` 同一份手搭几何（不是实验结构，几何误差计入实测差距）。
fn methanol() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("O", [0.0, 0.0, 1.420]),
            Atom::new("H", [0.0, 0.885, 1.760]),
            Atom::new("H", [0.0, 1.025, -0.363]),
            Atom::new("H", [0.888, -0.512, -0.363]),
            Atom::new("H", [-0.888, -0.512, -0.363]),
        ],
        0,
        1,
    )
}

/// 审计所用的三种刚性转动（与 `sv04_covariant_cavity.rs` 同一组）。
const ROTATIONS: [([f64; 3], f64); 3] = [
    ([0.0, 0.0, 1.0], 0.7),
    ([1.0, 0.0, 0.0], 1.3),
    ([0.3, -0.5, 0.81], 2.9),
];

struct Outcome {
    energy_hartree: f64,
    solvation_energy_hartree: f64,
    tesserae: usize,
    area_angstrom2: f64,
}

/// STO-3G / HF（与 `sv01_cosmo_scf.rs` 同一条 SCF 通路），空腔可选坐标架。
fn run_scf(
    registry: &Registry,
    molecule: &Molecule,
    epsilon: Option<f64>,
    points_per_atom: usize,
    frame: CavityFrame,
) -> Result<Outcome, EngineError> {
    let basis = InternalBasis::build(registry.get_basis("sto-3g")?, molecule)?;
    let tensor = two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse")?, molecule)
        .map_err(|err| EngineError::Internal(format!("grid: {err}")))?;
    let mut options = RksOptions::default();
    options.max_iter = 64;
    options.energy_threshold = 1.0e-9;
    options.density_threshold = 1.0e-7;
    let mut field: Option<Arc<CosmoScfField>> = None;
    let mut tesserae = 0;
    let mut area = 0.0;
    if let Some(epsilon) = epsilon {
        let built_cavity =
            cavity_with_frame(molecule, &bondi_radii(), 1.0, points_per_atom, frame)?;
        tesserae = built_cavity.tesserae.len();
        area = built_cavity.surface_area_angstrom2;
        let built = Arc::new(CosmoScfField::build(
            &basis,
            molecule,
            &built_cavity.tesserae,
            epsilon,
        )?);
        options.external_field = Some(built.handle());
        field = Some(built);
    }
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options)?;
    if !result.scf_converged {
        return Err(EngineError::Internal(format!(
            "cosmo scf did not converge for {} atoms",
            molecule.n_atoms()
        )));
    }
    Ok(Outcome {
        energy_hartree: result.energy_hartree,
        solvation_energy_hartree: field
            .as_ref()
            .map_or(0.0, |field| field.last_solvation_energy()),
        tesserae,
        area_angstrom2: area,
    })
}

/// 一个分子在给定模式下、reference + 三次转动的 `(M, A, ΔG kcal/mol)`。
fn rotation_sweep(
    registry: &Registry,
    name: &str,
    molecule: &Molecule,
    epsilon: f64,
    points: usize,
    frame: CavityFrame,
) -> Vec<(usize, f64, f64)> {
    let vacuum = run_scf(registry, molecule, None, points, frame).expect("vacuum scf");
    println!(
        "{name} [{frame:?}]: E_vac = {:.9} Ha",
        vacuum.energy_hartree
    );
    let mut results = Vec::new();
    let geometries: Vec<(&str, Molecule)> = std::iter::once(("reference", molecule.clone()))
        .chain(
            ROTATIONS
                .iter()
                .map(|(axis, angle)| ("rotated", rotate_molecule(molecule, *axis, *angle))),
        )
        .collect();
    for (label, geometry) in &geometries {
        let solvated =
            run_scf(registry, geometry, Some(epsilon), points, frame).expect("cosmo scf");
        let delta_g = (solvated.energy_hartree - vacuum.energy_hartree) * hartree_to_kcal();
        println!(
            "  {label}: M = {}, A = {:.4} A^2, dG = {delta_g:.4} kcal/mol \
             (field dG_el = {:.4})",
            solvated.tesserae,
            solvated.area_angstrom2,
            solvated.solvation_energy_hartree * hartree_to_kcal()
        );
        results.push((solvated.tesserae, solvated.area_angstrom2, delta_g));
    }
    let spread = results
        .iter()
        .map(|entry| entry.2)
        .fold(f64::NEG_INFINITY, f64::max)
        - results
            .iter()
            .map(|entry| entry.2)
            .fold(f64::INFINITY, f64::min);
    println!("  {name} [{frame:?}]: dG spread over the rigid rotations = {spread:.3e} kcal/mol");
    results
}

#[test]
fn covariant_cosmo_scf_converges_and_tracks_the_global_frame() {
    // 快道烟测：协变空腔在**真实** COSMO-SCF 里能收敛，且与同点数的全局模式给出同一个
    // 量级的 ΔG（换坐标架只该换采样取向，不该换物理）。100 点/原子的 M 足够小，秒级。
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let vacuum =
        run_scf(&registry, &molecule, None, 100, CavityFrame::Covariant).expect("vacuum scf");
    let global = run_scf(&registry, &molecule, Some(78.355), 100, CavityFrame::Global)
        .expect("global cosmo scf");
    let covariant = run_scf(
        &registry,
        &molecule,
        Some(78.355),
        100,
        CavityFrame::Covariant,
    )
    .expect("covariant cosmo scf");
    let global_delta = (global.energy_hartree - vacuum.energy_hartree) * hartree_to_kcal();
    let covariant_delta = (covariant.energy_hartree - vacuum.energy_hartree) * hartree_to_kcal();
    println!(
        "water 100 points/atom: M global {} (A {:.4}) dG {global_delta:.4} kcal/mol; \
         M covariant {} (A {:.4}) dG {covariant_delta:.4} kcal/mol",
        global.tesserae, global.area_angstrom2, covariant.tesserae, covariant.area_angstrom2
    );
    assert!(covariant_delta < 0.0, "solvation must stabilise");
    assert!(
        (covariant_delta - global_delta).abs() < 0.2,
        "the frame choice moved dG by {:.3} kcal/mol, which is beyond sampling error",
        covariant_delta - global_delta
    );
}

#[test]
#[ignore = "slow: 20 COSMO-SCF runs (water+methanol, both frames, reference + 3 rigid rotations), 82 s"]
fn solvation_energy_under_rigid_rotation() {
    // 验收数字：被换掉的全局模式上一轮记录 0.2410 / 1.2762 kcal/mol，本轮同一组转动实测
    // **0.1206（甲醇）/ 1.276（水）**（全局路径逐位未变，差异来自记录口径）。协变模式下空腔
    // 逐位不变（见 `sv04_covariant_cavity.rs`），所以这里只剩 SCF 的舍入（实测 1.78e-11 /
    // 1.78e-10）。判据取 1e-4 kcal/mol：比旧漂移小四个数量级，同时不把"SCF 自身的末位噪声"
    // 当成失败。
    assert_eq!(
        CavityFrame::default(),
        CavityFrame::Covariant,
        "the default frame must be the covariant one - `cavity()` feeds the production solvent path"
    );
    let registry = Registry::open(data_root()).expect("qwb-data");
    for (name, molecule, epsilon) in [("water", water(), 78.355), ("methanol", methanol(), 32.613)]
    {
        for frame in [CavityFrame::Covariant, CavityFrame::Global] {
            let results = rotation_sweep(&registry, name, &molecule, epsilon, 250, frame);
            let reference = results[0].2;
            let mut worst: f64 = 0.0;
            let mut counts = Vec::new();
            for (tesserae, area, delta_g) in &results {
                counts.push(*tesserae);
                worst = worst.max((delta_g - reference).abs());
                if frame == CavityFrame::Covariant {
                    assert!(
                        (area - results[0].1).abs() / results[0].1 < 1.0e-6,
                        "{name}: the covariant surface area moved under rotation"
                    );
                }
            }
            println!(
                "  {name} [{frame:?}]: worst |dG - dG_ref| = {worst:.3e} kcal/mol, M = {counts:?}"
            );
            if frame == CavityFrame::Covariant {
                assert_eq!(
                    counts
                        .iter()
                        .cloned()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                    1,
                    "{name}: the covariant tessera count moved under rotation"
                );
                assert!(
                    worst < 1.0e-4,
                    "{name}: dG_solv moved by {worst:.3e} kcal/mol under a rigid rotation - the \
                     cavity is not rotation covariant"
                );
            } else {
                // 被换掉的全局模式（第 435 轮之前是默认）：缺陷本身没改，所以漂移仍在。这条
                // 现在的作用是"当初为什么翻默认"的可执行记录；若有人把 Global 的实现也改了，
                // 它会红 —— 那时必须同步改这里与账本，而不是放宽它。
                assert!(
                    worst > 1.0e-4,
                    "{name}: the global frame no longer moves dG_solv under rotation (worst \
                     {worst:.3e} kcal/mol) - the global implementation changed; update this test \
                     and the ledger"
                );
            }
        }
    }
}

#[test]
#[ignore = "slow: 8 COSMO-SCF runs (water+methanol, both frames at 250 points/atom), 21 s"]
fn solvation_energy_against_the_experiment_at_250_points() {
    // SV-01 的第二条判据：ΔG_solv 与实验值差 < 1.5 kcal/mol。水在判据内，甲醇在判据外
    // （缺空化/色散/排斥三项，`docs/choices.md` 六十四记录过；几何也不是实验结构）。
    let registry = Registry::open(data_root()).expect("qwb-data");
    for (name, molecule, epsilon, experimental, criterion_met) in [
        ("water", water(), 78.355, -6.32, true),
        ("methanol", methanol(), 32.613, -3.71, false),
    ] {
        let vacuum =
            run_scf(&registry, &molecule, None, 250, CavityFrame::Covariant).expect("vacuum scf");
        for frame in [CavityFrame::Covariant, CavityFrame::Global] {
            let solvated =
                run_scf(&registry, &molecule, Some(epsilon), 250, frame).expect("cosmo scf");
            let delta_g = (solvated.energy_hartree - vacuum.energy_hartree) * hartree_to_kcal();
            let gap = delta_g - experimental;
            println!(
                "{name} [{frame:?}] STO-3G ({} tesserae, A = {:.4} A^2): dG = {delta_g:.4} kcal/mol, \
                 experiment {experimental:.2} -> gap {gap:+.4}",
                solvated.tesserae, solvated.area_angstrom2
            );
            assert!(delta_g < 0.0, "{name}: solvation must stabilise");
            if criterion_met && frame == CavityFrame::Covariant {
                assert!(
                    gap.abs() < 1.5,
                    "{name}: covariant dG {delta_g:.4} kcal/mol is {gap:+.3} from experiment"
                );
            }
            if !criterion_met {
                println!(
                    "  {name} [{frame:?}]: gap {gap:+.4} kcal/mol is outside the 1.5 criterion \
                     (recorded blocker: no cavitation/dispersion/repulsion terms)"
                );
            }
        }
    }
}
