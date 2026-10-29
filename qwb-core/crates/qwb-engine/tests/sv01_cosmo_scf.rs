//! SV-01（第三半）验收：COSMO 反应场接进 SCF，产出 `ΔG_solv`。
//!
//! 三条判据：
//! 1. 耦合自洽（`scf_converged`、`Σq ≈ 0`（中性溶质）、`E_总 − E_真空 = ΔG_el` 的恒等式）；
//! 2. `ε = 1` 时与真空 SCF **逐位相同**；
//! 3. 水/甲醇的 `ΔG_solv` 与实验值的差 —— 数值如实打印，判据（< 1.5 kcal/mol）按实测核对。
//!
//! 实验值（298.15 K、1 M 标准态、含全部非极性项）：水 −6.32 kcal/mol、甲醇 −3.71 kcal/mol
//! （SM8 表的常规引用；见 `docs/choices.md` 六十二里对出处强弱与标准态的说明）。
//! 本实现只做**静电**项，甲醇的几何也不是实验结构 —— 这两点都计入实测差距，不掩饰。
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use qwb_basis::InternalBasis;
use qwb_engine::RksBackend;
use qwb_engine::cosmo_scf::CosmoScfField;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, TaskOptions,
};
use qwb_integrals::two_electron::two_electron;
use qwb_props::cavity::cavity;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

/// Hartree → kcal/mol（CODATA 2018）。
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

/// 甲醇：C–O 1.42 Å、O–H 0.99 Å、C–H 1.10 Å、C–O–H 108.5° 的手搭结构。
/// **不是**实验结构、也不是我们自己的优化结果 —— 几何误差计入实测差距（choices 六十二）。
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

struct Outcome {
    energy_hartree: f64,
    converged: bool,
    solvation_energy: Option<f64>,
    total_charge: Option<f64>,
    tesserae: usize,
}

fn run_scf(
    registry: &Registry,
    molecule: &Molecule,
    epsilon: Option<f64>,
    points_per_atom: usize,
) -> Result<Outcome, EngineError> {
    let basis_k1 = registry.get_basis("sto-3g")?;
    let basis = InternalBasis::build(basis_k1, molecule)?;
    let tensor = two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse")?, molecule)
        .map_err(|err| EngineError::Internal(format!("grid: {err}")))?;
    let mut options = RksOptions::default();
    options.max_iter = 64;
    options.energy_threshold = 1.0e-9;
    options.density_threshold = 1.0e-7;
    let mut field: Option<Arc<CosmoScfField>> = None;
    let mut tesserae = 0;
    if let Some(epsilon) = epsilon {
        let cavity = cavity(molecule, &bondi_radii(), 1.0, points_per_atom)?;
        tesserae = cavity.tesserae.len();
        let built = Arc::new(CosmoScfField::build(
            &basis,
            molecule,
            &cavity.tesserae,
            epsilon,
        )?);
        options.external_field = Some(built.handle());
        field = Some(built);
    }
    let result = rks_scf(&basis, molecule, &tensor, &grid, &[], options)?;
    let (solvation_energy, total_charge) = match &field {
        Some(field) => (
            Some(field.last_solvation_energy()),
            Some(field.last_total_charge()),
        ),
        None => (None, None),
    };
    Ok(Outcome {
        energy_hartree: result.energy_hartree,
        converged: result.scf_converged,
        solvation_energy,
        total_charge,
        tesserae,
    })
}

/// `criterion_met` = 该分子是否落在 SV-01 的 1.5 kcal/mol 判据内。
/// 甲醇**不落在**里面（静电项没有空化/色散/排斥，系统性偏负 ~2 kcal/mol），
/// 这是 `docs/choices.md` 六十四记下来的实测差距，不是可以调容差糊过去的事。
fn report(
    name: &str,
    registry: &Registry,
    molecule: &Molecule,
    experimental: f64,
    criterion_met: bool,
) {
    let vacuum = run_scf(registry, molecule, None, 250).expect("vacuum scf");
    assert!(vacuum.converged, "{name}: vacuum SCF did not converge");
    let solvated = run_scf(registry, molecule, Some(78.355), 250).expect("cosmo scf");
    assert!(solvated.converged, "{name}: COSMO SCF did not converge");
    let delta_g = solvated.energy_hartree - vacuum.energy_hartree;
    let measured = delta_g * hartree_to_kcal();
    let field_energy = solvated.solvation_energy.expect("field energy");
    let total_charge = solvated.total_charge.expect("total charge");
    println!(
        "{name}/STO-3G ({} tesserae): E_vac={:.9}  E_solv={:.9}  dG={:.9} Ha = {measured:.3} kcal/mol",
        solvated.tesserae, vacuum.energy_hartree, solvated.energy_hartree, delta_g
    );
    println!(
        "    field dG_el={:.9} Ha ({:.3} kcal/mol)   sum q={total_charge:+.4} e   identity dG-dG_el={:.2e} Ha",
        field_energy,
        field_energy * hartree_to_kcal(),
        delta_g - field_energy
    );
    println!(
        "    experiment {experimental:.2} kcal/mol -> gap {:+.2} kcal/mol",
        measured - experimental
    );
    // 中性溶质的表面电荷总和必须接近 0（屏蔽不改变总电荷）
    assert!(
        total_charge.abs() < 0.2,
        "{name}: total surface charge {total_charge} should be ~0 for a neutral solute"
    );
    assert!(delta_g < 0.0, "{name}: solvation must stabilise");
    // 恒等式：SCF 的总能量差 == 场自己的静电溶剂化能（末步密度差异允许小残差）
    assert!(
        (delta_g - field_energy).abs() < 2.0e-3,
        "{name}: dG {delta_g} vs field dG_el {field_energy}"
    );
    if criterion_met {
        assert!(
            (measured - experimental).abs() < 1.5,
            "{name}: gap {:.2} kcal/mol exceeds the 1.5 kcal/mol criterion",
            measured - experimental
        );
    } else {
        // 差距必须**留在** 1.5 之外（否则说明记录已过期，文档要跟着改）
        assert!(
            (measured - experimental).abs() > 1.5,
            "{name}: the recorded gap is now inside 1.5 kcal/mol; update choices 64 and the \
             ledger instead of silently keeping a stale blocker"
        );
        assert!(
            measured < experimental,
            "{name}: the missing nonpolar terms are positive, so the electrostatic-only value \
             must come out too negative (measured {measured:.2}, experiment {experimental:.2})"
        );
    }
}

#[test]
fn sv01_water_solvation_energy_against_experiment() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 水：−7.274 kcal/mol vs 实验 −6.32 → 差 −0.95 kcal/mol（判据内）
    report("water", &registry, &water(), -6.32, true);
}

#[test]
fn sv01_methanol_solvation_energy_against_experiment() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 甲醇：−5.789 kcal/mol vs 实验 −3.71 → 差 −2.08 kcal/mol（判据外，卡点 = 缺非极性项）
    report("methanol", &registry, &methanol(), -3.71, false);
}

#[test]
fn sv01_solvation_energy_is_stable_against_the_tessera_count() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let vacuum = run_scf(&registry, &molecule, None, 100).expect("vacuum");
    let mut energies = Vec::new();
    for points in [100usize, 150, 250] {
        let solvated = run_scf(&registry, &molecule, Some(78.355), points).expect("cosmo");
        assert!(solvated.converged);
        let delta_g = (solvated.energy_hartree - vacuum.energy_hartree) * hartree_to_kcal();
        println!(
            "    {points:3} points/atom ({} tesserae): dG = {delta_g:.3} kcal/mol",
            solvated.tesserae
        );
        energies.push(delta_g);
    }
    // 离散化误差是单调的：碎片越多，屏蔽越完全（ΔG 更负）。它不是收敛的障碍，
    // 而是必须如实记下来的不确定度量级 —— 250 点/原子与 150 点/原子差 ~0.35 kcal/mol。
    for window in energies.windows(2) {
        assert!(
            window[1] < window[0],
            "dG should become more negative with more tesserae: {energies:?}"
        );
    }
    let spread = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - energies.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("    spread over tessera counts: {spread:.3} kcal/mol");
    assert!(
        spread < 1.2,
        "tessera-count sensitivity {spread:.3} kcal/mol is too large"
    );
}

#[test]
fn sv01_epsilon_one_reproduces_the_vacuum_energy() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let vacuum = run_scf(&registry, &molecule, None, 60).expect("vacuum scf");
    let epsilon_one = run_scf(&registry, &molecule, Some(1.0), 60).expect("cosmo scf");
    assert!(vacuum.converged && epsilon_one.converged);
    let field_energy = epsilon_one.solvation_energy.expect("field energy");
    println!(
        "E_vac={:.12}  E(eps=1)={:.12}  dE={:.3e}  field={field_energy:.3e}",
        vacuum.energy_hartree,
        epsilon_one.energy_hartree,
        epsilon_one.energy_hartree - vacuum.energy_hartree
    );
    assert!(
        field_energy.abs() < 1.0e-14,
        "eps=1 field energy {field_energy}"
    );
    assert!(
        (epsilon_one.energy_hartree - vacuum.energy_hartree).abs() < 1.0e-8,
        "eps=1 must reproduce the vacuum energy"
    );
}

#[test]
fn sv01_engine_emits_the_k8_solvation_block() {
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let mut input = JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "water",
        JobType::Sp,
        TaskOptions::default(),
    );
    input.options.scf_energy_threshold_hartree = 1.0e-9;
    input.options.scf_density_threshold = 1.0e-7;
    let job = backend.run(&input, &NullReporter).expect("cosmo run");
    let block = job.solvation.clone().expect("K8 solvation block");
    println!(
        "K8 block: model={} solvent={} total={:.9} el={:.9} A={:.3} V={:.3} M={} iters={} conv={}",
        block.model.as_str(),
        block.solvent_name,
        block.total_hartree,
        block.electrostatic_hartree,
        block.surface_area_angstrom2,
        block.surface_volume_angstrom3,
        block.n_tesserae,
        block.iterations,
        block.converged
    );
    block.validate().expect("K8 block must be self-consistent");
    assert!(block.converged, "K8 block must report convergence");
    assert_eq!(block.model.as_str(), "cosmo");
    assert_eq!(block.solvent_name, "water");
    // 分解对账：total = 静电 + 非极性（COSMO 没有非极性项）→ 差额必须为 0
    assert!(
        block.decomposition_gap_hartree().abs() < 1.0e-10,
        "decomposition gap {}",
        block.decomposition_gap_hartree()
    );
    assert!(block.n_tesserae > 300, "tesserae = {}", block.n_tesserae);

    // 与真空对照：`E_solv − E_vac` 必须等于块里的静电能
    let mut gas_input = input.clone();
    gas_input.solvent = "vacuum".into();
    let gas = backend.run(&gas_input, &NullReporter).expect("vacuum run");
    let delta = job.energy_hartree - gas.energy_hartree;
    println!(
        "E_solv - E_vac = {delta:.9} Ha = {:.3} kcal/mol (block el = {:.9})",
        delta * hartree_to_kcal(),
        block.electrostatic_hartree
    );
    assert!(
        (delta - block.electrostatic_hartree).abs() < 2.0e-3,
        "block electrostatic {} vs energy difference {delta}",
        block.electrostatic_hartree
    );
    // 与实验值（水，1 M 标准态）的差落在 SV-01 的判据内
    let measured = block.electrostatic_hartree * hartree_to_kcal();
    assert!(
        (measured - -6.32).abs() < 1.5,
        "water dG_solv {measured:.3} kcal/mol vs experiment -6.32"
    );
}

#[test]
fn sv01_solvated_gradients_are_rejected_until_sv02() {
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let input = JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "water",
        JobType::Opt,
        TaskOptions::default(),
    );
    let error = backend
        .run(&input, &NullReporter)
        .expect_err("solvated optimisation must be rejected");
    let message = error.to_string();
    println!("rejected: {message}");
    assert!(message.contains("SV-02"), "{message}");
    assert!(message.contains("sp"), "{message}");
}

#[test]
fn sv01_unknown_solvent_is_a_data_error() {
    let backend = RksBackend::new(Registry::open(data_root()).expect("qwb-data"));
    let input = JobInput::new(
        water(),
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "unobtainium",
        JobType::Sp,
        TaskOptions::default(),
    );
    let error = backend
        .run(&input, &NullReporter)
        .expect_err("unknown solvent must be rejected");
    let message = error.to_string();
    println!("rejected: {message}");
    assert!(message.contains("unobtainium"), "{message}");
}
