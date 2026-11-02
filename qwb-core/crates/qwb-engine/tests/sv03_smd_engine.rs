//! SV-03 的**引擎级**验收：`solvent_model: smd` 真的跑通，并且 `G_CDS` 与 props 层逐位一致。
//!
//! 这一轮补的是**接线**，不是新物理：
//!
//! * 非极性项是**一项** `G_CDS`（Marenich–Cramer–Truhlar 2009 eq. 4.77），可分成
//!   `Σ_j σ_j A_j`（eq. 4.78）与 `σ^[M] Σ_j A_j`（eq. 4.80）两半 —— 契约模块头写清了
//!   这个订正（K8 原来按"空化/色散/排斥三项"设计，那是把 eq. 4.80 的四个**描述符系数**
//!   当成了四个能量项）；
//! * SMD 的空腔半径是 eq. 16 的参数化半径，与 COSMO 用的 Bondi 半径是两套 —— 所以这里
//!   专门有一条测试证明"换了半径"，否则 `solvent_model: smd` 可能悄悄沿用 COSMO 的键。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, EngineError, GridLevel, JobInput, JobType, Molecule, SolventModel,
    TaskOptions,
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

fn methane() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.09]),
            Atom::new("H", [1.02709, 0.0, -0.36333]),
            Atom::new("H", [-0.51355, 0.88949, -0.36333]),
            Atom::new("H", [-0.51355, -0.88949, -0.36333]),
        ],
        0,
        1,
    )
}

fn input(molecule: Molecule, solvent: &str, model: SolventModel, jobtype: JobType) -> JobInput {
    JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        solvent,
        jobtype,
        TaskOptions::default().with_solvent_model(model),
    )
}

/// 直接走 props 层重算 `G_CDS`（引擎用 4096 点/原子，见 `run_single_point`）。
fn props_cds(molecule: &Molecule, solvent: &str) -> (f64, f64, f64) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let data = registry.get_smd_atomic("smd-atomic").expect("smd-atomic");
    let parameters = registry
        .get_solvent(solvent)
        .expect("solvent")
        .smd
        .clone()
        .expect("smd block");
    let water = solvent == "water";
    // 水相的 γ/φ/ψ 在 MNSOL 里是 -1 占位（数据文件里是 null），CDS 走单独的 σ 表、
    // 不需要描述符 —— 这正是"六个缺一个就是 None"的用武之地（与引擎里的
    // `smd_descriptors_or_none` 同一套规则）。
    let descriptors = match (
        parameters.abraham_alpha,
        parameters.abraham_beta,
        parameters.phi_sq,
        parameters.gamma_cal_per_mol_angstrom2,
        parameters.psi_sq,
        parameters.refractive_index_20c,
    ) {
        (Some(alpha), Some(beta), Some(phi), Some(gamma), Some(psi), Some(n)) => Some(
            qwb_props::smd::SmdDescriptors::new(alpha, beta, phi, gamma, psi, n),
        ),
        _ => None,
    };
    let surface = qwb_props::smd_surface::smd_surface(data, molecule, 4096, 0.0).expect("surface");
    let sts =
        qwb_props::smd_sigma::smd_sts(data, molecule, water, descriptors.as_ref()).expect("sts");
    let cssigm = qwb_props::smd::cds_constant_term(data, water, descriptors.as_ref());
    let zeros = vec![0.0_f64; surface.areas_angstrom2.len()];
    let atomic =
        qwb_props::smd_sigma::cds_energy_kcal_from_sts(&sts, 0.0, &surface.areas_angstrom2)
            .expect("atomic");
    let molecular =
        qwb_props::smd_sigma::cds_energy_kcal_from_sts(&zeros, cssigm, &surface.areas_angstrom2)
            .expect("molecular");
    let to_hartree = 1.0 / 627.509_474_063_6;
    (
        atomic * to_hartree,
        molecular * to_hartree,
        surface.total_area_angstrom2,
    )
}

#[test]
fn smd_in_water_reports_one_nonpolar_term_and_it_adds_up() {
    let result = backend()
        .run(
            &input(water(), "water", SolventModel::Smd, JobType::Sp),
            &NullReporter,
        )
        .expect("SMD 单点");
    assert!(result.scf_converged);
    let block = result.solvation.expect("SMD 结果必须带 solvation 块");
    assert_eq!(block.model, SolventModel::Smd);
    assert_eq!(block.solvent_name, "water");
    assert!(block.converged, "表面电荷迭代要单独报收敛");

    // ① 契约的对账恒等式：total = 静电 + G_CDS（< 1e-12 Ha）。
    assert!(
        block.decomposition_gap_hartree().abs() < 1e-12,
        "total {:.12} != el {:.12} + cds {:.12}",
        block.total_hartree,
        block.electrostatic_hartree,
        block.nonpolar_sum_hartree()
    );
    // ② eq. 4.77 的两半加起来就是报出来的 CDS。
    assert!(block.cds_hartree.is_some(), "SMD 必须报 G_CDS");
    assert!(block.cds_gap_hartree().abs() < 1e-15);
    // ③ 水相没有分子项（走单独的 σ_water 表）。
    assert!(
        block.cds_molecular_hartree.expect("molecular").abs() < 1e-15,
        "水相的分子表面张力项应为 0"
    );
    assert!(
        block.cds_hartree.expect("cds") > 0.0,
        "空腔项应当为正（形成空腔要付出代价）"
    );

    // ④ 与 props 层的同一次计算一致（同 4096 点/原子）—— 引擎没有自己另算一套。
    let (atomic, molecular, total_area) = props_cds(&water(), "water");
    assert!(
        (block.cds_atomic_hartree.expect("atomic") - atomic).abs() < 1e-12,
        "atomic {:.15} vs props {atomic:.15}",
        block.cds_atomic_hartree.expect("atomic")
    );
    assert!((block.cds_molecular_hartree.expect("molecular") - molecular).abs() < 1e-12);
    assert!(total_area > 0.0);

    // ⑤ 溶剂化自由能 = G_EP + G_CDS，量级与实验（water: -6.32 kcal/mol）在 1 kcal 内。
    let kcal = block.total_hartree * 627.509_474_063_6;
    println!(
        "smd/water: dG_el = {:.4} kcal/mol, G_CDS = {:.4}, dG_solv = {kcal:.4} (experiment -6.32)",
        block.electrostatic_hartree * 627.509_474_063_6,
        block.cds_hartree.expect("cds") * 627.509_474_063_6
    );
    assert!(
        (kcal + 6.32).abs() < 1.0,
        "SMD/H2O 的 ΔG_solv = {kcal:.3} kcal/mol 与实验 -6.32 差得太多"
    );
}

#[test]
fn a_nonaqueous_smd_run_reports_the_molecular_part() {
    // 甲苯：γ = 40.2 cal/(mol·Å²)、φ = 0.857（芳香性），所以 σ^[M] = 0.35γ /1 − 4.19φ² ≈ 11.0
    // —— 分子那一半必须非零，这正是"四项系数不是四项能量"的现场证据。
    let result = backend()
        .run(
            &input(methane(), "toluene", SolventModel::Smd, JobType::Sp),
            &NullReporter,
        )
        .expect("SMD 在甲苯里也要跑通");
    let block = result.solvation.expect("solvation block");
    assert_eq!(block.model, SolventModel::Smd);
    assert!(
        block.cds_molecular_hartree.expect("molecular").abs() > 1e-6,
        "甲苯的分子表面张力项不该是 0：{:?}",
        block.cds_molecular_hartree
    );
    assert!(block.cds_gap_hartree().abs() < 1e-15);
    assert!(block.decomposition_gap_hartree().abs() < 1e-12);

    let (atomic, molecular, _area) = props_cds(&methane(), "toluene");
    assert!((block.cds_atomic_hartree.expect("atomic") - atomic).abs() < 1e-12);
    assert!((block.cds_molecular_hartree.expect("molecular") - molecular).abs() < 1e-12);
}

#[test]
fn smd_switches_the_cavity_to_the_parameterised_radii() {
    // 甲苯里的 C：Bondi 1.70 Å vs SMD 参数化 1.85 Å —— 若 SMD 悄悄沿用 COSMO 的键，
    // 两个面积会**逐位相同**。这条测试就是钉住"换了半径"。
    let cosmo = backend()
        .run(
            &input(methane(), "toluene", SolventModel::Cosmo, JobType::Sp),
            &NullReporter,
        )
        .expect("COSMO 单点");
    let smd = backend()
        .run(
            &input(methane(), "toluene", SolventModel::Smd, JobType::Sp),
            &NullReporter,
        )
        .expect("SMD 单点");
    let cosmo_block = cosmo.solvation.expect("cosmo block");
    let smd_block = smd.solvation.expect("smd block");
    println!(
        "area: cosmo {:.3} A^2 vs smd {:.3} A^2; dG_el: {:.6} vs {:.6} Ha",
        cosmo_block.surface_area_angstrom2,
        smd_block.surface_area_angstrom2,
        cosmo_block.electrostatic_hartree,
        smd_block.electrostatic_hartree
    );
    assert!(
        (cosmo_block.surface_area_angstrom2 - smd_block.surface_area_angstrom2).abs() > 1e-3,
        "SMD 与 COSMO 的空腔面积相同 —— 半径没换（CH4 在甲苯里：Bondi C 1.70 vs SMD 1.85）"
    );
    assert!(
        (cosmo_block.electrostatic_hartree - smd_block.electrostatic_hartree).abs() > 1e-6,
        "静电项也应随半径改变"
    );
}

#[test]
fn smd_with_a_gradient_job_is_refused_and_names_sv02() {
    let error = backend()
        .run(
            &input(water(), "water", SolventModel::Smd, JobType::Opt),
            &NullReporter,
        )
        .expect_err("带溶剂的 opt 还没接（SV-02）");
    match error {
        EngineError::Unsupported(message) => {
            assert!(message.contains("SV-02"), "{message}");
            assert!(
                message.contains("solvation") || message.contains("solvent"),
                "{message}"
            );
        }
        other => panic!("期望 Unsupported（退出码 4），得到 {other:?}"),
    }
}
