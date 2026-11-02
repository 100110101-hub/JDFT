//! SV-03 的**整条链**外部对照：同一几何、同一基组下，我们的 SMD 对 PySCF 的 SMD。
//!
//! 两个实现装的是同一个模型（SMD 参数化半径 + C-PCM 型表面方程 + eq. 4.77 的 `G_CDS`），
//! 但**表面表示不同**：PySCF 用 SWIG 解析表面，我们用 Fibonacci tesserae + 球内侧切换带。
//! 所以这条对照要分开看两半：
//!
//! * **`G_CDS` 应当对得很紧**（它只依赖 SASA 与溶剂描述符，两边算的是同一个式子）；
//! * **静电那一半会差千分之几 Ha** —— 这个差值**量出来并写进 `docs/error-budget.md`**，
//!   而不是用一个宽松容差把它盖住。
//!
//! 顺带钉住**标准态**：`total_hartree` 是 1 M ↔ 1 M 的转移能，与文献的 `ΔG*_S`
//! （1 atm 气体 → 1 M 溶液）差 `RT ln(RT/p°) = +1.8943 kcal/mol`。
use std::path::PathBuf;

use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{
    Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, SolventModel, TaskOptions,
};
use qwb_registry::Registry;

const HARTREE_PER_KCAL: f64 = 627.509_474_063_6;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/smd_water.csv")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

/// 与 `scripts/pyscf_smd_reference.py` / `examples/h2o_smd.yaml` 逐字一致的几何。
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

fn input(molecule: Molecule, model: SolventModel) -> JobInput {
    JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Normal,
        "water",
        JobType::Sp,
        TaskOptions::default().with_solvent_model(model),
    )
}

/// 从参考 CSV 取一个量（kcal/mol 或 hartree）。
fn reference(quantity: &str) -> f64 {
    let text = std::fs::read_to_string(reference_path()).expect("smd_water.csv");
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 || fields[4] != quantity {
            continue;
        }
        assert_eq!(fields[0], "water");
        assert_eq!(fields[1], "water");
        return fields[5].parse().expect("value");
    }
    panic!("quantity `{quantity}` not found in smd_water.csv");
}

#[test]
fn the_cds_term_matches_pyscfs_smd_closely_and_the_electrostatics_is_recorded() {
    let result = backend()
        .run(&input(water(), SolventModel::Smd), &NullReporter)
        .expect("SMD 单点");
    let block = result.solvation.expect("solvation block");
    assert_eq!(block.model, SolventModel::Smd);

    // ① G_CDS：两边算的是同一个式子，只差表面采样 → 应当 < 1e-2 kcal/mol。
    let ours_cds = block.cds_hartree.expect("cds") * HARTREE_PER_KCAL;
    let theirs_cds = reference("cds_kcal_per_mol");
    let cds_gap = (ours_cds - theirs_cds).abs();
    assert!(
        cds_gap < 1.0e-2,
        "G_CDS：我们 {ours_cds:.6} vs PySCF {theirs_cds:.6} kcal/mol（差 {cds_gap:.2e}）"
    );

    // ② 静电：表面表示不同（SWIG vs tesserae），差值是**实测记录**的量，
    //    判据先定成 2 kcal/mol（当前实测约 1.30），超出就说明有一边改了实现。
    let ours_el = block.electrostatic_hartree * HARTREE_PER_KCAL;
    let theirs_el = reference("electrostatic_kcal_per_mol");
    let el_gap = (ours_el - theirs_el).abs();
    assert!(
        el_gap < 2.0,
        "静电：我们 {ours_el:.4} vs PySCF {theirs_el:.4} kcal/mol（差 {el_gap:.4}）"
    );

    // ③ 总溶剂化能（1 M ↔ 1 M）：两半的差在这里合计。
    let ours_total = block.total_hartree * HARTREE_PER_KCAL;
    let theirs_total = reference("total_kcal_per_mol");
    let total_gap = (ours_total - theirs_total).abs();
    println!(
        "SMD/water：G_CDS 我们 {ours_cds:.6} / PySCF {theirs_cds:.6}（差 {cds_gap:.2e}）；\
         静电 {ours_el:.4} / {theirs_el:.4}（差 {el_gap:.4}）；\
         总 {ours_total:.4} / {theirs_total:.4} kcal/mol（差 {total_gap:.4}）"
    );
    assert!(total_gap < 2.0, "总溶剂化能差 {total_gap:.4} kcal/mol");

    // ④ 标准态：1 M ↔ 1 M 与文献的 ΔG*_S 差 +1.8943 kcal/mol（eq. 4.64–4.65）。
    let standard = block.standard_state_hartree() * HARTREE_PER_KCAL;
    assert!(
        (standard - 1.8943).abs() < 1.0e-3,
        "标准态项 {standard:.4} kcal/mol"
    );
    assert!(
        ((block.delta_g_solv_1atm_hartree() - block.total_hartree) * HARTREE_PER_KCAL - standard)
            .abs()
            < 1e-12
    );

    // ⑤ 对账：total = 静电 + CDS（契约恒等式，与上面两半的对照互不相干）。
    assert!(block.decomposition_gap_hartree().abs() < 1e-12);
}

#[test]
fn gas_and_smd_energies_come_from_the_same_scf_path() {
    // 参考 CSV 里的气相能量必须与我们的气相 SCF 一致（否则「差」里混进了两个不同的 SCF）。
    let result = backend()
        .run(
            &JobInput::new(
                water(),
                "sto-3g",
                "hf",
                GridLevel::Normal,
                "vacuum",
                JobType::Sp,
                TaskOptions::default(),
            ),
            &NullReporter,
        )
        .expect("气相单点");
    let reference_gas = reference("energy_gas_hartree");
    // 判据 1e-6 Ha 而不是 1e-10：两边 SCF 的收敛阈值不同（我们 1e-8 的能量 / 1e-6 的密度，
    // PySCF `conv_tol=1e-11`），实测差 2.0e-7 Ha —— 这是收敛水平的差，不是两个不同的 SCF。
    assert!(
        (result.energy_hartree - reference_gas).abs() < 1.0e-6,
        "气相能量 {:.10} vs PySCF {reference_gas:.10}",
        result.energy_hartree
    );
}
