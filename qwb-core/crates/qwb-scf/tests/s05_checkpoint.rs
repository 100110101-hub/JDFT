//! S-05 验收：从检查点续算，结果与不中断的一次跑完相同。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, EngineError, Molecule};
use qwb_integrals::two_electron::two_electron;
use qwb_registry::Registry;
use qwb_scf::checkpoint::{load_checkpoint, save_checkpoint};
use qwb_scf::rks::{RksOptions, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.37]),
            Atom::new("H", [0.0, 0.0, 0.37]),
        ],
        0,
        1,
    )
}

#[test]
fn resume_from_checkpoint_reproduces_the_uninterrupted_result() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");

    // ① 一次跑完
    let mut complete = RksOptions::default();
    complete.max_iter = 400;
    let full = rks_scf(&basis, &molecule, &tensor, &grid, &[], complete).expect("full");

    // ② 只跑 3 步 → 不收敛，但检查点必须落盘
    let path = std::env::temp_dir().join(format!("qwb_scf_cp_{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let mut partial = RksOptions::default();
    partial.max_iter = 1;
    partial.checkpoint_path = Some(path.clone());
    partial.checkpoint_every = 1;
    let error =
        rks_scf(&basis, &molecule, &tensor, &grid, &[], partial).expect_err("3 步不可能收敛");
    assert!(
        matches!(error, EngineError::ScfNotConverged { .. }),
        "实得 {error}"
    );
    assert!(path.is_file(), "不收敛时必须落盘检查点");
    let checkpoint = load_checkpoint(&path).expect("load");
    println!(
        "检查点：iterations = {}，energy = {:?}",
        checkpoint.iterations, checkpoint.energy_hartree
    );
    assert_eq!(checkpoint.iterations, 1);

    // ③ 从检查点续算
    let mut resumed_options = RksOptions::default();
    resumed_options.max_iter = 400;
    resumed_options.resume_from = Some(checkpoint);
    let resumed =
        rks_scf(&basis, &molecule, &tensor, &grid, &[], resumed_options).expect("resumed");
    println!(
        "完整跑 E = {:.12}  续算 E = {:.12}  差 {:.3e}",
        full.energy_hartree,
        resumed.energy_hartree,
        (full.energy_hartree - resumed.energy_hartree).abs()
    );
    // 恢复 DIIS 子空间后，续算必须与不中断的跑法**逐位相同**（验收原文是 identical）。
    assert_eq!(
        full.energy_hartree.to_bits(),
        resumed.energy_hartree.to_bits(),
        "续算能量与完整跑不逐位相同：{} vs {}",
        full.energy_hartree,
        resumed.energy_hartree
    );
    assert_eq!(full.iterations, resumed.iterations, "迭代步数也应相同");
    assert_eq!(
        full.density
            .as_column_major()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        resumed
            .density
            .as_column_major()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "密度矩阵也应逐位相同"
    );
    assert!(resumed.scf_converged);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_round_trips_through_json() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = h2();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");
    let result = rks_scf(
        &basis,
        &molecule,
        &tensor,
        &grid,
        &[],
        RksOptions::default(),
    )
    .expect("scf");

    let path = std::env::temp_dir().join(format!("qwb_scf_rt_{}.json", std::process::id()));
    save_checkpoint(&path, &result.checkpoint).expect("save");
    let loaded = load_checkpoint(&path).expect("load");
    // serde_json 的 pretty 打印可能丢掉最后 1 ulp，所以能量用相对判据、密度要求逐位相同。
    assert_eq!(loaded.n_functions, result.checkpoint.n_functions);
    assert_eq!(loaded.iterations, result.checkpoint.iterations);
    assert_eq!(
        loaded.density, result.checkpoint.density,
        "密度必须逐位相同"
    );
    let (a_e, b_e) = (
        loaded.energy_hartree.expect("loaded"),
        result.checkpoint.energy_hartree.expect("orig"),
    );
    assert!(
        (a_e - b_e).abs() < 1.0e-15,
        "能量往返差 {:.3e}",
        (a_e - b_e).abs()
    );
    // 密度矩阵也能还原，且逐位相同
    let a = result.checkpoint.density_matrix().expect("a");
    let b = loaded.density_matrix().expect("b");
    assert_eq!(a.as_column_major(), b.as_column_major());
    let _ = std::fs::remove_file(&path);
}
