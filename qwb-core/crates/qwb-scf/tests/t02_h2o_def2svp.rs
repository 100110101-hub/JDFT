//! T2 验收：H2O / def2-SVP / PBE 单点能 + 性质 + 网格收敛 + 线程确定性。
//!
//! # 判定状态
//!
//! T2 规定「能量与 `tests/reference/h2o_def2svp_pbe.csv` 差 < 1 mHa」。**五条判据全部通过**
//! （2026-10-26）：最后那条"外部参考"由 **PySCF 2.14.0** 补齐 —— 同一几何/模型化学下
//! PySCF 给 `-76.271970822413 Ha`，本项目 grid-normal 是 `-76.2719421662`，
//! 差 **2.87e-5 Ha**（判据 1 mHa）。复现：`python3 scripts/pyscf_reference.py --grid-level 3`
//! （WSL 里的 pyscf wheel；Windows 侧 Python 3.12 没有 wheel，那条路一直走不通）。
//!
//! 决策记录：PySCF 是 §14 #12 允许的**黑盒对照**（只产数字，不拷代码/数据）；
//! def2-SVP 的指数/系数仍来自 BSE 导出，由 D-04 逐项 diff。
//!
//! 其余四条本来就是真判据：
//!
//! | 判据 | 实测 |
//! | --- | --- |
//! | 相邻网格等级能量差 | coarse vs normal 9.4e-5 Ha（< 0.1 mHa） |
//! | 1 线程 vs 4 线程 | 能量与密度指纹**逐位相同** |
//! | 偶极 vs 实验 1.85 D | 1.9544 D（差 0.104 D） |
//! | Tr(P·S) = 电子数 | 10.000000 |
#![allow(clippy::field_reassign_with_default)] // options 逐字段覆盖最清楚
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, RksResult, rks_scf};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
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

struct Outcome {
    result: RksResult,
    overlap: Matrix,
    dipole_debye: [f64; 3],
}

fn solve(grid_name: &str, threads: usize) -> Outcome {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(
        registry.get_basis("def2-svp").expect("def2-svp"),
        &molecule,
    )
    .expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron_with_threads(&basis, &molecule, threads)
        .expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name).expect("grid"), &molecule)
        .expect("grid");
    let functional = registry.get_functional("pbe").expect("pbe");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-9;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");

    // μ_α = Σ_μν P_μν ⟨μ|r_α|ν⟩ − Σ_A Z_A R_A,α，再乘 2.5417463 Debye/a.u.
    let integrals = qwb_integrals::dipole::dipole_integrals(&basis, &molecule).expect("dipole");
    let n = basis.n_functions();
    let mut dipole = [0.0_f64; 3];
    for (axis, slot) in dipole.iter_mut().enumerate() {
        for mu in 0..n {
            for nu in 0..n {
                *slot += result.density[(mu, nu)] * integrals[axis][(mu, nu)];
            }
        }
        for atom in &molecule.atoms {
            let z =
                f64::from(qwb_registry::elements::atomic_number(&atom.symbol).expect("element"));
            *slot -= z * atom.position_bohr()[axis];
        }
        *slot *= 2.541_746_3;
    }
    Outcome {
        result,
        overlap: one.overlap,
        dipole_debye: dipole,
    }
}

fn fingerprint(density: &Matrix) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in density.as_column_major() {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[test]
fn h2o_def2svp_pbe_single_point() {
    // 门禁版只用 grid-coarse 跑**一次** SCF（def2-SVP 每次 SCF 要几十秒到几分钟，
    // 全等级收敛与线程比较放在下面的 #[ignore] 测试里，数字见文件头）。
    let outcome = solve("grid-coarse", 0);
    println!(
        "H2O/def2-SVP/PBE: E = {:.10} Ha, E_xc = {:.10}, iters = {}",
        outcome.result.energy_hartree,
        outcome.result.exchange_correlation_hartree,
        outcome.result.iterations
    );
    assert!(outcome.result.scf_converged, "SCF 必须收敛");

    // ① 参考值（external：PySCF 2.14.0 的独立计算，见 csv 的来源标注）
    let reference_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference/h2o_def2svp_pbe.csv");
    let text = std::fs::read_to_string(&reference_path).expect("reference csv");
    let mut total_reference = None;
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields[0] == "pbe_total_energy_coarse" {
            total_reference = Some((
                fields[1].parse::<f64>().expect("value"),
                fields[2].parse::<f64>().expect("tolerance"),
            ));
        }
    }
    let (expected, tolerance) = total_reference.expect("pbe_total_energy_coarse 行");
    let difference = (outcome.result.energy_hartree - expected).abs();
    println!("参考 {expected} ± {tolerance}，差 {difference:.2e}");
    assert!(
        difference < tolerance,
        "T2 能量 {:.10} 与参考 {expected} 差 {difference:.2e}（容差 {tolerance}）",
        outcome.result.energy_hartree
    );

    // ② 偶极 vs 实验 1.85 D
    let magnitude = (outcome.dipole_debye[0].powi(2)
        + outcome.dipole_debye[1].powi(2)
        + outcome.dipole_debye[2].powi(2))
    .sqrt();
    println!("|μ| = {magnitude:.4} D（实验 1.85 D）");
    assert!(
        (magnitude - 1.85).abs() < 0.15,
        "偶极 {magnitude:.4} D 与实验 1.85 D 差得太多"
    );

    // ③ 电荷守恒：Tr(P·S) = 10
    let n = outcome.result.density.nrows();
    let mut electrons = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            electrons += outcome.result.density[(mu, nu)] * outcome.overlap[(mu, nu)];
        }
    }
    println!("Tr(P·S) = {electrons:.8}");
    assert!(
        (electrons - 10.0).abs() < 1.0e-6,
        "Tr(P·S) = {electrons} 应等于 10"
    );
}

/// 昂贵版本（约 16 分钟）：四个 def2-SVP SCF。数字已实测：
/// coarse −76.2720362190、normal −76.2719421662，差 9.41e-5 Ha（< 0.1 mHa）；
/// 1 线程与 4 线程的能量与密度指纹逐位相同（`129364e77400c84f`）。
#[test]
#[ignore = "耗时约 16 分钟（四个 def2-SVP SCF）；数字已实测并写在文档注释与 status.yaml 的 T2 里"]
fn h2o_def2svp_pbe_grid_and_threads() {
    let coarse = solve("grid-coarse", 0);
    let normal = solve("grid-normal", 0);
    let difference = (coarse.result.energy_hartree - normal.result.energy_hartree).abs();
    println!(
        "coarse {:.10}，normal {:.10}，差 {difference:.2e} Ha",
        coarse.result.energy_hartree, normal.result.energy_hartree
    );
    assert!(
        difference < 1.0e-4,
        "相邻网格等级差 {difference:.2e} Ha（要求 < 0.1 mHa）"
    );

    // 1 线程 vs 4 线程：能量与密度必须逐位相同（G7）
    let single = solve("grid-coarse", 1);
    let quad = solve("grid-coarse", 4);
    assert_eq!(
        single.result.energy_hartree.to_bits(),
        quad.result.energy_hartree.to_bits(),
        "1 线程与 4 线程的能量不是逐位相同"
    );
    assert_eq!(
        fingerprint(&single.result.density),
        fingerprint(&quad.result.density),
        "1 线程与 4 线程的密度不是逐位相同"
    );
    println!("1 vs 4 线程：能量与密度指纹逐位相同");
}
