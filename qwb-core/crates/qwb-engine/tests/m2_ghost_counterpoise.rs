//! M2 / VF-02 的前置能力：**幽灵原子**（counterpoise / BSSE）。
//!
//! 幽灵原子只提供基函数、不提供核电荷与电子。验收方式是把 PySCF 的 counterpoise 当 oracle：
//! 水二聚体的单体"在二聚体基组里"（伙伴换成 ghost）的能量必须与 PySCF 一致。
//!
//! 几何是固定的测试几何（不是 S22 的平衡构象）—— 这里验的是 ghost 的**语义**，不是化学。
use qwb_engine::RksBackend;
use qwb_engine_api::reporter::NullReporter;
use qwb_engine_api::{Atom, EngineBackend, GridLevel, JobInput, JobType, Molecule, TaskOptions};
use qwb_registry::Registry;
use std::path::PathBuf;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn backend() -> RksBackend {
    RksBackend::new(Registry::open(data_root()).expect("qwb-data"))
}

const WATER_A: [(&str, [f64; 3]); 3] = [
    ("O", [-1.551_0, 0.0, 0.0]),
    ("H", [-1.100_0, 0.0, 0.900_0]),
    ("H", [-2.400_0, 0.500_0, 0.300_0]),
];
const WATER_B: [(&str, [f64; 3]); 3] = [
    ("O", [1.300_0, 0.0, 2.000_0]),
    ("H", [1.900_0, 0.0, 2.600_0]),
    ("H", [1.600_0, 0.800_0, 1.500_0]),
];

fn input(molecule: Molecule, jobtype: JobType, options: TaskOptions) -> JobInput {
    JobInput::new(
        molecule,
        "sto-3g",
        "hf",
        GridLevel::Coarse,
        "vacuum",
        jobtype,
        options,
    )
}

fn water_dimer() -> Molecule {
    let mut atoms = Vec::new();
    for (symbol, position) in WATER_A.iter().chain(WATER_B.iter()) {
        atoms.push(Atom::new(*symbol, *position));
    }
    Molecule::new(atoms, 0, 1)
}

/// 单体 A + 单体 B 的 **ghost** 骨架（把 B 的三个原子换成幽灵）。
fn monomer_a_in_dimer_basis() -> Molecule {
    let mut atoms: Vec<Atom> = WATER_A
        .iter()
        .map(|(symbol, position)| Atom::new(*symbol, *position))
        .collect();
    for (symbol, position) in WATER_B.iter() {
        atoms.push(Atom::new_ghost(*symbol, *position));
    }
    Molecule::new(atoms, 0, 1)
}

fn energy(molecule: Molecule) -> f64 {
    let result = backend()
        .run(
            &input(molecule, JobType::Sp, TaskOptions::default()),
            &NullReporter,
        )
        .expect("sp");
    result.energy_hartree
}

#[test]
fn a_ghost_atom_is_the_counterpoise_reference() {
    // PySCF 2.14.0（STO-3G，同一几何）：二聚体 −149.9001711305、
    // 单体 A 在 AB 基组里（B 为 ghost）−74.9634869714、单体 A（无 ghost）−74.9634637933。
    let dimer = energy(water_dimer());
    let with_ghosts = energy(monomer_a_in_dimer_basis());
    let free = energy(Molecule::new(
        WATER_A
            .iter()
            .map(|(symbol, position)| Atom::new(*symbol, *position))
            .collect(),
        0,
        1,
    ));
    println!("dimer {dimer:.10}, A in AB basis {with_ghosts:.10}, A free {free:.10}");
    assert!(
        (dimer - (-149.900_171_130_5)).abs() < 1.0e-6,
        "二聚体能量 {dimer:.10} 与 PySCF 的 −149.9001711305 不符"
    );
    assert!(
        (with_ghosts - (-74.963_486_971_4)).abs() < 1.0e-6,
        "ghost 单体的能量 {with_ghosts:.10} 与 PySCF 的 −74.9634869714 不符"
    );
    assert!(
        (free - (-74.963_463_793_3)).abs() < 1.0e-6,
        "无 ghost 的单体能量 {free:.10} 与 PySCF 的 −74.9634637933 不符"
    );
    // BSSE 的量级：ghost 让单体能量**降低**（基组变大 → 变分更低）。
    let bsse = free - with_ghosts;
    println!("BSSE(A) = {bsse:.3e} Ha");
    assert!(bsse > 0.0, "ghost 基组下单体能量必须更低（变分原理）");
    assert!(bsse < 1.0e-3, "STO-3G 上的 BSSE 应该在 1e-4 量级");
}

#[test]
fn ghosts_are_refused_outside_the_audited_surface() {
    // 只审过"单点能量路径"：优化/频率、色散、ECP、溶剂都必须显式拒绝。
    let optimized = backend().run(
        &input(
            monomer_a_in_dimer_basis(),
            JobType::Opt,
            TaskOptions::default(),
        ),
        &NullReporter,
    );
    let message = format!("{}", optimized.expect_err("opt + ghosts must be refused"));
    println!("opt + ghosts -> {message}");
    assert!(message.contains("counterpoise"), "{message}");
    assert!(message.contains("sp"), "{message}");
}

#[test]
fn a_molecule_of_only_ghosts_is_refused() {
    let molecule = Molecule::new(
        WATER_A
            .iter()
            .map(|(symbol, position)| Atom::new_ghost(*symbol, *position))
            .collect(),
        0,
        1,
    );
    let error = molecule
        .validate()
        .expect_err("all-ghost molecule has no electrons");
    println!("all ghosts -> {error}");
    assert!(molecule.has_ghosts());
    assert_eq!(molecule.n_real_atoms(), 0);
}
