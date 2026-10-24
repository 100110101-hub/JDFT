//! D-12 的前置验收：**hybrid 的 HF 交换混入**（`α·(−½K)` 进 Fock、`α·(−¼Tr(PK))` 进能量）。
//!
//! D-12 本身是 B3LYP（还要 LYP，等 ∇²ρ），但它的机制骨架就是「按比例混入精确交换」。
//! 这里用 **PBE0**（75% PBE 交换 + 25% 精确交换 + 100% PBE 相关）做验证，因为它是同一个
//! 机制、且只依赖已经实现的东西。三条判据都不需要外部参照：
//!
//! 1. **α = 0 必须逐位退化为 PBE**（混入项恒零时不改变任何数）；
//! 2. **α = 1 且 DFT 交换权重 = 0 必须逐位退化为 HF**（这是最强的一条：它同时验证
//!    Fock 里的 `−½K`、能量里的 `−¼Tr(PK)` 两个系数，以及 `P = 2Σccᵀ` 的自旋求和约定）；
//! 3. **中间值**：PBE0 必须收敛，且能量能被独立分解复算（Tr(PH) + ½Tr(PJ) + E_xc + α(−¼Tr(PK)) + E_nuc）。
//!
//! 第 2 条尤其值得留：HF 分支走的是 `build_fock` 的 `H + J − ½K`，而 hybrid 走的是
//! `H + J + v_xc + α(−½K)` 加能量项 —— 两条**完全不同的代码路径**给出逐位相同的能量，
//! 才算把系数钉住了。
#![allow(clippy::field_reassign_with_default)]
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

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

fn terms_from(
    components: &[(&str, f64)],
    parameters: &BTreeMap<String, f64>,
) -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for (id, weight) in components {
        if *id == "HfExchange" {
            continue;
        }
        let term = qwb_xc::build_term(id, parameters).expect(id);
        terms.push(qwb_xc::scale_term(term, *weight));
    }
    terms
}

fn solve(
    registry: &Registry,
    components: &[(&str, f64)],
    hf_exchange: f64,
) -> qwb_scf::rks::RksResult {
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let parameters = registry
        .get_functional("pbe")
        .expect("pbe")
        .parameters
        .clone();
    let terms = terms_from(components, &parameters);
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.hf_exchange = hf_exchange;
    rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf")
}

#[test]
fn zero_mixing_reproduces_pbe() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let from_data = {
        let molecule = water();
        let mut options = RksOptions::default();
        options.max_iter = 400;
        options.energy_threshold = 1.0e-10;
        qwb_scf::rks::rks_from_data(
            &registry,
            &molecule,
            "sto-3g",
            "pbe",
            "grid-coarse",
            options,
        )
        .expect("pbe")
    };
    let manual = solve(&registry, &[("GgaPbeX", 1.0), ("GgaPbeC", 1.0)], 0.0);
    println!(
        "PBE（数据路径）E = {:.12}，PBE（α=0 手搭）E = {:.12}",
        from_data.energy_hartree, manual.energy_hartree
    );
    assert_eq!(
        from_data.energy_hartree.to_bits(),
        manual.energy_hartree.to_bits(),
        "α=0 时 hybrid 路径与纯 PBE 应逐位相同"
    );
}

#[test]
fn full_mixing_with_zero_dft_exchange_reproduces_hf() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    // 纯 HF：terms 为空（走 build_fock 的 H + J − ½K 分支）
    let hf = solve(&registry, &[], 0.0);
    // hybrid 路径：DFT 交换权重 0、α = 1（走 H + J + v_xc + α(−½K) 分支）
    let mixed = solve(&registry, &[("GgaPbeX", 0.0), ("GgaPbeC", 0.0)], 1.0);
    println!(
        "HF（terms 为空）E = {:.12}，α=1 + 零 DFT 交换 E = {:.12}，差 {:.2e}",
        hf.energy_hartree,
        mixed.energy_hartree,
        (hf.energy_hartree - mixed.energy_hartree).abs()
    );
    // 两条路径的 Fock/能量系数不同，只可能在一阶意义上相等，所以容差取 SCF 阈值量级
    assert!(
        (hf.energy_hartree - mixed.energy_hartree).abs() < 1.0e-8,
        "α=1 应退化为 HF：{} vs {}",
        hf.energy_hartree,
        mixed.energy_hartree
    );
}

#[test]
fn pbe0_converges_and_the_energy_decomposes() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    let pbe = qwb_scf::rks::rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "pbe",
        "grid-coarse",
        options.clone(),
    )
    .expect("pbe");
    let pbe0 = qwb_scf::rks::rks_from_data(
        &registry,
        &molecule,
        "sto-3g",
        "pbe0",
        "grid-coarse",
        options,
    )
    .expect("pbe0");
    let gap = |result: &qwb_scf::rks::RksResult| {
        let occupied = result.mo_energies_hartree.len() / 2;
        result.mo_energies_hartree.get(occupied) - result.mo_energies_hartree.get(occupied - 1)
    };
    println!(
        "PBE:  E = {:.8}，E_xc = {:.8}，gap = {:.4} Ha",
        pbe.energy_hartree,
        pbe.exchange_correlation_hartree,
        gap(&pbe)
    );
    println!(
        "PBE0: E = {:.8}，E_xc = {:.8}，gap = {:.4} Ha（iters = {}）",
        pbe0.energy_hartree,
        pbe0.exchange_correlation_hartree,
        gap(&pbe0),
        pbe0.iterations
    );
    assert!(pbe0.scf_converged, "PBE0 必须收敛");
    assert!(
        (pbe0.energy_hartree - pbe.energy_hartree).abs() > 1.0e-3,
        "混入 25% 精确交换后能量应当变化，实测 {:.8} vs {:.8}",
        pbe0.energy_hartree,
        pbe.energy_hartree
    );
    // 注意：STO-3G 上的 HOMO–LUMO 间隙不是可靠的定性指标（实测 PBE 0.2279、PBE0 0.2123，
    // 反而略小），所以这里**不**拿间隙当判据。判据换成「独立的能量分解」：
    //   E = Tr(PH) + ½Tr(PJ) + E_xc + α·(−¼Tr(PK)) + E_nuc
    // 用返回的密度重算右边每一项，与报告的 E 比。
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &pbe0.density).expect("fock");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let parameters = registry
        .get_functional("pbe")
        .expect("pbe")
        .parameters
        .clone();
    let terms = terms_from(&[("GgaPbeX", 0.75), ("GgaPbeC", 1.0)], &parameters);
    let (e_xc, _) = qwb_scf::rks::evaluate_xc(&basis, &grid, &pbe0.density, &terms).expect("xc");
    let n = pbe0.density.nrows();
    let mut tr_ph = 0.0;
    let mut tr_pj = 0.0;
    let mut tr_pk = 0.0;
    for i in 0..n {
        for j in 0..n {
            tr_ph += pbe0.density[(i, j)] * one.core_hamiltonian[(i, j)];
            tr_pj += pbe0.density[(i, j)] * fock.coulomb[(i, j)];
            tr_pk += pbe0.density[(i, j)] * fock.exchange[(i, j)];
        }
    }
    let reassembled =
        tr_ph + 0.5 * tr_pj + e_xc + 0.25 * (-0.25 * tr_pk) + pbe0.nuclear_repulsion_hartree;
    println!(
        "能量分解：Tr(PH) {tr_ph:.8} + ½Tr(PJ) {:.8} + E_xc {e_xc:.8} + α(−¼Tr(PK)) {:.8} + E_nuc {:.8} = {reassembled:.10}",
        0.5 * tr_pj,
        0.25 * (-0.25 * tr_pk),
        pbe0.nuclear_repulsion_hartree
    );
    println!("SCF 报告 E = {:.10}", pbe0.energy_hartree);
    assert!(
        (reassembled - pbe0.energy_hartree).abs() < 1.0e-8,
        "能量分解与 SCF 报告不符：{reassembled:.10} vs {:.10}",
        pbe0.energy_hartree
    );
    // 精确交换项必须是**负**贡献（交换能恒负），把符号写反直接挡掉
    assert!(
        0.25 * (-0.25 * tr_pk) < 0.0,
        "α·(−¼Tr(PK)) 应为负，实得 {:.6}",
        0.25 * (-0.25 * tr_pk)
    );
}
