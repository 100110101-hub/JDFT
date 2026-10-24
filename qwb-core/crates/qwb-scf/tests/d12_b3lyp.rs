//! B3LYP 的可执行验证（**程序化构造**，不走数据层）。
//!
//! # 为什么没有 b3lyp.yaml
//!
//! B3LYP 的四个组件里 LYP 依赖 `∇²ρ`，而 K1 的数据语义门禁（冻结的 phase-1 范围 §2.2）
//! 明确拒绝 `requires.laplacian = true` 的泛函（"no meta-GGA"）。所以：
//!
//! * **代码层**支持 B3LYP（本测试证明）；
//! * **数据层**暂时不允许注册它（`qwb data verify` 会报
//!   `requires.laplacian = true is out of first-phase scope`）。
//!
//! 这不是缺陷、也不是绕过：`data verify` 的拒绝是**冻结语义**的一部分，改动它要人类拍板。
//! 等 phase-2 放开 meta-GGA 后，只需把这里的组件按 0.08/0.72/0.2/0.81/0.19 写进 YAML。
//!
//! # 判据
//!
//! 1. **能量分解自洽**：`E = Tr(PH) + ½Tr(PJ) + E_xc + α(−¼Tr(PK)) + E_nuc`，其中
//!    `E_xc` 由四个加权组件给出（0.8 B88 + 0.81 LYP + 0.19 VWN5），复算与 SCF 报告一致；
//! 2. **权重生效**：把四个权重按同一比例缩放（×0.5）后，`E_xc` 必须同比缩放（线性性）；
//! 3. **收敛**：B3LYP 在 H2O/STO-3G 上收敛，且能量与 PBE、PBE0 都不同（混入确有作用）。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, RksResult, evaluate_xc, rks_scf};
use qwb_xc::FunctionalTerm;

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

fn parameters() -> BTreeMap<String, f64> {
    let mut p = BTreeMap::new();
    p.insert("slater_alpha".to_string(), 2.0 / 3.0);
    p.insert("b88_beta".to_string(), 0.0042);
    p.insert("lyp_a".to_string(), 0.049_18);
    p.insert("lyp_b".to_string(), 0.132);
    p.insert("lyp_c".to_string(), 0.253_3);
    p.insert("lyp_d".to_string(), 0.349);
    p.insert("vwn5_p_A".to_string(), 0.0310907);
    p.insert("vwn5_p_x0".to_string(), -0.409286);
    p.insert("vwn5_p_b".to_string(), 13.072);
    p.insert("vwn5_p_c".to_string(), 42.7198);
    p
}

/// B3LYP 的四个组件（scale = 1.0 是标准权重）。
fn b3lyp_terms(scale: f64) -> Vec<Box<dyn FunctionalTerm>> {
    let p = parameters();
    vec![
        qwb_xc::scale_term(
            qwb_xc::build_term("LdaSlaterX", &p).expect("slater"),
            0.08 * scale,
        ),
        qwb_xc::scale_term(
            qwb_xc::build_term("GgaB88X", &p).expect("b88"),
            0.72 * scale,
        ),
        qwb_xc::scale_term(
            qwb_xc::build_term("GgaLypC", &p).expect("lyp"),
            0.81 * scale,
        ),
        qwb_xc::scale_term(
            qwb_xc::build_term("LdaSvwn5", &p).expect("vwn5"),
            0.19 * scale,
        ),
    ]
}

fn solve(scale: f64) -> (RksResult, qwb_basis::InternalBasis, qwb_grid::Grid) {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let terms = b3lyp_terms(scale);
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-10;
    options.hf_exchange = 0.2 * scale;
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let mut result = rks_scf(&basis, &molecule, &tensor, &grid, &terms, options).expect("scf");
    // 保存 core Hamiltonian 供分解校验：用 Tr(PH) 从 one 重算，所以这里只需要返回基组与网格
    let _ = one;
    result.exchange_correlation_hartree *= 1.0;
    (result, basis, grid)
}

/// **曾经的失败（已修复）**：B3LYP 的 SCF 在 grid-coarse 上 3 步内就以
/// `density laplacian is not finite (NaN)` 中止。已排除的：
///
/// - LYP 项本身在三通道上都有限（用「类 SCF」的对角密度逐点扫过 29400 个点，非有限点 0 个，
///   σ 最大 6.8e5）；
/// - `evaluate_xc` 的能量与势在给定密度上满足变分自洽（`lyp_variational.rs`：残差随扰动
///   线性下降 3.4e-9 → 3.4e-13）。
///
/// 剩下两个嫌疑：① 网格太粗（29k 点的近核区把 ρ 与 ∇²ρ 都放大了）；② `evaluate_xc` 里
/// sigma/laplacian 通道用**未截断**的 ρ 去乘、而泛函内部用 `LOW_DENSITY_FLOOR` 截断过的 ρ，
/// 极小密度处两者不一致（LYP 的 ρ 依赖比 PBE 陡，放大了这个不一致）。
///
/// **真正的根因**（下一步排查发现的）：B88 用 sqrt(σ)，对偶数在 σ = 0 处导数发散，
/// x·x 的导数变成 2·0·∞ = NaN；远处网格点的 AO 梯度下溢成精确 0 时就会命中。
/// 改成用 t = σ/ρ^{8/3}（对 σ 线性）承载对偶数、A(x)=x·asinh x 用解析导数之后，
/// B3LYP 收敛到 −75.2933510460 Ha（21 步）。回归钉在 qwb-xc/tests/d11_b88_sigma_zero.rs。
#[test]
fn b3lyp_energy_decomposes_and_weights_are_linear() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = water();
    let (result, basis, grid) = solve(1.0);
    println!(
        "B3LYP/STO-3G: E = {:.10} Ha, E_xc = {:.10}, iters = {}, converged = {}",
        result.energy_hartree,
        result.exchange_correlation_hartree,
        result.iterations,
        result.scf_converged
    );
    assert!(result.scf_converged, "B3LYP 必须收敛");

    // ① 能量分解：E = Tr(PH) + ½Tr(PJ) + E_xc + α(−¼Tr(PK)) + E_nuc
    let one = qwb_integrals::one_electron::one_electron(&basis, &molecule).expect("1e");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &result.density).expect("fock");
    let (e_xc, _) = evaluate_xc(&basis, &grid, &result.density, &b3lyp_terms(1.0)).expect("xc");
    let n = result.density.nrows();
    let mut tr_ph = 0.0;
    let mut tr_pj = 0.0;
    let mut tr_pk = 0.0;
    for i in 0..n {
        for j in 0..n {
            tr_ph += result.density[(i, j)] * one.core_hamiltonian[(i, j)];
            tr_pj += result.density[(i, j)] * fock.coulomb[(i, j)];
            tr_pk += result.density[(i, j)] * fock.exchange[(i, j)];
        }
    }
    let reassembled =
        tr_ph + 0.5 * tr_pj + e_xc + 0.2 * (-0.25 * tr_pk) + result.nuclear_repulsion_hartree;
    println!(
        "分解：Tr(PH) {tr_ph:.8} + ½Tr(PJ) {:.8} + E_xc {e_xc:.8} + 0.2(−¼Tr(PK)) {:.8} + E_nuc {:.8} = {reassembled:.10}",
        0.5 * tr_pj,
        0.2 * (-0.25 * tr_pk),
        result.nuclear_repulsion_hartree
    );
    println!("SCF 报告 E = {:.10}", result.energy_hartree);
    assert!(
        (reassembled - result.energy_hartree).abs() < 1.0e-8,
        "B3LYP 能量分解不符：{reassembled:.10} vs {:.10}",
        result.energy_hartree
    );

    // ② 权重线性：同一密度下把四个权重全部 ×0.5，E_xc 必须正好减半
    let (half_xc, _) = evaluate_xc(&basis, &grid, &result.density, &b3lyp_terms(0.5)).expect("xc");
    println!("E_xc(scale=1) = {e_xc:.10}, E_xc(scale=0.5) = {half_xc:.10}");
    assert!(
        (half_xc - 0.5 * e_xc).abs() < 1.0e-10,
        "权重没有线性生效：{half_xc} vs {:.10}",
        0.5 * e_xc
    );

    // ③ 与 PBE / PBE0 都不同（混入确有作用）
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
    println!(
        "对比：PBE {:.8}，PBE0 {:.8}，B3LYP {:.8}",
        pbe.energy_hartree, pbe0.energy_hartree, result.energy_hartree
    );
    assert!(
        (result.energy_hartree - pbe.energy_hartree).abs() > 1.0e-3
            && (result.energy_hartree - pbe0.energy_hartree).abs() > 1.0e-3,
        "B3LYP 应与 PBE、PBE0 都不同"
    );
}
