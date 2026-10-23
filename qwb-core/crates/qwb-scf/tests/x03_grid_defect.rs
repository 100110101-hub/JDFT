//! KNOWN DEFECT（记录用）：重原子（O 的 1s，ζ≈130）的分子 XC 网格求积**不收敛**。
//!
//! 这条测试把实测数字钉在这里，不让「水的 PBE/X 能量」被当成可用数。它不是验收判据
//! （当前行为是错的），所以 `#[ignore]`；修好后应**反过来**断言收敛（∫ρ = 10 到 1e-8、
//! 相邻等级 E_xc 差 < 0.1 mHa），并把这条改成验收测试。
//!
//! # 已测到的现象
//!
//! 1. **固定密度的电子数**：用 grid-fine 收敛出来的密度 P（`Tr(P·S) = 10` 由库算得），
//!    在四个等级上算 `∫ρ dr = Σ w ρ`：
//!
//!    | 网格 | 点数 | ∫ρ dr（应为 10） | E_xc(P) |
//!    | --- | --- | --- | --- |
//!    | grid-coarse | 29400 | 10.09549604 | −8.40655109 |
//!    | grid-normal | 72900 | 10.89471413 | −11.56723753 |
//!    | grid-fine | 185625 | 9.80184228 | −8.57632002 |
//!    | grid-veryfine | 430080 | 9.58830844 | −7.78903635 |
//!
//! 2. **不是径向分辨率**：把径向点数从 99 加到 150/200/300/400（固定密度），
//!    `∫ρ` 收敛到一个**错的值** 9.81762435、`E_xc` 收敛到 −8.64913575；200 与 400 逐位相同
//!    ⇒ 径向已经收敛，收敛值就是错的。
//!
//! 3. **也不是角向点数不够**：把角向阶从 590 加到 900/1400/2000/3200（n = ⌈√order⌉ 的
//!    Gauss–Legendre × 等距 φ 乘积），`∫ρ` 依次 9.8176 / 10.7685 / 10.1996 / 9.9794 / 10.0027，
//!    `E_xc` 依次 −8.65 / −13.06 / −10.41 / −9.41 / −9.55 —— 在 10 附近乱跳，不单调收敛。
//!
//! 4. **也不是 Becke 锐度**：`becke_sharpness` 从 0.2 扫到 1.6（迭代 1→8），`∫ρ` 只从
//!    9.7944 动到 9.8419，`E_xc` 从 −8.55 到 −8.76，同样不收敛。
//!
//! 5. **每一层单独看都没问题**：
//!    * 以 O 为中心的原子网格对 `e^{-αr²}`（α = 1…300）相对误差 ≤ 4e-14；
//!    * 分子网格（Becke 分区）对同一批高斯也准到 1e-9；
//!    * `ao_value` 与手写收缩的最大差 1.1e-16；
//!    * 以 O 为中心的原子网格单独积 ρ 得 **10.00000000**。
//!
//! 6. **H 为中心**的原子网格对「O 处的 α=130 高斯」差 **43%**（α=1 时 9e-15）。也就是说：
//!    每个原子网格只在自己中心附近准确，分子网格的正确性完全依赖「分区权重把别人紧核处的
//!    贡献压掉」——而 Becke 权重在 ν→1 处只是多项式衰减，压不住 ρ ~ 10⁴ 的核尖，于是
//!    H 网格在 O 核心处贡献出不该有的份额。
//!
//! # 结论与修法
//!
//! 这不是「加点数」能解决的问题，需要把 X-03 缺的三件事补齐：
//! ① 真正的 Lebedev 角向表（现在是 `n = ⌈√lebedev_order⌉` 的 Gauss 乘积，阶数语义名不副实）；
//! ② Becke 分区的**尺寸调整** `ν = μ + a_ij(1−μ²)`（`a_ij` 来自原子半径比，现在没有）；
//! ③ Treutler–Ahlrichs 一类带核心标度的径向映射（现在 `a = cutoff/8` 只有外层尺度）。
//!
//! 影响面：含重原子的 XC / PBE / LDA **绝对能量不可引用**。HF 不经过网格，所以
//! H2O/STO-3G 的 HF = −74.9629 Ha 与外部参照一致这一步仍然成立。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, ao_value, evaluate_xc, rks_scf};

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

/// 固定密度下的 `∫ρ dr` 与 `E_xc`。
fn probe(
    basis: &qwb_basis::InternalBasis,
    grid: &qwb_grid::Grid,
    density: &qwb_linalg::Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> (f64, f64) {
    let n = basis.n_functions();
    let (exc, _) = evaluate_xc(basis, grid, density, terms).expect("xc");
    let mut electrons = 0.0;
    for point in &grid.points {
        let ao = ao_value(basis, point.position_bohr).expect("ao");
        let mut rho = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                rho += density[(mu, nu)] * ao[mu] * ao[nu];
            }
        }
        electrons += point.weight * rho;
    }
    (electrons, exc)
}

#[test]
#[ignore = "KNOWN DEFECT: 水的 XC 网格不收敛——径向在 9.8176 处收敛到错值、角向在 10 附近乱跳、Becke 锐度无效。见 docs/error-budget.md 第 2 节与 docs/choices.md 第十六节。"]
fn water_xc_quadrature_is_not_convergent() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = water();
    let basis = qwb_basis::InternalBasis::build(basis_k1, &molecule).expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let functional = registry.get_functional("pbe").expect("pbe");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    let definition = registry.get_grid("grid-fine").expect("grid");

    // 参考密度：默认参数下 grid-fine 收敛出来的 P
    let base_grid = qwb_grid::molecular_grid(definition, &molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-11;
    let density = rks_scf(&basis, &molecule, &tensor, &base_grid, &terms, options)
        .expect("scf")
        .density;

    // ① 四个等级
    for name in ["grid-coarse", "grid-normal", "grid-fine", "grid-veryfine"] {
        let level = registry.get_grid(name).expect("grid");
        let grid = qwb_grid::molecular_grid(level, &molecule).expect("grid");
        let (electrons, exc) = probe(&basis, &grid, &density, &terms);
        println!("{name:<14} ∫ρ = {electrons:.8}  E_xc = {exc:.8}");
        assert!(
            (electrons - 10.0).abs() > 1.0e-2,
            "{name} 的 ∫ρ 已经准到 1e-2 以内（{electrons}）——这条 known-defect 记录该撤了"
        );
    }

    // ② 径向加密：收敛到错值
    for radial in [99_u32, 200, 400] {
        let mut level = definition.clone();
        level.radial_points = radial;
        let grid = qwb_grid::molecular_grid(&level, &molecule).expect("grid");
        let (electrons, exc) = probe(&basis, &grid, &density, &terms);
        println!("径向 {radial:<4} ∫ρ = {electrons:.8}  E_xc = {exc:.8}");
    }

    // ③ 角向加密：乱跳
    for order in [900_u32, 1400, 2000] {
        let mut level = definition.clone();
        level.radial_points = 200;
        level.lebedev_order = order;
        let grid = qwb_grid::molecular_grid(&level, &molecule).expect("grid");
        let (electrons, exc) = probe(&basis, &grid, &density, &terms);
        println!("角向 {order:<5} ∫ρ = {electrons:.8}  E_xc = {exc:.8}");
    }
}
