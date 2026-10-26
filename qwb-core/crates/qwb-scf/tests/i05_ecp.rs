//! I-05 验收（project.md §6.2）：ECP 矩阵元（局域 + 非局域）与数值对照 < 1e-8；
//! Cu 的 d 通道单独断言。
//!
//! # 两条实现，互相验证
//!
//! * [`qwb_scf::ecp::one_center_matrix`]：**解析**。球对称算符在一中心时只有
//!   `(l, m)` 相同的元非零，径向积分是 `∫ r^m e^{−γr²} dr` 的闭式。
//! * [`qwb_scf::ecp::matrix_elements`]：**半数值**（生产路径）。ECP 中心上的
//!   径向 × 角向求积；非局域部分用实球谐投影
//!   `⟨μ|ΔV P_l|ν⟩ = Σ_i w_i r_i² ΔV(r_i) Σ_m c^μ_{lm} c^ν_{lm}`，
//!   这样一中心与异中心用同一套公式。
//!
//! 判据：一中心时两条实现的最大绝对差 < 1e-8 Ha（实测远小于此，数字写在断言信息里）；
//! 异中心（CuH/ZnO 里的 H、O 函数）没有解析对照，用**求积加密自收敛**：
//! `grid-fine` 与 `grid-veryfine` 的差 < 1e-6 Ha，且 `grid-veryfine` 与更密的
//! 合成等级（径向 220、角向预算 1450）差 < 1e-7 Ha。
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_registry::ecp::Ecp;
use qwb_registry::grid::GridLevelDef;
use qwb_scf::ecp::{EcpOperator, matrix_elements, one_center_matrix};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn registry() -> Registry {
    Registry::open(repo_root().join("qwb-data")).expect("qwb-data")
}

fn basis_for(
    registry: &Registry,
    elements: &[&str],
    molecule: &Molecule,
) -> qwb_basis::InternalBasis {
    // 用 lanl2dz（同时覆盖 H/O/Cu/Zn）——两个体系都用它。
    let _ = elements;
    qwb_basis::InternalBasis::build(registry.get_basis("lanl2dz").expect("lanl2dz"), molecule)
        .expect("basis")
}

fn operator(_registry: &Registry, name: &str, molecule: &Molecule) -> EcpOperator {
    let path = repo_root()
        .join("qwb-data")
        .join("ecp")
        .join(format!("{name}.yaml"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("读不到 {}: {err}", path.display()));
    let ecp: Ecp = serde_yaml::from_str(&text).expect("K1 ecp");
    EcpOperator::from_ecp(&ecp, molecule).expect("operator")
}

fn max_abs_difference(left: &Matrix, right: &Matrix) -> f64 {
    assert_eq!(left.nrows(), right.nrows());
    assert_eq!(left.ncols(), right.ncols());
    let mut worst = 0.0_f64;
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            worst = worst.max((left[(i, j)] - right[(i, j)]).abs());
        }
    }
    worst
}

fn max_abs(matrix: &Matrix) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            worst = worst.max(matrix[(i, j)].abs());
        }
    }
    worst
}

fn symmetry_error(matrix: &Matrix) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            worst = worst.max((matrix[(i, j)] - matrix[(j, i)]).abs());
        }
    }
    worst
}

fn atom(symbol: &str, position: [f64; 3]) -> Atom {
    Atom::new(symbol, position)
}

#[test]
fn copper_one_centre_analytic_and_numerical_agree() {
    let registry = registry();
    let molecule = Molecule::new(vec![atom("Cu", [0.0, 0.0, 0.0])], 0, 2);
    let basis = basis_for(&registry, &["Cu"], &molecule);
    let operator = operator(&registry, "lanl2dz-cu", &molecule);
    let level = registry.get_grid("grid-veryfine").expect("grid-veryfine");

    let analytic = one_center_matrix(&basis, &operator).expect("analytic");
    let numerical = matrix_elements(&basis, &operator, level).expect("numerical");
    let worst = max_abs_difference(&analytic, &numerical);
    let scale = max_abs(&analytic);
    println!("Cu/LANL2DZ：解析 vs 半数值最大绝对差 {worst:.3e} Ha（矩阵最大元 {scale:.3e}）");
    assert!(
        worst < 1e-8,
        "Cu 的一中心 ECP 矩阵元与解析值差 {worst:.3e} Ha，超过 1e-8"
    );
    assert!(scale > 1.0e-3, "矩阵不该全零（最大元 {scale:.3e}）");
    assert!(symmetry_error(&numerical) < 1e-12, "ECP 矩阵必须对称");
}

#[test]
fn zinc_one_centre_analytic_and_numerical_agree() {
    let registry = registry();
    let molecule = Molecule::new(vec![atom("Zn", [0.0, 0.0, 0.0])], 0, 1);
    let basis = basis_for(&registry, &["Zn"], &molecule);
    let operator = operator(&registry, "lanl2dz-zn", &molecule);
    assert_eq!(operator.projectors.len(), 3, "Zn 的投影通道是 l = 0,1,2");
    let level = registry.get_grid("grid-veryfine").expect("grid-veryfine");

    let analytic = one_center_matrix(&basis, &operator).expect("analytic");
    let numerical = matrix_elements(&basis, &operator, level).expect("numerical");
    let worst = max_abs_difference(&analytic, &numerical);
    println!(
        "Zn/LANL2DZ：解析 vs 半数值最大绝对差 {worst:.3e} Ha（矩阵最大元 {:.3e}）",
        max_abs(&analytic)
    );
    assert!(
        worst < 1e-8,
        "Zn 的一中心 ECP 矩阵元与解析值差 {worst:.3e} Ha，超过 1e-8"
    );
    assert!(symmetry_error(&numerical) < 1e-12);
}

#[test]
fn copper_d_channel_is_nonzero_and_matches_the_analytic_value() {
    // T5 点名的"Cu 的 l=2 通道单独断言"：LANL2DZ 把 l=2 当**局域**通道，
    // 所以 d 函数看到的正是 V_local。这里单独算"只有 V_local"的矩阵并与半数值对照。
    let registry = registry();
    let molecule = Molecule::new(vec![atom("Cu", [0.0, 0.0, 0.0])], 0, 2);
    let basis = basis_for(&registry, &["Cu"], &molecule);
    let mut operator = operator(&registry, "lanl2dz-cu", &molecule);
    assert!(
        operator.local.iter().any(|term| term.power == 1),
        "Cu 的局域通道应含 r 的一次项（BSE 里的系数是 -10）"
    );

    // 把投影通道拿掉 ⇒ 只剩 V_local ⇒ 两条路径都要给出同一个矩阵。
    operator.projectors.clear();
    let level = registry.get_grid("grid-veryfine").expect("grid-veryfine");
    let analytic = one_center_matrix(&basis, &operator).expect("analytic");
    let numerical = matrix_elements(&basis, &operator, level).expect("numerical");
    let worst = max_abs_difference(&analytic, &numerical);
    println!("Cu 的 l=2（局域）通道：解析 vs 半数值 {worst:.3e} Ha");
    assert!(worst < 1e-8, "l=2 通道差 {worst:.3e} Ha");

    // d 子块必须真的有内容。Cu/LANL2DZ 的壳层次序是 3×s（0–2）、3×p（3–11）、
    // 2×d（12–21，球谐 5 个/收缩），所以 d 子块是最后 10 个函数。
    let n = basis.n_functions();
    let mut d_block = 0.0_f64;
    let mut d_diag = 0.0_f64;
    for i in (n - 10)..n {
        d_diag = d_diag.max(numerical[(i, i)].abs());
        for j in (n - 10)..n {
            d_block = d_block.max(numerical[(i, j)].abs());
        }
    }
    println!("Cu d 子块最大元 {d_block:.3e} Ha（对角最大 {d_diag:.3e}）");
    // 紧的 d 收缩（指数 41.34…）在核附近看到 V_local，对角元是 1.9e-2 量级
    // （ECP 对 d 通道是小修正，不是主项）；弥散的第二个 d 收缩只有 ~1e-6。
    // 判据按实测取 1e-2：既排除"整块为零"，也不会因为换个基组就假红。
    assert!(
        d_block > 1.0e-2,
        "d 通道矩阵元应有 1e-2 量级（实测 {d_block:.3e}）"
    );
    assert!(
        d_diag > 1.0e-2,
        "d 对角元应有 1e-2 量级（实测 {d_diag:.3e}）"
    );
}

#[test]
fn off_centre_contributions_converge_with_the_quadrature() {
    // CuH：H 的 1s 在另一个中心上，没有解析对照，所以用求积加密自收敛来定精度。
    let registry = registry();
    let molecule = Molecule::new(
        vec![atom("Cu", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 2.6])],
        0,
        2,
    );
    let basis = basis_for(&registry, &["Cu", "H"], &molecule);
    let operator = operator(&registry, "lanl2dz-cu", &molecule);

    let fine = registry.get_grid("grid-fine").expect("grid-fine");
    let veryfine = registry.get_grid("grid-veryfine").expect("grid-veryfine");
    let fine_matrix = matrix_elements(&basis, &operator, fine).expect("fine");
    let veryfine_matrix = matrix_elements(&basis, &operator, veryfine).expect("veryfine");
    let difference = max_abs_difference(&fine_matrix, &veryfine_matrix);
    println!(
        "CuH：grid-fine vs grid-veryfine 的 ECP 矩阵最大差 {difference:.3e} Ha；\
         矩阵最大元 {:.3e}",
        max_abs(&veryfine_matrix)
    );
    assert!(
        difference < 1.0e-6,
        "两个网格等级的 ECP 矩阵差 {difference:.3e} Ha，求积没收敛"
    );

    // 异中心元的量级：H 的 1s 与 Cu 的 4s 重叠（相距 2.6 Bohr），不能是 0。
    let n = basis.n_functions();
    // H 的 2 个 s 函数是最后两个（lanl2dz 的 H = [2s]）。
    let mut off = 0.0_f64;
    for i in (n - 2)..n {
        for j in 0..(n - 2) {
            off = off.max(veryfine_matrix[(i, j)].abs());
        }
    }
    println!("Cu–H 异中心 ECP 矩阵元最大 {off:.3e} Ha");
    assert!(off > 1.0e-4, "异中心矩阵元不该全是 0（最大 {off:.3e}）");
}

#[test]
fn zero_ecp_is_the_identity_of_the_operator() {
    // 把项全清掉 ⇒ 矩阵必须是零矩阵（防止"某处偷偷加了点东西"）。
    let registry = registry();
    let molecule = Molecule::new(vec![atom("Cu", [0.0, 0.0, 0.0])], 0, 2);
    let basis = basis_for(&registry, &["Cu"], &molecule);
    let mut operator = operator(&registry, "lanl2dz-cu", &molecule);
    operator.local.clear();
    operator.projectors.clear();
    let level = registry.get_grid("grid-coarse").expect("grid-coarse");
    let matrix = matrix_elements(&basis, &operator, level).expect("matrix");
    assert_eq!(max_abs(&matrix), 0.0);
}

/// 合成的高密度等级：用来做"加密自收敛"的第三条参照。
fn denser_level() -> GridLevelDef {
    // `GridLevelDef` 是 K1 的 non_exhaustive 类型，测试里不能直接构造字面量，
    // 所以从 YAML 反序列化一份（与数据文件同一形态）。
    let text = "
name: ecp-veryfine-plus
version: 1.0.0
citation:
  - title: synthetic quadrature used only by tests/i05_ecp.rs
license:
  spdx: Apache-2.0
  commercial_ok: true
  redistribute_ok: true
created: 2026-10-26T00:00:00Z
level: veryfine
radial_points: 220
angular_rule: gauss-product
lebedev_order: 1450
becke_sharpness: 0.8
cutoff: 20.0
";
    serde_yaml::from_str(text).expect("synthetic grid level")
}

#[test]
fn very_fine_and_a_denser_quadrature_agree() {
    let registry = registry();
    let molecule = Molecule::new(
        vec![atom("Cu", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 2.6])],
        0,
        2,
    );
    let basis = basis_for(&registry, &["Cu", "H"], &molecule);
    let operator = operator(&registry, "lanl2dz-cu", &molecule);

    let veryfine = registry.get_grid("grid-veryfine").expect("grid-veryfine");
    let a = matrix_elements(&basis, &operator, veryfine).expect("veryfine");
    let b = matrix_elements(&basis, &operator, &denser_level()).expect("denser");
    let difference = max_abs_difference(&a, &b);
    println!("CuH：grid-veryfine vs 加密一级的 ECP 矩阵最大差 {difference:.3e} Ha");
    assert!(
        difference < 1.0e-7,
        "veryfine 与加密一级差 {difference:.3e} Ha，求积还不够"
    );
}
