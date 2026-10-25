//! G-01 的三块新机械，各自与**有限差分**对照。
//!
//! 把零件拆开验，是因为 KS 解析梯度里有四个独立来源（`α_hf` 缩放、Pulay 的 `v_xc` 接线、
//! 一阶 AO 导数、二阶/三阶 AO 导数）。整条梯度对不上时，先分别钉住零件再拼，
//! 比盯着一个 8.8e-2 的差值猜要快 —— 这次它就是靠这个方法定位到"纯泛函不该有 HF 交换"的。
//!
//! * `ao_hessian`：与 `ao_value_and_gradient` 的中心差分比。
//! * `ao_laplacian_derivative`：与 `ao_laplacian` 的中心差分比（LYP 的 ∇²ρ 通道用它）。
//! * `xc_gradient_terms`：`∂E_xc/∂R|_{P 固定}` 与「位移几何、重算网格与 AO、密度矩阵固定」
//!   的能量差分比；PBE（σ 通道）与 B3LYP 组合（σ + ∇²ρ 通道）各一条。
#![allow(clippy::field_reassign_with_default)]
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{
    ao_hessian, ao_laplacian, ao_laplacian_derivative, ao_value_and_gradient, evaluate_xc,
    xc_gradient_terms,
};

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn h2() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("H", [0.0, 0.0, -0.55]),
            Atom::new("H", [0.0, 0.0, 0.55]),
        ],
        0,
        1,
    )
}

fn displaced(molecule: &Molecule, atom: usize, axis: usize, shift_bohr: f64) -> Molecule {
    let mut moved = molecule.clone();
    moved.atoms[atom].position_angstrom[axis] += shift_bohr * ANGSTROM_PER_BOHR;
    moved
}

/// B3LYP 的四个组件（程序化构造：数据层因为 LYP 要 ∇²ρ 而不收 b3lyp.yaml）。
fn b3lyp_terms() -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let mut p: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    p.insert("slater_alpha".to_string(), 2.0 / 3.0);
    p.insert("b88_beta".to_string(), 0.0042);
    p.insert("lyp_a".to_string(), 0.049_18);
    p.insert("lyp_b".to_string(), 0.132);
    p.insert("lyp_c".to_string(), 0.2533);
    p.insert("lyp_d".to_string(), 0.349);
    p.insert("vwn5_p_A".to_string(), 0.0310907);
    p.insert("vwn5_p_x0".to_string(), -0.409286);
    p.insert("vwn5_p_b".to_string(), 13.072);
    p.insert("vwn5_p_c".to_string(), 42.7198);
    vec![
        qwb_xc::scale_term(qwb_xc::build_term("LdaSlaterX", &p).expect("slater"), 0.08),
        qwb_xc::scale_term(qwb_xc::build_term("GgaB88X", &p).expect("b88"), 0.72),
        qwb_xc::scale_term(qwb_xc::build_term("GgaLypC", &p).expect("lyp"), 0.81),
        qwb_xc::scale_term(qwb_xc::build_term("LdaSvwn5", &p).expect("vwn5"), 0.19),
    ]
}

fn pbe_terms(registry: &Registry) -> Vec<Box<dyn qwb_xc::FunctionalTerm>> {
    let functional = registry.get_functional("pbe").expect("pbe");
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters).expect("term");
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    terms
}

#[test]
fn ao_hessian_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let point = [0.13, -0.27, 0.41];
    let analytic = ao_hessian(&basis, point).expect("hessian");
    let delta = 1.0e-5;
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += delta;
        minus[axis] -= delta;
        let (_, grad_plus) = ao_value_and_gradient(&basis, plus).expect("plus");
        let (_, grad_minus) = ao_value_and_gradient(&basis, minus).expect("minus");
        for mu in 0..basis.n_functions() {
            for j in 0..3 {
                let numeric = (grad_plus[mu][j] - grad_minus[mu][j]) / (2.0 * delta);
                let error = (analytic[mu][axis][j] - numeric).abs();
                worst = worst.max(error);
                assert!(
                    error < 1.0e-7,
                    "AO {mu} ∂²/∂{axis}∂{j}: 解析 {:.9e} 数值 {:.9e} 差 {error:.2e}",
                    analytic[mu][axis][j],
                    numeric
                );
            }
        }
    }
    println!("ao_hessian vs 差分最大差 {worst:.2e}");
}

#[test]
fn ao_laplacian_derivative_matches_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let point = [0.13, -0.27, 0.41];
    let analytic = ao_laplacian_derivative(&basis, point).expect("d lap");
    let delta = 1.0e-5;
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += delta;
        minus[axis] -= delta;
        let lap_plus = ao_laplacian(&basis, plus).expect("plus");
        let lap_minus = ao_laplacian(&basis, minus).expect("minus");
        for mu in 0..basis.n_functions() {
            let numeric = (lap_plus[mu] - lap_minus[mu]) / (2.0 * delta);
            let error = (analytic[mu][axis] - numeric).abs();
            worst = worst.max(error);
            assert!(
                error < 1.0e-7,
                "AO {mu} ∂/∂{axis}(∇²φ): 解析 {:.9e} 数值 {:.9e} 差 {error:.2e}",
                analytic[mu][axis],
                numeric
            );
        }
    }
    println!("ao_laplacian_derivative vs 差分最大差 {worst:.2e}");
}

/// `∂E_xc/∂R|_{P 固定}` 与能量差分对照。`label` 只用于日志。
fn check_xc_energy_gradient(
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    hf_exchange: f64,
    label: &str,
) -> f64 {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = h2();
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let tensor = qwb_integrals::two_electron::two_electron(&basis, &molecule).expect("2e");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let mut options = qwb_scf::rks::RksOptions::default();
    options.max_iter = 400;
    options.energy_threshold = 1.0e-9;
    options.hf_exchange = hf_exchange;
    let result =
        qwb_scf::rks::rks_scf(&basis, &molecule, &tensor, &grid, terms, options).expect("scf");
    let density = result.density.clone();

    let analytic = xc_gradient_terms(&basis, &molecule, &grid, &density, terms).expect("xc grad");

    let delta = 1.0e-4;
    let mut worst = 0.0_f64;
    #[allow(clippy::needless_range_loop)] // 下面同时要 atom/axis 与 analytic 的下标
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut energies = [0.0_f64; 2];
            for (slot, sign) in [-1.0_f64, 1.0].iter().enumerate() {
                let moved = displaced(&molecule, atom, axis, sign * delta);
                let moved_basis = qwb_basis::InternalBasis::build(
                    registry.get_basis("sto-3g").expect("sto-3g"),
                    &moved,
                )
                .expect("basis");
                let moved_grid = qwb_grid::molecular_grid(
                    registry.get_grid("grid-coarse").expect("grid"),
                    &moved,
                )
                .expect("grid");
                let (energy, _) =
                    evaluate_xc(&moved_basis, &moved_grid, &density, terms).expect("xc energy");
                energies[slot] = energy;
            }
            let numeric = (energies[1] - energies[0]) / (2.0 * delta);
            let error = (analytic[atom][axis] - numeric).abs();
            println!(
                "{label} atom {atom} axis {axis}: 解析 {:>14.9}  数值 {:>14.9}  差 {error:.2e}",
                analytic[atom][axis], numeric
            );
            worst = worst.max(error);
        }
    }
    worst
}

#[test]
fn xc_gradient_terms_match_finite_differences_for_pbe() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let terms = pbe_terms(&registry);
    let worst = check_xc_energy_gradient(&terms, 0.0, "PBE");
    println!("PBE 的 XC 显式项 vs 能量差分最大差 {worst:.2e}");
    assert!(worst < 5.0e-5, "PBE 的 XC 显式项差 {worst:.2e}");
}

#[test]
fn xc_gradient_terms_match_finite_differences_for_b3lyp() {
    let terms = b3lyp_terms();
    let worst = check_xc_energy_gradient(&terms, 0.2, "B3LYP");
    println!("B3LYP（含 LYP 的 ∇²ρ 通道）XC 显式项 vs 能量差分最大差 {worst:.2e}");
    assert!(worst < 2.0e-4, "B3LYP 的 XC 显式项差 {worst:.2e}");
}
