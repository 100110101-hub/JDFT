//! LYP 的变分自洽：`evaluate_xc` 返回的势必须是能量的变分导数（**含 ∇²ρ 通道**）。
//!
//! 做法：给密度矩阵加一个小扰动 `ΔP = s·M`（M 取几个对称的 AO 外积），比较
//! `ΔE` 与 `Tr(v·ΔP)`。一阶变分自洽要求残差随 `s → 0` **线性下降**。
//!
//! 这条测试只有把第三个通道接对才可能通过：若按「只有 ρ、σ 两通道」实现，`v_xc` 会缺
//! `∫ (ρ∂ε/∂∇²ρ)·∇²(φ_μ φ_ν)` 一整项，残差会是 O(1) 而不是 O(s)。
use std::collections::BTreeMap;
use std::path::PathBuf;

use qwb_engine_api::{Atom, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::Registry;
use qwb_xc::FunctionalTerm;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn lyp() -> Box<dyn FunctionalTerm> {
    let mut parameters = BTreeMap::new();
    parameters.insert("lyp_a".to_string(), 0.049_18);
    parameters.insert("lyp_b".to_string(), 0.132);
    parameters.insert("lyp_c".to_string(), 0.253_3);
    parameters.insert("lyp_d".to_string(), 0.349);
    qwb_xc::build_term("GgaLypC", &parameters).expect("lyp")
}

#[test]
fn lyp_potential_is_the_variational_derivative() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let molecule = Molecule::new(
        vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.757, 0.586]),
            Atom::new("H", [0.0, -0.757, 0.586]),
        ],
        0,
        1,
    );
    let basis =
        qwb_basis::InternalBasis::build(registry.get_basis("sto-3g").expect("sto-3g"), &molecule)
            .expect("build");
    let grid = qwb_grid::molecular_grid(registry.get_grid("grid-coarse").expect("grid"), &molecule)
        .expect("grid");
    let n = basis.n_functions();
    let mut density = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let value = if i == j {
                1.0
            } else {
                0.4 / (1.0 + (i as f64 - j as f64).abs())
            };
            density.set(i, j, value);
        }
    }
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![lyp()];
    let (_, potential) = qwb_scf::rks::evaluate_xc(&basis, &grid, &density, &terms).expect("xc");
    assert!(
        terms[0].requires().laplacian,
        "LYP 必须声明需要 ∇²ρ（否则 evaluate_xc 不会算它）"
    );

    let mut perturbations = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 2), (3, 3), (4, 5)] {
        if i < n && j < n {
            let mut m = Matrix::zeros(n, n);
            m.set(i, j, 1.0);
            m.set(j, i, 1.0);
            perturbations.push(m);
        }
    }
    let mut previous = f64::INFINITY;
    for s in [1.0e-3_f64, 1.0e-4, 1.0e-5] {
        let mut worst = 0.0_f64;
        for m in &perturbations {
            let mut moved = density.clone();
            for i in 0..n {
                for j in 0..n {
                    moved.set(i, j, density[(i, j)] + s * m[(i, j)]);
                }
            }
            let (e0, _) = qwb_scf::rks::evaluate_xc(&basis, &grid, &density, &terms).expect("xc");
            let (e1, _) = qwb_scf::rks::evaluate_xc(&basis, &grid, &moved, &terms).expect("xc");
            let mut tr = 0.0;
            for i in 0..n {
                for j in 0..n {
                    tr += potential[(i, j)] * s * m[(i, j)];
                }
            }
            worst = worst.max((e1 - e0 - tr).abs());
        }
        println!("扰动 s = {s:.0e}: 最大残差 {worst:.2e}");
        assert!(
            worst < previous.max(f64::INFINITY),
            "残差没随 s 下降：{previous:.2e} -> {worst:.2e}"
        );
        previous = worst;
    }
    assert!(
        previous < 1.0e-6,
        "最小扰动上的变分自洽残差 {previous:.2e} 太大（第三通道没接对？）"
    );
}
