//! GGA 前置：AO 值 + 梯度。用有限差分校验梯度（与解析式独立）。
use std::path::PathBuf;

use qwb_basis::InternalBasis;
use qwb_engine_api::{Atom, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{ao_value, ao_value_and_gradient};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
}

fn molecule_h2o() -> Molecule {
    Molecule::new(
        vec![
            Atom::new("O", [0.1, -0.05, 0.02]),
            Atom::new("H", [0.3, 0.757, 0.586]),
            Atom::new("H", [-0.2, -0.757, 0.586]),
        ],
        0,
        1,
    )
}

#[test]
fn ao_gradients_match_finite_differences() {
    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = molecule_h2o();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");

    // 若干不对称的考察点（避免梯度恰好为零）
    let points = [[0.42, -0.31, 0.77], [-0.9, 0.2, -0.35], [1.3, 1.1, -0.8]];
    let step = 1.0e-5_f64;
    let mut worst = 0.0_f64;
    for point in points {
        let (values, gradients) = ao_value_and_gradient(&basis, point).expect("value+gradient");
        // 函数值必须与 ao_value 完全一致
        let plain = ao_value(&basis, point).expect("value");
        for (i, (a, b)) in values.iter().zip(&plain).enumerate() {
            // 两处实现的乘除顺序不同，末位可能差 1 ulp；数学上必须相同。
            let error = (a - b).abs();
            assert!(
                error <= 1.0e-15 * b.abs().max(1.0),
                "AO {i} 的值与 ao_value 不一致：{a} vs {b}（差 {error:e}）"
            );
        }
        for axis in 0..3 {
            let mut shifted = point;
            shifted[axis] += step;
            let plus = ao_value(&basis, shifted).expect("plus");
            shifted[axis] -= 2.0 * step;
            let minus = ao_value(&basis, shifted).expect("minus");
            for i in 0..values.len() {
                let finite_difference = (plus[i] - minus[i]) / (2.0 * step);
                let analytic = gradients[i][axis];
                let error = (finite_difference - analytic).abs();
                // 相对判据：逐步差分对**紧函数**（O 1s 的 α≈130，三阶导 ~α³）的截断
                // 误差会到 ~1e-5 量级，所以按梯度自身的量级判，而不是固定 1e-8。
                let scale = analytic.abs().max(1.0);
                let relative = error / scale;
                worst = worst.max(relative);
                assert!(
                    relative < 1.0e-7,
                    "AO {i} 的梯度[{axis}]：解析 {analytic} vs 差分 {finite_difference}（相对差 {relative:e}）"
                );
            }
        }
    }
    println!("AO 梯度 vs 有限差分：最大绝对误差 {worst:.3e}");
}

#[test]
fn density_gradient_is_consistent_with_the_orbital_gradients() {
    // ∇ρ = Σ_μν P_μν (∇φ_μ φ_ν + φ_μ ∇φ_ν)：用有限差分直接验证这条恒等式。
    use qwb_integrals::one_electron::one_electron;
    use qwb_integrals::two_electron::two_electron;
    use qwb_scf::rks::{RksOptions, rks_scf};

    let registry = Registry::open(data_root()).expect("qwb-data");
    let basis_k1 = registry.get_basis("sto-3g").expect("sto-3g");
    let molecule = molecule_h2o();
    let basis = InternalBasis::build(basis_k1, &molecule).expect("build");
    let _one = one_electron(&basis, &molecule).expect("1e");
    let tensor = two_electron(&basis, &molecule).expect("2e");
    let grid_level = registry.get_grid("grid-coarse").expect("grid");
    let grid = qwb_grid::molecular_grid(grid_level, &molecule).expect("grid");
    let mut options = RksOptions::default();
    options.max_iter = 400;
    let result = rks_scf(&basis, &molecule, &tensor, &grid, &[], options).expect("scf");
    let n = basis.n_functions();

    let density_at = |point: [f64; 3]| -> f64 {
        let values = ao_value(&basis, point).expect("ao");
        let mut rho = 0.0;
        for i in 0..n {
            for j in 0..n {
                rho += result.density[(i, j)] * values[i] * values[j];
            }
        }
        rho
    };

    let point = [0.35, 0.42, 0.61];
    let (values, gradients) = ao_value_and_gradient(&basis, point).expect("grad");
    let mut analytic = [0.0_f64; 3];
    for axis in 0..3 {
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum += result.density[(i, j)]
                    * (gradients[i][axis] * values[j] + values[i] * gradients[j][axis]);
            }
        }
        analytic[axis] = sum;
    }
    let step = 1.0e-5_f64;
    for axis in 0..3 {
        let mut shifted = point;
        shifted[axis] += step;
        let plus = density_at(shifted);
        shifted[axis] -= 2.0 * step;
        let minus = density_at(shifted);
        let finite_difference = (plus - minus) / (2.0 * step);
        let error = (finite_difference - analytic[axis]).abs();
        println!(
            "∇ρ[{axis}]：解析 {:.10} vs 差分 {:.10}（差 {error:.2e}）",
            analytic[axis], finite_difference
        );
        assert!(error < 1.0e-8, "∇ρ[{axis}] 不一致");
    }
}
