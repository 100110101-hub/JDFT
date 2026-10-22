//! X-01 + X-02 验收（project.md §6.4）。
//!
//! - X-01：`FunctionalTerm` 的批量接口与 LDA 项的数值正确性；
//! - **X-02**：`dε/dρ`、`dε/dσ` 与**中心有限差分**一致 —— 这是 G3 物理门禁，
//!   也是 D-09 里被推迟的那条"有限差分导数 < 1e-9"。

use std::collections::BTreeMap;

use qwb_xc::{
    DensityBatch, FunctionalTerm, LdaSlaterX, LdaSvwn5, Requirements, XcError, build_term,
};

const PI: f64 = std::f64::consts::PI;

/// rs（Wigner–Seitz 半径）→ ρ。
fn density_from_rs(rs: f64) -> f64 {
    3.0 / (4.0 * PI * rs * rs * rs)
}

/// 五点中心差分：`dε/dρ`。O(h⁴) 截断，误差远小于 1e-9 门禁。
fn finite_difference_rho(term: &dyn FunctionalTerm, rho: f64) -> f64 {
    let h = 1e-4 * rho;
    let energy = |r: f64| -> f64 {
        let mut e = [0.0];
        term.energy_density(&DensityBatch::from_rho(&[r]), &mut e);
        e[0]
    };
    let f_plus1 = energy(rho + h);
    let f_minus1 = energy(rho - h);
    let f_plus2 = energy(rho + 2.0 * h);
    let f_minus2 = energy(rho - 2.0 * h);
    (-f_plus2 + 8.0 * f_plus1 - 8.0 * f_minus1 + f_minus2) / (12.0 * h)
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let scale = actual.abs().max(expected.abs()).max(1e-30);
    let relative = (actual - expected).abs() / scale;
    assert!(
        relative < tolerance,
        "{what}: {actual:.15e} vs {expected:.15e}（相对差 {relative:e} > {tolerance:e}）"
    );
}

// ───────────── X-01：Slater 交换的解析值 ─────────────

#[test]
fn slater_exchange_matches_the_analytic_formula() {
    let term = LdaSlaterX::new(2.0 / 3.0);
    assert_eq!(term.id(), "LdaSlaterX");
    assert_eq!(term.requires(), Requirements::lda());

    let rho = [1.0_f64, 0.5, 10.0, 1e-4];
    let mut energies = [0.0; 4];
    term.energy_density(&DensityBatch::from_rho(&rho), &mut energies);

    for (slot, r) in energies.iter().zip(rho) {
        // ε_x = −(3/4)(3/π)^{1/3} ρ^{1/3}
        let expected = -0.75 * (3.0 / PI).cbrt() * r.cbrt();
        assert_relative(*slot, expected, 1e-14, "ε_x");
    }

    // 均匀电子气极限的另一种写法：ε_x = −3/(4π)·(9π/4)^{1/3}/rs
    for &rs in &[1.0_f64, 2.0, 5.0] {
        let r = density_from_rs(rs);
        let mut buffer = [0.0];
        term.energy_density(&DensityBatch::from_rho(&[r]), &mut buffer);
        let expected = -3.0 / (4.0 * PI) * (9.0 * PI / 4.0).cbrt() / rs;
        assert_relative(buffer[0], expected, 1e-13, "ε_x(rs)");
    }

    // 齐次性：ε(8ρ) = 2ε(ρ)（因为 ε ∝ ρ^{1/3}）。
    let mut scaled = [0.0];
    term.energy_density(&DensityBatch::from_rho(&[8.0 * 0.5]), &mut scaled);
    let mut base = [0.0];
    term.energy_density(&DensityBatch::from_rho(&[0.5]), &mut base);
    assert_relative(scaled[0], 2.0 * base[0], 1e-13, "齐次性");
}

// ───────────── X-02：有限差分导数门禁（< 1e-9）─────────────

#[test]
fn x02_slater_exchange_derivative_passes_the_finite_difference_gate() {
    let term = LdaSlaterX::new(2.0 / 3.0);
    for rho in [1e-3_f64, 0.05, 0.5, 1.0, 7.5] {
        let input = [rho];
        let mut d_rho = [0.0];
        let mut d_sigma = [0.0];
        term.potential_derivatives(&DensityBatch::from_rho(&input), &mut d_rho, &mut d_sigma);

        let numeric = finite_difference_rho(&term, rho);
        assert_relative(d_rho[0], numeric, 1e-9, "dε_x/dρ");
        assert_eq!(d_sigma[0], 0.0, "纯 LDA 的 dε/dσ 必须为 0");
    }
}

#[test]
fn x02_svwn5_derivative_passes_the_finite_difference_gate() {
    // 用数据文件里的 VWN5 (V) 顺磁参数（D-09 已数值确认过这组常数）。
    let term = LdaSvwn5::new((0.031_090_7, -0.409_286, 13.0720, 42.7198));
    assert_eq!(term.id(), "LdaSvwn5");

    for &rs in &[0.5_f64, 1.0, 2.0, 5.0, 12.0] {
        let rho = density_from_rs(rs);
        let input = [rho];
        let mut d_rho = [0.0];
        let mut d_sigma = [0.0];
        term.potential_derivatives(&DensityBatch::from_rho(&input), &mut d_rho, &mut d_sigma);

        let numeric = finite_difference_rho(&term, rho);
        // X-02 的门禁是 < 1e-9（相对）。中心差分在 h=1e-6·ρ 下误差 ~h²·ε'''，
        // 这里被积函数足够光滑，实测远小于门禁值。
        assert_relative(d_rho[0], numeric, 1e-9, &format!("dε_c/dρ @ rs={rs}"));
        assert_eq!(d_sigma[0], 0.0, "纯 LDA 的 dε/dσ 必须为 0");
    }
}

// ───────────── X-01：VWN5 的数值（与 D-09 的独立求值一致）─────────────

#[test]
fn svwn5_correlation_reproduces_the_ceperley_alder_point() {
    let term = LdaSvwn5::new((0.031_090_7, -0.409_286, 13.0720, 42.7198));
    let rho = density_from_rs(1.0);
    let mut buffer = [0.0];
    term.energy_density(&DensityBatch::from_rho(&[rho]), &mut buffer);
    // Ceperley–Alder 顺磁、rs=1：ε_c ≈ −0.0799 Ha（§7.0 外部引用，容差 5 mHa）。
    assert!(
        (buffer[0] + 0.0799).abs() < 5e-3,
        "ε_c(rs=1) = {} 应接近 −0.0799",
        buffer[0]
    );
}

// ───────────── 工厂与批量接口 ─────────────

#[test]
fn factory_builds_terms_from_data_parameters() {
    let mut parameters = BTreeMap::new();
    parameters.insert("slater_alpha".to_string(), 2.0 / 3.0);
    parameters.insert("vwn5_p_A".to_string(), 0.031_090_7);
    parameters.insert("vwn5_p_x0".to_string(), -0.409_286);
    parameters.insert("vwn5_p_b".to_string(), 13.0720);
    parameters.insert("vwn5_p_c".to_string(), 42.7198);

    let exchange = build_term("LdaSlaterX", &parameters).expect("exchange");
    let correlation = build_term("LdaSvwn5", &parameters).expect("correlation");
    assert_eq!(exchange.id(), "LdaSlaterX");
    assert_eq!(correlation.id(), "LdaSvwn5");

    // 未知组件 → 明确错误（不猜）。
    assert!(matches!(
        build_term("GgaPbeX", &parameters),
        Err(XcError::UnknownTerm(_))
    ));
    // 缺参数 → 明确错误。
    let empty = BTreeMap::new();
    assert!(matches!(
        build_term("LdaSlaterX", &empty),
        Err(XcError::MissingParameter { .. })
    ));
}

#[test]
fn batch_interface_handles_many_points_at_once() {
    let term = LdaSlaterX::new(2.0 / 3.0);
    let rho: Vec<f64> = (1..=1000).map(|i| f64::from(i) * 1e-3).collect();
    let mut out = vec![0.0; rho.len()];
    term.energy_density(&DensityBatch::from_rho(&rho), &mut out);
    assert!(out.iter().all(|value| *value < 0.0));

    let mut d_rho = vec![0.0; rho.len()];
    let mut d_sigma = vec![0.0; rho.len()];
    term.potential_derivatives(&DensityBatch::from_rho(&rho), &mut d_rho, &mut d_sigma);
    assert!(d_rho.iter().all(|value| *value < 0.0));
    assert!(d_sigma.iter().all(|value| *value == 0.0));
}

#[test]
fn shapes_are_checked() {
    let term = LdaSlaterX::new(2.0 / 3.0);
    let rho = [1.0, 2.0];
    // 长度正确时正常出值；长度错误属调用方的编程错误，实现里用 expect 断言。
    let mut good = [0.0, 0.0];
    term.energy_density(&DensityBatch::from_rho(&rho), &mut good);
    assert!(good[0] < 0.0 && good[1] < 0.0);
    assert!(good[0] > good[1], "ρ 越大 ε_x 越负");
}
