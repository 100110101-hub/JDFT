//! 回归测试：`σ = 0` 处 B88 的导数必须是**有限**的。
//!
//! 这是一个真实的 bug 的钉子。B88 原来写的是
//!
//! ```text
//! x = sqrt(σ)/ρ^{4/3}，ε = ε_lda − βρ^{1/3}·x²/(1 + 6β·x·asinh x)
//! ```
//!
//! `x` 对 `σ` 的导数是 `1/(2√σ)`，`σ = 0` 时发散；对偶数把 `x·x` 的导数算成 `2·0·∞ = NaN`，
//! 尽管 `x²` 对 `σ` 是线性的、导数有界。真实 SCF 里**远处网格点的 AO 梯度会下溢成精确的 0**，
//! 于是 v_xc 出现 NaN，整个 Fock 被污染——B3LYP 之前在 grid-coarse 上的 NaN 就是它。
//!
//! 修法：对偶数只承载 `t = x² = σ/ρ^{8/3}`，`A(x) = x·asinh(x)` 用解析导数 `dA/dt` 喂进去。
//!
//! 本测试在 `ρ` 跨越 1e-3 … 1e-222、`σ = 0` 与 `σ` 极小的组合上断言四个量全部有限，
//! 并且与有限差分一致（在 σ = 0 处用单侧差分）。
use std::collections::BTreeMap;

use qwb_xc::FunctionalTerm;

fn term() -> Box<dyn FunctionalTerm> {
    let mut parameters = BTreeMap::new();
    parameters.insert("b88_beta".to_string(), 0.0042);
    qwb_xc::build_term("GgaB88X", &parameters).expect("GgaB88X")
}

fn evaluate(term: &dyn FunctionalTerm, rho: f64, sigma: f64) -> (f64, f64, f64) {
    let rho_v = [rho];
    let sigma_v = [sigma];
    let batch = qwb_xc::DensityBatch::new(&rho_v, &sigma_v);
    let mut epsilon = [0.0_f64];
    let mut d_rho = [0.0_f64];
    let mut d_sigma = [0.0_f64];
    let mut d_laplacian = [0.0_f64];
    term.energy_density(&batch, &mut epsilon);
    term.potential_derivatives_laplacian(&batch, &mut d_rho, &mut d_sigma, &mut d_laplacian);
    (epsilon[0], d_rho[0], d_sigma[0])
}

#[test]
fn sigma_zero_derivatives_are_finite() {
    let b88 = term();
    for rho in [
        1.0e-3_f64,
        1.0e-10,
        1.0e-50,
        1.0e-120,
        3.525e-222,
        f64::MIN_POSITIVE,
    ] {
        for sigma in [0.0_f64, 1.0e-300, 1.0e-30, 1.0e-10] {
            let (epsilon, d_rho, d_sigma) = evaluate(b88.as_ref(), rho, sigma);
            assert!(
                epsilon.is_finite(),
                "ρ = {rho:e}, σ = {sigma:e}: ε 非有限（{epsilon}）"
            );
            assert!(
                d_rho.is_finite(),
                "ρ = {rho:e}, σ = {sigma:e}: dε/dρ 非有限（{d_rho}）"
            );
            assert!(
                d_sigma.is_finite(),
                "ρ = {rho:e}, σ = {sigma:e}: dε/dσ 非有限（{d_sigma}）"
            );
        }
    }
}

/// 在 `σ = 0` 处用单侧差分核对 `dε/dσ`（导数虽然有界，但此前是 NaN，这里再钉数值）。
#[test]
fn sigma_zero_slope_matches_one_sided_difference() {
    let b88 = term();
    for rho in [0.5_f64, 0.05, 5.0e-3] {
        let (_, _, analytic) = evaluate(b88.as_ref(), rho, 0.0);
        // σ 的相对步长：ε 对 σ 的依赖是 t = σ/ρ^{8/3}，在 σ=0 附近近似线性
        let h = 1.0e-6 * rho.powf(8.0 / 3.0);
        let (e0, _, _) = evaluate(b88.as_ref(), rho, 0.0);
        let (e1, _, _) = evaluate(b88.as_ref(), rho, h);
        let (e2, _, _) = evaluate(b88.as_ref(), rho, 2.0 * h);
        // 三点单侧二阶：(-3e0 + 4e1 - e2)/(2h)
        let numeric = (-3.0 * e0 + 4.0 * e1 - e2) / (2.0 * h);
        let scale = analytic.abs().max(numeric.abs()).max(1.0e-30);
        println!(
            "ρ = {rho}: dε/dσ 解析 {analytic:.10e}，单侧差分 {numeric:.10e}，相对 {:.2e}",
            (analytic - numeric).abs() / scale
        );
        assert!(
            (analytic - numeric).abs() / scale < 1.0e-5,
            "ρ = {rho}: dε/dσ {analytic} vs {numeric}"
        );
    }
}
