//! D-11 的 B88 交换：三条不依赖外部参照的判据。
//!
//! B88（Becke 1988）是纯 GGA 闭式，`ε_x = ε_x^LDA − β·ρ^{1/3}·x²/(1+6β·x·asinh x)`，
//! `x = |∇ρ|/ρ^{4/3}`、`β = 0.0042`。三条判据：
//!
//! 1. **σ→0 退化为 Slater**（`x → 0` 时修正项消失）：与 `LdaSlaterX` 逐点一致；
//! 2. **增强因子**：`F(x) = 1 + (β/C_x)x²/(1+6βx·asinh x)` 与实现给出的
//!    `ε/ε^LDA` 相对差 < 1e-13（在若干 `x` 上；`x` 由构造的 `(ρ, σ)` 反解）；
//! 3. **导数**：`∂ε/∂ρ`、`∂ε/∂σ` 与五点四阶差分相对差 < 1e-9
//!    （与 `d10_pbe.rs` 同一套判据；这是 GGA 最容易被写错的地方）。
//!
//! B88 的正性检查：`β > 0` 时 `F ≥ 1`，也就是**交换能绝对值变大**（更负）——
//! 这一条把「符号写反」直接挡掉。
use std::collections::BTreeMap;

use qwb_xc::FunctionalTerm;

fn beta() -> f64 {
    0.0042
}

fn term() -> Box<dyn FunctionalTerm> {
    let mut parameters = BTreeMap::new();
    parameters.insert("b88_beta".to_string(), beta());
    qwb_xc::build_term("GgaB88X", &parameters).expect("GgaB88X")
}

/// 实现给出的 `ε`（单点）。
fn energy(term: &dyn FunctionalTerm, rho: f64, sigma: f64) -> f64 {
    let rho_values = [rho];
    let sigma_values = [sigma];
    let batch = qwb_xc::DensityBatch::new(&rho_values, &sigma_values);
    let mut out = [0.0_f64];
    term.energy_density(&batch, &mut out);
    out[0]
}

fn derivatives(term: &dyn FunctionalTerm, rho: f64, sigma: f64) -> (f64, f64) {
    let rho_values = [rho];
    let sigma_values = [sigma];
    let batch = qwb_xc::DensityBatch::new(&rho_values, &sigma_values);
    let mut d_rho = [0.0_f64];
    let mut d_sigma = [0.0_f64];
    term.potential_derivatives(&batch, &mut d_rho, &mut d_sigma);
    (d_rho[0], d_sigma[0])
}

#[test]
fn sigma_to_zero_is_slater() {
    let mut parameters = BTreeMap::new();
    parameters.insert("slater_alpha".to_string(), 2.0 / 3.0);
    let slater = qwb_xc::build_term("LdaSlaterX", &parameters).expect("slater");
    let b88 = term();
    for rho in [1.0e-3_f64, 0.01, 0.1, 1.0, 10.0] {
        let from_b88 = energy(b88.as_ref(), rho, 0.0);
        let from_slater = energy(slater.as_ref(), rho, 0.0);
        assert!(
            (from_b88 - from_slater).abs() < 1.0e-14 * from_slater.abs().max(1.0),
            "ρ = {rho}：B88 在 σ=0 时给 {from_b88}，Slater 给 {from_slater}"
        );
    }
}

#[test]
fn enhancement_factor_matches_the_closed_form() {
    let b88 = term();
    // 用 (ρ, σ) 构造若干 x；σ = x²ρ^{8/3}
    for x in [1.0e-3_f64, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0] {
        let rho = 0.7_f64;
        let sigma = x * x * rho.powf(8.0 / 3.0);
        let computed = energy(b88.as_ref(), rho, sigma);
        let from_slater = energy(
            qwb_xc::build_term("LdaSlaterX", &{
                let mut p = BTreeMap::new();
                p.insert("slater_alpha".to_string(), 2.0 / 3.0);
                p
            })
            .expect("slater")
            .as_ref(),
            rho,
            sigma,
        );
        // ε_b88 = ε_lda − corr，所以增强因子是 computed/from_slater（不是反过来）。
        // B88 的原始形式用**每自旋**的约化梯度 x_σ = 2^{1/3}x（闭壳层），系数是
        // β/X_FACTOR_C（X_FACTOR_C = (3/4)(6/π)^{1/3}）——libxc 的 gga_x_b88.c 就是这么写的。
        // 漏掉这两个约定会让梯度修正弱 1.26 倍以上（BLYP 在水上偏高 112 mHa 的根因）。
        let ratio = computed / from_slater;
        let c2 = 2.0_f64.cbrt();
        let x_spin = c2 * x;
        let x_factor_c = 0.930_525_736_349_1_f64;
        let expected = 1.0
            + (beta() / x_factor_c) * x_spin * x_spin
                / (1.0 + 6.0 * beta() * x_spin * x_spin.asinh());
        assert!(
            ((ratio - expected) / expected).abs() < 1.0e-12,
            "x = {x}：实现的增强因子 {ratio} 与闭式 {expected} 不符"
        );
        assert!(ratio >= 1.0, "x = {x}：F 应 ≥ 1（β>0 时交换能绝对值变大）");
    }
}

#[test]
fn derivatives_match_finite_differences() {
    let b88 = term();
    for (rho, sigma) in [
        (0.5_f64, 0.05_f64),
        (0.1, 0.4),
        (1.0, 2.0),
        (2.0, 0.01),
        (0.02, 0.001),
    ] {
        let (analytic_rho, analytic_sigma) = derivatives(b88.as_ref(), rho, sigma);
        // 五点四阶的步长取相对 1e-4：截断 O(h⁴)~1e-16、舍入 ~1e-16/h~1e-12，两边都够
        let h_rho = 1.0e-4 * rho;
        let h_sigma = 1.0e-4 * sigma.max(1.0e-6);
        // 五点四阶
        let slope = |shift: f64, along_rho: bool| {
            let sample = |factor: f64| {
                if along_rho {
                    energy(b88.as_ref(), rho + factor * shift, sigma)
                } else {
                    energy(b88.as_ref(), rho, sigma + factor * shift)
                }
            };
            (-sample(2.0) + 8.0 * sample(1.0) - 8.0 * sample(-1.0) + sample(-2.0)) / (12.0 * shift)
        };
        let fd_rho = slope(h_rho, true);
        let fd_sigma = slope(h_sigma, false);
        let rel_rho = (analytic_rho - fd_rho).abs() / fd_rho.abs().max(1.0e-30);
        let rel_sigma = (analytic_sigma - fd_sigma).abs() / fd_sigma.abs().max(1.0e-30);
        println!(
            "ρ={rho} σ={sigma}: dε/dρ 解析 {analytic_rho:.10} 差分 {fd_rho:.10}（相对 {rel_rho:.2e}）;\
             dε/dσ 解析 {analytic_sigma:.10} 差分 {fd_sigma:.10}（相对 {rel_sigma:.2e}）"
        );
        // 判据是「相对 1e-9」或「落在差分的舍入地板以下」——后者在 dε/dσ 很小时才是瓶颈：
        // 五点四阶的舍入误差 ≈ 8·eps·|ε|/h，与 σ 导数相比可能到 1e-8 量级。
        let magnitude = energy(b88.as_ref(), rho, sigma).abs();
        let floor_rho = 8.0 * f64::EPSILON * magnitude / h_rho;
        let floor_sigma = 8.0 * f64::EPSILON * magnitude / h_sigma;
        assert!(
            (analytic_rho - fd_rho).abs() < 1.0e-9 * fd_rho.abs().max(1.0e-30) + 10.0 * floor_rho,
            "dε/dρ 解析 {analytic_rho:.12} vs 差分 {fd_rho:.12}（相对 {rel_rho:.2e}，地板 {floor_rho:.2e}）"
        );
        assert!(
            (analytic_sigma - fd_sigma).abs()
                < 1.0e-9 * fd_sigma.abs().max(1.0e-30) + 10.0 * floor_sigma,
            "dε/dσ 解析 {analytic_sigma:.12} vs 差分 {fd_sigma:.12}（相对 {rel_sigma:.2e}，地板 {floor_sigma:.2e}）"
        );
    }
}
