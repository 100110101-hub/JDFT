//! SC-01b（UKS 的 XC 侧）：**自旋分辨 LSDA**（VWN 的两端插值）。
//!
//! 注意参数集的名字：`svwn5.yaml` 用的是 libxc 的 VWN **RPA** 参数集，
//! 逐点对照在 `sc01b_spin_xc_pointwise.rs`（`LDA_C_VWN_RPA`），来龙去脉见
//! `docs/choices.md` 五十。
//!
//! 判据（phase-2 §4.4 / §9.2 G9）：
//!
//! 1. **ζ = 0 时退回闭壳层**：`ε_c(ρ/2, ρ/2)` 与闭壳层 `energy_density(ρ)` 逐点相同（< 1e-12）；
//! 2. **有限差分**：`∂ε/∂ρ_α`、`∂ε/∂ρ_β` 与中心差分差 < 1e-9（X-02 的判据）；
//! 3. **铁磁极限**：`ζ = 1`（`ρ_β → 0`）时能量等于铁磁参数集的值；
//! 4. **自旋极性单调**：固定总密度，`ε_c` 随 `|ζ|` 增大而增大（VWN5 的自旋插值方向）；
//! 5. 没有铁磁参数时 `energy_density_spin` 返回 `SpinUnsupported`（不猜、不静默）。

use qwb_xc::{DensityBatch, FunctionalTerm, LdaSvwn5, SpinDensityBatch, XcError};

/// 与 `qwb-data/functional/svwn5.yaml` 一致的参数（VWN5 参数化 V）。
const PARA: (f64, f64, f64, f64) = (0.031_090_7, -0.409_286, 13.072, 42.719_8);
const FERRO: (f64, f64, f64, f64) = (0.015_545_3, -0.743_294, 20.123_1, 101.578);
const ALPHA: (f64, f64, f64, f64) = (-0.053_051_647_697_298_45, -0.004_758_4, 1.131_07, 13.004_5);

fn svwn5() -> LdaSvwn5 {
    LdaSvwn5::new(PARA).with_spin_parameters(FERRO, ALPHA)
}

#[test]
fn zero_polarization_reduces_to_the_closed_shell_value() {
    let term = svwn5();
    for rho in [1.0e-3, 0.01, 0.1, 0.5, 1.0, 10.0] {
        let (spin, _, _) = term.spin_energy_at(0.5 * rho, 0.5 * rho);
        let mut closed = [0.0_f64];
        term.energy_density(&DensityBatch::from_rho(&[rho]), &mut closed);
        assert!(
            (spin - closed[0]).abs() < 1e-12,
            "ρ = {rho}: 自旋版 {spin:.15} vs 闭壳层 {:.15}",
            closed[0]
        );
    }
}

#[test]
fn derivatives_match_central_differences() {
    let term = svwn5();
    let step = 1.0e-6;
    for (alpha, beta) in [
        (0.25, 0.25),
        (0.4, 0.1),
        (0.05, 0.45),
        (0.9, 0.1),
        (0.01, 0.99),
        (1.0, 0.2),
    ] {
        let (_, d_alpha, d_beta) = term.spin_energy_at(alpha, beta);
        let plus = term.spin_energy_at(alpha + step, beta).0;
        let minus = term.spin_energy_at(alpha - step, beta).0;
        let numeric_alpha = (plus - minus) / (2.0 * step);
        let plus = term.spin_energy_at(alpha, beta + step).0;
        let minus = term.spin_energy_at(alpha, beta - step).0;
        let numeric_beta = (plus - minus) / (2.0 * step);
        assert!(
            (d_alpha - numeric_alpha).abs() < 1e-9,
            "ρ_α = {alpha}, ρ_β = {beta}: dε/dρ_α {d_alpha:.12} vs 差分 {numeric_alpha:.12}"
        );
        assert!(
            (d_beta - numeric_beta).abs() < 1e-9,
            "ρ_α = {alpha}, ρ_β = {beta}: dε/dρ_β {d_beta:.12} vs 差分 {numeric_beta:.12}"
        );
    }
}

#[test]
fn ferromagnetic_limit_uses_the_ferro_parameter_set() {
    let term = svwn5();
    // ρ_β = 0（ζ = 1）：能量必须等于铁磁参数集的闭壳层公式值。
    let ferro_only = LdaSvwn5::new(FERRO);
    for rho in [0.01, 0.1, 1.0, 5.0] {
        let (spin, _, _) = term.spin_energy_at(rho, 0.0);
        let mut closed = [0.0_f64];
        ferro_only.energy_density(&DensityBatch::from_rho(&[rho]), &mut closed);
        assert!(
            (spin - closed[0]).abs() < 1e-10,
            "ρ = {rho}: ζ=1 应当是铁磁值 {:.15}，实测 {spin:.15}",
            closed[0]
        );
    }
}

#[test]
fn correlation_energy_increases_with_polarization() {
    let term = svwn5();
    let total = 0.5;
    let unpolarized = term.spin_energy_at(0.5 * total, 0.5 * total).0;
    let polarized = term.spin_energy_at(0.75 * total, 0.25 * total).0;
    let fully = term.spin_energy_at(total, 0.0).0;
    assert!(
        unpolarized < polarized && polarized < fully,
        "VWN5 的相关能应当随极化增大（{unpolarized:.6} < {polarized:.6} < {fully:.6}）"
    );
}

#[test]
fn missing_spin_parameters_are_an_error_not_a_guess() {
    let closed_only = LdaSvwn5::new(PARA);
    assert!(!closed_only.supports_spin());
    let alpha = [0.25_f64];
    let beta = [0.25_f64];
    let batch = SpinDensityBatch::new(&alpha, &beta);
    let mut out = [0.0_f64];
    match closed_only.energy_density_spin(&batch, &mut out) {
        Err(XcError::SpinUnsupported(message)) => {
            assert!(message.contains("ferromagnetic"), "{message}")
        }
        other => panic!("期望 SpinUnsupported，得到 {other:?}"),
    }
}

#[test]
fn zeta_is_reported_and_clamped() {
    let alpha = [0.3_f64, 0.0, 1.0];
    let beta = [0.1, 0.0, 0.0];
    let batch = SpinDensityBatch::new(&alpha, &beta);
    assert!((batch.zeta(0) - 0.5).abs() < 1e-12);
    assert_eq!(batch.zeta(1), 0.0, "空区域 ζ = 0");
    assert_eq!(batch.zeta(2), 1.0);
    assert_eq!(batch.len(), 3);
}
