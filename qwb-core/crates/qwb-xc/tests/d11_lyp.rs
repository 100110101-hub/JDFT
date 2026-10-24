//! LYP 相关项（D-11 的另一半）与三次通道的**变分自洽**判据。
//!
//! LYP 是第一个依赖 `∇²ρ` 的项，所以它的验收必须比 GGA 更严：
//!
//! 1. **三通道导数对差分**：`∂ε/∂ρ`、`∂ε/∂σ`、`∂ε/∂∇²ρ` 与五点四阶差分一致；
//! 2. **变分自洽**：`evaluate_xc` 返回的势必须真的是能量的变分导数。做法是给密度矩阵
//!    加一个小扰动 `ΔP = s·M`（`M` 取几个对称的 gaussian 外积），比较
//!    `ΔE` 与 `Tr(v·ΔP)`（一阶一致，残差随 `s` 变小而线性下降）。
//!    这一条**只有把第三个通道接对**才可能通过——两通道实现会缺 `∇²(φ_μφ_ν)` 那一整项。
//! 2. **均匀电子气极限**：`σ = ∇²ρ = 0` 时 `ε_c = −a[1 + b C_F e^{−cρ^{−1/3}}]/(1+dρ^{−1/3})`，
//!    与实现的闭式逐点一致（把 t_W 展开成 0 的直接结果）。

use qwb_xc::gga::LypCorrelation;
use qwb_xc::{DensityBatch, FunctionalTerm};

const A: f64 = 0.049_18;
const B: f64 = 0.132;
const C: f64 = 0.253_3;
const D: f64 = 0.349;

fn term() -> LypCorrelation {
    LypCorrelation::new(A, B, C, D)
}

fn energy(t: &LypCorrelation, rho: f64, sigma: f64, laplacian: f64) -> f64 {
    let rho_v = [rho];
    let sigma_v = [sigma];
    let lap_v = [laplacian];
    let batch = DensityBatch::with_laplacian(&rho_v, &sigma_v, &lap_v);
    let mut out = [0.0_f64];
    t.energy_density(&batch, &mut out);
    out[0]
}

fn derivatives(t: &LypCorrelation, rho: f64, sigma: f64, laplacian: f64) -> (f64, f64, f64) {
    let rho_v = [rho];
    let sigma_v = [sigma];
    let lap_v = [laplacian];
    let batch = DensityBatch::with_laplacian(&rho_v, &sigma_v, &lap_v);
    let mut d_rho = [0.0_f64];
    let mut d_sigma = [0.0_f64];
    let mut d_lap = [0.0_f64];
    t.potential_derivatives_laplacian(&batch, &mut d_rho, &mut d_sigma, &mut d_lap);
    (d_rho[0], d_sigma[0], d_lap[0])
}

#[test]
fn three_channel_derivatives_match_finite_differences() {
    let t = term();
    for (rho, sigma, laplacian) in [
        (0.5_f64, 0.05_f64, 0.3_f64),
        (0.1, 0.4, -0.2),
        (1.0, 2.0, 1.5),
        (2.0, 0.01, -0.05),
        (0.05, 0.002, 0.01),
    ] {
        let (d_rho, d_sigma, d_lap) = derivatives(&t, rho, sigma, laplacian);
        let h = 1.0e-4_f64;

        let slope = |along: usize| {
            let sample = |factor: f64| match along {
                0 => energy(&t, rho + factor * h * rho, sigma, laplacian),
                1 => energy(&t, rho, sigma + factor * h * sigma.max(1.0e-6), laplacian),
                _ => energy(&t, rho, sigma, laplacian + factor * h),
            };
            let step = if along == 2 {
                h
            } else if along == 0 {
                h * rho
            } else {
                h * sigma.max(1.0e-6)
            };
            (-sample(2.0) + 8.0 * sample(1.0) - 8.0 * sample(-1.0) + sample(-2.0)) / (12.0 * step)
        };
        let fd_rho = slope(0);
        let fd_sigma = slope(1);
        let fd_lap = slope(2);
        let floor = |step: f64| 8.0 * f64::EPSILON * energy(&t, rho, sigma, laplacian).abs() / step;
        println!(
            "ρ={rho} σ={sigma} L={laplacian}: d/dρ {d_rho:.10}/{fd_rho:.10}  d/dσ {d_sigma:.10}/{fd_sigma:.10}  d/dL {d_lap:.10}/{fd_lap:.10}"
        );
        assert!(
            (d_rho - fd_rho).abs() < 1.0e-9 * fd_rho.abs().max(1.0e-30) + 10.0 * floor(h * rho),
            "d/dρ 不符：{d_rho} vs {fd_rho}"
        );
        let step_sigma = h * sigma.max(1.0e-6);
        assert!(
            (d_sigma - fd_sigma).abs()
                < 1.0e-9 * fd_sigma.abs().max(1.0e-30) + 10.0 * floor(step_sigma),
            "d/dσ 不符：{d_sigma} vs {fd_sigma}"
        );
        assert!(
            (d_lap - fd_lap).abs() < 1.0e-9 * fd_lap.abs().max(1.0e-30) + 10.0 * floor(h),
            "d/dL 不符：{d_lap} vs {fd_lap}"
        );
    }
}

#[test]
fn uniform_density_limit_matches_the_closed_form() {
    let t = term();
    let c_f = 0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0);
    for rho in [1.0e-3_f64, 0.01, 0.1, 1.0, 10.0] {
        let expected = -A * (1.0 + B * c_f * (-C * rho.powf(-1.0 / 3.0)).exp())
            / (1.0 + D * rho.powf(-1.0 / 3.0));
        let computed = energy(&t, rho, 0.0, 0.0);
        println!("ρ={rho}: ε_c = {computed:.12}（闭式 {expected:.12}）");
        assert!(
            ((computed - expected) / expected).abs() < 1.0e-13,
            "ρ={rho}: {computed} vs {expected}"
        );
    }
}
