//! FX-02（部分）：**M06-L / M06-2X 交换**与 libxc 的逐点对照。
//!
//! 判据：能量密度与 `∂e/∂ρ`、`∂e/∂σ`、`∂e/∂τ` 都与 libxc 一致。
//! 参考数据由 `scripts/pyscf_m06_exchange_reference.py` 生成。
//!
//! 三个内置自检（不依赖参考数据）：
//! 1. 均匀电子气（σ=0、τ=τ_unif）处 M06-L 交换**必须逐位退回 Slater**；
//! 2. M06-2X 的 `a[0] + 0.54 = 1`；
//! 3. 两族都只声明 `(ρ, σ, τ)`，不要 `∇²ρ`。

use qwb_xc::{
    DensityBatch, FunctionalTerm, Requirements,
    meta_gga::{M06Exchange, M06ExchangeParameters},
};

const M06L_SERIES: [f64; 12] = [
    0.398_775_6,
    0.254_821_9,
    0.392_399_4,
    -2.103_655,
    -6.302_147,
    10.976_15,
    30.972_73,
    -23.184_89,
    -56.734_80,
    21.603_64,
    34.218_14,
    -9.049_762,
];
const M06L_VS98: [f64; 6] = [
    0.601_224_4,
    0.004_748_822,
    -0.008_635_108,
    -0.000_009_308_062,
    0.000_044_828_11,
    0.0,
];
const M062X_SERIES: [f64; 12] = [
    4.600_000e-01,
    -2.206_052e-01,
    -9.431_788e-02,
    2.164_494e00,
    -2.556_466e00,
    -1.422_133e01,
    1.555_044e01,
    3.598_078e01,
    -2.722_754e01,
    -3.924_093e01,
    1.522_808e01,
    1.522_227e01,
];

fn m06l() -> M06Exchange {
    M06Exchange::new(
        "MggaXM06L",
        M06ExchangeParameters::new(M06L_SERIES, M06L_VS98, 0.001_867_26, true),
    )
}

fn m062x() -> M06Exchange {
    M06Exchange::new(
        "MggaXM062x",
        M06ExchangeParameters::new(M062X_SERIES, [0.0; 6], 0.0, false),
    )
}

fn evaluate(term: &dyn FunctionalTerm, rho: f64, sigma: f64, tau: f64) -> (f64, f64, f64, f64) {
    let rho_holder = [rho];
    let sigma_holder = [sigma];
    let tau_holder = [tau];
    let batch = DensityBatch::with_tau(&rho_holder, &sigma_holder, &tau_holder);
    let (mut epsilon, mut d_rho, mut d_sigma) = ([0.0_f64], [0.0_f64], [0.0_f64]);
    let (mut d_laplacian, mut d_tau) = ([0.0_f64], [0.0_f64]);
    term.energy_density(&batch, &mut epsilon);
    term.potential_derivatives_full(
        &batch,
        &mut d_rho,
        &mut d_sigma,
        &mut d_laplacian,
        &mut d_tau,
    );
    (epsilon[0], d_rho[0], d_sigma[0], d_tau[0])
}

fn check(component: &str, term: &dyn FunctionalTerm, tolerance: f64) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/reference/meta_gga_exchange.csv"
    );
    let text = std::fs::read_to_string(path).expect("read meta_gga_exchange.csv");
    let mut checked = 0usize;
    let mut worst = 0.0_f64;
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields[0] != component {
            continue;
        }
        let field = |index: usize| fields[index].parse::<f64>().expect("numeric");
        let (rho, sigma, tau) = (field(1), field(2), field(3));
        let (epsilon, d_rho, d_sigma, d_tau) = evaluate(term, rho, sigma, tau);
        // 我们给的是**每电子** ε 及其偏导；libxc 给的是每体积 e = ρε 及其偏导：
        //   ∂e/∂ρ = ε + ρ·∂ε/∂ρ、∂e/∂σ = ρ·∂ε/∂σ、∂e/∂τ = ρ·∂ε/∂τ
        let energy = epsilon * rho;
        let v_rho = epsilon + rho * d_rho;
        let v_sigma = rho * d_sigma;
        let v_tau = rho * d_tau;
        let expected = (field(4), field(5), field(6), field(7));
        let deviation = (energy - expected.0)
            .abs()
            .max((v_rho - expected.1).abs())
            .max((v_sigma - expected.2).abs())
            .max((v_tau - expected.3).abs());
        worst = worst.max(deviation);
        assert!(
            deviation < tolerance,
            "{component} ρ={rho} σ={sigma:.6e} τ={tau:.6e}: e {energy:.12e} vs {:.12e}｜\
             vρ {v_rho:.12e} vs {:.12e}｜vσ {v_sigma:.12e} vs {:.12e}｜vτ {v_tau:.12e} vs {:.12e}",
            expected.0,
            expected.1,
            expected.2,
            expected.3
        );
        checked += 1;
    }
    assert!(checked >= 8, "{component}: 只有 {checked} 行");
    println!("{component}: {checked} 行一致，最大偏差 {worst:.3e}");
}

#[test]
fn m06_l_exchange_matches_libxc() {
    check("mgga_x_m06_l", &m06l(), 1.0e-9);
}

#[test]
fn m06_2x_exchange_matches_libxc() {
    check("hyb_mgga_x_m06_2x", &m062x(), 1.0e-9);
}

#[test]
fn m06_l_exchange_recovers_lda_x_at_the_uniform_gas() {
    let k = 0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0);
    for rho in [0.1_f64, 0.5, 1.0, 2.0] {
        let tau = k * rho.powf(5.0 / 3.0);
        let (epsilon, _, _, _) = evaluate(&m06l(), rho, 0.0, tau);
        let expected = -0.75 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0) * rho.powf(1.0 / 3.0);
        assert!(
            (epsilon - expected).abs() < 1.0e-10,
            "ρ={rho}: ε = {epsilon:.12} vs LDA {expected:.12}"
        );
    }
}

#[test]
fn m06_2x_series_is_complementary_to_the_exact_exchange() {
    assert!((M062X_SERIES[0] + 0.54 - 1.0).abs() < 1.0e-12);
    assert!((M06L_SERIES[0] + M06L_VS98[0] - 1.0).abs() < 1.0e-12);
    let terms: Vec<Box<dyn FunctionalTerm>> = vec![Box::new(m06l()), Box::new(m062x())];
    for term in &terms {
        let requirements: Requirements = term.requires();
        assert!(requirements.kinetic_density && requirements.sigma);
        assert!(!requirements.laplacian);
    }
}
