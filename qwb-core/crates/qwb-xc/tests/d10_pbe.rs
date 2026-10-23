//! D-10 验收：PBE 交换/相关的解析导数与有限差分差 < 1e-9，且 σ→0 时退化为 LDA。
use qwb_xc::gga::{PbeCorrelation, PbeExchange};
use qwb_xc::{DensityBatch, FunctionalTerm, LdaSlaterX, LdaSvwn5};

const VWN5: (f64, f64, f64, f64) = (0.0310907, -0.409286, 13.0720, 42.7198);

fn exchange() -> PbeExchange {
    PbeExchange::new(0.804, 0.21951)
}

fn correlation() -> PbeCorrelation {
    PbeCorrelation::new(VWN5, 0.066_724_550_603_149_22, 0.031_090_690_869_654_9)
}

#[test]
fn pbe_reduces_to_lda_when_sigma_vanishes() {
    let rho = [0.02_f64, 0.1, 1.0, 5.0];
    let sigma = vec![0.0_f64; rho.len()];

    let mut pbe_x = vec![0.0; rho.len()];
    exchange().energy_density(&DensityBatch::new(&rho, &sigma), &mut pbe_x);
    let mut lda_x = vec![0.0; rho.len()];
    LdaSlaterX::new(2.0 / 3.0).energy_density(&DensityBatch::from_rho(&rho), &mut lda_x);

    let mut pbe_c = vec![0.0; rho.len()];
    correlation().energy_density(&DensityBatch::new(&rho, &sigma), &mut pbe_c);
    let mut lda_c = vec![0.0; rho.len()];
    LdaSvwn5::new(VWN5).energy_density(&DensityBatch::from_rho(&rho), &mut lda_c);

    for i in 0..rho.len() {
        println!(
            "rho = {:>5}: ε_x PBE {:.12} vs LDA {:.12}; ε_c PBE {:.12} vs LDA {:.12}",
            rho[i], pbe_x[i], lda_x[i], pbe_c[i], lda_c[i]
        );
        assert!(
            (pbe_x[i] - lda_x[i]).abs() < 1.0e-12,
            "σ=0 时 PBE 交换必须等于 LDA（rho = {}）",
            rho[i]
        );
        assert!(
            (pbe_c[i] - lda_c[i]).abs() < 1.0e-12,
            "σ=0 时 PBE 相关必须等于 LDA（rho = {}）",
            rho[i]
        );
    }
}

#[test]
fn pbe_exchange_enhancement_matches_the_formula_at_s_equals_one() {
    // s = 1 ⇔ σ = 4 k_F² ρ²；此时 F_x = 1 + κ − κ/(1 + μ/κ)。
    let rho = [0.05_f64, 0.3, 2.0];
    let kappa = 0.804_f64;
    let mu = 0.21951_f64;
    let expected_factor = 1.0 + kappa - kappa / (1.0 + mu / kappa);
    let sigma: Vec<f64> = rho
        .iter()
        .map(|r| {
            let kf = (3.0 * std::f64::consts::PI * std::f64::consts::PI * r).powf(1.0 / 3.0);
            4.0 * kf * kf * r * r
        })
        .collect();
    let mut pbe_x = vec![0.0; rho.len()];
    exchange().energy_density(&DensityBatch::new(&rho, &sigma), &mut pbe_x);
    let mut lda_x = vec![0.0; rho.len()];
    LdaSlaterX::new(2.0 / 3.0).energy_density(&DensityBatch::from_rho(&rho), &mut lda_x);
    for i in 0..rho.len() {
        let factor = pbe_x[i] / lda_x[i];
        println!(
            "rho = {}: F_x = {factor:.10}（解析 {expected_factor:.10}）",
            rho[i]
        );
        assert!(
            (factor - expected_factor).abs() < 1.0e-10,
            "F_x 与解析式不符：{factor} vs {expected_factor}"
        );
    }
}

#[test]
fn pbe_derivatives_match_finite_differences() {
    let rho = [0.02_f64, 0.15, 0.8, 3.0];
    let sigma = [1.0e-4_f64, 0.01, 0.5, 4.0];

    let terms: Vec<(&str, Box<dyn FunctionalTerm>)> = vec![
        ("GgaPbeX", Box::new(exchange())),
        ("GgaPbeC", Box::new(correlation())),
    ];
    for (label, term) in terms {
        let mut d_rho = vec![0.0; rho.len()];
        let mut d_sigma = vec![0.0; rho.len()];
        term.potential_derivatives(&DensityBatch::new(&rho, &sigma), &mut d_rho, &mut d_sigma);

        for i in 0..rho.len() {
            // 五点四阶 stencil：二阶中心差分的截断 ~h²，对小 σ 不够（会到 1e-8）；
            // 五点 stencil 的截断 ~h⁴，配 h = 1e-3·尺度 就能压到 1e-12。
            let value_at = |r: f64, s: f64| {
                let mut out = vec![0.0_f64; 1];
                term.energy_density(&DensityBatch::new(&[r], &[s]), &mut out);
                out[0]
            };
            let h_rho = 1.0e-3 * rho[i];
            let numeric_rho = (value_at(rho[i] - 2.0 * h_rho, sigma[i])
                - 8.0 * value_at(rho[i] - h_rho, sigma[i])
                + 8.0 * value_at(rho[i] + h_rho, sigma[i])
                - value_at(rho[i] + 2.0 * h_rho, sigma[i]))
                / (12.0 * h_rho);
            let h_sigma = 1.0e-3 * sigma[i];
            let numeric_sigma = (value_at(rho[i], sigma[i] - 2.0 * h_sigma)
                - 8.0 * value_at(rho[i], sigma[i] - h_sigma)
                + 8.0 * value_at(rho[i], sigma[i] + h_sigma)
                - value_at(rho[i], sigma[i] + 2.0 * h_sigma))
                / (12.0 * h_sigma);

            let scale_rho = numeric_rho.abs().max(1.0);
            let scale_sigma = numeric_sigma.abs().max(1.0e-6);
            let error_rho = (d_rho[i] - numeric_rho).abs() / scale_rho;
            let error_sigma = (d_sigma[i] - numeric_sigma).abs() / scale_sigma;
            println!(
                "{label} rho={:>5} σ={:>7}: d/dρ 解析 {:>13.9} 差分 {:>13.9} rel {error_rho:.2e} | d/dσ 解析 {:>13.9} 差分 {:>13.9} rel {error_sigma:.2e}",
                rho[i], sigma[i], d_rho[i], numeric_rho, d_sigma[i], numeric_sigma
            );
            assert!(
                error_rho < 1.0e-9,
                "{label}: dε/dρ 与差分差 {error_rho:.2e}（rho = {}）",
                rho[i]
            );
            assert!(
                error_sigma < 1.0e-9,
                "{label}: dε/dσ 与差分差 {error_sigma:.2e}（rho = {}）",
                rho[i]
            );
        }
    }
}
