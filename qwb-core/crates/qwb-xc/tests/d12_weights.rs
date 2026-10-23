//! 加权组件（D-12 的前置）：`ScaledTerm` 必须对**能量与两个导数**用同一个权重。
use qwb_xc::{DensityBatch, FunctionalTerm, LdaSlaterX, LdaSvwn5, scale_term};

#[test]
fn scaled_terms_scale_energy_and_both_derivatives_consistently() {
    let rho = [0.05_f64, 0.4, 2.0];
    let sigma = [0.0_f64; 3];
    let batch = DensityBatch::new(&rho, &sigma);
    let weight = 0.8_f64;

    let plain: Box<dyn FunctionalTerm> = Box::new(LdaSlaterX::new(2.0 / 3.0));
    let scaled = scale_term(Box::new(LdaSlaterX::new(2.0 / 3.0)), weight);

    let mut e_plain = vec![0.0; rho.len()];
    let mut e_scaled = vec![0.0; rho.len()];
    plain.energy_density(&batch, &mut e_plain);
    scaled.energy_density(&batch, &mut e_scaled);
    let mut d_plain = vec![0.0; rho.len()];
    let mut d_scaled = vec![0.0; rho.len()];
    let mut s_plain = vec![0.0; rho.len()];
    let mut s_scaled = vec![0.0; rho.len()];
    plain.potential_derivatives(&batch, &mut d_plain, &mut s_plain);
    scaled.potential_derivatives(&batch, &mut d_scaled, &mut s_scaled);

    for i in 0..rho.len() {
        assert!(
            (e_scaled[i] - weight * e_plain[i]).abs() < 1.0e-15,
            "能量缩放不对"
        );
        assert!(
            (d_scaled[i] - weight * d_plain[i]).abs() < 1.0e-15,
            "dε/dρ 缩放不对"
        );
        assert!(
            (s_scaled[i] - weight * s_plain[i]).abs() < 1.0e-15,
            "dε/dσ 缩放不对"
        );
    }
    // 权重为 1 时不应改变任何东西
    let unscaled = scale_term(
        Box::new(LdaSvwn5::new((0.0310907, -0.409286, 13.0720, 42.7198))),
        1.0,
    );
    let mut a = vec![0.0; rho.len()];
    let mut b = vec![0.0; rho.len()];
    unscaled.energy_density(&batch, &mut a);
    LdaSvwn5::new((0.0310907, -0.409286, 13.0720, 42.7198)).energy_density(&batch, &mut b);
    for i in 0..rho.len() {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "权重 1 必须逐位不变");
    }
}

#[test]
fn a_half_weighted_exchange_gives_half_the_exchange_energy() {
    // 复合泛函的线性性：0.5·LDA-X 的交换能必须正好是一半。
    let rho = [0.3_f64, 1.2];
    let sigma = [0.0_f64; 2];
    let batch = DensityBatch::new(&rho, &sigma);
    let mut half = vec![0.0; rho.len()];
    scale_term(Box::new(LdaSlaterX::new(2.0 / 3.0)), 0.5).energy_density(&batch, &mut half);
    let mut full = vec![0.0; rho.len()];
    LdaSlaterX::new(2.0 / 3.0).energy_density(&batch, &mut full);
    for i in 0..rho.len() {
        assert!((half[i] - 0.5 * full[i]).abs() < 1.0e-15);
    }
}
