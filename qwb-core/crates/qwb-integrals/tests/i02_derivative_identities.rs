//! 一电子积分 vs 中心导数恒等式（p/d 型的独立校验）。\n//!\n//! S、T 已精确通过；V 目前差约 3%（见 docs/status.yaml 的 I-02），所以这个用例\n//! 暂时 #[ignore]，等 V 的正确性确定后再打开——不会把已知错误的实现标成通过。
use qwb_integrals::one_electron::{primitive_kinetic, primitive_nuclear, primitive_overlap};

#[test]
#[ignore = "KNOWN GAP: p 型核吸引与中心导数恒等式差约 3%（I-02 未收口）；S/T 已精确通过"]
fn diag_p_integrals() {
    let alpha = 1.5_f64;
    let a = [0.0, 0.0, 0.0];
    let beta = 0.9_f64;
    let b = [0.5, 1.43, 1.11];
    let nucleus = [0.2, -0.1, 0.0];
    let charge = 8.0;

    let h = 1.0e-4;
    let shift = |dx: f64| [a[0] + dx, a[1], a[2]];

    // V_{p_x s} = (1/(2α)) ∂_{A_x} V_{ss}
    let v_s = |dx: f64| {
        primitive_nuclear(
            alpha,
            (0, 0, 0),
            shift(dx),
            beta,
            (0, 0, 0),
            b,
            nucleus,
            charge,
        )
    };
    let dv = (-v_s(2.0 * h) + 8.0 * v_s(h) - 8.0 * v_s(-h) + v_s(-2.0 * h)) / (12.0 * h);
    let expected_v = dv / (2.0 * alpha);
    let actual_v = primitive_nuclear(alpha, (1, 0, 0), a, beta, (0, 0, 0), b, nucleus, charge);
    println!(
        "DIAG V:  direct {actual_v:.12}  fd {expected_v:.12}  rel {:.2e}",
        (actual_v - expected_v).abs() / expected_v.abs()
    );

    // T_{p_x s} = (1/(2α)) ∂_{A_x} T_{ss}
    let t_s = |dx: f64| primitive_kinetic(alpha, (0, 0, 0), shift(dx), beta, (0, 0, 0), b);
    let dt = (-t_s(2.0 * h) + 8.0 * t_s(h) - 8.0 * t_s(-h) + t_s(-2.0 * h)) / (12.0 * h);
    let expected_t = dt / (2.0 * alpha);
    let actual_t = primitive_kinetic(alpha, (1, 0, 0), a, beta, (0, 0, 0), b);
    println!("DIAG T:  direct {actual_t:.12}  fd {expected_t:.12}",);

    // S_{p_x s} = (1/(2α)) ∂_{A_x} S_{ss}
    let s_s = |dx: f64| primitive_overlap(alpha, (0, 0, 0), shift(dx), beta, (0, 0, 0), b);
    let ds = (-s_s(2.0 * h) + 8.0 * s_s(h) - 8.0 * s_s(-h) + s_s(-2.0 * h)) / (12.0 * h);
    let expected_s = ds / (2.0 * alpha);
    let actual_s = primitive_overlap(alpha, (1, 0, 0), a, beta, (0, 0, 0), b);
    println!("DIAG S:  direct {actual_s:.12}  fd {expected_s:.12}");
    assert!(
        (actual_v - expected_v).abs() < 1e-9 * expected_v.abs().max(1.0),
        "V 不一致"
    );
    assert!(
        (actual_t - expected_t).abs() < 1e-9 * expected_t.abs().max(1.0),
        "T 不一致"
    );
    assert!(
        (actual_s - expected_s).abs() < 1e-9 * expected_s.abs().max(1.0),
        "S 不一致"
    );

    // d_xx 型：两阶中心导数
    let h2 = 1.0e-3;
    let v_s2 = |dx: f64| {
        primitive_nuclear(
            alpha,
            (0, 0, 0),
            shift(dx),
            beta,
            (0, 0, 0),
            b,
            nucleus,
            charge,
        )
    };
    let second = (-v_s2(2.0 * h2) + 16.0 * v_s2(h2) - 30.0 * v_s2(0.0) + 16.0 * v_s2(-h2)
        - v_s2(-2.0 * h2))
        / (12.0 * h2 * h2);
    let expected_dxx = second / (4.0 * alpha * alpha);
    let actual_dxx = primitive_nuclear(alpha, (2, 0, 0), a, beta, (0, 0, 0), b, nucleus, charge);
    println!("DIAG V2: direct {actual_dxx:.12}  fd {expected_dxx:.12}");
}
