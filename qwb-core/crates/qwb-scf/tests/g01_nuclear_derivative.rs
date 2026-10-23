//! 诊断：`nuclear_center_derivative` 与「对 primitive_nuclear 直接中心差分」逐点对照。
use qwb_integrals::one_electron::primitive_nuclear;
use qwb_scf::analytic_gradient::nuclear_center_derivative;

#[test]
fn diag_nuclear_center_derivative() {
    let cases = [
        (1.2_f64, [0.0, 0.0, 0.0], 0.9_f64, [0.0, 0.0, 1.4], [0.2, -0.1, 0.0], 1.0_f64),
        (1.5, [0.3, 0.1, -0.2], 0.8, [0.0, 0.4, 1.1], [0.1, 0.2, 0.3], 8.0),
        (0.6, [0.0, 0.0, 0.0], 0.6, [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], 1.0),
    ];
    let h = 1.0e-6_f64;
    for (alpha, a, beta, b, c, z) in cases {
        for axis in 0..3 {
            let analytic = nuclear_center_derivative(alpha, a, beta, b, c, z, axis);
            let mut cp = c;
            cp[axis] += h;
            let mut cm = c;
            cm[axis] -= h;
            let plus = primitive_nuclear(alpha, (0, 0, 0), a, beta, (0, 0, 0), b, cp, z);
            let minus = primitive_nuclear(alpha, (0, 0, 0), a, beta, (0, 0, 0), b, cm, z);
            let numeric = (plus - minus) / (2.0 * h);
            println!(
                "α={alpha} β={beta} axis={axis}: 解析 {analytic:>14.9}  差分 {numeric:>14.9}  比 {:.6}",
                if numeric != 0.0 { analytic / numeric } else { f64::NAN }
            );
        }
    }
}
