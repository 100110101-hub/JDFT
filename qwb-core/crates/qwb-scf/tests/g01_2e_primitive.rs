//! 诊断：`d_ssss` 与其槽位置换是否等于原函数积分对各自中心的中心差分。
use qwb_integrals::two_electron::primitive_ssss;
use qwb_scf::analytic_gradient::{d_ssss, d_ssss_slot};

#[test]
fn diag_d_ssss_slots() {
    let alpha = 1.1_f64;
    let beta = 0.7_f64;
    let gamma = 0.9_f64;
    let delta = 1.3_f64;
    let a = [0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 1.3];
    let c = [1.1, -0.2, 0.4];
    let d = [-0.3, 0.9, 1.7];
    let prims = [(alpha, a), (beta, b), (gamma, c), (delta, d)];
    let h = 1.0e-6_f64;
    let mut worst = 0.0_f64;
    for slot in 0..4 {
        for axis in 0..3 {
            let analytic = d_ssss_slot(&prims, slot, axis);
            let mut lo = prims;
            let mut hi = prims;
            lo[slot].1[axis] -= h;
            hi[slot].1[axis] += h;
            let f = |q: &[(f64, [f64; 3]); 4]| {
                primitive_ssss(
                    q[0].0, q[0].1, q[1].0, q[1].1, q[2].0, q[2].1, q[3].0, q[3].1,
                )
            };
            let numeric = (f(&hi) - f(&lo)) / (2.0 * h);
            let diff = (analytic - numeric).abs();
            worst = worst.max(diff);
            println!(
                "slot={slot} axis={axis}: 解析 {analytic:>15.10}  差分 {numeric:>15.10}  差 {diff:.3e}"
            );
        }
    }
    println!("最大差 {worst:.3e}");
    let direct = d_ssss(alpha, a, beta, b, gamma, c, delta, d, 1);
    println!(
        "d_ssss 原样(axis=1) {direct:>15.10} vs slot0 {:.10}",
        d_ssss_slot(&prims, 0, 1)
    );
}
