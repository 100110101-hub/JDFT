//! I-02：核吸引（任意角动量）的正确性，用一个**已被独立判定的**参照。
//!
//! 关键恒等式：把指数 γ 的归一化 s 函数压到核 C 上，则
//!
//! ```text
//! (ab|1/r_C) = lim_{γ→∞} (ab| φ_C φ_C )
//! ```
//!
//! 右侧是**两电子积分**，而 2e 路径已由 `i03_general_l.rs` 的独立数值路径逐分量
//! 判定到 1e-10。所以这条等式把 2e 的可信度「传导」给核吸引，不需要再写一套
//! 一电子参照。
//!
//! 这条测试抓到的真 bug：旧的 `primitive_nuclear` 把两中心的角动量直接相加、
//! 假定多项式中心在 P 上，丢掉了 (P−A)、(P−B)——s 型看不出来（所以 H2/STO-3G
//! 一直精确），p/d 型全错；改写后又发现 `∂^t F₀` 前**不该**有 (−1)^t
//! （2e 那边 (−1) 落在 ket 的 E 系数上，两者约定不同）。修复后 p_z 分量与参照
//! 差 1.7e-11。

use qwb_integrals::hermite2e::primitive_component;
use qwb_integrals::one_electron::{primitive_norm, primitive_nuclear};

/// 用「极窄 s 函数」的 2e 积分给出核吸引的量级（正数）。
fn point_charge_limit(
    powers: (u32, u32, u32),
    a: [f64; 3],
    alpha: f64,
    b: [f64; 3],
    beta: f64,
    c: [f64; 3],
) -> f64 {
    let gamma = 1.0e5_f64;
    let norm = primitive_norm((0, 0, 0), gamma);
    primitive_component(
        [powers.0, powers.1, powers.2],
        a,
        alpha,
        [0, 0, 0],
        b,
        beta,
        [0, 0, 0],
        c,
        gamma,
        [0, 0, 0],
        c,
        gamma,
    ) * norm
        * norm
}

#[test]
fn nuclear_attraction_matches_the_point_charge_limit() {
    let a = [0.0, 0.0, 0.0];
    let b = [0.5, 1.43, 1.11];
    let c = [0.2, -0.1, 0.0];
    let alpha = 1.5_f64;
    let beta = 0.9_f64;

    let cases: [((u32, u32, u32), &str); 5] = [
        ((0, 0, 0), "s"),
        ((1, 0, 0), "p_x"),
        ((0, 1, 0), "p_y"),
        ((0, 0, 1), "p_z"),
        ((2, 0, 0), "d_xx"),
    ];
    for (powers, label) in cases {
        let from_2e = point_charge_limit(powers, a, alpha, b, beta, c);
        // 核吸引是吸引的：V = −Z·lim(2e)
        let from_1e = primitive_nuclear(alpha, powers, a, beta, (0, 0, 0), b, c, 1.0);
        let relative = (from_1e + from_2e).abs() / from_2e.abs().max(1.0e-30);
        println!("{label:<5} 1e {from_1e:>15.10}  2e 极限 {from_2e:>15.10}  rel {relative:.2e}");
        assert!(
            relative < 1.0e-5,
            "{label}: 核吸引 {from_1e} 与点电荷极限 {from_2e} 相对差 {relative:.2e}"
        );
        assert!(from_2e.abs() > 1.0e-4, "{label}: 参照值太小，抓不住错");
    }
}

#[test]
fn nuclear_attraction_is_symmetric_in_the_two_centres() {
    // (ab|C) = (ba|C)：交换两中心（连同指数）不改变积分。
    let a = [0.0, 0.0, 0.0];
    let b = [0.5, 1.43, 1.11];
    let c = [0.2, -0.1, 0.0];
    let forward = primitive_nuclear(1.5, (1, 0, 0), a, 0.9, (0, 1, 0), b, c, 1.0);
    let backward = primitive_nuclear(0.9, (0, 1, 0), b, 1.5, (1, 0, 0), a, c, 1.0);
    assert!(
        (forward - backward).abs() < 1.0e-12 * forward.abs().max(1.0),
        "(ab|C) = {forward} ≠ (ba|C) = {backward}"
    );
}
