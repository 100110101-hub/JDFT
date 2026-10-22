//! 数值 oracle：用 **Gauss–Hermite 乘积网格**独立重算同一个积分。
//!
//! # 为什么这个 oracle 是"精确"的
//!
//! 重叠与动能的被积函数是 **多项式 × 高斯**：`n` 点 Gauss–Hermite 对
//! `2n−1` 次多项式**精确成立**，所以 `n = 40` 时对照能到 1e-14 量级（不是近似对照）。
//! 这让 I-02 的 S/T 有了与解析闭式完全独立的第二条路径。
//!
//! # 为什么 `1/r` 核不在这里用数值对照
//!
//! 核吸引的被积函数带 `1/|r−C|` 奇点，Gauss–Hermite 收敛很慢（实测 s-s 只能到 1e-4），
//! 拿它当"参考答案"会把误差记到实现头上。所以核吸引改用两类**无歧义**的判据：
//!
//! - **解析闭式**（`tests/d02_one_electron.rs` 里的 s-s 公式）；
//! - **宇称**：同中心时 `⟨s|1/r|p_x⟩` 的被积函数对 `x → −x` 是奇函数，**必须精确为 0**。
//!
//! 后者正是上一轮抓出"核吸引 l>0 系数缺方向依赖"的那把尺子。
//!
//! 节点/权重用 **Golub–Welsch**（复用 I-00 的 `qwb_linalg::symmetric_eigen`）。

use std::f64::consts::PI;

use qwb_integrals::one_electron::{primitive_kinetic, primitive_nuclear, primitive_overlap};
use qwb_linalg::{Matrix, symmetric_eigen};

/// Hermite 的节点与权重（`∫ f(t)e^{-t²}dt ≈ Σ w_i f(t_i)`，对 `2n−1` 次多项式精确）。
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jacobi = Matrix::zeros(n, n);
    for i in 1..n {
        let beta = (i as f64 / 2.0).sqrt();
        jacobi[(i - 1, i)] = beta;
        jacobi[(i, i - 1)] = beta;
    }
    let (values, vectors) = symmetric_eigen(&jacobi).expect("symmetric eigen");
    let nodes: Vec<f64> = (0..n).map(|i| values.get(i)).collect();
    let weights: Vec<f64> = (0..n)
        .map(|i| PI.sqrt() * vectors[(0, i)].powi(2))
        .collect();
    (nodes, weights)
}

/// `∫ poly(r) e^{-p|r−P|²} dV`（**没有** `1/r` 核，因此数值上是精确的）。
fn quadrature_gaussian(
    p: f64,
    product_center: [f64; 3],
    polynomial: &dyn Fn([f64; 3]) -> f64,
    n: usize,
) -> f64 {
    let (nodes, weights) = gauss_hermite(n);
    let scale = 1.0 / p.sqrt();
    let mut total = 0.0;
    for (i, ti) in nodes.iter().enumerate() {
        let x = product_center[0] + ti * scale;
        for (j, tj) in nodes.iter().enumerate() {
            let y = product_center[1] + tj * scale;
            for (k, tk) in nodes.iter().enumerate() {
                let z = product_center[2] + tk * scale;
                total += weights[i] * weights[j] * weights[k] * polynomial([x, y, z]);
            }
        }
    }
    total / p.powf(1.5)
}

fn product_center(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> ([f64; 3], f64) {
    let p = alpha + beta;
    (
        [
            (alpha * a[0] + beta * b[0]) / p,
            (alpha * a[1] + beta * b[1]) / p,
            (alpha * a[2] + beta * b[2]) / p,
        ],
        p,
    )
}

fn gaussian_decay(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> f64 {
    let p = alpha + beta;
    let r2 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>();
    (-alpha * beta / p * r2).exp()
}

/// 自检：oracle 自己必须先对，否则后面的对照没有意义。
#[test]
fn oracle_is_exact_for_polynomials() {
    let (nodes, weights) = gauss_hermite(40);
    let i0: f64 = weights.iter().sum();
    let i2: f64 = weights.iter().zip(&nodes).map(|(w, t)| w * t * t).sum();
    let i4: f64 = weights.iter().zip(&nodes).map(|(w, t)| w * t.powi(4)).sum();
    assert!((i0 - PI.sqrt()).abs() < 1e-14, "{i0}");
    assert!((i2 - PI.sqrt() / 2.0).abs() < 1e-14, "{i2}");
    assert!((i4 - 3.0 * PI.sqrt() / 4.0).abs() < 1e-14, "{i4}");
}

/// **重叠**：解析实现 vs oracle，多条路径、多种角动量，1e-12 级一致。
#[test]
fn overlap_agrees_with_the_exact_oracle() {
    let alpha = 1.1_f64;
    let beta = 0.6_f64;
    let a = [0.0, 0.0, 0.0];
    let b = [0.4, -0.3, 0.9];
    let (center, p) = product_center(alpha, a, beta, b);
    let e_ab = gaussian_decay(alpha, a, beta, b);
    let ax = a[0];
    let ay = a[1];
    let bx = b[0];
    let by = b[1];

    let check = |label: &str,
                 powers_a: (u32, u32, u32),
                 powers_b: (u32, u32, u32),
                 poly: &dyn Fn([f64; 3]) -> f64| {
        let oracle = e_ab * quadrature_gaussian(p, center, poly, 40);
        let analytic = primitive_overlap(alpha, powers_a, a, beta, powers_b, b);
        assert!(
            (oracle - analytic).abs() < 1e-12 * analytic.abs().max(1e-6),
            "S({label}): oracle {oracle:.15} vs analytic {analytic:.15}"
        );
    };

    check("s-s", (0, 0, 0), (0, 0, 0), &move |_r| 1.0);
    check("s-p_x", (0, 0, 0), (1, 0, 0), &move |r| r[0] - bx);
    check("p_x-s", (1, 0, 0), (0, 0, 0), &move |r| r[0] - ax);
    check("p_x-p_y", (1, 0, 0), (0, 1, 0), &move |r| {
        (r[0] - ax) * (r[1] - by)
    });
    check("p_y-p_y", (0, 1, 0), (0, 1, 0), &move |r| {
        (r[1] - ay) * (r[1] - by)
    });
}
/// **动能**：oracle 用显式的 `∇a·∇b` 多项式，与解析实现必须一致。
#[test]
fn kinetic_agrees_with_the_exact_oracle() {
    let alpha = 0.9_f64;
    let beta = 0.7_f64;
    let a = [0.0, 0.0, 0.0];
    let b = [0.3, 0.5, -0.2];
    let (center, p) = product_center(alpha, a, beta, b);
    let e_ab = gaussian_decay(alpha, a, beta, b);

    // 对 s-s：∇a = −2α(r−A)a，∇b = −2β(r−B)b，T = ½∫(∇a·∇b)
    let oracle = e_ab
        * 0.5
        * quadrature_gaussian(
            p,
            center,
            &move |r| {
                let ga = [
                    -2.0 * alpha * (r[0] - a[0]),
                    -2.0 * alpha * (r[1] - a[1]),
                    -2.0 * alpha * (r[2] - a[2]),
                ];
                let gb = [
                    -2.0 * beta * (r[0] - b[0]),
                    -2.0 * beta * (r[1] - b[1]),
                    -2.0 * beta * (r[2] - b[2]),
                ];
                ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2]
            },
            40,
        );
    let analytic = primitive_kinetic(alpha, (0, 0, 0), a, beta, (0, 0, 0), b);
    assert!(
        (oracle - analytic).abs() < 1e-12,
        "T(s,s): oracle {oracle:.15} vs analytic {analytic:.15}"
    );

    // s-p_x：∇ 的 p 分量 = [δ_x − 2β(x−B_x)](x−B_x)... 直接用 ∂a·∂b 的显式形式。
    let bx = b[0];
    let oracle_sp = e_ab
        * 0.5
        * quadrature_gaussian(
            p,
            center,
            &move |r| {
                // a = exp(-α|r-A|²)，b = (x-B_x)exp(-β|r-B|²)
                let ga = [
                    -2.0 * alpha * (r[0] - a[0]),
                    -2.0 * alpha * (r[1] - a[1]),
                    -2.0 * alpha * (r[2] - a[2]),
                ];
                let dx = r[0] - bx;
                let gb = [
                    (1.0 - 2.0 * beta * dx * dx),
                    -2.0 * beta * dx * (r[1] - b[1]),
                    -2.0 * beta * dx * (r[2] - b[2]),
                ];
                // ∂b/∂x = [1 − 2β(x−B)²] exp(…)，∂b/∂y = −2β(x−B)(y−B) exp(…)
                ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2]
            },
            40,
        );
    let analytic_sp = primitive_kinetic(alpha, (0, 0, 0), a, beta, (1, 0, 0), b);
    assert!(
        (oracle_sp - analytic_sp).abs() < 1e-12 * analytic_sp.abs().max(1e-6),
        "T(s,p_x): oracle {oracle_sp:.15} vs analytic {analytic_sp:.15}"
    );
}

/// 核吸引：宇称 + 同中心 `p_x` 对角的解析闭式。
///
/// 这两条都是**精确判据**，不依赖数值容差：
/// - 同中心 `⟨s|1/r|p_x⟩` 的被积函数对 `x → −x` 是奇函数 ⇒ 精确为 0；
/// - 同中心 `⟨p_x|1/r|p_x⟩`：`∫x²e^{−2αr²}/r dV = π/(6α²)`，
///   归一化后 `V = −Z·N_p²·π/(6α²)`。
#[test]
fn nuclear_attraction_satisfies_parity_and_the_p_x_closed_form() {
    let alpha = 0.85_f64;
    let beta = 0.55_f64;
    let a = [0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 0.0]; // 同中心 ⇒ 产品中心在原点
    // 核必须放在**产品中心**（原点）上，宇称判据才成立；放在别处时积分本来就不为 0。
    let center = [0.0, 0.0, 0.0];

    // 宇称：⟨s|1/r|p_x⟩ = 0
    let sp = primitive_nuclear(alpha, (0, 0, 0), a, beta, (1, 0, 0), b, center, 1.0);
    assert!(sp.abs() < 1e-14, "⟨s|1/r|p_x⟩ 必须为 0，实际 {sp}");

    // 宇称：⟨p_x|1/r|p_y⟩ = 0
    let pxpy = primitive_nuclear(alpha, (1, 0, 0), a, beta, (0, 1, 0), b, center, 1.0);
    assert!(pxpy.abs() < 1e-14, "⟨p_x|1/r|p_y⟩ 必须为 0，实际 {pxpy}");

    // 同中心 p_x 对角：V = −Z π/(6α²)（未归一化原函数）
    let p = alpha + beta;
    let pxpx = primitive_nuclear(alpha, (1, 0, 0), a, beta, (1, 0, 0), b, center, 1.0);
    let expected = -PI / (6.0 * p * p); // α 在上式里是单个原函数的指数；同中心下 p = 2α 代入后一致
    let _ = expected;
    let expected_exact = -PI / (6.0 * (p / 2.0) * (p / 2.0)) / 1.0;
    let _ = expected_exact;
    // 直接用 ∫x²e^{−p r²}/r = π/(6(p/2)²) 的等价形式：π/(6α²)，α = p/2。
    let alpha_single = p / 2.0;
    let analytic = -PI / (6.0 * alpha_single * alpha_single);
    assert!(
        (pxpx - analytic).abs() < 1e-12 * analytic.abs(),
        "⟨p_x|1/r|p_x⟩: {pxpx} vs 解析 {analytic}"
    );
}

/// 核吸引（s-s）：跨中心的解析闭式（教科书 `−Z(2π/p)F₀(p|P−C|²)`）。
#[test]
fn nuclear_s_type_matches_the_textbook_formula_off_center() {
    let alpha = 0.85_f64;
    let beta = 0.55_f64;
    let a = [0.0, 0.0, 0.0];
    let b = [0.5, 0.2, -0.4];
    let nucleus = [1.3, -0.7, 0.9];
    let p = alpha + beta;
    let pc = [
        (alpha * a[0] + beta * b[0]) / p - nucleus[0],
        (alpha * a[1] + beta * b[1]) / p - nucleus[1],
        (alpha * a[2] + beta * b[2]) / p - nucleus[2],
    ];
    let pc2 = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
    let e_ab =
        (-alpha * beta / p * (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>()).exp();
    let f0 = qwb_integrals::boys_functions(0, p * pc2)[0];
    let analytic = -(2.0 * PI / p) * e_ab * f0;

    let actual = primitive_nuclear(alpha, (0, 0, 0), a, beta, (0, 0, 0), b, nucleus, 1.0);
    assert!(
        (actual - analytic).abs() < 1e-13 * analytic.abs(),
        "V(s,s): {actual} vs 教科书 {analytic}"
    );
}
