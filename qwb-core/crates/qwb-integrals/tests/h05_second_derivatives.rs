#![allow(clippy::needless_range_loop)] // 四重下标循环是 144 个分量最直白的写法
//! RS-05 的二阶导数基元：`primitive_component_second_derivatives`。
//!
//! 三根独立的桩：
//!
//! 1. **中心差分**：对 `primitive_component`（已经过 I-03 的独立数值路径验证）做
//!    `O(h²)` 的混合中心差分。这条判据完全独立于二阶导数的实现——两边只共享
//!    「原函数积分值」这一个已经钉住的东西；
//! 2. **平动不变性**：`Σ_s ∂²/∂A_s∂A_t = 0`（四个中心一起平移时积分不变）；
//! 3. **交换对称**：`∂²/∂A_s,i∂A_t,j = ∂²/∂A_t,j∂A_s,i`。
//!
//! 另有一个 `#[ignore]` 的成本比测量（一阶 vs 二阶，同一批壳层四元组），它是
//! RS-05「2e 导数核能不能快 8×」的输入数字，不进主门禁。

use qwb_integrals::hermite2e::{
    primitive_component, primitive_component_derivatives, primitive_component_second_derivatives,
};

type Center = [f64; 3];

const H: f64 = 2.0e-4;

/// 只沿一个轴移动某个中心（`slot`/`axis` + `amount`），其余不动。
fn shifted(
    powers: [[u32; 3]; 4],
    centers: [Center; 4],
    exponents: [f64; 4],
    moves: &[(usize, usize, f64)],
) -> f64 {
    let mut moved = centers;
    for (slot, axis, amount) in moves {
        moved[*slot][*axis] += amount;
    }
    primitive_component(
        powers[0],
        moved[0],
        exponents[0],
        powers[1],
        moved[1],
        exponents[1],
        powers[2],
        moved[2],
        exponents[2],
        powers[3],
        moved[3],
        exponents[3],
    )
}

fn geometries() -> Vec<(&'static str, [Center; 4], [f64; 4])> {
    vec![
        (
            "H2-like",
            [
                [0.0, 0.0, -0.699_2],
                [0.0, 0.0, 0.699_2],
                [0.5, 0.1, 0.3],
                [-0.2, 0.4, 0.7],
            ],
            [1.2, 0.9, 1.1, 0.8],
        ),
        (
            "水分子一角",
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.430_5, 1.107_4],
                [0.0, -1.430_5, 1.107_4],
                [0.6, 0.9, -0.4],
            ],
            [1.5, 0.7, 1.0, 0.55],
        ),
        (
            "紧核-价层",
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.28],
                [0.0, 0.0, -0.28],
                [0.1, 0.2, 0.0],
            ],
            [130.0, 1.5, 0.5, 1.1],
        ),
    ]
}

fn power_cases() -> Vec<([[u32; 3]; 4], &'static str)> {
    vec![
        ([[0, 0, 0]; 4], "ss|ss"),
        ([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]], "px_s|pz_py"),
        ([[2, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]], "dxx_dxy|s_px"),
        ([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], "pxpx|pxpx"),
    ]
}

/// 二阶导数的中心差分参照。
fn finite_difference(
    powers: [[u32; 3]; 4],
    centers: [Center; 4],
    exponents: [f64; 4],
    s: usize,
    i: usize,
    t: usize,
    j: usize,
) -> f64 {
    if s == t {
        if i == j {
            let plus = shifted(powers, centers, exponents, &[(s, i, H)]);
            let minus = shifted(powers, centers, exponents, &[(s, i, -H)]);
            let zero = shifted(powers, centers, exponents, &[]);
            return (plus - 2.0 * zero + minus) / (H * H);
        }
        // 同一中心、两个不同分量：±i ±j 的十字差分。
        let pp = shifted(powers, centers, exponents, &[(s, i, H), (t, j, H)]);
        let pm = shifted(powers, centers, exponents, &[(s, i, H), (t, j, -H)]);
        let mp = shifted(powers, centers, exponents, &[(s, i, -H), (t, j, H)]);
        let mm = shifted(powers, centers, exponents, &[(s, i, -H), (t, j, -H)]);
        return (pp - pm - mp + mm) / (4.0 * H * H);
    }
    let pp = shifted(powers, centers, exponents, &[(s, i, H), (t, j, H)]);
    let pm = shifted(powers, centers, exponents, &[(s, i, H), (t, j, -H)]);
    let mp = shifted(powers, centers, exponents, &[(s, i, -H), (t, j, H)]);
    let mm = shifted(powers, centers, exponents, &[(s, i, -H), (t, j, -H)]);
    (pp - pm - mp + mm) / (4.0 * H * H)
}

#[test]
fn second_derivatives_match_central_differences() {
    let mut worst = 0.0_f64;
    for (name, centers, exponents) in geometries() {
        for (powers, label) in power_cases() {
            let actual = primitive_component_second_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            );
            for s in 0..4 {
                for t in 0..4 {
                    for i in 0..3 {
                        for j in 0..3 {
                            let expected =
                                finite_difference(powers, centers, exponents, s, i, t, j);
                            let scale = expected.abs().max(1.0e-8);
                            let relative = (actual[s][t][i][j] - expected).abs() / scale;
                            // H=2e-4 时截断误差 ~3e-6（相对）、噪声 ~1e-7（相对）。
                            // 只对「不算太小」的元素用相对判据，太小的元素用绝对判据
                            // （否则 1e-12 的元素会把噪声放大成假失配）。
                            let absolute = (actual[s][t][i][j] - expected).abs();
                            if expected.abs() >= 1.0e-4 {
                                assert!(
                                    relative < 1.0e-5,
                                    "{name}/{label} ∂²/∂A_{s},{i}∂A_{t},{j}: production {:.10e} vs finite difference {:.10e}（相对差 {relative:.2e}）",
                                    actual[s][t][i][j],
                                    expected
                                );
                                worst = worst.max(relative);
                            } else {
                                assert!(
                                    absolute < 1.0e-8,
                                    "{name}/{label} ∂²/∂A_{s},{i}∂A_{t},{j}: production {:.10e} vs finite difference {:.10e}（绝对差 {absolute:.2e}）",
                                    actual[s][t][i][j],
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("二阶导数 vs 中心差分的最大相对差：{worst:.2e}");
    assert!(worst < 1.0e-5, "最大相对差 {worst:.2e}");
}

#[test]
fn second_derivatives_are_translation_invariant() {
    for (name, centers, exponents) in geometries() {
        for (powers, label) in power_cases() {
            let value = primitive_component_second_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            );
            for t in 0..4 {
                for j in 0..3 {
                    for i in 0..3 {
                        let sum: f64 = (0..4).map(|s| value[s][t][i][j]).sum();
                        let scale = (0..4)
                            .map(|s| value[s][t][i][j].abs())
                            .fold(0.0_f64, f64::max)
                            .max(1.0e-10);
                        assert!(
                            sum.abs() < 1.0e-9 * scale,
                            "{name}/{label} ∂²/∂A_{t},{j} 的平动和 = {sum:.3e}（尺度 {scale:.3e}）"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn second_derivatives_are_symmetric_in_the_derivative_operators() {
    for (name, centers, exponents) in geometries() {
        for (powers, label) in power_cases() {
            let value = primitive_component_second_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            );
            for s in 0..4 {
                for t in 0..4 {
                    for i in 0..3 {
                        for j in 0..3 {
                            let forward = value[s][t][i][j];
                            let backward = value[t][s][j][i];
                            let scale = forward.abs().max(backward.abs()).max(1.0e-10);
                            assert!(
                                (forward - backward).abs() < 1.0e-10 * scale,
                                "{name}/{label}: ∂²/∂A_{s},{i}∂A_{t},{j} = {forward:.6e} ≠ ∂²/∂A_{t},{j}∂A_{s},{i} = {backward:.6e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn first_derivative_still_satisfies_translation_invariance() {
    // 顺带把一阶导数也钉一次：二阶导数的展开如果写错，最可能的形态是「降幂系数用了
    // 外层幂次」，那种错误在一阶导数的平动和上也会露出来。
    for (name, centers, exponents) in geometries() {
        for (powers, label) in power_cases() {
            let value = primitive_component_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            );
            for i in 0..3 {
                let sum: f64 = (0..4).map(|s| value[s][i]).sum();
                let scale = (0..4)
                    .map(|s| value[s][i].abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0e-10);
                assert!(
                    sum.abs() < 1.0e-9 * scale,
                    "{name}/{label}: 一阶导数平动和 {sum:.3e}（尺度 {scale:.3e}）"
                );
            }
        }
    }
}

/// 成本比：同一批典型壳层四元组上，二阶导数基元 ÷ 一阶导数基元。
///
/// 这是 RS-05 的输入数字（「2e 导数核要快 k ≥ 7.7 倍」的另一半：二阶导数本身比
/// 一阶贵多少）。默认 `#[ignore]`，用 `cargo nextest run --run-ignored only` 或
/// `cargo test -- --ignored --nocapture` 跑。
#[test]
#[ignore = "成本测量，不进主门禁"]
fn cost_ratio_second_over_first_derivative() {
    use std::time::Instant;
    // 水/def2-SVP 的真实角动量组合：O 有 s,s,s,p,p,d；H 有 s,s,p。
    let quartets: [[[u32; 3]; 4]; 4] = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]],
        [[2, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]],
        [[2, 0, 0], [0, 0, 2], [1, 1, 0], [0, 0, 0]],
    ];
    let centers: [Center; 4] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.430_5, 1.107_4],
        [0.0, -1.430_5, 1.107_4],
        [0.6, 0.9, -0.4],
    ];
    let exponents = [3.5, 0.8, 1.4, 0.55];
    let repeats = 2000_u32;

    let mut first_time = 0.0_f64;
    let mut second_time = 0.0_f64;
    let mut sink = 0.0_f64;
    for powers in &quartets {
        let start = Instant::now();
        for _ in 0..repeats {
            sink += primitive_component_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            )[0][0];
        }
        first_time += start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..repeats {
            sink += primitive_component_second_derivatives(
                powers[0],
                centers[0],
                exponents[0],
                powers[1],
                centers[1],
                exponents[1],
                powers[2],
                centers[2],
                exponents[2],
                powers[3],
                centers[3],
                exponents[3],
            )[0][0][0][0];
        }
        second_time += start.elapsed().as_secs_f64();
    }
    // 一次二阶导数调用覆盖 144 个分量，一次一阶覆盖 12 个；先报**每分量**成本，
    // 再报**每次调用**成本（Hessian 里两种口径都要用：前者决定收缩成本，后者
    // 决定是不是每个四元组都值得整块算）。
    let per_component = (second_time / 144.0) / (first_time / 12.0);
    let per_call = second_time / first_time;
    println!(
        "二阶/一阶：每次调用 {per_call:.2}×、每个分量 {per_component:.2}×（{repeats} 次 × {} 个四元组，sink {sink:.3e}）",
        quartets.len()
    );
    assert!(per_call.is_finite() && per_call > 0.0);
}
