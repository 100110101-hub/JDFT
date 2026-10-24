//! f（`l = 3`）球谐变换的验收：在**重叠度规**下必须正交归一。
//!
//! 与 d 的那条测试同一套判据（`d_transform_is_orthonormal_in_the_overlap_metric`），
//! 只是换了度量与维度：
//!
//! ```text
//! Tᵀ S T = I₇        S 是 10 个归一化直角 f 原函数的自重叠矩阵
//! ```
//!
//! 另外断言一条独立性质：七个 f 与 `r²·x` 正交（与 p 子空间分开，防止「形状对了但内容错」）。
//!
//! 背景：def2-TZVP 的 O/C/N 带 f，而 `spherical_transform` 原来只到 d，
//! 于是 H2O/def2-TZVP 直接报 Unsupported。这条测试就是那次缺口的验收。
#![allow(clippy::needless_range_loop)] // 矩阵收缩：两个下标各有含义

use qwb_basis::cartesian_components;
use qwb_basis::spherical::{f_overlap_metric, spherical_transform};
use qwb_registry::basis::Angular;

fn columns_of_f() -> [[f64; 10]; 7] {
    let transform = spherical_transform(3, Angular::Spherical).expect("f 变换应已实现");
    let mut columns = [[0.0_f64; 10]; 7];
    for (column, values) in columns.iter_mut().enumerate() {
        for (row, value) in values.iter_mut().enumerate() {
            *value = transform.get(row, column);
        }
    }
    columns
}

/// `aᵀ S b`
fn overlap(a: &[f64; 10], b: &[f64; 10]) -> f64 {
    let metric = f_overlap_metric();
    let mut total = 0.0;
    for i in 0..10 {
        for j in 0..10 {
            total += a[i] * metric.get(i, j) * b[j];
        }
    }
    total
}

#[test]
fn f_transform_is_orthonormal_in_the_overlap_metric() {
    let transform = spherical_transform(3, Angular::Spherical).expect("f 变换应已实现");
    assert_eq!(transform.n_cartesian, 10, "f 有 10 个直角分量");
    assert_eq!(transform.n_spherical, 7, "f 有 7 个球谐分量");

    let columns = columns_of_f();
    let mut worst = 0.0_f64;
    for a in 0..7 {
        for b in 0..7 {
            let value = overlap(&columns[a], &columns[b]);
            let expected = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((value - expected).abs());
        }
    }
    println!("Tᵀ S T 与单位阵的最大偏差 {worst:.2e}");
    assert!(worst < 1.0e-12, "f 变换在重叠度规下不正交归一：{worst:.2e}");
}

#[test]
fn f_is_orthogonal_to_the_radial_p_direction() {
    // x(x²+y²+z²) = x³ + xy² + xz²
    let mut radial_p = [0.0_f64; 10];
    for (index, powers) in cartesian_components(3).iter().enumerate() {
        if *powers == (3, 0, 0) || *powers == (1, 2, 0) || *powers == (1, 0, 2) {
            radial_p[index] = 1.0;
        }
    }
    let columns = columns_of_f();
    let mut worst = 0.0_f64;
    for column in &columns {
        worst = worst.max(overlap(&radial_p, column).abs());
    }
    println!("七个 f 与 r²·x 的最大投影 {worst:.2e}");
    assert!(worst < 1.0e-12, "f 与 r²·p 不正交：{worst:.2e}");
}

#[test]
fn f_has_the_ten_cubic_components() {
    assert_eq!(cartesian_components(3).len(), 10);
    let mut produced: Vec<[u32; 3]> = cartesian_components(3)
        .into_iter()
        .map(|powers| [powers.0, powers.1, powers.2])
        .collect();
    produced.sort_unstable();
    let mut expected: Vec<[u32; 3]> = Vec::new();
    for a in 0..=3u32 {
        for b in 0..=(3 - a) {
            expected.push([a, b, 3 - a - b]);
        }
    }
    expected.sort_unstable();
    assert_eq!(produced, expected, "三次分量列表不完整或有重复");
}
