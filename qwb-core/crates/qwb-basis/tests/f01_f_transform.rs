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
    // **`r²·x` 在「归一化直角基」下的系数不是 1。**
    //
    // 变换的基底是按分量归一化的直角高斯 `φ_i = N_i x^a y^b z^c e^{−αr²}`，
    // 其中 `N_i ∝ 1/√∏_axis (2l_axis−1)!!`。所以单项式
    // `x^a y^b z^c ∝ √∏(2l−1)!! · φ_i`，而
    // `r²x = x³ + xy² + xz²` 对应 `√15·e_{x³} + √3·e_{xy²} + √3·e_{xz²}`。
    //
    // 这条测试原来用 `1,1,1` 当系数，于是它和同样写错系数的实现"互相印证"，
    // 一起漏掉了「f 混进 r²·p 成分」这个错（详见 `docs/choices.md` 五十五）。
    let mut radial_p = [0.0_f64; 10];
    for (index, powers) in cartesian_components(3).iter().enumerate() {
        let odd = |l: u32| -> f64 {
            // (2l−1)!!：0→1, 1→1, 2→3, 3→15
            let mut value = 1.0;
            let mut current = 2 * l;
            while current > 1 {
                value *= f64::from(current) - 1.0;
                current -= 2;
            }
            value
        };
        if *powers == (3, 0, 0) || *powers == (1, 2, 0) || *powers == (1, 0, 2) {
            radial_p[index] = (odd(powers.0) * odd(powers.1) * odd(powers.2)).sqrt();
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

/// f 子空间必须**就是** l=3，而不只是「正交归一 + 与某三个向量正交」。
///
/// 上一版的 `Tᵀ S T = I` 与「与 r²p 正交」两条都过，但 `r²p` 向量写的是未归一化
/// 多项式的系数，于是它和同样写错的实现互相印证，一起漏掉了「f 混进 r²·p 成分」。
/// 这里换一条对「投影到哪个子空间」极敏感的判据：
///
/// ```text
/// T Tᵀ  =  Tₐ Tₐᵀ        （Tₐ 是独立 Gram–Schmidt 构造的 l=3 正交归一基）
/// ```
///
/// 两种基底之间只差一个 7×7 正交阵，所以 `T Tᵀ` 必须逐元素相等。
#[test]
fn f_subspace_is_the_l3_space() {
    let components = cartesian_components(3);
    let odd = |l: u32| -> f64 {
        let mut value = 1.0;
        let mut current = 2 * l;
        while current > 1 {
            value *= f64::from(current) - 1.0;
            current -= 2;
        }
        value
    };
    // 三个 r²p 向量在归一化直角基下的系数
    let mut r = [[0.0_f64; 10]; 3];
    for (index, powers) in components.iter().enumerate() {
        let scale = (odd(powers.0) * odd(powers.1) * odd(powers.2)).sqrt();
        let is_x = *powers == (3, 0, 0) || *powers == (1, 2, 0) || *powers == (1, 0, 2);
        let is_y = *powers == (2, 1, 0) || *powers == (0, 3, 0) || *powers == (0, 1, 2);
        let is_z = *powers == (2, 0, 1) || *powers == (0, 2, 1) || *powers == (0, 0, 3);
        if is_x {
            r[0][index] = scale;
        }
        if is_y {
            r[1][index] = scale;
        }
        if is_z {
            r[2][index] = scale;
        }
    }
    // 先把 r²p 归一（否则投影会多减）
    for vector in &mut r {
        let norm = overlap(vector, vector).sqrt();
        assert!(norm > 1.0e-8, "r²p 向量退化");
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
    // 独立构造：十个单位向量逐个 Gram–Schmidt（先打掉 r²p，再打掉已选列）
    let mut analytic: Vec<[f64; 10]> = Vec::new();
    for unit in 0..10 {
        let mut current = [0.0_f64; 10];
        current[unit] = 1.0;
        for basis in r.iter().chain(analytic.iter()) {
            let projection = overlap(&current, basis);
            for k in 0..10 {
                current[k] -= projection * basis[k];
            }
        }
        let norm = overlap(&current, &current).sqrt();
        if norm > 1.0e-8 {
            for value in current.iter_mut() {
                *value /= norm;
            }
            analytic.push(current);
        }
        if analytic.len() == 7 {
            break;
        }
    }
    assert_eq!(analytic.len(), 7, "l=3 补空间应为 7 维");

    let columns = columns_of_f();
    let mut worst = 0.0_f64;
    for i in 0..10 {
        for j in 0..10 {
            let from_t: f64 = (0..7).map(|c| columns[c][i] * columns[c][j]).sum();
            let from_analytic: f64 = (0..7).map(|c| analytic[c][i] * analytic[c][j]).sum();
            worst = worst.max((from_t - from_analytic).abs());
        }
    }
    println!("T Tᵀ 与独立构造的 T Tᵀ 最大差 {worst:.2e}");
    assert!(
        worst < 1.0e-10,
        "f 张成的子空间不是 l=3：T Tᵀ 差 {worst:.2e}"
    );
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
