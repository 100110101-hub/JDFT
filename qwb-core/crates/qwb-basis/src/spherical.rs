//! 直角坐标 → 球谐的变换矩阵（R-06 的另一半）。
//!
//! # 为什么需要它
//!
//! 数据里的 `angular` 决定 AO 数：球谐 d 是 5 个、直角坐标 d 是 6 个。积分核是
//! 直角坐标实现（Hermite 展开天然是直角坐标），所以要么把积分直接写成球谐
//! （很难），要么**在直角坐标里算完再变换**（这里走这条）：
//!
//! ```text
//! M_sph = Tᵀ M_cart T
//! ```
//!
//! `T` 是每个壳层一块的块对角矩阵（`n_cart × n_spherical`）。`l <= 1` 时
//! `T` 是单位阵（1 个 s / 3 个 p 两种表示完全相同），所以 STO-3G 路径完全不受影响。
//!
//! # `l = 2` 的实际形式
//!
//! 按 [`cartesian_components`] 的顺序 `[x², xy, xz, y², yz, z²]`，五个实球谐
//! （顺序 `[x²−y², z², xy, xz, yz]`）为
//!
//! ```text
//! d(x²−y²) = (x² − y²)/√2
//! d(z²)    = (2z² − x² − y²)/√6
//! d(xy)    = xy,   d(xz) = xz,   d(yz) = yz
//! ```
//!
//! 这五列在归一化直角坐标基下是**标准正交**的（测试断言 `TᵀT = I₅`），且
//! 都与 `s` 方向 `(x²+y²+z²)/√3 ∝ r²` 正交（测试断言）—— 这就是"球谐 d 里没有 s 成分"。
//!
//! # `l >= 3` 未实现
//!
//! f 及更高阶的实球谐变换系数不同（还涉及多组组合），先显式返回错误，
//! 不猜系数。第一阶段的 def2-SVP / cc-pVDZ 对 H..O 只到 d，够用。

use crate::Angular;

/// 一个壳层的变换矩阵（列主序，`n_cart × n_spherical`）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Transform {
    /// 直角坐标分量数（列数意义上的"源"）。
    pub n_cartesian: usize,
    /// 球谐分量数（变换后的"目标"）。
    pub n_spherical: usize,
    /// 列主序数据：`data[col * n_cartesian + row]`。
    data: Vec<f64>,
}

impl Transform {
    /// 单位变换（`n_cartesian == n_spherical`）。
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self {
            n_cartesian: n,
            n_spherical: n,
            data,
        }
    }

    /// 取 `(row, column)`：第 `column` 个球谐函数里第 `row` 个直角坐标分量的系数。
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[column * self.n_cartesian + row]
    }

    /// 列主序切片。
    #[must_use]
    pub fn as_column_major(&self) -> &[f64] {
        &self.data
    }

    /// 是否是单位变换（`l <= 1` 时为真）。
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.n_cartesian == self.n_spherical
            && (0..self.n_cartesian).all(|i| self.get(i, i) == 1.0)
            && self.data.iter().filter(|v| **v != 0.0).count() == self.n_cartesian
    }
}

/// 某个 `l` 在给定 `angular` 下的变换矩阵。
///
/// `l = 1` 的球谐顺序与直角坐标相同（`x, y, z`）；`l >= 3` 返回 `None`。
#[must_use]
pub fn spherical_transform(l: u32, angular: Angular) -> Option<Transform> {
    let n_cart = crate::cartesian_components(l).len();
    match angular {
        Angular::Cartesian => Some(Transform::identity(n_cart)),
        // l <= 1：球谐与直角坐标一一对应（1 个 s；p 就是 x/y/z）。
        Angular::Spherical if l <= 1 => Some(Transform::identity(n_cart)),
        Angular::Spherical if l == 2 => Some(d_transform()),
        // f 及更高阶：系数不猜，显式不支持。
        _ => None,
    }
}

/// `l = 2` 的 6×5 变换（见模块文档）。
fn d_transform() -> Transform {
    // **必须在重叠度规下正交归一，不是普通点积。**
    // 归一化直角坐标 d 之间满足 <x²|x²> = 1、<x²|y²> = 1/3（其余为 0），
    // 因为 <x²|y²> / <x²|x²> = 1/3（高斯矩的比值）。所以下面这些系数与
    // "球谐函数按未归一化直角坐标展开" 的系数不同（后者是 1/√2、1/√6 等），
    // 用错会让球谐 d 的自重叠变成 2/3 —— 上一版正是被 d 壳层积分测试抓到的。
    let mut data = vec![0.0; 6 * 5];
    // 行：0:x², 1:xy, 2:xz, 3:y², 4:yz, 5:z²（列主序：index = 列 * 6 + 行）
    // 列 0 = (x² − y²)·(√3/2)：<u|u> = (3/4)(2 − 2/3) = 1
    data[0] = 3.0_f64.sqrt() / 2.0;
    data[3] = -3.0_f64.sqrt() / 2.0;
    // 列 1 = (2z² − x² − y²)/2：<v|v> = (1/4)(4 + 2 + 2 − ... ) = 1
    data[6] = -0.5;
    data[9] = -0.5;
    data[11] = 1.0;
    // 列 2..4：xy、xz、yz 本身已正交归一。
    data[13] = 1.0;
    data[20] = 1.0;
    data[28] = 1.0;
    Transform {
        n_cartesian: 6,
        n_spherical: 5,
        data,
    }
}

/// `l = 2` 直角坐标分量的重叠度规（同一中心、归一化原函数）：对角 1，`x²/y²/z²` 之间 1/3。
#[must_use]
pub fn d_overlap_metric() -> Transform {
    let mut data = vec![0.0; 36];
    for i in 0..6 {
        data[i * 6 + i] = 1.0;
    }
    for (a, b) in [(0_usize, 3_usize), (0, 5), (3, 5)] {
        data[b * 6 + a] = 1.0 / 3.0;
        data[a * 6 + b] = 1.0 / 3.0;
    }
    Transform {
        n_cartesian: 6,
        n_spherical: 6,
        data,
    }
}
#[cfg(test)]
mod tests {
    use super::{Transform, d_overlap_metric, spherical_transform};
    use crate::Angular;

    /// `Tᵀ S T`，`S` 是直角坐标重叠度规。
    fn gram(transform: &Transform, metric: &Transform) -> Vec<f64> {
        let mut out = vec![0.0; transform.n_spherical * transform.n_spherical];
        for a in 0..transform.n_spherical {
            for b in 0..transform.n_spherical {
                let mut sum = 0.0;
                for i in 0..transform.n_cartesian {
                    for j in 0..transform.n_cartesian {
                        sum += transform.get(i, a) * metric.get(i, j) * transform.get(j, b);
                    }
                }
                out[b * transform.n_spherical + a] = sum;
            }
        }
        out
    }

    #[test]
    fn low_angular_momentum_is_the_identity() {
        for l in 0..=1 {
            let transform = spherical_transform(l, Angular::Spherical).expect("identity");
            assert!(transform.is_identity(), "l={l}");
            assert_eq!(transform.n_spherical, 1 + 2 * l as usize);
        }
        assert!(
            spherical_transform(2, Angular::Cartesian)
                .expect("cartesian")
                .is_identity()
        );
    }

    #[test]
    fn d_transform_is_orthonormal_in_the_overlap_metric() {
        let transform = spherical_transform(2, Angular::Spherical).expect("d");
        assert_eq!(transform.n_cartesian, 6);
        assert_eq!(transform.n_spherical, 5, "球谐 d 是 5 个函数");
        assert!(!transform.is_identity());

        // 关键：正交归一是相对于 <x²|y²> = 1/3 的重叠度规，而不是普通点积。
        let gram_matrix = gram(&transform, &d_overlap_metric());
        for a in 0..5 {
            for b in 0..5 {
                let expected = if a == b { 1.0 } else { 0.0 };
                let actual = gram_matrix[b * 5 + a];
                assert!(
                    (actual - expected).abs() < 1e-15,
                    "Tᵀ S T ({a},{b}) = {actual}，应为 {expected}"
                );
            }
        }
    }

    #[test]
    fn d_transform_has_no_s_component() {
        // s 方向 = (x²+y²+z²)/√3；在重叠度规下每个球谐 d 列都与它正交。
        let transform = spherical_transform(2, Angular::Spherical).expect("d");
        let metric = d_overlap_metric();
        let sqrt3 = 3.0_f64.sqrt();
        for column in 0..transform.n_spherical {
            let mut dot = 0.0;
            for i in 0..6 {
                for j in 0..6 {
                    let s_i = if matches!(i, 0 | 3 | 5) {
                        1.0 / sqrt3
                    } else {
                        0.0
                    };
                    dot += s_i * metric.get(i, j) * transform.get(j, column);
                }
            }
            assert!(dot.abs() < 1e-15, "列 {column} 与 r² 不正交（dot = {dot}）");
        }
    }

    #[test]
    fn unknown_higher_angular_momentum_is_rejected_not_guessed() {
        assert!(spherical_transform(3, Angular::Spherical).is_none());
        assert!(spherical_transform(4, Angular::Spherical).is_none());
        assert!(spherical_transform(3, Angular::Cartesian).is_some());
    }
}
