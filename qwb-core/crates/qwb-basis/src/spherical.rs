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
        Angular::Spherical if l == 3 => Some(f_transform()),
        Angular::Spherical if l == 4 => Some(g_transform()),
        // l >= 4：系数不猜，显式不支持。
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
/// 双阶乘 `n!!`（`n <= 0` 时按定义取 1，即 `(0)!! = (−1)!! = 1`）。
fn double_factorial(n: i64) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    let mut product = 1.0;
    let mut current = n;
    while current > 1 {
        product *= current as f64;
        current -= 2;
    }
    product
}

/// `(2l−1)!!`：归一化直角高斯的逐分量常数（`l = 0,1,2,3 → 1,1,3,15`）。
fn odd_double_factorial(l: u32) -> f64 {
    double_factorial(2 * i64::from(l) - 1)
}

/// `l = 3` 的 10 个直角分量在**重叠度规**下的自重叠比 `Π_axis (a+a'-1)!!/sqrt((2a-1)!!(2a'-1)!!)`。
///
/// 只有同一轴上两个幂次的奇偶性一致时才非零；否则该轴积分为 0。
fn f_overlap_metric_value(left: (u32, u32, u32), right: (u32, u32, u32)) -> f64 {
    let left = [left.0, left.1, left.2];
    let right = [right.0, right.1, right.2];
    let mut product = 1.0;
    for axis in 0..3 {
        let a = i64::from(left[axis]);
        let b = i64::from(right[axis]);
        if (a + b) % 2 == 1 {
            return 0.0;
        }
        let numerator = double_factorial(a + b - 1);
        let denominator = (double_factorial(2 * a - 1) * double_factorial(2 * b - 1)).sqrt();
        product *= numerator / denominator;
    }
    product
}

/// `l = 4` 的 15×9 变换（列主序，行序见 [`crate::cartesian_components`]）。
///
/// 与 [`f_transform`] 同一套构造（不手推归一化常数）：四次单项式空间分解为
/// `g(9) ⊕ r²·d(6)`，所以取九个实 g 谐波的多项式形式，在重叠度规下对六个
/// `r²·(二次单项式)` 方向做 Gram–Schmidt（打掉 d 成分），再正交归一化。
/// `Tᵀ S T = I₉` 与「与 r²·d 正交」都是构造保证的。
fn g_transform() -> Transform {
    use crate::cartesian_components;
    let components = cartesian_components(4);
    debug_assert_eq!(components.len(), 15, "l=4 应有 15 个直角分量");

    let mut component_scale = [0.0_f64; 15];
    for (i, powers) in components.iter().enumerate() {
        component_scale[i] = (double_factorial(i64::from(powers.0) * 2 - 1)
            * double_factorial(i64::from(powers.1) * 2 - 1)
            * double_factorial(i64::from(powers.2) * 2 - 1))
        .sqrt();
    }

    let mut metric = [[0.0_f64; 15]; 15];
    for i in 0..15 {
        for j in 0..15 {
            metric[i][j] = f_overlap_metric_value(components[i], components[j]);
        }
    }

    #[rustfmt::skip]
    let mut candidates: [[f64; 15]; 9] = [
        [3.000000000000, 0.000000000000, 0.000000000000, 6.000000000000, 0.000000000000, -24.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 3.000000000000, 0.000000000000, -24.000000000000, 0.000000000000, 8.000000000000],
        [0.000000000000, 0.000000000000, -3.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -3.000000000000, 0.000000000000, 4.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -3.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -3.000000000000, 0.000000000000, 4.000000000000, 0.000000000000],
        [-1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 6.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, -6.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, -1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -1.000000000000, 0.000000000000, 6.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -3.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 3.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [1.000000000000, 0.000000000000, 0.000000000000, -6.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, -1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
    ];

    #[rustfmt::skip]
    let mut radial: [[f64; 15]; 6] = [
        [1.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 1.000000000000],
    ];

    for vector in &mut candidates {
        for (index, value) in vector.iter_mut().enumerate() {
            *value *= component_scale[index];
        }
    }
    for vector in &mut radial {
        for (index, value) in vector.iter_mut().enumerate() {
            *value *= component_scale[index];
        }
    }

    let inner = |a: &[f64; 15], b: &[f64; 15]| -> f64 {
        let mut total = 0.0;
        for i in 0..15 {
            for j in 0..15 {
                total += a[i] * metric[i][j] * b[j];
            }
        }
        total
    };

    let mut radial_basis: Vec<[f64; 15]> = Vec::new();
    for vector in &radial {
        let mut current = *vector;
        for b in &radial_basis {
            let projection = inner(&current, b);
            for k in 0..15 {
                current[k] -= projection * b[k];
            }
        }
        let norm = inner(&current, &current).sqrt();
        if norm > 1.0e-8 {
            for value in &mut current {
                *value /= norm;
            }
            radial_basis.push(current);
        }
    }
    debug_assert_eq!(radial_basis.len(), 6, "r²·d 应是 6 维");

    let mut basis: Vec<[f64; 15]> = Vec::new();
    for vector in &candidates {
        let mut current = *vector;
        for b in radial_basis.iter().chain(basis.iter()) {
            let projection = inner(&current, b);
            for k in 0..15 {
                current[k] -= projection * b[k];
            }
        }
        let norm = inner(&current, &current).sqrt();
        assert!(norm > 1.0e-10, "g 候选向量退化（构造错误）");
        for value in &mut current {
            *value /= norm;
        }
        basis.push(current);
    }

    let mut data = vec![0.0; 9 * 15];
    for (col, vector) in basis.iter().enumerate() {
        for (row, value) in vector.iter().enumerate() {
            data[col * 15 + row] = *value;
        }
    }
    Transform {
        n_cartesian: 15,
        n_spherical: 9,
        data,
    }
}

/// `l = 3` 的 10×7 变换（列主序，行序见 [`crate::cartesian_components`]）。
///
/// # 怎么构造的（不用手推归一化常数）
///
/// 三次单项式空间（10 维）分解为 `f(7) ⊕ r²·p(3)`。所以：
///
/// 1. 取七个实立方谐波的**多项式**形式（`z(5z²−3r²)`、`x(5z²−r²)`、`z(x²−y²)`、`xyz`、
///    `x(x²−3y²)` 及其 ±y 对应）；
/// 2. 在**重叠度规** `S` 下把它们对 `r²·x, r²·y, r²·z` 做 Gram–Schmidt 投影（把 p 成分打掉）；
/// 3. 再在 `S` 下正交归一化。
///
/// 于是 `Tᵀ S T = I₇` 与「与 `r²·p` 正交」都是构造保证的，而**张成的子空间**正是
/// 真实的 f 子空间——闭壳层 SCF 用同一壳层内的任意正交归一基给出的能量与密度完全相同，
/// 所以这一步不需要（也不应该）去猜每一列的解析归一化常数。
fn f_transform() -> Transform {
    use crate::cartesian_components;
    let components = cartesian_components(3);
    debug_assert_eq!(components.len(), 10);

    // **多项式系数 → 归一化直角基下的系数。**
    //
    // 下面 `candidates` / `radial` 里写的是多项式 `Σ c_i x^a y^b z^c` 的系数，
    // 而这里的基函数是**按分量归一化**的直角高斯
    // `φ_i = N_i x^a y^b z^c e^{−αr²}`，`N_i = C·(4α)^{L/2}/√∏_axis (2l_axis−1)!!`
    // （`C` 与 `L` 对同一壳层的所有分量相同，可整体约掉）。所以单项式
    // `x^a y^b z^c = φ_i / N_i ∝ √∏_axis (2l_axis−1)!! · φ_i`：
    // 多项式系数必须**逐分量乘上 `√∏(2l−1)!!`**。
    //
    // 上一版漏了这一步，导致「去掉 r²·p 成分」的投影打掉的是另一个三维子空间，
    // 得到的七个函数虽然仍满足 `Tᵀ S T = I₇`（Gram–Schmidt 保证），却混进了
    // `r²·p` 成分。后果是 f 壳层只被虚轨道用到，所以总能量只偏 2.7e-6 Ha，
    // 但 H₂O/def2-TZVP 的 HF 与 PySCF 差 2.7e-6、轨道能偏 1e-4 —— 见
    // `docs/choices.md` 五十五与 `f_subspace_is_the_l3_space` 测试。
    let component_scale: [f64; 10] = {
        let mut out = [0.0_f64; 10];
        for (index, powers) in components.iter().enumerate() {
            out[index] = (odd_double_factorial(powers.0)
                * odd_double_factorial(powers.1)
                * odd_double_factorial(powers.2))
            .sqrt();
        }
        out
    };

    // 立方谐波（未归一化，允许任意整体缩放）
    let candidates: [[f64; 10]; 7] = [
        // z(5z² − r²) = 2z³ − 3x²z − 3y²z
        [0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 2.0],
        // x(5z² − r²) = 4xz² − x³ − xy²
        [-1.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        // y(5z² − r²) = 4yz² − y³ − x²y
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0],
        // z(x² − y²) = x²z − y²z
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        // xyz
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // x(x² − 3y²) = x³ − 3xy²
        [1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // y(3x² − y²) = 3x²y − y³
        [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    ];

    // 重叠度规
    let mut metric = [[0.0_f64; 10]; 10];
    for i in 0..10 {
        for j in 0..10 {
            metric[i][j] = f_overlap_metric_value(components[i], components[j]);
        }
    }
    // r²·x、r²·y、r²·z 在三项式基下的系数
    let radial: [[f64; 10]; 3] = [
        // x(x²+y²+z²) = x³ + xy² + xz²
        {
            let mut v = [0.0; 10];
            v[0] = 1.0;
            v[3] = 1.0;
            v[5] = 1.0;
            v
        },
        // y(x²+y²+z²)
        {
            let mut v = [0.0; 10];
            v[1] = 1.0;
            v[6] = 1.0;
            v[8] = 1.0;
            v
        },
        // z(x²+y²+z²)
        {
            let mut v = [0.0; 10];
            v[2] = 1.0;
            v[7] = 1.0;
            v[9] = 1.0;
            v
        },
    ];
    // 把多项式系数换成归一化直角基下的系数（见上面 `component_scale` 的说明）。
    let mut candidates = candidates;
    for vector in &mut candidates {
        for (index, value) in vector.iter_mut().enumerate() {
            *value *= component_scale[index];
        }
    }
    let mut radial = radial;
    for vector in &mut radial {
        for (index, value) in vector.iter_mut().enumerate() {
            *value *= component_scale[index];
        }
    }
    // 在 S 下的内积
    let inner = |a: &[f64; 10], b: &[f64; 10]| -> f64 {
        let mut total = 0.0;
        for i in 0..10 {
            for j in 0..10 {
                total += a[i] * metric[i][j] * b[j];
            }
        }
        total
    };
    // 先把径向向量在 S 下正交归一（作为投影的基底）
    let mut radial_basis: Vec<[f64; 10]> = Vec::new();
    for vector in &radial {
        let mut current = *vector;
        for basis in &radial_basis {
            let projection = inner(&current, basis);
            for k in 0..10 {
                current[k] -= projection * basis[k];
            }
        }
        let norm = inner(&current, &current).sqrt();
        if norm > 1.0e-12 {
            for value in current.iter_mut() {
                *value /= norm;
            }
            radial_basis.push(current);
        }
    }

    // 候选列：去掉径向成分，再逐列正交归一
    let mut columns: Vec<[f64; 10]> = Vec::new();
    for candidate in &candidates {
        let mut current = *candidate;
        for basis in &radial_basis {
            let projection = inner(&current, basis);
            for k in 0..10 {
                current[k] -= projection * basis[k];
            }
        }
        for basis in &columns {
            let projection = inner(&current, basis);
            for k in 0..10 {
                current[k] -= projection * basis[k];
            }
        }
        let norm = inner(&current, &current).sqrt();
        assert!(
            norm > 1.0e-8,
            "f 变换构造失败：某一列在去掉径向与已选成分后范数过小"
        );
        for value in current.iter_mut() {
            *value /= norm;
        }
        columns.push(current);
    }
    assert_eq!(columns.len(), 7, "f 必须构造出 7 列");

    let mut data = vec![0.0; 10 * 7];
    for (column, values) in columns.iter().enumerate() {
        for row in 0..10 {
            data[column * 10 + row] = values[row];
        }
    }
    Transform {
        n_cartesian: 10,
        n_spherical: 7,
        data,
    }
}

/// `l = 3` 的重叠度规（同一中心、归一化原函数，10×10）。
#[must_use]
pub fn f_overlap_metric() -> Transform {
    use crate::cartesian_components;
    let components = cartesian_components(3);
    let mut data = vec![0.0; 100];
    for i in 0..10 {
        for j in 0..10 {
            data[j * 10 + i] = f_overlap_metric_value(components[i], components[j]);
        }
    }
    Transform {
        n_cartesian: 10,
        n_spherical: 10,
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

    /// `l = 3`（f）现在**是**支持的（见 `f_transform`），`l >= 4` 仍拒绝猜系数。
    /// 这条测试以前断言 `l = 3` 返回 None——那是 def2-TZVP 之前的限制，已解除。
    #[test]
    fn unknown_higher_angular_momentum_is_rejected_not_guessed() {
        assert!(spherical_transform(3, Angular::Spherical).is_some());
        assert!(spherical_transform(5, Angular::Spherical).is_none());
        assert!(spherical_transform(5, Angular::Spherical).is_none());
        assert!(spherical_transform(3, Angular::Cartesian).is_some());
        assert_eq!(crate::cartesian_components(3).len(), 10);
        assert_eq!(
            spherical_transform(3, Angular::Spherical)
                .unwrap()
                .n_spherical,
            7
        );
    }
    #[test]
    fn g_transform_is_orthonormal_and_free_of_d() {
        use crate::cartesian_components;
        let components = cartesian_components(4);
        assert_eq!(components.len(), 15);
        let metric = |i: usize, j: usize| {
            crate::spherical::f_overlap_metric_value(components[i], components[j])
        };
        let t = crate::spherical::g_transform();
        assert_eq!(t.n_cartesian, 15);
        assert_eq!(t.n_spherical, 9);
        // Tᵀ S T = I₉
        for a in 0..9 {
            for b in 0..9 {
                let mut value = 0.0_f64;
                for i in 0..15 {
                    for j in 0..15 {
                        value += t.data[a * 15 + i] * metric(i, j) * t.data[b * 15 + j];
                    }
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((value - expected).abs() < 1.0e-10, "({a},{b}) = {value}");
            }
        }
        // 与 r²·(六个二次单项式) 正交 —— 这是「g 里没有 d 成分」的判据。
        // r² = x²+y²+z²，所以在单项式基下：r²·q 的系数是把 q 沿 x/y/z 各升两阶。
        for (quadratic, name) in [
            ((2_u32, 0, 0), "x²"),
            ((1, 1, 0), "xy"),
            ((1, 0, 1), "xz"),
            ((0, 2, 0), "y²"),
            ((0, 1, 1), "yz"),
            ((0, 0, 2), "z²"),
        ] {
            let mut vector = [0.0_f64; 15];
            for shift in [(2_u32, 0_u32, 0_u32), (0, 2, 0), (0, 0, 2)] {
                let target = (
                    quadratic.0 + shift.0,
                    quadratic.1 + shift.1,
                    quadratic.2 + shift.2,
                );
                let position = components
                    .iter()
                    .position(|c| *c == target)
                    .unwrap_or_else(|| panic!("找不到单项式 {target:?}"));
                vector[position] += 1.0;
            }
            // polynomial coefficients -> normalised Cartesian components (same conversion
            // g_transform does internally).
            let scale: Vec<f64> = components
                .iter()
                .map(|p| {
                    (crate::spherical::double_factorial(p.0 as i64 * 2 - 1)
                        * crate::spherical::double_factorial(p.1 as i64 * 2 - 1)
                        * crate::spherical::double_factorial(p.2 as i64 * 2 - 1))
                    .sqrt()
                })
                .collect();
            for (j, value) in vector.iter_mut().enumerate() {
                *value *= scale[j];
            }
            for a in 0..9 {
                let mut overlap = 0.0_f64;
                for i in 0..15 {
                    let t_i = t.data[a * 15 + i];
                    for (j, value) in vector.iter().enumerate() {
                        overlap += t_i * metric(i, j) * value;
                    }
                }
                assert!(
                    overlap.abs() < 1.0e-9,
                    "第 {a} 列与 r²·{name} 的重叠 = {overlap}"
                );
            }
        }
    }
}
