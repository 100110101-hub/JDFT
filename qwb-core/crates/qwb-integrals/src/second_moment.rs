//! 二阶矩 `⟨μ|(r−O)_i (r−O)_j|ν⟩`（GIAO 的**抗磁**项 `⟨(∂H/∂B_i)(∂H/∂B_j)⟩` 需要）。
//!
//! # 为什么不需要新的高斯递推
//!
//! 乘一个从 `O` 量的直角坐标分量是初等的（与 [`crate::giao`] 的磁矩积分同一条恒等式，
//! 只是把 `A_l` 换成 `A_l − O_l`）：
//!
//! ```text
//! (x_l − O_l) · χ_a = χ_{a+e_l} + (A_l − O_l) · χ_a
//! ```
//!
//! 用两次（`χ_{a+e_i}` 的中心仍是 `A`，所以第二次的因子还是 `A_j − O_j`）：
//!
//! ```text
//! (x_i − O_i)(x_j − O_j) χ_a
//!   = χ_{a+e_i+e_j}
//!   + (A_j − O_j) χ_{a+e_i} + (A_i − O_i) χ_{a+e_j}
//!   + (A_i − O_i)(A_j − O_j) χ_a
//! ```
//!
//! 每一项都是**普通重叠积分**，只是 bra 的幂次被抬高了一次或两次 ——
//! 复用 [`crate::one_electron::primitive_overlap`]（它对任意幂次都成立）与
//! `momentum` 里抬幂次的小工具即可，**没有**任何新的递推、新特殊函数。
//! 注意抬高的那一项**不**乘 `N_{a+e_l}`：归一化用原分量的 `N_a`（与
//! [`crate::giao`] 约定 1、`momentum` 第 399 轮的教训一致）。
//!
//! # 三个必须成立的性质（都由测试钉住）
//!
//! 1. `i ↔ j` 对称：上式的系数对 `i, j` 完全对称。
//! 2. `μ ↔ ν` 对称：`(r−O)_i (r−O)_j` 是厄米算符的实乘积。
//! 3. `Σ_i result[i][i] = ⟨μ||r−O|²|ν⟩`。
//!
//! 另外还有一条**强**尺子：换原点 `O → O+δ` 时
//!
//! ```text
//! result_ij(O+δ) = result_ij(O) − δ_j D_i − δ_i D_j + δ_i δ_j S
//! ```
//!
//! （`D_i = ⟨μ|(r−O)_i|ν⟩`、`S` 是重叠）。它同时锁住系数、原点处理和 `A_l` 项。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::EngineError;
use qwb_linalg::Matrix;

use crate::dipole::transform_matrix;
use crate::momentum::{power_along, with_power};
use crate::one_electron::{primitive_norm, primitive_overlap};

/// 九个矩阵 `result[i][j][μ,ν] = ⟨μ|(r−O)_i (r−O)_j|ν⟩`（原子单位 `e·bohr²`）。
///
/// `origin` 是电子位置量的原点 `O`，单位 **Bohr**（与
/// [`qwb_basis::InternalShell::center_bohr`] 同一坐标系）；要 `⟨μ|r_i r_j|ν⟩` 传
/// `[0.0; 3]`，或用便捷版本 [`second_moment_integrals_at_origin`]。
///
/// 注意与 [`crate::dipole::second_moment_integrals`] 的区别：那个返回**相对坐标原点**的
/// 六个分量 `[xx, yy, zz, xy, xz, yz]`，这里返回**相对任意 `O`** 的完整 `3×3`。
///
/// # Errors
///
/// 球谐变换未实现（`l` 过大）→ [`EngineError::Unsupported`]；矩阵维度不一致 →
/// [`EngineError::Internal`]（来自 [`transform_matrix`]）。
pub fn second_moment_integrals(
    basis: &InternalBasis,
    origin: [f64; 3],
) -> Result<[[Matrix; 3]; 3], EngineError> {
    let mut transforms = Vec::with_capacity(basis.shells().len());
    for shell in basis.shells() {
        let transform = qwb_basis::spherical::spherical_transform(shell.l, shell.angular)
            .ok_or_else(|| {
                EngineError::Unsupported(format!(
                    "spherical transformation for l={} is not implemented (R06)",
                    shell.l
                ))
            })?;
        transforms.push(transform);
    }
    let mut cart_offsets = Vec::with_capacity(transforms.len());
    let mut n_cartesian = 0usize;
    for transform in &transforms {
        cart_offsets.push(n_cartesian);
        n_cartesian += transform.n_cartesian;
    }
    let mut moments: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian)));

    let shells: Vec<_> = basis.shells().iter().collect();
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let components_a = qwb_basis::cartesian_components(shell_a.l);
            let components_b = qwb_basis::cartesian_components(shell_b.l);
            for (ca, powers_a) in components_a.iter().enumerate() {
                for (cb, powers_b) in components_b.iter().enumerate() {
                    let mut values = [[0.0_f64; 3]; 3];
                    for (alpha, coefficient_a) in
                        shell_a.exponents.iter().zip(&shell_a.coefficients)
                    {
                        let norm_a = primitive_norm(*powers_a, *alpha);
                        for (beta, coefficient_b) in
                            shell_b.exponents.iter().zip(&shell_b.coefficients)
                        {
                            let norm_b = primitive_norm(*powers_b, *beta);
                            let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                            let primitive = primitive_second_moments(
                                *powers_a,
                                shell_a.center_bohr,
                                *alpha,
                                *powers_b,
                                shell_b.center_bohr,
                                *beta,
                                origin,
                            );
                            for (first, row) in primitive.iter().enumerate() {
                                for (second, value) in row.iter().enumerate() {
                                    values[first][second] += weight * value;
                                }
                            }
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for (first, block) in moments.iter_mut().enumerate() {
                        for (second, moment) in block.iter_mut().enumerate() {
                            moment[(row, column)] = values[first][second];
                        }
                    }
                }
            }
        }
    }

    let n_spherical = basis.n_functions();
    let mut global_transform = Matrix::zeros(n_cartesian, n_spherical);
    for (shell_index, shell) in shells.iter().enumerate() {
        let transform = &transforms[shell_index];
        for row in 0..transform.n_cartesian {
            for column in 0..transform.n_spherical {
                global_transform[(
                    cart_offsets[shell_index] + row,
                    shell.first_function + column,
                )] = transform.get(row, column);
            }
        }
    }

    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n_spherical, n_spherical)));
    for (first, block) in out.iter_mut().enumerate() {
        for (second, matrix) in block.iter_mut().enumerate() {
            *matrix = transform_matrix(&moments[first][second], &global_transform)?;
        }
    }
    Ok(out)
}

/// [`second_moment_integrals`] 在 `O = [0.0; 3]` 处的便捷版本（`⟨μ|r_i r_j|ν⟩`）。
///
/// # Errors
///
/// 与 [`second_moment_integrals`] 相同。
pub fn second_moment_integrals_at_origin(
    basis: &InternalBasis,
) -> Result<[[Matrix; 3]; 3], EngineError> {
    second_moment_integrals(basis, [0.0; 3])
}

/// 一个原函数对的九个分量（**未归一化**原函数，不含收缩系数与 `N`）：
/// `⟨χ_a|(r−O)_i (r−O)_j|χ_b⟩`。
///
/// 见模块文档的展开：抬一次的重叠、抬两次的重叠、以及 `(A−O)` 因子。
fn primitive_second_moments(
    powers_a: (u32, u32, u32),
    a: [f64; 3],
    alpha: f64,
    powers_b: (u32, u32, u32),
    b: [f64; 3],
    beta: f64,
    origin: [f64; 3],
) -> [[f64; 3]; 3] {
    let overlap = |powers: (u32, u32, u32)| primitive_overlap(alpha, powers, a, beta, powers_b, b);
    let plain = overlap(powers_a);
    let mut single = [0.0_f64; 3];
    for (axis, value) in single.iter_mut().enumerate() {
        *value = overlap(with_power(powers_a, axis, power_along(powers_a, axis) + 1));
    }
    // `+e_i+e_j` 对 `(i, j)` 对称：只算上三角（省一半重叠调用）。
    let mut double = [[0.0_f64; 3]; 3];
    #[allow(clippy::needless_range_loop)] // 上三角写法必须同时用两个下标
    for first in 0..3 {
        for second in first..3 {
            let value = overlap(raise_twice(powers_a, first, second));
            double[first][second] = value;
            double[second][first] = value;
        }
    }

    let mut out = [[0.0_f64; 3]; 3];
    for (first, row) in out.iter_mut().enumerate() {
        let shifted_first = a[first] - origin[first];
        for (second, value) in row.iter_mut().enumerate() {
            let shifted_second = a[second] - origin[second];
            *value = double[first][second]
                + shifted_second * single[first]
                + shifted_first * single[second]
                + shifted_first * shifted_second * plain;
        }
    }
    out
}

/// 把第 `first`、`second` 两个轴的幂次各抬高一次（`a + e_first + e_second`）。
///
/// 必须**在抬高第一次之后**再读第二次的当前幂次：写
/// `with_power(with_power(a, i, a_i+1), j, a_j+1)` 在 `i == j` 时只抬一次
/// （外层把内层的结果覆盖掉）—— 这是本轮 R3 抓到的那半个幂次的错误。
fn raise_twice(powers: (u32, u32, u32), first: usize, second: usize) -> (u32, u32, u32) {
    let once = with_power(powers, first, power_along(powers, first) + 1);
    with_power(once, second, power_along(once, second) + 1)
}

/// 四条尺子（`R1`..`R4`）：
/// - `R1`：`Σ_i result[i][i] = ⟨μ||r−O|²|ν⟩`，右端由**原始坐标矩**（显式 `with_power`
///   抬高）另算一遍；
/// - `R2`：换原点的代数恒等式（最硬的一条：同时锁系数、原点与 `A_l`）；
/// - `R3`：**绝对**尺子 —— 与生产路径不共享任何矩求值，逐轴用
///   `∫ u^k e^{−p u²} du` 的闭式；
/// - `R4` 是"变异检查"，不是常驻测试，见报告。
#[cfg(test)]
mod tests {
    use super::{raise_twice, second_moment_integrals};
    use crate::dipole::{dipole_integrals, transform_matrix};
    use crate::momentum::{power_along, with_power};
    use crate::one_electron::{one_electron, primitive_norm, primitive_overlap};
    use qwb_basis::InternalBasis;
    use qwb_engine_api::{Atom, Molecule};
    use qwb_linalg::Matrix;
    use qwb_registry::Registry;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    /// 水，**三个原子中心全都离开原点**（`A_l − O_l` 不会意外为零）。
    fn displaced_water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.30, -0.25, 0.42]),
                Atom::new("H", [0.30, 0.507, 1.006]),
                Atom::new("H", [0.30, -1.007, 1.006]),
            ],
            0,
            1,
        )
    }

    fn basis(name: &str, molecule: &Molecule) -> InternalBasis {
        let registry = Registry::open(data_root()).expect("qwb-data");
        InternalBasis::build(registry.get_basis(name).expect("basis"), molecule)
            .expect("basis build")
    }

    /// 某个基函数所属壳层的中心。
    fn centre_of(basis: &InternalBasis, function: usize) -> [f64; 3] {
        for shell in basis.shells() {
            if (shell.first_function..shell.first_function + shell.n_functions())
                .contains(&function)
            {
                return shell.center_bohr;
            }
        }
        panic!("基函数下标 {function} 越界");
    }

    /// 参照路径的组装：只共用 `primitive_norm` 与 `transform_matrix`（球谐指标变换），
    /// **不**共用生产路径的任何矩求值。
    fn assemble_reference<F>(basis: &InternalBasis, kernel: &mut F) -> [[Matrix; 3]; 3]
    where
        F: FnMut((u32, u32, u32), [f64; 3], f64, (u32, u32, u32), [f64; 3], f64) -> [[f64; 3]; 3],
    {
        let mut transforms = Vec::with_capacity(basis.shells().len());
        for shell in basis.shells() {
            transforms.push(
                qwb_basis::spherical::spherical_transform(shell.l, shell.angular)
                    .expect("spherical transform"),
            );
        }
        let mut cart_offsets = Vec::with_capacity(transforms.len());
        let mut n_cartesian = 0usize;
        for transform in &transforms {
            cart_offsets.push(n_cartesian);
            n_cartesian += transform.n_cartesian;
        }
        let mut cartesian: [[Matrix; 3]; 3] = std::array::from_fn(|_| {
            std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian))
        });
        let shells: Vec<_> = basis.shells().iter().collect();
        for (ia, shell_a) in shells.iter().enumerate() {
            for (ib, shell_b) in shells.iter().enumerate() {
                let components_a = qwb_basis::cartesian_components(shell_a.l);
                let components_b = qwb_basis::cartesian_components(shell_b.l);
                for (ca, powers_a) in components_a.iter().enumerate() {
                    for (cb, powers_b) in components_b.iter().enumerate() {
                        let mut values = [[0.0_f64; 3]; 3];
                        for (alpha, coefficient_a) in
                            shell_a.exponents.iter().zip(&shell_a.coefficients)
                        {
                            let norm_a = primitive_norm(*powers_a, *alpha);
                            for (beta, coefficient_b) in
                                shell_b.exponents.iter().zip(&shell_b.coefficients)
                            {
                                let norm_b = primitive_norm(*powers_b, *beta);
                                let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                                let primitive = kernel(
                                    *powers_a,
                                    shell_a.center_bohr,
                                    *alpha,
                                    *powers_b,
                                    shell_b.center_bohr,
                                    *beta,
                                );
                                for (first, row) in primitive.iter().enumerate() {
                                    for (second, value) in row.iter().enumerate() {
                                        values[first][second] += weight * value;
                                    }
                                }
                            }
                        }
                        let row = cart_offsets[ia] + ca;
                        let column = cart_offsets[ib] + cb;
                        for (first, block) in cartesian.iter_mut().enumerate() {
                            for (second, matrix) in block.iter_mut().enumerate() {
                                matrix[(row, column)] = values[first][second];
                            }
                        }
                    }
                }
            }
        }

        let n_spherical = basis.n_functions();
        let mut global = Matrix::zeros(n_cartesian, n_spherical);
        for (shell_index, shell) in shells.iter().enumerate() {
            let transform = &transforms[shell_index];
            for row in 0..transform.n_cartesian {
                for column in 0..transform.n_spherical {
                    global[(
                        cart_offsets[shell_index] + row,
                        shell.first_function + column,
                    )] = transform.get(row, column);
                }
            }
        }
        let mut out: [[Matrix; 3]; 3] = std::array::from_fn(|_| {
            std::array::from_fn(|_| Matrix::zeros(n_spherical, n_spherical))
        });
        for (first, block) in out.iter_mut().enumerate() {
            for (second, matrix) in block.iter_mut().enumerate() {
                *matrix = transform_matrix(&cartesian[first][second], &global)
                    .expect("transform reference");
            }
        }
        out
    }

    /// `max |a − b|`（九个矩阵逐元）与 `max |a|`（尺度）。
    fn compare(a: &[[Matrix; 3]; 3], b: &[[Matrix; 3]; 3]) -> (f64, f64) {
        let n = a[0][0].nrows();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                for mu in 0..n {
                    for nu in 0..n {
                        let value = a[i][j].get(mu, nu);
                        scale = scale.max(value.abs());
                        worst = worst.max((value - b[i][j].get(mu, nu)).abs());
                    }
                }
            }
        }
        (worst, scale)
    }

    // ===== R1：Σ_i result[i][i] = ⟨μ||r−O|²|ν⟩（原始坐标矩的第二路径）=====

    /// 抬一次 / 抬两次的**未归一化**重叠（`with_power` 的显式使用）。
    fn raised_overlaps(
        powers_a: (u32, u32, u32),
        a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        b: [f64; 3],
        beta: f64,
    ) -> (f64, [f64; 3], [[f64; 3]; 3]) {
        let overlap =
            |powers: (u32, u32, u32)| primitive_overlap(alpha, powers, a, beta, powers_b, b);
        let plain = overlap(powers_a);
        let mut single = [0.0_f64; 3];
        for (axis, value) in single.iter_mut().enumerate() {
            *value = overlap(with_power(powers_a, axis, power_along(powers_a, axis) + 1));
        }
        let mut double = [[0.0_f64; 3]; 3];
        for (first, row) in double.iter_mut().enumerate() {
            for (second, cell) in row.iter_mut().enumerate() {
                *cell = overlap(raise_twice(powers_a, first, second));
            }
        }
        (plain, single, double)
    }

    /// `⟨χ_a||r−O|²|χ_b⟩` 的第二条路径：先算**原始**矩 `⟨x_i x_j⟩`、`⟨x_i⟩`，再按
    /// `|r−O|² = Σ_i (x_i² − 2O_i x_i + O_i²)` 组合。
    ///
    /// 与生产路径的差别是实质性的：生产把 `(A_l − O_l)` 直接吸收进系数，这里先算
    /// **不含 O** 的原始矩（`x_i²χ_a = χ_{a+2e_i} + 2A_i χ_{a+e_i} + A_i²χ_a`），
    /// 最后才减 `2O_i r_i`。两条路若在任何一处 `A`/`O` 上写错，结果就不一样。
    fn raw_squared_primitive(
        powers_a: (u32, u32, u32),
        a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        b: [f64; 3],
        beta: f64,
        origin: [f64; 3],
    ) -> f64 {
        let (plain, single, double) = raised_overlaps(powers_a, a, alpha, powers_b, b, beta);
        let mut total = 0.0_f64;
        for axis in 0..3 {
            let raw_single = single[axis] + a[axis] * plain;
            let raw_double =
                double[axis][axis] + 2.0 * a[axis] * single[axis] + a[axis] * a[axis] * plain;
            total +=
                raw_double - 2.0 * origin[axis] * raw_single + origin[axis] * origin[axis] * plain;
        }
        total
    }

    #[test]
    fn r1_the_trace_is_the_squared_radius() {
        let molecule = displaced_water();
        let origin = [0.13, -0.41, 0.27];
        for name in ["sto-3g", "def2-svp"] {
            let basis = basis(name, &molecule);
            let n = basis.n_functions();
            let production = second_moment_integrals(&basis, origin).expect("second moment");
            let reference =
                assemble_reference(&basis, &mut |powers_a, a, alpha, powers_b, b, beta| {
                    let value =
                        raw_squared_primitive(powers_a, a, alpha, powers_b, b, beta, origin);
                    [[value, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                });

            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for mu in 0..n {
                for nu in 0..n {
                    let mut trace = 0.0_f64;
                    for (axis, block) in production.iter().enumerate() {
                        scale = scale.max(block[axis].get(mu, nu).abs());
                        trace += block[axis].get(mu, nu);
                    }
                    worst = worst.max((trace - reference[0][0].get(mu, nu)).abs());
                }
            }
            // i↔j 与 μ↔ν 对称性（实基）。
            let mut transposed_ij = 0.0_f64;
            let mut transposed_munu = 0.0_f64;
            for (i, row) in production.iter().enumerate() {
                for (j, matrix) in row.iter().enumerate() {
                    for mu in 0..n {
                        for nu in 0..n {
                            transposed_ij = transposed_ij
                                .max((matrix.get(mu, nu) - production[j][i].get(mu, nu)).abs());
                            transposed_munu = transposed_munu
                                .max((matrix.get(mu, nu) - matrix.get(nu, mu)).abs());
                        }
                    }
                }
            }
            println!(
                "R1 {name}: max|trace − <|r−O|²>| = {worst:.3e}，max|result| = {scale:.3e}，\
                 max|R_ij − R_ji| = {transposed_ij:.3e}，max|R_μν − R_νμ| = {transposed_munu:.3e}"
            );
            assert!(
                scale > 1.0,
                "R1 {name}: 矩阵太小（{scale:.3e}），尺子是空的"
            );
            assert!(worst.is_finite(), "R1 {name}: 出现非有限值");
            assert!(
                worst / scale < 1.0e-12,
                "R1 {name}: 迹与 |r−O|² 差 {worst:.3e}（max|result| = {scale:.3e}，判据 1e-12 相对）"
            );
            assert!(
                transposed_ij / scale < 1.0e-12,
                "R1 {name}: i↔j 不对称 {transposed_ij:.3e}（判据 1e-12 相对）"
            );
            assert!(
                transposed_munu / scale < 1.0e-12,
                "R1 {name}: μ↔ν 不对称 {transposed_munu:.3e}（判据 1e-12 相对）"
            );
        }
    }

    /// 额外的一条交叉尺子：`O = 0` 时九个矩阵必须与**已有**的
    /// [`crate::dipole::second_moment_integrals`]（走 `overlap_1d` 递推，是另一份独立实现）
    /// 的分量逐个对上（顺序 `[xx, yy, zz, xy, xz, yz]`）。
    #[test]
    fn the_origin_case_matches_the_existing_dipole_second_moment() {
        let molecule = displaced_water();
        for name in ["sto-3g", "def2-svp"] {
            let basis = basis(name, &molecule);
            let n = basis.n_functions();
            let at_origin = second_moment_integrals(&basis, [0.0; 3]).expect("R(O=0)");
            let legacy =
                crate::dipole::second_moment_integrals(&basis, &molecule).expect("已有的二阶矩");
            let map = [
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
                (0, 1, 3),
                (0, 2, 4),
                (1, 2, 5),
            ];
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for (first, second, component) in map {
                for mu in 0..n {
                    for nu in 0..n {
                        let value = at_origin[first][second].get(mu, nu);
                        scale = scale.max(value.abs());
                        worst = worst.max((value - legacy[component].get(mu, nu)).abs());
                    }
                }
            }
            println!(
                "R1′ {name}: O=0 时与 dipole::second_moment_integrals 最差 {worst:.3e}\
                 （max|result| = {scale:.3e}）"
            );
            assert!(
                scale > 1.0,
                "R1′ {name}: 矩阵太小（{scale:.3e}），尺子是空的"
            );
            assert!(
                worst / scale < 1.0e-12,
                "R1′ {name}: 与已有二阶矩差 {worst:.3e}（max|result| = {scale:.3e}，判据 1e-12 相对）"
            );
        }
    }

    // ===== R2：换原点的代数恒等式（最硬的一条）=====
    #[test]
    fn r2_the_origin_shift_identity_holds() {
        let molecule = displaced_water();
        let origin = [0.13, -0.41, 0.27];
        for delta in [[0.31, -0.17, 0.23], [-0.42, 0.28, -0.19]] {
            let shifted_origin = [
                origin[0] + delta[0],
                origin[1] + delta[1],
                origin[2] + delta[2],
            ];
            for name in ["sto-3g", "def2-svp"] {
                let basis = basis(name, &molecule);
                let n = basis.n_functions();
                let at_origin = second_moment_integrals(&basis, origin).expect("R(O)");
                let at_shifted = second_moment_integrals(&basis, shifted_origin).expect("R(O+δ)");
                // D_i = ⟨μ|(r−O)_i|ν⟩ = ⟨μ|x_i|ν⟩ − O_i·S（同一原点 O）。
                let dipole = dipole_integrals(&basis, &molecule).expect("dipole");
                let overlap = one_electron(&basis, &molecule)
                    .expect("one electron")
                    .overlap;

                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                let mut predicted = 0.0_f64;
                for mu in 0..n {
                    for nu in 0..n {
                        let s = overlap.get(mu, nu);
                        let d: [f64; 3] =
                            std::array::from_fn(|axis| dipole[axis].get(mu, nu) - origin[axis] * s);
                        for (i, row) in at_origin.iter().enumerate() {
                            for (j, matrix) in row.iter().enumerate() {
                                let expected =
                                    matrix.get(mu, nu) - delta[j] * d[i] - delta[i] * d[j]
                                        + delta[i] * delta[j] * s;
                                let measured = at_shifted[i][j].get(mu, nu);
                                worst = worst.max((measured - expected).abs());
                                predicted = predicted.max((measured - matrix.get(mu, nu)).abs());
                                scale = scale.max(matrix.get(mu, nu).abs());
                                scale = scale.max(measured.abs());
                            }
                        }
                    }
                }
                println!(
                    "R2 {name} δ={delta:?}: max|result| = {scale:.3e}，max|Δ| = {predicted:.3e}，\
                     原点平移恒等式最差 {worst:.3e}（相对 {:.3e}）",
                    worst / scale
                );
                assert!(
                    predicted > 1.0e-6,
                    "R2 {name}: 平移前后整体没变化（{predicted:.3e}），尺子是空的"
                );
                assert!(
                    worst / scale < 1.0e-12,
                    "R2 {name} δ={delta:?}: 恒等式差 {worst:.3e}（max|result| = {scale:.3e}，判据 1e-12 相对）"
                );
            }
        }
    }

    // ===== R3：**绝对**尺子 = 显式多项式 × 1D 解析高斯矩 =====
    //
    // 与 giao 的动能矩测试同一风格：把 `(r−O)_i (r−O)_j` 与被积函数都写成
    // `u = x − P`（高斯乘积中心）的多项式，逐轴用
    // `∫ u^k e^{−p u²} du = (k−1)!!/(2p)^{k/2} √(π/p)`（k 偶）/ 0（k 奇）收缩。
    // 这条路不碰 `primitive_overlap`、`hermite_1d`、`overlap_1d`，也不碰生产路径的
    // `(A−O)` 展开。

    /// `∫ u^k e^{−p u²} du`。
    fn gaussian_moment(k: usize, p: f64) -> f64 {
        if k % 2 == 1 {
            return 0.0;
        }
        let mut double_factorial = 1.0_f64;
        let mut j = 1usize;
        while j < k {
            double_factorial *= j as f64;
            j += 2;
        }
        double_factorial / (2.0 * p).powf(0.5 * k as f64) * (std::f64::consts::PI / p).sqrt()
    }

    /// `C(n,k)`。
    fn binomial(n: usize, k: usize) -> f64 {
        let mut value = 1.0_f64;
        for i in 0..k {
            value = value * (n - i) as f64 / (i + 1) as f64;
        }
        value
    }

    /// `(u + offset)^n` 按 `u^k` 升幂的系数。
    fn shifted_powers(n: u32, offset: f64) -> Vec<f64> {
        let n = n as usize;
        (0..=n)
            .map(|k| binomial(n, k) * offset.powi((n - k) as i32))
            .collect()
    }

    /// 两个 1D 多项式相乘（升幂）。
    fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            if *ai == 0.0 {
                continue;
            }
            for (j, bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    }

    /// `∫ poly(u) e^{−p u²} du`。
    fn poly_contract(poly: &[f64], p: f64) -> f64 {
        poly.iter()
            .enumerate()
            .map(|(k, c)| c * gaussian_moment(k, p))
            .sum()
    }

    /// `⟨χ_a|(r−O)_i (r−O)_j|χ_b⟩` 的**闭式**值（未归一化原函数）。
    ///
    /// 算符在第 `l` 轴的因子是 `(x_l − O_l) = (u + P_l − O_l)`；`i == j` 时同一轴乘两次。
    fn reference_primitive(
        powers_a: (u32, u32, u32),
        a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        b: [f64; 3],
        beta: f64,
        origin: [f64; 3],
        first: usize,
        second: usize,
    ) -> f64 {
        let p = alpha + beta;
        let product: [f64; 3] = std::array::from_fn(|axis| (alpha * a[axis] + beta * b[axis]) / p);
        let ab2: f64 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum();
        let e_ab = (-alpha * beta / p * ab2).exp();
        let pa: [f64; 3] = std::array::from_fn(|axis| product[axis] - a[axis]);
        let pb: [f64; 3] = std::array::from_fn(|axis| product[axis] - b[axis]);

        let mut total = e_ab;
        for axis in 0..3 {
            let bra = shifted_powers(power_along(powers_a, axis), pa[axis]);
            let ket = shifted_powers(power_along(powers_b, axis), pb[axis]);
            let mut poly = poly_mul(&bra, &ket);
            // 算符在这一轴的因子；`first == second == axis` 时命中两次 ⇒ `(u+c)²`。
            let mut factor = vec![1.0_f64];
            for l in [first, second] {
                if l == axis {
                    factor = poly_mul(&factor, &[product[axis] - origin[axis], 1.0]);
                }
            }
            poly = poly_mul(&poly, &factor);
            total *= poly_contract(&poly, p);
        }
        total
    }

    #[test]
    fn r3_the_tensor_matches_the_analytic_gaussian_moment_reference() {
        let molecule = displaced_water();
        for (name, origin) in [
            ("sto-3g", [0.13, -0.41, 0.27]),
            ("def2-svp", [0.13, -0.41, 0.27]),
            ("sto-3g", [0.0, 0.0, 0.0]),
        ] {
            let basis = basis(name, &molecule);
            let production = second_moment_integrals(&basis, origin).expect("second moment");
            let reference =
                assemble_reference(&basis, &mut |powers_a, a, alpha, powers_b, b, beta| {
                    let mut values = [[0.0_f64; 3]; 3];
                    for (first, row) in values.iter_mut().enumerate() {
                        for (second, value) in row.iter_mut().enumerate() {
                            *value = reference_primitive(
                                powers_a, a, alpha, powers_b, b, beta, origin, first, second,
                            );
                        }
                    }
                    values
                });
            let (worst, scale) = compare(&production, &reference);
            println!(
                "R3 {name} O={origin:?}: max|production − 闭式矩| = {worst:.3e}，\
                 max|result| = {scale:.3e}（相对 {:.3e}）",
                worst / scale
            );
            assert!(
                scale > 1.0,
                "R3 {name}: 矩阵太小（{scale:.3e}），尺子是空的"
            );
            assert!(
                worst / scale < 1.0e-10,
                "R3 {name}: 与闭式矩参考差 {worst:.3e}（max|result| = {scale:.3e}，判据 1e-10 相对）"
            );

            // 点名几个具体矩阵元：一对 i≠j、以及一个**两中心不同**的元素。
            let n = basis.n_functions();
            let (mu, nu) = (0usize, n - 1);
            println!(
                "R3 {name} 抽查 R[0][1]({mu},{nu})：生产 {:.9e}，闭式 {:.9e}；\
                 R[2][2]({mu},{nu})：生产 {:.9e}，闭式 {:.9e}；\
                 中心 {mu} = {:?}、{nu} = {:?}（μ≠ν 两中心）",
                production[0][1].get(mu, nu),
                reference[0][1].get(mu, nu),
                production[2][2].get(mu, nu),
                reference[2][2].get(mu, nu),
                centre_of(&basis, mu),
                centre_of(&basis, nu),
            );
            assert!(
                production[0][1].get(mu, nu).abs() > 1.0e-12,
                "R3 {name}: 抽查的 i≠j 元素太小（{:.3e}），点名不是空的",
                production[0][1].get(mu, nu)
            );
            assert!(
                centre_of(&basis, mu) != centre_of(&basis, nu),
                "R3 {name}: 抽查的两个函数恰好在同一中心"
            );
        }
    }
}
