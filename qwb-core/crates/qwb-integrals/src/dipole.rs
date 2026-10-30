//! 偶极积分 `⟨μ|x|ν⟩`（S-06）。
//!
//! 用恒等式 `x = (x−A_x) + A_x`：
//!
//! ```text
//! ⟨μ|x|ν⟩ = N_μ N_ν · [ M(a_x+1, b_x) + A_x·M(a_x, b_x) ] · M(a_y,b_y) · M(a_z,b_z)
//! ```
//!
//! 其中 `M(i,j) = ∫ (x−A)^i (x−B)^j e^{−α(x−A)²−β(x−B)²} dx` 是一维重叠，用标准递推得到——
//! 所以不需要新的特殊函数，偶极积分就是「抬高一个幂次的重叠」。
//!
//! 单位：偶极算符用**原子单位**（`e·bohr`），`qwb-props` 里再换成 Debye。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::one_electron::primitive_norm;

/// 一维重叠 `M(i,j)` 的标准递推。
///
/// `pub(crate)`：`angmom` 模块的「抬高幂次 × 导数」也建在同一个递推上。
pub(crate) fn overlap_1d(i: u32, j: u32, alpha: f64, beta: f64, a: f64, b: f64) -> f64 {
    let p = alpha + beta;
    let center = (alpha * a + beta * b) / p;
    let (iu, ju) = (i as usize, j as usize);
    let mut table = vec![vec![0.0_f64; ju + 1]; iu + 1];
    table[0][0] = (std::f64::consts::PI / p).sqrt() * (-alpha * beta / p * (a - b).powi(2)).exp();
    for ii in 0..=iu {
        for jj in 0..=ju {
            if ii == 0 && jj == 0 {
                continue;
            }
            let mut value;
            if ii > 0 {
                value = (center - a) * table[ii - 1][jj];
                if ii >= 2 {
                    value += (ii as f64 - 1.0) / (2.0 * p) * table[ii - 2][jj];
                }
                if jj >= 1 {
                    value += jj as f64 / (2.0 * p) * table[ii - 1][jj - 1];
                }
            } else {
                value = (center - b) * table[ii][jj - 1];
                if jj >= 2 {
                    value += (jj as f64 - 1.0) / (2.0 * p) * table[ii][jj - 2];
                }
            }
            table[ii][jj] = value;
        }
    }
    table[iu][ju]
}

/// 某个笛卡尔分量对的三个偶极分量（原子单位），未归一化。
fn primitive_dipole(
    powers_a: (u32, u32, u32),
    a: [f64; 3],
    alpha: f64,
    powers_b: (u32, u32, u32),
    b: [f64; 3],
    beta: f64,
) -> [f64; 3] {
    let pa = [powers_a.0, powers_a.1, powers_a.2];
    let pb = [powers_b.0, powers_b.1, powers_b.2];
    let mut out = [0.0; 3];
    for axis in 0..3 {
        let mut value = 1.0;
        for other in 0..3 {
            if other == axis {
                // x·(x−A)^a(x−B)^b = (x−A)^(a+1)(x−B)^b + A_x·(x−A)^a(x−B)^b
                let raised = overlap_1d(pa[axis] + 1, pb[axis], alpha, beta, a[axis], b[axis]);
                let plain = overlap_1d(pa[axis], pb[axis], alpha, beta, a[axis], b[axis]);
                value *= raised + a[axis] * plain;
            } else {
                value *= overlap_1d(pa[other], pb[other], alpha, beta, a[other], b[other]);
            }
        }
        out[axis] = value;
    }
    out
}

/// 把每个指标按壳层的球谐变换处理（与 `one_electron` 完全相同的约定）。
///
/// `pub(crate)`：`angmom` 模块共用。
pub(crate) fn transform_matrix(m: &Matrix, transform: &Matrix) -> Result<Matrix, EngineError> {
    let transposed = transform.transpose();
    let left = qwb_linalg::matmul(&transposed, m)
        .map_err(|err| EngineError::Internal(format!("dipole T^T M: {err}")))?;
    qwb_linalg::matmul(&left, transform)
        .map_err(|err| EngineError::Internal(format!("dipole (T^T M) T: {err}")))
}

/// 三个偶极积分矩阵 `⟨μ|x|ν⟩, ⟨μ|y|ν⟩, ⟨μ|z|ν⟩`（原子单位，球谐基）。
///
/// # Errors
///
/// 球谐变换未实现（`l > 2`）。
pub fn dipole_integrals(
    basis: &InternalBasis,
    _molecule: &Molecule,
) -> Result<[Matrix; 3], EngineError> {
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
    let mut dipole = [
        Matrix::zeros(n_cartesian, n_cartesian),
        Matrix::zeros(n_cartesian, n_cartesian),
        Matrix::zeros(n_cartesian, n_cartesian),
    ];

    let shells: Vec<_> = basis.shells().iter().collect();
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let components_a = qwb_basis::cartesian_components(shell_a.l);
            let components_b = qwb_basis::cartesian_components(shell_b.l);
            for (ca, powers_a) in components_a.iter().enumerate() {
                for (cb, powers_b) in components_b.iter().enumerate() {
                    let mut values = [0.0_f64; 3];
                    for (i, (alpha, coefficient_a)) in shell_a
                        .exponents
                        .iter()
                        .zip(&shell_a.coefficients)
                        .enumerate()
                    {
                        let norm_a = primitive_norm(*powers_a, *alpha);
                        for (j, (beta, coefficient_b)) in shell_b
                            .exponents
                            .iter()
                            .zip(&shell_b.coefficients)
                            .enumerate()
                        {
                            let norm_b = primitive_norm(*powers_b, *beta);
                            let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                            let primitive = primitive_dipole(
                                *powers_a,
                                shell_a.center_bohr,
                                *alpha,
                                *powers_b,
                                shell_b.center_bohr,
                                *beta,
                            );
                            for axis in 0..3 {
                                values[axis] += weight * primitive[axis];
                            }
                            let _ = (i, j);
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for axis in 0..3 {
                        dipole[axis][(row, column)] = values[axis];
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
    let mut out = [
        Matrix::zeros(n_spherical, n_spherical),
        Matrix::zeros(n_spherical, n_spherical),
        Matrix::zeros(n_spherical, n_spherical),
    ];
    for axis in 0..3 {
        out[axis] = transform_matrix(&dipole[axis], &global_transform)?;
    }
    Ok(out)
}

/// 某个笛卡尔分量对的**二阶矩** `⟨μ|r_i r_j|ν⟩`（6 个分量：xx, yy, zz, xy, xz, yz）。
///
/// 与偶极积分同一个绕法：把 `r_i r_j` 拆成 `(x−A)` 的幂次组合，每一项都是
/// 「抬高一个/两个幂次的重叠」（`overlap_1d`），不需要新的特殊函数。
fn primitive_second_moments(
    powers_a: (u32, u32, u32),
    a: [f64; 3],
    alpha: f64,
    powers_b: (u32, u32, u32),
    b: [f64; 3],
    beta: f64,
) -> [f64; 6] {
    let pa = [powers_a.0, powers_a.1, powers_a.2];
    let pb = [powers_b.0, powers_b.1, powers_b.2];
    let m = |axis: usize, raise: u32| {
        overlap_1d(pa[axis] + raise, pb[axis], alpha, beta, a[axis], b[axis])
    };
    let m_plain = |axis: usize| overlap_1d(pa[axis], pb[axis], alpha, beta, a[axis], b[axis]);
    let diag = |axis: usize| -> f64 {
        let mut value = m(axis, 2) + 2.0 * a[axis] * m(axis, 1) + a[axis] * a[axis] * m(axis, 0);
        for other in 0..3 {
            if other != axis {
                value *= m_plain(other);
            }
        }
        value
    };
    let off = |first: usize, second: usize| -> f64 {
        let mut value = m(first, 1) * m(second, 1);
        value += a[first] * m(second, 1) * m(first, 0);
        value += a[second] * m(first, 1) * m(second, 0);
        value += a[first] * a[second] * m(first, 0) * m(second, 0);
        let other = 3 - first - second;
        value * m_plain(other)
    };
    [diag(0), diag(1), diag(2), off(0, 1), off(0, 2), off(1, 2)]
}

/// 二阶矩积分 `⟨μ|r_i r_j|ν⟩`（原子单位 `e·bohr²`），顺序 `[xx, yy, zz, xy, xz, yz]`。
///
/// # Errors
///
/// 球谐变换未实现（`l` 过大）时返回 [`EngineError::Unsupported`]。
pub fn second_moment_integrals(
    basis: &InternalBasis,
    _molecule: &Molecule,
) -> Result<[Matrix; 6], EngineError> {
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
    let mut moments: [Matrix; 6] = std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian));
    let shells: Vec<_> = basis.shells().iter().collect();
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let components_a = qwb_basis::cartesian_components(shell_a.l);
            let components_b = qwb_basis::cartesian_components(shell_b.l);
            for (ca, powers_a) in components_a.iter().enumerate() {
                for (cb, powers_b) in components_b.iter().enumerate() {
                    let mut values = [0.0_f64; 6];
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
                            );
                            for component in 0..6 {
                                values[component] += weight * primitive[component];
                            }
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for component in 0..6 {
                        moments[component][(row, column)] = values[component];
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
    let mut out: [Matrix; 6] = std::array::from_fn(|_| Matrix::zeros(n_spherical, n_spherical));
    for component in 0..6 {
        out[component] = transform_matrix(&moments[component], &global_transform)?;
    }
    Ok(out)
}
