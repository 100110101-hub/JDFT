//! 角动量积分 `⟨μ| (r − O) × ∇ |ν⟩`（RS-04 顺磁屏蔽要用）。
//!
//! 物理上的轨道角动量算符是 `L = −i (r − O) × ∇`，所以这里的矩阵是**实数**部分
//! `N_α = ε_αβγ ⟨μ| (r−O)_β ∂_γ |ν⟩`，`L_α = −i N_α`（在实数基下 `L` 纯虚反对称）。
//!
//! 不引入新的特殊函数：`∂_γ(x−B)^q e^{−β(x−B)²} = q(x−B)^{q−1} − 2β(x−B)^{q+1}`，
//! 所以「偶极（抬高一个幂次）× 导数（幂次 ±1）」仍然只是 [`overlap_1d`] 的组合 ——
//! 和偶极积分、二阶矩积分同一套绕法。
//!
//! 单位：原子单位（`ℏ = 1`）。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::dipole::{overlap_1d, transform_matrix};
use crate::one_electron::primitive_norm;

/// 把某个轴的幂次加减一个整数（调用方保证不越过 0）。
fn bump(powers: [u32; 3], axis: usize, delta: i32) -> [u32; 3] {
    let mut out = powers;
    out[axis] = (out[axis] as i32 + delta) as u32;
    out
}

/// 三个轴的 `overlap_1d` 乘积；`skip` 指定的轴不参与（那一轴由调用方手算）。
fn product(
    powers_a: [u32; 3],
    powers_b: [u32; 3],
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    skip: Option<usize>,
) -> f64 {
    let mut value = 1.0;
    for axis in 0..3 {
        if Some(axis) == skip {
            continue;
        }
        value *= overlap_1d(
            powers_a[axis],
            powers_b[axis],
            alpha,
            beta,
            a[axis],
            b[axis],
        );
    }
    value
}

/// `⟨φ_μ| (x_β − O_β) ∂_γ φ_ν⟩` 的笛卡尔原语（一个分量对）。
///
/// 把 `(x_β − O_β)` 写成 `(x_β − A_β) + (A_β − O_β)`：前一半是「抬高一个幂次」，
/// 后一半是常数因子；导数那一轴用 `q(x−B)^{q−1} − 2β(x−B)^{q+1}`。
fn primitive_moment_derivative(
    powers_a: [u32; 3],
    a: [f64; 3],
    alpha: f64,
    powers_b: [u32; 3],
    b: [f64; 3],
    beta: f64,
    origin: [f64; 3],
    moment_axis: usize,
    deriv_axis: usize,
) -> f64 {
    let shift = a[moment_axis] - origin[moment_axis];
    if moment_axis == deriv_axis {
        // 幂次抬高与导数落在同一根轴上。
        let k = moment_axis;
        let mut value = 0.0;
        if powers_b[k] > 0 {
            value += powers_b[k] as f64
                * product(
                    bump(powers_a, k, 2),
                    bump(powers_b, k, -1),
                    alpha,
                    beta,
                    a,
                    b,
                    Some(k),
                );
            value += shift
                * powers_b[k] as f64
                * product(
                    bump(powers_a, k, 1),
                    bump(powers_b, k, -1),
                    alpha,
                    beta,
                    a,
                    b,
                    Some(k),
                );
        }
        value += -2.0
            * beta
            * (product(
                bump(powers_a, k, 2),
                bump(powers_b, k, 1),
                alpha,
                beta,
                a,
                b,
                Some(k),
            ) + shift
                * product(
                    bump(powers_a, k, 1),
                    bump(powers_b, k, 1),
                    alpha,
                    beta,
                    a,
                    b,
                    Some(k),
                ));
        value
    } else {
        let moment = moment_axis;
        let deriv = deriv_axis;
        let mut value = 0.0;
        // extra = 1 对应 (x_β − A_β)，extra = 0 对应常数 × (A_β − O_β)。
        for (extra, weight) in [(1_u32, 1.0), (0_u32, shift)] {
            if weight == 0.0 {
                continue;
            }
            let mut powers = powers_a;
            powers[moment] += extra;
            if powers_b[deriv] > 0 {
                value += weight
                    * powers_b[deriv] as f64
                    * product(powers, bump(powers_b, deriv, -1), alpha, beta, a, b, None);
            }
            value += weight
                * -2.0
                * beta
                * product(powers, bump(powers_b, deriv, 1), alpha, beta, a, b, None);
        }
        value
    }
}

/// 一个分量对的三个实角动量分量 `N_α = ε_αβγ ⟨φ_μ|(r−O)_β ∂_γ|φ_ν⟩`。
fn primitive_angular_momentum(
    powers_a: [u32; 3],
    a: [f64; 3],
    alpha: f64,
    powers_b: [u32; 3],
    b: [f64; 3],
    beta: f64,
    origin: [f64; 3],
) -> [f64; 3] {
    let m = |moment: usize, deriv: usize| {
        primitive_moment_derivative(powers_a, a, alpha, powers_b, b, beta, origin, moment, deriv)
    };
    [m(1, 2) - m(2, 1), m(2, 0) - m(0, 2), m(0, 1) - m(1, 0)]
}

/// 三个实角动量积分矩阵 `⟨μ|((r−O) × ∇)_α|ν⟩`（原子单位，球谐基）。
///
/// `origin_bohr` 是规范原点 `O`；`r` 用的是**电子坐标**（与 `dipole_integrals` 一致）。
/// 物理算符 `L_α = −i N_α`。
///
/// # Errors
///
/// 球谐变换未实现（`l > 2`）。
pub fn angular_momentum_integrals(
    basis: &InternalBasis,
    _molecule: &Molecule,
    origin_bohr: [f64; 3],
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
    let mut angular = [
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
                    let pa = [powers_a.0, powers_a.1, powers_a.2];
                    let pb = [powers_b.0, powers_b.1, powers_b.2];
                    let mut values = [0.0_f64; 3];
                    for (alpha, coefficient_a) in
                        shell_a.exponents.iter().zip(&shell_a.coefficients)
                    {
                        let norm_a = primitive_norm(*powers_a, *alpha);
                        for (beta, coefficient_b) in
                            shell_b.exponents.iter().zip(&shell_b.coefficients)
                        {
                            let norm_b = primitive_norm(*powers_b, *beta);
                            let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                            let primitive = primitive_angular_momentum(
                                pa,
                                shell_a.center_bohr,
                                *alpha,
                                pb,
                                shell_b.center_bohr,
                                *beta,
                                origin_bohr,
                            );
                            for axis in 0..3 {
                                values[axis] += weight * primitive[axis];
                            }
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for axis in 0..3 {
                        angular[axis][(row, column)] = values[axis];
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
        out[axis] = transform_matrix(&angular[axis], &global_transform)?;
    }
    Ok(out)
}
