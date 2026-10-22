//! 一电子积分的实现（McMurchie–Davidson Hermite 展开）。

use qwb_basis::{InternalBasis, InternalShell};
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::elements::atomic_number;

use crate::boys::boys_functions;

const PI: f64 = std::f64::consts::PI;

/// 直角坐标分量 `(lx, ly, lz)`。
type Powers = (u32, u32, u32);

/// `(2n−1)!!`，约定 `(−1)!! = 1`。
fn double_factorial_odd(n: u32) -> f64 {
    let mut value = 1.0;
    let mut k = 1_u32;
    while k < 2 * n {
        value *= f64::from(k);
        k += 2;
    }
    value
}

/// 归一化原函数常数。
///
/// **按分量**归一：`N = (2α/π)^{3/4} (4α)^{L/2} / sqrt(∏_axes (2l_axis − 1)!!)`。
///
/// 注意不能用总的 `(2L−1)!!`：`(x−A)²` 与 `(x−A)(y−A)` 都是 L=2，但
/// `∫x⁴ ≠ ∫x²y²`，两者的归一化常数不同（用 `(2L−1)!!` 会让 xy 型分量的
/// 自重叠变成 1/3 —— 这正是测试抓到的错误）。
#[must_use]
pub fn primitive_norm(powers: Powers, alpha: f64) -> f64 {
    let l_total = powers.0 + powers.1 + powers.2;
    let product = double_factorial_odd(powers.0)
        * double_factorial_odd(powers.1)
        * double_factorial_odd(powers.2);
    (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(0.5 * f64::from(l_total)) / product.sqrt()
}

/// 一维 Hermite 系数 `E_t^{i,j}`（`t = 0..=i+j`）。
///
/// 递推（McMurchie–Davidson）：
/// `E_t^{i+1,j} = (1/2p)E_{t-1}^{i,j} + (P−A)_x E_t^{i,j} + (t+1)E_{t+1}^{i,j}`，
/// `j` 方向同理（用 `(P−B)_x`）。
fn hermite_1d(i: u32, j: u32, p: f64, pa: f64, pb: f64) -> Vec<f64> {
    let i_max = i as usize;
    let j_max = j as usize;
    let t_max = i_max + j_max;
    let mut table = vec![vec![vec![0.0_f64; t_max + 1]; j_max + 1]; i_max + 1];
    table[0][0][0] = 1.0;
    let inv_2p = 0.5 / p;

    for ii in 0..=i_max {
        for jj in 0..=j_max {
            if ii == 0 && jj == 0 {
                continue;
            }
            for t in 0..=t_max {
                let mut value = 0.0;
                if ii > 0 {
                    let previous = &table[ii - 1][jj];
                    if t > 0 {
                        value += inv_2p * previous[t - 1];
                    }
                    value += pa * previous[t];
                    if t < t_max {
                        value += f64::from(t as u32 + 1) * previous[t + 1];
                    }
                } else {
                    let previous = &table[ii][jj - 1];
                    if t > 0 {
                        value += inv_2p * previous[t - 1];
                    }
                    value += pb * previous[t];
                    if t < t_max {
                        value += f64::from(t as u32 + 1) * previous[t + 1];
                    }
                }
                table[ii][jj][t] = value;
            }
        }
    }
    table[i_max][j_max].clone()
}

/// 高斯积：`p, P, E_AB` 与三个方向的 Hermite 系数。
fn gaussian_product(
    alpha: f64,
    a_powers: Powers,
    a_center: [f64; 3],
    beta: f64,
    b_powers: Powers,
    b_center: [f64; 3],
) -> (f64, f64, [f64; 3], [Vec<f64>; 3]) {
    let p = alpha + beta;
    let p_center = [
        (alpha * a_center[0] + beta * b_center[0]) / p,
        (alpha * a_center[1] + beta * b_center[1]) / p,
        (alpha * a_center[2] + beta * b_center[2]) / p,
    ];
    let r2 = (a_center[0] - b_center[0]).powi(2)
        + (a_center[1] - b_center[1]).powi(2)
        + (a_center[2] - b_center[2]).powi(2);
    let e_ab = (-alpha * beta / p * r2).exp();

    let pa = [
        p_center[0] - a_center[0],
        p_center[1] - a_center[1],
        p_center[2] - a_center[2],
    ];
    let pb = [
        p_center[0] - b_center[0],
        p_center[1] - b_center[1],
        p_center[2] - b_center[2],
    ];
    let hermite = [
        hermite_1d(a_powers.0, b_powers.0, p, pa[0], pb[0]),
        hermite_1d(a_powers.1, b_powers.1, p, pa[1], pb[1]),
        hermite_1d(a_powers.2, b_powers.2, p, pa[2], pb[2]),
    ];
    (e_ab, p, p_center, hermite)
}

/// 两个（未归一化）原函数的重叠。
#[must_use]
pub fn primitive_overlap(
    alpha: f64,
    a_powers: Powers,
    a_center: [f64; 3],
    beta: f64,
    b_powers: Powers,
    b_center: [f64; 3],
) -> f64 {
    let (e_ab, p, _, hermite) =
        gaussian_product(alpha, a_powers, a_center, beta, b_powers, b_center);
    e_ab * (PI / p).powf(1.5) * hermite[0][0] * hermite[1][0] * hermite[2][0]
}

/// 两个（未归一化）原函数的动能 `⟨a|−∇²/2|b⟩`。
///
/// 用分部积分 `T = (1/2)∫(∇a)·(∇b)`，而 `∂a/∂x = [i(x−A)^{i−1} − 2α(x−A)^{i+1}] e^{…}`，
/// 于是每个分量都化成**错位重叠**——不需要第二套核函数。
#[must_use]
pub fn primitive_kinetic(
    alpha: f64,
    a_powers: Powers,
    a_center: [f64; 3],
    beta: f64,
    b_powers: Powers,
    b_center: [f64; 3],
) -> f64 {
    let a = [a_powers.0, a_powers.1, a_powers.2];
    let b = [b_powers.0, b_powers.1, b_powers.2];
    let mut total = 0.0;

    for axis in 0..3 {
        let ai = a[axis];
        let bi = b[axis];

        // 把该轴的幂次错位后重算重叠。
        let shifted = |delta_a: i32, delta_b: i32| -> f64 {
            let mut a_p = a;
            let mut b_p = b;
            a_p[axis] = (i64::from(ai) + i64::from(delta_a)) as u32;
            b_p[axis] = (i64::from(bi) + i64::from(delta_b)) as u32;
            primitive_overlap(
                alpha,
                (a_p[0], a_p[1], a_p[2]),
                a_center,
                beta,
                (b_p[0], b_p[1], b_p[2]),
                b_center,
            )
        };

        let mut axis_sum = 0.0;
        if ai >= 1 && bi >= 1 {
            axis_sum += f64::from(ai * bi) * shifted(-1, -1);
        }
        if ai >= 1 {
            axis_sum -= 2.0 * beta * f64::from(ai) * shifted(-1, 1);
        }
        if bi >= 1 {
            axis_sum -= 2.0 * alpha * f64::from(bi) * shifted(1, -1);
        }
        axis_sum += 4.0 * alpha * beta * shifted(1, 1);
        total += axis_sum;
    }
    0.5 * total
}

/// 两个（未归一化）原函数的核吸引 `⟨a|−Z/|r−C||b⟩`。
///
/// # 已知限制（I-02 未完成项）
///
/// 本函数用 `R^n = (−2p)^n F_n` 的简化 Hermite Coulomb 系数，它**只在 `l_total = 0`
/// 时正确**。l>0 时该系数缺少方向依赖（`(P−C)` 的奇次项），例如同中心的
/// `⟨s|1/r|p_x⟩` 本应为 0，这个公式却给出非零值 —— 由
/// `tests/d02_one_electron.rs::nuclear_attraction_for_l_above_zero_is_reported_unsupported`
/// 记录。正确实现需要完整的 Hermite Coulomb 递推（Helgaker 9.4 或 Rys 求积），
/// 排在 I-05（那里本来就要和数值积分对照 < 1e-8）。
///
/// 因此 [one_electron] 在遇到任何 l > 0 的壳层时返回 Unsupported，而**不会**
/// 给出看起来像结果、其实错的值。
#[must_use]
#[allow(clippy::too_many_arguments)] // 数学签名：两对(指数,分量,中心) + 核位置 + 电荷
pub fn primitive_nuclear(
    alpha: f64,
    a_powers: Powers,
    a_center: [f64; 3],
    beta: f64,
    b_powers: Powers,
    b_center: [f64; 3],
    nucleus_center: [f64; 3],
    nucleus_charge: f64,
) -> f64 {
    let (e_ab, p, p_center, hermite) =
        gaussian_product(alpha, a_powers, a_center, beta, b_powers, b_center);

    let pc2 = (p_center[0] - nucleus_center[0]).powi(2)
        + (p_center[1] - nucleus_center[1]).powi(2)
        + (p_center[2] - nucleus_center[2]).powi(2);

    let l_total = a_powers.0 + a_powers.1 + a_powers.2 + b_powers.0 + b_powers.1 + b_powers.2;
    let boys = boys_functions(l_total as usize, p * pc2);

    let mut sum = 0.0;
    for (t, e_t) in hermite[0].iter().enumerate() {
        if *e_t == 0.0 {
            continue;
        }
        for (u, e_u) in hermite[1].iter().enumerate() {
            if *e_u == 0.0 {
                continue;
            }
            for (v, e_v) in hermite[2].iter().enumerate() {
                if *e_v == 0.0 {
                    continue;
                }
                let n = t + u + v;
                // Hermite Coulomb 积分 R^n = (-2p)^n F_n
                let r = (-2.0 * p).powi(i32::try_from(n).expect("n fits i32")) * boys[n];
                sum += e_t * e_u * e_v * r;
            }
        }
    }

    -nucleus_charge * (2.0 * PI / p) * e_ab * sum
}

/// 一个壳层的原函数数据（系数保持数据原样；归一化在积分时按分量施加）。
struct ShellPrimitives {
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    components: Vec<Powers>,
    first_function: usize,
    center: [f64; 3],
    l: u32,
}

fn shell_primitives(shell: &InternalShell) -> ShellPrimitives {
    ShellPrimitives {
        exponents: shell.exponents.clone(),
        coefficients: shell.coefficients.clone(),
        components: qwb_basis::cartesian_components(shell.l),
        first_function: shell.first_function,
        center: shell.center_bohr,
        l: shell.l,
    }
}

/// 一电子积分矩阵（全部对称）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OneElectronIntegrals {
    /// 重叠矩阵 `S`。
    pub overlap: Matrix,
    /// 动能矩阵 `T`。
    pub kinetic: Matrix,
    /// 核吸引矩阵 `V`（对全部核求和）。
    pub nuclear: Matrix,
    /// 核心哈密顿量 `H = T + V`。
    pub core_hamiltonian: Matrix,
}

/// 计算一个分子基组下的全部一电子积分。
///
/// # Errors
///
/// 分子里出现元素表之外的符号（拿不到核电荷）→ [`EngineError::Internal`]。
pub fn one_electron(
    basis: &InternalBasis,
    molecule: &Molecule,
) -> Result<OneElectronIntegrals, EngineError> {
    let n = basis.n_functions();
    let mut overlap = Matrix::zeros(n, n);
    let mut kinetic = Matrix::zeros(n, n);
    let mut nuclear = Matrix::zeros(n, n);

    // 核电荷（ECP 会在 I-05 改成有效核电荷；这里先全电子）。
    let mut nuclei: Vec<([f64; 3], f64)> = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        let z = atomic_number(&atom.symbol).ok_or_else(|| {
            EngineError::Internal(format!(
                "unknown element `{}` while building nuclear attraction",
                atom.symbol
            ))
        })?;
        nuclei.push((atom.position_bohr(), f64::from(z)));
    }

    let shells: Vec<ShellPrimitives> = basis.shells().iter().map(shell_primitives).collect();

    // I-02 的已知缺口：核吸引的 l>0 项需要完整的 Hermite Coulomb 递推（见
    // `primitive_nuclear` 的文档）。这里显式报 Unsupported，而不是返回错值。
    if let Some(shell) = shells.iter().find(|shell| shell.l > 0) {
        return Err(EngineError::Unsupported(format!(
            "nuclear attraction for l>0 shells is not implemented yet (I-02 gap, \
             planned together with I-05): the basis has an l={} shell",
            shell.l
        )));
    }

    for (shell_a_index, shell_a) in shells.iter().enumerate() {
        for shell_b in shells.iter().skip(shell_a_index) {
            for (component_a, powers_a) in shell_a.components.iter().enumerate() {
                for (component_b, powers_b) in shell_b.components.iter().enumerate() {
                    // 按分量归一（xy 型与 x² 型的归一化常数不同）。
                    let norms_a: Vec<f64> = shell_a
                        .exponents
                        .iter()
                        .map(|alpha| primitive_norm(*powers_a, *alpha))
                        .collect();
                    let norms_b: Vec<f64> = shell_b
                        .exponents
                        .iter()
                        .map(|beta| primitive_norm(*powers_b, *beta))
                        .collect();

                    let mut s = 0.0;
                    let mut t = 0.0;
                    let mut v = 0.0;
                    for (i, (alpha, coefficient_a)) in shell_a
                        .exponents
                        .iter()
                        .zip(&shell_a.coefficients)
                        .enumerate()
                    {
                        for (j, (beta, coefficient_b)) in shell_b
                            .exponents
                            .iter()
                            .zip(&shell_b.coefficients)
                            .enumerate()
                        {
                            let weight = coefficient_a * coefficient_b * norms_a[i] * norms_b[j];
                            s += weight
                                * primitive_overlap(
                                    *alpha,
                                    *powers_a,
                                    shell_a.center,
                                    *beta,
                                    *powers_b,
                                    shell_b.center,
                                );
                            t += weight
                                * primitive_kinetic(
                                    *alpha,
                                    *powers_a,
                                    shell_a.center,
                                    *beta,
                                    *powers_b,
                                    shell_b.center,
                                );
                            for (nucleus_center, charge) in &nuclei {
                                v += weight
                                    * primitive_nuclear(
                                        *alpha,
                                        *powers_a,
                                        shell_a.center,
                                        *beta,
                                        *powers_b,
                                        shell_b.center,
                                        *nucleus_center,
                                        *charge,
                                    );
                            }
                        }
                    }

                    let row = shell_a.first_function + component_a;
                    let column = shell_b.first_function + component_b;
                    overlap[(row, column)] = s;
                    kinetic[(row, column)] = t;
                    nuclear[(row, column)] = v;
                    if row != column {
                        overlap[(column, row)] = s;
                        kinetic[(column, row)] = t;
                        nuclear[(column, row)] = v;
                    }
                }
            }
        }
    }

    let mut core_hamiltonian = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            core_hamiltonian[(i, j)] = kinetic[(i, j)] + nuclear[(i, j)];
        }
    }

    Ok(OneElectronIntegrals {
        overlap,
        kinetic,
        nuclear,
        core_hamiltonian,
    })
}
