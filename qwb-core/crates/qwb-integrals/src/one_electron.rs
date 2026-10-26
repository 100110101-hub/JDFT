//! 一电子积分的实现（McMurchie–Davidson Hermite 展开）。

use qwb_basis::{InternalBasis, InternalShell};
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::elements::atomic_number;

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
pub(crate) fn hermite_1d(i: u32, j: u32, p: f64, pa: f64, pb: f64) -> Vec<f64> {
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

/// `n!`（`n <= 12` 足够：第一阶段 `l <= 4`）。
#[allow(dead_code)]
fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product::<f64>().max(1.0)
}

/// 二项系数 `C(n,k)`。
#[allow(dead_code)]
fn binomial(n: u32, k: u32) -> f64 {
    if k > n {
        0.0
    } else {
        factorial(n) / (factorial(k) * factorial(n - k))
    }
}

/// 两个（未归一化）原函数的核吸引 `⟨a|−Z/|r−C||b⟩`，**任意角动量**。
///
/// # 推导（不再是"凭记忆的递推"）
///
/// 记 `α = i_a+i_b` 等三个方向的幂次之和、`x = P − C`、`u = p|x|²`。对
/// `G(s) = ∫ e^{−p|v|²} e^{s·v}/|v−x| d³v` 做 Laplace 变换并换元 `y = t/(1+t)`，
/// 得到生成函数
///
/// ```text
/// G(s) = (π/p) ∫₀¹ [ Σ_axes ... ] 即每一对 s 变量配对时给出传播子 1/(4p(1+t))，
/// ```
///
/// 提取 `s_x^α s_y^β s_z^γ` 的系数后，所有 `t` 积分都化成**广义 Boys 函数**
///
/// ```text
/// I(m,N;u) = ∫₀¹ y^{m−1/2}(1−y)^N e^{−uy} dy = 2 Σ_{k=0}^{N} (−1)^k C(N,k) F_{m+k}(u)
/// ```
///
/// （`F_n` 是标准 Boys 函数；`y = s²` 代换给出那个因子 2）。于是
///
/// ```text
/// V = −Z E_AB (π/p) Σ_{a,b,c} W · x_x^{α−2a} x_y^{β−2b} x_z^{γ−2c} (4p)^{−(a+b+c)} I(m,N;u)
/// W = α!/((α−2a)!a!) · β!/((β−2b)!b!) · γ!/((γ−2c)!c!)
/// N = a+b+c,  m = L − 2N,  L = α+β+γ
/// ```
///
/// 两个自检：`L = 0` 时退化为 `−Z(2π/p)E_AB F₀(u)`（教科书 s-s 公式）；
/// `x = 0` 且 `α` 为奇数时 `x_x^{α−2a}` 全部为 0 ⇒ 积分为 0（宇称）。
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
    let p = alpha + beta;
    let p_center = [
        (alpha * a_center[0] + beta * b_center[0]) / p,
        (alpha * a_center[1] + beta * b_center[1]) / p,
        (alpha * a_center[2] + beta * b_center[2]) / p,
    ];
    let ab2 = (0..3)
        .map(|axis| (a_center[axis] - b_center[axis]).powi(2))
        .sum::<f64>();
    let e_ab = (-alpha * beta / p * ab2).exp();

    // Hermite（McMurchie–Davidson）展开：把 (r−A)^a (r−B)^b 展成 Hermite 高斯，
    // 势的对角元于是变成「Boys 函数对 P 的高阶导数」——这正是 hermite2e 里那份
    // 按定义求导的 derivative_boys。
    //
    // 旧实现把两中心的角动量直接相加、并假定多项式中心就在 P 上，等于丢掉了
    // (P−A)、(P−B) 的信息：s 型（a=b=0）看不出来，p/d 型会全错。
    let factors: Vec<Vec<f64>> = (0..3)
        .map(|axis| {
            hermite_1d(
                a_powers_axis(a_powers, axis),
                a_powers_axis(b_powers, axis),
                p,
                p_center[axis] - a_center[axis],
                p_center[axis] - b_center[axis],
            )
        })
        .collect();

    let r = [
        p_center[0] - nucleus_center[0],
        p_center[1] - nucleus_center[1],
        p_center[2] - nucleus_center[2],
    ];

    let n_x = a_powers.0 + b_powers.0;
    let n_y = a_powers.1 + b_powers.1;
    let n_z = a_powers.2 + b_powers.2;
    let mut total = 0.0;
    for tx in 0..=n_x {
        let fx = factors[0][tx as usize];
        if fx == 0.0 {
            continue;
        }
        for ty in 0..=n_y {
            let fy = factors[1][ty as usize];
            if fy == 0.0 {
                continue;
            }
            for tz in 0..=n_z {
                let fz = factors[2][tz as usize];
                if fz == 0.0 {
                    continue;
                }
                // 核吸引：V_t = ∂^t F_0(p|P−C|²)，**不带** (−1)^t。
                // （2e 那边是 (−1) 落在 ket 的 E 系数上，两者约定不同，别混用。）
                total += fx * fy * fz * crate::hermite2e::derivative_boys(tx, ty, tz, p, r);
            }
        }
    }

    -nucleus_charge * 2.0 * PI / p * e_ab * total
}

/// 取幂次的第 `axis` 个分量。
fn a_powers_axis(powers: Powers, axis: usize) -> u32 {
    match axis {
        0 => powers.0,
        1 => powers.1,
        _ => powers.2,
    }
}
/// 一个壳层的原函数数据（系数保持数据原样；归一化在积分时按分量施加）。
struct ShellPrimitives {
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    components: Vec<Powers>,
    center: [f64; 3],
}

fn shell_primitives(shell: &InternalShell) -> ShellPrimitives {
    ShellPrimitives {
        exponents: shell.exponents.clone(),
        coefficients: shell.coefficients.clone(),
        components: qwb_basis::cartesian_components(shell.l),
        center: shell.center_bohr,
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
/// 核电荷取元素表里的 `Z`（全电子）。有 ECP 的体系用
/// [`one_electron_with_charges`] 传入有效核电荷。
///
/// # Errors
///
/// 分子里出现元素表之外的符号（拿不到核电荷）→ [`EngineError::Internal`]。
pub fn one_electron(
    basis: &InternalBasis,
    molecule: &Molecule,
) -> Result<OneElectronIntegrals, EngineError> {
    let mut charges = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        let z = atomic_number(&atom.symbol).ok_or_else(|| {
            EngineError::Internal(format!(
                "unknown element `{}` while building nuclear attraction",
                atom.symbol
            ))
        })?;
        charges.push(f64::from(z));
    }
    one_electron_with_charges(basis, molecule, &charges)
}

/// 与 [`one_electron`] 相同，但核电荷由调用方给（I-05：ECP 元素的核吸引用 `z_eff`）。
///
/// `charges[i]` 是第 `i` 个原子对**电子**的核电荷。ECP 的电子-核有效势里，
/// 点电荷吸引那一半用 `z_eff = Z − n_core`，其余（局域 + 非局域投影）由
/// `qwb_scf::ecp` 单独算成矩阵再加进 `H_core`。
///
/// # Errors
///
/// `charges` 长度与分子原子数不符 → [`EngineError::Internal`]；球谐变换缺失 →
/// [`EngineError::Unsupported`]。
pub fn one_electron_with_charges(
    basis: &InternalBasis,
    molecule: &Molecule,
    charges: &[f64],
) -> Result<OneElectronIntegrals, EngineError> {
    if charges.len() != molecule.n_atoms() {
        return Err(EngineError::Internal(format!(
            "one_electron_with_charges: {} charges for {} atoms",
            charges.len(),
            molecule.n_atoms()
        )));
    }
    // 每个壳层一个变换（l<=1 是单位阵；l=2 是 6x5；l>=3 未实现 → 显式报错）。
    let mut transforms = Vec::with_capacity(basis.shells().len());
    for shell in basis.shells() {
        let transform = qwb_basis::spherical::spherical_transform(shell.l, shell.angular)
            .ok_or_else(|| {
                EngineError::Unsupported(format!(
                    "spherical transformation for l={} is not implemented (R-06; \
                     first phase covers s/p/d)",
                    shell.l
                ))
            })?;
        transforms.push(transform);
    }
    // 直角坐标索引空间（d 是 6 个），球谐空间是 basis.n_functions()。
    let mut cart_offsets = Vec::with_capacity(transforms.len());
    let mut n_cartesian = 0usize;
    for transform in &transforms {
        cart_offsets.push(n_cartesian);
        n_cartesian += transform.n_cartesian;
    }
    let n = n_cartesian;
    let mut overlap = Matrix::zeros(n, n);
    let mut kinetic = Matrix::zeros(n, n);
    let mut nuclear = Matrix::zeros(n, n);

    // 核电荷：全电子用元素表的 Z；有 ECP 时由调用方传入 z_eff（I-05）。
    let mut nuclei: Vec<([f64; 3], f64)> = Vec::with_capacity(molecule.n_atoms());
    for (atom, charge) in molecule.atoms.iter().zip(charges) {
        nuclei.push((atom.position_bohr(), *charge));
    }

    let shells: Vec<ShellPrimitives> = basis.shells().iter().map(shell_primitives).collect();

    for (shell_a_index, shell_a) in shells.iter().enumerate() {
        for (shell_b_index, shell_b) in shells.iter().enumerate().skip(shell_a_index) {
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

                    let row = cart_offsets[shell_a_index] + component_a;
                    let column = cart_offsets[shell_b_index] + component_b;
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

    // 组装块对角的 T_g（直角坐标 × 球谐），然后 M_sph = T_gᵀ M_cart T_g。
    let n_spherical = basis.n_functions();
    let mut global_transform = Matrix::zeros(n_cartesian, n_spherical);
    for (shell_index, shell) in basis.shells().iter().enumerate() {
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

    let overlap = transform_matrix(&overlap, &global_transform)?;
    let kinetic = transform_matrix(&kinetic, &global_transform)?;
    let nuclear = transform_matrix(&nuclear, &global_transform)?;

    let mut core_hamiltonian = Matrix::zeros(n_spherical, n_spherical);
    for i in 0..n_spherical {
        for j in 0..n_spherical {
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

/// `Tᵀ M T`（`T` 是 `n_cart × n_sph`）。
fn transform_matrix(m: &Matrix, transform: &Matrix) -> Result<Matrix, EngineError> {
    let transposed = transform.transpose();
    let left = qwb_linalg::matmul(&transposed, m)
        .map_err(|err| EngineError::Internal(format!("transform T^T M: {err}")))?;
    qwb_linalg::matmul(&left, transform)
        .map_err(|err| EngineError::Internal(format!("transform (T^T M) T: {err}")))
}
