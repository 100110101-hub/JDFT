//! GIAO 抗磁（dia）项缺的两块积分：libcint 的 `int1e_giao_a11part` 与 `int1e_a01gp`。
//!
//! RS-04 的 GIAO 层需要 `pyscf.prop.nmr.rhf.giao_dia` 里那两个 9 分量张量积分。两块都是
//! 「**多项式 / r_A³**」的形状，而本 crate 只有 1/r 的原函数核（[`primitive_nuclear`]），
//! 所以这里沿用 [`crate::angmom::hyperfine_integrals`] 的绕法：把
//! `(r−R_A)_γ/r_A³ = −∂_γ(1/r_A)` 换成一次导数，再分部积分把导数挪到基函数上 ——
//! 结果全是「导数基函数 × 导数基函数」的普通 1/r 积分。**不引入新的特殊函数。**
//!
//! # 两个算子的确切定义（都是**测量**出来的，不是从文献抄的）
//!
//! 参考矩阵由 `scripts/pyscf_giao_dia_reference.py` 用 PySCF 2.14.0 的 libcint 逐元素导出
//! （水，STO-3G 与 LANL2DZ，三个核，9 个 (α,β) 块）；两个定义都经过元素级拟合确认，
//! 残差见 crate 测试 `rs04_giao_dia.rs` 与生成脚本的 `--check`。def2-SVP 也测过（按球谐 d 的
//! AO 重排后差 8.7e-15…1.8e-14），只是没进参考文件（见生成脚本里的说明）。
//!
//! 1. [`giao_a11_integrals`]（libcint `int1e_giao_a11part`）：
//!
//!    ```text
//!    A_αβ[μν] = −½ ⟨μ| (r−R_A)_α (r−R_ν)_β / |r−R_A|³ |ν⟩
//!    ```
//!
//!    第二因子是 **(r−R_ν)**（**ket** 基函数的中心），不是 `(r−R_A)`：这是 libcint 的
//!    定义，也是它 α↔β 不对称的来源（水/STO-3G 上 `max|A_αβ − A_βα| = 1.784790e-1`，
//!    而 `max|A| = 1.262885`）。O1s–O1s 元素是 `−1.262884566 = −½·(1/3)⟨1/r_A⟩`。
//!    **本函数含那个 `−½`**，与 `int1e_giao_a11part` 逐元素同号同值。
//!
//! 2. [`giao_a01_integrals`]（libcint `int1e_a01gp`）：
//!
//!    ```text
//!    B_αβ[μν] = −½ Σ_{jl} Σ_{γδ} ε_{αjl} ε_{βγδ} (R_ν−R_μ)_j
//!                        ⟨μ| r_l · ∂_δ[ (r−R_A)_γ / |r−R_A|³ ] · φ_ν ⟩
//!             = −½ ⟨μ| [ (R_ν−R_μ) × r ]_α · ((r−R_A) × ∇)_β / |r−R_A|³ |ν⟩
//!    ```
//!
//!    也就是「**相位导数 × 超精细算符**」：`[(R_ν−R_μ) × r]_α` 是 GIAO 相位对磁场的
//!    导数那一类因子，`((r−R_A)×∇)_β/r_A³` 是核自旋的超精细算符（`∇` 只作用在 ket 上）。
//!    **本函数含那个 `−½`。**
//!
//!    两件事必须写下来：
//!
//!    * `r` 是**绝对**电子坐标（不是 `r−R_A`）：所以 `B` 对整体平移**不**不变 ——
//!      水/STO-3G 上把分子平移到核在原点，`max|a01gp(平移前) − a01gp(平移后)| = 4.147e-2`
//!      （`max|a01gp| = 1.806e-1`）。这不是 bug：GIAO 的 dia/para **分解**依赖绝对原点，
//!      只有两者之和才规范不变（`scripts/pyscf_giao_nmr_reference.py` 顶部也记了这一条），
//!      所以这一块只能按 libcint 的绝对坐标约定复现。`A`（a11）则对平移不变
//!      （同一测量里差值 2.3e-16）， потому它不含绝对 `r`。
//!    * `(R_ν−R_μ)` 是 **ket 中心减 bra 中心**（列减行），于是同一中心上的矩阵元为零：
//!      对角元 `B_αβ[μμ] = 0` 精确成立（测试钉住这一条）。
//!
//! # 为什么不需要 `1/r³` 的新积分
//!
//! `(r−R_A)_γ/r_A³ = −∂_γ(1/r_A)`，分部积分：
//!
//! ```text
//! ⟨f| ∂_γ(1/r_A) |g⟩ = −⟨∂_γ f| 1/r_A |g⟩ − ⟨f| 1/r_A |∂_γ g⟩
//! ```
//!
//! 于是 a11 化成
//!
//! ```text
//! ⟨μ|(r−R_A)_α (r−R_ν)_β/r_A³|ν⟩
//!   = ⟨∂_αμ| (r−R_ν)_β/r_A |ν⟩ + δ_αβ⟨μ|1/r_A|ν⟩ + ⟨μ|(r−R_ν)_β/r_A|∂_αν⟩
//! ```
//!
//! （第二、三项来自 `∂_α[(x−R_ν)_β φ_ν]`），a01 化成
//!
//! ```text
//! Q_{lγδ} = ⟨μ| r_l (r−R_A)_γ ∂_δ/r_A³|ν⟩            （r_l = (x−A)_l + A_l 是**绝对**坐标）
//!         = ⟨∂_γ[μ r_l]| 1/r_A |∂_δν⟩ + ⟨μ r_l| 1/r_A |∂_γ∂_δν⟩
//! ```
//!
//! 每一项都是 [`primitive_nuclear`] 的「抬高幂次 + 一次/两次导数」，与
//! `hyperfine_integrals` 用的是同一套 1/r 核（那边的 `1/r³` 也是这么绕的）。
//!
//! # 归一化约定（与 `momentum` / `giao` 的磁矩那一节一致）
//!
//! 抬高幂次时乘的仍是**原分量**的 `N_a`：`x_l [N_a χ_a] = N_a χ_{a+e_l}`，
//! 抬高的那一项**不**再乘 `N_{a+e_l}`。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::dipole::transform_matrix;
use crate::one_electron::{primitive_norm, primitive_nuclear};

/// 把某轴幂次加一。
fn raise(powers: [u32; 3], axis: usize) -> [u32; 3] {
    let mut out = powers;
    out[axis] += 1;
    out
}

/// `∂_axis` 作用在一个分量上的两项：`(幂次, 权重)`。
///
/// `∂_x (x−A)^p e^{−α(x−A)²} = p (x−A)^{p−1} e^{…} − 2α (x−A)^{p+1} e^{…}`
fn derivative_terms(powers: [u32; 3], axis: usize, exponent: f64) -> Vec<([u32; 3], f64)> {
    let mut terms = Vec::with_capacity(2);
    if powers[axis] > 0 {
        let mut lowered = powers;
        lowered[axis] -= 1;
        terms.push((lowered, f64::from(powers[axis])));
    }
    terms.push((raise(powers, axis), -2.0 * exponent));
    terms
}

/// 纯 `⟨a|1/|r−C||b⟩`（**未归一化**原函数，不含收缩系数与 `N`）。
///
/// [`primitive_nuclear`] 的返回值带吸引符号（负），取负得纯 1/r 积分。
fn inv_r(
    alpha: f64,
    powers_a: [u32; 3],
    a: [f64; 3],
    beta: f64,
    powers_b: [u32; 3],
    b: [f64; 3],
    nucleus: [f64; 3],
) -> f64 {
    -primitive_nuclear(
        alpha,
        (powers_a[0], powers_a[1], powers_a[2]),
        a,
        beta,
        (powers_b[0], powers_b[1], powers_b[2]),
        b,
        nucleus,
        1.0,
    )
}

/// `ε_{abc}` 里**非零**的那两项：`[(b, c, +1), (c, b, −1)]`。
///
/// `ε_{abc} = +1` 当 `(b, c)` 是 `(a+1, a+2)` 的循环排列（模 3）。这样写出来就没有
/// 「跳过 0 项」的循环，并且 3 个 `a` 一共只有 6 个 `(b, c)` —— 正好是 9 个块里真正用到的那几对。
fn levi_civita_pairs(a: usize) -> [(usize, usize, f64); 2] {
    let plus = (a + 1) % 3;
    let minus = (a + 2) % 3;
    [(plus, minus, 1.0), (minus, plus, -1.0)]
}

/// 逐原函数对的核：给定两套（幂次、中心、指数）与 `R_μ`、`R_ν`、`R_A`，
/// 返回 9 个 `(α,β)` 块的原函数值（**未归一化**，不含收缩系数与 `N`）。
type DiaKernel<'k> = dyn FnMut(
        [u32; 3],
        [f64; 3],
        f64,
        [u32; 3],
        [f64; 3],
        f64,
        [f64; 3],
        [f64; 3],
        [f64; 3],
    ) -> [[f64; 3]; 3]
    + 'k;

/// 按「分量对 → 原函数对 → 球谐变换」组装 9 个块（与 `angmom::hyperfine_integrals` 同一套绕法）。
fn assemble(
    basis: &InternalBasis,
    nucleus: [f64; 3],
    kernel: &mut DiaKernel<'_>,
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
    let mut cartesian: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian)));

    let shells: Vec<_> = basis.shells().iter().collect();
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let components_a = qwb_basis::cartesian_components(shell_a.l);
            let components_b = qwb_basis::cartesian_components(shell_b.l);
            for (ca, powers_a) in components_a.iter().enumerate() {
                for (cb, powers_b) in components_b.iter().enumerate() {
                    let pa = [powers_a.0, powers_a.1, powers_a.2];
                    let pb = [powers_b.0, powers_b.1, powers_b.2];
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
                                pa,
                                shell_a.center_bohr,
                                *alpha,
                                pb,
                                shell_b.center_bohr,
                                *beta,
                                shell_a.center_bohr,
                                shell_b.center_bohr,
                                nucleus,
                            );
                            for (first, block) in primitive.iter().enumerate() {
                                for (second, value) in block.iter().enumerate() {
                                    values[first][second] += weight * value;
                                }
                            }
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for (first, block) in values.iter().enumerate() {
                        for (second, value) in block.iter().enumerate() {
                            cartesian[first][second][(row, column)] = *value;
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
    for first in 0..3 {
        for second in 0..3 {
            out[first][second] = transform_matrix(&cartesian[first][second], &global_transform)?;
        }
    }
    Ok(out)
}

/// 9 个块 `A_αβ[μν] = **−½** ⟨μ| (r−R_A)_α (r−R_ν)_β / |r−R_A|³ |ν⟩`（原子单位，球谐基）。
///
/// 与 libcint 的 `int1e_giao_a11part`（`comp=9`，`R_A` 用 `with_rinv_origin` 指定）逐元素一致；
/// 定义与那两处「要留意」的地方见模块文档：第二因子是 **(r−R_ν)**（ket 中心），
/// 且 `−½` **含在返回值里**。
///
/// # Errors
///
/// 球谐变换未实现（`l > 2`）或维度不一致。
pub fn giao_a11_integrals(
    basis: &InternalBasis,
    _molecule: &Molecule,
    nucleus_bohr: [f64; 3],
) -> Result<[[Matrix; 3]; 3], EngineError> {
    assemble(
        basis,
        nucleus_bohr,
        &mut |pa, a, alpha, pb, b, beta, _cmu, _cnu, nucleus| {
            // `(r−R_ν)_β` 在 ket 侧：R_ν 就是 ket 壳层中心，所以只是把 β 轴幂次抬高。
            // 分部积分后是两项：
            //   ⟨∂_αμ| (r−R_ν)_β/r |ν⟩
            // + ⟨μ| 1/r |∂_α[(r−R_ν)_β φ_ν]⟩，而后一项里
            //   `∂_α[(x−R_ν)_β φ_ν] = δ_αβ φ_ν + (x−R_ν)_β ∂_αφ_ν`。
            // 那个 `δ_αβ` 由 `derivative_terms` 在**抬高后**的幂次上自动给出，
            // 所以**不要**再单独加一次 `⟨μ|1/r|ν⟩`（加了会正好多出 −½⟨1/r⟩ 的偏差）。
            let mut out = [[0.0_f64; 3]; 3];
            for (alpha_axis, block) in out.iter_mut().enumerate() {
                for (beta_axis, value) in block.iter_mut().enumerate() {
                    let raised = raise(pb, beta_axis);
                    let mut total = 0.0_f64;
                    // ⟨∂_α μ| (r−R_ν)_β/r |ν⟩
                    for (shifted, weight) in derivative_terms(pa, alpha_axis, alpha) {
                        total += weight * inv_r(alpha, shifted, a, beta, raised, b, nucleus);
                    }
                    // ⟨μ| 1/r |∂_α[(r−R_ν)_β φ_ν]⟩（导数打在**抬高后**的 ket 上）
                    for (shifted, weight) in derivative_terms(raised, alpha_axis, beta) {
                        total += weight * inv_r(alpha, pa, a, beta, shifted, b, nucleus);
                    }
                    *value = -0.5 * total;
                }
            }
            out
        },
    )
}

/// 单个「抬高后的 bra」上的归约：
///
/// ```text
/// R_{γδ}[bra] = ⟨bra| (r−R_A)_γ ∂_δ / r_A³ |ν⟩
///              = ⟨∂_γ bra| 1/r_A |∂_δν⟩ + ⟨bra| 1/r_A |∂_γ∂_δν⟩
/// ```
///
/// （`(r−R_A)_γ/r_A³ = −∂_γ(1/r_A)` 加一次分部积分。）
fn reduced_block(
    bra: [u32; 3],
    a: [f64; 3],
    alpha: f64,
    pb: [u32; 3],
    b: [f64; 3],
    beta: f64,
    nucleus: [f64; 3],
) -> [[f64; 3]; 3] {
    // 第一轴是 `γ`、第二轴是 `δ`（与调用方 `q[l][γ][δ]` 的布局一致）。
    let ket_first: Vec<Vec<([u32; 3], f64)>> = (0..3)
        .map(|derivative| derivative_terms(pb, derivative, beta))
        .collect();
    let mut out = [[0.0_f64; 3]; 3];
    for (gamma, row) in out.iter_mut().enumerate() {
        for (derivative, value) in row.iter_mut().enumerate() {
            let ket_terms = &ket_first[derivative];
            let mut total = 0.0_f64;
            // 第一项：∂_γ 打在 bra 上。
            for (bra_powers, bra_weight) in derivative_terms(bra, gamma, alpha) {
                for (ket_powers, ket_weight) in ket_terms {
                    total += bra_weight
                        * ket_weight
                        * inv_r(alpha, bra_powers, a, beta, *ket_powers, b, nucleus);
                }
            }
            // 第二项：∂_γ 接在 ∂_δ 后面，仍只作用在 ket 上。
            for (ket_powers, ket_weight) in ket_terms {
                for (second_powers, second_weight) in derivative_terms(*ket_powers, gamma, beta) {
                    total += ket_weight
                        * second_weight
                        * inv_r(alpha, bra, a, beta, second_powers, b, nucleus);
                }
            }
            *value = total;
        }
    }
    out
}

/// 9 个块 `B_αβ[μν] = **−½** ⟨μ| [(R_ν−R_μ) × r]_α · ((r−R_A) × ∇)_β / |r−R_A|³ |ν⟩`
/// （原子单位，球谐基）。
///
/// 与 libcint 的 `int1e_a01gp`（`comp=9`）逐元素一致。`r` 是**绝对**电子坐标，`∇` 只作用在 ket
/// 上，`R_μ`/`R_ν` 分别是 bra/ket 壳层中心 —— 所以这个矩阵对整体平移**不**不变（见模块文档里
/// 的测量数字）。`−½` **含在返回值里**。
///
/// # Errors
///
/// 球谐变换未实现（`l > 2`）或维度不一致。
pub fn giao_a01_integrals(
    basis: &InternalBasis,
    _molecule: &Molecule,
    nucleus_bohr: [f64; 3],
) -> Result<[[Matrix; 3]; 3], EngineError> {
    assemble(
        basis,
        nucleus_bohr,
        &mut |pa, a, alpha, pb, b, beta, cmu, cnu, nucleus| {
            // Q_{lγδ} = ⟨μ| r_l (r−R_A)_γ ∂_δ / r_A³ |ν⟩
            //         = ⟨∂_γ[μ r_l]|1/r|∂_δν⟩ + ⟨μ r_l|1/r|∂_γ∂_δν⟩
            // `r_l = (x−A)_l + A_l`：前一半是「抬高 bra 幂次」，后一半是常数因子。绝对坐标 `r_l`
            // 是 libcint 的约定（这就是这一块对平移不不变的原因），不能换成 `(r−R_A)_l`。
            let plain = reduced_block(pa, a, alpha, pb, b, beta, nucleus);
            let mut q = [[[0.0_f64; 3]; 3]; 3];
            for (moment, plane) in q.iter_mut().enumerate() {
                let raised = reduced_block(raise(pa, moment), a, alpha, pb, b, beta, nucleus);
                let shift = a[moment];
                for (gamma, row) in plane.iter_mut().enumerate() {
                    for (derivative, value) in row.iter_mut().enumerate() {
                        *value = raised[gamma][derivative] + shift * plain[gamma][derivative];
                    }
                }
            }
            let delta_r = [cnu[0] - cmu[0], cnu[1] - cmu[1], cnu[2] - cmu[2]];
            let mut out = [[0.0_f64; 3]; 3];
            for (alpha_axis, block) in out.iter_mut().enumerate() {
                for (beta_axis, value) in block.iter_mut().enumerate() {
                    let mut total = 0.0_f64;
                    for (j, moment, sign_first) in levi_civita_pairs(alpha_axis) {
                        for (gamma, derivative, sign_second) in levi_civita_pairs(beta_axis) {
                            total += sign_first
                                * delta_r[j]
                                * sign_second
                                * q[moment][gamma][derivative];
                        }
                    }
                    *value = -0.5 * total;
                }
            }
            out
        },
    )
}
