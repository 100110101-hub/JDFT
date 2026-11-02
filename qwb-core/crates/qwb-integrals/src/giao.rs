//! GIAO（London 轨道）的**第一个积分**：重叠矩阵对磁场的导数（`RS-04` 的 GIAO 层）。
//!
//! # 公式与为什么它不需要新积分
//!
//! London 轨道给每个基函数乘一个依赖磁场 `B` 与规范原点 `O` 的相位：
//!
//! ```text
//! χ_μ(B) = exp(−(i/2) B·[(R_μ − O) × r]) χ_μ(r)
//! ```
//!
//! 对 `B` 求导并取 `B = 0`：
//!
//! ```text
//! ∂S_μν/∂B_k = (i/2) Σ_{jl} ε_{kjl} (R_μ − R_ν)_j ⟨μ| r_l |ν⟩
//! ```
//!
//! 两件事值得写下来：
//!
//! 1. **规范原点 `O` 精确抵消**（差值里只出现 `R_μ − R_ν`）。这不是近似，是 London 相位
//!    的共同原点性质 —— 所以这个函数**不需要** `O` 参数，`origin` 传什么都一样，这一点由测试钉住。
//! 2. 它只需要**普通的偶极积分** `⟨μ|r|ν⟩`（`dipole::dipole_integrals` 已有），不需要新的
//!    积分递推。这也是 GIAO 层的入口比看起来便宜的原因。
//!
//! 返回的矩阵 `C_k` 约定成 `∂S/∂B_k = (i/2) C_k`（`C_k` 是**实**矩阵），因为复数不该泄漏到
//! 上游的实矩阵接口里 —— 相位带来的虚部由调用方按这个约定乘进去。
//!
//! **还差的（GIAO 要完整落地还需要）**：`∂H/∂B`、GIAO 的电子排斥导数、拉曼/抗磁的 GIAO 项，
//! 以及 CPHF 右端的改写。这一块只交付**重叠导数**，因为它是唯一能用"复相位的有限差分"独立
//! 验证的一步 —— 其余几块没有独立的尺子之前不写（见 `docs/status-v2.yaml` 的 RS-04）。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;
use qwb_registry::elements::atomic_number;

use crate::dipole::{dipole_integrals, transform_matrix};
use crate::momentum::{power_along, with_power};
use crate::one_electron::{primitive_kinetic, primitive_norm, primitive_nuclear};

/// 球谐（或笛卡尔）基下的 `∂S/∂B_k = (i/2) C_k` 的三个实矩阵。
///
/// # Errors
///
/// 基组里出现 `l` 大于已实现的球谐变换、或矩阵维度不一致。
pub fn giao_overlap_derivative(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
) -> Result<[Matrix; 3], EngineError> {
    let dipole = dipole_integrals(basis, molecule)?;
    let n = dipole[0].nrows();
    for component in &dipole {
        if component.nrows() != n || component.ncols() != n {
            return Err(EngineError::Internal(
                "giao: dipole matrices have inconsistent dimensions".to_string(),
            ));
        }
    }

    // 每个基函数的中心（同一壳层的所有函数共享中心），用于 `R_μ − R_ν`。
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n);
    for shell in basis.shells() {
        for _ in 0..shell.n_functions() {
            centers.push(shell.center_bohr);
        }
    }
    if centers.len() != n {
        return Err(EngineError::Internal(format!(
            "giao: {} basis functions but {} centers",
            n,
            centers.len()
        )));
    }

    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    for mu in 0..n {
        for nu in 0..n {
            let delta = [
                centers[mu][0] - centers[nu][0],
                centers[mu][1] - centers[nu][1],
                centers[mu][2] - centers[nu][2],
            ];
            let d = [
                dipole[0].get(mu, nu),
                dipole[1].get(mu, nu),
                dipole[2].get(mu, nu),
            ];
            // (Δ × d)_k
            let cross = [
                delta[1] * d[2] - delta[2] * d[1],
                delta[2] * d[0] - delta[0] * d[2],
                delta[0] * d[1] - delta[1] * d[0],
            ];
            for (k, value) in cross.iter().enumerate() {
                out[k][(mu, nu)] = *value;
            }
        }
    }
    Ok(out)
}

/// 单个壳层对的重叠导数（**笛卡尔**分量级，`C_k` 同上）。
///
/// 存在的理由是测试：验证用「复相位重叠的有限差分」做尺子，而那个尺子只能在**笛卡尔**
/// 分量级上写（球谐变换会把不同分量混起来，测试就不再独立了）。生产路径用
/// [`giao_overlap_derivative`]。
///
/// # Errors
///
/// 指数/系数长度不一致，或维度越界。
pub fn giao_overlap_derivative_cartesian(
    shell_a: &qwb_basis::InternalShell,
    shell_b: &qwb_basis::InternalShell,
) -> Result<[Vec<f64>; 3], EngineError> {
    let components_a = qwb_basis::cartesian_components(shell_a.l);
    let components_b = qwb_basis::cartesian_components(shell_b.l);
    let rows = components_a.len();
    let columns = components_b.len();
    let mut out = [
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
    ];
    let delta = [
        shell_a.center_bohr[0] - shell_b.center_bohr[0],
        shell_a.center_bohr[1] - shell_b.center_bohr[1],
        shell_a.center_bohr[2] - shell_b.center_bohr[2],
    ];
    for (ca, powers_a) in components_a.iter().enumerate() {
        for (cb, powers_b) in components_b.iter().enumerate() {
            let mut d = [0.0_f64; 3];
            for (alpha, coefficient_a) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                for (beta, coefficient_b) in shell_b.exponents.iter().zip(&shell_b.coefficients) {
                    let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                    let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                    let p = alpha + beta;
                    let product = [
                        (alpha * shell_a.center_bohr[0] + beta * shell_b.center_bohr[0]) / p,
                        (alpha * shell_a.center_bohr[1] + beta * shell_b.center_bohr[1]) / p,
                        (alpha * shell_a.center_bohr[2] + beta * shell_b.center_bohr[2]) / p,
                    ];
                    let overlap = crate::one_electron::primitive_overlap(
                        *alpha,
                        *powers_a,
                        shell_a.center_bohr,
                        *beta,
                        *powers_b,
                        shell_b.center_bohr,
                    );
                    for axis in 0..3 {
                        d[axis] += weight * overlap * product[axis];
                    }
                }
            }
            let cross = [
                delta[1] * d[2] - delta[2] * d[1],
                delta[2] * d[0] - delta[0] * d[2],
                delta[0] * d[1] - delta[1] * d[0],
            ];
            for (k, value) in cross.iter().enumerate() {
                out[k][ca * columns + cb] = *value;
            }
        }
    }
    Ok(out)
}

// ===== 磁矩积分 `⟨μ| r_l W |ν⟩`（`W = T` 或核吸引 `V`；`∂H_core/∂B` 的其余两项）=====
//
// 推导（不需要任何新的高斯递推）：
//
// `r_l` 是实值乘性函数，所以对厄米算符 `W` 有 `⟨μ| r_l W |ν⟩ = ⟨r_l μ| W |ν⟩`。
// 而乘一个直角坐标分量是初等的：
//
// ```text
// x_l · χ_a = χ_{a+e_l} + A_l · χ_a
// ```
//
// （同指数、同中心、同收缩系数；`a` 是笛卡尔幂次向量、`A` 是该函数中心）。于是每个磁矩
// 矩阵都是「bra 抬高一个幂次的普通积分」+ `A_l` ×「普通积分」——正好复用
// `primitive_kinetic` / `primitive_nuclear`（它们对任意幂次都成立）与 `momentum` 里
// 那几个抬幂次的小工具。
//
// 两个约定和 `dipole` 完全一致：
//
// 1. 归一化用**原分量**的 `N_a`。`x_l [N_a χ_a] = N_a (χ_{a+e_l} + A_l χ_a)`，抬高的那一项
//    **不**乘 `N_{a+e_l}`（这是 `momentum` 第 399 轮踩过的坑：多做一次归一化比值）。
// 2. 抬高的函数幂次比壳层高 1，但存储下标仍是**原分量**；球谐变换作用在原壳层的直角分量上。

/// 逐原函数对的核函数：给定两套（幂次、指数、中心），返回三个 `l` 的
/// `⟨x_l φ_a | W | φ_b⟩`（**未归一化**原函数，不含收缩系数与 `N`）。
type MomentKernel<'k> =
    dyn FnMut((u32, u32, u32), f64, [f64; 3], (u32, u32, u32), f64, [f64; 3]) -> [f64; 3] + 'k;

/// 三个矩阵 `A[l][μ,ν] = ⟨μ| r_l T |ν⟩`（`T = −∇²/2`，原子单位）。
///
/// 用法与 [`crate::momentum::momentum_integrals`] 一致：只要基组。
///
/// 尺子见测试模块 `moment_tests`：交换子恒等式 `A_l − A_lᵀ = p_l`（`p_l` 是
/// [`crate::momentum::momentum_integrals`]）+ 一条独立的解析矩路径。
///
/// # Errors
///
/// 球谐变换不可用（`l` 过大）。
pub fn kinetic_moment_integrals(basis: &InternalBasis) -> Result<[Matrix; 3], EngineError> {
    assemble_moments(basis, &mut |powers_a, alpha, a, powers_b, beta, b| {
        let plain = primitive_kinetic(alpha, powers_a, a, beta, powers_b, b);
        let mut values = [0.0_f64; 3];
        for (axis, value) in values.iter_mut().enumerate() {
            let raised = with_power(powers_a, axis, power_along(powers_a, axis) + 1);
            *value = primitive_kinetic(alpha, raised, a, beta, powers_b, b) + a[axis] * plain;
        }
        values
    })
}

/// 三个矩阵 `A[l][μ,ν] = ⟨μ| r_l V |ν⟩`（`V = Σ_A −Z_A/|r−R_A|`，核电荷取元素表的 `Z`）。
///
/// # Errors
///
/// 分子里出现元素表之外的符号 → [`EngineError::Internal`]；球谐变换不可用 →
/// [`EngineError::Unsupported`]。
pub fn nuclear_moment_integrals(
    basis: &InternalBasis,
    molecule: &Molecule,
) -> Result<[Matrix; 3], EngineError> {
    let mut charges = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        let z = atomic_number(&atom.symbol).ok_or_else(|| {
            EngineError::Internal(format!(
                "unknown element `{}` while building nuclear moment integrals",
                atom.symbol
            ))
        })?;
        charges.push(f64::from(z));
    }
    nuclear_moment_integrals_with_charges(basis, molecule, &charges)
}

/// 与 [`nuclear_moment_integrals`] 相同，但核电荷由调用方给（ECP 的 `z_eff`，与
/// [`crate::one_electron::one_electron_with_charges`] 同一约定）。
///
/// # Errors
///
/// `charges` 长度与原子数不符 → [`EngineError::Internal`]；球谐变换不可用 →
/// [`EngineError::Unsupported`]。
pub fn nuclear_moment_integrals_with_charges(
    basis: &InternalBasis,
    molecule: &Molecule,
    charges: &[f64],
) -> Result<[Matrix; 3], EngineError> {
    if charges.len() != molecule.n_atoms() {
        return Err(EngineError::Internal(format!(
            "nuclear_moment_integrals_with_charges: {} charges for {} atoms",
            charges.len(),
            molecule.n_atoms()
        )));
    }
    let nuclei: Vec<([f64; 3], f64)> = molecule
        .atoms
        .iter()
        .zip(charges)
        .map(|(atom, charge)| (atom.position_bohr(), *charge))
        .collect();
    assemble_moments(basis, &mut |powers_a, alpha, a, powers_b, beta, b| {
        let mut values = [0.0_f64; 3];
        for (axis, value) in values.iter_mut().enumerate() {
            let raised = with_power(powers_a, axis, power_along(powers_a, axis) + 1);
            let mut total = 0.0_f64;
            for (nucleus_center, charge) in &nuclei {
                total += primitive_nuclear(
                    alpha,
                    raised,
                    a,
                    beta,
                    powers_b,
                    b,
                    *nucleus_center,
                    *charge,
                ) + a[axis]
                    * primitive_nuclear(
                        alpha,
                        powers_a,
                        a,
                        beta,
                        powers_b,
                        b,
                        *nucleus_center,
                        *charge,
                    );
            }
            *value = total;
        }
        values
    })
}

/// 按「分量对 → 原函数对 → 球谐变换」组装三个 `l` 的矩阵（与
/// [`crate::dipole::dipole_integrals`] 同一套绕法）。
fn assemble_moments(
    basis: &InternalBasis,
    acting: &mut MomentKernel<'_>,
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
    let mut moments: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian));

    let shells: Vec<_> = basis.shells().iter().collect();
    for (ia, shell_a) in shells.iter().enumerate() {
        for (ib, shell_b) in shells.iter().enumerate() {
            let components_a = qwb_basis::cartesian_components(shell_a.l);
            let components_b = qwb_basis::cartesian_components(shell_b.l);
            for (ca, powers_a) in components_a.iter().enumerate() {
                for (cb, powers_b) in components_b.iter().enumerate() {
                    let mut values = [0.0_f64; 3];
                    for (alpha, coefficient_a) in
                        shell_a.exponents.iter().zip(&shell_a.coefficients)
                    {
                        // 抬高项用的是**原分量**的归一化常数（见本节顶部约定 1）。
                        let norm_a = primitive_norm(*powers_a, *alpha);
                        for (beta, coefficient_b) in
                            shell_b.exponents.iter().zip(&shell_b.coefficients)
                        {
                            let norm_b = primitive_norm(*powers_b, *beta);
                            let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                            let primitive = acting(
                                *powers_a,
                                *alpha,
                                shell_a.center_bohr,
                                *powers_b,
                                *beta,
                                shell_b.center_bohr,
                            );
                            for (axis, value) in values.iter_mut().enumerate() {
                                *value += weight * primitive[axis];
                            }
                        }
                    }
                    let row = cart_offsets[ia] + ca;
                    let column = cart_offsets[ib] + cb;
                    for (axis, moment) in moments.iter_mut().enumerate() {
                        moment[(row, column)] = values[axis];
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
    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n_spherical, n_spherical));
    for (axis, moment) in moments.iter().enumerate() {
        out[axis] = transform_matrix(moment, &global_transform)?;
    }
    Ok(out)
}

/// `[e_k × a]_l = ε_{lkm} a_m`（与 [`crate::core_hamiltonian`] 同一个叉积符号）。
fn cross_unit(a: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
}

/// 单电子核心哈密顿量对磁场的 GIAO 导数：三个**实**矩阵 `Y_k`，`∂H/∂B_k = i Y_k`。
///
/// # 公式与推导（与 [`crate::core_hamiltonian::core_hamiltonian_field_derivative`] 的区别）
///
/// London 轨道下 `⟨φ_μ|h|φ_ν⟩` 用**右矢**的相位做规范变换：
///
/// ```text
/// e^{iK_ν·r} h(B) e^{−iK_ν·r} = ½(p + ½B×(r−R_ν))² + V          ← 规范原点变成 R_ν
/// ```
///
/// 于是（`B = 0` 处，一次项）
///
/// ```text
/// ∂H_μν/∂B_k = (i/2) Σ_l [e_k×(R_μ−R_ν)]_l M_l[μν] + ½ ⟨μ| L_k(R_ν) |ν⟩
/// M_l = ⟨μ| r_l (T+V) |ν⟩（[`kinetic_moment_integrals`] + [`nuclear_moment_integrals`]）
/// ```
///
/// 写成实矩阵（`L_k(R) = L_k(0) − (R×p)_k`，且 `L = −iN`、`p = −i·∂`）：
///
/// ```text
/// Y_k = ½ Σ_l [e_k×(R_μ−R_ν)]_l M_l − ½ N_k(0) + ½ (e_k×R_ν)·∂
/// ```
///
/// **规范原点 `O` 不出现**（`R_μ−R_ν` 里精确抵消）。
///
/// 这不是 [`crate::core_hamiltonian::core_hamiltonian_field_derivative`]：那个函数把
/// 相位的两端都写成 `R−O`，等价于这里的 `Y_k + ½ N_k(O)` —— 差一个绕 `O` 的角动量。
/// 物理的 `∂H/∂B` 只有一个，本函数与 PySCF 的 GIAO 单电子 Fock 导数一致
/// （`crates/qwb-integrals/tests/reference/giao_fock_derivative.csv`，逐元素 ≤ 1e-11）。
///
/// # Errors
///
/// 球谐变换不可用（`l` 过大）、分子里出现元素表之外的符号。
pub fn giao_core_field_derivative(
    basis: &InternalBasis,
    molecule: &Molecule,
) -> Result<[Matrix; 3], EngineError> {
    let n = basis.n_functions();
    let kinetic = kinetic_moment_integrals(basis)?;
    let nuclear = nuclear_moment_integrals(basis, molecule)?;
    let angular = crate::angmom::angular_momentum_integrals(basis, molecule, [0.0; 3])?;
    let momentum = crate::momentum::momentum_integrals(basis)?;
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n);
    for shell in basis.shells() {
        for _ in 0..shell.n_functions() {
            centers.push(shell.center_bohr);
        }
    }
    if centers.len() != n {
        return Err(EngineError::Internal(format!(
            "giao_core_field_derivative: {n} basis functions but {} centers",
            centers.len()
        )));
    }
    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for mu in 0..n {
        for nu in 0..n {
            let delta = [
                centers[mu][0] - centers[nu][0],
                centers[mu][1] - centers[nu][1],
                centers[mu][2] - centers[nu][2],
            ];
            let cross_delta = cross_unit(delta);
            let cross_center = cross_unit(centers[nu]);
            for k in 0..3 {
                let mut value = 0.0_f64;
                for l in 0..3 {
                    let core_moment = kinetic[l].get(mu, nu) + nuclear[l].get(mu, nu);
                    value += 0.5 * cross_delta[k][l] * core_moment;
                }
                value -= 0.5 * angular[k].get(mu, nu);
                for l in 0..3 {
                    value += 0.5 * cross_center[k][l] * momentum[l].get(mu, nu);
                }
                out[k][(mu, nu)] = value;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{giao_overlap_derivative, giao_overlap_derivative_cartesian};
    use qwb_engine_api::{Atom, Molecule};
    use qwb_registry::Registry;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    fn water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.0]),
                Atom::new("H", [0.0, 0.757, 0.586]),
                Atom::new("H", [0.0, -0.757, 0.586]),
            ],
            0,
            1,
        )
    }

    /// 复相位重叠的**精确**值：`S^GIAO(B)_μν = Σ w S^prims exp(i K·P − K²/(4p))`，
    /// 其中 `K = (1/2) B × (R_μ − R_ν)`、`P = (αR_μ + βR_ν)/p`。
    fn giao_overlap_cartesian(
        shell_a: &qwb_basis::InternalShell,
        shell_b: &qwb_basis::InternalShell,
        field: [f64; 3],
    ) -> (Vec<f64>, Vec<f64>) {
        let components_a = qwb_basis::cartesian_components(shell_a.l);
        let components_b = qwb_basis::cartesian_components(shell_b.l);
        let rows = components_a.len();
        let columns = components_b.len();
        let delta = [
            shell_a.center_bohr[0] - shell_b.center_bohr[0],
            shell_a.center_bohr[1] - shell_b.center_bohr[1],
            shell_a.center_bohr[2] - shell_b.center_bohr[2],
        ];
        let k = [
            0.5 * (field[1] * delta[2] - field[2] * delta[1]),
            0.5 * (field[2] * delta[0] - field[0] * delta[2]),
            0.5 * (field[0] * delta[1] - field[1] * delta[0]),
        ];
        let mut real = vec![0.0_f64; rows * columns];
        let mut imaginary = vec![0.0_f64; rows * columns];
        for (ca, powers_a) in components_a.iter().enumerate() {
            for (cb, powers_b) in components_b.iter().enumerate() {
                let mut sum_re = 0.0_f64;
                let mut sum_im = 0.0_f64;
                for (alpha, coefficient_a) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                    let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                    for (beta, coefficient_b) in shell_b.exponents.iter().zip(&shell_b.coefficients)
                    {
                        let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                        let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                        let p = alpha + beta;
                        let product = [
                            (alpha * shell_a.center_bohr[0] + beta * shell_b.center_bohr[0]) / p,
                            (alpha * shell_a.center_bohr[1] + beta * shell_b.center_bohr[1]) / p,
                            (alpha * shell_a.center_bohr[2] + beta * shell_b.center_bohr[2]) / p,
                        ];
                        let overlap = crate::one_electron::primitive_overlap(
                            *alpha,
                            *powers_a,
                            shell_a.center_bohr,
                            *beta,
                            *powers_b,
                            shell_b.center_bohr,
                        );
                        let phase = k[0] * product[0] + k[1] * product[1] + k[2] * product[2];
                        let damping = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) / (4.0 * p);
                        let magnitude = weight * overlap * (-damping).exp();
                        sum_re += magnitude * phase.cos();
                        sum_im += magnitude * phase.sin();
                    }
                }
                real[ca * columns + cb] = sum_re;
                imaginary[ca * columns + cb] = sum_im;
            }
        }
        (real, imaginary)
    }

    #[test]
    fn the_overlap_derivative_matches_a_finite_difference_of_the_complex_phase() {
        // 独立的尺子：直接按相位因子算 S^GIAO(B)（复矩阵），对 B 做中心差分，再与生产公式
        // （偶极积分 × (R_μ − R_ν)）比。两条路只共用 `primitive_overlap`，不需要别的假设。
        let registry = Registry::open(data_root()).expect("qwb-data");
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &water(),
        )
        .expect("basis");
        let shells = basis.shells();
        let step = 1.0e-5_f64;
        let mut worst: f64 = 0.0;
        for shell_a in shells {
            for shell_b in shells {
                let coefficients = giao_overlap_derivative_cartesian(shell_a, shell_b).expect("C");
                for axis in 0..3 {
                    let mut field_plus = [0.0_f64; 3];
                    let mut field_minus = [0.0_f64; 3];
                    field_plus[axis] = step;
                    field_minus[axis] = -step;
                    let (_, plus) = giao_overlap_cartesian(shell_a, shell_b, field_plus);
                    let (_, minus) = giao_overlap_cartesian(shell_a, shell_b, field_minus);
                    for index in 0..plus.len() {
                        // ∂S/∂B = (i/2) C  ⇒  Im(∂S/∂B) = C/2 ⇒ C = 2·dIm/dB
                        let numeric = (plus[index] - minus[index]) / (2.0 * step);
                        let expected = 0.5 * coefficients[axis][index];
                        worst = worst.max((numeric - expected).abs());
                    }
                }
            }
        }
        println!("GIAO 重叠导数 vs 复相位有限差分：最差 {worst:.3e}");
        assert!(
            worst < 1.0e-8,
            "GIAO 重叠导数与复相位差分差 {worst:.3e}（判据 1e-8）"
        );
    }

    #[test]
    fn the_production_matrix_is_the_cartesian_one_assembled() {
        // 生产函数（球谐/笛卡尔基上的块矩阵）与分量级公式必须一致：抽查两个矩阵元的对称性
        // 与反对称性 —— `C_k` 必须是**实的反对称**（因为 iC 是厄米的、C 实）。
        let registry = Registry::open(data_root()).expect("qwb-data");
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &water(),
        )
        .expect("basis");
        let matrices = giao_overlap_derivative(&basis, &water()).expect("C");
        let n = matrices[0].nrows();
        for (k, matrix) in matrices.iter().enumerate() {
            for mu in 0..n {
                for nu in 0..n {
                    let value = matrix.get(mu, nu);
                    assert!(value.is_finite());
                    // (R_μ − R_ν) × d 对 (μ,ν) 是反对称的（d 对称、Δ 反对称）。
                    assert!(
                        (value + matrix.get(nu, mu)).abs() < 1.0e-10,
                        "C_{k} 必须反对称：{mu},{nu} = {value:.6e}"
                    );
                }
            }
        }
    }
}

/// 磁矩积分 `⟨μ| r_l T |ν⟩`、`⟨μ| r_l V |ν⟩` 的**尺子**。
///
/// # 主尺子：交换子恒等式（先自己把符号代数推一遍，再写成测试）
///
/// `T = −∇²/2`，`[∂_l, x_l] = 1` ⇒ `[∂_l², x_l] = 2∂_l` ⇒ **`[T, x_l] = −∂_l`**。
/// 记 `A_l[μ,ν] = ⟨μ| x_l T |ν⟩`、`p_l[μ,ν] = ⟨μ| ∂_l |ν⟩`（`p_l` 就是已经验证过的
/// [`crate::momentum::momentum_integrals`]）。实基下 `T` 厄米、`x_l` 实：
///
/// ```text
/// ⟨μ| T x_l |ν⟩ = ⟨Tμ| x_l ν⟩ = ⟨x_l ν| T |μ⟩ = A_l[ν,μ]
/// ```
///
/// 于是 `A_l[μ,ν] − A_l[ν,μ] = ⟨μ| (x_l T − T x_l) |ν⟩ = −⟨μ|[T,x_l]|ν⟩ = +p_l[μ,ν]`。
///
/// 所以判据是 **`A_l − A_lᵀ = +p_l`**（注意符号：不是 `−p_l`；上面的推导给出正号）。
/// 这条恒等式**不引入任何新积分**，纯粹是两条独立实现的互相对照。
///
/// 它确实能抓住 `A_l` 那一项的错误：若 `A_l` 项算错，差值里会多出
/// `(A_l(μ) − A_l(ν))·T[μ,ν]`，只有在**不同壳层中心不同 `A_l`** 时才非零 ——
/// 这就是「所有原子中心都离开原点」的几何为什么重要。
///
/// # 次尺子
///
/// `V` 是乘性势，`[V, x_l] = 0` ⇒ `⟨μ| r_l V |ν⟩` 对 `μ↔ν` **对称**（每个 `l`）。
#[cfg(test)]
mod moment_tests {
    use super::{kinetic_moment_integrals, nuclear_moment_integrals};
    use crate::momentum::{momentum_integrals, power_along};
    use qwb_engine_api::{Atom, Molecule};
    use qwb_linalg::Matrix;
    use qwb_registry::Registry;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    /// 水，**三个原子中心全都离开原点**（`A_l` 项不会意外为零）。
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

    /// 水，氧核**正好在原点**（`A_l` 对氧壳层为零的另一种极端）。
    fn origin_water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.0]),
                Atom::new("H", [0.0, 0.757, 0.586]),
                Atom::new("H", [0.0, -0.757, 0.586]),
            ],
            0,
            1,
        )
    }

    fn basis(name: &str, molecule: &Molecule) -> qwb_basis::InternalBasis {
        let registry = Registry::open(data_root()).expect("qwb-data");
        qwb_basis::InternalBasis::build(registry.get_basis(name).expect("basis"), molecule)
            .expect("basis build")
    }

    /// `max |A[l] − A[l]ᵀ − p[l]|`、`max |A|`、`max |A − Aᵀ|`（最后一项用来说明恒等式非平凡）。
    fn commutator_deviation(moment: &[Matrix; 3], momentum: &[Matrix; 3]) -> (f64, f64, f64) {
        let n = moment[0].nrows();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut antisymmetric = 0.0_f64;
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            for mu in 0..n {
                for nu in 0..n {
                    let a = moment[axis].get(mu, nu);
                    scale = scale.max(a.abs());
                    let transposed = moment[axis].get(nu, mu);
                    antisymmetric = antisymmetric.max((a - transposed).abs());
                    worst = worst.max((a - transposed - momentum[axis].get(mu, nu)).abs());
                }
            }
        }
        (worst, scale, antisymmetric)
    }

    /// `max |M[l] − M[l]ᵀ|` 与 `max |M|`。
    fn symmetry_deviation(moment: &[Matrix; 3]) -> (f64, f64) {
        let n = moment[0].nrows();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for matrix in moment {
            for mu in 0..n {
                for nu in 0..n {
                    scale = scale.max(matrix.get(mu, nu).abs());
                    worst = worst.max((matrix.get(mu, nu) - matrix.get(nu, mu)).abs());
                }
            }
        }
        (worst, scale)
    }

    /// 对给定几何与基组跑一次交换子尺子，返回（deviation、scale、antisymmetric）。
    fn run_ruler(name: &str, molecule: &Molecule) -> (f64, f64, f64) {
        let basis = basis(name, molecule);
        let moment = kinetic_moment_integrals(&basis).expect("kinetic moment");
        let momentum = momentum_integrals(&basis).expect("momentum");
        assert_eq!(moment[0].nrows(), momentum[0].nrows());
        commutator_deviation(&moment, &momentum)
    }

    #[test]
    fn the_kinetic_moment_obeys_the_commutator_ruler() {
        // 两组基（含 d 壳层的 def2-svp）+ 两种几何（全部中心离开原点 / 氧核在原点），
        // 判据 1e-10（相对最大矩阵元）。
        let cases = [
            ("sto-3g", displaced_water()),
            ("def2-svp", displaced_water()),
            ("sto-3g", origin_water()),
            ("def2-svp", origin_water()),
        ];
        for (name, molecule) in cases {
            let (worst, scale, antisymmetric) = run_ruler(name, &molecule);
            let relative = worst / scale;
            println!(
                "{name} @ {:.2},{:.2},{:.2}: max|A−Aᵀ−p| = {worst:.3e}（max|A| = {scale:.3e}，相对 {relative:.3e}）；\
                 max|A−Aᵀ| = {antisymmetric:.3e}",
                molecule.atoms[0].position_angstrom[0],
                molecule.atoms[0].position_angstrom[1],
                molecule.atoms[0].position_angstrom[2],
            );
            assert!(worst.is_finite(), "{name}: 交换子尺子出现非有限值");
            assert!(
                relative < 1.0e-10,
                "{name}: 交换子恒等式 A−Aᵀ=p 偏差 {worst:.3e}（max|A| = {scale:.3e}，相对 {relative:.3e}，判据 1e-10）"
            );
            // 恒等式必须**非平凡**：A 的反对称部分就是 p，p 不该等于零矩阵。
            assert!(
                antisymmetric > 1.0e-6,
                "{name}: A−Aᵀ 的最大元只有 {antisymmetric:.3e}，尺子是空的"
            );
        }
    }

    #[test]
    fn the_nuclear_moment_is_symmetric() {
        // [V, x_l] = 0 ⇒ ⟨μ| r_l V |ν⟩ 对 μ↔ν 对称（每个 l）。
        for (name, molecule) in [
            ("sto-3g", displaced_water()),
            ("def2-svp", displaced_water()),
        ] {
            let basis = basis(name, &molecule);
            let moment = nuclear_moment_integrals(&basis, &molecule).expect("nuclear moment");
            let (worst, scale) = symmetry_deviation(&moment);
            let relative = worst / scale;
            println!(
                "{name} 核磁矩对称性：max|V−Vᵀ| = {worst:.3e}（max|V| = {scale:.3e}，相对 {relative:.3e}）"
            );
            assert!(
                scale > 1.0,
                "{name}: 核磁矩最大元只有 {scale:.3e}，检查是否算成了零矩阵"
            );
            assert!(
                relative < 1.0e-10,
                "{name}: 核磁矩不对称 {worst:.3e}（相对 {relative:.3e}，判据 1e-10）"
            );
        }
    }

    // ===== 第二条（**绝对**的、完全独立的）尺子：显式多项式 × 1D 解析高斯矩 =====
    //
    // 上面那条交换子恒等式只锁住 `A` 的**反对称部分**（`A − Aᵀ`）。这里再写一条绝对路径：
    // 直接把 `T = −∇²/2` 作用在 ket 上写成**显式多项式**，再逐轴用解析矩
    // `∫ y^k e^{−py²}dy = (k−1)!!/(2p)^{k/2} √(π/p)`（`k` 偶）/ 0（`k` 奇）相乘。
    // 这条路不碰 `primitive_kinetic`、`hermite_1d`、`overlap_1d`，只共用归一化常数与球谐变换。

    /// `∫ y^k e^{−p y²} dy`（`y = x − P`）。
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

    /// `(y + offset)^n` 按 `y^k` 升幂的系数。
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

    /// `target += weight · term`（长度不够就补零）。
    fn poly_add_scaled(target: &mut Vec<f64>, term: &[f64], weight: f64) {
        if weight == 0.0 {
            return;
        }
        if target.len() < term.len() {
            target.resize(term.len(), 0.0);
        }
        for (k, c) in term.iter().enumerate() {
            target[k] += weight * c;
        }
    }

    /// `∫ poly(y) e^{−p y²} dy`。
    fn poly_contract(poly: &[f64], p: f64) -> f64 {
        poly.iter()
            .enumerate()
            .map(|(k, c)| c * gaussian_moment(k, p))
            .sum()
    }

    /// `⟨x_l χ_a | T | χ_b⟩`（**未归一化**原函数）的解析值。
    fn reference_primitive(
        powers_a: (u32, u32, u32),
        a: [f64; 3],
        alpha: f64,
        powers_b: (u32, u32, u32),
        b: [f64; 3],
        beta: f64,
        axis_l: usize,
    ) -> f64 {
        let p = alpha + beta;
        let product: [f64; 3] = std::array::from_fn(|axis| (alpha * a[axis] + beta * b[axis]) / p);
        let ab2: f64 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum();
        let e_ab = (-alpha * beta / p * ab2).exp();
        let pa: [f64; 3] = std::array::from_fn(|axis| product[axis] - a[axis]);
        let pb: [f64; 3] = std::array::from_fn(|axis| product[axis] - b[axis]);

        // bra：每轴 `(y + PA)^a`；`l` 轴再乘 `x_l = y + P_l`。
        let bra = |axis: usize| -> Vec<f64> {
            let powers = shifted_powers(power_along(powers_a, axis), pa[axis]);
            if axis == axis_l {
                poly_mul(&powers, &[product[axis], 1.0])
            } else {
                powers
            }
        };
        let ket =
            |axis: usize| -> Vec<f64> { shifted_powers(power_along(powers_b, axis), pb[axis]) };
        // `−1/2 ∂²[(y+PB)^n e^{−βy²}] = −1/2 [n(n−1)(y+PB)^{n−2} − 2β(2n+1)(y+PB)^n
        //   + 4β²(y+PB)^{n+2}] e^{…}`
        let laplacian = |axis: usize| -> Vec<f64> {
            let n = power_along(powers_b, axis);
            let mut poly: Vec<f64> = vec![0.0];
            if n >= 2 {
                poly_add_scaled(
                    &mut poly,
                    &shifted_powers(n - 2, pb[axis]),
                    f64::from(n) * f64::from(n - 1),
                );
            }
            poly_add_scaled(
                &mut poly,
                &shifted_powers(n, pb[axis]),
                -2.0 * beta * f64::from(2 * n + 1),
            );
            poly_add_scaled(
                &mut poly,
                &shifted_powers(n + 2, pb[axis]),
                4.0 * beta * beta,
            );
            poly.iter().map(|c| -0.5 * c).collect()
        };

        let mut total = 0.0_f64;
        for derivative_axis in 0..3 {
            let mut term = e_ab;
            for axis in 0..3 {
                let factor = if axis == derivative_axis {
                    poly_mul(&bra(axis), &laplacian(axis))
                } else {
                    poly_mul(&bra(axis), &ket(axis))
                };
                term *= poly_contract(&factor, p);
            }
            total += term;
        }
        total
    }

    /// 用解析矩路径组装同一个 `kinetic_moment_integrals`（球谐基）。
    fn reference_moments(basis: &qwb_basis::InternalBasis) -> [Matrix; 3] {
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
        let mut cartesian: [Matrix; 3] =
            std::array::from_fn(|_| Matrix::zeros(n_cartesian, n_cartesian));
        let shells: Vec<_> = basis.shells().iter().collect();
        for (ia, shell_a) in shells.iter().enumerate() {
            for (ib, shell_b) in shells.iter().enumerate() {
                let components_a = qwb_basis::cartesian_components(shell_a.l);
                let components_b = qwb_basis::cartesian_components(shell_b.l);
                for (ca, powers_a) in components_a.iter().enumerate() {
                    for (cb, powers_b) in components_b.iter().enumerate() {
                        let mut values = [0.0_f64; 3];
                        for (alpha, coefficient_a) in
                            shell_a.exponents.iter().zip(&shell_a.coefficients)
                        {
                            let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                            for (beta, coefficient_b) in
                                shell_b.exponents.iter().zip(&shell_b.coefficients)
                            {
                                let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                                let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                                for (axis, value) in values.iter_mut().enumerate() {
                                    *value += weight
                                        * reference_primitive(
                                            *powers_a,
                                            shell_a.center_bohr,
                                            *alpha,
                                            *powers_b,
                                            shell_b.center_bohr,
                                            *beta,
                                            axis,
                                        );
                                }
                            }
                        }
                        let row = cart_offsets[ia] + ca;
                        let column = cart_offsets[ib] + cb;
                        for (axis, moment) in cartesian.iter_mut().enumerate() {
                            moment[(row, column)] = values[axis];
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
        let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n_spherical, n_spherical));
        for (axis, moment) in cartesian.iter().enumerate() {
            out[axis] = crate::dipole::transform_matrix(moment, &global_transform)
                .expect("transform moments");
        }
        out
    }

    #[test]
    fn the_kinetic_moment_matches_the_analytic_moment_reference() {
        for (name, molecule) in [
            ("sto-3g", displaced_water()),
            ("def2-svp", displaced_water()),
        ] {
            let basis = basis(name, &molecule);
            let production = kinetic_moment_integrals(&basis).expect("kinetic moment");
            let reference = reference_moments(&basis);
            let n = basis.n_functions();
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for (axis, matrix) in reference.iter().enumerate() {
                for mu in 0..n {
                    for nu in 0..n {
                        scale = scale.max(matrix.get(mu, nu).abs());
                        worst =
                            worst.max((production[axis].get(mu, nu) - matrix.get(mu, nu)).abs());
                    }
                }
            }
            println!("{name} 动磁矩 vs 解析矩参考：最差 {worst:.3e}（max|A| = {scale:.3e}）");
            assert!(
                scale > 1.0,
                "{name}: 参考矩阵太小（{scale:.3e}），尺子没有内容"
            );
            assert!(
                worst < 1.0e-10 * scale,
                "{name}: 与解析矩参考差 {worst:.3e}（max|A| = {scale:.3e}，判据 1e-10 相对）"
            );
        }
    }
}
