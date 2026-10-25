//! S-02：闭壳层 **RKS SCF**（LDA 泛函）。
//!
//! # 能量表达
//!
//! ```text
//! E = Tr(P H_core) + ½ Tr(P G) + E_xc + E_nuc
//! G = J（纯库仑；交换关联全部在 E_xc/v_xc 里）,  E_xc = Σ_g w_g ρ_g ε(ρ_g)
//! v_xc[i,j] = Σ_g w_g φ_i φ_j (ε + ρ dε/dρ)
//! ```
//!
//! # 办法
//!
//! 1. `X = S^{-1/2}`（对称特征分解，I-00），把广义本征问题变标准形式；
//! 2. 初猜：对角化 `H_core`；
//! 3. 迭代：密度 → 网格上的 ρ → XC 能量与 `v_xc` → `F` → 对角化 → 新 P；
//!    直到 `|ΔE| < threshold`；
//! 4. 每步都检查 `ΔP`，两者都达标才算收敛。
//!
//! # 第一阶段的范围（写清楚，免得误以为支持）
//!
//! - **只支持 LDA**（`requires.sigma == false`）。需要 σ 的 GGA 返回 `Unsupported`
//!   —— 网格上还没有梯度求值。
//! - **只支持 l <= 1 的 AO 求值**（s、p 直角坐标）；l >= 2 需要球谐变换后的 AO 求值，
//!   而积分层已经能算 l=2（R-06 已接），但网格上还缺球谐 AO 求值，故此处显式拒绝。
//! - 无 DIIS（S-03）、无 UKS（S-04）、无 checkpoint（S-05）。

#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use qwb_basis::{InternalBasis, cartesian_components};
use qwb_engine_api::{EngineError, Molecule};
use qwb_grid::{Grid, GridPoint};
use qwb_integrals::one_electron::{one_electron, primitive_norm};
use qwb_integrals::two_electron::{TwoElectronIntegrals, two_electron};
use qwb_linalg::{LinalgError, Matrix, Vector, matmul, symmetric_eigen};
use qwb_registry::elements::atomic_number;
use qwb_xc::{DensityBatch, FunctionalTerm};

use crate::diis::Diis;
use crate::{FockMatrices, build_fock};

/// SCF 控制参数。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RksOptions {
    /// 最大迭代步数。
    pub max_iter: u32,
    /// 能量收敛阈值（Hartree）。
    pub energy_threshold: f64,
    /// 密度矩阵收敛阈值。
    pub density_threshold: f64,
    /// HF 交换的混入比例 `α`（hybrid 泛函；纯泛函为 0）。
    ///
    /// 数据侧写在 functional 的 `components` 里：`- id: HfExchange, weight: 0.25`
    /// （`rks_from_data` 会把它读进这个字段，而不是当成能量密度项）。
    /// 进入 Fock 的是 `α·(−½K)`，进入能量的是 `α·(−¼Tr(PK))`。
    pub hf_exchange: f64,
    /// 是否用 Pulay DIIS 加速（默认开）。
    pub use_diis: bool,
    /// DIIS 子空间最多保留多少组向量。
    pub diis_max_vectors: usize,
    /// 初猜里 `H_core` 非对角块的缩放系数（默认 1.0 = 纯 core guess）。
    ///
    /// N2/STO-3G 这类近简并体系上，纯 core guess 会稳定收敛到**激发态**（N2 实测高
    /// 0.684 Ha）。配合 [`rks_scf_multistart`] 用几个不同的缩放因子各跑一次、取最低解。
    pub initial_guess_scale: f64,
    /// 从检查点续算（S-05）。
    pub resume_from: Option<crate::checkpoint::ScfCheckpoint>,
    /// 检查点文件路径（`None` = 不写检查点）。
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// 每多少步写一次检查点（0 = 只在结束时写）。
    pub checkpoint_every: u32,
}

impl Default for RksOptions {
    fn default() -> Self {
        Self {
            max_iter: 128,
            energy_threshold: 1.0e-8,
            density_threshold: 1.0e-6,
            hf_exchange: 0.0,
            use_diis: true,
            diis_max_vectors: 8,
            initial_guess_scale: 1.0,
            resume_from: None,
            checkpoint_path: None,
            checkpoint_every: 0,
        }
    }
}

/// SCF 结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RksResult {
    /// 总能量，Hartree。
    pub energy_hartree: f64,
    /// 是否收敛。
    pub scf_converged: bool,
    /// 实际迭代步数。
    pub iterations: u32,
    /// 电子能量（不含核排斥）。
    pub electronic_hartree: f64,
    /// 交换相关能。
    pub exchange_correlation_hartree: f64,
    /// 核排斥能。
    pub nuclear_repulsion_hartree: f64,
    /// 密度矩阵。
    pub density: Matrix,
    /// 轨道能（升序）。
    pub mo_energies_hartree: Vector,
    /// **MO 系数矩阵**（列 = MO，AO 为基；`C` 满足 `Cᵀ S C = 1`）。
    ///
    /// 之前 `RksResult` 只带密度矩阵，于是 `qwb-engine` 拿密度去填 K2 的
    /// `JobResult.mo_coeffs` —— 字段名与内容不符（P-10 的轨道等值面也因此做不了）。
    pub coefficients: Matrix,
    /// 最后一次能量变化。
    pub last_energy_change: f64,
    /// 网格点数（诊断用）。
    pub grid_points: usize,
    /// 终态检查点（可直接存盘续算）。
    pub checkpoint: crate::checkpoint::ScfCheckpoint,
}

fn linalg(err: LinalgError) -> EngineError {
    EngineError::Internal(format!("linear algebra: {err}"))
}

/// 求 `S^{-1/2}`（对称正交化）。
fn inverse_sqrt_symmetric(s: &Matrix) -> Result<Matrix, EngineError> {
    let (values, vectors) = symmetric_eigen(s).map_err(linalg)?;
    let n = s.nrows();
    let largest = (0..n).map(|i| values.get(i)).fold(0.0_f64, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return Err(EngineError::Internal(format!(
            "overlap matrix is degenerate (largest eigenvalue {largest})"
        )));
    }
    // 秩截断的规范正交化：近线性相关的基组（6-311G(d,p) 这类共享指数的三重分裂）
    // 会出现 ~1e-16 的本征值。标准做法是**丢弃**它们（对应的方向不进正交化基），
    // 而不是断言矩阵正定——断言只会让这类基组直接不可用。
    let threshold = largest * 1.0e-10;
    let mut dropped = 0usize;
    for i in 0..n {
        let lambda = values.get(i);
        if !lambda.is_finite() || lambda < -threshold {
            return Err(EngineError::Internal(format!(
                "overlap matrix is not positive semidefinite (eigenvalue {lambda})"
            )));
        }
        if lambda < threshold {
            dropped += 1;
        }
    }
    if dropped > 0 {
        // 每个进程只提示一次：SCF 每次迭代都会调这个函数，逐次打印会把真正的输出淹掉。
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "warning: overlap matrix has {dropped} near-zero eigenvalue(s) below {threshold:.2e}; \
using rank-truncated orthogonalization (basis is nearly linearly dependent)"
            );
        });
    }
    // X = U Λ^{-1/2} Uᵀ，被截断的方向取 0
    let mut scaled = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let lambda = values.get(j);
            scaled[(i, j)] = if lambda < threshold {
                0.0
            } else {
                vectors[(i, j)] / lambda.sqrt()
            };
        }
    }
    let vectors_t = vectors.transpose();
    matmul(&scaled, &vectors_t).map_err(linalg)
}

/// 核排斥能 `Σ_{A>B} Z_A Z_B / R_AB`。
fn nuclear_repulsion(molecule: &Molecule) -> Result<f64, EngineError> {
    let mut charges = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        let z = atomic_number(&atom.symbol)
            .ok_or_else(|| EngineError::Internal(format!("unknown element {}", atom.symbol)))?;
        charges.push((atom.position_bohr(), f64::from(z)));
    }
    let mut energy = 0.0;
    for i in 0..charges.len() {
        for j in (i + 1)..charges.len() {
            let distance = (0..3)
                .map(|axis| (charges[i].0[axis] - charges[j].0[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            energy += charges[i].1 * charges[j].1 / distance;
        }
    }
    Ok(energy)
}

/// 一个 AO 在给定点处的值（只支持 l <= 1）。pub 是为了让测试能直接查有限性。
#[allow(clippy::needless_range_loop)] // 分量下标与 components 一一对应，比迭代器更清楚
pub fn ao_value(basis: &InternalBasis, point: [f64; 3]) -> Result<Vec<f64>, EngineError> {
    let n = basis.n_functions();
    let mut values = vec![0.0; n];
    // 热路径：网格上每个点都要调用，所以这里**不做任何堆分配**（`MAX_CARTESIAN` 够 l ≤ 3）。
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let mut cartesian = [0.0_f64; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let mut value = 0.0;
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm(*powers, *alpha);
                let d = [
                    point[0] - shell.center_bohr[0],
                    point[1] - shell.center_bohr[1],
                    point[2] - shell.center_bohr[2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let polynomial = d[0].powi(i32::try_from(powers.0).expect("small"))
                    * d[1].powi(i32::try_from(powers.1).expect("small"))
                    * d[2].powi(i32::try_from(powers.2).expect("small"));
                value += coefficient * norm * polynomial * (-alpha * r2).exp();
            }
            cartesian[component] = value; // 显式下标：与 components 一一对应
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian[..components.len()],
            shell.l,
            shell.angular,
        )?;
    }
    Ok(values)
}

/// 把直角分量写进 AO 数组：`Cartesian` 直接抄，`Spherical` 乘球谐变换矩阵。
///
/// `l ≤ 1` 时直角与球谐是同一组函数（`T` 是单位阵），所以这个函数对 STO-3G 完全不改行为。
#[allow(clippy::needless_range_loop)] // 矩阵收缩：两个下标各有含义，range 最直观
fn write_transformed(
    values: &mut [f64],
    first: usize,
    cartesian: &[f64],
    l: u32,
    angular: qwb_registry::Angular,
) -> Result<(), EngineError> {
    let count = cartesian.len();
    match qwb_basis::spherical::spherical_transform(l, angular) {
        None => {
            for (component, value) in cartesian.iter().enumerate() {
                values[first + component] = *value;
            }
            Ok(())
        }
        Some(transform) => {
            if transform.n_spherical > count {
                return Err(EngineError::Internal(format!(
                    "球谐变换维度 {} 超过直角分量数 {count}",
                    transform.n_spherical
                )));
            }
            for spherical in 0..transform.n_spherical {
                let mut sum = 0.0;
                for component in 0..count {
                    sum += transform.get(component, spherical) * cartesian[component];
                }
                values[first + spherical] = sum;
            }
            Ok(())
        }
    }
}

/// AO 值 + 梯度（GGA 的前置）。
///
/// 对笛卡尔高斯 `x^a y^b z^c e^{−α|r−A|²}`（`u = r − A`）：
///
/// ```text
/// ∂/∂x = [ a·u_x^{a−1} − 2α·u_x^{a+1} ] · u_y^b · u_z^c · e^{−α|u|²}
/// ```
///
/// 也就是说梯度的每个分量仍是「同一族单项式」，只是 x 方向的幂次减一或加二。
///
/// # Errors
///
/// 与 [`ao_value`] 相同：遇到 `l ≥ 2`。
pub fn ao_value_and_gradient(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<(Vec<f64>, Vec<[f64; 3]>), EngineError> {
    let n = basis.n_functions();
    let mut values = vec![0.0; n];
    let mut gradients = vec![[0.0_f64; 3]; n];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian_values = [0.0_f64; MAX_CARTESIAN];
        let mut cartesian_gradients = [[0.0_f64; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let powers = [powers.0, powers.1, powers.2];
            let mut value = 0.0;
            let mut gradient = [0.0_f64; 3];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((powers[0], powers[1], powers[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                value += weight * monomial(powers);
                for axis in 0..3 {
                    // a·u^(a−1) 项（a ≥ 1 才有）
                    let mut derivative = 0.0;
                    if powers[axis] >= 1 {
                        let mut lowered = powers;
                        lowered[axis] -= 1;
                        derivative += f64::from(powers[axis]) * monomial(lowered);
                    }
                    // −2α·u^(a+1) 项
                    let mut raised = powers;
                    raised[axis] += 1;
                    derivative -= 2.0 * alpha * monomial(raised);
                    gradient[axis] += weight * derivative;
                }
            }
            cartesian_values[component] = value;
            cartesian_gradients[component] = gradient;
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian_values[..components.len()],
            shell.l,
            shell.angular,
        )?;
        // 梯度是对**函数**逐分量线性组合，所以同一个变换矩阵作用在三个分量上。
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        let mut slice = [0.0_f64; MAX_CARTESIAN];
        for axis in 0..3 {
            for component in 0..components.len() {
                slice[component] = cartesian_gradients[component][axis];
            }
            let mut flat = [0.0_f64; MAX_CARTESIAN];
            write_transformed(
                &mut flat,
                0,
                &slice[..components.len()],
                shell.l,
                shell.angular,
            )?;
            for slot in 0..range {
                gradients[shell.first_function + slot][axis] = flat[slot];
            }
        }
    }
    Ok((values, gradients))
}

/// 球谐函数个数：`l ≤ 1` 与直角相同，`l = 2` 是 5。
const MAX_CARTESIAN: usize = 10;

/// 球谐函数个数：l ≤ 1 与直角相同，l = 2 是 5。
fn spherical_count(l: u32) -> usize {
    (2 * l as usize + 1).min(6)
}

/// 每个 AO 在 `point` 处的**拉普拉斯量** `∇²φ`。
///
/// 对笛卡尔分量 `φ = N·u_x^a u_y^b u_z^c e^{−αu²}`（`u = r − A`）：
///
/// ```text
/// ∂²/∂u_x² [u_x^a e^{−αu²}] = a(a−1)u_x^{a−2} − 2α(2a+1)u_x^a + 4α²u_x^{a+2}
/// ```
///
/// 三个轴各按这个式子取一项（只改本轴的幂次），求和即 `∇²φ`。球谐壳层同样按
/// `spherical_transform` 的列做线性组合——与 `ao_value_and_gradient` 一致。
///
/// # Errors
///
/// 形状或球谐变换维度不一致。
#[allow(clippy::needless_range_loop)]
pub fn ao_laplacian(basis: &InternalBasis, point: [f64; 3]) -> Result<Vec<f64>, EngineError> {
    let mut values = vec![0.0_f64; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let mut cartesian = [0.0_f64; MAX_CARTESIAN];
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (component, powers) in components.iter().enumerate() {
            let powers = [powers.0, powers.1, powers.2];
            let monomial = |exponents: [u32; 3]| {
                d[0].powi(i32::try_from(exponents[0]).expect("small"))
                    * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                    * d[2].powi(i32::try_from(exponents[2]).expect("small"))
            };
            let mut value = 0.0;
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((powers[0], powers[1], powers[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let mut laplacian = 0.0;
                for axis in 0..3 {
                    // a(a−1)u^{a−2}
                    if powers[axis] >= 2 {
                        let mut lowered = powers;
                        lowered[axis] -= 2;
                        laplacian += f64::from(powers[axis])
                            * f64::from(powers[axis] - 1)
                            * monomial(lowered);
                    }
                    // −2α(2a+1)u^a
                    laplacian -=
                        2.0 * alpha * (2.0 * f64::from(powers[axis]) + 1.0) * monomial(powers);
                    // +4α²u^{a+2}
                    let mut raised = powers;
                    raised[axis] += 2;
                    laplacian += 4.0 * alpha * alpha * monomial(raised);
                }
                value += weight * laplacian;
            }
            cartesian[component] = value;
        }
        write_transformed(
            &mut values,
            shell.first_function,
            &cartesian[..components.len()],
            shell.l,
            shell.angular,
        )?;
    }
    Ok(values)
}

/// 在网格上求密度与 XC 能量/v_xc。
///
/// # Errors
///
/// 网格上的 AO 数据（`φ`、`∇φ`、`∇²φ`），与密度无关，**可以在 SCF 迭代之间复用**。
///
/// SCF 每步都要在整张网格上重算 AO 三元组，这是 XC 求值的主要开销（苯/STO-3G 实测约 5 s/步，
/// 400 步就是半小时）。网格与基组在 SCF 期间不变，所以这里把它算一次、后面只做密度缩并。
pub struct GridAoCached {
    values: Vec<Vec<f64>>,
    gradients: Vec<Vec<[f64; 3]>>,
    laplacians: Vec<Vec<f64>>,
}

impl GridAoCached {
    /// 按需求（是否需要 σ / ∇²ρ）预取网格上的 AO 数据。
    ///
    /// # Errors
    /// AO 求值失败（例如基函数角动量超出现有实现）。
    pub fn build(
        basis: &InternalBasis,
        grid: &Grid,
        needs_sigma: bool,
        needs_laplacian: bool,
    ) -> Result<Self, EngineError> {
        let mut values = Vec::with_capacity(grid.len());
        let mut gradients = Vec::with_capacity(grid.len());
        let mut laplacians = Vec::with_capacity(grid.len());
        for GridPoint { position_bohr, .. } in &grid.points {
            let (value, gradient) = if needs_sigma || needs_laplacian {
                ao_value_and_gradient(basis, *position_bohr)?
            } else {
                (ao_value(basis, *position_bohr)?, Vec::new())
            };
            let laplacian = if needs_laplacian {
                ao_laplacian(basis, *position_bohr)?
            } else {
                Vec::new()
            };
            values.push(value);
            gradients.push(gradient);
            laplacians.push(laplacian);
        }
        Ok(Self {
            values,
            gradients,
            laplacians,
        })
    }
}

/// AO 求值遇到 l>=2，或密度出现非有限值。
pub fn evaluate_xc(
    basis: &InternalBasis,
    grid: &Grid,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<(f64, Matrix), EngineError> {
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let cache = GridAoCached::build(basis, grid, needs_sigma, needs_laplacian)?;
    evaluate_xc_cached(basis, grid, &cache, density, terms)
}

/// 与 [`evaluate_xc`] 相同，但 AO 数据由调用方预先算好（SCF 里每个迭代步复用）。
///
/// # Errors
/// 密度或密度梯度出现非有限值。
pub fn evaluate_xc_cached(
    basis: &InternalBasis,
    grid: &Grid,
    cache: &GridAoCached,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<(f64, Matrix), EngineError> {
    let n = basis.n_functions();
    // Requirements 是 non_exhaustive，只能逐字段问，不能构造字面量
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);

    let mut rho_values = Vec::with_capacity(grid.len());
    let mut sigma_values = Vec::with_capacity(grid.len());
    let mut laplacian_values = Vec::with_capacity(grid.len());
    let mut ao_values: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    let mut ao_gradients: Vec<Vec<[f64; 3]>> = Vec::with_capacity(grid.len());
    let mut ao_laplacians: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    let mut grad_rho: Vec<[f64; 3]> = Vec::with_capacity(grid.len());

    for (index, GridPoint { .. }) in grid.points.iter().enumerate() {
        // AO 三元组来自预取缓存（见 GridAoCached）：与密度无关，SCF 各步复用。
        let values = cache.values[index].clone();
        let gradients = cache.gradients[index].clone();
        let ao_laplacian_here = cache.laplacians[index].clone();
        // ∇²ρ = Σ_μν P_μν (∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν)
        let laplacian_here = if needs_laplacian {
            let mut total = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let p = density[(i, j)];
                    if p == 0.0 {
                        continue;
                    }
                    let dot = gradients[i][0] * gradients[j][0]
                        + gradients[i][1] * gradients[j][1]
                        + gradients[i][2] * gradients[j][2];
                    total += p
                        * (ao_laplacian_here[i] * values[j]
                            + 2.0 * dot
                            + values[i] * ao_laplacian_here[j]);
                }
            }
            if !total.is_finite() {
                // 诊断：把参与求和的量抽样吐出来，定位是 P、φ、∇φ 还是 ∇²φ 出了问题
                let mut max_p = 0.0_f64;
                let mut max_value = 0.0_f64;
                let mut max_gradient = 0.0_f64;
                let mut max_laplacian = 0.0_f64;
                for i in 0..n {
                    max_value = max_value.max(values[i].abs());
                    max_laplacian = max_laplacian.max(ao_laplacian_here[i].abs());
                    for value in &gradients[i] {
                        max_gradient = max_gradient.max(value.abs());
                    }
                    for j in 0..n {
                        max_p = max_p.max(density[(i, j)].abs());
                    }
                }
                return Err(EngineError::Internal(format!(
                    "density laplacian is not finite ({total}) at a grid point: \
                     max|P| {max_p:.3e}, max|φ| {max_value:.3e}, max|∇φ| {max_gradient:.3e}, \
                     max|∇²φ| {max_laplacian:.3e}"
                )));
            }
            total
        } else {
            0.0
        };
        laplacian_values.push(laplacian_here);
        let mut rho = 0.0;
        for i in 0..n {
            for j in 0..n {
                rho += values[i] * density[(i, j)] * values[j];
            }
        }
        // 防御：网格坐标或 AO 求值一旦产生非有限值，这里必须报错。
        // 历史教训：`f64::max(0.0)` 会把 NaN 静默变成 0，于是密度"看起来正常"，
        // 而 v_xc 里保留着 NaN —— 上一版就是这样把 NaN 带进 SCF 的。
        if !rho.is_finite() {
            return Err(EngineError::Internal(format!(
                "density is not finite ({rho}) at a grid point; grid coordinates or AO \
                 values are corrupted"
            )));
        }
        // ∇ρ = Σ_μν P_μν (∇φ_μ φ_ν + φ_μ ∇φ_ν)
        let mut gradient_here = [0.0_f64; 3];
        if needs_sigma {
            for i in 0..n {
                for j in 0..n {
                    let p = density[(i, j)];
                    if p == 0.0 {
                        continue;
                    }
                    for axis in 0..3 {
                        gradient_here[axis] +=
                            p * (gradients[i][axis] * values[j] + values[i] * gradients[j][axis]);
                    }
                }
            }
            if !gradient_here.iter().all(|value| value.is_finite()) {
                return Err(EngineError::Internal(
                    "density gradient is not finite at a grid point".to_string(),
                ));
            }
        }
        rho_values.push(rho.max(0.0));
        sigma_values.push(
            gradient_here[0] * gradient_here[0]
                + gradient_here[1] * gradient_here[1]
                + gradient_here[2] * gradient_here[2],
        );
        grad_rho.push(gradient_here);
        ao_values.push(values);
        ao_gradients.push(gradients);
        ao_laplacians.push(ao_laplacian_here);
    }

    let batch = if needs_laplacian {
        DensityBatch::with_laplacian(&rho_values, &sigma_values, &laplacian_values)
    } else if needs_sigma {
        DensityBatch::new(&rho_values, &sigma_values)
    } else {
        DensityBatch::from_rho(&rho_values)
    };
    let mut epsilon = vec![0.0; grid.len()];
    let mut d_epsilon = vec![0.0; grid.len()];
    let mut d_sigma = vec![0.0; grid.len()];
    let mut d_laplacian = vec![0.0; grid.len()];

    for term in terms {
        let mut local_epsilon = vec![0.0; grid.len()];
        let mut local_d_rho = vec![0.0; grid.len()];
        let mut local_d_sigma = vec![0.0; grid.len()];
        let mut local_d_laplacian = vec![0.0; grid.len()];
        term.energy_density(&batch, &mut local_epsilon);
        term.potential_derivatives_laplacian(
            &batch,
            &mut local_d_rho,
            &mut local_d_sigma,
            &mut local_d_laplacian,
        );
        for index in 0..grid.len() {
            epsilon[index] += local_epsilon[index];
            d_epsilon[index] += local_d_rho[index];
            d_sigma[index] += local_d_sigma[index];
            d_laplacian[index] += local_d_laplacian[index];
        }
    }

    let mut energy = 0.0;
    let mut v_xc = Matrix::zeros(n, n);
    for (index, GridPoint { weight, .. }) in grid.points.iter().enumerate() {
        let rho = rho_values[index];
        energy += weight * rho * epsilon[index];
        // LDA 部分：v = ε + ρ dε/dρ；ρ=0 时不乘（0×inf = NaN）。
        let potential = if rho > 0.0 {
            epsilon[index] + rho * d_epsilon[index]
        } else {
            epsilon[index]
        };
        // GGA 部分：∫ φ_μ(−∇·(2·∂f/∂σ·∇ρ))φ_ν 用分部积分转成
        //   + 2·∂f/∂σ·∇ρ·∇(φ_μ φ_ν)，其中 ∂f/∂σ = ρ·dε/dσ。
        // 这样不需要在网格上做数值散度，矩阵元直接可达。
        let sigma_coefficient = if needs_sigma && rho > 0.0 {
            2.0 * rho * d_sigma[index]
        } else {
            0.0
        };
        // ∇²ρ 通道：∫ φ_μ(∇²(ρ·∂ε/∂∇²ρ))φ_ν（一遍分部积分）
        //   = ∫ (ρ·∂ε/∂∇²ρ)·∇²(φ_μ φ_ν)，而
        //   ∇²(φ_μ φ_ν) = ∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν ——三样都已经在手上。
        let laplacian_coefficient = if needs_laplacian && rho > 0.0 {
            rho * d_laplacian[index]
        } else {
            0.0
        };
        // 临时诊断：哪个系数先坏
        if !(potential.is_finite()
            && sigma_coefficient.is_finite()
            && laplacian_coefficient.is_finite())
        {
            return Err(EngineError::Internal(format!(
                "coefficient not finite at grid point {index}: potential {potential:.3e}, \
                 sigma_coeff {sigma_coefficient:.3e}, laplacian_coeff {laplacian_coefficient:.3e}, \
                 rho {rho:.3e}, d_eps {:.3e}, d_sigma {:.3e}, d_lap {:.3e}",
                d_epsilon[index], d_sigma[index], d_laplacian[index]
            )));
        }
        let values = &ao_values[index];
        let gradients = &ao_gradients[index];
        let laplacians = &ao_laplacians[index];
        let gradient_rho = grad_rho[index];
        for i in 0..n {
            if values[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                let mut element = values[i] * values[j] * potential;
                if sigma_coefficient != 0.0 {
                    // ∇ρ · ∇(φ_μ φ_ν)
                    let dot = (0..3)
                        .map(|axis| {
                            gradient_rho[axis]
                                * (gradients[i][axis] * values[j] + values[i] * gradients[j][axis])
                        })
                        .sum::<f64>();
                    element += sigma_coefficient * dot;
                }
                if laplacian_coefficient != 0.0 {
                    let dot = gradients[i]
                        .iter()
                        .zip(gradients[j].iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    element += laplacian_coefficient
                        * (laplacians[i] * values[j] + 2.0 * dot + values[i] * laplacians[j]);
                }
                v_xc[(i, j)] += weight * element;
            }
        }
    }
    Ok((energy, v_xc))
}

/// 闭壳层 RKS SCF。
///
/// # Errors
///
/// - 电子数为奇数 → `InvalidInput`（闭壳层要求偶数）；
/// - 泛函需要 σ（GGA）或 AO 需要 l>=2 → `Unsupported`；
/// - 线性代数失败 → `Internal`。
pub fn rks_scf(
    basis: &InternalBasis,
    molecule: &Molecule,
    tensor: &TwoElectronIntegrals,
    grid: &Grid,
    terms: &[Box<dyn FunctionalTerm>],
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let n = basis.n_functions();
    if tensor.n_functions != n {
        return Err(EngineError::Internal(
            "two-electron tensor size does not match the basis".to_string(),
        ));
    }
    let mut electrons = 0i64 - i64::from(molecule.charge);
    for atom in &molecule.atoms {
        let z = atomic_number(&atom.symbol)
            .ok_or_else(|| EngineError::Internal(format!("unknown element {}", atom.symbol)))?;
        electrons += i64::from(z);
    }
    if electrons <= 0 || electrons % 2 != 0 {
        return Err(EngineError::InvalidInput(format!(
            "closed-shell RKS needs an even, positive electron count, got {electrons}"
        )));
    }
    let n_occ = (electrons / 2) as usize;

    let integrals = one_electron(basis, molecule)?;
    let overlap = &integrals.overlap;
    let hcore = &integrals.core_hamiltonian;
    let x = inverse_sqrt_symmetric(overlap)?;
    let e_nuc = nuclear_repulsion(molecule)?;
    let mut diis = if options.use_diis {
        Some(Diis::new(options.diis_max_vectors))
    } else {
        None
    };

    // 用 Xᵀ H X 对角化得到初猜（`initial_guess_scale` 缩放非对角块，多初猜时用）。
    let guess_hcore = if (options.initial_guess_scale - 1.0).abs() < f64::EPSILON {
        hcore.clone()
    } else {
        let scale = options.initial_guess_scale;
        let mut scaled = hcore.clone();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    scaled[(i, j)] = hcore[(i, j)] * scale;
                }
            }
        }
        scaled
    };
    let mut density = core_guess(&guess_hcore, &x, n_occ)?;
    let mut previous_energy = f64::NAN;
    let mut completed_steps = 0u32;
    if let Some(checkpoint) = &options.resume_from {
        if checkpoint.n_functions != n {
            return Err(EngineError::InvalidInput(format!(
                "resume checkpoint has {} functions but this basis has {n}",
                checkpoint.n_functions
            )));
        }
        density = checkpoint.density_matrix()?;
        if let Some(energy) = checkpoint.energy_hartree {
            previous_energy = energy;
        }
        completed_steps = checkpoint.iterations;
    }
    // S-05：恢复 DIIS 子空间，否则续算的轨迹与不中断的跑法不同（末位会差几个 ulp）。
    if let Some(checkpoint) = &options.resume_from
        && let Some(accelerator) = diis.as_mut()
        && !checkpoint.diis_focks.is_empty()
    {
        *accelerator = Diis::restore(
            options.diis_max_vectors,
            n,
            checkpoint.diis_focks.clone(),
            checkpoint.diis_errors.clone(),
        )
        .map_err(linalg)?;
    }

    // 统一的检查点构造（带上 DIIS 子空间）。
    // 用嵌套 fn 而不是闭包：闭包会长期借用 `diis`，与循环里的 `diis.as_mut()` 冲突。
    fn build_checkpoint(
        density: &Matrix,
        energy: f64,
        steps: u32,
        diis: Option<&Diis>,
    ) -> crate::checkpoint::ScfCheckpoint {
        let base = crate::checkpoint::ScfCheckpoint::new(density, Some(energy), steps);
        match diis {
            Some(accelerator) => {
                let (focks, errors) = accelerator.snapshot();
                base.with_diis(focks, errors)
            }
            None => base,
        }
    }

    let mut iterations = 0;
    let mut converged = false;
    let mut last_change = f64::INFINITY;
    let mut mo_energies = Vector::zeros(n);
    // 最后一次对角化的 MO 系数（列 = MO）；收敛后即 K2 mo_coeffs 的真实内容。
    let mut final_coefficients = Matrix::zeros(n, n);
    let mut e_xc = 0.0;
    let mut electronic = 0.0;

    // AO 三元组与密度无关，整次 SCF 只算一次（苯/STO-3G 的网格求值实测占每步开销的大头）。
    let xc_cache = if terms.is_empty() {
        None
    } else {
        let needs_sigma = terms.iter().any(|term| term.requires().sigma);
        let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
        Some(GridAoCached::build(
            basis,
            grid,
            needs_sigma,
            needs_laplacian,
        )?)
    };

    for step in 0..options.max_iter {
        iterations = step + 1;
        let fock_data: FockMatrices = build_fock(hcore, tensor, &density).map_err(linalg)?;
        // 纯 HF（terms 为空）**不碰网格**：E_xc = 0、v_xc = 0 是精确结果，
        // 而走一遍网格要按点求 AO 值与密度（H2O/STO-3G 的 grid-coarse 是 29400 点 × 7 个 AO，
        // 每步都算一次会让 HF 的 SCF 慢两个数量级——实测每个 SCF 约 30 秒）。
        let (xc_energy, v_xc) = if terms.is_empty() {
            (0.0, Matrix::zeros(basis.n_functions(), basis.n_functions()))
        } else {
            match &xc_cache {
                Some(cache) => evaluate_xc_cached(basis, grid, cache, &density, terms)?,
                None => evaluate_xc(basis, grid, &density, terms)?,
            }
        };
        e_xc = xc_energy;

        // 两种约定：
        //  * Kohn–Sham（有 XC 泛函）：F = h + J + v_xc，交换关联全部经 E_xc/v_xc 进入；
        //  * Hartree–Fock（terms 为空）：F = h + J − ½K，即 build_fock 给出的 HF Fock。
        let hf_mode = terms.is_empty();
        let mut fock = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] = if hf_mode {
                    fock_data.fock[(i, j)]
                } else {
                    // KS：F = h + J + v_xc（+ hybrid 的 α·(−½K)）
                    hcore[(i, j)] + fock_data.coulomb[(i, j)] + v_xc[(i, j)]
                        - 0.5 * options.hf_exchange * fock_data.exchange[(i, j)]
                };
            }
        }

        // DIIS：把当步 Fock 与交换子误差推入子空间，用外推的 Fock 去对角化。
        let mut fock_used = fock.clone();
        if let Some(accelerator) = diis.as_mut() {
            let error = Diis::commutator(&fock, &density, overlap).map_err(linalg)?;
            accelerator.push(fock.clone(), error);
            if let Some(extrapolated) = accelerator.extrapolate().map_err(linalg)? {
                fock_used = extrapolated;
            }
        }

        // F' = Xᵀ F X，对角化，C = X C'
        let xt = x.transpose();
        let f_prime = matmul(&matmul(&xt, &fock_used).map_err(linalg)?, &x).map_err(linalg)?;
        let (values, vectors) = symmetric_eigen(&f_prime).map_err(linalg)?;
        mo_energies = values;
        let coefficients = matmul(&x, &vectors).map_err(linalg)?;
        final_coefficients = coefficients.clone();

        // 新密度：P = 2 Σ_{i<n_occ} c_i c_iᵀ
        let mut new_density = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for orbital in 0..n_occ.min(n) {
                    sum += coefficients[(i, orbital)] * coefficients[(j, orbital)];
                }
                new_density[(i, j)] = 2.0 * sum;
            }
        }

        // E = Tr(P H) + ½ Tr(P G) + E_xc + E_nuc
        let mut tr_ph = 0.0;
        let mut tr_pg = 0.0;
        let mut tr_pk = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ph += new_density[(i, j)] * hcore[(i, j)];
                // KS 的 2e 项是纯库仑 J；HF 的 2e 项才是 J − ½K。
                tr_pk += new_density[(i, j)] * fock_data.exchange[(i, j)];
                tr_pg += new_density[(i, j)]
                    * if hf_mode {
                        fock_data.fock[(i, j)] - hcore[(i, j)]
                    } else {
                        fock_data.coulomb[(i, j)]
                    };
            }
        }
        // hybrid：E += α·(−¼Tr(PK))；纯 HF 走的是 hf_mode（build_fock 已经含 −½K），不重复计
        let hybrid = if hf_mode {
            0.0
        } else {
            options.hf_exchange * (-0.25 * tr_pk)
        };
        electronic = tr_ph + 0.5 * tr_pg + xc_energy + hybrid;
        let energy = electronic + e_nuc;

        let mut max_dp = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_dp = max_dp.max((new_density[(i, j)] - density[(i, j)]).abs());
            }
        }
        last_change = if previous_energy.is_nan() {
            f64::INFINITY
        } else {
            (energy - previous_energy).abs()
        };
        density = new_density;
        previous_energy = energy;

        // S-05：按间隔落盘，进程被 kill 也能从最近一步续算。
        if let Some(path) = options.checkpoint_path.as_deref()
            && options.checkpoint_every > 0
            && (step as usize + 1).is_multiple_of(options.checkpoint_every as usize)
        {
            let snapshot = build_checkpoint(
                &density,
                previous_energy,
                completed_steps + iterations,
                diis.as_ref(),
            );
            crate::checkpoint::save_checkpoint(path, &snapshot)?;
        }

        if last_change < options.energy_threshold && max_dp < options.density_threshold {
            converged = true;
            break;
        }
    }

    if !converged {
        // S-05：不收敛也落盘，否则「kill 后续算」拿不到状态。
        if let Some(path) = options.checkpoint_path.as_deref() {
            let snapshot = build_checkpoint(
                &density,
                previous_energy,
                completed_steps + iterations,
                diis.as_ref(),
            );
            crate::checkpoint::save_checkpoint(path, &snapshot)?;
        }
        return Err(EngineError::ScfNotConverged {
            iters: iterations,
            max: options.max_iter,
        });
    }

    let checkpoint = build_checkpoint(
        &density,
        previous_energy,
        completed_steps + iterations,
        diis.as_ref(),
    );

    Ok(RksResult {
        energy_hartree: previous_energy + e_nuc - e_nuc, // 保持写法对称：electronic + e_nuc
        scf_converged: converged,
        // 续算时报**总**步数（含检查点之前完成的），不是本次新增的步数。
        iterations: completed_steps + iterations,
        electronic_hartree: electronic,
        exchange_correlation_hartree: e_xc,
        nuclear_repulsion_hartree: e_nuc,
        density,
        mo_energies_hartree: mo_energies,
        coefficients: final_coefficients,
        last_energy_change: last_change,
        grid_points: grid.len(),
        checkpoint,
    })
}

/// 初猜密度：对角化 `H_core` 并占据前 `n_occ` 个轨道（`P = 2 sum c c^T`）。
///
/// # Errors
///
/// 线性代数失败。
pub fn core_guess(hcore: &Matrix, x: &Matrix, n_occ: usize) -> Result<Matrix, EngineError> {
    let n = hcore.nrows();
    let xt = x.transpose();
    let h_prime = matmul(&matmul(&xt, hcore).map_err(linalg)?, x).map_err(linalg)?;
    let (_values, vectors) = symmetric_eigen(&h_prime).map_err(linalg)?;
    let coefficients = matmul(x, &vectors).map_err(linalg)?;
    let mut density = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for orbital in 0..n_occ.min(n) {
                sum += coefficients[(i, orbital)] * coefficients[(j, orbital)];
            }
            density[(i, j)] = 2.0 * sum;
        }
    }
    Ok(density)
}

/// 多初猜 SCF：用若干个 `initial_guess_scale` 各跑一次，返回**能量最低**的收敛解。
///
/// 动机（实测）：N2/STO-3G 的纯 core guess 会收敛到激发态——总能量比 CCCBDB 的
/// HF 值高 **0.684 Ha**，占据轨道能级也明显不对（1π_u 不成对简并、HOMO 被抬到 -0.18 Ha）。
/// 近简并体系（三键、同核双原子）上这是 SCF 初猜的经典问题，多初猜是标准对策。
///
/// # Errors
///
/// 所有初猜都不收敛，或线性代数失败。
pub fn rks_scf_multistart(
    basis: &InternalBasis,
    molecule: &Molecule,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    grid: &qwb_grid::Grid,
    terms: &[Box<dyn FunctionalTerm>],
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let mut best: Option<RksResult> = None;
    let mut last_error: Option<EngineError> = None;
    for scale in [1.0, 0.5, 1.5, 0.25, 2.0] {
        let mut attempt = options.clone();
        attempt.initial_guess_scale = scale;
        match rks_scf(basis, molecule, tensor, grid, terms, attempt) {
            Ok(result) => {
                let better = match &best {
                    None => true,
                    Some(current) => result.energy_hartree < current.energy_hartree - 1.0e-12,
                };
                if better {
                    best = Some(result);
                }
            }
            Err(err) => last_error = Some(err),
        }
    }
    match (best, last_error) {
        (Some(result), _) => Ok(result),
        (None, Some(err)) => Err(err),
        (None, None) => Err(EngineError::Internal(
            "multistart SCF: no attempt produced a result".to_string(),
        )),
    }
}

/// 便捷入口：从注册表里的基组/泛函/网格名做一次 RKS（T1 路径用）。
///
/// # Errors
///
/// 名字不存在、数据错误、SCF 输入非法。
pub fn rks_from_data(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional_name: &str,
    grid_name: &str,
    options: RksOptions,
) -> Result<RksResult, EngineError> {
    let basis_k1 = registry.get_basis(basis_name)?;
    let functional = registry.get_functional(functional_name)?;
    let grid_level = registry.get_grid(grid_name)?;
    let basis = InternalBasis::build(basis_k1, molecule)?;
    let tensor = two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(grid_level, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;

    let parameters: BTreeMap<String, f64> = functional.parameters.clone();
    let mut options = options;
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        // `HfExchange` 不是一个能量密度项，而是 hybrid 的混入比例：读进 options，不建 term。
        if component.id == "HfExchange" {
            options.hf_exchange += component.weight;
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters)
            .map_err(|err| EngineError::Data(format!("functional `{functional_name}`: {err}")))?;
        // 权重 ≠ 1（复合泛函）由 qwb-xc 的 ScaledTerm 处理：能量与两个导数用同一权重。
        let term = qwb_xc::scale_term(term, component.weight);
        // GGA（σ 依赖）现在由 evaluate_xc 支持（D-10）。
        terms.push(term);
    }

    rks_scf(&basis, molecule, &tensor, &grid, &terms, options)
}

// ---------------------------------------------------------------------------
// Kohn–Sham 解析梯度的 XC 显式项（G-01 / T6）
// ---------------------------------------------------------------------------

/// 每个 AO 在 `point` 处的**二阶导数张量** `∂²φ_μ/∂r_i∂r_j`（对称 3×3）。
///
/// 为什么需要它：GGA 的 XC 梯度里有 σ 通道，`σ = |∇ρ|²`，于是
/// `∂σ/∂R_A = 2∇ρ·∂(∇ρ)/∂R_A`，而 `∂(∇φ_μ)/∂R_A = −∂²φ_μ/∂r_i∂r_j`
/// —— 就是这里的量。只靠一阶 AO 导数写不出这一项（HF 没有它，所以这个缺口
/// 直到把 Fock 换成 KS 才暴露）。
///
/// 对笛卡尔分量 `φ = N·u_x^a u_y^b u_z^c e^{−αu²}`（`u = r − A`）：
///
/// ```text
/// ∂²/∂u_i∂u_j (g·e^{−αu²}) = e^{−αu²}[ ∂_i∂_j g − 2α(u_j ∂_i g + u_i ∂_j g) − 2α δ_ij g + 4α² u_i u_j g ]
/// ```
///
/// 令 `j = i` 就退回 `ao_laplacian` 里那条闭式（`a(a−1)u^{a−2} − 2α(2a+1)u^a + 4α²u^{a+2}`），
/// 两条公式的一致性由 `gga_ao_gradient` 的有限差分测试覆盖。
///
/// # Errors
///
/// 球谐变换维度不一致（`l >= 2` 的球谐壳层）。
pub fn ao_hessian(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<Vec<[[f64; 3]; 3]>, EngineError> {
    let mut hessian = vec![[[0.0_f64; 3]; 3]; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian = [[[0.0_f64; 3]; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let p = [powers.0, powers.1, powers.2];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((p[0], p[1], p[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                for i in 0..3 {
                    for j in 0..3 {
                        let mut term = 0.0;
                        // ∂_i∂_j g（单变量二阶 / 混合二阶）
                        if i == j {
                            if p[i] >= 2 {
                                let mut lowered = p;
                                lowered[i] -= 2;
                                term += f64::from(p[i] * (p[i] - 1)) * monomial(lowered);
                            }
                        } else if p[i] >= 1 && p[j] >= 1 {
                            let mut lowered = p;
                            lowered[i] -= 1;
                            lowered[j] -= 1;
                            term += f64::from(p[i] * p[j]) * monomial(lowered);
                        }
                        // −2α(u_j ∂_i g + u_i ∂_j g)
                        if p[i] >= 1 {
                            let mut shifted = p;
                            shifted[i] -= 1;
                            shifted[j] += 1;
                            term -= 2.0 * alpha * f64::from(p[i]) * monomial(shifted);
                        }
                        if p[j] >= 1 {
                            let mut shifted = p;
                            shifted[j] -= 1;
                            shifted[i] += 1;
                            term -= 2.0 * alpha * f64::from(p[j]) * monomial(shifted);
                        }
                        // −2α δ_ij g
                        if i == j {
                            term -= 2.0 * alpha * monomial(p);
                        }
                        // +4α² u_i u_j g
                        let mut raised = p;
                        raised[i] += 1;
                        raised[j] += 1;
                        term += 4.0 * alpha * alpha * monomial(raised);
                        cartesian[component][i][j] += weight * term;
                    }
                }
            }
        }
        // 球谐变换作用在每个 (i,j) 分量上（与 `ao_value_and_gradient` 同一套写法）。
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        for i in 0..3 {
            for j in 0..3 {
                let mut slice = [0.0_f64; MAX_CARTESIAN];
                for component in 0..components.len() {
                    slice[component] = cartesian[component][i][j];
                }
                let mut flat = [0.0_f64; MAX_CARTESIAN];
                write_transformed(
                    &mut flat,
                    0,
                    &slice[..components.len()],
                    shell.l,
                    shell.angular,
                )?;
                for slot in 0..range {
                    hessian[shell.first_function + slot][i][j] = flat[slot];
                }
            }
        }
    }
    Ok(hessian)
}

/// 每个 AO 的**拉普拉斯量的中心导数** `∂(∇²φ_μ)/∂r_a`（`a = 0,1,2`）。
///
/// 为什么需要它：LYP（B3LYP 的组件之一）的能量密度依赖 `∇²ρ`，于是 XC 梯度里有
/// `(ρ·∂ε/∂∇²ρ)·∂(∇²ρ)/∂R` 一项，而 `∂(∇²ρ)` 里含 `∂(∇²φ_μ)/∂R = −∂_a(∇²φ_μ)`。
///
/// 对笛卡尔分量 `∇²φ = Σ_prims w·L(u)`（`L` 是 `ao_laplacian` 里那串多项式），
/// 把 `∂_a` 作用上去：`∂_a(w·L) = w·(∂_a L − 2α u_a L)`。
///
/// # Errors
///
/// 球谐变换维度不一致。
pub fn ao_laplacian_derivative(
    basis: &InternalBasis,
    point: [f64; 3],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let mut derivatives = vec![[0.0_f64; 3]; basis.n_functions()];
    for shell in basis.shells() {
        let components = cartesian_components(shell.l);
        let d = [
            point[0] - shell.center_bohr[0],
            point[1] - shell.center_bohr[1],
            point[2] - shell.center_bohr[2],
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut cartesian = [[0.0_f64; 3]; MAX_CARTESIAN];
        for (component, powers) in components.iter().enumerate() {
            let p = [powers.0, powers.1, powers.2];
            for (alpha, coefficient) in shell.exponents.iter().zip(&shell.coefficients) {
                let norm = primitive_norm((p[0], p[1], p[2]), *alpha);
                let weight = coefficient * norm * (-alpha * r2).exp();
                let monomial = |exponents: [u32; 3]| {
                    d[0].powi(i32::try_from(exponents[0]).expect("small"))
                        * d[1].powi(i32::try_from(exponents[1]).expect("small"))
                        * d[2].powi(i32::try_from(exponents[2]).expect("small"))
                };
                // L(u) 本身（与 ao_laplacian 同式）
                let mut lap_polynomial = 0.0;
                for k in 0..3 {
                    if p[k] >= 2 {
                        let mut lowered = p;
                        lowered[k] -= 2;
                        lap_polynomial += f64::from(p[k] * (p[k] - 1)) * monomial(lowered);
                    }
                    lap_polynomial -= 2.0 * alpha * (2.0 * f64::from(p[k]) + 1.0) * monomial(p);
                    let mut raised = p;
                    raised[k] += 2;
                    lap_polynomial += 4.0 * alpha * alpha * monomial(raised);
                }
                for a in 0..3 {
                    // ∂_a L(u)
                    let mut derivative = 0.0;
                    for k in 0..3 {
                        if p[k] >= 2 {
                            let mut lowered = p;
                            lowered[k] -= 2;
                            let mut target = lowered;
                            if target[a] >= 1 {
                                target[a] -= 1;
                                derivative += f64::from(p[k] * (p[k] - 1))
                                    * f64::from(lowered[a])
                                    * monomial(target);
                            }
                        }
                        if p[a] >= 1 {
                            let mut target = p;
                            target[a] -= 1;
                            derivative -= 2.0
                                * alpha
                                * (2.0 * f64::from(p[k]) + 1.0)
                                * f64::from(p[a])
                                * monomial(target);
                        }
                        let mut raised = p;
                        raised[k] += 2;
                        if raised[a] >= 1 {
                            let mut target = raised;
                            target[a] -= 1;
                            derivative +=
                                4.0 * alpha * alpha * f64::from(raised[a]) * monomial(target);
                        }
                    }
                    cartesian[component][a] +=
                        weight * (derivative - 2.0 * alpha * d[a] * lap_polynomial);
                }
            }
        }
        let range = if shell.angular == qwb_registry::Angular::Spherical {
            spherical_count(shell.l)
        } else {
            components.len()
        };
        for a in 0..3 {
            let mut slice = [0.0_f64; MAX_CARTESIAN];
            for component in 0..components.len() {
                slice[component] = cartesian[component][a];
            }
            let mut flat = [0.0_f64; MAX_CARTESIAN];
            write_transformed(
                &mut flat,
                0,
                &slice[..components.len()],
                shell.l,
                shell.angular,
            )?;
            for slot in 0..range {
                derivatives[shell.first_function + slot][a] = flat[slot];
            }
        }
    }
    Ok(derivatives)
}

/// XC 能量的**显式基组运动项** `∂E_xc/∂R_A`（Kohn–Sham 解析梯度缺的那一块）。
///
/// Kohn–Sham 的能量是 `E = Tr(PH) + E_2e + E_xc(P) + E_nuc`，对核坐标求导时
/// `Tr(dP·F_KS) = −Tr(W dS)` 那一项由 `W = ½P F_KS P` 的 Pulay 项负责（`F_KS` 里
/// 已经含 `v_xc`），但 `E_xc` 对**基函数位置**的显式依赖必须单独加：
///
/// ```text
/// ∂E_xc/∂R_A,x = Σ_g w_g [ (ε + ρ ε_ρ)·∂ρ/∂R_A,x + (ρ ε_σ)·∂σ/∂R_A,x ]
/// ∂ρ/∂R_A,x    = Σ_{μ∈A} (∂φ_μ/∂R_A,x)·Ỹ_μ
/// ∂∇ρ/∂R_A,x   = Σ_{μ∈A} [ (∂∇φ_μ/∂R_A,x)·Ỹ_μ + (∂φ_μ/∂R_A,x)·Z̃_μ ]
/// ∂σ/∂R_A,x    = 2 ∇ρ·∂∇ρ/∂R_A,x
/// ```
///
/// 其中 `Ỹ_μ = Σ_ν (P_μν + P_νμ) φ_ν`、`Z̃_μ = Σ_ν (P_μν + P_νμ) ∇φ_ν`
/// （用 `P + Pᵀ` 是为了不假设 P 严格对称），`∂φ_μ/∂R_A,x = −∂φ_μ/∂r_x`（平移不变）。
///
/// **网格权重导数没有算**：这是主流代码的通行做法（网格随原子移动的那一项通常是
/// 1e-6 量级或更小）。这里不靠"通常"说服人 —— `g01_analytic_pbe` / `t6_b3lyp_gradient`
/// 直接拿它与**数值梯度**比，差多少就写多少。
///
/// # Errors
///
/// 需要 `∇²ρ` 的 meta-GGA 不在解析梯度的支持范围内（显式拒绝，不静默忽略）；
/// 密度或 AO 数据出现非有限值。
pub fn xc_gradient_terms(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &Grid,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<Vec<[f64; 3]>, EngineError> {
    let n = basis.n_functions();
    let natoms = molecule.atoms.len();
    let mut gradient = vec![[0.0_f64; 3]; natoms];
    if density.nrows() != n || density.ncols() != n {
        return Err(EngineError::Internal(
            "xc_gradient_terms: density size mismatch".to_string(),
        ));
    }
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    // LYP（B3LYP 的组件之一）需要 ∇²ρ，所以这一路也要算：`∂(∇²ρ)/∂R` 用
    // `ao_laplacian_derivative`（AO 拉普拉斯量再对中心求一次导）。
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    if terms.is_empty() {
        return Ok(gradient);
    }

    // AO → 原子（球谐/直角都按壳层的 function 区间取）
    let mut owner = vec![usize::MAX; n];
    for shell in basis.shells() {
        for slot in 0..shell.n_functions() {
            owner[shell.first_function + slot] = shell.atom_index;
        }
    }

    for GridPoint {
        position_bohr,
        weight,
        ..
    } in &grid.points
    {
        let point = *position_bohr;
        // ∇φ 一直要：LDA 的显式项也靠它（∂φ/∂R = −∂φ/∂r），只有 Hessian 能省。
        let (values, gradients) = ao_value_and_gradient(basis, point)?;
        let hessian = if needs_sigma {
            ao_hessian(basis, point)?
        } else {
            Vec::new()
        };
        let ao_laplacians = if needs_laplacian {
            ao_laplacian(basis, point)?
        } else {
            Vec::new()
        };
        let ao_laplacian_derivatives = if needs_laplacian {
            ao_laplacian_derivative(basis, point)?
        } else {
            Vec::new()
        };
        // ∇²ρ = Σ P (∇²φ_μ φ_ν + 2 ∇φ_μ·∇φ_ν + φ_μ ∇²φ_ν)
        let mut density_laplacian = 0.0;
        if needs_laplacian {
            for mu in 0..n {
                for nu in 0..n {
                    let p = density[(mu, nu)];
                    if p == 0.0 {
                        continue;
                    }
                    let dot = gradients[mu][0] * gradients[nu][0]
                        + gradients[mu][1] * gradients[nu][1]
                        + gradients[mu][2] * gradients[nu][2];
                    density_laplacian += p
                        * (ao_laplacians[mu] * values[nu]
                            + 2.0 * dot
                            + values[mu] * ao_laplacians[nu]);
                }
            }
        }

        // ρ 与 ∇ρ（与 SCF 的 evaluate_xc 同一条缩并）
        let mut rho = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                rho += values[mu] * density[(mu, nu)] * values[nu];
            }
        }
        if !rho.is_finite() {
            return Err(EngineError::Internal(
                "xc_gradient_terms: density is not finite at a grid point".to_string(),
            ));
        }
        let mut grad_rho = [0.0_f64; 3];
        if needs_sigma {
            for mu in 0..n {
                for nu in 0..n {
                    let p = density[(mu, nu)];
                    if p == 0.0 {
                        continue;
                    }
                    for axis in 0..3 {
                        grad_rho[axis] += p
                            * (gradients[mu][axis] * values[nu] + values[mu] * gradients[nu][axis]);
                    }
                }
            }
        }
        let sigma =
            grad_rho[0] * grad_rho[0] + grad_rho[1] * grad_rho[1] + grad_rho[2] * grad_rho[2];
        let rho = rho.max(0.0);

        // 泛函在这一点的一阶导数
        let rho_slice = [rho];
        let sigma_slice = [sigma];
        let laplacian_slice = [density_laplacian];
        let batch = if needs_laplacian {
            DensityBatch::with_laplacian(&rho_slice, &sigma_slice, &laplacian_slice)
        } else if needs_sigma {
            DensityBatch::new(&rho_slice, &sigma_slice)
        } else {
            DensityBatch::from_rho(&rho_slice)
        };
        let mut epsilon = [0.0_f64];
        let mut d_rho = [0.0_f64];
        let mut d_sigma = [0.0_f64];
        let mut d_laplacian = [0.0_f64];
        for term in terms {
            let mut local_epsilon = [0.0_f64];
            let mut local_rho = [0.0_f64];
            let mut local_sigma = [0.0_f64];
            let mut local_laplacian = [0.0_f64];
            term.energy_density(&batch, &mut local_epsilon);
            term.potential_derivatives_laplacian(
                &batch,
                &mut local_rho,
                &mut local_sigma,
                &mut local_laplacian,
            );
            epsilon[0] += local_epsilon[0];
            d_rho[0] += local_rho[0];
            d_sigma[0] += local_sigma[0];
            d_laplacian[0] += local_laplacian[0];
        }
        let v_laplacian = if needs_laplacian && rho > 0.0 {
            rho * d_laplacian[0]
        } else {
            0.0
        };
        let v_rho = if rho > 0.0 {
            epsilon[0] + rho * d_rho[0]
        } else {
            epsilon[0]
        };
        let v_sigma = if needs_sigma && rho > 0.0 {
            rho * d_sigma[0]
        } else {
            0.0
        };

        // Ỹ_μ = Σ_ν (P_μν + P_νμ) φ_ν 、 Z̃_μ = Σ_ν (P_μν + P_νμ) ∇φ_ν
        let mut y_tilde = vec![0.0_f64; n];
        let mut z_tilde = vec![[0.0_f64; 3]; n];
        for mu in 0..n {
            let mut y = 0.0;
            let mut z = [0.0_f64; 3];
            for nu in 0..n {
                let p = density[(mu, nu)] + density[(nu, mu)];
                if p == 0.0 {
                    continue;
                }
                y += p * values[nu];
                if needs_sigma {
                    for axis in 0..3 {
                        z[axis] += p * gradients[nu][axis];
                    }
                }
            }
            y_tilde[mu] = y;
            z_tilde[mu] = z;
        }
        // L̃_μ = Σ_ν (P_μν + P_νμ) ∇²φ_ν（只在需要 ∇²ρ 通道时用）
        let mut lap_tilde = vec![0.0_f64; n];
        if needs_laplacian {
            for mu in 0..n {
                let mut total = 0.0;
                for nu in 0..n {
                    let p = density[(mu, nu)] + density[(nu, mu)];
                    if p != 0.0 {
                        total += p * ao_laplacians[nu];
                    }
                }
                lap_tilde[mu] = total;
            }
        }

        for mu in 0..n {
            let atom = owner[mu];
            if atom == usize::MAX || y_tilde[mu] == 0.0 {
                continue;
            }
            for axis in 0..3 {
                let d_phi = -gradients[mu][axis];
                let mut sigma_axis = 0.0;
                if needs_sigma {
                    let mut grad_rho_axis = [0.0_f64; 3];
                    for j in 0..3 {
                        grad_rho_axis[j] =
                            -hessian[mu][axis][j] * y_tilde[mu] + d_phi * z_tilde[mu][j];
                    }
                    sigma_axis = 2.0
                        * (grad_rho[0] * grad_rho_axis[0]
                            + grad_rho[1] * grad_rho_axis[1]
                            + grad_rho[2] * grad_rho_axis[2]);
                }
                let mut laplacian_axis = 0.0;
                if needs_laplacian {
                    // ∂_A,a(∇²ρ) = Σ_{μ∈A} [ (∂_a∇²φ_μ)·Ỹ_μ + 2(∂_a∇φ_μ)·Z̃_μ + (∂_aφ_μ)·L̃_μ ]
                    laplacian_axis =
                        -ao_laplacian_derivatives[mu][axis] * y_tilde[mu] + d_phi * lap_tilde[mu];
                    for j in 0..3 {
                        laplacian_axis += 2.0 * (-hessian[mu][axis][j]) * z_tilde[mu][j];
                    }
                }
                gradient[atom][axis] += weight
                    * (v_rho * d_phi * y_tilde[mu]
                        + v_sigma * sigma_axis
                        + v_laplacian * laplacian_axis);
            }
        }
    }

    for slot in &gradient {
        if !slot.iter().all(|value| value.is_finite()) {
            return Err(EngineError::Internal(
                "xc_gradient_terms: gradient is not finite".to_string(),
            ));
        }
    }
    Ok(gradient)
}

/// Kohn–Sham 解析梯度：`骨架项 + ½α_hf 交换 + Pulay(W = ½P F_KS P) + XC 显式项`。
///
/// # 为什么必须由这里自己搭 Fock
///
/// Pulay 项用 `W = ½ P F_KS P`，而 `F_KS = H + J − ½α_hf·K + v_xc`：
///
/// * 纯泛函（LDA/GGA）**没有** HF 交换（`α_hf = 0`），`F` 里不能出现 `−½K`；
/// * 杂化泛函（B3LYP 0.20 / PBE0 0.25）要按 `α_hf` 缩放；
/// * 2e 骨架项同理：库仑恒有，交换乘 `α_hf`。
///
/// 把 `fock` 交给调用方去拼，就等于把"别把 HF 交换算进 KS 能量"这件事交给运气
/// —— 实测差 8.8e-2 Ha/Bohr，正是 `g01_analytic_pbe` 一开始失败的原因。
///
/// # Errors
///
/// 形状不匹配、AO 角动量超出实现、或 XC 是 meta-GGA（需要 `∇²ρ`，未支持）。
#[allow(clippy::too_many_arguments)]
pub fn rks_analytic_gradient(
    basis: &InternalBasis,
    molecule: &Molecule,
    grid: &Grid,
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
    hf_exchange: f64,
) -> Result<Vec<[f64; 3]>, EngineError> {
    let fock_hf = build_fock(hcore, tensor, density).map_err(|err| {
        EngineError::Internal(format!("rks_analytic_gradient: build_fock: {err}"))
    })?;
    let (_, v_xc) = evaluate_xc(basis, grid, density, terms)?;
    // F_KS = H + J − ½α_hf K + v_xc
    let n = basis.n_functions();
    let mut fock = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            fock[(mu, nu)] = hcore[(mu, nu)] + fock_hf.coulomb[(mu, nu)]
                - 0.5 * hf_exchange * fock_hf.exchange[(mu, nu)]
                + v_xc[(mu, nu)];
        }
    }
    let components =
        crate::analytic_gradient::gradient_components(basis, molecule, density, &fock)?;
    let xc = xc_gradient_terms(basis, molecule, grid, density, terms)?;

    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for (atom, slot) in gradient.iter_mut().enumerate() {
        for (axis, value) in slot.iter_mut().enumerate() {
            *value = components.kinetic[atom][axis]
                + components.nuclear_ao[atom][axis]
                + components.nuclear_center[atom][axis]
                + components.nuclear_repulsion[atom][axis]
                + components.two_electron_coulomb[atom][axis]
                + hf_exchange * components.two_electron_exchange[atom][axis]
                + components.pulay[atom][axis]
                + xc[atom][axis];
        }
    }
    Ok(gradient)
}
