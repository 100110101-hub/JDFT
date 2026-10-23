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
    /// 是否用 Pulay DIIS 加速（默认开）。
    pub use_diis: bool,
    /// DIIS 子空间最多保留多少组向量。
    pub diis_max_vectors: usize,
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
            use_diis: true,
            diis_max_vectors: 8,
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
    for i in 0..n {
        let lambda = values.get(i);
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(EngineError::Internal(format!(
                "overlap matrix is not positive definite (eigenvalue {lambda})"
            )));
        }
    }
    // X = U Λ^{-1/2} Uᵀ
    let mut scaled = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            scaled[(i, j)] = vectors[(i, j)] / values.get(j).sqrt();
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

/// 在网格上求密度与 XC 能量/v_xc。
///
/// # Errors
///
/// AO 求值遇到 l>=2，或密度出现非有限值。
pub fn evaluate_xc(
    basis: &InternalBasis,
    grid: &Grid,
    density: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<(f64, Matrix), EngineError> {
    let n = basis.n_functions();
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);

    let mut rho_values = Vec::with_capacity(grid.len());
    let mut sigma_values = Vec::with_capacity(grid.len());
    let mut ao_values: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    let mut ao_gradients: Vec<Vec<[f64; 3]>> = Vec::with_capacity(grid.len());
    let mut grad_rho: Vec<[f64; 3]> = Vec::with_capacity(grid.len());

    for GridPoint { position_bohr, .. } in &grid.points {
        let (values, gradients) = if needs_sigma {
            ao_value_and_gradient(basis, *position_bohr)?
        } else {
            (ao_value(basis, *position_bohr)?, Vec::new())
        };
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
    }

    let batch = if needs_sigma {
        DensityBatch::new(&rho_values, &sigma_values)
    } else {
        DensityBatch::from_rho(&rho_values)
    };
    let mut epsilon = vec![0.0; grid.len()];
    let mut d_epsilon = vec![0.0; grid.len()];
    let mut d_sigma = vec![0.0; grid.len()];

    for term in terms {
        let mut local_epsilon = vec![0.0; grid.len()];
        let mut local_d_rho = vec![0.0; grid.len()];
        let mut local_d_sigma = vec![0.0; grid.len()];
        term.energy_density(&batch, &mut local_epsilon);
        term.potential_derivatives(&batch, &mut local_d_rho, &mut local_d_sigma);
        for index in 0..grid.len() {
            epsilon[index] += local_epsilon[index];
            d_epsilon[index] += local_d_rho[index];
            d_sigma[index] += local_d_sigma[index];
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
        let values = &ao_values[index];
        let gradients = &ao_gradients[index];
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

    // 用 Xᵀ H X 对角化得到初猜。
    let mut density = core_guess(hcore, &x, n_occ)?;
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
    let mut e_xc = 0.0;
    let mut electronic = 0.0;

    for step in 0..options.max_iter {
        iterations = step + 1;
        let fock_data: FockMatrices = build_fock(hcore, tensor, &density).map_err(linalg)?;
        // 纯 HF（terms 为空）**不碰网格**：E_xc = 0、v_xc = 0 是精确结果，
        // 而走一遍网格要按点求 AO 值与密度（H2O/STO-3G 的 grid-coarse 是 29400 点 × 7 个 AO，
        // 每步都算一次会让 HF 的 SCF 慢两个数量级——实测每个 SCF 约 30 秒）。
        let (xc_energy, v_xc) = if terms.is_empty() {
            (0.0, Matrix::zeros(basis.n_functions(), basis.n_functions()))
        } else {
            evaluate_xc(basis, grid, &density, terms)?
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
                    hcore[(i, j)] + fock_data.coulomb[(i, j)] + v_xc[(i, j)]
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
        for i in 0..n {
            for j in 0..n {
                tr_ph += new_density[(i, j)] * hcore[(i, j)];
                // KS 的 2e 项是纯库仑 J；HF 的 2e 项才是 J − ½K。
                tr_pg += new_density[(i, j)]
                    * if hf_mode {
                        fock_data.fock[(i, j)] - hcore[(i, j)]
                    } else {
                        fock_data.coulomb[(i, j)]
                    };
            }
        }
        electronic = tr_ph + 0.5 * tr_pg + xc_energy;
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
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &parameters)
            .map_err(|err| EngineError::Data(format!("functional `{functional_name}`: {err}")))?;
        // 权重 ≠ 1（复合泛函）由 qwb-xc 的 ScaledTerm 处理：能量与两个导数用同一权重。
        let term = qwb_xc::scale_term(term, component.weight);
        // GGA（σ 依赖）现在由 evaluate_xc 支持（D-10）。
        terms.push(term);
    }

    rks_scf(&basis, molecule, &tensor, &grid, &terms, options)
}
