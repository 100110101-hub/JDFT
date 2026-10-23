//! 几何优化（T3 的基础）：BFGS + 线搜索。
//!
//! # 为什么先做 BFGS 而不是「梯度下降足够用」
//!
//! T3 的判据是「最大力 < 4.5e-4 Ha/Bohr」——这要求真的收敛到驻点，而不是「能量降得差不多了」。
//! 纯梯度下降在水分子这种近简谐体系上会来回震荡，步长稍大就发散；BFGS 用割线条件逼近
//! Hessian，两三步就能把弯曲方向的慢模式提上来。
//!
//! # 梯度来源
//!
//! 优先用解析梯度（`qwb_scf::analytic_gradient`），它只支持 **s 型基组 + HF**（见
//! `docs/status.yaml` 的 G-01）；拿不到时退到数值梯度（`gradient::numerical_gradient`，
//! 每次位移重跑 SCF）。退化的规则写在返回值里（`OptResult.used_analytic_gradient`），
//! 不靠调用方猜。
//!
//! # 收敛
//!
//! 判据是力的**最大分量**（不是 RMS），与 T3 的写法一致。线搜索用 Armijo 回溯，
//! 每步步长上限 `max_step_bohr`，防止从很差的初值一步跳到别的势能面分支上。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

use crate::gradient::{GradientOptions, numerical_gradient};

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

/// 优化器的选项。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptOptions {
    /// 基组名。
    pub basis: String,
    /// 泛函名（空字符串表示 HF）。
    pub functional: String,
    /// 网格等级名。
    pub grid: String,
    /// 最多迭代多少步。
    pub max_iterations: u32,
    /// 收敛判据：力的最大分量（Ha/Bohr）。
    pub force_threshold: f64,
    /// 单步步长上限（Bohr）。
    pub max_step_bohr: f64,
}

impl OptOptions {
    /// 默认值：力阈值 4.5e-4 Ha/Bohr（T3 判据），步长上限 0.2 Bohr。
    #[must_use]
    pub fn new(
        basis: impl Into<String>,
        functional: impl Into<String>,
        grid: impl Into<String>,
    ) -> Self {
        Self {
            basis: basis.into(),
            functional: functional.into(),
            grid: grid.into(),
            max_iterations: 60,
            force_threshold: 4.5e-4,
            max_step_bohr: 0.2,
        }
    }
}

/// 优化结果。
#[derive(Debug, Clone)]
pub struct OptResult {
    /// 优化后的几何。
    pub molecule: Molecule,
    /// 优化后的能量（Hartree）。
    pub energy_hartree: f64,
    /// 优化后的梯度（Ha/Bohr）。
    pub gradient: Vec<[f64; 3]>,
    /// 实际迭代步数。
    pub iterations: u32,
    /// 是否满足力判据。
    pub converged: bool,
    /// 本次是否用了解析梯度（false = 数值梯度）。
    pub used_analytic_gradient: bool,
    /// 每一步的 `(能量, 最大力)`，第 0 项是初值（G-03 的「多步轨迹」）。
    pub trajectory: Vec<(f64, f64)>,
}

/// 把几何摊成 `[x1, y1, z1, x2, …]`（单位 Bohr）。
fn flatten(molecule: &Molecule) -> Vec<f64> {
    let mut out = Vec::with_capacity(molecule.atoms.len() * 3);
    for atom in &molecule.atoms {
        out.extend_from_slice(&atom.position_bohr());
    }
    out
}

/// 把摊平的坐标写回分子（输入单位 Bohr）。
fn unflatten(molecule: &Molecule, coordinates: &[f64]) -> Molecule {
    let mut out = molecule.clone();
    for (atom, chunk) in out.atoms.iter_mut().zip(coordinates.chunks_exact(3)) {
        atom.position_angstrom = [
            chunk[0] * ANGSTROM_PER_BOHR,
            chunk[1] * ANGSTROM_PER_BOHR,
            chunk[2] * ANGSTROM_PER_BOHR,
        ];
    }
    out
}

fn flatten_gradient(gradient: &[[f64; 3]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(gradient.len() * 3);
    for row in gradient {
        out.extend_from_slice(row);
    }
    out
}

/// 能量 + 梯度。优先解析（s 型 + HF），否则数值。
fn energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>, bool), EngineError> {
    let mut scf_options = RksOptions::default();
    scf_options.max_iter = 400;

    // 解析路径目前只对 **HF** 成立：KS 还缺一项「XC 能量的显式基组运动项」
    // `∫ v_xc·(∂ρ/∂R)|_{P 固定}`（见 tests/g01_analytic_pbe.rs 与 status.yaml 的 G-01）。
    if options.functional.is_empty() {
        let basis_k1 = registry.get_basis(&options.basis)?;
        let basis = qwb_basis::InternalBasis::build(basis_k1, molecule)?;
        let one = qwb_integrals::one_electron::one_electron(&basis, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        let terms = functional_terms(registry, options)?;
        let result = rks_scf(&basis, molecule, &tensor, &grid, &terms, scf_options)?;
        let mut fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &result.density)
            .map_err(|err| EngineError::Internal(format!("build_fock: {err}")))?
            .fock;
        if !terms.is_empty() {
            // KS：Pulay 项要求 F 是**完整的** Kohn–Sham Fock，即 H + J + v_xc
            let (_, v_xc) = qwb_scf::rks::evaluate_xc(&basis, &grid, &result.density, &terms)?;
            for mu in 0..fock.nrows() {
                for nu in 0..fock.ncols() {
                    fock.set(mu, nu, fock.get(mu, nu) + v_xc.get(mu, nu));
                }
            }
        }
        if let Ok(gradient) =
            qwb_scf::analytic_gradient::analytic_gradient(&basis, molecule, &result.density, &fock)
        {
            return Ok((result.energy_hartree, gradient, true));
        }
    }

    // 数值路径（任何基组/泛函）
    let mut gradient_options = GradientOptions::new(
        options.basis.clone(),
        options.functional.clone(),
        options.grid.clone(),
    );
    gradient_options.delta_bohr = 1.0e-3;
    let gradient = numerical_gradient(registry, molecule, &gradient_options)?;
    let energy = single_point(registry, molecule, options)?;
    Ok((energy, gradient, false))
}

fn single_point(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<f64, EngineError> {
    let mut scf_options = RksOptions::default();
    scf_options.max_iter = 400;
    if options.functional.is_empty() {
        let basis_k1 = registry.get_basis(&options.basis)?;
        let basis = qwb_basis::InternalBasis::build(basis_k1, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        Ok(rks_scf(&basis, molecule, &tensor, &grid, &[], scf_options)?.energy_hartree)
    } else {
        Ok(rks_from_data(
            registry,
            molecule,
            &options.basis,
            &options.functional,
            &options.grid,
            scf_options,
        )?
        .energy_hartree)
    }
}

/// 从注册表构造泛函项（空名字表示 HF）。
fn functional_terms(
    registry: &qwb_registry::Registry,
    options: &OptOptions,
) -> Result<Vec<Box<dyn qwb_xc::FunctionalTerm>>, EngineError> {
    let mut terms: Vec<Box<dyn qwb_xc::FunctionalTerm>> = Vec::new();
    if options.functional.is_empty() {
        return Ok(terms);
    }
    let functional = registry.get_functional(&options.functional)?;
    for component in &functional.components {
        let term = qwb_xc::build_term(&component.id, &functional.parameters)
            .map_err(|err| EngineError::Data(format!("functional `{}`: {err}", functional.name)))?;
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    Ok(terms)
}
fn max_force(gradient: &[f64]) -> f64 {
    gradient
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 对称正定矩阵乘向量（按行优先存）。
fn mat_vec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

/// BFGS 优化。
///
/// # Errors
///
/// 任何一步的能量/梯度计算失败，或坐标出现非有限值。
pub fn optimize(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<OptResult, EngineError> {
    let dimension = molecule.atoms.len() * 3;
    let mut coordinates = flatten(molecule);
    let (mut energy, mut gradient, used_analytic) =
        energy_and_gradient(registry, molecule, options)?;
    let mut flat_gradient = flatten_gradient(&gradient);

    // 逆 Hessian 初值：对角近似（0.7 Ha/Bohr² 是分子体系的常用量级）
    let mut inverse_hessian: Vec<Vec<f64>> = (0..dimension)
        .map(|i| {
            (0..dimension)
                .map(|j| if i == j { 1.0 / 0.7 } else { 0.0 })
                .collect()
        })
        .collect();

    let mut iterations = 0;
    let mut converged = max_force(&flat_gradient) < options.force_threshold;
    let mut trajectory = vec![(energy, max_force(&flat_gradient))];

    while !converged && iterations < options.max_iterations {
        iterations += 1;
        // 方向：p = −H g；若不再是下降方向就重置 H
        let mut direction: Vec<f64> = mat_vec(&inverse_hessian, &flat_gradient)
            .iter()
            .map(|value| -value)
            .collect();
        if dot(&direction, &flat_gradient) >= 0.0 {
            inverse_hessian = (0..dimension)
                .map(|i| {
                    (0..dimension)
                        .map(|j| if i == j { 1.0 / 0.7 } else { 0.0 })
                        .collect()
                })
                .collect();
            direction = flat_gradient.iter().map(|value| -value).collect();
        }
        // 步长上限
        let norm = dot(&direction, &direction).sqrt();
        let step = if norm > options.max_step_bohr {
            options.max_step_bohr / norm
        } else {
            1.0
        };
        direction.iter_mut().for_each(|value| *value *= step);

        // Armijo 回溯
        let slope = dot(&direction, &flat_gradient);
        let mut alpha = 1.0_f64;
        let mut accepted = None;
        for _ in 0..20 {
            let trial: Vec<f64> = coordinates
                .iter()
                .zip(&direction)
                .map(|(x, p)| x + alpha * p)
                .collect();
            if trial.iter().all(|value| value.is_finite()) {
                let trial_molecule = unflatten(molecule, &trial);
                let trial_energy = single_point(registry, &trial_molecule, options)?;
                if trial_energy <= energy + 1.0e-4 * alpha * slope {
                    accepted = Some((trial, trial_molecule, trial_energy));
                    break;
                }
            }
            alpha *= 0.5;
        }
        let Some((new_coordinates, new_molecule, new_energy)) = accepted else {
            // 线搜索失败：把这一步当成已收敛的前兆，交给调用方判断
            break;
        };

        let (_, new_gradient, _) = energy_and_gradient(registry, &new_molecule, options)?;
        let new_flat = flatten_gradient(&new_gradient);

        // BFGS 逆 Hessian 更新：H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
        let s: Vec<f64> = new_coordinates
            .iter()
            .zip(&coordinates)
            .map(|(a, b)| a - b)
            .collect();
        let y: Vec<f64> = new_flat
            .iter()
            .zip(&flat_gradient)
            .map(|(a, b)| a - b)
            .collect();
        let s_dot_y = dot(&s, &y);
        if s_dot_y > 1.0e-12 {
            let rho = 1.0 / s_dot_y;
            // Hy 与 yᵀH
            let hy = mat_vec(&inverse_hessian, &y);
            let y_h: Vec<f64> = (0..dimension)
                .map(|j| (0..dimension).map(|i| y[i] * inverse_hessian[i][j]).sum())
                .collect();
            let y_hy = dot(&y, &hy);
            let factor = (1.0 + y_hy * rho) * rho;
            for i in 0..dimension {
                for j in 0..dimension {
                    inverse_hessian[i][j] = inverse_hessian[i][j]
                        - rho * (s[i] * y_h[j] + hy[i] * s[j])
                        + factor * s[i] * s[j];
                }
            }
        } else {
            // 曲率为负/太小：重置，保持正定
            inverse_hessian = (0..dimension)
                .map(|i| {
                    (0..dimension)
                        .map(|j| if i == j { 1.0 / 0.7 } else { 0.0 })
                        .collect()
                })
                .collect();
        }

        coordinates = new_coordinates;
        energy = new_energy;
        gradient = new_gradient;
        flat_gradient = new_flat;
        converged = max_force(&flat_gradient) < options.force_threshold;
        trajectory.push((energy, max_force(&flat_gradient)));
    }

    Ok(OptResult {
        molecule: unflatten(molecule, &coordinates),
        energy_hartree: energy,
        gradient,
        iterations,
        converged,
        used_analytic_gradient: used_analytic,
        trajectory,
    })
}
