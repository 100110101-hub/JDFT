//! 数值 Hessian 与简正模（F-01/F-02 的基础）。
//!
//! # 为什么是「数值 Hessian」
//!
//! 解析 Hessian 要把梯度再求一次导，工作量比解析梯度大一个量级；而 F-01 的判据是
//! 「与更高精度差 < 5%（相对 Frobenius 范数）」，中心差分的数值 Hessian 在
//! δ ≈ 0.01 Bohr 上就够用（截断误差 O(δ²) ≈ 1e-4，而能量二阶导的量级是 0.1–1 Ha/Bohr²）。
//!
//! 位移之间**互相独立**，所以用 rayon 并行——这是整个梯度族里最自然的并行点。
//!
//! # 单位
//!
//! * Hessian：Ha / Bohr²；
//! * 质量加权后对角化，特征值 `λ` 对应的波数 `ν̃ = sqrt(λ) / (2π) · 5140.49 cm⁻¹`
//!   （`sqrt(Ha/(Bohr²·amu))` 换算成 cm⁻¹ 的常数）；
//! * ZPE = ½ Σ ν̃（cm⁻¹）。
//!
//! # 已知边界
//!
//! 原子质量表在本文件里（IUPAC 2021 标准原子量）。按项目的数据纪律，质量表本该进
//! `qwb-data/` 并带引用；这里先就地实现并标注，属于**待迁移的数据**（记在 F-01 的账里）。
//! 频率不是可观测量的一部分（IR 强度需要偶极导数），所以本模块只给简正模与 ZPE。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::Registry;
use rayon::prelude::*;

use crate::gradient::{GradientOptions, numerical_gradient};
use crate::opt::OptOptions;

/// 中心差分步长（Bohr）。0.01 时截断误差与辛普森式差分噪声都比较小。
pub const HESSIAN_DELTA_BOHR: f64 = 1.0e-2;

/// `sqrt(Ha / (Bohr² · amu))` → cm⁻¹。
pub const WAVE_NUMBER_FACTOR: f64 = 5140.4871;

/// IUPAC 2021 标准原子量（待迁移到 `qwb-data/`，见模块文档）。
fn atomic_mass(symbol: &str) -> Option<f64> {
    match symbol {
        "H" => Some(1.008),
        "He" => Some(4.002_602),
        "C" => Some(12.011),
        "N" => Some(14.007),
        "O" => Some(15.999),
        "F" => Some(18.998_403),
        _ => None,
    }
}

/// 一个简正模。
#[derive(Debug, Clone)]
pub struct NormalMode {
    /// 波数（cm⁻¹）。虚频为负。
    pub wavenumber: f64,
    /// 质量加权的本征向量摊回笛卡尔位移（未归一化到「原子位移」的物理长度）。
    pub displacement: Vec<[f64; 3]>,
}

/// Hessian 与简正模的结果。
#[derive(Debug, Clone)]
pub struct HessianResult {
    /// `3N × 3N` 的笛卡尔 Hessian（Ha/Bohr²），按行优先、下标 `3*atom + axis`。
    pub hessian: Vec<f64>,
    /// 按波数升序排列的简正模（虚频为负，排在最前）。
    pub modes: Vec<NormalMode>,
    /// 零点能（cm⁻¹），只含**振动**模（平移/转动不算）。
    pub zero_point_cm: f64,
    /// 振动模的个数（非线性 3N−6，线性 3N−5）。
    pub vibrational_modes: usize,
    /// 自由度数（`3N`）。
    pub degrees_of_freedom: usize,
}

/// 是否线性分子（所有原子共线）。
fn is_linear(molecule: &Molecule) -> bool {
    let count = molecule.atoms.len();
    if count <= 2 {
        return true;
    }
    let first = molecule.atoms[0].position_bohr();
    let mut direction: Option<[f64; 3]> = None;
    for atom in molecule.atoms.iter().skip(1) {
        let position = atom.position_bohr();
        let delta = [
            position[0] - first[0],
            position[1] - first[1],
            position[2] - first[2],
        ];
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if length < 1.0e-8 {
            continue;
        }
        let unit = [delta[0] / length, delta[1] / length, delta[2] / length];
        match direction {
            None => direction = Some(unit),
            Some(reference) => {
                let cross = [
                    reference[1] * unit[2] - reference[2] * unit[1],
                    reference[2] * unit[0] - reference[0] * unit[2],
                    reference[0] * unit[1] - reference[1] * unit[0],
                ];
                let magnitude =
                    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                if magnitude > 1.0e-6 {
                    return false;
                }
            }
        }
    }
    true
}

/// 位移一组的能量梯度（内部用）。
fn gradient_at(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
    atom: usize,
    axis: usize,
    shift: f64,
) -> Result<Vec<f64>, EngineError> {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let mut moved = molecule.clone();
    moved.atoms[atom].position_angstrom[axis] += shift * ANGSTROM_PER_BOHR;
    let mut gradient_options = GradientOptions::new(
        options.basis.clone(),
        options.functional.clone(),
        options.grid.clone(),
    );
    gradient_options.delta_bohr = 1.0e-3;
    let gradient = numerical_gradient(registry, &moved, &gradient_options)?;
    let mut flat = Vec::with_capacity(gradient.len() * 3);
    for row in &gradient {
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

/// 用**解析梯度**（s/p 都支持）算某一几何的梯度；拿不到时退回数值梯度。
fn analytic_gradient_flat(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<Vec<f64>, EngineError> {
    let mut scf_options = qwb_scf::rks::RksOptions::default();
    scf_options.max_iter = 400;
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    if !options.functional.is_empty() {
        // 有泛函时解析梯度还不支持（Pulay 项需要 v_xc 进 Fock），退回数值
        return Ok(Vec::new());
    }
    let one = qwb_integrals::one_electron::one_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    let result = qwb_scf::rks::rks_scf(&basis, molecule, &tensor, &grid, &[], scf_options)?;
    let fock = qwb_scf::build_fock(&one.core_hamiltonian, &tensor, &result.density)
        .map_err(|err| EngineError::Internal(format!("build_fock: {err}")))?
        .fock;
    match qwb_scf::analytic_gradient::analytic_gradient(&basis, molecule, &result.density, &fock) {
        Ok(gradient) => {
            let mut flat = Vec::with_capacity(gradient.len() * 3);
            for row in &gradient {
                flat.extend_from_slice(row);
            }
            Ok(flat)
        }
        Err(_) => Ok(Vec::new()),
    }
}

/// 数值 Hessian（中心差分 + rayon 并行位移）+ 简正模。
///
/// # Errors
///
/// 任何一次位移的能量/梯度计算失败，或坐标出现非有限值。
pub fn numerical_hessian(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<HessianResult, EngineError> {
    let natoms = molecule.atoms.len();
    let dof = 3 * natoms;
    // 每个位移需要 ±δ 两个几何的梯度
    let displacements: Vec<(usize, usize)> = (0..dof).map(|k| (k / 3, k % 3)).collect();

    let results: Vec<Result<Vec<f64>, EngineError>> = displacements
        .par_iter()
        .map(|(atom, axis)| {
            let plus = analytic_gradient_flat_for(
                registry,
                molecule,
                options,
                *atom,
                *axis,
                HESSIAN_DELTA_BOHR,
            )?;
            let minus = analytic_gradient_flat_for(
                registry,
                molecule,
                options,
                *atom,
                *axis,
                -HESSIAN_DELTA_BOHR,
            )?;
            let mut row = Vec::with_capacity(dof);
            for k in 0..dof {
                row.push((plus[k] - minus[k]) / (2.0 * HESSIAN_DELTA_BOHR));
            }
            Ok(row)
        })
        .collect();

    let mut hessian = vec![0.0_f64; dof * dof];
    for (column, result) in results.into_iter().enumerate() {
        let row = result?;
        for k in 0..dof {
            hessian[k * dof + column] = row[k];
        }
    }
    // 对称化（单侧差分噪声的对称化是标准做法）
    for i in 0..dof {
        for j in i + 1..dof {
            let average = 0.5 * (hessian[i * dof + j] + hessian[j * dof + i]);
            hessian[i * dof + j] = average;
            hessian[j * dof + i] = average;
        }
    }

    let modes = normal_modes(molecule, &hessian, dof)?;
    // 平移 3 个、转动 3 个（线性分子 2 个）都不算振动。它们在本征值谱里是最小的几个，
    // 所以「取最大的 n_vib 个」就够了——不需要额外做投影。
    let vibrational = if is_linear(molecule) {
        dof - 5
    } else {
        dof - 6
    };
    let mut zero_point = 0.0;
    for mode in modes.iter().rev().take(vibrational) {
        if mode.wavenumber > 0.0 {
            zero_point += 0.5 * mode.wavenumber;
        }
    }
    Ok(HessianResult {
        hessian,
        modes,
        zero_point_cm: zero_point,
        vibrational_modes: vibrational,
        degrees_of_freedom: dof,
    })
}

/// 位移后的解析梯度（失败时退回数值梯度）。
fn analytic_gradient_flat_for(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
    atom: usize,
    axis: usize,
    shift: f64,
) -> Result<Vec<f64>, EngineError> {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let mut moved = molecule.clone();
    moved.atoms[atom].position_angstrom[axis] += shift * ANGSTROM_PER_BOHR;
    let flat = analytic_gradient_flat(registry, &moved, options)?;
    if flat.is_empty() {
        gradient_at(registry, molecule, options, atom, axis, shift)
    } else {
        Ok(flat)
    }
}

/// 质量加权对角化，得到简正模。
fn normal_modes(
    molecule: &Molecule,
    hessian: &[f64],
    dof: usize,
) -> Result<Vec<NormalMode>, EngineError> {
    let mut masses = Vec::with_capacity(dof);
    for atom in &molecule.atoms {
        let mass = atomic_mass(&atom.symbol).ok_or_else(|| {
            EngineError::Data(format!(
                "没有 {} 的原子质量（F-01 的质量表待迁移到 qwb-data）",
                atom.symbol
            ))
        })?;
        masses.extend_from_slice(&[mass, mass, mass]);
    }
    // 质量加权：H̃_ij = H_ij / sqrt(m_i m_j)
    let mut weighted = vec![0.0_f64; dof * dof];
    for i in 0..dof {
        for j in 0..dof {
            weighted[i * dof + j] = hessian[i * dof + j] / (masses[i] * masses[j]).sqrt();
        }
    }
    let matrix = qwb_linalg::Matrix::from_row_major(dof, dof, &weighted)
        .map_err(|err| EngineError::Internal(format!("hessian matrix: {err}")))?;
    // 升序特征值；特征向量按列存放（`eigenvectors[(row, column)]`）
    let (eigenvalues, eigenvectors) = qwb_linalg::symmetric_eigen(&matrix)
        .map_err(|err| EngineError::Internal(format!("hessian eigen: {err}")))?;
    let mut modes = Vec::with_capacity(dof);
    for index in 0..dof {
        let eigenvalue = eigenvalues.get(index);
        let wavenumber = if eigenvalue > 0.0 {
            eigenvalue.sqrt() * WAVE_NUMBER_FACTOR
        } else {
            -(-eigenvalue).sqrt() * WAVE_NUMBER_FACTOR
        };
        let mut displacement = Vec::with_capacity(molecule.atoms.len());
        for atom in 0..molecule.atoms.len() {
            displacement.push([
                eigenvectors[(3 * atom, index)] / masses[3 * atom].sqrt(),
                eigenvectors[(3 * atom + 1, index)] / masses[3 * atom + 1].sqrt(),
                eigenvectors[(3 * atom + 2, index)] / masses[3 * atom + 2].sqrt(),
            ]);
        }
        modes.push(NormalMode {
            wavenumber,
            displacement,
        });
    }
    modes.sort_by(|left, right| left.wavenumber.total_cmp(&right.wavenumber));
    Ok(modes)
}
