//! 过渡态搜索（`RP-01`）：Hessian 引导的局部搜索。
//!
//! 算法是**本征向量跟随（eigenvector following）的 Newton 变体**，每一步做三件事：
//!
//! 1. 在**质量加权**坐标下对角化 Hessian：`H̃ = M^{-1/2} H M^{-1/2}`；
//! 2. 把梯度也换成质量加权，沿每个本征方向取 `Δx̃_k = −(g̃·v_k)/λ_k^eff`，
//!    其中**最低的那个本征方向的 `λ` 取成 `−|λ|`** —— 于是沿反应坐标是**爬坡**，
//!    其余方向照常下山；
//! 3. 换回笛卡尔位移、按步长上限截断、必要时折半（重试），直到 `max|g| < 阈值`。
//!
//! # 为什么不做 P-RFO / dimer
//!
//! 这个项目要的是「给一个初猜、在附近找一个一阶鞍点」，不是全局 TS 搜索。本征向量跟随的
//! 最简形式（翻一个符号的 Newton 步）在**初猜已经在鞍点盆地内**时够用，而它每一步只有一个
//! 可解释的自由度（"沿最软的方向往上走"），比 P-RFO 的两套本征值和信赖域参数更容易核对。
//! 它的失败模式也清楚：初猜离鞍点太远、或最低本征方向中途换了一支 —— 所以收敛之后**必须**
//! 重新算一次 Hessian 并核对"恰好一个虚频"，那一步是这条路径的验收，不是装饰。
//!
//! # 边界（写在这里而不是留给使用者猜）
//!
//! * 只做闭壳层 HF/KS（与 `freq` 同一条梯度路径），ECP/溶剂/色散/开壳层一律拒绝；
//! * 收敛判据是 `max|g| < options.ts.gradient_threshold`，**外加**"恰好一个虚频"；
//! * 每步都重算完整数值 Hessian（`6N` 次解析梯度）：对了小分子是对的取舍，对中大分子太贵 ——
//!   真正省钱的版本要 BFGS 式的 Hessian 更新，那是另一个 WP 的事。
#![deny(unsafe_code)]

use qwb_engine_api::ts::TsResult;
use qwb_engine_api::{EngineError, Molecule};

use crate::hessian::{WAVE_NUMBER_FACTOR, atomic_mass, numerical_hessian};
use crate::opt::{OptOptions, analytic_energy_and_gradient};

/// 一步搜索需要的东西（给 `run_ts` 用，也便于单测）。
pub struct StepContext {
    /// 质量加权的 Hessian 本征值（升序）。
    pub eigenvalues: Vec<f64>,
    /// 质量加权的 Hessian 本征向量（列向量，与 `eigenvalues` 同序）。
    pub eigenvectors: Vec<Vec<f64>>,
    /// 当前能量（Hartree）。
    pub energy_hartree: f64,
    /// 当前梯度（Ha/Bohr，`[3*atom + axis]`）。
    pub gradient: Vec<f64>,
    /// 当前的虚频个数（质量加权 Hessian 的负本征值）。
    pub imaginary: usize,
}

/// 质量加权 Hessian 的对角化结果（`[3N][3N]` 行主序输入）。
///
/// # Errors
///
/// 元素质量未知、矩阵维度不对、对角化失败。
pub fn analyse_hessian(
    molecule: &Molecule,
    hessian: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), EngineError> {
    let dof = molecule.n_atoms() * 3;
    if hessian.len() != dof * dof {
        return Err(EngineError::Internal(format!(
            "ts: hessian has {} entries for {dof} coordinates",
            hessian.len()
        )));
    }
    let mut weights = Vec::with_capacity(dof);
    for atom in &molecule.atoms {
        let mass = atomic_mass(&atom.symbol).ok_or_else(|| {
            EngineError::Data(format!("ts: unknown atomic mass for `{}`", atom.symbol))
        })?;
        let root = mass.sqrt();
        for _ in 0..3 {
            weights.push(root);
        }
    }
    let mut data = Vec::with_capacity(dof * dof);
    for row in 0..dof {
        for column in 0..dof {
            data.push(hessian[row * dof + column] / (weights[row] * weights[column]));
        }
    }
    // 对称化：数值 Hessian 已经对称化过，这里只是把舍入误差抹平。
    for row in 0..dof {
        for column in row + 1..dof {
            let average = 0.5 * (data[row * dof + column] + data[column * dof + row]);
            data[row * dof + column] = average;
            data[column * dof + row] = average;
        }
    }
    let matrix = qwb_linalg::Matrix::from_column_major(dof, dof, data)
        .map_err(|error| EngineError::Internal(format!("ts: hessian matrix: {error}")))?;
    let (eigenvalues, eigenvectors) = qwb_linalg::symmetric_eigen(&matrix)
        .map_err(|error| EngineError::Internal(format!("ts: hessian eigen: {error}")))?;
    let values: Vec<f64> = (0..dof).map(|index| eigenvalues.get(index)).collect();
    let mut vectors = Vec::with_capacity(dof);
    for column in 0..dof {
        vectors.push((0..dof).map(|row| eigenvectors.get(row, column)).collect());
    }
    Ok((values, vectors))
}

/// 由质量加权 Hessian 的本征值算虚频（cm⁻¹，负值）。
#[must_use]
pub fn imaginary_wavenumber(eigenvalue: f64) -> f64 {
    if eigenvalue >= 0.0 {
        0.0
    } else {
        -((-eigenvalue).sqrt() * WAVE_NUMBER_FACTOR)
    }
}

/// 一次搜索迭代：返回下一步的笛卡尔几何（未截断/未验证），以及诊断。
///
/// `mode` 是被**翻符号**的那个本征方向的序号（0 = 最软）。若该方向的 `|λ|` 太小
/// （< `1e-6`，平移到转动那一档），用 `1e-3` 兜底 —— 否则步长会炸。
///
/// # Errors
///
/// 维度不一致或出现非有限值。
pub fn take_step(
    molecule: &Molecule,
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    gradient: &[f64],
    mode: usize,
    max_step_bohr: f64,
) -> Result<Molecule, EngineError> {
    let dof = molecule.n_atoms() * 3;
    if eigenvalues.len() != dof || gradient.len() != dof {
        return Err(EngineError::Internal(
            "ts: eigenvalue/gradient dimension mismatch".to_string(),
        ));
    }
    let mut weights = Vec::with_capacity(dof);
    for atom in &molecule.atoms {
        let mass = atomic_mass(&atom.symbol)
            .ok_or_else(|| EngineError::Data(format!("ts: unknown mass for `{}`", atom.symbol)))?;
        for _ in 0..3 {
            weights.push(mass.sqrt());
        }
    }
    let weighted_gradient: Vec<f64> = gradient
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value / weight)
        .collect();

    let mut displacement = vec![0.0_f64; dof];
    for (index, vector) in eigenvectors.iter().enumerate() {
        let mut projection = 0.0_f64;
        for coordinate in 0..dof {
            projection += vector[coordinate] * weighted_gradient[coordinate];
        }
        let mut eigenvalue = eigenvalues[index];
        if index == mode {
            // 反应坐标方向：**往上走**。
            eigenvalue = -eigenvalue.abs().max(1.0e-3);
        } else if eigenvalue.abs() < 1.0e-6 {
            // 平移/转动那一档（本该恰为零）：不让它贡献步长。
            continue;
        }
        let scale = -projection / eigenvalue;
        for coordinate in 0..dof {
            displacement[coordinate] += scale * vector[coordinate];
        }
    }
    // 质量加权 → 笛卡尔位移，再按步长上限截断。
    let mut cartesian: Vec<f64> = displacement
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value / weight)
        .collect();
    let norm = cartesian
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > max_step_bohr && norm > 0.0 {
        let factor = max_step_bohr / norm;
        cartesian.iter_mut().for_each(|value| *value *= factor);
    }
    if cartesian.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Internal("ts: step is not finite".into()));
    }
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let mut moved = molecule.clone();
    for atom in 0..molecule.n_atoms() {
        for axis in 0..3 {
            moved.atoms[atom].position_angstrom[axis] +=
                cartesian[3 * atom + axis] * ANGSTROM_PER_BOHR;
        }
    }
    Ok(moved)
}

/// 局部 TS 搜索的主循环。
///
/// # Errors
///
/// 任何一步的 Hessian/梯度计算失败、步长搜索失败，或收敛后核对发现虚频个数不是 1。
pub fn search(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    options: &OptOptions,
    max_steps: u32,
    threshold: f64,
    reporter: &dyn crate::Reporter,
) -> Result<TsSearchOutcome, EngineError> {
    let mut current = molecule.clone();
    let mut iterations = 0_u32;
    loop {
        let (energy, gradient) = analytic_energy_and_gradient(registry, &current, options)?;
        let flat: Vec<f64> = gradient
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let max_force = flat.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let hessian = numerical_hessian(registry, &current, options)?;
        let (eigenvalues, eigenvectors) = analyse_hessian(&current, &hessian.hessian)?;
        // 只数**真虚频**（波数 < −100 cm⁻¹）：3N 个本征值里有 6 个零模，数值噪声实测到
        // −48.7（H2O）、−16.4（NH3）—— 把它们算成虚频，"恰好一个虚频"这条判据就永远不成立。
        let wavenumbers: Vec<f64> = eigenvalues
            .iter()
            .map(|value| imaginary_wavenumber(*value))
            .collect();
        let real: Vec<f64> = wavenumbers
            .iter()
            .copied()
            .filter(|value| *value < qwb_engine_api::ts::IMAGINARY_THRESHOLD_CM)
            .collect();
        let imaginary = real.len();
        reporter.log(&format!(
            "ts step {iterations}: E = {energy:.10} Ha, max|g| = {max_force:.3e} Ha/Bohr, \
             imaginary = {imaginary}, lowest eigenvalue = {:.6e}",
            eigenvalues.first().copied().unwrap_or(f64::NAN)
        ));
        if max_force < threshold {
            let imaginary_wavenumbers = real.clone();
            if imaginary != 1 {
                return Err(EngineError::Unsupported(format!(
                    "transition-state search converged to a stationary point with {imaginary} \
                     imaginary mode(s), not 1: {imaginary_wavenumbers:?} cm^-1. A gradient-norm \
                     criterion alone cannot tell a minimum from a saddle, so this is a failure, \
                     not a result"
                )));
            }
            return Ok(TsSearchOutcome {
                molecule: current,
                energy_hartree: energy,
                gradient: flat,
                iterations,
                imaginary_wavenumber_cm: imaginary_wavenumbers
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::min),
                n_imaginary: u32::try_from(imaginary).unwrap_or(0),
                hessian_eigenvalues: eigenvalues,
            });
        }
        if iterations >= max_steps {
            return Err(EngineError::Unsupported(format!(
                "transition-state search did not converge in {iterations} steps \
                 (max|g| = {max_force:.3e} Ha/Bohr)"
            )));
        }
        iterations += 1;
        // 步长回退：整步让 |g| 变小就接受；否则把**步长上限**折半重试（最多 3 次）。
        let mut alpha = 1.0_f64;
        let mut accepted: Option<Molecule> = None;
        for _ in 0..3 {
            let moved = take_step(
                &current,
                &eigenvalues,
                &eigenvectors,
                &flat,
                0,
                options.max_step_bohr * alpha,
            )?;
            let (_, new_gradient) = analytic_energy_and_gradient(registry, &moved, options)?;
            let new_force = new_gradient
                .iter()
                .flat_map(|row| row.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            if new_force.is_finite() && (new_force < max_force || alpha <= 0.25) {
                accepted = Some(moved);
                break;
            }
            alpha *= 0.5;
        }
        let Some(next) = accepted else {
            return Err(EngineError::Unsupported(
                "transition-state search: every trial step increased |gradient|".to_string(),
            ));
        };
        current = next;
    }
}

/// 搜索的结果（与 K2 的 `TsResult` 一一对应，多带几何与本征值）。
pub struct TsSearchOutcome {
    /// 收敛后的几何。
    pub molecule: Molecule,
    /// 能量（Hartree）。
    pub energy_hartree: f64,
    /// 梯度（Ha/Bohr，`[3*atom + axis]`）。
    pub gradient: Vec<f64>,
    /// 迭代步数。
    pub iterations: u32,
    /// 虚频个数（只计波数 < −100 cm⁻¹ 的真虚频）。
    pub n_imaginary: u32,
    /// 最负的虚频波数（cm⁻¹，负值）；没有时 0。
    pub imaginary_wavenumber_cm: f64,
    /// 质量加权 Hessian 的本征值（升序）。
    pub hessian_eigenvalues: Vec<f64>,
}

/// 把搜索结果装进 K2 的块。
#[must_use]
pub fn to_api(outcome: &TsSearchOutcome) -> TsResult {
    TsResult::new(
        outcome.iterations,
        outcome.energy_hartree,
        outcome
            .gradient
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        outcome.n_imaginary,
        outcome.imaginary_wavenumber_cm,
    )
}
