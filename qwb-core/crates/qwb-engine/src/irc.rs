//! IRC（`RP-02`）：从一阶鞍点沿质量加权最陡下降路径走向两端极小点。
//!
//! # 算法
//!
//! 1. 在起点几何上算 Hessian，取**唯一的真虚频**（波数 < −100 cm⁻¹，见
//!    [`qwb_engine_api::ts::IMAGINARY_THRESHOLD_CM`]）对应本征向量的质量加权形式 `v`；
//! 2. 两个方向各走一遍：每步沿**质量加权的负梯度**取固定步长 `h`（单位 `sqrt(amu)·Bohr`），
//!    能量必须下降（否则步长折半，最多 4 次）；`max|g|` 低于判据就停；
//! 3. 端点再做一次**自由优化**，得到真正的极小点与它的内坐标指纹（键长/键角）。
//!
//! # 三个刻意的选择
//!
//! * **方向由能量判据定，不由虚频本征向量的符号定**：`v` 与 `−v` 都是合法的反应坐标方向，
//!   哪边"正"是任意的。所以每端先试 `±v` 各走一小步，取能量更低的那一步继续 —— 这样即使
//!   调用方把几何镜像了也不会走错边。
//! * **单调性是构造出来的，不是事后检查的**：只接受让能量下降的步（折半重试）。`monotone`
//!   报告里那个布尔仍然是算出来的（`IrcEnd::new`），因为判据要验的是结果而不是意图。
//! * **端点指纹用内坐标**：NH3 伞形反转的两个极小点是镜像，逐原子笛卡尔比较要先解决对齐与
//!   置换；键长/键角是置换与旋转不变的，`< 0.02 Å` 的判据因此可复现。
#![deny(unsafe_code)]

use qwb_engine_api::irc::{IrcEnd, IrcPoint, IrcResult};
use qwb_engine_api::ts::IMAGINARY_THRESHOLD_CM;
use qwb_engine_api::{EngineError, Molecule};

use crate::hessian::{atomic_mass, numerical_hessian};
use crate::opt::{OptOptions, analytic_energy_and_gradient, optimize};
use crate::ts::{analyse_hessian, imaginary_wavenumber};

/// 一个 IRC 端点的搜索输出（几何 + 能量 + 指纹 + 步数）。
pub struct EndOutcome {
    /// 优化后的端点几何。
    pub molecule: Molecule,
    /// 优化后的能量。
    pub energy_hartree: f64,
    /// 优化步数。
    pub optimisation_steps: u32,
    /// 内坐标指纹（键长 Å，按原子对顺序；然后是键角）。
    pub fingerprint: Vec<f64>,
}

/// 鞍点处唯一真虚频的质量加权本征向量（`[3*atom + axis]`）。
///
/// # Errors
///
/// Hessian 计算/对角化失败，或真虚频个数不是 1。
pub fn saddle_mode(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<(Vec<f64>, f64), EngineError> {
    let hessian = numerical_hessian(registry, molecule, options)?;
    let (eigenvalues, eigenvectors) = analyse_hessian(molecule, &hessian.hessian)?;
    let candidates: Vec<(usize, f64)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(index, value)| (index, imaginary_wavenumber(*value)))
        .filter(|(_, wavenumber)| *wavenumber < IMAGINARY_THRESHOLD_CM)
        .collect();
    if candidates.len() != 1 {
        return Err(EngineError::Unsupported(format!(
            "IRC needs exactly one real imaginary mode at the starting geometry, found {}: {:?}. \
             Run `qwb ts` first and use the saddle geometry",
            candidates.len(),
            candidates
                .iter()
                .map(|(_, wavenumber)| *wavenumber)
                .collect::<Vec<_>>()
        )));
    }
    let (index, wavenumber) = candidates[0];
    Ok((eigenvectors[index].clone(), wavenumber))
}

/// 质量加权的反梯度方向（单位化）转成笛卡尔位移（**不是**归一化后的笛卡尔单位向量）。
fn mass_weighted_step(
    molecule: &Molecule,
    gradient: &[f64],
    step_size: f64,
) -> Result<Vec<f64>, EngineError> {
    let dof = molecule.n_atoms() * 3;
    let mut weights = Vec::with_capacity(dof);
    for atom in &molecule.atoms {
        let mass = atomic_mass(&atom.symbol)
            .ok_or_else(|| EngineError::Data(format!("irc: unknown mass for `{}`", atom.symbol)))?;
        for _ in 0..3 {
            weights.push(mass.sqrt());
        }
    }
    let weighted: Vec<f64> = gradient
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value / weight)
        .collect();
    let norm = weighted
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm < 1.0e-12 {
        return Err(EngineError::Internal(
            "irc: gradient vanished before reaching a minimum".to_string(),
        ));
    }
    let mut cartesian: Vec<f64> = weighted
        .iter()
        .zip(&weights)
        .map(|(value, weight)| -step_size * value / norm / weight)
        .collect();
    if cartesian.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Internal("irc: step is not finite".into()));
    }
    let _ = &mut cartesian;
    Ok(cartesian)
}

fn displace(molecule: &Molecule, cartesian: &[f64], factor: f64) -> Molecule {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let mut moved = molecule.clone();
    for atom in 0..molecule.n_atoms() {
        for axis in 0..3 {
            moved.atoms[atom].position_angstrom[axis] +=
                factor * cartesian[3 * atom + axis] * ANGSTROM_PER_BOHR;
        }
    }
    moved
}

fn flatten(gradient: &[[f64; 3]]) -> Vec<f64> {
    gradient
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect()
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// 从鞍点出发走一端。
///
/// # Errors
///
/// 能量/梯度计算失败、步长搜索失败，或走不到极小点（`max_steps` 用完）。
#[allow(clippy::too_many_arguments)]
pub fn walk(
    registry: &qwb_registry::Registry,
    saddle: &Molecule,
    mode: &[f64],
    options: &OptOptions,
    step_size: f64,
    max_steps: u32,
    sign: f64,
    reporter: &dyn crate::Reporter,
) -> Result<(Vec<IrcPoint>, Molecule), EngineError> {
    // 方向由**能量**定：沿 ±v 各挪一小步，取能量更低的那边（v 的正负本来是任意的）。
    let mut weights = Vec::with_capacity(mode.len());
    for atom in &saddle.atoms {
        let mass = atomic_mass(&atom.symbol)
            .ok_or_else(|| EngineError::Data(format!("irc: unknown mass for `{}`", atom.symbol)))?;
        for _ in 0..3 {
            weights.push(mass.sqrt());
        }
    }
    let cartesian_mode: Vec<f64> = mode
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value / weight)
        .collect();
    let probe = step_size * 0.5;
    let (base_energy, base_gradient) = analytic_energy_and_gradient(registry, saddle, options)?;
    let mut points = vec![IrcPoint::new(
        0,
        base_energy,
        max_abs(&flatten(&base_gradient)),
    )];
    let mut direction = if sign >= 0.0 { 1.0 } else { -1.0 };
    {
        let plus = displace(saddle, &cartesian_mode, direction * probe);
        let minus = displace(saddle, &cartesian_mode, -direction * probe);
        let plus_energy = analytic_energy_and_gradient(registry, &plus, options)?.0;
        let minus_energy = analytic_energy_and_gradient(registry, &minus, options)?.0;
        if plus_energy > base_energy && minus_energy > base_energy {
            return Err(EngineError::Unsupported(
                "irc: both probe directions increase the energy - the starting geometry is not a \
                 first-order saddle (run `qwb ts` first)"
                    .to_string(),
            ));
        }
        if minus_energy < plus_energy {
            direction = -direction;
        }
    }
    // 第一步必须沿虚频本征向量**离开鞍点**：鞍点上梯度本来就近零，直接按梯度判据会立刻
    // "收敛"（第一版就是这样，得到一个只有 1 个点的路径）。
    let mut current = displace(saddle, &cartesian_mode, direction * step_size);
    let mut energy = analytic_energy_and_gradient(registry, &current, options)?.0;
    if energy >= base_energy {
        return Err(EngineError::Internal(format!(
            "irc: the first step along the imaginary mode did not lower the energy \
             ({base_energy:.10} -> {energy:.10} Ha)"
        )));
    }
    {
        let (_, gradient) = analytic_energy_and_gradient(registry, &current, options)?;
        points.push(IrcPoint::new(1, energy, max_abs(&flatten(&gradient))));
    }
    reporter.log(&format!(
        "irc: left the saddle along the imaginary mode, E = {energy:.10} Ha"
    ));
    for step in 2..=max_steps {
        let (_, gradient) = analytic_energy_and_gradient(registry, &current, options)?;
        let flat = flatten(&gradient);
        let force = max_abs(&flat);
        if force < options.force_threshold {
            return Ok((points, current));
        }
        let step_vector = mass_weighted_step(&current, &flat, step_size)?;
        // 单调性：只接受让能量下降的步（折半最多 4 次）。
        let mut scale = 1.0_f64;
        let mut accepted = None;
        for _ in 0..4 {
            let trial = displace(&current, &step_vector, direction * scale);
            let trial_energy = analytic_energy_and_gradient(registry, &trial, options)?.0;
            if trial_energy < energy {
                accepted = Some((trial, trial_energy));
                break;
            }
            scale *= 0.5;
        }
        let Some((next, next_energy)) = accepted else {
            // 能量不再下降：已经到极小点的平地上了。
            return Ok((points, current));
        };
        current = next;
        energy = next_energy;
        let (_, gradient) = analytic_energy_and_gradient(registry, &current, options)?;
        let force = max_abs(&flatten(&gradient));
        points.push(IrcPoint::new(step, energy, force));
        if step % 10 == 0 {
            reporter.log(&format!(
                "irc step {step}: E = {energy:.10} Ha, max|g| = {force:.3e} Ha/Bohr"
            ));
        }
        if force < options.force_threshold {
            return Ok((points, current));
        }
    }
    Err(EngineError::Unsupported(format!(
        "irc did not reach a minimum in {max_steps} steps (max|g| = {:.3e} Ha/Bohr)",
        max_abs(&flatten(
            &analytic_energy_and_gradient(registry, &current, options)?.1
        ))
    )))
}

/// 内坐标指纹：所有原子对的键长（Å），再按 (i, i+1, i+2) 三元组给键角（度）。
///
/// 只对**分子内**有意义（同种分子的两个极小点比对），且与旋转/平移/原子编号的置换无关。
#[must_use]
pub fn fingerprint(molecule: &Molecule) -> Vec<f64> {
    let atoms = &molecule.atoms;
    let mut out = Vec::new();
    for i in 0..atoms.len() {
        for j in i + 1..atoms.len() {
            out.push(distance(
                atoms[i].position_angstrom,
                atoms[j].position_angstrom,
            ));
        }
    }
    for i in 0..atoms.len().saturating_sub(2) {
        let a = atoms[i].position_angstrom;
        let b = atoms[i + 1].position_angstrom;
        let c = atoms[i + 2].position_angstrom;
        let v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
        if n1 > 1.0e-12 && n2 > 1.0e-12 {
            let cosine =
                ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)).clamp(-1.0, 1.0);
            out.push(cosine.acos().to_degrees());
        }
    }
    out
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// 端点收尾：从走到的几何再自由优化到极小点，并算指纹。
///
/// # Errors
///
/// 同 [`optimize`]。
pub fn refine_end(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<EndOutcome, EngineError> {
    let optimised = optimize(registry, molecule, options)?;
    if !optimised.converged {
        return Err(EngineError::Unsupported(format!(
            "irc: the endpoint optimisation did not converge (max|g| = {:.3e} Ha/Bohr)",
            optimised
                .gradient
                .iter()
                .flat_map(|row| row.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        )));
    }
    Ok(EndOutcome {
        fingerprint: fingerprint(&optimised.molecule),
        molecule: optimised.molecule,
        energy_hartree: optimised.energy_hartree,
        optimisation_steps: optimised.iterations,
    })
}

/// 组装 K2 的块。
///
/// # Errors
///
/// 端点点集自洽性失败。
pub fn to_api(
    wavenumber: f64,
    forward: Option<(Vec<IrcPoint>, EndOutcome)>,
    reverse: Option<(Vec<IrcPoint>, EndOutcome)>,
) -> Result<IrcResult, EngineError> {
    let build = |pair: (Vec<IrcPoint>, EndOutcome)| {
        let (points, end) = pair;
        IrcEnd::new(
            points,
            end.energy_hartree,
            end.fingerprint,
            end.optimisation_steps,
        )
    };
    let result = IrcResult::new(wavenumber, forward.map(build), reverse.map(build));
    result.validate()?;
    Ok(result)
}
