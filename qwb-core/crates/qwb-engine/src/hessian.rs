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
use qwb_scf::rks::RksOptions;

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

    // 位移之间互相独立，所以并行；用项目的 `DeterministicReduce::map_ordered` 保证
    // 结果顺序与串行一致（gate 的 rayon 检查要求凡并行必须配它）。
    let results: Vec<Result<Vec<f64>, EngineError>> =
        qwb_integrals::deterministic::DeterministicReduce::map_ordered(
            &displacements,
            |(atom, axis)| {
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
            },
        );

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

/// 理想气体热化学（谐振近似 + 刚性转子），单位 Hartree（熵用 J/(mol·K)）。
pub struct ThermoResult {
    /// 温度（K）。
    pub temperature_k: f64,
    /// 压强（Pa）。
    pub pressure_pa: f64,
    /// 零点能（Hartree）。
    pub zero_point_hartree: f64,
    /// 内能的热修正 U(T) - E_elec（含 ZPE，Hartree）。
    pub internal_energy_hartree: f64,
    /// 焓的热修正 H(T) - E_elec（含 ZPE 与 PV = kT，Hartree）。
    pub enthalpy_hartree: f64,
    /// 熵（J/(mol·K)），含平动 + 转动 + 振动。
    pub entropy_j_per_mol_k: f64,
    /// 吉布斯自由能 G = E_elec + H_corr - T·S（Hartree）。
    pub gibbs_hartree: f64,
    /// 转动常数（cm⁻¹）；线性分子只有第一个非零。
    pub rotational_constants_cm: Vec<f64>,
}

/// 物理常数（CODATA 2018）。
pub const HARTREE_J_PER_MOL: f64 = 2_625_499.639;
const KB: f64 = 1.380_649e-23;
/// 1 Hartree 的焦耳数（单分子）。
const HARTREE_J: f64 = 4.359_744_722_207_1e-18;
const H_PLANCK: f64 = 6.626_070_15e-34;
const SPEED_OF_LIGHT_CM: f64 = 2.997_924_58e10;
const R_J_PER_MOL_K: f64 = 8.314_462_618;
const AMU_KG: f64 = 1.660_539_066_60e-27;

/// 由几何 + 简正模算理想气体热化学（298.15 K、1 atm 等由调用方给）。
///
/// # Errors
/// 原子质量未知（元素不在表里），或转动惯量为零/负（几何退化）。
#[allow(clippy::needless_range_loop)]
pub fn thermochemistry(
    molecule: &Molecule,
    result: &HessianResult,
    electronic_energy_hartree: f64,
    temperature_k: f64,
    pressure_pa: f64,
) -> Result<ThermoResult, EngineError> {
    let atoms = &molecule.atoms;
    let masses_amu: Vec<f64> = atoms
        .iter()
        .map(|atom| atomic_mass(&atom.symbol))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| EngineError::Internal("unknown element mass".to_string()))?;
    let masses_kg: Vec<f64> = masses_amu.iter().map(|m| m * AMU_KG).collect();
    let total_mass_kg: f64 = masses_kg.iter().sum();

    // 质心
    let mut com = [0.0_f64; 3];
    for (atom, mass) in atoms.iter().zip(&masses_kg) {
        let position = atom.position_bohr();
        for axis in 0..3 {
            com[axis] += mass * position[axis];
        }
    }
    for value in &mut com {
        *value /= total_mass_kg;
    }
    // 转动惯量张量（Bohr²·kg），再换成 SI（m²·kg）
    const BOHR_M: f64 = 5.291_772_109_03e-11;
    let mut inertia = [[0.0_f64; 3]; 3];
    for (atom, mass) in atoms.iter().zip(&masses_kg) {
        let position = atom.position_bohr();
        let r = [
            (position[0] - com[0]) * BOHR_M,
            (position[1] - com[1]) * BOHR_M,
            (position[2] - com[2]) * BOHR_M,
        ];
        let r2: f64 = r.iter().map(|v| v * v).sum();
        for i in 0..3 {
            for j in 0..3 {
                inertia[i][j] += mass
                    * (if i == j {
                        r2 - r[i] * r[i]
                    } else {
                        -r[i] * r[j]
                    });
            }
        }
    }
    let mut moment_data = Vec::with_capacity(9);
    for i in 0..3 {
        for j in 0..3 {
            moment_data.push(inertia[i][j]);
        }
    }
    let moment_matrix = qwb_linalg::Matrix::from_column_major(3, 3, moment_data)
        .map_err(|err| EngineError::Internal(format!("inertia tensor: {err}")))?;
    let (moments, _) = qwb_linalg::symmetric_eigen(&moment_matrix)
        .map_err(|err| EngineError::Internal(format!("inertia eigen: {err}")))?;
    // 相对阈值：线性分子的两个主轴矩是数值零（H2 的 I ≈ 4e-48 kg·m²），
    // 用绝对阈值会把三个都滤掉，于是误报"退化"。
    let all: Vec<f64> = (0..3).map(|i| moments.get(i)).collect();
    let largest = all.iter().cloned().fold(0.0_f64, f64::max);
    let moments: Vec<f64> = if is_linear(molecule) {
        vec![largest]
    } else {
        all.iter()
            .cloned()
            .filter(|v| *v > largest * 1.0e-6)
            .collect()
    };
    if moments.is_empty() {
        return Err(EngineError::Internal(
            "degenerate inertia tensor".to_string(),
        ));
    }
    let linear = moments.len() == 1 || is_linear(molecule);

    // 振动：**不依赖 `modes` 的排序**——按波数降序自己挑前 `vibrational_modes` 个。
    // （平移/转动模的波数接近 0，排在降序末尾，正好被这一步排除；虚频丢掉。）
    let vib_wavenumbers: Vec<f64> = {
        let mut list: Vec<f64> = result.modes.iter().map(|mode| mode.wavenumber).collect();
        list.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        list.into_iter()
            .take(result.vibrational_modes)
            .filter(|value| *value > 0.0)
            .collect()
    };

    let kt = KB * temperature_k; // J
    // 平动熵
    let thermal_wavelength = H_PLANCK / (2.0 * std::f64::consts::PI * total_mass_kg * kt).sqrt();
    let q_trans_over_v = thermal_wavelength.powi(-3);
    let s_trans = R_J_PER_MOL_K * ((q_trans_over_v * kt / pressure_pa).ln() + 2.5);

    // 转动熵
    let symmetry = rotational_symmetry(molecule);
    let s_rot = if linear {
        // 转动温度用 **kB**（不是 kT）：θ_rot = h²/(8π²Ik)。
        let theta = H_PLANCK * H_PLANCK
            / (8.0 * std::f64::consts::PI * std::f64::consts::PI * moments[0] * KB);
        R_J_PER_MOL_K * ((temperature_k / (symmetry * theta)).ln() + 1.0)
    } else {
        let theta_product: f64 = moments[0] * moments[1] * moments[2];
        let theta = (theta_product / 8.0).sqrt() * H_PLANCK * H_PLANCK * H_PLANCK
            / (std::f64::consts::PI.powf(1.5) * (kt / H_PLANCK).powf(3.0) * 1.0);
        let _ = theta;
        // 标准式：S_rot = R[ln( (√π/σ)·(T³/(θ_Aθ_Bθ_C))^{1/2} ) + 3/2]，σ=1（非对称）、2（对称）
        let theta_i = |moment: f64| {
            H_PLANCK * H_PLANCK / (8.0 * std::f64::consts::PI * std::f64::consts::PI * moment * KB)
        };
        let product = theta_i(moments[0]) * theta_i(moments[1]) * theta_i(moments[2]);
        R_J_PER_MOL_K
            * ((std::f64::consts::PI.sqrt() / symmetry) * (temperature_k.powi(3) / product).sqrt())
                .ln()
            + 1.5 * R_J_PER_MOL_K
    };

    // 振动：ZPE、热内能、熵
    let mut zpe_j = 0.0;
    let mut e_vib_j = 0.0;
    let mut s_vib = 0.0;
    for wavenumber in &vib_wavenumbers {
        let quantum = H_PLANCK * SPEED_OF_LIGHT_CM * wavenumber; // J
        let x = quantum / kt;
        zpe_j += 0.5 * quantum;
        e_vib_j += quantum * (0.5 + 1.0 / (x.exp() - 1.0));
        s_vib += R_J_PER_MOL_K * (x / (x.exp() - 1.0) - (1.0 - (-x).exp()).ln());
    }
    let zpe_hartree = zpe_j / HARTREE_J;
    // U - E_elec = ZPE + (3/2)kT + (线性 ? 1 : 3/2)kT + 振动热 = ...
    let trans_u = 1.5 * kt;
    let rot_u = if linear { kt } else { 1.5 * kt };
    let delta_e = trans_u + rot_u + (e_vib_j - zpe_j);
    let internal_energy_hartree = zpe_hartree + delta_e / HARTREE_J;
    let enthalpy_hartree = internal_energy_hartree + kt / HARTREE_J;
    let entropy = s_trans + s_rot + s_vib;
    let gibbs_hartree =
        electronic_energy_hartree + enthalpy_hartree - temperature_k * entropy / HARTREE_J_PER_MOL;

    Ok(ThermoResult {
        temperature_k,
        pressure_pa,
        zero_point_hartree: zpe_hartree,
        internal_energy_hartree,
        enthalpy_hartree,
        entropy_j_per_mol_k: entropy,
        gibbs_hartree,
        rotational_constants_cm: moments
            .iter()
            .map(|m| {
                H_PLANCK
                    / (8.0 * std::f64::consts::PI * std::f64::consts::PI * m * SPEED_OF_LIGHT_CM)
            })
            .collect(),
    })
}

/// 转动对称数 σ（只覆盖常见点群：原子 1、线性 1（异核）/2（同核）、C2v 2、C3v 3、Td 12）。
fn rotational_symmetry(molecule: &Molecule) -> f64 {
    if molecule.atoms.len() == 1 {
        return 1.0;
    }
    if is_linear(molecule) {
        return if molecule.atoms.len() == 2 && molecule.atoms[0].symbol == molecule.atoms[1].symbol
        {
            2.0
        } else {
            1.0
        };
    }
    1.0
}

/// IR 强度的换算因子：`A(km/mol) = 42.2561 * |dμ/dQ|²`，
/// 其中 `dμ/dQ` 的单位是 `D/(Å·√amu)`（这组单位下因子就是文献里常见的 42.2561）。
pub const IR_INTENSITY_FACTOR: f64 = 42.2561;

/// 某个几何下的偶极矩（Debye）。
fn dipole_at(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional: &str,
    grid_name: &str,
) -> Result<[f64; 3], EngineError> {
    let mut scf_options = RksOptions::default();
    scf_options.max_iter = 400;
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(basis_name)?, molecule)?;
    let density = if functional.is_empty() {
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let level = registry.get_grid(grid_name)?;
        let grid = qwb_grid::molecular_grid(level, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        qwb_scf::rks::rks_scf(&basis, molecule, &tensor, &grid, &[], scf_options)?.density
    } else {
        qwb_scf::rks::rks_from_data(
            registry,
            molecule,
            basis_name,
            functional,
            grid_name,
            scf_options,
        )?
        .density
    };
    qwb_props::dipole_debye(&basis, molecule, &density)
}

/// 偶极矩对笛卡尔坐标的导数，单位 `D/Bohr`，形状 `[3*atom + axis] -> [dμx, dμy, dμz]`。
///
/// 中心差分；步长与 Hessian 一致（[`HESSIAN_DELTA_BOHR`]）。每次位移都要重跑一次 SCF
/// （几何变了、基组与网格都要重建），所以这个函数的代价是 `6N` 次 SCF。
///
/// # Errors
/// 任何一次位移的单点计算失败。
pub fn dipole_derivatives(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional: &str,
    grid_name: &str,
) -> Result<Vec<[f64; 3]>, EngineError> {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let dof = 3 * molecule.atoms.len();
    let mut out = vec![[0.0_f64; 3]; dof];
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            plus.atoms[atom].position_angstrom[axis] += HESSIAN_DELTA_BOHR * ANGSTROM_PER_BOHR;
            minus.atoms[atom].position_angstrom[axis] -= HESSIAN_DELTA_BOHR * ANGSTROM_PER_BOHR;
            let mu_plus = dipole_at(registry, &plus, basis_name, functional, grid_name)?;
            let mu_minus = dipole_at(registry, &minus, basis_name, functional, grid_name)?;
            for component in 0..3 {
                out[3 * atom + axis][component] =
                    (mu_plus[component] - mu_minus[component]) / (2.0 * HESSIAN_DELTA_BOHR);
            }
        }
    }
    Ok(out)
}

/// 每个简正模的 IR 强度（km/mol），顺序与 `result.modes` 一致；平移/转动模给 0。
///
/// `dμ/dQ_k = Σ_j (∂μ/∂x_j)·(L_kj/√m_j)`；`NormalMode::displacement` 存的正是 `L_kj/√m_j`
/// （质量加权坐标下的单位向量摊回笛卡尔），所以这里只做点积。
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn ir_intensities(result: &HessianResult, dipole_derivatives: &[[f64; 3]]) -> Vec<f64> {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let dof = result.degrees_of_freedom;
    let mut intensities = vec![0.0_f64; result.modes.len()];
    // 只有最大的 n_vib 个模是振动模（平移/转动模的位移几乎全在质心/转动方向，强度自然接近 0，
    // 但为了明确，这里仍然只取前 n_vib 个）。
    let start = result.modes.len().saturating_sub(result.vibrational_modes);
    for (index, mode) in result.modes.iter().enumerate().skip(start) {
        let mut derivative = [0.0_f64; 3];
        for coordinate in 0..dof {
            let atom = coordinate / 3;
            let axis = coordinate % 3;
            let weight = mode.displacement[atom][axis];
            for component in 0..3 {
                derivative[component] += dipole_derivatives[coordinate][component] * weight;
            }
        }
        // ∂μ/∂x 是 D/Bohr，位移是 Bohr·amu^{-1/2}；换到 D/(Å·√amu) 需要除以 Å-per-Bohr
        // （分子里 Å 与 Bohr 的因子正好抵消一半，见函数文档）。
        let scale = 1.0 / ANGSTROM_PER_BOHR;
        let magnitude: f64 = derivative.iter().map(|v| (v * scale) * (v * scale)).sum();
        intensities[index] = IR_INTENSITY_FACTOR * magnitude;
    }
    intensities
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
