//! RS-06 · Raman 活性与（相对）强度：`∂α/∂R` 的有限差分 + 简正模投影。
//!
//! # 物理量与其约定（这一段就是「我们实现了什么」的定义）
//!
//! 极化率张量 `α_ij = 4 d_iᵀ (A+B)⁻¹ d_j` 由 `qwb_response` 的 CPHF 求解器给出（与
//! [`qwb_response::static_polarizability`] 同一套方程与同一个 `(A+B)` 算子；这里在引擎侧
//! 取对称化后的 `(α+αᵀ)/2`，理由见 [`polarizability_at`] 的注释）。单位原子单位（`bohr³`）。
//! Raman 的强度只需要它**对核位移的导数**投影到简正模上：
//!
//! ```text
//! ∂α_ij/∂Q_k = Σ_c (∂α_ij/∂x_c) · (L_kc / √m_c)
//! ```
//!
//! 其中 `c = 3·atom + axis` 走全部 `3N` 个笛卡尔坐标，`L_kc/√m_c` 正是
//! [`crate::hessian::NormalMode::displacement`] 存的东西（质量加权本征向量摊回笛卡尔）。
//! `∂α_ij/∂x_c` 由**中心差分**得到（`(α(R+δ) − α(R−δ)) / 2δ`，`δ` 单位 Bohr），
//! 所以 `∂α/∂Q` 的自然单位是 `bohr²·amu^{-1/2}`；报告前换算到 `Å²·amu^{-1/2}`
//! （乘 `Å_per_Bohr²`，**只乘一次** —— 这是本项目历史上最容易多乘/少乘的地方）。
//!
//! 然后按 Placzek 的**常规定义**给每个模的**活性**（Raman activity，Gaussian/ORCA 也报这个数）：
//!
//! ```text
//! ᾱ'_k = (1/3)(α'_xx + α'_yy + α'_zz)
//! γ'_k² = ½[(α'_xx−α'_yy)² + (α'_yy−α'_zz)² + (α'_zz−α'_xx)²] + 3(α'_xy² + α'_xz² + α'_yz²)
//! A_k   = 45 ᾱ'_k² + 7 γ'_k²          [Å⁴/amu]
//! ```
//!
//! 45 与 7 是 Placzek 的**各向同性/各向异性权重**（来源：Placzek 1934 的经典结果，也是
//! Gaussian `Freq=Raman` 与绝大多数代码用的同一组系数）。张量的偏导是对称张量的偏导，
//! 所以三个非对角分量取对称化后的值（`(α'_xy+α'_yx)/2`）—— 这是差分噪声的抑制，不改公式。
//!
//! **强度**（可选，报告里给的是**相对**强度）用同一个 Placzek 活性乘上散射的
//! `ν⁴ / (ν (1 − e^{-hν/kT}))` 因子：
//!
//! ```text
//! I_k ∝ A_k · (ν̃_0 − ν̃_k)⁴ / [ ν̃_k (1 − exp(−h c ν̃_k / k_B T)) ]
//! ```
//!
//! `ν̃_0` 是**激发光的波数**（调用方给，报告里同时落盘；532 nm = 18797.2 cm⁻¹、
//! 1064 nm = 9398.6 cm⁻¹ 是两条常见线），`T` 是温度（缺省 298.15 K）。这个式子是
//! **相对**强度（任意单位），报告里归一化到「最强模 = 100」；它的 4 次幂因子意味着
//! 「相对强度」与「活性」是**两个不同的数**，不能混着比。绝对微分散射截面还要乘上
//! `π²/ε₀² · h/(8π²cν̃_k)` 这类常数，本模块**不给**绝对截面（那需要一套 SI 常数，
//! 而 §6 的判据只要相对强度排序；标度因子按判据要求单列，见 `docs` 的账本）。
//!
//! # 与 IR 的关系（为什么这里能复用 `hessian` 的约定）
//!
//! [`crate::hessian::dipole_derivatives`] 用**同一个**步长、同一条 `Å → Bohr` 路径算
//! `∂μ/∂x`，再用**同一个** `NormalMode::displacement` 投影。所以本模块的结构是那种做法的
//! 逐行类比：把「偶极」换成「极化率张量」，把「IR 的 42.2561·|∂μ/∂Q|²」换成
//! 「45ᾱ'² + 7γ'²」。
//!
//! # 已知边界
//!
//! * 只做**闭壳层** RHF/RKS 参考（`qwb_response::static_polarizability` 的适用面）；
//! * 只有**数值** `∂α/∂R`（RS-05 的解析二阶导仍是 blocked，所以这里也只能走差分）；
//! * 虚频（鞍点）一律由调用方拒绝（同 `freq` 的判据），因为「取最大的 `3N−6` 个模当振动」
//!   这条规则只在极小点成立。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::Registry;
use qwb_response::cphf::CphfOptions;
use qwb_response::mo_integrals::build_cphf_integrals;
use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

use crate::hessian::{HESSIAN_DELTA_BOHR, HESSIAN_METHOD, HessianResult};

/// `Å per Bohr`（与 [`crate::hessian`] 里的常量逐位一致；那个名字在 hessian.rs 里叫
/// `ANGSTROM_PER_BOHR`，但语义就是「1 Bohr = 多少 Å」）。
pub(crate) const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

/// `1 bohr²` 折算成 `Å²`。
pub(crate) const ANGSTROM2_PER_BOHR2: f64 = ANGSTROM_PER_BOHR * ANGSTROM_PER_BOHR;

/// 活性单位的冻结字符串（进 `result.json` 的 `raman.activity_unit`）。
pub const RAMAN_ACTIVITY_UNIT: &str = "angstrom^4/amu";

/// 活性的来源（进 `result.json` 的 `raman.method`）。
///
/// 「有限差分极化率 + 数值 Hessian 的模式」这件事必须能被读结果的人看见 —— 它是**数值**
/// 导数，不是解析导数。
pub const RAMAN_METHOD: &str = "finite-difference-polarizability";

/// Placzek 的各向同性权重（`45 ᾱ'²`）。
pub const PLACZEK_ISOTROPIC_WEIGHT: f64 = 45.0;

/// Placzek 的各向异性权重（`7 γ'²`）。
pub const PLACZEK_ANISOTROPY_WEIGHT: f64 = 7.0;

/// `h c / k_B`，单位 `cm·K`（CODATA 2018：1.438 776 877 cm·K）。
pub const HC_OVER_KB_CM_K: f64 = 1.438_776_87;

/// 532 nm（倍频 Nd:YAG）对应的波数（cm⁻¹）：`1e7 / 532`。报告缺省用它。
pub const DEFAULT_EXCITATION_WAVELENGTH_NM: f64 = 532.0;

/// 相对强度的缺省温度（K）。
pub const DEFAULT_TEMPERATURE_K: f64 = 298.15;

/// 一个模的 Raman 分解（活性 + 两个 Placzek 不变量）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ModeRamanActivity {
    /// `∂α/∂Q_k` 的各向同性部分 `ᾱ'_k`，单位 `Å²·amu^{-1/2}`。
    pub isotropic_derivative: f64,
    /// `∂α/∂Q_k` 的各向异性 `γ'_k`（不是平方），单位 `Å²·amu^{-1/2}`。
    pub anisotropy_derivative: f64,
    /// 活性 `45 ᾱ'² + 7 γ'²`，单位 `Å⁴/amu`。
    pub activity_a4_per_amu: f64,
}

impl ModeRamanActivity {
    /// 由 `∂α/∂Q_k` 的 3×3 张量（单位 `Å²·amu^{-1/2}`）算两个不变量与活性。
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn from_derivative_tensor(tensor: [[f64; 3]; 3]) -> Self {
        // 对称化：α 本身对称，所以 ∂α/∂Q 也对称；差分噪声可能让 yx/xy 差 1e-12 量级，
        // 取平均不改变公式，只是让「哪一个是真值」不再依赖下标顺序。
        let mut t = tensor;
        for i in 0..3 {
            for j in i + 1..3 {
                let average = 0.5 * (t[i][j] + t[j][i]);
                t[i][j] = average;
                t[j][i] = average;
            }
        }
        let isotropic = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let anisotropy_squared = 0.5
            * ((t[0][0] - t[1][1]).powi(2)
                + (t[1][1] - t[2][2]).powi(2)
                + (t[2][2] - t[0][0]).powi(2))
            + 3.0 * (t[0][1].powi(2) + t[0][2].powi(2) + t[1][2].powi(2));
        Self {
            isotropic_derivative: isotropic,
            anisotropy_derivative: anisotropy_squared.max(0.0).sqrt(),
            activity_a4_per_amu: PLACZEK_ISOTROPIC_WEIGHT * isotropic * isotropic
                + PLACZEK_ANISOTROPY_WEIGHT * anisotropy_squared,
        }
    }
}

/// 闭壳层电子数（奇电子/非正 → 错误）。与 `qwb-engine` 里的同名私有助手同一套约定。
fn closed_shell_electrons(molecule: &Molecule) -> Result<usize, EngineError> {
    let mut electrons = -i64::from(molecule.charge);
    for atom in &molecule.atoms {
        let z = qwb_registry::elements::atomic_number(&atom.symbol).ok_or_else(|| {
            EngineError::Data(format!("Raman: unknown element `{}`", atom.symbol))
        })?;
        electrons += i64::from(z);
    }
    if electrons <= 0 || electrons % 2 != 0 {
        return Err(EngineError::InvalidInput(format!(
            "Raman activities need a closed-shell reference with an even, positive electron \
             count, got {electrons}"
        )));
    }
    usize::try_from(electrons)
        .map_err(|_| EngineError::Internal("electron count does not fit usize".to_string()))
}

/// 某个几何下的静态极化率张量（原子单位，`bohr³`）。
///
/// 与 [`crate::hessian::dipole_derivatives`] 的 `dipole_at` 同一个形状：**每个位移重跑一次
/// SCF**（几何变了，基组与 2e 张量都得重建），再解一次 CPHF 拿 `α = 4 dᵀ(A+B)⁻¹d`。
///
/// # Errors
///
/// SCF/CPHF 不收敛、基组或数据缺失、电子数非偶 → 见 [`EngineError`]。
pub fn polarizability_at(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional: &str,
    grid_name: &str,
) -> Result<[[f64; 3]; 3], EngineError> {
    let n_occupied = closed_shell_electrons(molecule)? / 2;
    let mut scf_options = RksOptions::default();
    scf_options.max_iter = 400;
    let max_iter = scf_options.max_iter;
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(basis_name)?, molecule)?;
    let result = if functional.is_empty() {
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let level = registry.get_grid(grid_name)?;
        let grid = qwb_grid::molecular_grid(level, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        rks_scf(&basis, molecule, &tensor, &grid, &[], scf_options)?
    } else {
        rks_from_data(
            registry,
            molecule,
            basis_name,
            functional,
            grid_name,
            scf_options,
        )?
    };
    if !result.scf_converged {
        return Err(EngineError::ScfNotConverged {
            iters: result.iterations,
            max: max_iter,
        });
    }
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let dipole_ao = qwb_integrals::dipole::dipole_integrals(&basis, molecule)?;
    let integrals = build_cphf_integrals(&tensor, &result.coefficients, &dipole_ao, n_occupied)?;
    let orbital_energies: Vec<f64> = (0..result.mo_energies_hartree.len())
        .map(|index| result.mo_energies_hartree.get(index))
        .collect();
    // **为什么比 RS-01 的缺省更严**：`∂α/∂R` 是两个相邻几何的 α 相减再除以 `2δ`，所以 α
    // 的求解残差会被放大 `1/(2δ) = 50` 倍；缺省的 `1e-8` 残差在差分里会变成 1e-6 量级的
    // 导数噪声。这里收到 `1e-10`。
    let mut cphf_options = CphfOptions::default();
    cphf_options.tolerance = 1.0e-10;
    // **为什么在这里自己组装、而不是调 `static_polarizability`**：那个函数的最后一步是
    // `Polarizability::validate()`，它要求张量逐元素对称到 `1e-10`（绝对值）。位移几何上
    // 会出现**真值很小**的元素（水/STO-3G 的 α_yz 在位移后是 3.3e-6），而 CPHF 迭代解的
    // 不对称量是 2.4e-10 —— 相对 1e-4，绝对 2.4e-10，于是那条检查在位**移几何上偶发失败**。
    // 精确的 α 张量本来是对称的，(A+B) 与偶极右端都是对称的，不对称纯粹是迭代解的残差；
    // 有限差分需要的正是这个对称量，所以这里取 `(α + αᵀ)/2`。**没有放宽任何检查**：
    // `Polarizability::validate` 一个字节没动，它检查的是「RS-01 的静态极化率」，而这里
    // 是「差分用的对称化响应」。实测差分量级：δ = 1e-2 时活性与 δ = 2e-3 的差 < 1%
    // （`tests/rs06_raman.rs::the_finite_difference_step_is_converged_at_the_hessian_convention`）。
    let rhs: Vec<Vec<f64>> = (0..3).map(|axis| integrals.dipole[axis].clone()).collect();
    let outcome = qwb_response::solve_cphf(&integrals, &orbital_energies, &rhs, cphf_options)?;
    if !outcome.converged {
        return Err(EngineError::ScfNotConverged {
            iters: outcome.iterations.iter().copied().max().unwrap_or(0),
            max: cphf_options.max_iter,
        });
    }
    let mut tensor = [[0.0_f64; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for (p, value) in integrals.dipole[i].iter().enumerate() {
                sum += value * outcome.solutions[j][p];
            }
            tensor[i][j] = 4.0 * sum;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in i + 1..3 {
            let average = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = average;
            tensor[j][i] = average;
        }
    }
    Ok(tensor)
}

/// `∂α_ij/∂x_c`（单位原子单位：`bohr²`），形状 `[3*atom + axis] -> 3×3 张量`。
///
/// 中心差分，步长 [`HESSIAN_DELTA_BOHR`]（**与 `dipole_derivatives` 完全同一个约定**：
/// 位移量是 `δ_bohr · Å_per_Bohr` 加在 `position_angstrom` 上，所以 Bohr 只在
/// `Å → Bohr` 的那一处出现一次）。代价是 `6N` 次 SCF + CPHF。
///
/// # Errors
///
/// 任何一次位移的单点/CPHF 失败。
pub fn polarizability_derivatives(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional: &str,
    grid_name: &str,
) -> Result<Vec<[[f64; 3]; 3]>, EngineError> {
    polarizability_derivatives_with_delta(
        registry,
        molecule,
        basis_name,
        functional,
        grid_name,
        HESSIAN_DELTA_BOHR,
    )
}

/// 与 [`polarizability_derivatives`] 相同，但显式给差分步长（单位 **Bohr**）。
///
/// 存在的理由与 [`crate::hessian::numerical_hessian_with_delta`] 一样：差分步长的收敛性
/// 要用实测说话（`tests/rs06_raman.rs` 里有一条只量不判的步长扫描）。
///
/// # Errors
///
/// 同 [`polarizability_derivatives`]。
pub fn polarizability_derivatives_with_delta(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional: &str,
    grid_name: &str,
    delta_bohr: f64,
) -> Result<Vec<[[f64; 3]; 3]>, EngineError> {
    let dof = 3 * molecule.atoms.len();
    let mut out = vec![[[0.0_f64; 3]; 3]; dof];
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            plus.atoms[atom].position_angstrom[axis] += delta_bohr * ANGSTROM_PER_BOHR;
            minus.atoms[atom].position_angstrom[axis] -= delta_bohr * ANGSTROM_PER_BOHR;
            let alpha_plus = polarizability_at(registry, &plus, basis_name, functional, grid_name)?;
            let alpha_minus =
                polarizability_at(registry, &minus, basis_name, functional, grid_name)?;
            for i in 0..3 {
                for j in 0..3 {
                    out[3 * atom + axis][i][j] =
                        (alpha_plus[i][j] - alpha_minus[i][j]) / (2.0 * delta_bohr);
                }
            }
        }
    }
    Ok(out)
}

/// 每个简正模的 Raman 活性（`Å⁴/amu`），顺序与 [`HessianResult::modes`] 一致；
/// 平移/转动模给 0（与 [`crate::hessian::ir_intensities`] 的处置逐条相同）。
///
/// `derivatives` 是 [`polarizability_derivatives`] 的输出（原子单位，`bohr²`）。
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn raman_activities(
    result: &HessianResult,
    derivatives: &[[[f64; 3]; 3]],
) -> Vec<ModeRamanActivity> {
    let dof = result.degrees_of_freedom;
    let mut out =
        vec![ModeRamanActivity::from_derivative_tensor([[0.0; 3]; 3]); result.modes.len()];
    // 只有最大的 n_vib 个模是振动模（与 `ir_intensities` 同一个 `start`）。
    let start = result.modes.len().saturating_sub(result.vibrational_modes);
    for (index, mode) in result.modes.iter().enumerate().skip(start) {
        let mut tensor = [[0.0_f64; 3]; 3];
        for coordinate in 0..dof {
            let atom = coordinate / 3;
            let axis = coordinate % 3;
            let weight = mode.displacement[atom][axis];
            for i in 0..3 {
                for j in 0..3 {
                    tensor[i][j] += derivatives[coordinate][i][j] * weight;
                }
            }
        }
        // bohr²/√amu → Å²/√amu：**只乘一次** Å_per_Bohr²。
        for row in &mut tensor {
            for value in row.iter_mut() {
                *value *= ANGSTROM2_PER_BOHR2;
            }
        }
        out[index] = ModeRamanActivity::from_derivative_tensor(tensor);
    }
    out
}

/// Raman 的**相对**强度（任意单位）：`A_k (ν̃_0−ν̃_k)⁴ / [ν̃_k (1 − e^{−hcν̃_k/kT})]`。
///
/// 与 `activities`/`wavenumbers` 一一对应；`ν̃_k <= 0`（虚频）或 `ν̃_0 <= ν̃_k`（激发线
/// 低于 Stokes 位移，物理上不可能）时给 0，而不是给一个负数或 NaN。
///
/// # Errors
///
/// `activities` 与 `wavenumbers` 长度不一致、或参数非有限/非正。
pub fn raman_intensities(
    activities: &[f64],
    wavenumbers: &[f64],
    excitation_wavenumber_cm: f64,
    temperature_k: f64,
) -> Result<Vec<f64>, EngineError> {
    if activities.len() != wavenumbers.len() {
        return Err(EngineError::Internal(format!(
            "raman: {} activities but {} wavenumbers",
            activities.len(),
            wavenumbers.len()
        )));
    }
    if !excitation_wavenumber_cm.is_finite() || excitation_wavenumber_cm <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "raman: excitation wavenumber must be finite and positive, got {excitation_wavenumber_cm}"
        )));
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "raman: temperature must be finite and positive, got {temperature_k}"
        )));
    }
    let thermal = HC_OVER_KB_CM_K / temperature_k;
    let mut out = Vec::with_capacity(activities.len());
    for (activity, wavenumber) in activities.iter().zip(wavenumbers) {
        let stokes = excitation_wavenumber_cm - wavenumber;
        if *wavenumber <= 0.0 || stokes <= 0.0 || !activity.is_finite() {
            out.push(0.0);
            continue;
        }
        let boltzmann = 1.0 - (-thermal * wavenumber).exp();
        out.push(activity * stokes.powi(4) / (wavenumber * boltzmann));
    }
    Ok(out)
}

/// 把一组强度归一化到「最强 = 100」。报告里的 `intensities` 就是这个数。
#[must_use]
pub fn relative_to_strongest(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|value| 100.0 * value / max).collect()
}

/// 把波长（nm）换成波数（cm⁻¹）：`1e7 / λ_nm`。
///
/// # Errors
///
/// 波长非有限或非正。
pub fn wavenumber_from_wavelength_nm(wavelength_nm: f64) -> Result<f64, EngineError> {
    if !wavelength_nm.is_finite() || wavelength_nm <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "raman: excitation wavelength must be finite and positive, got {wavelength_nm}"
        )));
    }
    Ok(1.0e7 / wavelength_nm)
}

/// 频率来源的字符串（与 [`HESSIAN_METHOD`] 一起写进报告，说明模式是从哪来的）。
#[must_use]
pub fn mode_source() -> &'static str {
    HESSIAN_METHOD
}

#[cfg(test)]
mod tests {
    use super::{
        ANGSTROM2_PER_BOHR2, ModeRamanActivity, PLACZEK_ANISOTROPY_WEIGHT,
        PLACZEK_ISOTROPIC_WEIGHT, raman_intensities, relative_to_strongest,
        wavenumber_from_wavelength_nm,
    };

    #[test]
    fn isotropic_derivative_alone_gives_the_placzek_activity() {
        // ∂α/∂Q = 2·I（Å²/√amu）：ᾱ' = 2、γ' = 0 ⇒ A = 45·4 = 180。
        let tensor = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let activity = ModeRamanActivity::from_derivative_tensor(tensor);
        assert!((activity.isotropic_derivative - 2.0).abs() < 1e-14);
        assert!(activity.anisotropy_derivative < 1e-14);
        assert!(
            (activity.activity_a4_per_amu - PLACZEK_ISOTROPIC_WEIGHT * 4.0).abs() < 1e-12,
            "{}",
            activity.activity_a4_per_amu
        );
    }

    #[test]
    fn a_pure_anisotropy_derivative_uses_the_seven_weight() {
        // ∂α/∂Q = diag(1, −1, 0)：ᾱ' = 0，γ'² = ½(4 + 1 + 1) = 3 ⇒ A = 7·3 = 21。
        let tensor = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]];
        let activity = ModeRamanActivity::from_derivative_tensor(tensor);
        assert!(activity.isotropic_derivative.abs() < 1e-15);
        assert!((activity.anisotropy_derivative - 3.0_f64.sqrt()).abs() < 1e-12);
        assert!((activity.activity_a4_per_amu - PLACZEK_ANISOTROPY_WEIGHT * 3.0).abs() < 1e-12);
    }

    #[test]
    fn off_diagonal_terms_are_symmetrised_before_the_anisotropy() {
        // xy = 1, yx = 0 → 对称化后 0.5：γ'² = 3·0.25 = 0.75，A = 7·0.75 = 5.25。
        let tensor = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let activity = ModeRamanActivity::from_derivative_tensor(tensor);
        assert!(activity.activity_a4_per_amu > 0.0);
        let expected = PLACZEK_ANISOTROPY_WEIGHT * 0.75;
        assert!(
            (activity.activity_a4_per_amu - expected).abs() < 1e-12,
            "{} vs {expected}",
            activity.activity_a4_per_amu
        );
    }

    #[test]
    fn bohr_squared_is_converted_once() {
        // 0.529177210903² = 0.2800285205390781；这条钉住「Å→Bohr 只乘一次」。
        assert!(
            (ANGSTROM2_PER_BOHR2 - 0.280_028_520_539_078_1).abs() < 1e-12,
            "{ANGSTROM2_PER_BOHR2}"
        );
    }

    #[test]
    fn intensities_follow_the_fourth_power_and_the_boltzmann_factor() {
        let activities = [1.0, 1.0];
        let wavenumbers = [1000.0, 2000.0];
        let intensities =
            raman_intensities(&activities, &wavenumbers, 18797.2, 298.15).expect("intensities");
        // 手算：(17797.2)⁴/(1000·(1−e^{-0.004825·1000})) 与 (16797.2)⁴/(2000·(1−e^{-0.004825·2000}))
        let thermal = super::HC_OVER_KB_CM_K / 298.15;
        let expected_low = 17797.2_f64.powi(4) / (1000.0 * (1.0 - (-thermal * 1000.0).exp()));
        let expected_high = 16797.2_f64.powi(4) / (2000.0 * (1.0 - (-thermal * 2000.0).exp()));
        assert!(
            (intensities[0] / expected_low - 1.0).abs() < 1e-12,
            "{} vs {expected_low}",
            intensities[0]
        );
        assert!(
            (intensities[1] / expected_high - 1.0).abs() < 1e-12,
            "{} vs {expected_high}",
            intensities[1]
        );
        // 4 次幂 ⇒ 低波数那支更强（同样的活性）。
        assert!(intensities[0] > intensities[1]);
    }

    #[test]
    fn imaginary_modes_and_impossible_stokes_shifts_give_zero() {
        let intensities =
            raman_intensities(&[5.0, 5.0, 5.0], &[-200.0, 0.0, 20000.0], 18797.2, 298.15)
                .expect("intensities");
        assert_eq!(intensities, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn bad_parameters_are_rejected() {
        assert!(raman_intensities(&[1.0], &[1.0, 2.0], 18797.2, 298.15).is_err());
        assert!(raman_intensities(&[1.0], &[1.0], 0.0, 298.15).is_err());
        assert!(raman_intensities(&[1.0], &[1.0], 18797.2, -1.0).is_err());
        assert!(wavenumber_from_wavelength_nm(0.0).is_err());
        assert!(
            (wavenumber_from_wavelength_nm(532.0).expect("532 nm") - 18_796.992_481_2).abs() < 1e-6
        );
    }

    #[test]
    fn relative_intensities_are_normalised_to_the_strongest_mode() {
        let values = relative_to_strongest(&[2.0, 4.0, 1.0]);
        assert_eq!(values, vec![50.0, 100.0, 25.0]);
        assert_eq!(relative_to_strongest(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
