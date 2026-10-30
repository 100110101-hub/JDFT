//! 数值梯度（G-01 的前置，也是 T3 的基础）。
//!
//! 解析梯度的验收是「与**数值**梯度最大分量差 < 1e-6 Ha/Bohr」，所以先把数值梯度做成
//! 可靠的一等公民：
//!
//! * 中心差分 `dE/dR_a[k] = (E(R+δ) − E(R−δ)) / (2δ)`，`δ` 以 Bohr 计，默认 1e-3；
//! * 每次位移都**重建**基组、2e 张量与网格并重跑 SCF——不做任何几何假设；
//! * 结果可被三条与参照无关的不变量检验（见 `qwb-engine/tests/g01_numeric_gradient.rs`）：
//!   平移不变 `Σ_A dE/dR_A = 0`、H2 在极小点的梯度为零、力方向正确。
//!
//! 注意：数值梯度的精度受 `δ` 与 SCF 收敛阈值共同限制。做解析梯度比对时要给解析侧
//! 留 ~1e-6 的余地（§6.2 的判据就是这个量级）。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::Registry;

use qwb_scf::rks::{RksOptions, rks_from_data, rks_scf};

/// 数值梯度的选项。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GradientOptions {
    /// 基组名。
    pub basis: String,
    /// 泛函名（空字符串表示 HF）。
    pub functional: String,
    /// 网格等级名。
    pub grid: String,
    /// 差分步长，单位 **Bohr**。
    pub delta_bohr: f64,
    /// 自旋设置；`Uhf`/`Uks` 走开壳层的单点（SC-02）。
    pub scf_type: Option<qwb_engine_api::ScfType>,
}

impl GradientOptions {
    /// 默认值（`delta = 1e-3` Bohr）。
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
            delta_bohr: 1.0e-3,
            scf_type: None,
        }
    }
}

/// 把分子的第 `atom` 个原子沿 `axis` 移动 `shift_bohr`。
pub(crate) fn displaced(
    molecule: &Molecule,
    atom: usize,
    axis: usize,
    shift_bohr: f64,
) -> Molecule {
    const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
    let mut moved = molecule.clone();
    let mut position = moved.atoms[atom].position_angstrom;
    position[axis] += shift_bohr * ANGSTROM_PER_BOHR;
    moved.atoms[atom].position_angstrom = position;
    moved
}

/// 在**固定密度矩阵**下，XC 能量对核坐标的导数（中心差分）。
///
/// # 为什么需要它（一个真 bug 的收尾）
///
/// `qwb_scf::rks::xc_gradient_terms` 只算了"密度随原子移动"那一半：ρ 通道、σ 通道、
/// LYP 的 `∇²ρ` 通道。**没算**网格自己随原子移动那一半（Becke 权重的导数 + 网格点
/// 位置的导数）。H2 上被对称性抹掉（8.4e-7），H2O 上是 **3.30e-4 Ha/Bohr**
/// —— 足以把 def2-SVP 的优化几何带偏（O–H 0.9745 vs 0.9682 Å、∠HOH 102.08° vs ~104.5°）。
///
/// 这里用"固定 P、移动几何、重算网格与 `E_xc`、中心差分"求这一整块：
///
/// ```text
/// ∂E_xc[P]/∂R_A,x = ( E_xc[P](R + δ e) − E_xc[P](R − δ e) ) / (2δ)
/// ```
///
/// 它精确包含 AO 随核移动**和**网格随核移动两部分（所以生产路径用它**替代**
/// `xc_gradient_terms`，不是叠加）。代价是每算一次梯度多 `6N` 次 XC 求值
/// （不用重跑 SCF）—— 对水分子约等于一次 SCF 的量级。
///
/// 剩下的缺口记在 `docs/status.yaml` 的 G-01：Becke 权重导数的**解析**版本还没写，
/// 现在是 `O(δ²)` 的有限差分。
///
/// # Errors
///
/// 任一几何的 XC 求值失败（基组/网格/泛函数据问题）。
pub fn xc_geometric_derivative(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    grid_name: &str,
    density: &qwb_linalg::Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    delta_bohr: f64,
) -> Result<Vec<[f64; 3]>, EngineError> {
    if terms.is_empty() {
        // HF：没有 XC，这一项恒为 0（省掉 6N 次求值）
        return Ok(vec![[0.0_f64; 3]; molecule.atoms.len()]);
    }
    if !(delta_bohr.is_finite() && delta_bohr > 0.0) {
        return Err(EngineError::InvalidInput(format!(
            "xc geometric derivative delta must be positive and finite, got {delta_bohr}"
        )));
    }
    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for (atom, row) in gradient.iter_mut().enumerate() {
        for (axis, slot) in row.iter_mut().enumerate() {
            let plus = xc_energy_at(
                registry,
                &displaced(molecule, atom, axis, delta_bohr),
                basis_name,
                grid_name,
                density,
                terms,
            )?;
            let minus = xc_energy_at(
                registry,
                &displaced(molecule, atom, axis, -delta_bohr),
                basis_name,
                grid_name,
                density,
                terms,
            )?;
            *slot = (plus - minus) / (2.0 * delta_bohr);
        }
    }
    Ok(gradient)
}

/// 固定**两条自旋密度**下的 `E_xc` 几何导数（UKS 的 XC 网格项；SC-02）。
///
/// 与 [`xc_geometric_derivative`] 同一套做法（固定密度、中心差分），只是把
/// `E_xc[rho^a, rho^b]` 换成自旋分辨的求值。
///
/// # Errors
///
/// 任一几何的 XC 求值失败，或 `delta_bohr` 非法。
#[allow(clippy::too_many_arguments)]
pub fn xc_geometric_derivative_spin(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    grid_name: &str,
    density_alpha: &qwb_linalg::Matrix,
    density_beta: &qwb_linalg::Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    delta_bohr: f64,
) -> Result<Vec<[f64; 3]>, EngineError> {
    if terms.is_empty() {
        return Ok(vec![[0.0_f64; 3]; molecule.atoms.len()]);
    }
    if !(delta_bohr.is_finite() && delta_bohr > 0.0) {
        return Err(EngineError::InvalidInput(format!(
            "xc geometric derivative delta must be positive and finite, got {delta_bohr}"
        )));
    }
    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for (atom, row) in gradient.iter_mut().enumerate() {
        for (axis, slot) in row.iter_mut().enumerate() {
            let plus = xc_energy_spin_at(
                registry,
                &displaced(molecule, atom, axis, delta_bohr),
                basis_name,
                grid_name,
                density_alpha,
                density_beta,
                terms,
            )?;
            let minus = xc_energy_spin_at(
                registry,
                &displaced(molecule, atom, axis, -delta_bohr),
                basis_name,
                grid_name,
                density_alpha,
                density_beta,
                terms,
            )?;
            *slot = (plus - minus) / (2.0 * delta_bohr);
        }
    }
    Ok(gradient)
}

/// 固定两条自旋密度下的 `E_xc`（网格按当前几何重建）。
fn xc_energy_spin_at(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    grid_name: &str,
    density_alpha: &qwb_linalg::Matrix,
    density_beta: &qwb_linalg::Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> Result<f64, EngineError> {
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(basis_name)?, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    let cache =
        qwb_scf::rks::GridAoCached::build(&basis, &grid, needs_sigma, needs_laplacian, needs_tau)?;
    let evaluation = qwb_scf::uks::evaluate_xc_spin_cached(
        &basis,
        &grid,
        &cache,
        density_alpha,
        density_beta,
        terms,
    )?;
    Ok(evaluation.energy_hartree)
}

/// 固定密度矩阵下的 `E_xc`（网格按 `molecule` 的**当前**几何重建）。
fn xc_energy_at(
    registry: &Registry,
    molecule: &Molecule,
    basis_name: &str,
    grid_name: &str,
    density: &qwb_linalg::Matrix,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
) -> Result<f64, EngineError> {
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(basis_name)?, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(grid_name)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    let (energy, _potential) = qwb_scf::rks::evaluate_xc(&basis, &grid, density, terms)?;
    Ok(energy)
}

/// 给定几何的总能量（单点 SCF）。空 functional 名表示走 HF 分支。
fn single_point_energy(
    registry: &Registry,
    molecule: &Molecule,
    options: &GradientOptions,
) -> Result<f64, EngineError> {
    let mut scf_options = RksOptions::default();
    scf_options.max_iter = 400;
    // 开壳层的数值参照（SC-02）：UHF 走 `uhf_scf`，与解析梯度用的是同一套 SCF。
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        // UKS（开壳层 DFT）的数值参照：与解析路径用同一套 SCF。
        let level = registry.get_grid(&options.grid)?;
        let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let grid = qwb_grid::molecular_grid(level, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
        let mut uks_options = qwb_scf::uks::UksOptions::default();
        uks_options.max_iter = 400;
        uks_options.hf_exchange = hf_exchange;
        let result = qwb_scf::uks::uks_scf(&basis, molecule, &tensor, &grid, &terms, uks_options)?;
        return Ok(result.energy_hartree);
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
        let mut uks_options = qwb_scf::uks::UksOptions::default();
        uks_options.max_iter = 400;
        uks_options.hf_exchange = hf_exchange;
        let result = crate::opt::uks_scf_with_convergence_aids(
            &basis,
            molecule,
            &tensor,
            &grid,
            &terms,
            uks_options,
        )?;
        return Ok(result.energy_hartree);
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uhf)) {
        if !options.functional.is_empty() {
            return Err(EngineError::Unsupported(
                "options.scf_type = `uhf` is Hartree-Fock only; use `uks` for open-shell DFT"
                    .into(),
            ));
        }
        let basis_k1 = registry.get_basis(&options.basis)?;
        let basis = qwb_basis::InternalBasis::build(basis_k1, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let mut uhf_options = qwb_scf::uhf::UhfOptions::default();
        uhf_options.max_iter = 400;
        let result = qwb_scf::uhf::uhf_scf(&basis, molecule, &tensor, uhf_options)?;
        return Ok(result.energy_hartree);
    }
    if options.functional.is_empty() {
        // HF：空 terms
        let basis_k1 = registry.get_basis(&options.basis)?;
        let basis = qwb_basis::InternalBasis::build(basis_k1, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let level = registry.get_grid(&options.grid)?;
        let grid = qwb_grid::molecular_grid(level, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        let result = rks_scf(&basis, molecule, &tensor, &grid, &[], scf_options)?;
        Ok(result.energy_hartree)
    } else {
        let result = rks_from_data(
            registry,
            molecule,
            &options.basis,
            &options.functional,
            &options.grid,
            scf_options,
        )?;
        Ok(result.energy_hartree)
    }
}

/// 数值梯度 `dE/dR_A[k]`，单位 Ha/Bohr，形状 `[natoms][3]`。
///
/// # Errors
///
/// 任何一次位移的单点计算失败。
pub fn numerical_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &GradientOptions,
) -> Result<Vec<[f64; 3]>, EngineError> {
    if !(options.delta_bohr.is_finite() && options.delta_bohr > 0.0) {
        return Err(EngineError::InvalidInput(format!(
            "gradient delta must be positive and finite, got {}",
            options.delta_bohr
        )));
    }
    let mut gradient = vec![[0.0_f64; 3]; molecule.atoms.len()];
    for (atom, row) in gradient.iter_mut().enumerate() {
        for (axis, slot) in row.iter_mut().enumerate() {
            let plus = displaced(molecule, atom, axis, options.delta_bohr);
            let minus = displaced(molecule, atom, axis, -options.delta_bohr);
            let e_plus = single_point_energy(registry, &plus, options)?;
            let e_minus = single_point_energy(registry, &minus, options)?;
            *slot = (e_plus - e_minus) / (2.0 * options.delta_bohr);
        }
    }
    Ok(gradient)
}
