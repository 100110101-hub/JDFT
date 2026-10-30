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
//! 优先用解析梯度：HF 走 `qwb_scf::analytic_gradient`，KS（LDA/GGA/hybrid）走
//! `qwb_scf::rks::rks_analytic_gradient`（含 `α_hf` 缩放、`v_xc` 进 Pulay、XC 的
//! σ/`∇²ρ` 显式项 —— 见 docs/choices.md 的三十八节）。只有解析路径**报错**时才退到
//! 数值梯度（`gradient::numerical_gradient`，每次位移重跑 SCF），并把原因留在
//! `OptResult.analytic_fallback` 里 —— 不靠调用方猜，也不静默降级。
//!
//! 解析梯度把每一步的代价从「`6N` 次 SCF」降到「1 次 SCF + 1 次梯度」：水分子
//! （`N=3`，`6N=18`）差 18 倍，T8 的 PBE/def2-SVP 优化就是靠这条才跑得动。
//!
//! # 收敛
//!
//! 判据是力的**最大分量**（不是 RMS），与 T3 的写法一致。线搜索用 Armijo 回溯，
//! 每步步长上限 `max_step_bohr`，防止从很差的初值一步跳到别的势能面分支上。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::Registry;
use qwb_scf::rks::{RksOptions, rks_scf};

use crate::gradient::{GradientOptions, numerical_gradient};

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

/// XC 几何导数的差分步长（Bohr）。
///
/// 与数值梯度的 `δ = 1e-3` 同一个量级：`O(δ²)` 误差在 `δ=1e-3` 上约 1e-8 Ha/Bohr
/// （实测见 `tests/g01_analytic_def2svp.rs` 与 `g01_gradient_localize.rs`）。
pub(crate) const XC_DERIVATIVE_DELTA_BOHR: f64 = 1.0e-3;

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
    /// 每一步 SCF 的最大迭代步数。
    pub scf_max_iter: u32,
    /// 每一步 SCF 的能量收敛阈值（Hartree）。
    pub scf_energy_threshold: f64,
    /// 每一步 SCF 的密度收敛阈值。
    pub scf_density_threshold: f64,
    /// 自旋设置（`None`/`Rks`/`Rhf` = 闭壳层；`Uhf` = 开壳层 UHF，SC-02）。
    pub scf_type: Option<qwb_engine_api::ScfType>,
    /// 每一步 SCF 的热启动密度（`None` = core guess）。
    ///
    /// 用途是 Hessian：同一条势能面上相邻几何的密度几乎一样，从参考密度出发
    /// SCF 迭代数能从十几次掉到几次（RS-05 的墙钟优化，见 choices 七十二）。
    pub scf_initial_density: Option<qwb_linalg::Matrix>,
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
            // 与 `RksOptions::default` 的取值一致：优化里的每一步 SCF 都要真的收敛，
            // 否则梯度就是"未收敛密度"的梯度，BFGS 会沿着噪声走。
            scf_max_iter: 400,
            scf_energy_threshold: 1.0e-8,
            scf_density_threshold: 1.0e-6,
            scf_type: None,
            scf_initial_density: None,
        }
    }

    pub(crate) fn scf_options(&self) -> RksOptions {
        let mut options = RksOptions::default();
        options.max_iter = self.scf_max_iter;
        options.energy_threshold = self.scf_energy_threshold;
        options.density_threshold = self.scf_density_threshold;
        if let Some(density) = &self.scf_initial_density {
            options.resume_from = Some(qwb_scf::checkpoint::ScfCheckpoint::new(density, None, 0));
        }
        options
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
    /// 走数值梯度的**原因**（解析梯度成功时为 `None`）。
    ///
    /// 有这一条，"解析梯度可用" 这件事就不是靠 `used_analytic_gradient == true` 猜的：
    /// 一旦解析路径因为基组角动量/XC 导数实现不全而报错，原因会一路带到结果目录的
    /// `artifacts/optimization.json`。
    pub analytic_fallback: Option<String>,
    /// 每一步的 `(能量, 最大力)`，第 0 项是初值（G-03 的「多步轨迹」）。
    pub trajectory: Vec<(f64, f64)>,
}

/// 一步的观测值（优化过程中回调给调用方，用来打进度）。
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct OptStep {
    /// 第几步（0 = 初值）。
    pub iteration: u32,
    /// 这一步的能量（Hartree）。
    pub energy_hartree: f64,
    /// 这一步的最大力分量（Ha/Bohr）。
    pub max_force: f64,
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

/// 一次「SCF + 解析梯度」：同一次 SCF 的密度同时喂给能量和梯度，不做重复收敛。
///
/// XC 的**几何**导数（网格权重/网格点随核移动那一半）走
/// [`crate::gradient::xc_geometric_derivative`]（固定密度的中心差分）—— 见
/// `qwb_scf::rks::rks_analytic_gradient_with_xc` 的文档。
///
/// 公开是给"要在这个价位上做梯度"的调用方（Hessian、梯度验收测试）用的：
/// 生产梯度只有这一条路，测试和生产不能各走一条。
///
/// # Errors
///
/// 基组/网格/泛函数据不可用，SCF 未收敛，或梯度实现覆盖不到的角动量。
pub fn analytic_energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>), EngineError> {
    // SC-02：开壳层（UHF）走 `uhf_scf` + `uhf_analytic_gradient`。UKS 还需要把
    // XC 项按自旋拆开（以及 `v_xc` 进 Pulay），是 SC-02 的剩余一半 —— 显式拒绝。
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uhf)) {
        return uhf_analytic_energy_and_gradient(registry, molecule, options);
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        return uks_analytic_energy_and_gradient(registry, molecule, options);
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        // UKS 的梯度**装配**已经写好并过了闭壳层极限测试
        // （`tests/sc02_uks_gradient.rs`），但开壳层的 UKS SCF 在 OH/SVWN5/STO-3G 上
        // 400 步不收敛（DIIS/阻尼/能级移动/初猜缩放都不行），所以还不敢把它接进
        // 生产路径 —— 那会变成一个"梯度算得出来但密度没收敛"的假结果。
        return Err(EngineError::Unsupported(
            "options.scf_type = `uks` gradients are not enabled yet: the UKS gradient assembly              is verified (closed-shell limit) but open-shell UKS SCF convergence is the              remaining half of SC-02"
                .into(),
        ));
    }
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
    let one = qwb_integrals::one_electron::one_electron(&basis, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;

    let mut scf_options = options.scf_options();
    scf_options.hf_exchange = hf_exchange;
    let result = rks_scf(&basis, molecule, &tensor, &grid, &terms, scf_options)?;
    let xc = crate::gradient::xc_geometric_derivative(
        registry,
        molecule,
        &options.basis,
        &options.grid,
        &result.density,
        &terms,
        XC_DERIVATIVE_DELTA_BOHR,
    )?;
    let gradient = qwb_scf::rks::rks_analytic_gradient_with_xc(
        &basis,
        molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &result.density,
        &terms,
        hf_exchange,
        &xc,
    )?;
    Ok((result.energy_hartree, gradient))
}

/// 开壳层（UHF）的一次「SCF + 解析梯度」（SC-02）。
///
/// 只支持纯 HF：`functional` 非空时显式拒绝（UKS 的 XC 自旋拆分是 SC-02 的剩余一半）。
///
/// # Errors
///
/// 泛函非空、SCF 未收敛、或梯度尺寸不符。
/// 驻点（Brillouin）判据：`‖F^σ P^σ S − S P^σ F^σ‖_max`（两条自旋取最大）。
///
/// 解析梯度假定它为零。**能级移动分支的不动点不满足**（实测 1.1e-2，见 `choices` 七十五），
/// 所以梯度路径必须先查这一条，否则会静默给出错的梯度。
fn brillouin_residual(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    grid: &qwb_grid::Grid,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    result: &qwb_scf::uks::UksResult,
) -> Result<f64, EngineError> {
    let one = qwb_integrals::one_electron::one_electron(basis, molecule)?;
    let n = basis.n_functions();
    let mut total = qwb_linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            total[(i, j)] = result.density_alpha[(i, j)] + result.density_beta[(i, j)];
        }
    }
    let coulomb = qwb_scf::build_fock(&one.core_hamiltonian, tensor, &total)
        .map_err(|err| EngineError::Internal(format!("uks: J: {err}")))?
        .coulomb;
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let needs_laplacian = terms.iter().any(|term| term.requires().laplacian);
    let needs_tau = terms.iter().any(|term| term.requires().kinetic_density);
    let cache =
        qwb_scf::rks::GridAoCached::build(basis, grid, needs_sigma, needs_laplacian, needs_tau)?;
    let xc = qwb_scf::uks::evaluate_xc_spin_cached(
        basis,
        grid,
        &cache,
        &result.density_alpha,
        &result.density_beta,
        terms,
    )?;
    let mut worst = 0.0_f64;
    for (density, potential) in [
        (&result.density_alpha, &xc.v_alpha),
        (&result.density_beta, &xc.v_beta),
    ] {
        let mut fock = qwb_linalg::Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] = one.core_hamiltonian[(i, j)] + coulomb[(i, j)] + potential[(i, j)];
            }
        }
        let fp = qwb_linalg::matmul(&fock, density)
            .map_err(|err| EngineError::Internal(format!("uks: FP: {err}")))?;
        let fps = qwb_linalg::matmul(&fp, &one.overlap)
            .map_err(|err| EngineError::Internal(format!("uks: FPS: {err}")))?;
        let sp = qwb_linalg::matmul(&one.overlap, density)
            .map_err(|err| EngineError::Internal(format!("uks: SP: {err}")))?;
        let spf = qwb_linalg::matmul(&sp, &fock)
            .map_err(|err| EngineError::Internal(format!("uks: SPF: {err}")))?;
        for i in 0..n {
            for j in 0..n {
                worst = worst.max((fps[(i, j)] - spf[(i, j)]).abs());
            }
        }
    }
    Ok(worst)
}

/// UKS 的收敛与**驻点**保证：正统算法先试；不行就走「能级移动找解 → 无移动重启抛光」。
///
/// 抛光后仍要查 [`brillouin_residual`]，超过 1e-6 就报错 —— 梯度不能建在非驻点上。
pub(crate) fn uks_scf_with_convergence_aids(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    tensor: &qwb_integrals::two_electron::TwoElectronIntegrals,
    grid: &qwb_grid::Grid,
    terms: &[Box<dyn qwb_xc::FunctionalTerm>],
    base: qwb_scf::uks::UksOptions,
) -> Result<qwb_scf::uks::UksResult, EngineError> {
    let mut plain = base.clone();
    plain.level_shift = 0.0;
    plain.initial_density = None;
    if let Ok(result) = qwb_scf::uks::uks_scf(basis, molecule, tensor, grid, terms, plain)
        && brillouin_residual(basis, molecule, tensor, grid, terms, &result)? < 1.0e-6
    {
        return Ok(result);
    }
    let mut last_error: Option<EngineError> = None;
    for (damping, level_shift, guess_scale) in [(0.3, 0.2, 1.0), (0.3, 0.2, 0.5), (0.3, 0.2, 1.5)] {
        let mut aid = base.clone();
        aid.use_diis = false;
        aid.damping = damping;
        aid.level_shift = level_shift;
        aid.initial_guess_scale = guess_scale;
        match qwb_scf::uks::uks_scf(basis, molecule, tensor, grid, terms, aid) {
            Ok(found) => {
                // **能级移动的延拓抛光**：λ 逐级降到 0，每一步都从上一解热启动。
                // 一次性从 λ=0.2 直接跳到 0，OH 这类体系又会飘走（实测 400 步不收敛）；
                // 逐级降 λ 能把解连续地带到真驻点上（残差随之收敛到 < 1e-6）。
                let mut current = found;
                for level_shift in [0.05_f64, 0.01, 0.002, 0.0005, 0.0] {
                    let mut step = base.clone();
                    step.level_shift = level_shift;
                    step.damping = 0.0;
                    step.use_diis = true;
                    step.initial_density =
                        Some((current.density_alpha.clone(), current.density_beta.clone()));
                    match qwb_scf::uks::uks_scf(basis, molecule, tensor, grid, terms, step) {
                        Ok(result) => current = result,
                        Err(err) => {
                            let _ = err;
                            break;
                        }
                    }
                }
                let residual = brillouin_residual(basis, molecule, tensor, grid, terms, &current)?;
                if residual < 1.0e-6 {
                    return Ok(current);
                }
                return Ok(current).and_then(|_| {
                    Err(EngineError::Internal(format!(
                        "uks: the converged density is not a Brillouin point \
                         (||FPS-SPF|| = {residual:.3e}); a gradient from it would be wrong"
                    )))
                });
            }
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.unwrap_or_else(|| EngineError::Internal("uks: no attempt ran".into())))
}

fn uhf_analytic_energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>), EngineError> {
    if !options.functional.is_empty() {
        return Err(EngineError::Unsupported(
            "open-shell analytic gradients are wired for UHF only; UKS needs the spin-resolved \
             XC terms (the remaining half of SC-02)"
                .into(),
        ));
    }
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let mut uhf_options = qwb_scf::uhf::UhfOptions::default();
    uhf_options.max_iter = options.scf_max_iter;
    uhf_options.energy_threshold = options.scf_energy_threshold;
    uhf_options.density_threshold = options.scf_density_threshold;
    let result = qwb_scf::uhf::uhf_scf(&basis, molecule, &tensor, uhf_options)?;
    if !result.scf_converged {
        return Err(EngineError::ScfNotConverged {
            iters: result.iterations,
            max: options.scf_max_iter,
        });
    }
    let w_alpha = qwb_scf::analytic_gradient::energy_weighted_density(
        &result.coefficients_alpha,
        &result.mo_energies_alpha,
        result.n_alpha,
    )?;
    let w_beta = qwb_scf::analytic_gradient::energy_weighted_density(
        &result.coefficients_beta,
        &result.mo_energies_beta,
        result.n_beta,
    )?;
    let gradient = qwb_scf::analytic_gradient::uhf_analytic_gradient(
        &basis,
        molecule,
        &result.density_alpha,
        &result.density_beta,
        &w_alpha,
        &w_beta,
    )?;
    Ok((result.energy_hartree, gradient))
}

/// 开壳层 DFT（UKS）的一次「SCF + 解析梯度」（SC-02）。
///
/// `v_xc` 按自旋进 `W`（Pulay 项），XC 的网格项走
/// [`crate::gradient::xc_geometric_derivative_spin`]（固定自旋密度的中心差分）。
///
/// # Errors
///
/// 泛函数据不可用、泛函不支持自旋、SCF 未收敛。
fn uks_analytic_energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>), EngineError> {
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
    let one = qwb_integrals::one_electron::one_electron(&basis, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
    let mut uks_options = qwb_scf::uks::UksOptions::default();
    uks_options.max_iter = options.scf_max_iter;
    uks_options.energy_threshold = options.scf_energy_threshold;
    uks_options.density_threshold = options.scf_density_threshold;
    uks_options.hf_exchange = hf_exchange;
    let result =
        uks_scf_with_convergence_aids(&basis, molecule, &tensor, &grid, &terms, uks_options)?;
    let xc = crate::gradient::xc_geometric_derivative_spin(
        registry,
        molecule,
        &options.basis,
        &options.grid,
        &result.density_alpha,
        &result.density_beta,
        &terms,
        XC_DERIVATIVE_DELTA_BOHR,
    )?;
    let gradient = qwb_scf::analytic_gradient::uks_analytic_gradient(
        &basis,
        molecule,
        &grid,
        &one.core_hamiltonian,
        &tensor,
        &result.density_alpha,
        &result.density_beta,
        &terms,
        hf_exchange,
        &xc,
    )?;
    Ok((result.energy_hartree, gradient))
}

/// 一次"能量 + 梯度"的结果：`(能量, 梯度, 退到数值梯度的原因)`。
type EnergyAndGradient = (f64, Vec<[f64; 3]>, Option<String>);

/// 能量 + 梯度：先解析，解析失败才退数值（并把原因交给调用方）。
fn energy_and_gradient(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<EnergyAndGradient, EngineError> {
    match analytic_energy_and_gradient(registry, molecule, options) {
        Ok((energy, gradient)) => Ok((energy, gradient, None)),
        Err(analytic_error) => {
            // 数值路径（任何基组/泛函）。这是**降级**，不是等价路径：把解析失败的原因
            // 带出去，别让"梯度算出来了"看起来像"解析梯度没问题"。
            let mut gradient_options = GradientOptions::new(
                options.basis.clone(),
                options.functional.clone(),
                options.grid.clone(),
            );
            gradient_options.delta_bohr = 1.0e-3;
            gradient_options.scf_type = options.scf_type;
            let gradient = numerical_gradient(registry, molecule, &gradient_options)?;
            let energy = single_point(registry, molecule, options)?;
            Ok((energy, gradient, Some(analytic_error.to_string())))
        }
    }
}

fn single_point(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
) -> Result<f64, EngineError> {
    let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
        let mut uks_options = qwb_scf::uks::UksOptions::default();
        uks_options.max_iter = options.scf_max_iter;
        uks_options.energy_threshold = options.scf_energy_threshold;
        uks_options.density_threshold = options.scf_density_threshold;
        uks_options.hf_exchange = hf_exchange;
        return Ok(
            qwb_scf::uks::uks_scf(&basis, molecule, &tensor, &grid, &terms, uks_options)?
                .energy_hartree,
        );
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uks)) {
        let basis = qwb_basis::InternalBasis::build(registry.get_basis(&options.basis)?, molecule)?;
        let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
        let grid = qwb_grid::molecular_grid(registry.get_grid(&options.grid)?, molecule)
            .map_err(|err| EngineError::Data(format!("grid: {err}")))?;
        let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
        let mut uks_options = qwb_scf::uks::UksOptions::default();
        uks_options.max_iter = options.scf_max_iter;
        uks_options.energy_threshold = options.scf_energy_threshold;
        uks_options.density_threshold = options.scf_density_threshold;
        uks_options.hf_exchange = hf_exchange;
        return Ok(uks_scf_with_convergence_aids(
            &basis,
            molecule,
            &tensor,
            &grid,
            &terms,
            uks_options,
        )?
        .energy_hartree);
    }
    if matches!(options.scf_type, Some(qwb_engine_api::ScfType::Uhf)) {
        let mut uhf_options = qwb_scf::uhf::UhfOptions::default();
        uhf_options.max_iter = options.scf_max_iter;
        uhf_options.energy_threshold = options.scf_energy_threshold;
        uhf_options.density_threshold = options.scf_density_threshold;
        return Ok(qwb_scf::uhf::uhf_scf(&basis, molecule, &tensor, uhf_options)?.energy_hartree);
    }
    let (terms, hf_exchange) = crate::terms_and_hf_exchange(registry, &options.functional)?;
    let mut scf_options = options.scf_options();
    scf_options.hf_exchange = hf_exchange;
    Ok(rks_scf(&basis, molecule, &tensor, &grid, &terms, scf_options)?.energy_hartree)
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
    optimize_with_progress(registry, molecule, options, &mut |_| true)
}

/// 与 [`optimize`] 相同，但每走一步都把 `(步数, 能量, 最大力)` 回调出去。
///
/// 回调返回 `false` 表示**中止**（协作取消；返回
/// [`EngineError::InvalidInput`]，与 K2 的"启动前被取消"同一个变体）。
///
/// 存在的理由：T8 的 `opt` 作业要跑几分钟到几十分钟，GUI 的 Job Monitor 需要有东西
/// 可看。回调在**工作线程**里执行（引擎本来就是阻塞调用），调用方自己保证线程安全。
///
/// # Errors
///
/// 同 [`optimize`]，外加回调中止时的 [`EngineError::InvalidInput`]。
pub fn optimize_with_progress(
    registry: &Registry,
    molecule: &Molecule,
    options: &OptOptions,
    on_step: &mut dyn FnMut(&OptStep) -> bool,
) -> Result<OptResult, EngineError> {
    let dimension = molecule.atoms.len() * 3;
    let mut coordinates = flatten(molecule);
    let (mut energy, mut gradient, mut analytic_fallback) =
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
    if !on_step(&OptStep {
        iteration: 0,
        energy_hartree: energy,
        max_force: max_force(&flat_gradient),
    }) {
        return Err(EngineError::InvalidInput(
            "job cancelled during optimisation".into(),
        ));
    }

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

        let (_, new_gradient, fallback) = energy_and_gradient(registry, &new_molecule, options)?;
        if analytic_fallback.is_none() {
            analytic_fallback = fallback;
        }
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
        if !on_step(&OptStep {
            iteration: iterations,
            energy_hartree: energy,
            max_force: max_force(&flat_gradient),
        }) {
            return Err(EngineError::InvalidInput(
                "job cancelled during optimisation".into(),
            ));
        }
    }

    Ok(OptResult {
        molecule: unflatten(molecule, &coordinates),
        energy_hartree: energy,
        gradient,
        iterations,
        converged,
        used_analytic_gradient: analytic_fallback.is_none(),
        analytic_fallback,
        trajectory,
    })
}
