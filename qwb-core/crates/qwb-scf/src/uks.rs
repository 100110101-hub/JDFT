//! SC-01b / M1b：**UKS**（自旋极化 Kohn–Sham）—— 自由基也能用 DFT 算了。
//!
//! # 与 UHF、RKS 的关系
//!
//! ```text
//!         闭壳层                         开壳层
//! HF      RHF（阶段一）                  UHF（SC-01 / M1）
//! DFT     RKS（阶段一）                  UKS（本模块 / SC-01b / M1b）
//! ```
//!
//! 数值骨架与 [`crate::uhf`] 相同（双套密度、每个自旋通道一套 Fock 与 DIIS），
//! 只是把"交换项"换成"自旋分辨的 XC 势"：
//!
//! ```text
//! P       = P_α + P_β
//! J       = J(P)                          （库仑只看总密度）
//! F_σ     = H + J + v_σ − α_hf·K(P_σ)     （v_σ = ∂E_xc/∂ρ_σ，含 σ 通道）
//! E_elec  = Σ_σ Tr[P_σ H] + ½ Tr[P J] + E_xc − ½ α_hf Σ_σ Tr[P_σ K(P_σ)]
//! ```
//!
//! 能量的写法**不用** `½ Σ_σ Tr[P_σ (H + F_σ)]`：那个式子在混入 XC 之后会把
//! `½∫ρ_σ v_σ` 也算进来，必须再减掉才是 E_xc（这是 KS 与 HF 最容易搞错的一处）。
//! 上面这条是把 XC 显式写出来的版本，退化检查：`α_hf = 1`、`E_xc = v_σ = 0` 时
//! 与 [`crate::uhf::uhf_scf`] 的能量公式逐项相同。
//!
//! # 闭壳层退化（M1b 的判据之一）
//!
//! `ρ_α = ρ_β = ρ/2` 时，`qwb-xc` 的自旋接口在 ζ = 0 处逐位退化为闭壳层实现
//! （`tests/sc01b_spin_xc_pointwise.rs` 钉住这一点），于是 UKS 与 RKS 的能量
//! 只差 DIIS 轨迹的舍入（判据 < 1e-8）。
//!
//! # 哪些泛函能跑 UKS
//!
//! 只有实现了 [`qwb_xc::FunctionalTerm::evaluate_spin`] 的组件：`LdaSlaterX`、
//! `LdaSvwn5`、`GgaPbeX`、`GgaPbeC`。B88/LYP 还没有自旋实现，所以 BLYP/B3LYP 的
//! UKS 会**显式报错**（而不是安静地用闭壳层公式跑出错的能量）。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_grid::Grid;
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::{Matrix, Vector, matmul, symmetric_eigen};
use qwb_xc::{FunctionalTerm, SpinDensityBatch, SpinXcOutput};

use crate::build_fock;
use crate::diis::Diis;
use crate::rks::{
    GridAoCached, default_charges, inverse_sqrt_symmetric, linalg, nuclear_repulsion,
};
use crate::uhf::{UhfInitialGuess, max_abs_difference, mix, split_spins};

/// UKS 控制参数（字段语义与 [`crate::uhf::UhfOptions`] 一一对应）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UksOptions {
    /// 最大迭代步数。
    pub max_iter: u32,
    /// 能量收敛阈值（Hartree）。
    pub energy_threshold: f64,
    /// 密度矩阵收敛阈值（两个自旋通道取最大）。
    pub density_threshold: f64,
    /// 是否用 Pulay DIIS 加速（每个自旋通道一套）。
    pub use_diis: bool,
    /// DIIS 子空间上限。
    pub diis_max_vectors: usize,
    /// 密度线性混合系数（`0` = 直接替换）。
    pub damping: f64,
    /// 静态能级移动 `λ`。
    pub level_shift: f64,
    /// 能级移动的**逐步衰减因子**（`0` = 不衰减，恒定 `level_shift`）。
    ///
    /// 退火用：λ_k = λ_0 · decay^k，衰减到 1e-4 以下就当 0。OH/SVWN5 这类在两个
    /// 自旋解之间跳的体系，恒定 λ 只能停在"移位哈密顿量的不动点"（不是驻点），
    /// 逐步退火才可能被带到真驻点上（`docs/choices.md` 七十六）。
    pub level_shift_decay: f64,
    /// 初猜的非对角缩放（多初猜用）。
    pub initial_guess_scale: f64,
    /// 初猜策略（默认原子密度叠加）。
    pub initial_guess: UhfInitialGuess,
    /// 每个原子的核电荷（ECP 体系传 `z_eff`）。
    pub nuclear_charges: Option<Vec<f64>>,
    /// ECP 的电子-核有效势矩阵元。
    pub ecp_potential: Option<Matrix>,
    /// hybrid 泛函的精确交换混入比例（纯泛函 = 0，PBE0 = 0.25）。
    pub hf_exchange: f64,
    /// 自旋密度热启动（`None` = 用 `initial_guess`）。
    ///
    /// 用途：先靠「阻尼 + 能级移动」那条辅助分支找到解，再**从这条密度**用无移动的正统
    /// 算法收敛到真驻点 —— 辅助分支的不动点不是 Brillouin 点（实测交换子残差 1.1e-2，
    /// 见 `docs/choices.md` 七十五）。
    pub initial_density: Option<(Matrix, Matrix)>,
}

impl Default for UksOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            energy_threshold: 1.0e-8,
            density_threshold: 1.0e-6,
            use_diis: true,
            diis_max_vectors: 8,
            damping: 0.0,
            level_shift: 0.0,
            level_shift_decay: 0.0,
            initial_guess_scale: 1.0,
            initial_guess: UhfInitialGuess::default(),
            nuclear_charges: None,
            ecp_potential: None,
            hf_exchange: 0.0,
            initial_density: None,
        }
    }
}

/// UKS 结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UksResult {
    /// 总能量（含核排斥），Hartree。
    pub energy_hartree: f64,
    /// 是否收敛。
    pub scf_converged: bool,
    /// 迭代步数。
    pub iterations: u32,
    /// 电子能量（不含核排斥）。
    pub electronic_hartree: f64,
    /// 核排斥能。
    pub nuclear_repulsion_hartree: f64,
    /// 交换相关能 `E_xc = ∫ e_xc dV`。
    pub xc_energy_hartree: f64,
    /// **总**密度 `P_α + P_β`。
    pub density: Matrix,
    /// α 密度。
    pub density_alpha: Matrix,
    /// β 密度。
    pub density_beta: Matrix,
    /// α 轨道能（升序）。
    pub mo_energies_alpha: Vector,
    /// β 轨道能（升序）。
    pub mo_energies_beta: Vector,
    /// α MO 系数（列 = MO）。
    pub coefficients_alpha: Matrix,
    /// β MO 系数（列 = MO）。
    pub coefficients_beta: Matrix,
    /// α 电子数。
    pub n_alpha: usize,
    /// β 电子数。
    pub n_beta: usize,
    /// `⟨S²⟩`。
    pub s_squared: f64,
    /// 自旋污染 `⟨S²⟩ − S(S+1)`。
    pub spin_contamination: f64,
    /// 最后一次能量变化。
    pub last_energy_change: f64,
}

/// 一次自旋分辨 XC 求值的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpinXcEvaluation {
    /// `E_xc = ∫ e_xc dV`（Hartree）。
    pub energy_hartree: f64,
    /// α 通道的 XC 势矩阵（`∫` 的一部分，已乘网格权重）。
    pub v_alpha: Matrix,
    /// β 通道的 XC 势矩阵。
    pub v_beta: Matrix,
}

/// 检查所有泛函项是否支持自旋分辨求值。
///
/// # Errors
///
/// 任一项没实现 `evaluate_spin`（例如 B88/LYP）时返回 [`EngineError::Unsupported`]，
/// 消息里点名是哪个组件 —— "静默用闭壳层公式"是这类代码最坏的失败方式。
pub fn require_spin_support(terms: &[Box<dyn FunctionalTerm>]) -> Result<(), EngineError> {
    for term in terms {
        if !term.spin_supported() {
            return Err(EngineError::Unsupported(format!(
                "functional component `{}` has no spin-resolved implementation yet, so \
                 scf_type = `uks` cannot use it (SC-01b covers SVWN5 and PBE; B88/LYP are \
                 phase-2 follow-ups). Use scf_type = `rks` for closed shells.",
                term.id()
            )));
        }
    }
    Ok(())
}

/// 在网格上求自旋分辨的 `E_xc`、`v_α`、`v_β`。
///
/// # Errors
///
/// - 密度或梯度出现非有限值；
/// - 泛函项没有自旋实现。
pub fn evaluate_xc_spin_cached(
    basis: &qwb_basis::InternalBasis,
    grid: &Grid,
    cache: &GridAoCached,
    density_alpha: &Matrix,
    density_beta: &Matrix,
    terms: &[Box<dyn FunctionalTerm>],
) -> Result<SpinXcEvaluation, EngineError> {
    let n = basis.n_functions();
    let points = grid.len();
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    if terms.iter().any(|term| term.requires().laplacian) {
        return Err(EngineError::Unsupported(
            "spin-resolved XC does not have a ∇²ρ channel yet (LYP-class functionals); \
             scf_type = `uks` supports LDA and GGA without ∇²ρ (SC-01b)"
                .to_string(),
        ));
    }

    // ── 每个点的 ρ_σ、∇ρ_σ、σ ────────────────────────────────────
    let mut rho_alpha = vec![0.0_f64; points];
    let mut rho_beta = vec![0.0_f64; points];
    let mut sigma_alpha = vec![0.0_f64; points];
    let mut sigma_beta = vec![0.0_f64; points];
    let mut sigma_alpha_beta = vec![0.0_f64; points];
    let mut grad_alpha = vec![[0.0_f64; 3]; points];
    let mut grad_beta = vec![[0.0_f64; 3]; points];

    for index in 0..points {
        let values = cache.values_at(index);
        let mut alpha = 0.0;
        let mut beta = 0.0;
        for i in 0..n {
            for j in 0..n {
                let phi = values[i] * values[j];
                alpha += phi * density_alpha[(i, j)];
                beta += phi * density_beta[(i, j)];
            }
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(EngineError::Internal(format!(
                "spin density is not finite at grid point {index} ({alpha}, {beta}); \
                 grid coordinates or AO values are corrupted"
            )));
        }
        rho_alpha[index] = alpha.max(0.0);
        rho_beta[index] = beta.max(0.0);
        if needs_sigma {
            let gradients = cache.gradients_at(index);
            let mut ga = [0.0_f64; 3];
            let mut gb = [0.0_f64; 3];
            for i in 0..n {
                for j in 0..n {
                    let pa = density_alpha[(i, j)];
                    let pb = density_beta[(i, j)];
                    if pa == 0.0 && pb == 0.0 {
                        continue;
                    }
                    for axis in 0..3 {
                        let grad_phi =
                            gradients[i][axis] * values[j] + values[i] * gradients[j][axis];
                        ga[axis] += pa * grad_phi;
                        gb[axis] += pb * grad_phi;
                    }
                }
            }
            if !ga.iter().all(|value| value.is_finite())
                || !gb.iter().all(|value| value.is_finite())
            {
                return Err(EngineError::Internal(
                    "spin density gradient is not finite at a grid point".to_string(),
                ));
            }
            grad_alpha[index] = ga;
            grad_beta[index] = gb;
            sigma_alpha[index] = ga[0] * ga[0] + ga[1] * ga[1] + ga[2] * ga[2];
            sigma_beta[index] = gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2];
            sigma_alpha_beta[index] = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
        }
    }

    let batch = if needs_sigma {
        SpinDensityBatch::with_sigma(
            &rho_alpha,
            &rho_beta,
            &sigma_alpha,
            &sigma_beta,
            &sigma_alpha_beta,
        )
    } else {
        SpinDensityBatch::new(&rho_alpha, &rho_beta)
    };

    // ── 逐项求值并组装 ───────────────────────────────────────────
    let mut energy = 0.0;
    let mut v_alpha_matrix = Matrix::zeros(n, n);
    let mut v_beta_matrix = Matrix::zeros(n, n);
    let mut energy_density = vec![0.0_f64; points];
    let mut potential_alpha = vec![0.0_f64; points];
    let mut potential_beta = vec![0.0_f64; points];
    let mut d_sigma_alpha = vec![0.0_f64; points];
    let mut d_sigma_beta = vec![0.0_f64; points];
    let mut d_sigma_alpha_beta = vec![0.0_f64; points];

    for term in terms {
        let mut out = SpinXcOutput::new(
            &mut energy_density,
            &mut potential_alpha,
            &mut potential_beta,
            &mut d_sigma_alpha,
            &mut d_sigma_beta,
            &mut d_sigma_alpha_beta,
        );
        term.evaluate_spin(&batch, &mut out)
            .map_err(|err| EngineError::Unsupported(format!("{}: {err}", term.id())))?;

        for (index, point) in grid.points.iter().enumerate() {
            let weight = point.weight;
            energy += weight * energy_density[index];
            let values = cache.values_at(index);
            // σ 通道的向量系数（分部积分后：∫ c_σ · ∇(φ_μ φ_ν)）。
            let (coefficient_alpha, coefficient_beta) = if needs_sigma {
                let ga = grad_alpha[index];
                let gb = grad_beta[index];
                let mut ca = [0.0_f64; 3];
                let mut cb = [0.0_f64; 3];
                for axis in 0..3 {
                    ca[axis] = 2.0 * d_sigma_alpha[index] * ga[axis]
                        + d_sigma_alpha_beta[index] * gb[axis];
                    cb[axis] =
                        2.0 * d_sigma_beta[index] * gb[axis] + d_sigma_alpha_beta[index] * ga[axis];
                }
                (ca, cb)
            } else {
                ([0.0_f64; 3], [0.0_f64; 3])
            };
            let gradients = if needs_sigma {
                Some(cache.gradients_at(index))
            } else {
                None
            };
            for i in 0..n {
                let value_i = values[i];
                if value_i == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let phi = value_i * values[j];
                    let mut element_alpha = potential_alpha[index] * phi;
                    let mut element_beta = potential_beta[index] * phi;
                    if let Some(gradients) = gradients {
                        let mut grad_phi = [0.0_f64; 3];
                        for axis in 0..3 {
                            grad_phi[axis] =
                                gradients[i][axis] * values[j] + value_i * gradients[j][axis];
                        }
                        for axis in 0..3 {
                            element_alpha += coefficient_alpha[axis] * grad_phi[axis];
                            element_beta += coefficient_beta[axis] * grad_phi[axis];
                        }
                    }
                    v_alpha_matrix[(i, j)] += weight * element_alpha;
                    v_beta_matrix[(i, j)] += weight * element_beta;
                }
            }
        }
    }

    Ok(SpinXcEvaluation {
        energy_hartree: energy,
        v_alpha: v_alpha_matrix,
        v_beta: v_beta_matrix,
    })
}

/// 自旋极化 Kohn–Sham（SC-01b）。
///
/// # Errors
///
/// - 电子数/多重度不自洽 → [`EngineError::InvalidInput`]；
/// - 泛函项没有自旋实现、或带 `∇²ρ` 通道 → [`EngineError::Unsupported`]；
/// - `max_iter` 步内不收敛 → [`EngineError::ScfNotConverged`]。
pub fn uks_scf(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    tensor: &TwoElectronIntegrals,
    grid: &Grid,
    terms: &[Box<dyn FunctionalTerm>],
    options: UksOptions,
) -> Result<UksResult, EngineError> {
    let n = basis.n_functions();
    if tensor.n_functions != n {
        return Err(EngineError::Internal(
            "two-electron tensor size does not match the basis".to_string(),
        ));
    }
    if terms.is_empty() {
        return Err(EngineError::Unsupported(
            "scf_type = `uks` needs a density functional; for Hartree-Fock use scf_type = `uhf` \
             (SC-01)"
                .to_string(),
        ));
    }
    require_spin_support(terms)?;

    let charges = match &options.nuclear_charges {
        Some(charges) => {
            if charges.len() != molecule.n_atoms() {
                return Err(EngineError::Internal(format!(
                    "UksOptions::nuclear_charges has {} entries for {} atoms",
                    charges.len(),
                    molecule.n_atoms()
                )));
            }
            charges.clone()
        }
        None => default_charges(molecule)?,
    };
    let mut electrons = 0i64 - i64::from(molecule.charge);
    for charge in &charges {
        electrons += charge.round() as i64;
    }
    let (n_alpha, n_beta) = split_spins(electrons, molecule.multiplicity)?;

    let integrals =
        qwb_integrals::one_electron::one_electron_with_charges(basis, molecule, &charges)?;
    let overlap = &integrals.overlap;
    let mut hcore = integrals.core_hamiltonian.clone();
    if let Some(potential) = &options.ecp_potential {
        if potential.nrows() != n || potential.ncols() != n {
            return Err(EngineError::Internal(format!(
                "UksOptions::ecp_potential is {}x{}, expected {n}x{n}",
                potential.nrows(),
                potential.ncols()
            )));
        }
        for mu in 0..n {
            for nu in 0..n {
                hcore[(mu, nu)] += potential[(mu, nu)];
            }
        }
    }
    let x = inverse_sqrt_symmetric(overlap)?;
    let e_nuc = nuclear_repulsion(&charges, molecule)?;

    // AO 三元组整次 SCF 只算一次。
    // 自旋分辨的 meta-GGA（τ_α/τ_β）还没做：这里显式拒绝，而不是把 τ=0 喂给泛函。
    if terms.iter().any(|term| term.requires().kinetic_density) {
        return Err(EngineError::Unsupported(
            "spin-resolved meta-GGA (τ_α/τ_β) is not implemented yet; `scf_type = uks` \
             supports LDA and GGA without τ (SC-01b). Closed-shell meta-GGA works through \
             `scf_type = rks` (FX-01)."
                .to_string(),
        ));
    }
    let needs_sigma = terms.iter().any(|term| term.requires().sigma);
    let cache = GridAoCached::build(basis, grid, needs_sigma, false, false)?;

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
    // 初猜：优先用调用方给的热启动密度，其次 spin-averaged SAD（core guess 做退路）。
    let (mut density_alpha, mut density_beta) =
        if let Some((alpha, beta)) = &options.initial_density {
            if alpha.nrows() != n || beta.nrows() != n {
                return Err(EngineError::Internal(format!(
                    "uks: initial_density is {0}x{0}/{1}x{1} for {n} AO",
                    alpha.nrows(),
                    beta.nrows()
                )));
            }
            (alpha.clone(), beta.clone())
        } else {
            match options.initial_guess {
                UhfInitialGuess::AtomicDensities => {
                    let (alpha, beta) = crate::guess::sad_guess(basis, molecule)?;
                    if alpha.as_column_major().iter().all(|value| *value == 0.0) {
                        let (a, _, _) = density_from_fock(&guess_hcore, &x, n_alpha)?;
                        let b = if n_beta == n_alpha {
                            a.clone()
                        } else {
                            density_from_fock(&guess_hcore, &x, n_beta)?.0
                        };
                        (a, b)
                    } else {
                        (alpha, beta)
                    }
                }
                UhfInitialGuess::Core => {
                    let (a, _, _) = density_from_fock(&guess_hcore, &x, n_alpha)?;
                    let b = if n_beta == n_alpha {
                        a.clone()
                    } else {
                        density_from_fock(&guess_hcore, &x, n_beta)?.0
                    };
                    (a, b)
                }
            }
        };

    let mut diis_alpha = if options.use_diis {
        Some(Diis::new(options.diis_max_vectors))
    } else {
        None
    };
    let mut diis_beta = if options.use_diis {
        Some(Diis::new(options.diis_max_vectors))
    } else {
        None
    };

    let mut previous_energy = f64::NAN;
    let mut last_change = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    let mut coefficients_alpha = Matrix::zeros(n, n);
    let mut coefficients_beta = Matrix::zeros(n, n);
    let mut mo_energies_alpha = Vector::zeros(n);
    let mut mo_energies_beta = Vector::zeros(n);
    let mut electronic = 0.0;
    let mut xc_energy = 0.0;

    for step in 0..options.max_iter {
        iterations = step + 1;
        let total_density = add(&density_alpha, &density_beta)?;
        let xc =
            evaluate_xc_spin_cached(basis, grid, &cache, &density_alpha, &density_beta, terms)?;

        // J 只看总密度；K 只在 hybrid 时需要。
        let coulomb = build_fock(&hcore, tensor, &total_density)
            .map_err(linalg)?
            .coulomb;
        let (exchange_alpha, exchange_beta) = if options.hf_exchange.abs() > 0.0 {
            (
                build_fock(&hcore, tensor, &density_alpha)
                    .map_err(linalg)?
                    .exchange,
                build_fock(&hcore, tensor, &density_beta)
                    .map_err(linalg)?
                    .exchange,
            )
        } else {
            (Matrix::zeros(n, n), Matrix::zeros(n, n))
        };

        // 退火：λ_k = λ_0 · decay^k（decay = 0 时恒为 λ_0）。
        let effective_shift = if options.level_shift_decay > 0.0 {
            let factor = options.level_shift_decay.powi(step as i32);
            let value = options.level_shift * factor;
            if value < 1.0e-4 { 0.0 } else { value }
        } else {
            options.level_shift
        };
        let fock_alpha = assemble_fock(
            &hcore,
            &coulomb,
            &xc.v_alpha,
            &exchange_alpha,
            options.hf_exchange,
            effective_shift,
            &density_alpha,
            overlap,
        )?;
        let fock_beta = assemble_fock(
            &hcore,
            &coulomb,
            &xc.v_beta,
            &exchange_beta,
            options.hf_exchange,
            effective_shift,
            &density_beta,
            overlap,
        )?;

        // DIIS（每个通道一套）。
        let (mut f_alpha, mut f_beta) = (fock_alpha.clone(), fock_beta.clone());
        let err_alpha = Diis::commutator(&fock_alpha, &density_alpha, overlap).ok();
        let err_beta = Diis::commutator(&fock_beta, &density_beta, overlap).ok();
        if let (Some(alpha), Some(beta), Some(err_alpha), Some(err_beta)) =
            (diis_alpha.as_mut(), diis_beta.as_mut(), err_alpha, err_beta)
        {
            alpha.push(fock_alpha.clone(), err_alpha);
            beta.push(fock_beta.clone(), err_beta);
            if alpha.len() >= 2
                && let Ok(Some(extrapolated)) = alpha.extrapolate()
            {
                f_alpha = extrapolated;
            }
            if beta.len() >= 2
                && let Ok(Some(extrapolated)) = beta.extrapolate()
            {
                f_beta = extrapolated;
            }
        }

        let (new_alpha, energies_alpha, coeffs_alpha) = density_from_fock(&f_alpha, &x, n_alpha)?;
        let (new_beta, energies_beta, coeffs_beta) = density_from_fock(&f_beta, &x, n_beta)?;
        mo_energies_alpha = energies_alpha;
        mo_energies_beta = energies_beta;
        coefficients_alpha = coeffs_alpha;
        coefficients_beta = coeffs_beta;

        // 能量：**所有项都用同一套（更新后的）密度**。
        //
        // 原来的写法是「新密度的 2e 项 + 旧密度的 XC 能量」，那种混用让能量不是任何一套
        // 密度的泛函值，SCF 的迭代映射因此可能没有不动点 —— OH/SVWN5 就是这样：19 种
        // 配方全都不收敛（`docs/choices.md` 七十六）。这里改成用新密度重算 J/K/XC
        // （每步多 2 次 Fock 构建 + 1 次 XC 求值，STO-3G 上是毫秒量级）。
        let total_new = add(&new_alpha, &new_beta)?;
        let coulomb_new = build_fock(&hcore, tensor, &total_new)
            .map_err(linalg)?
            .coulomb;
        let (exchange_alpha_new, exchange_beta_new) = if options.hf_exchange.abs() > 0.0 {
            (
                build_fock(&hcore, tensor, &new_alpha)
                    .map_err(linalg)?
                    .exchange,
                build_fock(&hcore, tensor, &new_beta)
                    .map_err(linalg)?
                    .exchange,
            )
        } else {
            (Matrix::zeros(n, n), Matrix::zeros(n, n))
        };
        let xc_new = evaluate_xc_spin_cached(basis, grid, &cache, &new_alpha, &new_beta, terms)?;
        xc_energy = xc_new.energy_hartree;
        let mut tr_ph = 0.0;
        let mut tr_pj = 0.0;
        let mut tr_pk = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr_ph += new_alpha[(i, j)] * hcore[(i, j)] + new_beta[(i, j)] * hcore[(i, j)];
                tr_pj += total_new[(i, j)] * coulomb_new[(i, j)];
                tr_pk += new_alpha[(i, j)] * exchange_alpha_new[(i, j)]
                    + new_beta[(i, j)] * exchange_beta_new[(i, j)];
            }
        }
        electronic = tr_ph + 0.5 * tr_pj + xc_energy - 0.5 * options.hf_exchange * tr_pk;
        let energy = electronic + e_nuc;
        last_change = if previous_energy.is_nan() {
            f64::INFINITY
        } else {
            (energy - previous_energy).abs()
        };
        previous_energy = energy;

        let damping = options.damping.clamp(0.0, 1.0);
        let mixed_alpha = mix(&density_alpha, &new_alpha, damping);
        let mixed_beta = mix(&density_beta, &new_beta, damping);
        let max_dp = max_abs_difference(&density_alpha, &mixed_alpha)
            .max(max_abs_difference(&density_beta, &mixed_beta));
        density_alpha = mixed_alpha;
        density_beta = mixed_beta;

        if std::env::var_os("QWB_SCF_TRACE").is_some() {
            eprintln!(
                "uks step {iterations}: E = {energy:+.9}  dE = {last_change:.3e}  max|dP| = {max_dp:.3e}  E_xc = {xc_energy:+.9}"
            );
        }

        if last_change < options.energy_threshold && max_dp < options.density_threshold {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(EngineError::ScfNotConverged {
            iters: iterations,
            max: options.max_iter,
        });
    }

    let s2 = crate::uhf::s_squared(&density_alpha, &density_beta, overlap)?;
    let spin = 0.5 * ((n_alpha as f64) - (n_beta as f64));
    let total = add(&density_alpha, &density_beta)?;
    Ok(UksResult {
        energy_hartree: previous_energy,
        scf_converged: converged,
        iterations,
        electronic_hartree: electronic,
        nuclear_repulsion_hartree: e_nuc,
        xc_energy_hartree: xc_energy,
        density: total,
        density_alpha,
        density_beta,
        mo_energies_alpha,
        mo_energies_beta,
        coefficients_alpha,
        coefficients_beta,
        n_alpha,
        n_beta,
        s_squared: s2,
        spin_contamination: s2 - spin * (spin + 1.0),
        last_energy_change: last_change,
    })
}

/// `F_σ = H + J + v_σ − α_hf·K_σ`（可选能级移动）。
// 数学签名：F = H + J + v_xc + hybrid·K，每一项都是独立的矩阵，再加能级移动的两个矩阵。
#[allow(clippy::too_many_arguments)]
fn assemble_fock(
    hcore: &Matrix,
    coulomb: &Matrix,
    v_xc: &Matrix,
    exchange: &Matrix,
    hf_exchange: f64,
    level_shift: f64,
    density: &Matrix,
    overlap: &Matrix,
) -> Result<Matrix, EngineError> {
    let n = hcore.nrows();
    let mut fock = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            fock[(i, j)] =
                hcore[(i, j)] + coulomb[(i, j)] + v_xc[(i, j)] - hf_exchange * exchange[(i, j)];
        }
    }
    if level_shift > 0.0 {
        let sp = matmul(overlap, density).map_err(linalg)?;
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] += level_shift * (overlap[(i, j)] - sp[(i, j)]);
            }
        }
    }
    Ok(fock)
}

/// `P_α + P_β`。
fn add(left: &Matrix, right: &Matrix) -> Result<Matrix, EngineError> {
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        return Err(EngineError::Internal(
            "density matrices have different shapes".into(),
        ));
    }
    let mut out = Matrix::zeros(left.nrows(), left.ncols());
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            out.set(i, j, left.get(i, j) + right.get(i, j));
        }
    }
    Ok(out)
}

/// 对角化 `F`（`X` 正交基下）并取前 `n_occ` 个轨道构造密度（不带因子 2）。
fn density_from_fock(
    fock: &Matrix,
    x: &Matrix,
    n_occ: usize,
) -> Result<(Matrix, Vector, Matrix), EngineError> {
    let fx = matmul(fock, x).map_err(linalg)?;
    let mut f_tilde = Matrix::zeros(x.nrows(), x.ncols());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let mut sum = 0.0;
            for k in 0..x.nrows() {
                sum += x.get(k, i) * fx.get(k, j);
            }
            f_tilde.set(i, j, sum);
        }
    }
    let (values, vectors) = symmetric_eigen(&f_tilde).map_err(linalg)?;
    if values.len() < n_occ {
        return Err(EngineError::Internal(
            "not enough orbitals for the requested occupation".into(),
        ));
    }
    let n = fock.nrows();
    let mut coefficients = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x.get(i, k) * vectors.get(k, j);
            }
            coefficients.set(i, j, sum);
        }
    }
    let mut density = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n_occ {
                sum += coefficients.get(i, k) * coefficients.get(j, k);
            }
            density.set(i, j, sum);
        }
    }
    Ok((density, values, coefficients))
}

/// 由注册表数据直接跑 UKS（`rks_from_data` 的自旋版）。
///
/// # Errors
///
/// 数据缺项（基组/泛函/网格）或 SCF 不收敛。
pub fn uks_from_data(
    registry: &qwb_registry::Registry,
    molecule: &Molecule,
    basis_name: &str,
    functional_name: &str,
    grid_name: &str,
    options: UksOptions,
) -> Result<UksResult, EngineError> {
    let basis_k1 = registry.get_basis(basis_name)?;
    let functional = registry.get_functional(functional_name)?;
    let grid_level = registry.get_grid(grid_name)?;
    let basis = qwb_basis::InternalBasis::build(basis_k1, molecule)?;
    let tensor = qwb_integrals::two_electron::two_electron(&basis, molecule)?;
    let grid = qwb_grid::molecular_grid(grid_level, molecule)
        .map_err(|err| EngineError::Data(format!("grid: {err}")))?;

    let parameters: std::collections::BTreeMap<String, f64> = functional.parameters.clone();
    let mut options = options;
    let mut terms: Vec<Box<dyn FunctionalTerm>> = Vec::new();
    for component in &functional.components {
        if component.id == "HfExchange" {
            options.hf_exchange += component.weight;
            continue;
        }
        let term = qwb_xc::build_term(&component.id, &parameters)
            .map_err(|err| EngineError::Data(format!("functional `{functional_name}`: {err}")))?;
        terms.push(qwb_xc::scale_term(term, component.weight));
    }
    uks_scf(&basis, molecule, &tensor, &grid, &terms, options)
}
