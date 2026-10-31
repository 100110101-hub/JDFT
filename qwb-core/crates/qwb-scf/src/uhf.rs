//! SC-01（phase-2 §6.5）：**UHF**（非限制性 Hartree–Fock）与自旋诊断。
//!
//! # 为什么先做 UHF 而不是 UKS
//!
//! UKS 需要给每个泛函项写**自旋插值**（交换的 `½[E_x(2ρ_α)+E_x(2ρ_β)]`、
//! VWN5/PBE-C/LYP 的相关自旋插值），每一项都要单独对照；UHF 只有交换项，且形式上
//! 与闭壳层 RHF 只差"每个自旋通道用自己的密度算交换"。所以先用 UHF 把**自旋机制**
//! （双套密度、⟨S²⟩、污染报告、闭壳层退化）钉死，UKS 再在这个骨架上加 XC（`SC-01b`）。
//!
//! # 公式
//!
//! ```text
//! P      = P_α + P_β
//! J      = J(P)                     （库仑用**总**密度）
//! F_σ    = H_core + J − K(P_σ)      （交换只用**本通道**密度）
//! E_elec = ½ Tr[P_α (H + F_α)] + ½ Tr[P_β (H + F_β)]
//! E      = E_elec + E_nuc
//! ```
//!
//! 闭壳层退化（对 H2/STO-3G 实测到 1e-12）：`P_α = P_β = P/2` 时
//! `K(P_σ) = ½K(P)`，于是 `F_σ = H + J − ½K(P) = F_RHF`，能量与 RHF 逐位相同。
//!
//! # ⟨S²⟩ 与自旋污染
//!
//! ```text
//! ⟨S²⟩ = S_z(S_z + 1) + n_β − Tr[(P_α S)(P_β S)]
//! S_z  = (n_α − n_β)/2
//! ```
//!
//! **这是 UHF 的卖点也是它的病**：UHF 不强制纯自旋态，所以 O2 三重态实测
//! `⟨S²⟩ = 2.0034`（不是 2.0）、OH 双峰实测 `0.7533`（不是 0.75）。
//! 参考值来自 PySCF 黑盒（`tests/reference/uhf_open_shell.csv`），而且**污染量必须报出来**：
//! "收敛了就不提自旋"是不可接受的（phase-2 §4.4 门禁 ②）。
//!
//! 想要纯自旋态请用 ROHF（`SC-01b`）—— 那里 `⟨S²⟩ = S(S+1)` 精确成立。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::{Matrix, Vector, matmul, symmetric_eigen};

use crate::build_fock;
use crate::diis::Diis;
use crate::rks::{default_charges, inverse_sqrt_symmetric, linalg, nuclear_repulsion};

/// 初猜策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum UhfInitialGuess {
    /// `H_core` 对角化（phase-1 的老做法）。快，但近简并体系会落到较高的解上。
    Core,
    /// 原子密度叠加（`crate::guess::sad_guess`）。球对称、不破坏简并，是默认做法。
    #[default]
    AtomicDensities,
}

/// UHF 控制参数（与 `RksOptions` 的对应字段语义相同；检查点/展宽暂不支持）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UhfOptions {
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
    /// 静态能级移动 `λ`；`F_σ ← F_σ + λ (S − S P_σ S)`。
    pub level_shift: f64,
    /// 能级移动的**逐步衰减因子**（`0` = 不衰减，用恒定 `λ`）。
    ///
    /// 与 [`crate::uks::UksOptions::level_shift_decay`] 同义：`λ_k = λ_0 · decay^k`，
    /// 衰减到 1e-4 以下就当 0。静态移位只能把 SCF 停到「移位哈密顿量的不动点」（不是驻点）；
    /// 逐步退火才可能被带到真驻点上（`docs/choices.md` 七十六）。
    pub level_shift_decay: f64,
    /// 初猜的非对角缩放（多初猜用；默认 1.0 = 纯 core guess）。
    pub initial_guess_scale: f64,
    /// 初猜策略（默认原子密度叠加）。
    pub initial_guess: UhfInitialGuess,
    /// 每个原子的核电荷（ECP 体系传 `z_eff`）。
    pub nuclear_charges: Option<Vec<f64>>,
    /// ECP 的电子-核有效势矩阵元。
    pub ecp_potential: Option<Matrix>,
}

impl Default for UhfOptions {
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
        }
    }
}

/// UHF 结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UhfResult {
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
    /// **总**密度 `P_α + P_β`（偶极/多极等只关心电荷分布的量用它）。
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
    /// 自旋污染 `⟨S²⟩ − S(S+1)`（纯自旋态 ≈ 0）。
    pub spin_contamination: f64,
    /// 最后一次能量变化。
    pub last_energy_change: f64,
}

/// 电子数与自旋多重度 → `(n_α, n_β)`。
///
/// # Errors
///
/// 电子数非正、多重度与电子数奇偶性矛盾、或 `n_β < 0` 时返回 [`EngineError::InvalidInput`]。
pub(crate) fn split_spins(
    electrons: i64,
    multiplicity: u32,
) -> Result<(usize, usize), EngineError> {
    if electrons <= 0 {
        return Err(EngineError::InvalidInput(
            "open-shell SCF needs a positive electron count".into(),
        ));
    }
    if multiplicity == 0 {
        return Err(EngineError::InvalidInput(
            "multiplicity must be >= 1".into(),
        ));
    }
    let unpaired = i64::from(multiplicity) - 1;
    // N 与 (2S) 必须同奇偶：否则"n_α − n_β = 2S"解不出整数。
    if (electrons - unpaired) % 2 != 0 {
        return Err(EngineError::InvalidInput(format!(
            "multiplicity {multiplicity} is inconsistent with {electrons} electrons \
             (n_alpha - n_beta must be {}), the electron count decides the parity",
            unpaired
        )));
    }
    let n_alpha = (electrons + unpaired) / 2;
    let n_beta = (electrons - unpaired) / 2;
    if n_beta < 0 {
        return Err(EngineError::InvalidInput(format!(
            "multiplicity {multiplicity} needs more unpaired electrons than {electrons} \
             electrons can provide"
        )));
    }
    Ok((n_alpha as usize, n_beta as usize))
}

/// `⟨S²⟩ = S_z(S_z+1) + n_β − Tr[(P_α S)(P_β S)]`。
///
/// 两个电子数由密度的迹给出（`Tr[P_σ S]`），这样"算 ⟨S²⟩"与"用掉的电子数"永远一致。
///
/// # Errors
///
/// 矩阵尺寸不一致。
pub fn s_squared(
    density_alpha: &Matrix,
    density_beta: &Matrix,
    overlap: &Matrix,
) -> Result<f64, EngineError> {
    let pa_s = matmul(density_alpha, overlap).map_err(linalg)?;
    let pb_s = matmul(density_beta, overlap).map_err(linalg)?;
    let mut overlap_trace = 0.0;
    for i in 0..pa_s.nrows() {
        for k in 0..pa_s.ncols() {
            overlap_trace += pa_s.get(i, k) * pb_s.get(k, i);
        }
    }
    let n_alpha = trace_of(&pa_s)?;
    let n_beta = trace_of(&pb_s)?;
    let s_z = 0.5 * (n_alpha - n_beta);
    Ok(s_z * (s_z + 1.0) + n_beta - overlap_trace)
}

/// 迹。
fn trace_of(matrix: &Matrix) -> Result<f64, EngineError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(EngineError::Internal("trace needs a square matrix".into()));
    }
    let mut trace = 0.0;
    for i in 0..matrix.nrows() {
        trace += matrix.get(i, i);
    }
    Ok(trace)
}

/// 求自旋通道的 Fock 与能量。
pub(crate) fn fock_and_energy(
    hcore: &Matrix,
    tensor: &TwoElectronIntegrals,
    density_alpha: &Matrix,
    density_beta: &Matrix,
    level_shift: f64,
    overlap: &Matrix,
) -> Result<(Matrix, Matrix, f64, f64, f64), EngineError> {
    // J 用总密度；K 用本通道密度。`build_fock` 一次给 (J, K, F)，所以调两次：
    // 一次用 P_α（拿 K_α），一次用 P_β（拿 K_β），J 加起来即可（J 对密度线性）。
    let alpha = build_fock(hcore, tensor, density_alpha).map_err(linalg)?;
    let beta = build_fock(hcore, tensor, density_beta).map_err(linalg)?;
    let n = hcore.nrows();
    let mut fock_alpha = Matrix::zeros(n, n);
    let mut fock_beta = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let coulomb = alpha.coulomb.get(i, j) + beta.coulomb.get(i, j);
            fock_alpha.set(i, j, hcore.get(i, j) + coulomb - alpha.exchange.get(i, j));
            fock_beta.set(i, j, hcore.get(i, j) + coulomb - beta.exchange.get(i, j));
        }
    }
    if level_shift > 0.0 {
        apply_level_shift(&mut fock_alpha, density_alpha, overlap, level_shift);
        apply_level_shift(&mut fock_beta, density_beta, overlap, level_shift);
    }
    // E_elec = ½ Σ_σ Tr[P_σ (H + F_σ)]。注意 `F_σ` **已经含 H**，所以不能只乘 F ——
    // 少这一项会让单电子能量少一半（H2/STO-3G 实测差 1.25 Ha，正是这一项）。
    let e_alpha =
        0.5 * (trace_product(density_alpha, &fock_alpha)? + trace_product(density_alpha, hcore)?);
    let e_beta =
        0.5 * (trace_product(density_beta, &fock_beta)? + trace_product(density_beta, hcore)?);
    Ok((fock_alpha, fock_beta, e_alpha + e_beta, e_alpha, e_beta))
}

/// `F ← F + λ (S − S P S)`（把虚轨道抬高 λ）。
fn apply_level_shift(fock: &mut Matrix, density: &Matrix, overlap: &Matrix, shift: f64) {
    let Ok(sp) = matmul(overlap, density) else {
        return;
    };
    let Ok(sps) = matmul(&sp, overlap) else {
        return;
    };
    let n = fock.nrows();
    for i in 0..n {
        for j in 0..n {
            fock.set(
                i,
                j,
                fock.get(i, j) + shift * (overlap.get(i, j) - sps.get(i, j)),
            );
        }
    }
}

/// `½ Tr[P F]`。
pub(crate) fn trace_product(density: &Matrix, fock: &Matrix) -> Result<f64, EngineError> {
    let mut trace = 0.0;
    for i in 0..density.nrows() {
        for k in 0..density.ncols() {
            trace += density.get(i, k) * fock.get(k, i);
        }
    }
    Ok(trace)
}

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

pub(crate) fn max_abs_difference(left: &Matrix, right: &Matrix) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            worst = worst.max((left.get(i, j) - right.get(i, j)).abs());
        }
    }
    worst
}

/// 对角化 `F`（在 `X` 的正交化基下）并取前 `n_occ` 个轨道构造密度。
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
    // C = X · C_tilde
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

/// 非限制性 Hartree–Fock（SC-01）。
///
/// # Errors
///
/// - 电子数/多重度不自洽 → [`EngineError::InvalidInput`]；
/// - 2e 张量尺寸不匹配、ECP 势尺寸不对 → [`EngineError::Internal`]；
/// - `max_iter` 步内不收敛 → [`EngineError::ScfNotConverged`]。
pub fn uhf_scf(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    tensor: &TwoElectronIntegrals,
    options: UhfOptions,
) -> Result<UhfResult, EngineError> {
    let n = basis.n_functions();
    if tensor.n_functions != n {
        return Err(EngineError::Internal(
            "two-electron tensor size does not match the basis".to_string(),
        ));
    }
    let charges = match &options.nuclear_charges {
        Some(charges) => {
            if charges.len() != molecule.n_atoms() {
                return Err(EngineError::Internal(format!(
                    "UhfOptions::nuclear_charges has {} entries for {} atoms",
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
                "UhfOptions::ecp_potential is {}x{}, expected {n}x{n}",
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
    // 初猜：对 H_core（可缩放）做一次对角化，α 取前 n_α 个、β 取前 n_β 个轨道。
    //
    // 注意**不能**用 `rks::core_guess`：那个函数给的是闭壳层**总**密度（里面乘了 2），
    // 拿来做 P_α 会让电子数翻倍。自旋通道的密度是 `Σ_{i<n_σ} C Cᵀ`，不带因子 2。
    let (mut density_alpha, mut density_beta) = match options.initial_guess {
        UhfInitialGuess::AtomicDensities => {
            let (alpha, beta) = crate::guess::sad_guess(basis, molecule)?;
            // 全零 = SAD 不可用（已警告），退回 core guess。
            if alpha.as_column_major().iter().all(|value| *value == 0.0) {
                let (alpha, _, _) = density_from_fock(&guess_hcore, &x, n_alpha)?;
                let beta = if n_beta == n_alpha {
                    alpha.clone()
                } else {
                    density_from_fock(&guess_hcore, &x, n_beta)?.0
                };
                (alpha, beta)
            } else {
                (alpha, beta)
            }
        }
        UhfInitialGuess::Core => {
            let (alpha, _, _) = density_from_fock(&guess_hcore, &x, n_alpha)?;
            let beta = if n_beta == n_alpha {
                alpha.clone()
            } else {
                density_from_fock(&guess_hcore, &x, n_beta)?.0
            };
            (alpha, beta)
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

    let mut current_shift = options.level_shift;
    for step in 0..options.max_iter {
        iterations = step + 1;
        let (fock_alpha, fock_beta, e_elec, _, _) = fock_and_energy(
            &hcore,
            tensor,
            &density_alpha,
            &density_beta,
            current_shift,
            overlap,
        )?;
        electronic = e_elec;
        // 退火：λ_k = λ_0 · decay^k（decay <= 0 表示不衰减；衰减到 1e-4 以下直接关掉）。
        if options.level_shift_decay > 0.0 && options.level_shift_decay < 1.0 {
            let decayed = options.level_shift * options.level_shift_decay.powi(step as i32 + 1);
            current_shift = if decayed < 1.0e-4 { 0.0 } else { decayed };
        }
        let energy = e_elec + e_nuc;
        last_change = (energy - previous_energy).abs();
        previous_energy = energy;

        // DIIS 外推（每个自旋通道一套；误差向量是各自的对易子）。
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

        let damping = options.damping.clamp(0.0, 1.0);
        let mixed_alpha = mix(&density_alpha, &new_alpha, damping);
        let mixed_beta = mix(&density_beta, &new_beta, damping);
        let max_dp = max_abs_difference(&density_alpha, &mixed_alpha)
            .max(max_abs_difference(&density_beta, &mixed_beta));
        density_alpha = mixed_alpha;
        density_beta = mixed_beta;

        if std::env::var_os("QWB_SCF_TRACE").is_some() {
            eprintln!(
                "uhf step {iterations}: E = {energy:+.9}  dE = {last_change:.3e}  max|dP| = {max_dp:.3e}"
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

    let s2 = s_squared(&density_alpha, &density_beta, overlap)?;
    let spin = 0.5 * ((n_alpha as f64) - (n_beta as f64));
    let total = add(&density_alpha, &density_beta)?;
    Ok(UhfResult {
        energy_hartree: previous_energy,
        scf_converged: converged,
        iterations,
        electronic_hartree: electronic,
        nuclear_repulsion_hartree: e_nuc,
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

/// `P ← (1 − d) P_old + d P_new`。
pub(crate) fn mix(old: &Matrix, new: &Matrix, damping: f64) -> Matrix {
    if damping <= 0.0 {
        return new.clone();
    }
    let mut out = Matrix::zeros(old.nrows(), old.ncols());
    for i in 0..old.nrows() {
        for j in 0..old.ncols() {
            out.set(
                i,
                j,
                (1.0 - damping) * old.get(i, j) + damping * new.get(i, j),
            );
        }
    }
    out
}
