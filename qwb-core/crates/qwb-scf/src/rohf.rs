//! ROHF（限制性开壳层 HF）：**Roothaan 的等效 Fock**，纯自旋。
//!
//! 开壳层体系用 UHF 会自旋污染（`⟨S²⟩ ≠ S(S+1)`）；ROHF 让闭壳层与开壳层**共用空间轨道**，
//! 于是波函数是纯自旋的。代价是"Fock 矩阵"不再唯一 —— 必须构造一个**等效 Fock**，它的本征
//! 向量给出正确的驻点。这里用 Roothaan 的构造（与 PySCF 的 `get_roothaan_fock` 同一个表）：
//!
//! ```text
//!          closed   open   virtual
//! closed    Fc      Fb      Fc
//! open      Fb      Fc      Fa
//! virtual   Fc      Fa      Fc      （Fc = (Fa + Fb)/2）
//! ```
//!
//! 其中 `Fa`/`Fb` 由**同一组空间轨道**构造的两份密度给出：
//!
//! ```text
//! D_c = Σ_{i<闭壳层} C_i C_iᵀ,   D_o = Σ_{开壳层} C_i C_iᵀ
//! P_α = D_c + D_o,  P_β = D_c        （总密度 2D_c + D_o）
//! Fa = h + J(P_α+P_β) − K(P_α),  Fb = h + J(P_α+P_β) − K(P_β)
//! ```
//!
//! 投影子 `P_c = D_c S`、`P_o = D_o S`、`P_v = I − (D_c+D_o) S`，组装
//!
//! ```text
//! F_eff = ½(P_cᵀF_cP_c + P_oᵀF_cP_o + P_vᵀF_cP_v) + P_oᵀF_bP_c + P_oᵀF_aP_v + P_vᵀF_cP_c + h.c.
//! ```
//!
//! **返回的 `⟨S²⟩` 是 `S(S+1)` 精确值**（波函数按 CSF 算），而不是单个行列式的
//! `⟨S²⟩`（那个仍会略大于 `S(S+1)`）。这与 PySCF 的 `ROHF.spin_square` 一致。
//!
//! 能量仍按"行列式 + 这两份密度"的公式算：`E = ½[Tr(P_α(h+Fa)) + Tr(P_β(h+Fb))] + E_nuc`
//! —— 对 ROHF 的驻点成立（PySCF 的 `energy_tot` 同样这么算）。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::{Matrix, Vector, matmul, symmetric_eigen};

use crate::diis::Diis;
use crate::rks::{default_charges, inverse_sqrt_symmetric, linalg, nuclear_repulsion};
use crate::uhf::{UhfOptions, fock_and_energy, split_spins};

/// ROHF 的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RohfResult {
    /// 总能量（Hartree）。
    pub energy_hartree: f64,
    /// 空间轨道系数（列；前 `n_closed` 个是闭壳层、接着 `n_open` 个是开壳层）。
    pub coefficients: Matrix,
    /// **等效 Fock** 的本征值。注意它们**不是** Koopmans 轨道能（Roothaan 等效 Fock 的性质）。
    pub mo_energies: Vector,
    /// 闭壳层密度 `D_c`。
    pub density_closed: Matrix,
    /// 开壳层密度 `D_o`。
    pub density_open: Matrix,
    /// SCF 迭代步数。
    pub iterations: u32,
    /// 是否收敛。
    pub scf_converged: bool,
    /// `⟨S²⟩`：纯自旋的精确值 `S(S+1)`。
    pub s_squared: f64,
    /// α 电子数（= 闭壳层数 + 开壳层数）。
    pub n_alpha: usize,
    /// β 电子数（= 闭壳层数）。
    pub n_beta: usize,
    /// 电子能量（不含核排斥）。
    pub electronic_hartree: f64,
    /// 总密度 `2D_c + D_o`（偶极矩等一次性性质用它）。
    pub density_total: Matrix,
}

/// 按 **Roothaan 的占据规则**选轨道（PySCF `_fill_rohf_occ` 同款）：
///
/// * 闭壳层 = 等效 Fock 本征值最低的 `n_closed` 个；
/// * 开壳层 = **其余**轨道里 `⟨φ_i|F_a|φ_i⟩` 最低的 `n_open` 个。
///
/// 第二句是必须的：直接按等效 Fock 本征值排序会把开壳层放到错的轨道上（O2/OH 上差几十 mHa），
/// 因为等效 Fock 的本征值不是轨道能。
fn occupations(
    routhaan_energies: &Vector,
    diagonal_alpha: &Vector,
    n_closed: usize,
    n_open: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = routhaan_energies.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        routhaan_energies
            .get(*a)
            .total_cmp(&routhaan_energies.get(*b))
    });
    let closed: Vec<usize> = order[..n_closed].to_vec();
    let mut rest: Vec<usize> = order[n_closed..].to_vec();
    rest.sort_by(|a, b| diagonal_alpha.get(*a).total_cmp(&diagonal_alpha.get(*b)));
    let open: Vec<usize> = rest[..n_open].to_vec();
    (closed, open)
}

/// 按列下标构造密度 `Σ_{i∈indices} C_i C_iᵀ`。
fn density_from_indices(coefficients: &Matrix, indices: &[usize]) -> Result<Matrix, EngineError> {
    let n = coefficients.nrows();
    let mut out = Matrix::zeros(n, n);
    for &column in indices {
        for i in 0..n {
            let ci = coefficients.get(i, column);
            if ci == 0.0 {
                continue;
            }
            for j in 0..n {
                out[(i, j)] += ci * coefficients.get(j, column);
            }
        }
    }
    Ok(out)
}

/// `⟨φ_i|F|φ_i⟩`（Roothaan 轨道已正交归一，不需要 S）。
fn diagonal_expectation(coefficients: &Matrix, fock: &Matrix) -> Vector {
    let n = coefficients.nrows();
    let mut out = Vector::zeros(n);
    for column in 0..n {
        let mut sum = 0.0;
        for mu in 0..n {
            let c_mu = coefficients.get(mu, column);
            if c_mu == 0.0 {
                continue;
            }
            for nu in 0..n {
                sum += c_mu * fock.get(mu, nu) * coefficients.get(nu, column);
            }
        }
        out.set(column, sum);
    }
    out
}

/// 投影子 `P = D·S`（`D` 是某一段轨道的密度）。
fn projector(density: &Matrix, overlap: &Matrix) -> Result<Matrix, EngineError> {
    matmul(density, overlap).map_err(linalg)
}

/// 把 `fock` 在正交基下对角化，返回 `(本征值, C = X·C̃)`。
fn orbitals_from_fock(fock: &Matrix, x: &Matrix) -> Result<(Vector, Matrix), EngineError> {
    let fx = matmul(fock, x).map_err(linalg)?;
    let n = x.nrows();
    let mut f_tilde = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x.get(k, i) * fx.get(k, j);
            }
            f_tilde.set(i, j, sum);
        }
    }
    let (values, vectors) = symmetric_eigen(&f_tilde).map_err(linalg)?;
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
    Ok((values, coefficients))
}

/// Roothaan 的等效 Fock（见模块文档的表）。
fn effective_fock(
    fock_alpha: &Matrix,
    fock_beta: &Matrix,
    p_closed: &Matrix,
    p_open: &Matrix,
    p_virtual: &Matrix,
) -> Result<Matrix, EngineError> {
    let n = fock_alpha.nrows();
    let mut fock_closed = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            fock_closed.set(i, j, 0.5 * (fock_alpha.get(i, j) + fock_beta.get(i, j)));
        }
    }
    // 每个块：**P_leftᵀ** · F · P_right（P = D·S 不对称，漏掉左边的转置会把块算错）。
    let block = |left: &Matrix, fock: &Matrix, right: &Matrix| -> Result<Matrix, EngineError> {
        let mut transposed = Matrix::zeros(left.ncols(), left.nrows());
        for i in 0..left.nrows() {
            for j in 0..left.ncols() {
                transposed[(j, i)] = left.get(i, j);
            }
        }
        let first = matmul(&transposed, fock).map_err(linalg)?;
        matmul(&first, right).map_err(linalg)
    };
    let mut fock = Matrix::zeros(n, n);
    // 对角块乘 ½、交叉块乘 1 —— 与 PySCF `get_roothaan_fock` 的系数逐项对应：
    // 对角块在下面的 `fock + fockᵀ` 里翻倍成 1，交叉块翻倍成 2（即两侧对称块各 1）。
    for (left, middle, right, weight) in [
        (p_closed, &fock_closed, p_closed, 0.5),
        (p_open, &fock_closed, p_open, 0.5),
        (p_virtual, &fock_closed, p_virtual, 0.5),
        (p_open, fock_beta, p_closed, 1.0),
        (p_open, fock_alpha, p_virtual, 1.0),
        (p_virtual, &fock_closed, p_closed, 1.0),
    ] {
        let piece = block(left, middle, right)?;
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] += weight * piece.get(i, j);
            }
        }
    }
    // 对称化：F_eff = fock + fockᵀ
    let mut symmetric = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            symmetric[(i, j)] = fock.get(i, j) + fock.get(j, i);
        }
    }
    Ok(symmetric)
}

/// ROHF SCF（纯自旋）。`options` 复用 [`UhfOptions`] 的收敛/DIIS/预算字段。
///
/// # Errors
///
/// 电子数与多重度不自洽、两电子张量尺寸不符、未在预算内收敛（[`EngineError::ScfNotConverged`]）。
pub fn rohf_scf(
    basis: &InternalBasis,
    molecule: &Molecule,
    tensor: &TwoElectronIntegrals,
    options: &UhfOptions,
) -> Result<RohfResult, EngineError> {
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
    // 开壳层 = 未配对电子数；ROHF 要求它们全部在 α 通道（`n_alpha >= n_beta` 由 split_spins 保证）。
    let n_closed = n_beta;
    let n_open = n_alpha - n_beta;
    if n_open == 0 {
        return Err(EngineError::InvalidInput(
            "rohf: multiplicity 1 has no open shell; use RHF".to_string(),
        ));
    }

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

    // 初猜：H_core 的轨道，按本征值排序（symmetric_eigen 升序）取闭壳层/开壳层。
    let (_, guess_coefficients) = orbitals_from_fock(&hcore, &x)?;
    let mut coefficients = guess_coefficients;

    let mut diis = if options.use_diis {
        Some(Diis::new(options.diis_max_vectors))
    } else {
        None
    };
    let mut previous_energy = f64::NAN;
    let mut iterations = 0u32;

    // 占据在迭代之间**传递**：上一轮按 Roothaan 规则选出的闭/开壳层下标，这一轮就用它建密度。
    // （初猜按本征值升序切成 0..n_closed 与 n_closed..n_closed+n_open。）
    let mut closed_indices: Vec<usize> = (0..n_closed).collect();
    let mut open_indices: Vec<usize> = (n_closed..n_closed + n_open).collect();
    let (energy_hartree, mo_energies, rho_closed, rho_open) = loop {
        let closed = density_from_indices(&coefficients, &closed_indices)?;
        let open = density_from_indices(&coefficients, &open_indices)?;
        let mut alpha_density = closed.clone();
        for i in 0..n {
            for j in 0..n {
                alpha_density[(i, j)] += open.get(i, j);
            }
        }
        let (fock_alpha, fock_beta, e_elec, _, _) =
            fock_and_energy(&hcore, tensor, &alpha_density, &closed, 0.0, overlap)?;
        let energy = e_elec + e_nuc;
        let change = (energy - previous_energy).abs();
        previous_energy = energy;

        let p_closed = projector(&closed, overlap)?;
        let p_open = projector(&open, overlap)?;
        let mut occupied = closed.clone();
        for i in 0..n {
            for j in 0..n {
                occupied[(i, j)] += open.get(i, j);
            }
        }
        let occupied_projector = projector(&occupied, overlap)?;
        let mut p_virtual = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let identity = if i == j { 1.0 } else { 0.0 };
                p_virtual[(i, j)] = identity - occupied_projector.get(i, j);
            }
        }
        let mut fock = effective_fock(&fock_alpha, &fock_beta, &p_closed, &p_open, &p_virtual)?;

        // DIIS：误差向量取总密度的对易子（与 UHF 同一个判据）。
        let mut total = closed.clone();
        for i in 0..n {
            for j in 0..n {
                total[(i, j)] = 2.0 * closed.get(i, j) + open.get(i, j);
            }
        }
        if let (Some(space), Some(error)) =
            (diis.as_mut(), Diis::commutator(&fock, &total, overlap).ok())
        {
            space.push(fock.clone(), error);
            if space.len() >= 2
                && let Ok(Some(extrapolated)) = space.extrapolate()
            {
                fock = extrapolated;
            }
        }

        let (values, new_coefficients) = orbitals_from_fock(&fock, &x)?;
        let previous_coefficients = coefficients;
        coefficients = new_coefficients;
        iterations += 1;

        // 占据：闭壳层按等效 Fock 本征值、开壳层按 `⟨F_a⟩` 对角值（见 `occupations`）。
        let diagonal_alpha = diagonal_expectation(&coefficients, &fock_alpha);
        let (closed_indices_next, open_indices_next) =
            occupations(&values, &diagonal_alpha, n_closed, n_open);
        let new_closed = density_from_indices(&coefficients, &closed_indices_next)?;
        let new_open = density_from_indices(&coefficients, &open_indices_next)?;
        let mut max_change = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let new_total = 2.0 * new_closed.get(i, j) + new_open.get(i, j);
                max_change = max_change.max((new_total - total.get(i, j)).abs());
            }
        }
        let _ = &previous_coefficients;
        if std::env::var_os("QWB_SCF_TRACE").is_some() {
            eprintln!(
                "rohf step {iterations}: E = {energy:+.9}  dE = {change:.3e}  max|dP| = {max_change:.3e}  occ = {closed_indices_next:?} + {open_indices_next:?}"
            );
        }
        if max_change < options.density_threshold && change < options.energy_threshold {
            break (energy, values, new_closed, new_open);
        }
        closed_indices = closed_indices_next;
        open_indices = open_indices_next;
        if iterations >= options.max_iter {
            return Err(EngineError::ScfNotConverged {
                iters: iterations,
                max: options.max_iter,
            });
        }
    };

    // 走到这里一定收敛过（不收敛在上面就 `Err` 了），所以 `scf_converged` 恒为 true。
    let mut density_total = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            density_total[(i, j)] = 2.0 * rho_closed.get(i, j) + rho_open.get(i, j);
        }
    }
    let angular = 0.5 * f64::from(molecule.multiplicity.saturating_sub(1));
    Ok(RohfResult {
        energy_hartree,
        coefficients,
        mo_energies,
        density_closed: rho_closed,
        density_open: rho_open,
        iterations,
        scf_converged: true,
        n_alpha,
        n_beta,
        electronic_hartree: energy_hartree - e_nuc,
        density_total,
        s_squared: angular * (angular + 1.0),
    })
}
