//! 静态极化率（RS-01 的交付物）。
//!
//! `α_ij = 4 d_iᵀ (A+B)⁻¹ d_j`（推导见 [`crate::cphf`]）。三个方向的 RHS 一起解，
//! 然后组装 3×3 张量。
//!
//! 报告里同时给**各向同性** `α_iso = (α_xx+α_yy+α_zz)/3` 与**各向异性**
//! `Δα = sqrt( (3/2) Σ_{ij} (α_ij − α_iso δ_ij)² )`：前者可以直接和有限场/文献比，
//! 后者是转动不变量，可以当"张量没写错"的自检（旋转分子后不变）。

use qwb_engine_api::EngineError;

use crate::cphf::{CphfOptions, CphfResult, solve_cphf};
use crate::mo_integrals::CphfIntegrals;

/// 极化率张量与不变量（原子单位，`bohr³`；1 a.u. = 0.1481847 Å³）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Polarizability {
    /// 3×3 张量（对称）。
    pub tensor_au: [[f64; 3]; 3],
    /// 各向同性 `(α_xx+α_yy+α_zz)/3`。
    pub isotropic_au: f64,
    /// 各向异性 `sqrt((3/2) Σ_{ij}(α_ij − α_iso δ_ij)²)`。
    pub anisotropy_au: f64,
}

impl Polarizability {
    /// 由张量构造（同时算两个不变量）。
    #[must_use]
    pub fn new(tensor_au: [[f64; 3]; 3]) -> Self {
        let isotropic = (tensor_au[0][0] + tensor_au[1][1] + tensor_au[2][2]) / 3.0;
        let mut sum = 0.0;
        for (i, row) in tensor_au.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let deviation = if i == j { value - isotropic } else { *value };
                sum += deviation * deviation;
            }
        }
        Self {
            tensor_au,
            isotropic_au: isotropic,
            anisotropy_au: (1.5 * sum).sqrt(),
        }
    }

    /// 校验：张量对称、各向同性值有限、对角为正（SPD 的必要条件）。
    ///
    /// # Errors
    ///
    /// 张量不对称、出现非有限值、或某条对角非正 → [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        for i in 0..3 {
            for j in 0..3 {
                if !self.tensor_au[i][j].is_finite() {
                    return Err(EngineError::Internal(format!(
                        "polarizability: 张量元素 [{i}][{j}] 非有限"
                    )));
                }
                if (self.tensor_au[i][j] - self.tensor_au[j][i]).abs()
                    > 1.0e-10 * self.tensor_au[i][j].abs().max(1.0)
                {
                    return Err(EngineError::Internal(format!(
                        "polarizability: 张量不对称 ([{i}][{j}] = {}, [{j}][{i}] = {})",
                        self.tensor_au[i][j], self.tensor_au[j][i]
                    )));
                }
            }
            if self.tensor_au[i][i] <= 0.0 {
                return Err(EngineError::Internal(format!(
                    "polarizability: 对角元 [{i}][{i}] = {} 非正；α = 4 dᵀ(A+B)⁻¹d 应当正定",
                    self.tensor_au[i][i]
                )));
            }
        }
        Ok(())
    }
}

/// 静态极化率（闭壳层 RHF 参考）。
///
/// # Errors
///
/// CPHF 未收敛 → [`EngineError::ScfNotConverged`]；其余见
/// [`crate::cphf::solve_cphf`]。
pub fn static_polarizability(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    options: CphfOptions,
) -> Result<(Polarizability, CphfResult), EngineError> {
    // 三个 RHS：d_x, d_y, d_z。解的是 (A+B) X = d（符号见 cphf 的推导：
    // α = 4 dᵀ (A+B)⁻¹ d，所以直接以 d 为右端）。
    let rhs: Vec<Vec<f64>> = (0..3).map(|axis| integrals.dipole[axis].clone()).collect();
    let outcome = solve_cphf(integrals, orbital_energies, &rhs, options)?;
    if !outcome.converged {
        return Err(EngineError::ScfNotConverged {
            iters: outcome.iterations.iter().copied().max().unwrap_or(0),
            max: options.max_iter,
        });
    }
    // 3×3 张量的三个轴固定是 x/y/z，用下标比迭代器清楚（`dipole[i]` 与
    // `solutions[j]` 也是按轴索引的）。
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
    let polarizability = Polarizability::new(tensor);
    polarizability.validate()?;
    Ok((polarizability, outcome))
}
