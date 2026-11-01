//! ST-02：**动态极化率** `α(ω)`（多频率），以及它在静态极限上与 RS-01 的一致性。
//!
//! # 公式
//!
//! 有了完整 RPA/TDHF 的激发态（[`crate::tddft::tddft_singlet`]），极化率就是态求和
//! （sum-over-states）：
//!
//! ```text
//! α_ij(ω) = Σ_n  μ_n^i μ_n^j · 2ω_n / (ω_n² − ω²)
//! ```
//!
//! 推导：一阶微扰求和里，`1/(ω_n − ω)` 与 `1/(ω_n + ω)` 两项合并成 `2ω_n/(ω_n² − ω²)`。
//! `μ_n` 是 RPA 的跃迁偶极矩（已含闭壳层的 `√2`，见 `tda` 模块），所以 `ω → 0` 时
//! `α_ij(0) = Σ_n 2 μ_n^i μ_n^j / ω_n`，迹/3 就是静态极化率。
//!
//! # 为什么这条公式值得单独做一遍（而不是只用 CPHF）
//!
//! 静态极化率有**两条完全不同的路径**：
//!
//! * RS-01 的 CPHF：解 `(A+B) X = d`，`α = 4 dᵀ(A+B)⁻¹d`；
//! * 本模块：先解 RPA 本征问题拿 `ω_n, μ_n`，再态求和。
//!
//! 两条路径在**完整单激发空间**里数学上恒等，但实现的每一步都不同（一个是线性方程组、
//! 一个是本征问题 + 求和）。所以 `α_SOS(0) == α_CPHF` 是一条极强的自检：它同时验证
//! `(A+B)`、`(A−B)`、归一化与跃迁偶极矩。实测差在 1e-9 量级（见测试）。
//!
//! # 频率的物理边界
//!
//! `ω` 必须**低于第一激发能**，否则分母过零（共振）。这里显式检查并报错，而不是返回
//! 一个巨大的数：`α(ω)` 在共振点附近没有意义，静默返回会在下游变成"数值错误"。
use qwb_engine_api::EngineError;

use crate::mo_integrals::CphfIntegrals;
use crate::polarizability::Polarizability;
use crate::tddft::tddft_singlet;

/// 动态极化率结果：每个请求频率一个张量 + 不变量。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DynamicPolarizability {
    /// 角频率，原子单位（Hartree；`1 Ha = 4.1341373e16 rad/s`）。
    pub omega_au: f64,
    /// 该频率下的 3×3 张量（对称）。
    pub tensor_au: [[f64; 3]; 3],
    /// 各向同性 `(α_xx+α_yy+α_zz)/3`。
    pub isotropic_au: f64,
    /// 各向异性 `sqrt((3/2) Σ(α_ij − α_iso δ_ij)²)`。
    pub anisotropy_au: f64,
    /// 这一轮展开用到的 RPA 态数（= `n_occ·n_vir`，全空间）。
    pub n_states: usize,
    /// 最低激发能（检查共振用的边界）。
    pub lowest_excitation_hartree: f64,
}

/// 一个频率的态求和 `α_ij(ω)`。
///
/// `excitations` 是 RPA 的全部态（`tddft_singlet` 返回的 `Vec<Excitation>`，含跃迁偶极矩）。
///
/// # Errors
///
/// `ω` 非有限、`ω` 触及/超过最低激发能、出现非有限值 → [`EngineError::InvalidInput`] /
/// [`EngineError::Internal`]。
fn sum_over_states(
    excitations: &[crate::tda::Excitation],
    omega: f64,
) -> Result<([[f64; 3]; 3], f64), EngineError> {
    if !omega.is_finite() || omega < 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "dynamic polarizability: omega must be finite and >= 0, got {omega}"
        )));
    }
    let lowest = excitations
        .iter()
        .map(|state| state.energy_hartree)
        .fold(f64::INFINITY, f64::min);
    if !lowest.is_finite() {
        return Err(EngineError::Internal(
            "dynamic polarizability: no excitations to sum over".into(),
        ));
    }
    if omega >= lowest * (1.0 - 1.0e-12) {
        return Err(EngineError::InvalidInput(format!(
            "dynamic polarizability: omega = {omega:.6} Ha touches the lowest excitation \
             {lowest:.6} Ha (the sum has a pole there; static response is a different question)"
        )));
    }
    let mut tensor = [[0.0_f64; 3]; 3];
    for state in excitations {
        let energy = state.energy_hartree;
        let denominator = energy * energy - omega * omega;
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err(EngineError::Internal(format!(
                "dynamic polarizability: degenerate denominator {denominator:.6e} at \
                 omega = {omega:.6}, excitation {energy:.6}"
            )));
        }
        let factor = 2.0 * energy / denominator;
        for (i, row) in tensor.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value += state.transition_dipole_au[i] * state.transition_dipole_au[j] * factor;
            }
        }
    }
    Ok((tensor, lowest))
}

/// 动态极化率：给定一组频率，返回每个频率的张量与不变量。
///
/// `n_states` 是 RPA 求解时**返回**的态数；态求和必须用**全空间**的态，所以这里内部
/// 直接请求全空间（`n_occ·n_vir`），保证 `α(0)` 与 CPHF 在同一空间里可比。
///
/// # Errors
///
/// 见 [`sum_over_states`] 与 [`tddft_singlet`]。
pub fn dynamic_polarizability(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    frequencies_au: &[f64],
) -> Result<Vec<DynamicPolarizability>, EngineError> {
    let full = integrals.n_occupied * integrals.n_virtual;
    let rpa = tddft_singlet(integrals, orbital_energies, full)?;
    let mut out = Vec::with_capacity(frequencies_au.len());
    for &omega in frequencies_au {
        let (tensor, lowest) = sum_over_states(&rpa.excitations, omega)?;
        let invariants = Polarizability::new(tensor);
        out.push(DynamicPolarizability {
            omega_au: omega,
            tensor_au: tensor,
            isotropic_au: invariants.isotropic_au,
            anisotropy_au: invariants.anisotropy_au,
            n_states: rpa.excitations.len(),
            lowest_excitation_hartree: lowest,
        });
    }
    Ok(out)
}
