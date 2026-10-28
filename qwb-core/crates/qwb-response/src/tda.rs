//! RS-02 / RS-03：**TDA（Tamm–Dancoff 近似）**激发能与振子强度。
//!
//! # 方程
//!
//! 闭壳层、单重态、TDA（= RPA 的 A 矩阵本征问题，丢掉 B 块与去激发）：
//!
//! ```text
//! A_{ia,jb} = δ_ij δ_ab (ε_a − ε_i) + 2 (ia|jb) − (ij|ab)
//! A X = ω X
//! ```
//!
//! `2(ia|jb)` 是单重态的库仑项（三重态为 0，本项目先只做单重态），`(ij|ab)` 是交换项。
//! **`HF` 参考下这就是 CIS** —— RS-03 的判据「`hf` 方法时 CIS 与 TDA 差 < 1e-6 a.u.」
//! 因此是同一套方程的两个名字；`qwb-response` 只保留一份实现，`method` 只在报告里区分。
//!
//! # 振子强度
//!
//! ```text
//! μ_n = √2 Σ_{ai} X^n_{ai} ⟨a|r|i⟩,     f_n = (2/3) ω_n |μ_n|²
//! ```
//!
//! `√2` 来自闭壳层双占据（态矢量的归一化：`Σ X² = 1`）。
//!
//! # 自洽检查
//!
//! `Σ_n f_n`（全空间）应当接近电子数 `N` —— TRK 求和规则。CIS 只近似满足它
//! （典型差几个百分点），所以测试把实测值如实打出来，不当成 1e-3 级的判据。
use qwb_engine_api::EngineError;
use qwb_linalg::{Vector, symmetric_eigen};

use crate::mo_integrals::{CphfIntegrals, pair_index};

/// 一个激发态。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Excitation {
    /// 激发能（Hartree）。
    pub energy_hartree: f64,
    /// 跃迁偶极矩（a.u.，`e·bohr`）。
    pub transition_dipole_au: [f64; 3],
    /// 振子强度（无量纲）。
    pub oscillator_strength: f64,
}

/// TDA 的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TdaResult {
    /// 按能量升序的激发态（只含能量为正的）。
    pub excitations: Vec<Excitation>,
    /// 振子强度之和（全空间；应与电子数近似相等）。
    pub oscillator_sum: f64,
    /// 参与展开的 (占据, 虚) 对数（= 本征问题的维数）。
    pub dimension: usize,
}

/// 单重态 TDA（`method: tda` / `cis`）。
///
/// `n_states` 是**返回**的最低激发态个数；本征问题始终在全 `n_occ·n_vir` 空间里解，
/// 这样 `oscillator_sum` 才是全空间的求和。
///
/// # Errors
///
/// 维度不匹配、本征分解失败、出现非有限值 → [`EngineError::Internal`]。
pub fn tda_singlet(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    n_states: usize,
) -> Result<TdaResult, EngineError> {
    let n_occ = integrals.n_occupied;
    let n_vir = integrals.n_virtual;
    let n_pairs = n_occ * n_vir;
    if orbital_energies.len() < integrals.n_functions {
        return Err(EngineError::Internal(format!(
            "TDA: only {} orbital energies for {} functions",
            orbital_energies.len(),
            integrals.n_functions
        )));
    }
    if n_pairs == 0 {
        return Err(EngineError::InvalidInput(
            "TDA: 没有 (占据, 虚) 对可用于展开".into(),
        ));
    }
    // A 矩阵
    let mut a = qwb_linalg::Matrix::zeros(n_pairs, n_pairs);
    for i in 0..n_occ {
        for va in 0..n_vir {
            let p = pair_index(n_vir, i, va);
            for j in 0..n_occ {
                for vb in 0..n_vir {
                    let q = pair_index(n_vir, j, vb);
                    let mut value = 2.0 * integrals.t1(p, q) - integrals.t2(p, q);
                    if p == q {
                        value += orbital_energies[n_occ + va] - orbital_energies[i];
                    }
                    a[(p, q)] = value;
                }
            }
        }
    }
    let (values, vectors) = symmetric_eigen(&a)
        .map_err(|err| EngineError::Internal(format!("TDA: 本征分解失败：{err}")))?;

    let mut excitations = Vec::new();
    let mut oscillator_sum = 0.0;
    for column in 0..n_pairs {
        let energy = values.get(column);
        if !energy.is_finite() {
            return Err(EngineError::Internal("TDA: 本征值出现非有限值".into()));
        }
        let mut dipole = [0.0_f64; 3];
        // x/y/z 三个轴固定，用下标比迭代器清楚（dipole 与 integrals.dipole 都按轴索引）。
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            let mut sum = 0.0;
            for p in 0..n_pairs {
                sum += vectors[(p, column)] * integrals.dipole[axis][p];
            }
            dipole[axis] = std::f64::consts::SQRT_2 * sum;
        }
        let norm2 = dipole.iter().map(|value| value * value).sum::<f64>();
        let oscillator = (2.0 / 3.0) * energy * norm2;
        oscillator_sum += oscillator;
        if energy > 1.0e-10 && excitations.len() < n_states {
            excitations.push(Excitation {
                energy_hartree: energy,
                transition_dipole_au: dipole,
                oscillator_strength: oscillator,
            });
        }
    }
    Ok(TdaResult {
        excitations,
        oscillator_sum,
        dimension: n_pairs,
    })
}

/// 把一个 `Vector` 解包成 `Vec<f64>`（给调用方省一次样板）。
#[must_use]
pub fn eigenvalues_to_vec(values: &Vector) -> Vec<f64> {
    (0..values.len()).map(|i| values.get(i)).collect()
}
