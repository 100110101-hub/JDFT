//! RS-04（顺磁项本体）：**顺磁屏蔽** `σ_p`（核中心共原规范，CPHF 耦合形式）。
//!
//! # 公式
//!
//! 核 A 的各向同性顺磁屏蔽：
//!
//! ```text
//! σ_p(A) = (1/(3c²)) Σ_α  L_α(O_A)^ovᵀ  (A+B)⁻¹  V_α(A)^ov          [a.u.]
//! ```
//!
//! * `V_α(A)` 是超精细算符 `L_α(A)/r_A³` 的 AO 矩阵（[`hyperfine_integrals`]，已与
//!   cint 的 `int1e_prinvxp` 逐元素对照过）；
//! * `L_α(O_A)` 是**以核 A 为原点**的角动量算符（[`angular_momentum_integrals`]，`L = −iN`）；
//!   `^ov` 表示取占据-虚（`i`↔`a`）块；
//! * `(A+B)⁻¹ V` 就是线性响应方程的解 —— 顺磁项必须**含轨道弛豫**，把 `⟨a|V|i⟩/(ε_i−ε_a)`
//!   的未耦合求和当结果会差 10% 以上（`docs/choices.md` 七十八的反面）。
//!
//! 系数 `1/(3c²)` 与 `1/3` 的迹平均、以及 `(A+B)` 里闭壳层的自旋因子都沿用与
//! [`crate::polarizability`] 同一套约定（那里 `α = 4 dᵀ (A+B)⁻¹ d` 已被有限场验证到 1e-7）。
//! 交叉二阶导 `E^{(2)}_{λμ} = −2 Wᵀ (A+B)⁻¹ V` 与它同源：取 `W = V = d` 就回到 `−½α`。
//!
//! 于是 `σ_p = −(1/(3c²)) Σ_α E^{(2)}`（其中 `W_α = ½L_α`，与有限场脚本的约定一致）：
//!
//! ```text
//! σ_p(A) = (1/(3c²)) Σ_α L_α^ovᵀ (A+B)⁻¹ V_α^ov × 1e6   [ppm]
//! ```
//!
//! # 必须核对的符号
//!
//! `L = −iN`、且这里用的是**实**矩阵 `N`：两次虚微扰的乘积带来 `i·i = −1`，所以用 `N` 代替
//! `L` 时整体差一个符号。本模块的符号是靠**外部参考**定的（水/STO-3G 的 `σ_p`：O
//! −236.764662 ppm、H −3.891219 ppm，来自 PySCF 有限场），而不是靠"看起来对"。
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::angmom::{angular_momentum_integrals, hyperfine_integrals};
use qwb_linalg::Matrix;

use crate::cphf::{CphfOptions, solve_cphf};
use crate::mo_integrals::{CphfIntegrals, pair_index};

/// 光速（原子单位），与仓库其它地方同一个常数。
const C: f64 = 137.035_999_084;

/// 一个核的顺磁屏蔽（ppm）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ParamagneticShielding {
    /// 核下标。
    pub nucleus_index: usize,
    /// 元素符号。
    pub symbol: String,
    /// 各向同性顺磁屏蔽，ppm。
    pub sigma_p_ppm: f64,
}

/// 把 AO 矩阵变换到 MO 的占据-虚块，并按 `p = i·n_vir + a` 展平。
fn occupied_virtual_block(
    operator: &Matrix,
    coefficients: &Matrix,
    n_occupied: usize,
    n_functions: usize,
) -> Vec<f64> {
    let n_virtual = n_functions - n_occupied;
    // 先做一次完整的 Cᵀ O C 太贵（n³）；这里只需要 ov 块：O_mo[a][i] = Σ_μν C_μa O_μν C_νi。
    let mut block = vec![0.0_f64; n_occupied * n_virtual];
    for i in 0..n_occupied {
        for a in 0..n_virtual {
            let mut sum = 0.0;
            for mu in 0..n_functions {
                let c_mu_a = coefficients[(mu, n_occupied + a)];
                if c_mu_a == 0.0 {
                    continue;
                }
                for nu in 0..n_functions {
                    sum += c_mu_a * operator[(mu, nu)] * coefficients[(nu, i)];
                }
            }
            block[pair_index(n_virtual, i, a)] = sum;
        }
    }
    block
}

/// 计算每个核的各向同性顺磁屏蔽（ppm）。
///
/// 规范：**每个核用自己的位置当原点**（nucleus-centred common origin）。这不是 GIAO ——
/// 结果依赖这个选择，所以 RS-04 的 GIAO 判据仍未满足（见账本条目）。
///
/// # Errors
///
/// 维度不匹配、CPHF 未收敛、角动量积分不支持该角量子数 → [`EngineError`]。
pub fn paramagnetic_shielding_ppm(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
    coefficients: &Matrix,
    orbital_energies: &[f64],
    n_occupied: usize,
    integrals: &CphfIntegrals,
    options: CphfOptions,
) -> Result<Vec<ParamagneticShielding>, EngineError> {
    let n = basis.n_functions();
    if coefficients.nrows() != n || coefficients.ncols() != n {
        return Err(EngineError::Internal(format!(
            "paramagnetic: coefficients are {}x{}, basis has {n} functions",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    let mut out = Vec::with_capacity(molecule.atoms.len());
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let origin = atom.position_bohr();
        let angular = angular_momentum_integrals(basis, molecule, origin)?;
        let hyperfine = hyperfine_integrals(basis, molecule, origin)?;
        let mut isotropic = 0.0;
        for axis in 0..3 {
            let l_block = occupied_virtual_block(&angular[axis], coefficients, n_occupied, n);
            let v_block = occupied_virtual_block(&hyperfine[axis], coefficients, n_occupied, n);
            let rhs = vec![v_block];
            let outcome = solve_cphf(integrals, orbital_energies, &rhs, options)?;
            if !outcome.converged {
                return Err(EngineError::ScfNotConverged {
                    iters: outcome.iterations.iter().copied().max().unwrap_or(0),
                    max: options.max_iter,
                });
            }
            let solution = &outcome.solutions[0];
            let mut term = 0.0;
            for p in 0..l_block.len() {
                term += l_block[p] * solution[p];
            }
            isotropic += term;
        }
        // 三次虚微扰乘积给 (−i)² = −1；实测符号与这个一致（见模块文档与测试）。
        let sigma_p = -(isotropic / (3.0 * C * C)) * 1.0e6;
        out.push(ParamagneticShielding {
            nucleus_index: index,
            symbol: atom.symbol.clone(),
            sigma_p_ppm: sigma_p,
        });
    }
    Ok(out)
}
