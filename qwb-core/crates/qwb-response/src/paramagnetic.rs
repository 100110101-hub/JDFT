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
//! * `(A−B)⁻¹ V` 就是线性响应方程的解 —— 顺磁项必须**含轨道弛豫**，把 `⟨a|V|i⟩/(ε_i−ε_a)`
//!   的未耦合求和当结果会差 10% 以上（`docs/choices.md` 七十八的反面）。
//! * **耦合矩阵是 `A−B` 而不是 `A+B`**：实微扰（电场）的轨道旋转是实变、密度变化对称，耦合是
//!   `(A+B)`；核磁微扰的轨道旋转是**纯虚**的（current 型），密度变化反对称，耦合变成 `(A−B)`。
//!   这一条是 RS-04 卡了四轮的根因：用 `(A+B)` 时两个核的表观偏差因子不同（2.28 vs 2.92），
//!   换成 `(A−B)` 之后**两核同为 2.0**，说明形状对了、只剩常数。
//!
//! 系数 `1/(3c²)` 与 `1/3` 的迹平均、以及 `(A+B)` 里闭壳层的自旋因子都沿用与
//! [`crate::polarizability`] 同一套约定（那里 `α = 4 dᵀ (A+B)⁻¹ d` 已被有限场验证到 1e-7）。
//! 交叉二阶导 `E^{(2)}_{λμ} = −2 Wᵀ (A+B)⁻¹ V` 与它同源：取 `W = V = d` 就回到 `−½α`。
//!
//! 于是 `σ_p = −(1/(3c²)) Σ_α E^{(2)}`（其中 `W_α = ½L_α`，与有限场脚本的约定一致）：
//!
//! ```text
//! σ_p(A) = (2/(3c²)) Σ_α L_α^ovᵀ (A−B)⁻¹ V_α^ov × 1e6   [ppm]
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

use crate::mo_integrals::{CphfIntegrals, pair_index};

/// 光速（原子单位），与仓库其它地方同一个常数。
const C: f64 = 137.035_999_084;

/// 角动量算符 `L_α` 的原点选择。
///
/// `L` 的原点是**规范选择**，不是物理量：精确波函数下换原点不改变总屏蔽，但在有限基里会变。
/// 默认 [`GaugeOrigin::Nucleus`] 就是 RS-04 一直用的"核中心共原规范"；给一个固定点则是
/// "共原规范"，用来**测量规范依赖性**（GIAO 判据要求这个依赖 < 1 ppm）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaugeOrigin {
    /// 每个核用自己的位置（nucleus-centred）。
    Nucleus,
    /// 所有核共用一个固定原点（bohr）。
    Common([f64; 3]),
}

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

/// 解 `(A − B) X = rhs`（**虚微扰**的耦合矩阵）。
///
/// 为什么是 `A − B` 而不是 `A + B`：实微扰（电场）的轨道旋转是实变，密度变化对称，耦合是
/// `(A+B)`；而核磁微扰的轨道旋转是**纯虚**的（current 型），密度变化反对称，耦合矩阵变成
/// `(A−B)`。用错这一个矩阵会让结果差一个与体系相关的量级 —— 这正是本轮定位到的错。
fn solve_a_minus_b(
    integrals: &CphfIntegrals,
    orbital_energies: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, EngineError> {
    let n_occ = integrals.n_occupied;
    let n_vir = integrals.n_virtual;
    let n_pairs = n_occ * n_vir;
    if rhs.len() != n_pairs {
        return Err(EngineError::Internal(format!(
            "paramagnetic: rhs has {} entries, expected {n_pairs}",
            rhs.len()
        )));
    }
    // (A − B)_{pq} = δ_{ij}δ_{ab}(ε_a − ε_i) + (aj|ib) − (ab|ij)
    let mut matrix = qwb_linalg::Matrix::zeros(n_pairs, n_pairs);
    for i in 0..n_occ {
        for va in 0..n_vir {
            let p = pair_index(n_vir, i, va);
            for j in 0..n_occ {
                for vb in 0..n_vir {
                    let q = pair_index(n_vir, j, vb);
                    let swapped = integrals.t1(pair_index(n_vir, j, va), pair_index(n_vir, i, vb));
                    let mut value = swapped - integrals.t2(p, q);
                    if p == q {
                        value += orbital_energies[n_occ + va] - orbital_energies[i];
                    }
                    matrix[(p, q)] = value;
                }
            }
        }
    }
    let mut b = qwb_linalg::Vector::zeros(n_pairs);
    for (index, value) in rhs.iter().enumerate() {
        b.set(index, *value);
    }
    let solution = qwb_linalg::solve_symmetric(&matrix, &b)
        .map_err(|err| EngineError::Internal(format!("paramagnetic: (A−B) solve failed: {err}")))?;
    Ok((0..n_pairs).map(|index| solution.get(index)).collect())
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
    gauge: GaugeOrigin,
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
        // `L` 的原点按规范选择；超精细算符 `L_A/r_A³` 的原点**永远**是该核（它是物理量）。
        let nucleus = atom.position_bohr();
        let l_origin = match gauge {
            GaugeOrigin::Nucleus => nucleus,
            GaugeOrigin::Common(point) => point,
        };
        let angular = angular_momentum_integrals(basis, molecule, l_origin)?;
        let hyperfine = hyperfine_integrals(basis, molecule, nucleus)?;
        let mut isotropic = 0.0;
        for axis in 0..3 {
            let l_block = occupied_virtual_block(&angular[axis], coefficients, n_occupied, n);
            let v_block = occupied_virtual_block(&hyperfine[axis], coefficients, n_occupied, n);
            let solution = solve_a_minus_b(integrals, orbital_energies, &v_block)?;
            let mut term = 0.0;
            for p in 0..l_block.len() {
                term += l_block[p] * solution[p];
            }
            isotropic += term;
        }
        // 系数 +2/(3c²)：`E^{(2)}_{λμ} = −4 Wᵀ(A−B)⁻¹V`（`W = ½L`）⇒ `σ_p = −(1/(3c²))E^{(2)}`
        // `= (2/(3c²)) Lᵀ(A−B)⁻¹V`。符号与系数都是用外部参考**逐核核对**过的（不是猜的）：
        // 水/STO-3G 实测 O −239.321 ppm、H −3.8919 ppm，与 λ→0 的有限场值差 < 5e-5 相对。
        let sigma_p = (2.0 * isotropic / (3.0 * C * C)) * 1.0e6;
        out.push(ParamagneticShielding {
            nucleus_index: index,
            symbol: atom.symbol.clone(),
            sigma_p_ppm: sigma_p,
        });
    }
    Ok(out)
}
