//! RS-04（GIAO 全量屏蔽）：**London 轨道**下的抗磁 + 顺磁屏蔽与 GIAO-CPHF 响应解。
//!
//! # 与 [`crate::paramagnetic`] 的关系
//!
//! [`crate::paramagnetic`] 用的是**核中心共原（common-origin）**规范：`L` 的原点取该核，
//! 波函数没有 London 相位。它的结果是规范依赖的（水/STO-3G 实测 7.1229 ppm 的规范误差，见
//! `rs04_paramagnetic.rs::the_common_origin_gauge_error_is_measured`），所以 RS-04 的
//! 「GIAO/London 规范不变性 < 1 ppm」判据只能由本模块满足。
//!
//! 本模块**照搬** PySCF `pyscf/prop/nmr/rhf.py`（v1.7.6，2.x 删掉了 `prop` 包）的三段式，
//! 而那个转写的真伪由 PySCF 自己写在文件里的自检值钉住（`scripts/pyscf_giao_nmr_reference.py`
//! 顶部记了 FH/6-31G 的 375.232839 / 483.002139 / 449.032227）。
//!
//! ```text
//! σ_dia[αβ](A) = Σ_μν D_μν [ A11_αβ[μν] − δ_αβ tr_γ A11_γγ[μν] + A01_αβ[μν] ]      (×1e6/c²)
//! H1_k         = ∂H/∂B_k / i = giao_core_field_derivative + giao_two_electron_field_derivative
//! S1_k         = ∂S/∂B_k / i = ½ · giao_overlap_derivative
//! H1_mo[p][i]  = Σ_μν C_μp H1_μν C_νi ,  S1_mo 同理
//! hs           = H1_mo − S1_mo·ε_i                       ← `−s1·ε_i` 耦合
//! X_oo         = −½ S1_oo                                 ← `−½ s1_oo` 密度块
//! (A−B) X_ov   = −( hs_ov + F[X_oo]_ov )
//! σ_para[αβ](A)= 2 Σ_μν (D^oo + D^vo)_μν H01^A_νμ        (×1e6/c²)
//! ```
//!
//! `D^oo`、`D^vo` 分别是 `C_o(2X_oo)C_oᵀ`、`C_v(2X_ov)C_oᵀ`；`H01^A = int1e_prinvxp`（核 A 上的
//! 超精细算符），本仓库的 [`hyperfine_integrals`] 与它**差一个整体符号**（`rs04_hyperfine.rs`
//! 把这条钉住了：`ours = −cint`），所以这里显式取负。
//!
//! # 常数是怎么钉的（三条，都是量出来的，不是选的）
//!
//! 1. **`1e6/c²`**：PySCF 的 `unit = nist.ALPHA**2 * 1e6`，实测
//!    `53.251354478344666`；本仓库的 `1e6/c²`（`c = 137.035999084`，与
//!    [`crate::paramagnetic`] 和 `qwb-props` 的抗磁项同一个常数）是 `53.251354520432891`，
//!    相对差 **7.9e-10**（对应 380 ppm 上 3e-7 ppm），所以两边可以互相当参考。
//!    `1/3` 的迹平均出现在各向同性那一处（`tr/3`），`2` 的因子来自闭壳层密度 `P = 2Σ c cᵀ`
//!    —— 与 [`crate::paramagnetic`] 的 `2/(3c²)` 是同一套约定（那边写成
//!    `(2/(3c²))·L(A−B)⁻¹V`，把 `2` 显式乘出来；这里 `2` 在密度里、`1/3` 在迹里）。
//! 2. **`(A−B)` 的符号**：PySCF 的 `cphf.solve_withs1` 解的是
//!    `mo1[vir] = (hs + fvind(mo1)) /(ε_i − ε_a)`，而 `fvind` 对纯虚（current 型）微扰是
//!    `−½·K(dm1)`（`scf/_response_functions.py` 的 `hermi == 2` 分支，**没有** Coulomb 项）。
//!    把 `dm1 = D − Dᵀ` 展开后，`X_{ai}` 的耦合系数是 `(ab|ij) − (aj|ib)`，即
//!    「带 `ε_i − ε_a` 对角」的那套矩阵；它恰好是 [`crate::paramagnetic`] 里 `(A−B)`
//!    **逐元取负**。所以本模块复用同一个 `(A−B)`（`ε_a − ε_i` 在上、正定，Cholesky 能用），
//!    把右端取负：`(A−B)X = −rhs` ⇔ `−(A−B)X = rhs`。
//! 3. **`−½ S1_oo` 必须进响应解的密度块**：它不是可选项。本模块的 A3 变异实验（把 `x_oo`
//!    置零，即 `mo1[occ] = 0`）实测：水/STO-3G 的 O 从 365.672798 掉到 **230.667249** ppm
//!    （|Δ| = 1.35e2 ppm），A2 的规范 spread 从 8.6e-8 涨到 **6.208** ppm。同一变异在 PySCF 侧
//!    （把 `cphf.solve_withs1` 的 `mo1base[:,occidx] = -s1[:,occidx]*.5` 改成 `0`，其余逐字不动）
//!    给出 **230.667219** ppm —— 两侧一致到 3e-5 ppm，说明这个量级不是本实现的口径问题。
//!    作为对照，把 GIAO 的 `S1` 换成非 GIAO 的 `∂S/∂B = 0` 给出 405.657121 ppm（PySCF 侧
//!    405.657071），spread 7.796 ppm。**两者都不是「≈10 ppm」**：任务书里"365.7 → 375.6 ppm"
//!    那条背景数字用这两种标准变异都复现不出来（两边的 PySCF 原始求解器都试过），见报告。
//!
//! # 绝对值验收
//!
//! `rs04_giao_nmr.rs` 用 `tests/reference/giao_nmr.csv`（PySCF 2.14 + libcint GIAO 积分 +
//! 转写的 `prop/nmr/rhf.py`）逐核验收到 **0.1 ppm**：水/STO-3G O 365.672764、H 33.682096；
//! 水/6-31G O 326.753598、H 32.725715；甲烷/STO-3G C 234.929788、H 32.663107。
//! 分解出来的抗磁张量另有中间量对照（O 359.0486/393.5013/380.5008、
//! H 23.2536/38.2721/30.5263、H 的 yz/zy 7.5044/8.4207）。
//!
//! # 规范不变性（GIAO 的**全部意义**）
//!
//! GIAO 的 dia/para **分解**依赖坐标绝对原点（`giao_a01_integrals` 的 `r` 是绝对电子坐标，
//! 见 `qwb-integrals/src/giao_dia.rs`），只有**两者之和**规范不变。所以本模块的规范检验
//! 不是"换一个 `L` 的原点"（GIAO 里那个原点精确抵消、根本没有这个自由度），而是
//! **把整个分子平移到另一个坐标原点**再算一遍：分解会动、总和不动。判据 < 1 ppm，
//! 实测在 1e-9 ppm 量级（`rs04_giao_nmr.rs`）。

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_integrals::angmom::hyperfine_integrals;
use qwb_integrals::two_electron::TwoElectronIntegrals;
use qwb_linalg::{Matrix, Vector};

use crate::mo_integrals::{CphfIntegrals, pair_index};

/// 原子单位下的光速（CODATA 2018）：与 [`crate::paramagnetic`] 和 `qwb-props` 的抗磁项同一个常数。
const SPEED_OF_LIGHT: f64 = 137.035_999_084;

/// 原子单位（`c⁻²` 量纲）→ ppm：`1e6/c²`。
///
/// 等价于 PySCF 的 `nist.ALPHA**2 * 1e6`（相对差 7.9e-10，见模块文档）。
const PPM_PER_AU: f64 = 1.0e6 / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);

/// 一个核的 GIAO 屏蔽张量（ppm），dia / para / 总和都留下。
///
/// 三个张量都**不是**对称的（`α01` 与响应解都不对称），所以用 `[[f64; 3]; 3]` 而不是
/// 6 个独立分量 —— 转置元素 `yz`/`zy` 在参考里就不等（水/STO-3G 的 H 是 −7.5044 / −8.4207）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GiaoShielding {
    /// 核下标。
    pub nucleus_index: usize,
    /// 元素符号。
    pub symbol: String,
    /// 抗磁张量 `σ_dia[α][β]`（ppm）。
    pub dia_ppm: [[f64; 3]; 3],
    /// 顺磁张量 `σ_para[α][β]`（ppm）。
    pub para_ppm: [[f64; 3]; 3],
    /// 总屏蔽张量 `σ_dia + σ_para`（ppm）。
    pub total_ppm: [[f64; 3]; 3],
    /// 各向同性屏蔽 `⅓ tr(σ_total)`（ppm）。
    ///
    /// **这一条与规范原点无关**，是与实验/NMR 化学位移对接的那个量。
    pub sigma_iso_ppm: f64,
}

/// `P_μν = 2 Σ_{i<n_occ} C_μi C_νi`（AO 密度，含自旋求和 —— 与 PySCF 的 `make_rdm1()` 同一定义）。
fn ao_density(coefficients: &Matrix, n_occupied: usize, n_functions: usize) -> Matrix {
    let mut density = Matrix::zeros(n_functions, n_functions);
    for mu in 0..n_functions {
        for nu in 0..n_functions {
            let mut sum = 0.0;
            for i in 0..n_occupied {
                sum += 2.0 * coefficients[(mu, i)] * coefficients[(nu, i)];
            }
            density[(mu, nu)] = sum;
        }
    }
    density
}

/// `Σ_μν C_μp O_μν C_νi` 的**全部** `nmo × n_occ` 块，按 `p * n_occ + i` 展平。
///
/// 与 [`crate::paramagnetic`] 的 `occupied_virtual_block` 只差"第一下标跑全部 MO 而不只是虚
/// 轨道"：`hs` 与 `−½ S1_oo` 都要占据行，响应方程只要虚行。
fn mo_columns(
    operator: &Matrix,
    coefficients: &Matrix,
    n_occupied: usize,
    n_functions: usize,
) -> Vec<f64> {
    let mut block = vec![0.0_f64; n_functions * n_occupied];
    for p in 0..n_functions {
        for i in 0..n_occupied {
            let mut sum = 0.0;
            for mu in 0..n_functions {
                let c_mu_p = coefficients[(mu, p)];
                if c_mu_p == 0.0 {
                    continue;
                }
                for nu in 0..n_functions {
                    sum += c_mu_p * operator[(mu, nu)] * coefficients[(nu, i)];
                }
            }
            block[p * n_occupied + i] = sum;
        }
    }
    block
}

/// `C_o (2x) C_oᵀ`（`x` 是 `n_occ × n_occ`，按 `l * n_occ + k` 展平）。
fn occupied_density(
    coefficients: &Matrix,
    x: &[f64],
    n_occupied: usize,
    n_functions: usize,
) -> Matrix {
    let mut density = Matrix::zeros(n_functions, n_functions);
    for mu in 0..n_functions {
        for nu in 0..n_functions {
            let mut sum = 0.0;
            for l in 0..n_occupied {
                let c_mu_l = coefficients[(mu, l)];
                if c_mu_l == 0.0 {
                    continue;
                }
                for k in 0..n_occupied {
                    sum += c_mu_l * 2.0 * x[l * n_occupied + k] * coefficients[(nu, k)];
                }
            }
            density[(mu, nu)] = sum;
        }
    }
    density
}

/// `C_v (2x) C_oᵀ`（`x` 是 `n_vir × n_occ`，按 `a * n_occ + i` 展平）。
fn virtual_occupied_density(
    coefficients: &Matrix,
    x: &[f64],
    n_occupied: usize,
    n_virtual: usize,
    n_functions: usize,
) -> Matrix {
    let mut density = Matrix::zeros(n_functions, n_functions);
    for mu in 0..n_functions {
        for nu in 0..n_functions {
            let mut sum = 0.0;
            for a in 0..n_virtual {
                let c_mu_a = coefficients[(mu, n_occupied + a)];
                if c_mu_a == 0.0 {
                    continue;
                }
                for i in 0..n_occupied {
                    sum += c_mu_a * 2.0 * x[a * n_occupied + i] * coefficients[(nu, i)];
                }
            }
            density[(mu, nu)] = sum;
        }
    }
    density
}

/// `−½ K(dm)`，其中 `K(dm)_μν = Σ_λσ (μλ|νσ) dm_λσ`。
///
/// PySCF 的 `gen_response(..., hermi=2)` 对纯虚（current 型）微扰只留交换项、系数 `−½`
/// （`scf/_response_functions.py` 的 `else: v1 = -.5 * mf.get_k(...)`）：反对称密度的 Coulomb
/// 项恒为零，所以这里**不**做 `J`。
fn exchange_response(density: &Matrix, tensor: &TwoElectronIntegrals) -> Matrix {
    let n = tensor.n_functions;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for lambda in 0..n {
                for sigma in 0..n {
                    let value = density[(lambda, sigma)];
                    if value == 0.0 {
                        continue;
                    }
                    sum += tensor.get(mu, lambda, nu, sigma) * value;
                }
            }
            out[(mu, nu)] = -0.5 * sum;
        }
    }
    out
}

/// 计算每个核的 **GIAO 屏蔽张量**（ppm），抗磁/顺磁/总和都返回。
///
/// 输入里的 `coefficients` / `orbital_energies` / `n_occupied` 是收敛的闭壳层 RHF 态；
/// `density`（`2Σ_{i occ} c_i c_iᵀ`）在内部由系数构造，所以调用方不需要再传一份
/// —— 这与 PySCF 的 `mf.make_rdm1()` 精确同一定义（`qwb-scf` 的密度就是
/// `Σ_orbital f_i c_i c_iᵀ`、整数占据时 `f_i = 2`）。
///
/// # 参数
///
/// * `integrals`：CPHF 的 `(A−B)` 两块（[`CphfIntegrals`]，与 [`crate::paramagnetic`] 共用）；
/// * `tensor`：AO 两电子张量。`−½ S1_oo` 那块密度要的是 `(虚 占 | 占 占)` 块，
///   [`CphfIntegrals`] 按设计只存 `(虚 占 | 虚 占)` 与 `(虚 虚 | 占 占)`，所以这里直接从
///   AO 张量缩并一次（`O(n⁴)`，**一次**，不是每次迭代一次）。
///
/// # Errors
///
/// 维度不匹配、占据/虚轨道分界非法、CPHF 矩阵非正定（Cholesky 失败）→ [`EngineError`]。
pub fn giao_shielding_ppm(
    basis: &InternalBasis,
    molecule: &Molecule,
    coefficients: &Matrix,
    orbital_energies: &[f64],
    n_occupied: usize,
    integrals: &CphfIntegrals,
    tensor: &TwoElectronIntegrals,
) -> Result<Vec<GiaoShielding>, EngineError> {
    let n = basis.n_functions();
    if coefficients.nrows() != n || coefficients.ncols() != n {
        return Err(EngineError::Internal(format!(
            "giao: coefficients are {}x{}, basis has {n} functions",
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if orbital_energies.len() < n {
        return Err(EngineError::Internal(format!(
            "giao: only {} orbital energies for {n} functions",
            orbital_energies.len()
        )));
    }
    if n_occupied == 0 || n_occupied >= n {
        return Err(EngineError::InvalidInput(format!(
            "giao: needs at least one occupied and one virtual orbital (n_occ = {n_occupied}, n = {n})"
        )));
    }
    let n_virtual = n - n_occupied;
    let n_pairs = n_occupied * n_virtual;
    if integrals.n_occupied != n_occupied
        || integrals.n_virtual != n_virtual
        || integrals.n_functions != n
    {
        return Err(EngineError::Internal(format!(
            "giao: CPHF integrals are ({} occ, {} vir, {} functions), expected ({n_occupied}, {n_virtual}, {n})",
            integrals.n_occupied, integrals.n_virtual, integrals.n_functions
        )));
    }
    if tensor.n_functions != n {
        return Err(EngineError::Internal(format!(
            "giao: two-electron tensor has {} functions, basis has {n}",
            tensor.n_functions
        )));
    }

    let density = ao_density(coefficients, n_occupied, n);

    // ── 一阶导数（AO）：H1_k = ∂H/∂B_k / i、S1_k = ∂S/∂B_k / i ──────────────
    let core = qwb_integrals::giao::giao_core_field_derivative(basis, molecule)?;
    let two_electron =
        qwb_integrals::giao_2e::giao_two_electron_field_derivative(basis, molecule, &density)?;
    let overlap = qwb_integrals::giao::giao_overlap_derivative(basis, molecule)?;
    let mut h1: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    let mut s1: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for k in 0..3 {
        for mu in 0..n {
            for nu in 0..n {
                h1[k][(mu, nu)] = core[k][(mu, nu)] + two_electron[k][(mu, nu)];
                // PySCF 的 `s1 = -int1e_igovlp`；本仓库的 `giao_overlap_derivative` 返回 `C_k`
                // 且 `∂S/∂B = (i/2)C_k`，所以 `S1 = ½C`（rs04_giao_derivatives.rs 钉的这一条）。
                s1[k][(mu, nu)] = 0.5 * overlap[k][(mu, nu)];
            }
        }
    }

    // ── MO 块与 `hs = H1_mo − S1_mo·ε_i` ────────────────────────────────────
    let mut hs: Vec<Vec<f64>> = (0..3)
        .map(|k| mo_columns(&h1[k], coefficients, n_occupied, n))
        .collect();
    let s1_mo: Vec<Vec<f64>> = (0..3)
        .map(|k| mo_columns(&s1[k], coefficients, n_occupied, n))
        .collect();
    for k in 0..3 {
        for p in 0..n {
            for i in 0..n_occupied {
                hs[k][p * n_occupied + i] -= s1_mo[k][p * n_occupied + i] * orbital_energies[i];
            }
        }
    }

    // ── (A−B)：与 crate::paramagnetic 逐元相同（那边 `(aj|ib) − (ab|ij) + (ε_a−ε_i)`）──
    let mut a_minus_b = Matrix::zeros(n_pairs, n_pairs);
    for i in 0..n_occupied {
        for va in 0..n_virtual {
            let p = pair_index(n_virtual, i, va);
            for j in 0..n_occupied {
                for vb in 0..n_virtual {
                    let q = pair_index(n_virtual, j, vb);
                    let exchanged =
                        integrals.t1(pair_index(n_virtual, j, va), pair_index(n_virtual, i, vb));
                    let mut value = exchanged - integrals.t2(p, q);
                    if p == q {
                        value += orbital_energies[n_occupied + va] - orbital_energies[i];
                    }
                    a_minus_b[(p, q)] = value;
                }
            }
        }
    }

    // ── 逐分量解 GIAO-CPHF，并累积顺磁张量 ─────────────────────────────────
    // `H01 = int1e_prinvxp`（核中心）与分量无关，先一次算好；本仓库的
    // [`hyperfine_integrals`] 与它差一个整体符号（`rs04_hyperfine.rs` 钉的）。
    let mut hyperfines = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        hyperfines.push(hyperfine_integrals(basis, molecule, atom.position_bohr())?);
    }
    let mut para = vec![[[0.0_f64; 3]; 3]; molecule.n_atoms()];
    for k in 0..3 {
        // X_oo = −½ S1_oo（`cphf.solve_withs1` 的 `mo1base[:,occidx] = -s1[:,occidx]*.5`）。
        let mut x_oo = vec![0.0_f64; n_occupied * n_occupied];
        for l in 0..n_occupied {
            for m in 0..n_occupied {
                x_oo[l * n_occupied + m] = -0.5 * s1_mo[k][l * n_occupied + m];
            }
        }
        // 这块密度对 Fock 响应的贡献：`dm1 = D − Dᵀ`（`vind` 里就是这么反对称化的），
        // 再取 `−½K`、投到虚-占块上。
        let d_oo = occupied_density(coefficients, &x_oo, n_occupied, n);
        let mut dm1_oo = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                dm1_oo[(mu, nu)] = d_oo[(mu, nu)] - d_oo[(nu, mu)];
            }
        }
        let response = exchange_response(&dm1_oo, tensor);
        let mut from_oo = vec![0.0_f64; n_virtual * n_occupied];
        for a in 0..n_virtual {
            for i in 0..n_occupied {
                let mut sum = 0.0;
                for mu in 0..n {
                    let c_mu_a = coefficients[(mu, n_occupied + a)];
                    if c_mu_a == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        sum += c_mu_a * response[(mu, nu)] * coefficients[(nu, i)];
                    }
                }
                from_oo[a * n_occupied + i] = sum;
            }
        }

        // `(A−B) X = −(hs_ov + F[X_oo])`：耦合算子在 GIAO 这一侧是 paramagnetic 那套的**负**，
        // 见模块文档第 2 条。
        let mut rhs = Vector::zeros(n_pairs);
        for i in 0..n_occupied {
            for a in 0..n_virtual {
                let p = pair_index(n_virtual, i, a);
                rhs.set(
                    p,
                    -(hs[k][(n_occupied + a) * n_occupied + i] + from_oo[a * n_occupied + i]),
                );
            }
        }
        let solution = qwb_linalg::solve_symmetric(&a_minus_b, &rhs).map_err(|err| {
            EngineError::Internal(format!("giao: (A−B) solve failed at component {k}: {err}"))
        })?;
        let mut x_ov = vec![0.0_f64; n_virtual * n_occupied];
        for i in 0..n_occupied {
            for a in 0..n_virtual {
                x_ov[a * n_occupied + i] = solution.get(pair_index(n_virtual, i, a));
            }
        }

        // 两个一阶密度（**不**反对称化：`giao_para` 里就是 `C(2x)Cᵀ`）。
        let d_ov = virtual_occupied_density(coefficients, &x_ov, n_occupied, n_virtual, n);
        let mut total_dm = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                total_dm[(mu, nu)] = d_oo[(mu, nu)] + d_ov[(mu, nu)];
            }
        }
        for index in 0..molecule.n_atoms() {
            let hyperfine = &hyperfines[index];
            for beta in 0..3 {
                let mut sum = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        sum += total_dm[(mu, nu)] * -hyperfine[beta][(nu, mu)];
                    }
                }
                para[index][k][beta] = 2.0 * sum * PPM_PER_AU;
            }
        }
    }

    // ── 抗磁项 + 组装 ──────────────────────────────────────────────────────
    let mut out = Vec::with_capacity(molecule.n_atoms());
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let nucleus = atom.position_bohr();
        let a11 = qwb_integrals::giao_dia::giao_a11_integrals(basis, molecule, nucleus)?;
        let a01 = qwb_integrals::giao_dia::giao_a01_integrals(basis, molecule, nucleus)?;
        let mut dia = [[0.0_f64; 3]; 3];
        for alpha in 0..3 {
            for beta in 0..3 {
                let mut sum = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        sum += a11[alpha][beta][(mu, nu)] * density[(mu, nu)];
                    }
                }
                dia[alpha][beta] = sum;
            }
        }
        // 迹只在 a11 那一段上取（PySCF：`e11 -= eye(3)*e11.trace()` 发生在加 a01gp **之前**）。
        let trace = dia[0][0] + dia[1][1] + dia[2][2];
        for alpha in 0..3 {
            for beta in 0..3 {
                let mut sum = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        sum += a01[alpha][beta][(mu, nu)] * density[(mu, nu)];
                    }
                }
                dia[alpha][beta] += sum;
                if alpha == beta {
                    dia[alpha][beta] -= trace;
                }
            }
        }
        let mut dia_ppm = [[0.0_f64; 3]; 3];
        let mut total_ppm = [[0.0_f64; 3]; 3];
        for alpha in 0..3 {
            for beta in 0..3 {
                dia_ppm[alpha][beta] = dia[alpha][beta] * PPM_PER_AU;
                total_ppm[alpha][beta] = dia_ppm[alpha][beta] + para[index][alpha][beta];
            }
        }
        let iso = (total_ppm[0][0] + total_ppm[1][1] + total_ppm[2][2]) / 3.0;
        out.push(GiaoShielding {
            nucleus_index: index,
            symbol: atom.symbol.clone(),
            dia_ppm,
            para_ppm: para[index],
            total_ppm,
            sigma_iso_ppm: iso,
        });
    }
    Ok(out)
}
