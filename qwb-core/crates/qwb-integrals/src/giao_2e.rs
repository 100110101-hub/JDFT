//! RS-04 的 GIAO 层：**两电子积分的相位矩** `X_l[μν|λσ] = ∫∫ r₁,ₗ χ_μχ_ν r₁₂⁻¹ χ_λχ_σ`。
//!
//! # 为什么它必须存在
//!
//! London 轨道 `χ_μ^B = exp(−iK_μ·r)χ_μ`、`K_μ = ½B×(R_μ−O)` 让两电子积分也带相位：
//!
//! ```text
//! (μν|λσ)_B = ∫∫ e^{i(K_μ−K_ν)·r₁ + i(K_λ−K_σ)·r₂} χ_μχ_ν r₁₂⁻¹ χ_λχ_σ
//! K_μ − K_ν = ½B × (R_μ − R_ν)          ← 规范原点 O **精确抵消**
//! ```
//!
//! 所以 `B = 0` 处的导数只需要**一个**新积分族：
//!
//! ```text
//! ∂(μν|λσ)/∂B_k = (i/2) Σ_l { [e_k×(R_μ−R_ν)]_l X_l[μν|λσ] + [e_k×(R_λ−R_σ)]_l X_l[λσ|μν] }
//! ```
//!
//! `X_l` 不需要新的原函数：`r_l χ = χ_{μ+e_l} + R_{μ,l} χ_μ`（抬一次幂），于是
//! [`crate::two_electron`] 的同一套 Hermite 机器就够了
//! （[`crate::two_electron::cartesian_block_raised`]）。
//!
//! # 这一项是"装饰"吗？不是（实测）
//!
//! 水/STO-3G、PySCF 的 GIAO 公式（`scripts/pyscf_giao_nmr_reference.py`）：把这一层从
//! Fock 导数里拿掉，屏蔽从 O **365.673** ppm 变成 **2396.038**、H **33.682** 变成 **−2.036**。
//! 它与单电子项同量级 —— **只靠单电子 `∂H/∂B` 与 `∂S/∂B` 拼出来的"GIAO"不是 GIAO**。
//!
//! # 把 `X_l` 收缩成 `∂F₂ₑ/∂B_k`（这一层已与 PySCF 逐元素对上）
//!
//! 见 [`giao_two_electron_field_derivative`] 的文档。一句话：`X_l` 本身**不够** —— 收缩不是一次
//! 密度缩并，而是「Coulomb 的第二个电子对」+「交换项的两个挂法之差 `E_k − E_kᵀ`」两种模式；
//! 反过来，Coulomb 里那个把密度挂在**第一个**电子对上的转置项（`X_l[λσ|μν]`）对**对称**的
//! 密度矩阵**精确为零**（实测 3.8e-17），所以它不需要单独实现。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::two_electron::{ShellBlock, cartesian_block_raised, contract, shell_block, strides};

/// 三个 `X_l[i][j][k][l] = ∫∫ r₁,ₗ χ_iχ_j r₁₂⁻¹ χ_kχ_l`（原子单位，扁平 `((i·n+j)·n+k)·n+l`）。
///
/// **不是置换对称的**：`X_l` 只在 `i↔j` 下不变、并且在把 `r₁` 换成 `r₂` 的同时对
/// `(ij)↔(kl)` 不变；所以这里对每个壳层四元组只写一次，不做 8 重散射。
///
/// # 代价
///
/// 三个 `n⁴` 张量（`3·n⁴` 个 `f64`）—— 与调用方本来就要持有的两电子张量同一量级。
///
/// # Errors
///
/// 基组出现 `l > 2`（球谐变换未实现）→ [`EngineError::Unsupported`]。
pub fn giao_two_electron_moments(
    basis: &InternalBasis,
    molecule: &Molecule,
) -> Result<[Vec<f64>; 3], EngineError> {
    let _ = molecule;
    let shells: Vec<ShellBlock> = basis
        .shells()
        .iter()
        .map(shell_block)
        .collect::<Result<_, _>>()?;
    moment_tensors(basis, &shells)
}

/// 见 [`giao_two_electron_moments`]（壳层预装配的版本）。
fn moment_tensors(
    basis: &InternalBasis,
    shells: &[ShellBlock],
) -> Result<[Vec<f64>; 3], EngineError> {
    let n = basis.n_functions();
    let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n * n * n * n]);
    for a in shells {
        for b in shells {
            for c in shells {
                for d in shells {
                    for (axis, target) in out.iter_mut().enumerate() {
                        let mut data = cartesian_block_raised(a, b, c, d, axis);
                        let mut dims = [
                            a.components.len(),
                            b.components.len(),
                            c.components.len(),
                            d.components.len(),
                        ];
                        for (position, shell) in [a, b, c, d].iter().enumerate() {
                            let (next, next_dims) =
                                contract(&data, dims, position, &shell.transform);
                            data = next;
                            dims = next_dims;
                        }
                        let out_strides = strides(dims);
                        let base = [
                            a.first_function,
                            b.first_function,
                            c.first_function,
                            d.first_function,
                        ];
                        for i in 0..dims[0] {
                            for j in 0..dims[1] {
                                for k in 0..dims[2] {
                                    for l in 0..dims[3] {
                                        let value = data[i * out_strides[0]
                                            + j * out_strides[1]
                                            + k * out_strides[2]
                                            + l * out_strides[3]];
                                        if value == 0.0 {
                                            continue;
                                        }
                                        let (p, q, r, s) =
                                            (base[0] + i, base[1] + j, base[2] + k, base[3] + l);
                                        target[((p * n + q) * n + r) * n + s] += value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `ε_{klm}`（Levi-Civita；与 [`crate::giao`] 的叉积同一个符号）。
fn levi_civita(k: usize, l: usize, m: usize) -> f64 {
    match (k, l, m) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (1, 0, 2) | (2, 1, 0) => -1.0,
        _ => 0.0,
    }
}

/// `w_{k,l}(a,b) = Σ_m ε_{klm} (R_a − R_b)_m`，即 `[e_k × (R_a − R_b)]_l`。
fn pair_weight(k: usize, l: usize, delta: [f64; 3]) -> f64 {
    // ε 只有两个非零的 m，直接枚举比查表清楚。
    (0..3).map(|m| levi_civita(k, l, m) * delta[m]).sum::<f64>()
}

/// GIAO 电子排斥对**磁场**的一阶导数：三个**实**矩阵 `Y_k`，`∂F₂ₑ/∂B_k = i·Y_k`。
///
/// # 为什么它不是"密度 × `X_l`"一次收缩
///
/// 两电子积分的四个基函数**都**带相位，所以
///
/// ```text
/// ∂(μν|λσ)/∂B_k = (i/2) Σ_l { [e_k×(R_μ−R_ν)]_l X_l[μν|λσ] + [e_k×(R_λ−R_σ)]_l X_l[λσ|μν] }
/// ```
///
/// 而 Fock 矩阵 `F_μν = H_μν + Σ_λσ D_λσ[(μν|λσ) − ½(μλ|νσ)]` 求导后出现**四种**不同的
/// 指标挂法（`w_k(a,b) ≡ Σ_m ε_{klm}(R_a−R_b)_m`）：
///
/// ```text
/// J_l[μν]    = Σ_λσ D_λσ X_l[μν|λσ]                     ← Coulomb：密度挂第二个电子对
/// T_l[μλν]   = Σ_σ D_λσ X_l[μλ|σν]                      ← 交换：权重里含一个占据指标
/// Y_k[μν]    = −½ Σ_l w_k(μ,ν)_l J_l[μν] + ¼ (E_k[μν] − E_k[νμ])
/// E_k[μν]    = Σ_l Σ_λ w_k(μ,λ)_l T_l[μλν]
/// ```
///
/// 两点值得写下来（都是**实测**得到的，不是推导出来就完事）：
///
/// 1. **Coulomb 的"另一个电子"那一项恒为零**：`Σ_λσ D_λσ w_k(λ,σ)_l X_l[λσ|μν]` 里
///    `w_k(λ,σ)` 对 `λ↔σ` 反对称、`X_l` 与 `D` 都对称，缩并精确为零（实测 3.8e-17）。
///    所以密度只挂在 `X_l` 的第二个电子对上。
/// 2. **交换项必须反对称化**（`E_k − E_kᵀ`），不是对称组合 —— 这一条被参考值钉死：
///    换成对称组合 `±½(E_k + E_kᵀ)` 时，水/STO-3G 的 O 屏蔽从 365.673 ppm 变成
///    4384.289 ppm（PySCF 侧的实测，等价地本函数的逐元素残留从 7e-8 跳到 O(1)）。
///
/// # 尺子
///
/// `qwb-response/tests/reference/giao_fock_derivative.csv` 的
/// `two_electron_field_derivative` 列（PySCF 的 `int2e_ig1` 组装，441 行）。逐元素残留见
/// `crates/qwb-response/tests/rs04_giao_derivative_2e.rs`：用本仓库自己的 SCF 密度是
/// **7.4e-8**（量级 1.2e0），与 `∂H/∂B` 的 3.9e-8 同一量级，来源是 Å→Bohr 常数差 1.84e-8
/// （见 `qwb-integrals/src/giao.rs` 与 `rs04_giao_derivatives.rs` 的说明）。
///
/// 单位：`density` 与返回值都是原子单位（密度是 AO 基下的 `P_μν`，含自旋求和）。
///
/// # Errors
///
/// 密度维度与基函数数不符 → [`EngineError::Internal`]；`X_l` 算不出来（`l` 过大）→
/// [`EngineError::Unsupported`]。
pub fn giao_two_electron_field_derivative(
    basis: &InternalBasis,
    molecule: &Molecule,
    density: &Matrix,
) -> Result<[Matrix; 3], EngineError> {
    let n = basis.n_functions();
    if density.nrows() != n || density.ncols() != n {
        return Err(EngineError::Internal(format!(
            "giao_two_electron_field_derivative: density is {}x{}, basis has {n} functions",
            density.nrows(),
            density.ncols()
        )));
    }
    let moments = giao_two_electron_moments(basis, molecule)?;
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n);
    for shell in basis.shells() {
        for _ in 0..shell.n_functions() {
            centers.push(shell.center_bohr);
        }
    }
    if centers.len() != n {
        return Err(EngineError::Internal(format!(
            "giao_two_electron_field_derivative: {n} functions but {} centers",
            centers.len()
        )));
    }
    let basis_index = |mu: usize, nu: usize, lambda: usize, sigma: usize| {
        ((mu * n + nu) * n + lambda) * n + sigma
    };

    // ── J_l[μν] = Σ_λσ D_λσ X_l[μν|λσ]（Coulomb）────────────────────
    let mut coulomb: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    // ── T_l[μλν] = Σ_σ D_λσ X_l[μλ|σν]（交换；第二个电子对的顺序按 Fock 的 (μλ|νσ)）──
    let mut exchange = vec![0.0_f64; 3 * n * n * n];
    for (l, moment) in moments.iter().enumerate() {
        for mu in 0..n {
            for lambda in 0..n {
                for nu in 0..n {
                    let mut sum = 0.0;
                    for sigma in 0..n {
                        let d = density[(lambda, sigma)];
                        if d == 0.0 {
                            continue;
                        }
                        sum += d * moment[basis_index(mu, lambda, sigma, nu)];
                    }
                    exchange[(l * n + mu) * n * n + lambda * n + nu] = sum;
                }
            }
            for nu in 0..n {
                let mut sum = 0.0;
                for lambda in 0..n {
                    for sigma in 0..n {
                        let d = density[(lambda, sigma)];
                        if d == 0.0 {
                            continue;
                        }
                        sum += d * moment[basis_index(mu, nu, lambda, sigma)];
                    }
                }
                coulomb[l][(mu, nu)] = sum;
            }
        }
    }

    // ── 组装 Y_k = −½ w∘J + ¼(E_k − E_kᵀ) ──────────────────────────
    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for (k, target) in out.iter_mut().enumerate() {
        let mut coulomb_weighted = vec![0.0_f64; n * n];
        let mut exchange_k = Matrix::zeros(n, n);
        for l in 0..3 {
            for mu in 0..n {
                for nu in 0..n {
                    let delta = [
                        centers[mu][0] - centers[nu][0],
                        centers[mu][1] - centers[nu][1],
                        centers[mu][2] - centers[nu][2],
                    ];
                    let weight = pair_weight(k, l, delta);
                    if weight != 0.0 {
                        coulomb_weighted[mu * n + nu] += weight * coulomb[l][(mu, nu)];
                    }
                }
            }
            for mu in 0..n {
                for lambda in 0..n {
                    let delta = [
                        centers[mu][0] - centers[lambda][0],
                        centers[mu][1] - centers[lambda][1],
                        centers[mu][2] - centers[lambda][2],
                    ];
                    let weight = pair_weight(k, l, delta);
                    if weight == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        let value = exchange[(l * n + mu) * n * n + lambda * n + nu];
                        if value != 0.0 {
                            exchange_k[(mu, nu)] += weight * value;
                        }
                    }
                }
            }
        }
        for mu in 0..n {
            for nu in 0..n {
                target[(mu, nu)] = -0.5 * coulomb_weighted[mu * n + nu]
                    + 0.25 * (exchange_k[(mu, nu)] - exchange_k[(nu, mu)]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::giao_two_electron_moments;
    use crate::two_electron::{cartesian_block_raised, shell_block, two_electron};
    use qwb_engine_api::{Atom, Molecule};
    use qwb_registry::Registry;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    fn water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.117_790]),
                Atom::new("H", [0.0, 0.755_450, -0.471_160]),
                Atom::new("H", [0.0, -0.755_450, -0.471_160]),
            ],
            0,
            1,
        )
    }

    fn water_moments() -> (qwb_basis::InternalBasis, Molecule, [Vec<f64>; 3]) {
        let registry = Registry::open(data_root()).expect("qwb-data");
        let molecule = water();
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &molecule,
        )
        .expect("basis");
        let moments = giao_two_electron_moments(&basis, &molecule).expect("moments");
        (basis, molecule, moments)
    }

    /// **绝对尺子**：四个函数都在同一个中心 `O` 上、且四个函数的 `axis` 幂次之和为**偶**时，
    /// `X_axis[μν|λσ] = O_axis · (μν|λσ)`（`(r−O)_axis` 是奇函数，被偶的被积函数积掉）。
    ///
    /// 水/STO-3G 的氧 `1s`(下标 0) 与 `2s`(下标 1) 都是 s 函数，所以在 `{0,1}⁴` 上恒等式成立。
    #[test]
    fn the_moment_integrals_reduce_to_the_center_times_the_plain_integrals() {
        let (basis, molecule, moments) = water_moments();
        let plain = two_electron(&basis, &molecule).expect("2e");
        let n = basis.n_functions();
        let oxygen = molecule.atoms[0].position_bohr();
        let s_functions = [0usize, 1];
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in s_functions {
            for j in s_functions {
                for k in s_functions {
                    for l in s_functions {
                        for (axis, moment) in moments.iter().enumerate() {
                            let reference = oxygen[axis] * plain.get(i, j, k, l);
                            let value = moment[((i * n + j) * n + k) * n + l];
                            worst = worst.max((value - reference).abs());
                            scale = scale.max(reference.abs());
                        }
                    }
                }
            }
        }
        println!("单中心恒等式：max|X − O·(ij|kl)| = {worst:.3e}（max|ref| = {scale:.3e}）");
        assert!(worst < 1.0e-11, "矩积分不满足单中心恒等式：{worst:.3e}");
    }

    /// `X_l[μν|λσ]` 对 `μ↔ν` 对称（`r_l` 落在哪个函数上不该改变积分）。
    #[test]
    fn the_moment_integrals_are_symmetric_in_the_bra_pair() {
        let (basis, _, moments) = water_moments();
        let n = basis.n_functions();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for moment in &moments {
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            let forward = moment[((i * n + j) * n + k) * n + l];
                            let swapped = moment[((j * n + i) * n + k) * n + l];
                            worst = worst.max((forward - swapped).abs());
                            scale = scale.max(forward.abs());
                        }
                    }
                }
            }
        }
        println!("μ↔ν 对称性：max|Δ| = {worst:.3e}（max|X| = {scale:.3e}）");
        assert!(worst < 1.0e-11, "矩积分不对称：{worst:.3e}");
    }

    /// 抬高 bra 与抬高 ket 给出同一个积分（`r_l χ_a χ_b = (r_lχ_a)χ_b = χ_a(r_lχ_b)`）。
    #[test]
    fn raising_the_bra_and_raising_the_ket_agree() {
        let (basis, _, _) = water_moments();
        let shells: Vec<_> = basis
            .shells()
            .iter()
            .map(shell_block)
            .collect::<Result<Vec<_>, _>>()
            .expect("shells");
        let (c, d) = (&shells[0], &shells[0]);
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for a in &shells {
            for b in &shells {
                for axis in 0..3 {
                    let left = cartesian_block_raised(a, b, c, d, axis);
                    let right = cartesian_block_raised(b, a, c, d, axis);
                    let (da, db) = (a.components.len(), b.components.len());
                    for i in 0..da {
                        for j in 0..db {
                            worst = worst.max((left[i * db + j] - right[j * da + i]).abs());
                            scale = scale.max(left[i * db + j].abs());
                        }
                    }
                }
            }
        }
        println!("抬高 bra vs 抬高 ket：max|Δ| = {worst:.3e}（max|X| = {scale:.3e}）");
        assert!(worst < 1.0e-11, "抬高 bra/ket 不一致：{worst:.3e}");
    }
}
