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
//! # 交付边界（如实写在这里）
//!
//! 这里交付的是**积分层** `X_l` 本身（三条结构尺子全绿，见下方测试：单中心恒等式、`μ↔ν`
//! 对称性、抬高 bra 与抬高 ket 一致）。把 `X_l` 收缩成 `∂F₂ₑ/∂B_k` 的那一步**还没有**与
//! PySCF 逐元素对上（残留约 1.8e-1，量级 1.2e0），所以它**没有**进 `qwb-response` 的屏蔽装配
//! —— 宁可少一个功能，也不把没对上的数当结果。参考值已经落在
//! `qwb-response/tests/reference/giao_fock_derivative.csv`（`two_electron_field_derivative`
//! 那一列），谁接这一步都能立刻验。
#![deny(unsafe_code)]

use qwb_basis::InternalBasis;
use qwb_engine_api::{EngineError, Molecule};

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
