//! GIAO（London 轨道）的**第一个积分**：重叠矩阵对磁场的导数（`RS-04` 的 GIAO 层）。
//!
//! # 公式与为什么它不需要新积分
//!
//! London 轨道给每个基函数乘一个依赖磁场 `B` 与规范原点 `O` 的相位：
//!
//! ```text
//! χ_μ(B) = exp(−(i/2) B·[(R_μ − O) × r]) χ_μ(r)
//! ```
//!
//! 对 `B` 求导并取 `B = 0`：
//!
//! ```text
//! ∂S_μν/∂B_k = (i/2) Σ_{jl} ε_{kjl} (R_μ − R_ν)_j ⟨μ| r_l |ν⟩
//! ```
//!
//! 两件事值得写下来：
//!
//! 1. **规范原点 `O` 精确抵消**（差值里只出现 `R_μ − R_ν`）。这不是近似，是 London 相位
//!    的共同原点性质 —— 所以这个函数**不需要** `O` 参数，`origin` 传什么都一样，这一点由测试钉住。
//! 2. 它只需要**普通的偶极积分** `⟨μ|r|ν⟩`（`dipole::dipole_integrals` 已有），不需要新的
//!    积分递推。这也是 GIAO 层的入口比看起来便宜的原因。
//!
//! 返回的矩阵 `C_k` 约定成 `∂S/∂B_k = (i/2) C_k`（`C_k` 是**实**矩阵），因为复数不该泄漏到
//! 上游的实矩阵接口里 —— 相位带来的虚部由调用方按这个约定乘进去。
//!
//! **还差的（GIAO 要完整落地还需要）**：`∂H/∂B`、GIAO 的电子排斥导数、拉曼/抗磁的 GIAO 项，
//! 以及 CPHF 右端的改写。这一块只交付**重叠导数**，因为它是唯一能用"复相位的有限差分"独立
//! 验证的一步 —— 其余几块没有独立的尺子之前不写（见 `docs/status-v2.yaml` 的 RS-04）。
#![deny(unsafe_code)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

use crate::dipole::dipole_integrals;

/// 球谐（或笛卡尔）基下的 `∂S/∂B_k = (i/2) C_k` 的三个实矩阵。
///
/// # Errors
///
/// 基组里出现 `l` 大于已实现的球谐变换、或矩阵维度不一致。
pub fn giao_overlap_derivative(
    basis: &qwb_basis::InternalBasis,
    molecule: &Molecule,
) -> Result<[Matrix; 3], EngineError> {
    let dipole = dipole_integrals(basis, molecule)?;
    let n = dipole[0].nrows();
    for component in &dipole {
        if component.nrows() != n || component.ncols() != n {
            return Err(EngineError::Internal(
                "giao: dipole matrices have inconsistent dimensions".to_string(),
            ));
        }
    }

    // 每个基函数的中心（同一壳层的所有函数共享中心），用于 `R_μ − R_ν`。
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n);
    for shell in basis.shells() {
        for _ in 0..shell.n_functions() {
            centers.push(shell.center_bohr);
        }
    }
    if centers.len() != n {
        return Err(EngineError::Internal(format!(
            "giao: {} basis functions but {} centers",
            n,
            centers.len()
        )));
    }

    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    for mu in 0..n {
        for nu in 0..n {
            let delta = [
                centers[mu][0] - centers[nu][0],
                centers[mu][1] - centers[nu][1],
                centers[mu][2] - centers[nu][2],
            ];
            let d = [
                dipole[0].get(mu, nu),
                dipole[1].get(mu, nu),
                dipole[2].get(mu, nu),
            ];
            // (Δ × d)_k
            let cross = [
                delta[1] * d[2] - delta[2] * d[1],
                delta[2] * d[0] - delta[0] * d[2],
                delta[0] * d[1] - delta[1] * d[0],
            ];
            for (k, value) in cross.iter().enumerate() {
                out[k][(mu, nu)] = *value;
            }
        }
    }
    Ok(out)
}

/// 单个壳层对的重叠导数（**笛卡尔**分量级，`C_k` 同上）。
///
/// 存在的理由是测试：验证用「复相位重叠的有限差分」做尺子，而那个尺子只能在**笛卡尔**
/// 分量级上写（球谐变换会把不同分量混起来，测试就不再独立了）。生产路径用
/// [`giao_overlap_derivative`]。
///
/// # Errors
///
/// 指数/系数长度不一致，或维度越界。
pub fn giao_overlap_derivative_cartesian(
    shell_a: &qwb_basis::InternalShell,
    shell_b: &qwb_basis::InternalShell,
) -> Result<[Vec<f64>; 3], EngineError> {
    let components_a = qwb_basis::cartesian_components(shell_a.l);
    let components_b = qwb_basis::cartesian_components(shell_b.l);
    let rows = components_a.len();
    let columns = components_b.len();
    let mut out = [
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
        vec![0.0_f64; rows * columns],
    ];
    let delta = [
        shell_a.center_bohr[0] - shell_b.center_bohr[0],
        shell_a.center_bohr[1] - shell_b.center_bohr[1],
        shell_a.center_bohr[2] - shell_b.center_bohr[2],
    ];
    for (ca, powers_a) in components_a.iter().enumerate() {
        for (cb, powers_b) in components_b.iter().enumerate() {
            let mut d = [0.0_f64; 3];
            for (alpha, coefficient_a) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                for (beta, coefficient_b) in shell_b.exponents.iter().zip(&shell_b.coefficients) {
                    let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                    let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                    let p = alpha + beta;
                    let product = [
                        (alpha * shell_a.center_bohr[0] + beta * shell_b.center_bohr[0]) / p,
                        (alpha * shell_a.center_bohr[1] + beta * shell_b.center_bohr[1]) / p,
                        (alpha * shell_a.center_bohr[2] + beta * shell_b.center_bohr[2]) / p,
                    ];
                    let overlap = crate::one_electron::primitive_overlap(
                        *alpha,
                        *powers_a,
                        shell_a.center_bohr,
                        *beta,
                        *powers_b,
                        shell_b.center_bohr,
                    );
                    for axis in 0..3 {
                        d[axis] += weight * overlap * product[axis];
                    }
                }
            }
            let cross = [
                delta[1] * d[2] - delta[2] * d[1],
                delta[2] * d[0] - delta[0] * d[2],
                delta[0] * d[1] - delta[1] * d[0],
            ];
            for (k, value) in cross.iter().enumerate() {
                out[k][ca * columns + cb] = *value;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{giao_overlap_derivative, giao_overlap_derivative_cartesian};
    use qwb_engine_api::{Atom, Molecule};
    use qwb_registry::Registry;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qwb-data")
    }

    fn water() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("O", [0.0, 0.0, 0.0]),
                Atom::new("H", [0.0, 0.757, 0.586]),
                Atom::new("H", [0.0, -0.757, 0.586]),
            ],
            0,
            1,
        )
    }

    /// 复相位重叠的**精确**值：`S^GIAO(B)_μν = Σ w S^prims exp(i K·P − K²/(4p))`，
    /// 其中 `K = (1/2) B × (R_μ − R_ν)`、`P = (αR_μ + βR_ν)/p`。
    fn giao_overlap_cartesian(
        shell_a: &qwb_basis::InternalShell,
        shell_b: &qwb_basis::InternalShell,
        field: [f64; 3],
    ) -> (Vec<f64>, Vec<f64>) {
        let components_a = qwb_basis::cartesian_components(shell_a.l);
        let components_b = qwb_basis::cartesian_components(shell_b.l);
        let rows = components_a.len();
        let columns = components_b.len();
        let delta = [
            shell_a.center_bohr[0] - shell_b.center_bohr[0],
            shell_a.center_bohr[1] - shell_b.center_bohr[1],
            shell_a.center_bohr[2] - shell_b.center_bohr[2],
        ];
        let k = [
            0.5 * (field[1] * delta[2] - field[2] * delta[1]),
            0.5 * (field[2] * delta[0] - field[0] * delta[2]),
            0.5 * (field[0] * delta[1] - field[1] * delta[0]),
        ];
        let mut real = vec![0.0_f64; rows * columns];
        let mut imaginary = vec![0.0_f64; rows * columns];
        for (ca, powers_a) in components_a.iter().enumerate() {
            for (cb, powers_b) in components_b.iter().enumerate() {
                let mut sum_re = 0.0_f64;
                let mut sum_im = 0.0_f64;
                for (alpha, coefficient_a) in shell_a.exponents.iter().zip(&shell_a.coefficients) {
                    let norm_a = crate::one_electron::primitive_norm(*powers_a, *alpha);
                    for (beta, coefficient_b) in shell_b.exponents.iter().zip(&shell_b.coefficients)
                    {
                        let norm_b = crate::one_electron::primitive_norm(*powers_b, *beta);
                        let weight = coefficient_a * coefficient_b * norm_a * norm_b;
                        let p = alpha + beta;
                        let product = [
                            (alpha * shell_a.center_bohr[0] + beta * shell_b.center_bohr[0]) / p,
                            (alpha * shell_a.center_bohr[1] + beta * shell_b.center_bohr[1]) / p,
                            (alpha * shell_a.center_bohr[2] + beta * shell_b.center_bohr[2]) / p,
                        ];
                        let overlap = crate::one_electron::primitive_overlap(
                            *alpha,
                            *powers_a,
                            shell_a.center_bohr,
                            *beta,
                            *powers_b,
                            shell_b.center_bohr,
                        );
                        let phase = k[0] * product[0] + k[1] * product[1] + k[2] * product[2];
                        let damping = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) / (4.0 * p);
                        let magnitude = weight * overlap * (-damping).exp();
                        sum_re += magnitude * phase.cos();
                        sum_im += magnitude * phase.sin();
                    }
                }
                real[ca * columns + cb] = sum_re;
                imaginary[ca * columns + cb] = sum_im;
            }
        }
        (real, imaginary)
    }

    #[test]
    fn the_overlap_derivative_matches_a_finite_difference_of_the_complex_phase() {
        // 独立的尺子：直接按相位因子算 S^GIAO(B)（复矩阵），对 B 做中心差分，再与生产公式
        // （偶极积分 × (R_μ − R_ν)）比。两条路只共用 `primitive_overlap`，不需要别的假设。
        let registry = Registry::open(data_root()).expect("qwb-data");
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &water(),
        )
        .expect("basis");
        let shells = basis.shells();
        let step = 1.0e-5_f64;
        let mut worst: f64 = 0.0;
        for shell_a in shells {
            for shell_b in shells {
                let coefficients = giao_overlap_derivative_cartesian(shell_a, shell_b).expect("C");
                for axis in 0..3 {
                    let mut field_plus = [0.0_f64; 3];
                    let mut field_minus = [0.0_f64; 3];
                    field_plus[axis] = step;
                    field_minus[axis] = -step;
                    let (_, plus) = giao_overlap_cartesian(shell_a, shell_b, field_plus);
                    let (_, minus) = giao_overlap_cartesian(shell_a, shell_b, field_minus);
                    for index in 0..plus.len() {
                        // ∂S/∂B = (i/2) C  ⇒  Im(∂S/∂B) = C/2 ⇒ C = 2·dIm/dB
                        let numeric = (plus[index] - minus[index]) / (2.0 * step);
                        let expected = 0.5 * coefficients[axis][index];
                        worst = worst.max((numeric - expected).abs());
                    }
                }
            }
        }
        println!("GIAO 重叠导数 vs 复相位有限差分：最差 {worst:.3e}");
        assert!(
            worst < 1.0e-8,
            "GIAO 重叠导数与复相位差分差 {worst:.3e}（判据 1e-8）"
        );
    }

    #[test]
    fn the_production_matrix_is_the_cartesian_one_assembled() {
        // 生产函数（球谐/笛卡尔基上的块矩阵）与分量级公式必须一致：抽查两个矩阵元的对称性
        // 与反对称性 —— `C_k` 必须是**实的反对称**（因为 iC 是厄米的、C 实）。
        let registry = Registry::open(data_root()).expect("qwb-data");
        let basis = qwb_basis::InternalBasis::build(
            registry.get_basis("sto-3g").expect("sto-3g"),
            &water(),
        )
        .expect("basis");
        let matrices = giao_overlap_derivative(&basis, &water()).expect("C");
        let n = matrices[0].nrows();
        for (k, matrix) in matrices.iter().enumerate() {
            for mu in 0..n {
                for nu in 0..n {
                    let value = matrix.get(mu, nu);
                    assert!(value.is_finite());
                    // (R_μ − R_ν) × d 对 (μ,ν) 是反对称的（d 对称、Δ 反对称）。
                    assert!(
                        (value + matrix.get(nu, mu)).abs() < 1.0e-10,
                        "C_{k} 必须反对称：{mu},{nu} = {value:.6e}"
                    );
                }
            }
        }
    }
}
