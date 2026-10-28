//! SV-01 的第二半：**CO/SMO 表面电荷求解**。
//!
//! # 方程
//!
//! 把空腔表面离散成 `M` 个 tessera（[`crate::cavity::Tessera`]），每个碎片上取分片常数
//! 面电荷密度 `σ_K`（e/Bohr²），导体极限下「表面电势为零」给出
//!
//! ```text
//! Σ_{K'} ∫_{cell K'} σ_K' / |r_K − r'| dA' = −V_K ,    K = 1..M
//! ```
//!
//! 非对角碎片用 `∫ ≈ A_K'/d_KK'`；**对角碎片不能这样算**（积分在该碎片上有限但
//! 「中心到自身」发散）。把碎片近似成半径 `a_K = √(A_K/π)` 的圆盘，圆盘上的
//! `∫ dA'/|r_K − r'| = 2π a_K = 2√(π A_K)`。乘上 Klamt–Schüürmann 1993 的经验因子
//! `1.07`（圆盘近似偏小，补 7%）就是对角元：
//!
//! ```text
//! S_KK = 1.07 · 2√(π A_K)
//! ```
//!
//! # 为什么写成「电荷」而不是「密度」
//!
//! 令 `q_K = A_K σ_K`（每个碎片上的电荷），方程变成
//!
//! ```text
//! Σ_{K'} q_K' / d_KK' = −V_K      （K'≠K），  q_K 的对角系数 `S_KK`
//! ```
//!
//! 左端矩阵 `M_KK' = 1/d_KK'` **对称**，可以用对称求解器；若按密度写，
//! `A_K' σ_K'` 的系数矩阵不对称。所以本模块直接解 `q`，`σ = q/A` 只在输出时出现。
//!
//! # 对称性自检（这个错误很容易漏）
//!
//! 一个自洽的离散化必须让「均匀电荷的球壳」的**行和**收敛到 `4πR`：
//! 非对角和 `→ 4πR − 2√(πA)`，对角 `1.07·2√(πA)`，合计 `4πR + O(√A)`。
//! 若把对角写成 `1.07√(4π/A)`（乘以 `A` 之前的密度形式）就会得到 `4πR + O(√M)`
//! 的**发散**行和 —— 实测 `M=2000` 时高 24%。本模块的测试直接钉住这条。
//!
//! # 介电标度与能量
//!
//! C-PCM：`q_K^solvent = f(ε)·q_K^cond`，`f(ε) = (ε−1)/ε`；静电溶剂化能
//! `ΔG = ½ Σ_K V_K q_K^solvent`。
//!
//! # 与解析解的关系（本模块的验收）
//!
//! 中心点电荷 + **球形**空腔有解析解 `ΔG_Born = −½(1 − 1/ε) q²/R`。
//! 测试拿它当外部判据，误差越小、随 `M` 下降即视为实现正确 —— 不依赖任何文献表。

use qwb_engine_api::EngineError;
use qwb_linalg::{Matrix, Vector};

use crate::cavity::Tessera;

/// COSMO 的求解选项。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CosmoOptions {
    /// 线性求解的相对残差阈值。
    pub tolerance: f64,
    /// 对角元的 Klamt 经验因子（1.07）。
    pub self_term_constant: f64,
}

impl Default for CosmoOptions {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-10,
            self_term_constant: 1.07,
        }
    }
}

/// COSMO 的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CosmoResult {
    /// 每个 tessera 上的屏蔽**电荷**（e），`q_K = f(ε)·q_K^cond`。
    pub charges: Vec<f64>,
    /// 导体极限下的表面电荷**密度** `σ_K`（e/Bohr²，未乘 `f(ε)`）。
    pub conductor_charges: Vec<f64>,
    /// 静电溶剂化能（Hartree）。
    pub solvation_energy_hartree: f64,
    /// 相对残差 `‖Mq + V‖ / ‖V‖`。
    pub residual: f64,
    /// 残差是否达标。
    pub converged: bool,
}

/// C-PCM 的介电标度 `f(ε) = (ε−1)/ε`。
///
/// # Errors
///
/// `ε < 1` 或非有限 → [`EngineError::InvalidInput`]。
pub fn dielectric_scale(epsilon: f64) -> Result<f64, EngineError> {
    if !epsilon.is_finite() || epsilon < 1.0 {
        return Err(EngineError::InvalidInput(format!(
            "cosmo: epsilon must be finite and >= 1, got {epsilon}"
        )));
    }
    Ok((epsilon - 1.0) / epsilon)
}

/// 解 COSMO 的表面电荷方程。
///
/// `potential[k]` 是溶质在 `tesserae[k].center_angstrom` 处的静电势（Hartree/e）。
///
/// 复杂度：一次 `M×M` 稠密对称分解，`O(M³)` —— 生产路径的 `M` 由
/// `points_per_atom` 控制（水分子 ~10³ 量级）。
///
/// # Errors
///
/// 维度不匹配、空腔为空、`ε < 1`、矩阵非正定或出现非有限值 →
/// [`EngineError::InvalidInput`] / [`EngineError::Internal`]。
pub fn cosmo_surface_charges(
    tesserae: &[Tessera],
    potential: &[f64],
    epsilon: f64,
    options: CosmoOptions,
) -> Result<CosmoResult, EngineError> {
    let n = tesserae.len();
    if n == 0 {
        return Err(EngineError::InvalidInput(
            "cosmo: the cavity has no tesserae".into(),
        ));
    }
    if potential.len() != n {
        return Err(EngineError::InvalidInput(format!(
            "cosmo: {} potentials for {n} tesserae",
            potential.len()
        )));
    }
    if !options.self_term_constant.is_finite() || options.self_term_constant <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "cosmo: self term constant must be positive, got {}",
            options.self_term_constant
        )));
    }
    let scale = dielectric_scale(epsilon)?;
    let bohr_per_angstrom = qwb_engine_api::molecule::BOHR_PER_ANGSTROM;

    let areas_bohr2: Vec<f64> = tesserae
        .iter()
        .map(|tessera| {
            if !tessera.area_angstrom2.is_finite() || tessera.area_angstrom2 <= 0.0 {
                f64::NAN
            } else {
                tessera.area_angstrom2 * bohr_per_angstrom * bohr_per_angstrom
            }
        })
        .collect();
    if let Some(index) = areas_bohr2.iter().position(|area| !area.is_finite()) {
        return Err(EngineError::InvalidInput(format!(
            "cosmo: tessera {index} has an invalid area"
        )));
    }

    let mut matrix = Matrix::zeros(n, n);
    for i in 0..n {
        // 对角：圆盘上的自积分 2√(πA) 乘 Klamt 的 1.07
        matrix.set(
            i,
            i,
            options.self_term_constant * 2.0 * (std::f64::consts::PI * areas_bohr2[i]).sqrt(),
        );
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = tesserae[i].center_angstrom[0] - tesserae[j].center_angstrom[0];
            let dy = tesserae[i].center_angstrom[1] - tesserae[j].center_angstrom[1];
            let dz = tesserae[i].center_angstrom[2] - tesserae[j].center_angstrom[2];
            let distance_bohr = (dx * dx + dy * dy + dz * dz).sqrt() * bohr_per_angstrom;
            if distance_bohr <= 0.0 {
                return Err(EngineError::InvalidInput(format!(
                    "cosmo: tesserae {i} and {j} coincide"
                )));
            }
            // 对称：M_KK' = 1/d_KK'（电荷形式，不是面积加权的密度形式）
            matrix.set(i, j, 1.0 / distance_bohr);
        }
    }

    let mut rhs = Vector::zeros(n);
    for (i, value) in potential.iter().enumerate() {
        rhs.set(i, -value);
    }
    let conductor = solve_dense(&matrix, &rhs).map_err(|err| {
        EngineError::Internal(format!(
            "cosmo: solving the conductor surface charge system failed: {err}"
        ))
    })?;
    let solved: Vec<f64> = (0..n).map(|i| conductor.get(i)).collect();
    // 电荷 → 面电荷密度 σ_K = q_K / A_K（COSMO 文献的惯用表示）
    let conductor_charges: Vec<f64> = solved
        .iter()
        .zip(&areas_bohr2)
        .map(|(value, area)| value / area)
        .collect();

    // 残差：方程是在**电荷**形式下解的
    let mut residual2 = 0.0;
    let mut rhs2 = 0.0;
    for (i, value) in potential.iter().enumerate() {
        let mut sum = 0.0;
        for (j, charge) in solved.iter().enumerate() {
            sum += matrix.get(i, j) * charge;
        }
        let difference = sum + value;
        residual2 += difference * difference;
        rhs2 += value * value;
    }
    let residual = if rhs2 > 0.0 {
        (residual2 / rhs2).sqrt()
    } else {
        0.0
    };

    let charges: Vec<f64> = solved.iter().map(|value| scale * value).collect();
    let mut energy = 0.0;
    for k in 0..n {
        energy += 0.5 * potential[k] * charges[k];
    }
    if !energy.is_finite() {
        return Err(EngineError::Internal(
            "cosmo: non-finite solvation energy".into(),
        ));
    }
    Ok(CosmoResult {
        charges,
        conductor_charges,
        solvation_energy_hartree: energy,
        residual,
        converged: residual < options.tolerance.max(1.0e-8),
    })
}

/// 稠密线性方程组 `Ax = b`，**带部分选主元的高斯消元**（`O(n³)`）。
///
/// 为什么不用 [`qwb_linalg::solve_symmetric`]：Coulomb 核矩阵 `1/d_ij` 是**不定**的
/// （不是正定），加上对角自作用项后 Cholesky 仍可能失败 —— 实测 `R = 1.5 Å`、
/// `M = 400` 的球就会报「not positive definite」。所以这里用一般消元。
///
/// # Errors
///
/// 维数不匹配、出现非有限值、或主元退化（矩阵奇异）→ 错误字符串。
fn solve_dense(matrix: &Matrix, rhs: &Vector) -> Result<Vector, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n {
        return Err(format!(
            "dimension mismatch: {}x{} matrix, rhs length {}",
            matrix.nrows(),
            matrix.ncols(),
            rhs.len()
        ));
    }
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = matrix.get(i, j);
            if !value.is_finite() {
                return Err(format!("non-finite matrix element ({i}, {j})"));
            }
            a[i * n + j] = value;
        }
    }
    let mut b: Vec<f64> = (0..n).map(|i| rhs.get(i)).collect();
    if b.iter().any(|value| !value.is_finite()) {
        return Err("non-finite right-hand side".into());
    }
    let scale = a.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let threshold = scale * 1.0e-14;

    for column in 0..n {
        // 部分选主元
        let mut pivot_row = column;
        let mut pivot_value = a[column * n + column].abs();
        for row in (column + 1)..n {
            let candidate = a[row * n + column].abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = row;
            }
        }
        if pivot_value <= threshold {
            return Err(format!("singular matrix at column {column}"));
        }
        if pivot_row != column {
            for k in 0..n {
                a.swap(column * n + k, pivot_row * n + k);
            }
            b.swap(column, pivot_row);
        }
        let pivot = a[column * n + column];
        for row in (column + 1)..n {
            let factor = a[row * n + column] / pivot;
            if factor == 0.0 {
                continue;
            }
            a[row * n + column] = 0.0;
            for k in (column + 1)..n {
                a[row * n + k] -= factor * a[column * n + k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err("non-finite solution".into());
    }
    Ok(Vector::new(x))
}

/// 中心点电荷 + 球形空腔的**解析** Born 溶剂化能（Hartree）：`−½(1 − 1/ε) q²/R`。
///
/// `radius_angstrom` 是空腔半径（Å），`charge` 是点电荷（e）。
///
/// # Errors
///
/// 半径非正/非有限、`ε < 1` → [`EngineError::InvalidInput`]。
pub fn born_solvation_energy(
    radius_angstrom: f64,
    charge: f64,
    epsilon: f64,
) -> Result<f64, EngineError> {
    if !radius_angstrom.is_finite() || radius_angstrom <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "cosmo: cavity radius must be positive, got {radius_angstrom}"
        )));
    }
    let scale = dielectric_scale(epsilon)?;
    let radius_bohr = radius_angstrom * qwb_engine_api::molecule::BOHR_PER_ANGSTROM;
    Ok(-0.5 * scale * charge * charge / radius_bohr)
}
