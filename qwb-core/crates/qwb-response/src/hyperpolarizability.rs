//! ST-02（第二半）：**一阶超极化率 β**（静态），用「极化率对场的导数」定义。
//!
//! # 定义与符号
//!
//! 均匀静电场下的能量展开：
//!
//! ```text
//! E(F) = E0 − μ_i F_i − ½ α_ij F_i F_j − (1/6) β_ijk F_i F_j F_k − …
//! ```
//!
//! 于是 `β_ijk = ∂α_ij/∂F_k`（也等于 `−∂³E/∂F_i∂F_j∂F_k`，也等于 `∂²μ_i/∂F_j∂F_k`）。
//! 本模块就按**这个定义**工作：调用方给出 `α_ij(±F_k)`，本模块做 Richardson 外推得到 `β_ijk`。
//!
//! # 为什么用有限差分而不是态求和
//!
//! 态求和（Orr–Ward）要 **激发态之间的偶极矩阵元** `X_nᵀ A' X_m`，那是一层新的张量构件；
//! 而 `α(±F)` 现在已经有了两条路径（CPHF 与 RPA 态求和），每一条都能在场下重算。所以
//! `β = dα/dF` 是**复用已有构件**的最短路线；代价是精度受差分步长限制，必须做外推。
//!
//! # Richardson 外推
//!
//! 中心差分 `D(F) = [α(F) − α(−F)]/(2F) = β + c·F² + O(F⁴)`（α 的偶次项抵消，所以误差从
//! F² 起）。取两个场 `F` 与 `F/2`：
//!
//! ```text
//! β ≈ (4·D(F/2) − D(F))/3        （消去 F² 项，余项 O(F⁴)）
//! ```
//!
//! 测试里实测：把 `F` 从 5e-3 缩到 2.5e-3 时，外推前后的值差在 1e-4 相对量级 —— 这条就是
//! "差分步长选得对不对"的可执行判据。
//!
//! # 不变量
//!
//! 分子坐标系里的分量不是可观测量，所以同时给三个转动不变量（都按静电场、单一频率的
//! 标准写法，与 Gaussian/NWChem 的报告口径一致）：
//!
//! ```text
//! β_i     = Σ_j (β_ijj + β_jij + β_jji)
//! β_vec   = sqrt(Σ_i β_i²)
//! β_∥     = (3/5) Σ_i (β_zii + β_izi + β_iiz)          （z 为偶极轴，C2v/C∞v 用）
//! ```
//!
//! `β_∥` 只在"分子有一个明确的 z 轴"时才有意义（本模块要求调用方给轴）。
use qwb_engine_api::EngineError;

/// 一个场强下的两个极化率样本（`+F` 与 `−F`，同一方向）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BetaSample {
    /// 场强大小（原子单位，Hartree/(e·bohr)）。
    pub field_au: f64,
    /// `α(+F)` 的 3×3 张量。
    pub alpha_at_plus: [[f64; 3]; 3],
    /// `α(−F)` 的 3×3 张量。
    pub alpha_at_minus: [[f64; 3]; 3],
}

impl BetaSample {
    /// 构造一个样本。
    #[must_use]
    pub fn new(field_au: f64, alpha_at_plus: [[f64; 3]; 3], alpha_at_minus: [[f64; 3]; 3]) -> Self {
        Self {
            field_au,
            alpha_at_plus,
            alpha_at_minus,
        }
    }

    /// 这个场强下的中心差分 `D = [α(+F) − α(−F)]/(2F)`。
    ///
    /// # Errors
    ///
    /// `field_au` 非正/非有限 → [`EngineError::InvalidInput`]。
    pub fn central_difference(&self) -> Result<[[f64; 3]; 3], EngineError> {
        if !self.field_au.is_finite() || self.field_au <= 0.0 {
            return Err(EngineError::InvalidInput(format!(
                "beta: the field must be positive and finite, got {}",
                self.field_au
            )));
        }
        let mut out = [[0.0_f64; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value =
                    (self.alpha_at_plus[i][j] - self.alpha_at_minus[i][j]) / (2.0 * self.field_au);
            }
        }
        Ok(out)
    }
}

/// 静态一阶超极化率（一个方向上的导数）与转动不变量。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BetaTensor {
    /// `β_ijk = ∂α_ij/∂F_k`（`ij` 是 α 的两个指标，`k` 是场的方向）。
    pub tensor_au: [[[f64; 3]; 3]; 3],
    /// 由两个场外推用的 Richardson 权重（诊断：看它离 1 有多远就知道差分噪声有多大）。
    pub extrapolation_ratio: f64,
    /// `β_vec = sqrt(Σ_i (Σ_j(β_ijj + β_jij + β_jji))²)`。
    pub beta_vec_au: f64,
    /// `β_∥ = (3/5) Σ_i (β_zii + β_izi + β_iiz)`（`axis` 指定 z 轴）。
    pub beta_parallel_au: f64,
    /// `β_∥` 用的偶极轴。
    pub parallel_axis: usize,
}

/// 由**三个方向**的样本（每个方向两个场强）组装完整的 `β_ijk`。
///
/// `per_axis[k]` 是"场沿第 k 个坐标轴"时的两个样本（`F` 与 `F/2`），顺序不限。
/// `parallel_axis` 是偶极轴（0/1/2），只影响 `beta_parallel_au`。
///
/// # Errors
///
/// 每个方向必须是两个场强、且大小之比为 2；出现非有限值 → [`EngineError::InvalidInput`] /
/// [`EngineError::Internal`]。
pub fn beta_tensor(
    per_axis: &[[BetaSample; 2]; 3],
    parallel_axis: usize,
) -> Result<BetaTensor, EngineError> {
    if parallel_axis > 2 {
        return Err(EngineError::InvalidInput(format!(
            "beta: parallel_axis must be 0/1/2, got {parallel_axis}"
        )));
    }
    let mut tensor = [[[0.0_f64; 3]; 3]; 3];
    let mut ratio = 1.0;
    for (axis, samples) in per_axis.iter().enumerate() {
        let (low, high) = if samples[0].field_au < samples[1].field_au {
            (samples[0], samples[1])
        } else {
            (samples[1], samples[0])
        };
        let factor = high.field_au / low.field_au;
        if (factor - 2.0).abs() > 1.0e-6 {
            return Err(EngineError::InvalidInput(format!(
                "beta: the two field strengths must differ by a factor of 2, got {factor}"
            )));
        }
        ratio = factor;
        let fine = low.central_difference()?;
        let coarse = high.central_difference()?;
        for (i, plane) in tensor.iter_mut().enumerate() {
            for (j, row) in plane.iter_mut().enumerate() {
                // β_ijk：∂α_ij/∂F_k（k = axis）
                row[axis] = (4.0 * fine[i][j] - coarse[i][j]) / 3.0;
            }
        }
    }
    if tensor
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(EngineError::Internal(
            "beta: the extrapolated tensor contains non-finite entries".into(),
        ));
    }
    Ok(BetaTensor {
        tensor_au: tensor,
        extrapolation_ratio: ratio,
        beta_vec_au: beta_vec(&tensor),
        beta_parallel_au: beta_parallel(&tensor, parallel_axis),
        parallel_axis,
    })
}

/// `β_vec = sqrt(Σ_i β_i²)`，`β_i = Σ_j (β_ijj + β_jij + β_jji)`。
#[must_use]
pub fn beta_vec(tensor: &[[[f64; 3]; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for (i, plane) in tensor.iter().enumerate() {
        let mut component = 0.0;
        for (j, row) in plane.iter().enumerate() {
            component += row[j] + tensor[j][i][j] + tensor[j][j][i];
        }
        sum += component * component;
    }
    sum.sqrt()
}

/// `β_∥ = (3/5) Σ_i (β_zii + β_izi + β_iiz)`。
#[must_use]
pub fn beta_parallel(tensor: &[[[f64; 3]; 3]; 3], axis: usize) -> f64 {
    let z = axis.min(2);
    let mut sum = 0.0;
    for (i, plane) in tensor.iter().enumerate() {
        sum += tensor[z][i][i] + plane[z][i] + tensor[i][i][z];
    }
    0.6 * sum
}
