//! Pulay 的「能量加权密度」（W 矩阵）与它的基本恒等式（G-01 的前置）。
//!
//! # 为什么单独做这一步
//!
//! 解析梯度里最容易被写错、又最不容易自证的一项，是**分子轨道随几何弛豫**带来的贡献。
//! 对闭壳层 SCF，它在梯度里以
//!
//! ```text
//! −Tr(W dS/dR)         W = ½ · P F P
//! ```
//!
//! 的形式出现。系数 ½、F 出现的位置、以及整体的符号，光靠记忆写三次里能错两次。
//!
//! 但这条恒等式**可以独立验证**，不需要任何外部参照：
//!
//! ```text
//! 对固定的核位移 dR：  Tr(F dP) = −Tr(W dS)   （一阶）
//! ```
//!
//! 只要把两个相邻几何都各自收敛，就能用 `dP = P(R+δ) − P(R)`、`dS = S(R+δ) − S(R)`
//! 直接检验它。本条测试通过之后，梯度实现里就可以放心地用 `W = ½PFP`。
//!
//! # 推导（供对照）
//!
//! `P = 2Σ_i c_i c_iᵀ`、`F c_i = S c_i ε_i`（收敛后 MO 基下 F 对角），于是
//! `P F P = 4 Σ_i ε_i c_i c_iᵀ`；再由 `CᵀSC = I` 的微分把 `dc_i` 换成 `dS`，
//! 就得到上面那条恒等式。

use qwb_engine_api::EngineError;
use qwb_linalg::{LinalgError, Matrix, matmul};

/// `W = ½ P F P`（闭壳层能量加权密度）。
///
/// # Errors
///
/// 矩阵形状不匹配。
pub fn pulay_w(density: &Matrix, fock: &Matrix) -> Result<Matrix, EngineError> {
    let n = density.nrows();
    if fock.nrows() != n || fock.ncols() != n {
        return Err(EngineError::Internal(format!(
            "pulay_w: density is {n}x{}, fock is {}x{}",
            density.ncols(),
            fock.nrows(),
            fock.ncols()
        )));
    }
    let pf = matmul(density, fock).map_err(linalg)?;
    let pfp = matmul(&pf, density).map_err(linalg)?;
    let mut w = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = 0.5 * pfp[(i, j)];
        }
    }
    Ok(w)
}

/// `Tr(A B)`。
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut total = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            total += a[(i, j)] * b[(j, i)];
        }
    }
    total
}

/// `Tr(F·dP) + Tr(W·dS)`：恒等式成立时它应为一阶小量（≈ 0）。
///
/// 拆出来单独暴露，方便测试给出「残差相对量级」而不是只给一个布尔值。
#[must_use]
pub fn pulay_residual(fock: &Matrix, d_density: &Matrix, w: &Matrix, d_overlap: &Matrix) -> f64 {
    trace_product(fock, d_density) + trace_product(w, d_overlap)
}

fn linalg(error: LinalgError) -> EngineError {
    EngineError::Internal(format!("pulay_w: {error}"))
}
