//! `faer` 后端（可选 feature，R-11 的换后端保险）。
//!
//! 与 `nalgebra` 后端**同一个接口**：`matmul` / `symmetric_eigen` / `cholesky`。
//! `solve_symmetric` 不在这里实现 —— 它由抽象层用 `cholesky` 统一实现，
//! 这样两个后端不可能在三角回代上出现分歧。
//!
//! 只用 faer 的稳定入口（`linalg::matmul::matmul`、`prelude` 的
//! `self_adjoint_eigen` / `llt`），不碰内部布局。

use faer::{Accum, Par, Side};

use crate::{LinalgError, Matrix, Vector, canonicalize_eigen};

fn to_faer(matrix: &Matrix) -> faer::Mat<f64> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let data = matrix.as_column_major();
    faer::Mat::<f64>::from_fn(rows, columns, |i, j| data[j * rows + i])
}

fn from_faer_ref(matrix: faer::MatRef<'_, f64>) -> Matrix {
    let mut out = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            out[(i, j)] = matrix[(i, j)];
        }
    }
    out
}

fn require_square(matrix: &Matrix, context: &'static str) -> Result<(), LinalgError> {
    if matrix.is_square() {
        Ok(())
    } else {
        Err(LinalgError::Dimension {
            context,
            left_rows: matrix.nrows(),
            left_cols: matrix.ncols(),
            right_rows: matrix.nrows(),
            right_cols: matrix.nrows(),
        })
    }
}

/// `a * b`。
///
/// # Errors
///
/// `a.ncols() != b.nrows()`。
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, LinalgError> {
    if a.ncols() != b.nrows() {
        return Err(LinalgError::Dimension {
            context: "matmul",
            left_rows: a.nrows(),
            left_cols: a.ncols(),
            right_rows: b.nrows(),
            right_cols: b.ncols(),
        });
    }
    let lhs = to_faer(a);
    let rhs = to_faer(b);
    let mut dst = faer::Mat::<f64>::zeros(a.nrows(), b.ncols());
    // 确定性：(beta=Replace, alpha=1, Par::Seq) —— 不引入并行归约顺序（G7）。
    faer::linalg::matmul::matmul(
        dst.as_mut(),
        Accum::Replace,
        lhs.as_ref(),
        rhs.as_ref(),
        1.0,
        Par::Seq,
    );
    Ok(from_faer_ref(dst.as_ref()))
}

/// 对称矩阵的完全特征分解（升序特征值 + 特征向量按列，符号规范化）。
///
/// # Errors
///
/// 非方阵，或 faer 的特征分解失败。
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vector, Matrix), LinalgError> {
    require_square(a, "symmetric_eigen")?;
    let m = to_faer(a);
    let eigen = m
        .self_adjoint_eigen(Side::Lower)
        .map_err(|err| LinalgError::Backend {
            context: "symmetric_eigen",
            detail: format!("{err:?}"),
        })?;

    let n = a.nrows();
    let diagonal = eigen.S();
    let values: Vec<f64> = (0..n).map(|i| diagonal[i]).collect();
    let vectors = from_faer_ref(eigen.U());
    Ok(canonicalize_eigen(values, vectors))
}

/// Cholesky 分解 `A = L Lᵀ`，返回下三角 `L`。
///
/// # Errors
///
/// 非方阵，或不是正定矩阵。
pub fn cholesky(a: &Matrix) -> Result<Matrix, LinalgError> {
    require_square(a, "cholesky")?;
    let m = to_faer(a);
    let llt = m
        .llt(Side::Lower)
        .map_err(|err| LinalgError::NotPositiveDefinite {
            context: "cholesky",
            detail: format!("{err:?}"),
        })?;

    let n = a.nrows();
    let lower = llt.L();
    let mut out = Matrix::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            out[(i, j)] = lower[(i, j)];
        }
    }
    Ok(out)
}
