//! `nalgebra` 后端（默认，R-11/§14 #5）。

use nalgebra::DMatrix;

use crate::{LinalgError, Matrix, Vector, canonicalize_eigen};

fn to_nalgebra(matrix: &Matrix) -> DMatrix<f64> {
    DMatrix::from_column_slice(matrix.nrows(), matrix.ncols(), matrix.as_column_major())
}

fn from_nalgebra(matrix: &DMatrix<f64>) -> Matrix {
    Matrix::from_column_major(matrix.nrows(), matrix.ncols(), matrix.as_slice().to_vec())
        .expect("nalgebra dimensions agree by construction")
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
    let product = to_nalgebra(a) * to_nalgebra(b);
    Ok(from_nalgebra(&product))
}

/// 对称矩阵的完全特征分解：返回（升序特征值，特征向量按列）。
///
/// 只读下三角（调用方保证对称；`Engine` 侧用 `is_symmetric` 断言）。
///
/// # Errors
///
/// 非方阵。
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vector, Matrix), LinalgError> {
    require_square(a, "symmetric_eigen")?;
    let eigen = nalgebra::linalg::SymmetricEigen::new(to_nalgebra(a));
    let values: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    let vectors = from_nalgebra(&eigen.eigenvectors);
    Ok(canonicalize_eigen(values, vectors))
}

/// Cholesky 分解 `A = L Lᵀ`，返回下三角 `L`。
///
/// # Errors
///
/// 非方阵，或不是正定矩阵。
pub fn cholesky(a: &Matrix) -> Result<Matrix, LinalgError> {
    require_square(a, "cholesky")?;
    let factorization = nalgebra::linalg::Cholesky::new(to_nalgebra(a)).ok_or_else(|| {
        LinalgError::NotPositiveDefinite {
            context: "cholesky",
            detail: "nalgebra returned None (matrix is not positive definite)".to_string(),
        }
    })?;
    let mut lower = factorization.l();
    // `l()` 可能带上三角的零，但为保险起见显式清零上三角。
    for i in 0..lower.nrows() {
        for j in (i + 1)..lower.ncols() {
            lower[(i, j)] = 0.0;
        }
    }
    Ok(from_nalgebra(&lower))
}

// 解 `A x = b`（对称正定）在抽象层统一实现（`crate::solve_symmetric`）：
// 本后端只提供 `cholesky`，三角回代是共享代码，两个后端不可能给出不同的 `x`。
