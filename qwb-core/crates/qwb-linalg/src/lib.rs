//! I-00：线性代数抽象层。`nalgebra` 是默认后端，`faer` 是可选后端（R-11）。
//!
//! # 为什么要有这一层
//!
//! R-11 的风险是"nalgebra 性能不够、换后端要动全工程"。所以引擎**不直接用**
//! `nalgebra`/`faer` 的类型，只用本 crate 的 [`Matrix`] / [`Vector`]；后端在编译期
//! 通过 feature 选一个（`--no-default-features --features faer`）。
//!
//! # 可复现性（G7 / R-13 的延伸）
//!
//! 两个后端必须给出**同样的数**，否则"换后端"会改变结果位模式。因此本层在两个后端
//! 之上强制两条规范化：
//!
//! 1. 特征值按**升序**排列（特征向量按列同步重排）；
//! 2. 特征向量的**符号规范化**：绝对值最大的分量取正。
//!
//! 没有这两条，`nalgebra` 与 `faer` 的特征向量会差一个符号、顺序也可能不同。
//!
//! # 约定
//!
//! 矩阵是**列主序**稠密 `f64`（与 Fortran/LAPACK 一致，MKL/BLAS 换后端时不需要转置）。
#![deny(unsafe_code)]
#![deny(missing_docs)]

use thiserror::Error;

#[cfg(all(feature = "nalgebra", feature = "faer"))]
compile_error!("qwb-linalg: enable exactly one backend feature, not both (`nalgebra` or `faer`)");

#[cfg(not(any(feature = "nalgebra", feature = "faer")))]
compile_error!("qwb-linalg: enable exactly one backend feature (`nalgebra` or `faer`)");

#[cfg(feature = "faer")]
mod faer_backend;
#[cfg(feature = "nalgebra")]
mod nalgebra_backend;

#[cfg(feature = "faer")]
pub use faer_backend::{cholesky, matmul, symmetric_eigen};
#[cfg(feature = "nalgebra")]
pub use nalgebra_backend::{cholesky, matmul, symmetric_eigen};

/// 解 `A x = b`（`A` 对称正定），用 Cholesky 前代/回代。
///
/// 实现在**抽象层**而不是各后端里：两个后端只提供 `cholesky`，三角回代是共享代码，
/// 因此不可能出现"nalgebra 与 faer 解出不同的 x"。这也让 I-00 的后端替换面更小。
///
/// # Errors
///
/// 非方阵、`b` 长度不匹配、`A` 非正定。
pub fn solve_symmetric(a: &Matrix, b: &Vector) -> Result<Vector, LinalgError> {
    if !a.is_square() {
        return Err(LinalgError::Dimension {
            context: "solve_symmetric",
            left_rows: a.nrows(),
            left_cols: a.ncols(),
            right_rows: a.nrows(),
            right_cols: a.nrows(),
        });
    }
    let n = a.nrows();
    if b.len() != n {
        return Err(LinalgError::Dimension {
            context: "solve_symmetric",
            left_rows: n,
            left_cols: n,
            right_rows: b.len(),
            right_cols: 1,
        });
    }

    let lower = cholesky(a)?;

    // 前代：L y = b
    let mut y = Vector::zeros(n);
    for i in 0..n {
        let mut sum = b.get(i);
        for j in 0..i {
            sum -= lower[(i, j)] * y.get(j);
        }
        let diagonal = lower[(i, i)];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return Err(LinalgError::NotPositiveDefinite {
                context: "solve_symmetric",
                detail: format!("zero or non-finite diagonal at row {i}"),
            });
        }
        y.set(i, sum / diagonal);
    }

    // 回代：Lᵀ x = y
    let mut x = Vector::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y.get(i);
        for j in (i + 1)..n {
            sum -= lower[(j, i)] * x.get(j);
        }
        x.set(i, sum / lower[(i, i)]);
    }
    Ok(x)
}

/// 当前编译进去的后端名（日志/`env.json` 用）。
#[must_use]
pub fn backend_name() -> &'static str {
    if cfg!(feature = "faer") {
        "faer"
    } else {
        "nalgebra"
    }
}

/// 线性代数错误。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LinalgError {
    /// 维度不匹配。
    #[error(
        "dimension mismatch: {context}: left is {left_rows}x{left_cols}, right is {right_rows}x{right_cols}"
    )]
    Dimension {
        /// 出问题的运算。
        context: &'static str,
        /// 左操作数行数。
        left_rows: usize,
        /// 左操作数列数。
        left_cols: usize,
        /// 右操作数行数。
        right_rows: usize,
        /// 右操作数列数。
        right_cols: usize,
    },
    /// 矩阵不是正定（Cholesky / 对称求解失败）。
    #[error("matrix is not positive definite ({context}): {detail}")]
    NotPositiveDefinite {
        /// 出问题的运算。
        context: &'static str,
        /// 后端给出的细节。
        detail: String,
    },
    /// 后端报的其他错误。
    #[error("backend error ({context}): {detail}")]
    Backend {
        /// 出问题的运算。
        context: &'static str,
        /// 后端给出的细节。
        detail: String,
    },
}

/// 稠密列主序矩阵。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// 用**列主序**数据构造。
    ///
    /// # Errors
    ///
    /// `data.len() != rows * cols`。
    pub fn from_column_major(
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    ) -> Result<Self, LinalgError> {
        if data.len() != rows * cols {
            return Err(LinalgError::Dimension {
                context: "Matrix::from_column_major",
                left_rows: rows,
                left_cols: cols,
                right_rows: data.len(),
                right_cols: 1,
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// 用**行主序**数据构造（`result.json` 的 list-of-list 就是这个顺序）。
    ///
    /// # Errors
    ///
    /// `data.len() != rows * cols`。
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> Result<Self, LinalgError> {
        if data.len() != rows * cols {
            return Err(LinalgError::Dimension {
                context: "Matrix::from_row_major",
                left_rows: rows,
                left_cols: cols,
                right_rows: data.len(),
                right_cols: 1,
            });
        }
        let mut column_major = vec![0.0; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                column_major[j * rows + i] = data[i * cols + j];
            }
        }
        Ok(Self {
            rows,
            cols,
            data: column_major,
        })
    }

    /// 全零矩阵。
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// 单位矩阵。
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut matrix = Self::zeros(n, n);
        for i in 0..n {
            matrix[(i, i)] = 1.0;
        }
        matrix
    }

    /// 行数。
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// 列数。
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// 是否方阵。
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// 列主序切片。
    #[must_use]
    pub fn as_column_major(&self) -> &[f64] {
        &self.data
    }

    /// 按 `(row, col)` 读。
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.rows + row]
    }

    /// 按 `(row, col)` 写。
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[col * self.rows + row] = value;
    }

    /// 转置。
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// 迹（方阵）。
    ///
    /// # Errors
    ///
    /// 非方阵。
    pub fn trace(&self) -> Result<f64, LinalgError> {
        if !self.is_square() {
            return Err(LinalgError::Dimension {
                context: "Matrix::trace",
                left_rows: self.rows,
                left_cols: self.cols,
                right_rows: self.rows,
                right_cols: self.rows,
            });
        }
        Ok((0..self.rows).map(|i| self[(i, i)]).sum())
    }

    /// 是否（在容差内）对称。
    #[must_use]
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self[(i, j)] - self[(j, i)]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[col * self.rows + row]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[col * self.rows + row]
    }
}

/// 稠密向量。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// 从 `Vec<f64>` 构造。
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// 全零向量。
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// 长度。
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 数据切片。
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// 拥有所有权的数据。
    #[must_use]
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// 按索引读。
    #[must_use]
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// 按索引写。
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// 欧氏范数。
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// 点积。
    ///
    /// # Errors
    ///
    /// 长度不同。
    pub fn dot(&self, other: &Self) -> Result<f64, LinalgError> {
        if self.len() != other.len() {
            return Err(LinalgError::Dimension {
                context: "Vector::dot",
                left_rows: self.len(),
                left_cols: 1,
                right_rows: other.len(),
                right_cols: 1,
            });
        }
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }
}

/// 把特征值按升序重排，并对特征向量列做符号规范化（两个后端共用，保证位模式一致）。
///
/// `eigenvalues[i]` 对应 `eigenvectors` 的第 `i` 列。
pub(crate) fn canonicalize_eigen(
    mut eigenvalues: Vec<f64>,
    mut eigenvectors: Matrix,
) -> (Vector, Matrix) {
    let n = eigenvalues.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        eigenvalues[*a]
            .partial_cmp(&eigenvalues[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_values: Vec<f64> = order.iter().map(|i| eigenvalues[*i]).collect();
    let mut sorted_vectors = Matrix::zeros(n, n);
    for (target_col, source_col) in order.iter().enumerate() {
        for row in 0..n {
            sorted_vectors[(row, target_col)] = eigenvectors[(row, *source_col)];
        }
    }
    eigenvalues.clone_from(&sorted_values);

    // 符号规范化：每列里绝对值最大的分量取正。
    for col in 0..n {
        let mut best_row = 0;
        let mut best = 0.0_f64;
        for row in 0..n {
            let value = sorted_vectors[(row, col)].abs();
            if value > best {
                best = value;
                best_row = row;
            }
        }
        if sorted_vectors[(best_row, col)] < 0.0 {
            for row in 0..n {
                sorted_vectors[(row, col)] = -sorted_vectors[(row, col)];
            }
        }
    }

    eigenvectors = sorted_vectors;
    (Vector::new(eigenvalues), eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::{LinalgError, Matrix, Vector, backend_name};

    #[test]
    fn backend_name_is_one_of_the_two() {
        assert!(matches!(backend_name(), "nalgebra" | "faer"));
    }

    #[test]
    fn row_major_roundtrip() {
        let m = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
        // 列主序存储：第 0 列是 [1, 4]。
        assert_eq!(m.as_column_major(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        assert!(matches!(
            Matrix::from_row_major(2, 2, &[1.0, 2.0, 3.0]),
            Err(LinalgError::Dimension { .. })
        ));
    }

    #[test]
    fn identity_and_trace() {
        let i = Matrix::identity(3);
        assert_eq!(i.trace().expect("square"), 3.0);
        assert!(i.is_symmetric(0.0));
        let t = i.transpose();
        assert_eq!(t, i);
    }

    #[test]
    fn vector_ops() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b).expect("same len"), 32.0);
        assert!((a.norm() - 14.0_f64.sqrt()).abs() < 1e-15);
        assert!(a.dot(&Vector::new(vec![1.0])).is_err());
    }
}
