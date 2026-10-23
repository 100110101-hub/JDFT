//! Pulay DIIS（direct inversion in the iterative subspace）收敛加速器。
//!
//! 每步把 Fock 矩阵 `F_i` 与它的交换子误差 `e_i = F_i P_i S − S P_i F_i`
//! 存进子空间，然后解
//!
//! ```text
//! [ B  −1 ] [c]   [0]
//! [ −1  0 ] [λ] = [−1]        B_ij = ⟨e_i, e_j⟩,  Σ c_i = 1
//! ```
//!
//! 用 `F = Σ c_i F_i` 代替当步 Fock 去对角化。解法退化（矩阵奇异、系数非有限）
//! 时不抛错也不硬来，直接返回 `None` 让调用方退回普通迭代——加速器失效不该
//! 让整个 SCF 变成错误。

use qwb_linalg::{LinalgError, Matrix, matmul};

/// 解 DIIS 的加边线性系统（对称但**不定**，所以不能用 Cholesky）。
///
/// 部分主元高斯消元；遇到奇异或非有限值时返回 `None`。系统只有 `m+1` 阶
/// （`m` = 子空间大小，通常 ≤ 8），自己做一遍比给 `qwb-linalg` 的冻结接口
/// 加一个通用求解器更省事，也不会动 I-00 的契约。
fn solve_bordered(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let mut column = 0;
    while column < n {
        // 部分主元：选该列绝对值最大的行。
        let mut pivot_row = column;
        let mut pivot_value = a[column][column].abs();
        let mut row = column + 1;
        while row < n {
            let candidate = a[row][column].abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = row;
            }
            row += 1;
        }
        if pivot_value.is_nan() || pivot_value <= 1.0e-14 {
            return None;
        }
        a.swap(column, pivot_row);
        b.swap(column, pivot_row);

        let pivot = a[column][column];
        let (upper, lower) = a.split_at_mut(column + 1);
        let pivot_values = &upper[column];
        let mut offset = 0;
        while offset < lower.len() {
            let row_values = &mut lower[offset];
            let factor = row_values[column] / pivot;
            if factor != 0.0 {
                let mut col = column;
                while col < n {
                    row_values[col] -= factor * pivot_values[col];
                    col += 1;
                }
                b[column + 1 + offset] -= factor * b[column];
            }
            offset += 1;
        }
        column += 1;
    }

    // 回代。
    let mut x = vec![0.0_f64; n];
    let mut row = n;
    while row > 0 {
        row -= 1;
        let row_values = &a[row];
        let mut sum = b[row];
        let mut col = row + 1;
        while col < n {
            sum -= row_values[col] * x[col];
            col += 1;
        }
        x[row] = sum / row_values[row];
        if !x[row].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// DIIS 子空间。
#[derive(Debug, Clone)]
pub struct Diis {
    max_vectors: usize,
    focks: Vec<Matrix>,
    errors: Vec<Matrix>,
}

impl Diis {
    /// 建一个最多保留 `max_vectors` 组向量的加速器（下限按 2 处理）。
    #[must_use]
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors: max_vectors.max(2),
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// 当前子空间里的向量组数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.focks.len()
    }

    /// 子空间是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.focks.is_empty()
    }

    /// 清空子空间（重启 DIIS 时用）。
    pub fn clear(&mut self) {
        self.focks.clear();
        self.errors.clear();
    }

    /// 交换子误差 `e = F P S − S P F`。
    ///
    /// 收敛时 `F` 与 `P` 对易，`e` 归零，所以它同时是收敛程度的度量。
    ///
    /// # Errors
    ///
    /// 三个矩阵形状不一致。
    pub fn commutator(
        fock: &Matrix,
        density: &Matrix,
        overlap: &Matrix,
    ) -> Result<Matrix, LinalgError> {
        let fps = matmul(&matmul(fock, density)?, overlap)?;
        let spf = matmul(&matmul(overlap, density)?, fock)?;
        let mut error = Matrix::zeros(fps.nrows(), fps.ncols());
        for i in 0..fps.nrows() {
            for j in 0..fps.ncols() {
                error.set(i, j, fps.get(i, j) - spf.get(i, j));
            }
        }
        Ok(error)
    }

    /// 推入一组 `(F, e)`；超过上限时丢掉最旧的一组。
    pub fn push(&mut self, fock: Matrix, error: Matrix) {
        if self.focks.len() >= self.max_vectors {
            self.focks.remove(0);
            self.errors.remove(0);
        }
        self.focks.push(fock);
        self.errors.push(error);
    }

    /// 导出子空间（用于检查点序列化）：返回 `(Fock 们, 误差们)`，都是列主序展平。
    #[must_use]
    pub fn snapshot(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        (
            self.focks
                .iter()
                .map(|f| f.as_column_major().to_vec())
                .collect(),
            self.errors
                .iter()
                .map(|e| e.as_column_major().to_vec())
                .collect(),
        )
    }

    /// 从检查点恢复子空间。
    ///
    /// # Errors
    ///
    /// 任意矩阵的展平长度与 `n*n` 不符，或 Fock/误差数量不一致。
    pub fn restore(
        max_vectors: usize,
        n: usize,
        focks: Vec<Vec<f64>>,
        errors: Vec<Vec<f64>>,
    ) -> Result<Self, LinalgError> {
        if focks.len() != errors.len() {
            return Err(LinalgError::Dimension {
                context: "diis restore: focks and errors differ in count",
                left_rows: focks.len(),
                left_cols: 1,
                right_rows: errors.len(),
                right_cols: 1,
            });
        }
        let mut accelerator = Self::new(max_vectors);
        for (fock, error) in focks.into_iter().zip(errors) {
            if fock.len() != n * n || error.len() != n * n {
                return Err(LinalgError::Dimension {
                    context: "diis restore: flattened vector size mismatch",
                    left_rows: fock.len().max(error.len()),
                    left_cols: 1,
                    right_rows: n * n,
                    right_cols: 1,
                });
            }
            accelerator.push(
                Matrix::from_column_major(n, n, fock)?,
                Matrix::from_column_major(n, n, error)?,
            );
        }
        Ok(accelerator)
    }

    /// 最近一组误差向量的 Frobenius 范数（给调用方做诊断/收敛判据）。
    #[must_use]
    pub fn error_norm(&self) -> f64 {
        self.errors.last().map_or(0.0, |error| {
            error
                .as_column_major()
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
    }

    /// 外推得到组合 Fock 矩阵；子空间不足 2 组或线性系统退化时返回 `None`。
    ///
    /// # Errors
    ///
    /// 只在矩阵构造返回错误时冒泡；退化情形按 `Ok(None)` 处理。
    pub fn extrapolate(&self) -> Result<Option<Matrix>, LinalgError> {
        let m = self.focks.len();
        if m < 2 {
            return Ok(None);
        }
        let dim = m + 1;
        let mut a = vec![vec![0.0_f64; dim]; dim];
        let mut i = 0;
        while i < m {
            let ei = self.errors[i].as_column_major();
            let mut j = 0;
            while j < m {
                let ej = self.errors[j].as_column_major();
                let mut dot = 0.0;
                for (x, y) in ei.iter().zip(ej.iter()) {
                    dot += x * y;
                }
                // 极小对角正则：误差向量近线性相关时避免整个系统退化。
                a[i][j] = if i == j { dot + 1.0e-12 } else { dot };
                j += 1;
            }
            a[i][m] = -1.0;
            a[m][i] = -1.0;
            i += 1;
        }
        a[m][m] = 0.0;

        let mut rhs = vec![0.0_f64; dim];
        rhs[m] = -1.0;

        let Some(coefficients) = solve_bordered(a, rhs) else {
            return Ok(None);
        };

        let rows = self.focks[0].nrows();
        let cols = self.focks[0].ncols();
        let mut flat = vec![0.0_f64; rows * cols];
        for (index, fock) in self.focks.iter().enumerate() {
            let weight = coefficients[index];
            if !weight.is_finite() {
                return Ok(None);
            }
            for (accumulated, value) in flat.iter_mut().zip(fock.as_column_major()) {
                *accumulated += weight * value;
            }
        }
        Ok(Some(Matrix::from_column_major(rows, cols, flat)?))
    }
}
