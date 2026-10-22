//! I-00 验收：同一套接口、两个后端、同一批数值。
//!
//! 这个测试**不依赖 feature**：`verify.sh` 里跑默认后端（nalgebra），CI 另外跑
//! `--no-default-features --features faer`。两边都必须与这里的 golden 值一致，
//! 这就是"同一接口 roundtrip identical"的可执行定义。
//!
//! golden 值都是**解析可算**的，不是从某个后端抄来的：
//!
//! - `A = [[4, 1], [1, 3]]`：特征值 `(7 ∓ √5)/2`，即 2.381966… 与 4.618033…
//! - Cholesky：`L = [[2, 0], [0.5, √(11)/2]]`
//! - `A x = b`，`b = [1, 2]`：`x = [1/11, 7/11]`

use qwb_linalg::{
    LinalgError, Matrix, Vector, backend_name, cholesky, matmul, solve_symmetric, symmetric_eigen,
};

fn a() -> Matrix {
    Matrix::from_row_major(2, 2, &[4.0, 1.0, 1.0, 3.0]).expect("shape")
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let tolerance = 1e-13 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "[{}] {what}: {actual} vs {expected} (tol {tolerance:e})",
        backend_name()
    );
}

#[test]
fn backend_is_reported() {
    assert!(matches!(backend_name(), "nalgebra" | "faer"));
}

#[test]
fn matmul_matches_hand_computation() {
    // A * A = [[4,1],[1,3]]^2 = [[17, 7], [7, 10]]
    let product = matmul(&a(), &a()).expect("2x2 * 2x2");
    assert_close(product[(0, 0)], 17.0, "c00");
    assert_close(product[(0, 1)], 7.0, "c01");
    assert_close(product[(1, 0)], 7.0, "c10");
    assert_close(product[(1, 1)], 10.0, "c11");

    // 非方阵乘法的形状与维度检查。
    let b = Matrix::from_row_major(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("shape");
    let p = matmul(&a(), &b).expect("2x2 * 2x3");
    assert_eq!((p.nrows(), p.ncols()), (2, 3));
    assert!(matches!(
        matmul(&b, &a()),
        Err(LinalgError::Dimension { .. })
    ));
}

#[test]
fn symmetric_eigenvalues_are_analytic() {
    let (values, vectors) = symmetric_eigen(&a()).expect("2x2 symmetric");

    // (7 - sqrt(5))/2 与 (7 + sqrt(5))/2，升序。
    let root5 = 5.0_f64.sqrt();
    assert_close(values.get(0), (7.0 - root5) / 2.0, "lambda0");
    assert_close(values.get(1), (7.0 + root5) / 2.0, "lambda1");
    assert_close(values.get(0) + values.get(1), 7.0, "trace check");

    // 特征向量必须满足 A v = lambda v，且是单位向量。
    for column in 0..2 {
        let v = Vector::new((0..2).map(|row| vectors[(row, column)]).collect());
        let lam = values.get(column);
        for row in 0..2 {
            let av = a()[(row, 0)] * v.get(0) + a()[(row, 1)] * v.get(1);
            assert_close(av, lam * v.get(row), "A v = lambda v");
        }
        assert_close(v.norm(), 1.0, "eigenvector norm");
        // 符号规范化：绝对值最大的分量取正。
        let mut best_row = 0;
        let mut best = 0.0;
        for row in 0..2 {
            if v.get(row).abs() > best {
                best = v.get(row).abs();
                best_row = row;
            }
        }
        assert!(
            v.get(best_row) > 0.0,
            "eigenvector sign must be canonicalized"
        );
    }
}

#[test]
fn eigen_order_and_signs_are_deterministic_across_backends() {
    // 对角化一个 3x3 对称矩阵两次，结果必须逐位相同（同一后端内），
    // 且满足“升序 + 最大分量取正”的规范化 —— 这是两个后端能对齐的前提。
    let m = Matrix::from_row_major(3, 3, &[2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
        .expect("shape");
    let (values_a, vectors_a) = symmetric_eigen(&m).expect("eigen");
    let (values_b, vectors_b) = symmetric_eigen(&m).expect("eigen");

    assert_eq!(values_a, values_b);
    assert_eq!(vectors_a, vectors_b);

    // 解析特征值：2, 2±√2（升序）。
    let root2 = 2.0_f64.sqrt();
    assert_close(values_a.get(0), 2.0 - root2, "lambda0");
    assert_close(values_a.get(1), 2.0, "lambda1");
    assert_close(values_a.get(2), 2.0 + root2, "lambda2");
}

#[test]
fn cholesky_is_analytic() {
    let lower = cholesky(&a()).expect("positive definite");
    // L = [[2, 0], [0.5, sqrt(11)/2]]
    assert_close(lower[(0, 0)], 2.0, "l00");
    assert_close(lower[(0, 1)], 0.0, "l01 must be zero (lower triangular)");
    assert_close(lower[(1, 0)], 0.5, "l10");
    assert_close(lower[(1, 1)], 11.0_f64.sqrt() / 2.0, "l11");

    // L Lᵀ == A
    let product = matmul(&lower, &lower.transpose()).expect("L * L^T");
    for i in 0..2 {
        for j in 0..2 {
            assert_close(product[(i, j)], a()[(i, j)], "L L^T = A");
        }
    }
}

#[test]
fn cholesky_rejects_indefinite_matrices() {
    let indefinite = Matrix::from_row_major(2, 2, &[1.0, 2.0, 2.0, 1.0]).expect("shape");
    assert!(matches!(
        cholesky(&indefinite),
        Err(LinalgError::NotPositiveDefinite { .. })
    ));
}

#[test]
fn solve_symmetric_is_analytic() {
    let b = Vector::new(vec![1.0, 2.0]);
    let x = solve_symmetric(&a(), &b).expect("solve");
    // 精确解 x = [1/11, 7/11]。
    assert_close(x.get(0), 1.0 / 11.0, "x0");
    assert_close(x.get(1), 7.0 / 11.0, "x1");

    // 回代检查 A x = b。
    for row in 0..2 {
        let left = a()[(row, 0)] * x.get(0) + a()[(row, 1)] * x.get(1);
        assert_close(left, b.get(row), "A x = b");
    }

    assert!(matches!(
        solve_symmetric(&a(), &Vector::new(vec![1.0])),
        Err(LinalgError::Dimension { .. })
    ));
}
