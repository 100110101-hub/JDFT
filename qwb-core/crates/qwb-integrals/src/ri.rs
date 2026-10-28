//! IN-01：**RI（密度拟合）积分** —— 3 中心 `(ij|P)` 与 2 中心库仑 `(P|Q)`。
//!
//! # 记号
//!
//! 全部用**化学家记号**，与 [`crate::two_electron`] 一致：
//!
//! ```text
//! (ij|P) = ∫∫ φ_i(1)φ_j(1) r₁₂⁻¹ φ_P(2) d1 d2      i,j 是轨道基，P 是辅助基
//! (P|Q)  = ∫∫ φ_P(1)      r₁₂⁻¹ φ_Q(2) d1 d2
//! ```
//!
//! # 为什么这两个量是"同一个核函数的不同退化"
//!
//! 4 中心积分是 `(ab|cd)`：左边两个原函数、右边两个原函数。把右边**塌成一个**
//! 原函数（另一个的角动量取 0、指数取 0）就退化成 3 中心 `(ab|c)`；再把左边也
//! 塌掉就是 2 中心 `(P|Q)`。所以这里不写第二套 Hermite 递推 ——
//! 直接复用 [`crate::hermite2e::primitive_component`]：
//!
//! ```text
//! (ab|c) = primitive_component(pa, A, α, pb, B, β, pc, C, γ, (0,0,0), C, 0.0)
//! ```
//!
//! 塌掉的那一侧 `β = 0` 给出 `p = α`、`P = A`、`e_ab = 1`、`E^{0,0}_t = δ_{t0}`，
//! 于是 `Σ_t E_t·R_t` 只剩 `t = 0` 项 —— 恰好是 3 中心积分的定义式。
//! 这条"退化"把 IN-01 的数值风险压到最小：Hermite 核、Boys、归一化、球谐变换
//! 全都走已经被 4 中心积分测试钉住的同一条路径。
//!
//! # RI 拟合怎么用这两个量
//!
//! 密度拟合把 4 中心积分换成
//!
//! ```text
//! (ij|kl) ≈ Σ_{P,Q} (ij|P) [V⁻¹]_{PQ} (Q|kl),   V_{PQ} = (P|Q)
//! ```
//!
//! 拟合误差不是零（辅助基不完备），所以 SC-05 必须把残差落盘
//! （`ri_error_hartree`）。`V` 的逆不开方在这里做：RI-MP2 用 `V^{-1/2}`
//! （[`inverse_sqrt`]），RI-J 用 `V⁻¹`。

use qwb_engine_api::EngineError;
use qwb_linalg::Matrix;
use qwb_registry::Registry;

use crate::hermite2e::primitive_component;
use crate::one_electron::primitive_norm;
use crate::two_electron::{ShellBlock, shell_block};

/// 3 中心 RI 积分 `(ij|P)`。
///
/// 存储顺序是 `P` 在最外层：`values[(P * n_orbital + i) * n_orbital + j]`。
/// 这样 RI-MP2 里"对每个辅助函数取一个 `n_orb × n_orb` 矩阵"是一次连续读。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ThreeCenterIntegrals {
    /// 轨道基的基函数个数。
    pub n_orbital: usize,
    /// 辅助基的基函数个数。
    pub n_auxiliary: usize,
    /// 扁平存储的 `(ij|P)`，`P` 在最外层。
    pub values: Vec<f64>,
}

impl ThreeCenterIntegrals {
    /// 取 `(i j | P)`。
    #[must_use]
    pub fn get(&self, i: usize, j: usize, p: usize) -> f64 {
        self.values[(p * self.n_orbital + i) * self.n_orbital + j]
    }

    /// 某个辅助函数 `P` 上的整块 `(ij|P)`（`n_orb × n_orb`，行优先）。
    #[must_use]
    pub fn matrix_for(&self, p: usize) -> &[f64] {
        let n = self.n_orbital;
        &self.values[p * n * n..(p + 1) * n * n]
    }
}

/// 一维归一化常数（按分量归一，与 4 中心积分同一套）。
fn component_norm(powers: [u32; 3], gamma: f64) -> f64 {
    primitive_norm((powers[0], powers[1], powers[2]), gamma)
}

/// 3 中心 `(ij|P)` 的直角坐标块（含归范化与收缩）。
fn cartesian_block_three(a: &ShellBlock, b: &ShellBlock, c: &ShellBlock) -> Vec<f64> {
    let dims = [a.components.len(), b.components.len(), c.components.len()];
    let mut block = vec![0.0; dims[0] * dims[1] * dims[2]];
    for (ma, &pa) in a.components.iter().enumerate() {
        for (mb, &pb) in b.components.iter().enumerate() {
            for (mc, &pc) in c.components.iter().enumerate() {
                let mut sum = 0.0;
                for (i, &alpha) in a.exponents.iter().enumerate() {
                    let weight_a = a.coefficients[i] * component_norm(pa, alpha);
                    if weight_a == 0.0 {
                        continue;
                    }
                    for (j, &beta) in b.exponents.iter().enumerate() {
                        let weight_b = b.coefficients[j] * component_norm(pb, beta);
                        if weight_b == 0.0 {
                            continue;
                        }
                        for (k, &gamma) in c.exponents.iter().enumerate() {
                            let weight_c = c.coefficients[k] * component_norm(pc, gamma);
                            if weight_c == 0.0 {
                                continue;
                            }
                            // 第四侧塌成"指数 0 的 s 原函数"：见模块文档。
                            sum += weight_a
                                * weight_b
                                * weight_c
                                * primitive_component(
                                    pa,
                                    a.center,
                                    alpha,
                                    pb,
                                    b.center,
                                    beta,
                                    pc,
                                    c.center,
                                    gamma,
                                    [0, 0, 0],
                                    c.center,
                                    0.0,
                                );
                        }
                    }
                }
                block[(ma * dims[1] + mb) * dims[2] + mc] = sum;
            }
        }
    }
    block
}

/// 3 中心积分 `(ij|P)`。
///
/// `basis` 是轨道基，`aux` 是辅助基（两者的元素必须覆盖分子）。
///
/// # Errors
///
/// 球谐变换缺 `l`（辅助基里出现 `l >= 4` 的 g 函数时就是这条）→
/// [`EngineError::Unsupported`]。
pub fn three_center(
    basis: &qwb_basis::InternalBasis,
    aux: &qwb_basis::InternalBasis,
    _molecule: &qwb_engine_api::Molecule,
) -> Result<ThreeCenterIntegrals, EngineError> {
    let orbital: Vec<ShellBlock> = basis
        .shells()
        .iter()
        .map(shell_block)
        .collect::<Result<_, _>>()?;
    let auxiliary: Vec<ShellBlock> = aux
        .shells()
        .iter()
        .map(shell_block)
        .collect::<Result<_, _>>()?;
    let n_orb = basis.n_functions();
    let n_aux = aux.n_functions();
    let mut values = vec![0.0; n_aux * n_orb * n_orb];

    for c in auxiliary.iter() {
        for (ia, a) in orbital.iter().enumerate() {
            for b in orbital.iter().skip(ia) {
                let mut data = cartesian_block_three(a, b, c);
                let mut dims = [a.components.len(), b.components.len(), c.components.len()];
                for (axis, shell) in [a, b, c].iter().enumerate() {
                    let (next, next_dims) = contract3(&data, dims, axis, &shell.transform);
                    data = next;
                    dims = next_dims;
                }
                // `(ij|P)` 对 i,j 对称：散射到两个位置。
                let base = [a.first_function, b.first_function, c.first_function];
                let mut index = 0;
                for i in 0..dims[0] {
                    for j in 0..dims[1] {
                        for p in 0..dims[2] {
                            let value = data[index];
                            index += 1;
                            let gi = base[0] + i;
                            let gj = base[1] + j;
                            let gp = base[2] + p;
                            values[(gp * n_orb + gi) * n_orb + gj] = value;
                            values[(gp * n_orb + gj) * n_orb + gi] = value;
                        }
                    }
                }
            }
        }
    }

    Ok(ThreeCenterIntegrals {
        n_orbital: n_orb,
        n_auxiliary: n_aux,
        values,
    })
}

/// 3 个指标的球谐变换（与 4 中心的 [`contract`] 同一套逻辑，只减一维）。
fn contract3(
    data: &[f64],
    dims: [usize; 3],
    axis: usize,
    transform: &qwb_basis::spherical::Transform,
) -> (Vec<f64>, [usize; 3]) {
    let n_spherical = transform.n_spherical;
    let mut out_dims = dims;
    out_dims[axis] = n_spherical;
    let in_stride = [dims[1] * dims[2], dims[2], 1];
    let out_stride = [out_dims[1] * out_dims[2], out_dims[2], 1];
    let total: usize = dims.iter().product();
    let mut out = vec![0.0; out_dims.iter().product()];
    for (flat, &value) in data.iter().enumerate().take(total) {
        if value == 0.0 {
            continue;
        }
        let mut index = [0usize; 3];
        for k in 0..3 {
            index[k] = (flat / in_stride[k]) % dims[k];
        }
        let mut out_index = index;
        out_index[axis] = 0;
        let out_base: usize = (0..3).map(|k| out_index[k] * out_stride[k]).sum();
        for spherical_index in 0..n_spherical {
            let weight = transform.get(index[axis], spherical_index);
            if weight != 0.0 {
                out[out_base + spherical_index * out_stride[axis]] += weight * value;
            }
        }
    }
    (out, out_dims)
}

/// 2 中心库仑积分 `(P|Q)`（辅助基上的 `V` 矩阵）。
///
/// # Errors
///
/// 同 [`three_center`]。
pub fn two_center_coulomb(
    aux: &qwb_basis::InternalBasis,
    _molecule: &qwb_engine_api::Molecule,
) -> Result<Matrix, EngineError> {
    let shells: Vec<ShellBlock> = aux
        .shells()
        .iter()
        .map(shell_block)
        .collect::<Result<_, _>>()?;
    let n = aux.n_functions();
    let mut values = Matrix::zeros(n, n);
    for (index, c) in shells.iter().enumerate() {
        for d in shells.iter().skip(index) {
            let mut data = cartesian_block_two(c, d);
            let mut dims = [c.components.len(), d.components.len()];
            for (axis, shell) in [c, d].iter().enumerate() {
                let (next, next_dims) = contract2(&data, dims, axis, &shell.transform);
                data = next;
                dims = next_dims;
            }
            let mut index = 0;
            for k in 0..dims[0] {
                for l in 0..dims[1] {
                    let value = data[index];
                    index += 1;
                    let row = c.first_function + k;
                    let column = d.first_function + l;
                    values[(row, column)] = value;
                    values[(column, row)] = value;
                }
            }
        }
    }
    Ok(values)
}

/// 2 中心 `(P|Q)` 的直角坐标块：两侧都塌成 s 原函数。
fn cartesian_block_two(c: &ShellBlock, d: &ShellBlock) -> Vec<f64> {
    let dims = [c.components.len(), d.components.len()];
    let mut block = vec![0.0; dims[0] * dims[1]];
    for (mc, &pc) in c.components.iter().enumerate() {
        for (md, &pd) in d.components.iter().enumerate() {
            let mut sum = 0.0;
            for (k, &gamma) in c.exponents.iter().enumerate() {
                let weight_c = c.coefficients[k] * component_norm(pc, gamma);
                if weight_c == 0.0 {
                    continue;
                }
                for (l, &delta) in d.exponents.iter().enumerate() {
                    let weight_d = d.coefficients[l] * component_norm(pd, delta);
                    if weight_d == 0.0 {
                        continue;
                    }
                    sum += weight_c
                        * weight_d
                        * primitive_component(
                            pc,
                            c.center,
                            gamma,
                            [0, 0, 0],
                            c.center,
                            0.0,
                            pd,
                            d.center,
                            delta,
                            [0, 0, 0],
                            d.center,
                            0.0,
                        );
                }
            }
            block[mc * dims[1] + md] = sum;
        }
    }
    block
}

/// 2 个指标的球谐变换。
fn contract2(
    data: &[f64],
    dims: [usize; 2],
    axis: usize,
    transform: &qwb_basis::spherical::Transform,
) -> (Vec<f64>, [usize; 2]) {
    let n_spherical = transform.n_spherical;
    let mut out_dims = dims;
    out_dims[axis] = n_spherical;
    let in_stride = [dims[1], 1];
    let out_stride = [out_dims[1], 1];
    let total: usize = dims.iter().product();
    let mut out = vec![0.0; out_dims.iter().product()];
    for (flat, &value) in data.iter().enumerate().take(total) {
        if value == 0.0 {
            continue;
        }
        let mut index = [0usize; 2];
        for k in 0..2 {
            index[k] = (flat / in_stride[k]) % dims[k];
        }
        let mut out_index = index;
        out_index[axis] = 0;
        let out_base: usize = (0..2).map(|k| out_index[k] * out_stride[k]).sum();
        for spherical_index in 0..n_spherical {
            let weight = transform.get(index[axis], spherical_index);
            if weight != 0.0 {
                out[out_base + spherical_index * out_stride[axis]] += weight * value;
            }
        }
    }
    (out, out_dims)
}

/// Jacobi 对称特征分解：返回 `(升序特征值, 特征向量按列)`，都是行优先 `n×n`。
///
/// 用 Jacobi 而不是 `qwb-linalg` 里已有的分解：RI 辅助基经常近线性相关，
/// Jacobi 在**半正定/近奇异**上比 Cholesky 类做法稳，而且不引入新的依赖。
/// 收敛判据是「非对角元的平方和 < 1e-30」，最多 100 轮（n ≤ 三位数时典型 6–10 轮）。
fn jacobi(matrix: &Matrix) -> (Vec<f64>, Vec<f64>) {
    let n = matrix.nrows();
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = matrix[(i, j)];
        }
    }
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1.0e-30 {
            break;
        }
        // 循环 Jacobi：每轮扫一遍所有非对角元。
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1.0e-300 {
                    continue;
                }
                let theta = 0.5 * (a[q * n + q] - a[p * n + p]) / apq;
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// 对称矩阵的特征值（升序）。
///
/// 主要给测试与 `inverse_sqrt` 共用：`(P|Q)` 的特征值在"壳层内部任意正交变换"
/// 下不变，所以它可以当**外部参考的对照量**（见
/// `tests/in01_ri_integrals.rs` 与 `tests/reference/ri_invariants.csv`）。
///
/// # Errors
///
/// 非方阵 → [`EngineError::Internal`]。
pub fn symmetric_eigenvalues(matrix: &Matrix) -> Result<Vec<f64>, EngineError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(EngineError::Internal(format!(
            "symmetric_eigenvalues: 需要方阵，得到 {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    let (mut values, _) = jacobi(matrix);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(values)
}

/// 对称矩阵的 `V^{-1/2}`（RI 的"白化"变换）。
///
/// `V = U Λ Uᵀ ⇒ V^{-1/2} = U Λ^{-1/2} Uᵀ`。
///
/// # Errors
///
/// 维度不是方阵、出现非有限值、或最大特征值非正（`V` 不正定）→
/// [`EngineError::Internal`]。特征值小于 `tolerance·λ_max` 的方向被视为
/// 线性相关而**丢弃**（返回矩阵在这些方向上是 0），这与「fitting basis 里
/// 线性相关函数不贡献」的标准处理一致。
pub fn inverse_sqrt(matrix: &Matrix, tolerance: f64) -> Result<Matrix, EngineError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(EngineError::Internal(format!(
            "inverse_sqrt: 需要方阵，得到 {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    let (mut eigenvalues, v) = jacobi(matrix);
    let max = eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() || max <= 0.0 {
        return Err(EngineError::Internal(
            "inverse_sqrt: (P|Q) 不是正定矩阵（最大特征值非正）".into(),
        ));
    }
    let threshold = max * tolerance;
    for value in &mut eigenvalues {
        if *value < threshold {
            *value = 0.0;
        }
    }
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                if eigenvalues[k] == 0.0 {
                    continue;
                }
                sum += v[i * n + k] * eigenvalues[k].sqrt().recip() * v[j * n + k];
            }
            if !sum.is_finite() {
                return Err(EngineError::Internal(
                    "inverse_sqrt: 结果出现非有限值".into(),
                ));
            }
            out[(i, j)] = sum;
        }
    }
    Ok(out)
}

/// 从注册表加载辅助基（`DA-07` 的三个 RI 基都在 `qwb-data/basis/`）。
///
/// # Errors
///
/// 基组不存在 → [`EngineError::Data`]；元素不覆盖分子 → [`EngineError::Data`]。
pub fn load_aux_basis(
    registry: &Registry,
    name: &str,
    molecule: &qwb_engine_api::Molecule,
) -> Result<qwb_basis::InternalBasis, EngineError> {
    let basis = registry
        .get_basis(name)
        .map_err(|err| EngineError::Data(format!("辅助基 `{name}`：{err}")))?;
    qwb_basis::InternalBasis::build(basis, molecule)
}
