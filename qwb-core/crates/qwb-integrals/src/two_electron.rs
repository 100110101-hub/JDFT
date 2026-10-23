//! 两电子积分张量（化学家记号 `(ij|kl)`）。
//!
//! # 层次
//!
//! * **原函数层**：`hermite2e.rs` 给出任意角动量的未归一化 `(ab|cd)`；
//! * **壳层层**（本文件）：按分量归一化 + 收缩，得到直角坐标块；
//! * **球谐层**：用 `qwb_basis::spherical` 的 `T_g` 对**四个指标**各收缩一次；
//! * **张量层**：按 `(ij|kl)` 的置换对称性填 8 个等价位置。
//!
//! 对称性只用来**少算**，不用来"填出"没算过的值：每个球谐位置都由某一次真实的
//! 收缩得到，然后被写到它的 8 个等价下标上。
//!
//! # 归一化
//!
//! 每个原函数按它**自己的分量**（`x^a y^b z^c`）归一化（`xy` 型与 `x²` 型的归一
//! 化常数不同），再乘收缩系数。这一点与一电子积分（`one_electron.rs`）完全一致，
//! 否则 SCF 的能量会随基组静默漂移。

use qwb_basis::spherical::{Transform, spherical_transform};
use qwb_basis::{InternalBasis, InternalShell, cartesian_components};
use qwb_engine_api::{EngineError, Molecule};

use crate::boys::boys_functions;
use crate::hermite2e::primitive_component;
use crate::one_electron::primitive_norm;

const PI: f64 = std::f64::consts::PI;

/// 由两个 s 型（未归一化）原函数构成的高斯乘积：`p`、`P`、`E_AB`。
fn s_pair(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> (f64, [f64; 3], f64) {
    let p = alpha + beta;
    let center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let r2 = (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum::<f64>();
    (p, center, (-alpha * beta / p * r2).exp())
}

/// 四个（未归一化）s 型原函数的 `(ss|ss)`。
///
/// 闭合式：`2π^{5/2}/(p q √(p+q)) E_AB E_CD F_0(α)`。这条闭式同时是 I-03 交叉
/// 验证的基准（见 `tests/common/oracle.rs`）。
#[must_use]
pub fn primitive_ssss(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
) -> f64 {
    let (p, p_center, e_ab) = s_pair(alpha, a, beta, b);
    let (q, q_center, e_cd) = s_pair(gamma, c, delta, d);
    let pq = p + q;
    let r2 = (0..3)
        .map(|axis| (p_center[axis] - q_center[axis]).powi(2))
        .sum::<f64>();
    let alpha_boys = p * q / pq * r2;
    let f0 = boys_functions(0, alpha_boys)[0];

    2.0 * PI.powf(2.5) / (p * q * pq.sqrt()) * e_ab * e_cd * f0
}

/// 两电子积分张量（化学家记号 `(ij|kl)`，扁平 `i*n³ + j*n² + k*n + l`）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TwoElectronIntegrals {
    /// 基函数个数（球谐空间的 AO 个数）。
    pub n_functions: usize,
    /// 扁平存储的 `(ij|kl)`。
    pub values: Vec<f64>,
}

impl TwoElectronIntegrals {
    /// 取 `(i j | k l)`。
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.values[self.index(i, j, k, l)]
    }

    fn index(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        let n = self.n_functions;
        ((i * n + j) * n + k) * n + l
    }
}

/// 一个壳层装配时需要的全部信息。
struct ShellBlock {
    center: [f64; 3],
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    /// 直角坐标分量（与 `Transform` 的行一一对应）。
    components: Vec<[u32; 3]>,
    transform: Transform,
    /// 该壳层第一个球谐 AO 的全局下标。
    first_function: usize,
}

fn shell_block(shell: &InternalShell) -> Result<ShellBlock, EngineError> {
    let transform = spherical_transform(shell.l, shell.angular).ok_or_else(|| {
        EngineError::Unsupported(format!(
            "spherical transformation for l={} is not implemented (R-06; first phase covers s/p/d)",
            shell.l
        ))
    })?;
    let components: Vec<[u32; 3]> = cartesian_components(shell.l)
        .into_iter()
        .map(|(x, y, z)| [x, y, z])
        .collect();
    if components.len() != transform.n_cartesian {
        return Err(EngineError::Internal(format!(
            "shell l={}: {} cartesian components but transform expects {}",
            shell.l,
            components.len(),
            transform.n_cartesian
        )));
    }
    Ok(ShellBlock {
        center: shell.center_bohr,
        exponents: shell.exponents.clone(),
        coefficients: shell.coefficients.clone(),
        components,
        transform,
        first_function: shell.first_function,
    })
}

/// 四维扁平索引的步长。
fn strides(dims: [usize; 4]) -> [usize; 4] {
    [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1]
}

/// 一个壳层四元组的**直角坐标**块（含归一化与收缩）。
fn cartesian_block(a: &ShellBlock, b: &ShellBlock, c: &ShellBlock, d: &ShellBlock) -> Vec<f64> {
    let dims = [
        a.components.len(),
        b.components.len(),
        c.components.len(),
        d.components.len(),
    ];
    let mut block = vec![0.0; dims[0] * dims[1] * dims[2] * dims[3]];
    for (ma, &pa) in a.components.iter().enumerate() {
        for (mb, &pb) in b.components.iter().enumerate() {
            for (mc, &pc) in c.components.iter().enumerate() {
                for (md, &pd) in d.components.iter().enumerate() {
                    let mut sum = 0.0;
                    for (i, &alpha) in a.exponents.iter().enumerate() {
                        let weight_a =
                            a.coefficients[i] * primitive_norm((pa[0], pa[1], pa[2]), alpha);
                        if weight_a == 0.0 {
                            continue;
                        }
                        for (j, &beta) in b.exponents.iter().enumerate() {
                            let weight_b =
                                b.coefficients[j] * primitive_norm((pb[0], pb[1], pb[2]), beta);
                            if weight_b == 0.0 {
                                continue;
                            }
                            for (k, &gamma) in c.exponents.iter().enumerate() {
                                let weight_c = c.coefficients[k]
                                    * primitive_norm((pc[0], pc[1], pc[2]), gamma);
                                if weight_c == 0.0 {
                                    continue;
                                }
                                for (l, &delta) in d.exponents.iter().enumerate() {
                                    let weight_d = d.coefficients[l]
                                        * primitive_norm((pd[0], pd[1], pd[2]), delta);
                                    if weight_d == 0.0 {
                                        continue;
                                    }
                                    sum += weight_a
                                        * weight_b
                                        * weight_c
                                        * weight_d
                                        * primitive_component(
                                            pa, a.center, alpha, pb, b.center, beta, pc, c.center,
                                            gamma, pd, d.center, delta,
                                        );
                                }
                            }
                        }
                    }
                    let index = ((ma * dims[1] + mb) * dims[2] + mc) * dims[3] + md;
                    block[index] = sum;
                }
            }
        }
    }
    block
}

#[allow(clippy::needless_range_loop)] // flat 同时用于算 axis 下标与写回
/// 对第 `axis` 个指标做一次球谐变换（`T` 的列数即该壳层的球谐维数）。
fn contract(
    data: &[f64],
    dims: [usize; 4],
    axis: usize,
    transform: &Transform,
) -> (Vec<f64>, [usize; 4]) {
    let n_spherical = transform.n_spherical;
    let mut out_dims = dims;
    out_dims[axis] = n_spherical;
    let in_stride = strides(dims);
    let out_stride = strides(out_dims);
    let total: usize = dims.iter().product();
    let mut out = vec![0.0; out_dims.iter().product()];
    for flat in 0..total {
        let value = data[flat];
        if value == 0.0 {
            continue;
        }
        let cartesian_index = (flat / in_stride[axis]) % dims[axis];
        let base = flat - cartesian_index * in_stride[axis];
        for spherical_index in 0..n_spherical {
            let weight = transform.get(cartesian_index, spherical_index);
            if weight != 0.0 {
                out[base + spherical_index * out_stride[axis]] += weight * value;
            }
        }
    }
    (out, out_dims)
}

/// 把球谐块写到 `(ij|kl)` 的 8 个等价下标上。
fn scatter(
    values: &mut [f64],
    n: usize,
    block: &[f64],
    dims: [usize; 4],
    shells: [&ShellBlock; 4],
) {
    const PERMUTATIONS: [[usize; 4]; 8] = [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 1, 0],
    ];
    let base = [
        shells[0].first_function,
        shells[1].first_function,
        shells[2].first_function,
        shells[3].first_function,
    ];
    let mut index = 0;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                for l in 0..dims[3] {
                    let value = block[index];
                    index += 1;
                    let position = [base[0] + i, base[1] + j, base[2] + k, base[3] + l];
                    for permutation in PERMUTATIONS {
                        let p = [
                            position[permutation[0]],
                            position[permutation[1]],
                            position[permutation[2]],
                            position[permutation[3]],
                        ];
                        values[((p[0] * n + p[1]) * n + p[2]) * n + p[3]] = value;
                    }
                }
            }
        }
    }
}

/// 计算全部分子轨道的两电子积分。
///
/// # Errors
///
/// 出现 `l > 2` 的壳层（球谐变换未实现，R-06）→ [`EngineError::Unsupported`]。
pub fn two_electron(
    basis: &InternalBasis,
    _molecule: &Molecule,
) -> Result<TwoElectronIntegrals, EngineError> {
    let shells: Vec<ShellBlock> = basis
        .shells()
        .iter()
        .map(shell_block)
        .collect::<Result<_, _>>()?;
    let n = basis.n_functions();
    let mut values = vec![0.0; n * n * n * n];

    for (ia, a) in shells.iter().enumerate() {
        for (ib, b) in shells.iter().enumerate() {
            if ib < ia {
                continue; // (ab|· ·) 与 (ba|· ·) 等价
            }
            for (ic, c) in shells.iter().enumerate() {
                for (id, d) in shells.iter().enumerate() {
                    if id < ic {
                        continue;
                    }
                    // 壳层对的字典序：只算 (a,b) <= (c,d) 这一半。
                    if (ia, ib) > (ic, id) {
                        continue;
                    }
                    let block = cartesian_block(a, b, c, d);
                    let mut data = block;
                    let mut dims = [
                        a.components.len(),
                        b.components.len(),
                        c.components.len(),
                        d.components.len(),
                    ];
                    for (axis, shell) in [a, b, c, d].iter().enumerate() {
                        let (next, next_dims) = contract(&data, dims, axis, &shell.transform);
                        data = next;
                        dims = next_dims;
                    }
                    scatter(&mut values, n, &data, dims, [a, b, c, d]);
                }
            }
        }
    }

    Ok(TwoElectronIntegrals {
        n_functions: n,
        values,
    })
}
