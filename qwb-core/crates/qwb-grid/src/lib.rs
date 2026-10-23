//! X-03：XC 网格 —— 角向规则、径向规则、Becke 分区。
//!
//! # 角向规则由**数据**决定
//!
//! `grid/*.yaml` 的 `angular_rule` 字段（K1 v1.2 增量）选规则：
//!
//! - `lebedev`：需要该阶数的 Lebedev 点集表。仓库里**还没有**表（见
//!   `docs/choices.md` §9），所以这个分支显式返回 [`GridError::LebedevUnavailable`]，
//!   不会悄悄换成别的规则。
//! - `gauss-product`：`Gauss–Legendre(cosθ) × 均匀 φ`。`lebedev_order` 当作
//!   **角向点数预算**：`n_θ = n_φ = ⌈√N⌉`。`Σw = 4π` 精确成立（GL 权重和为 2，
//!   φ 方向均匀规则的总权为 2π）。
//!
//! # 径向规则
//!
//! 映射 `r = a·t²/(1−t²)`（`t ∈ [0,1)`，`a = cutoff/8`），`t` 上用 Gauss–Legendre。
//! 这样权重里已经带上 `4πr²·dr/dt` 的 Jacobian，`Σ w·f(r) ≈ ∫₀^∞ 4πr² f dr`。
//!
//! # Becke 分区
//!
//! 标准的模糊单元：`ν_ij = (|r−R_i| − |r−R_j|)/|R_i−R_j|`，三次多项式步进
//! `s(ν) = ½(3ν − ν³)` 迭代 `becke_sharpness` 推导出的次数，然后
//! `W_i = Π_{j≠i} s(ν_ij)` 再归一化 —— 因此**每点 Σ_i W_i = 1 恒成立**（测试断言）。

#![deny(unsafe_code)]
#![deny(missing_docs)]

use qwb_engine_api::Molecule;
use qwb_linalg::{Matrix, symmetric_eigen};
pub mod manager;

use qwb_registry::grid::{AngularRule, GridLevelDef};
use thiserror::Error;

/// 网格错误。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GridError {
    /// 数据要求 Lebedev，但本构建没有对应点集表。
    #[error(
        "angular_rule `lebedev` needs the point table for order {order}, which is not in this build (docs/choices.md 9); declare `angular_rule: gauss-product` in the grid data"
    )]
    LebedevUnavailable {
        /// 需要的阶数。
        order: u32,
    },
    /// 参数非法（点数太小等）。
    #[error("invalid grid level `{0}`: {1}")]
    Invalid(&'static str, String),
}

/// 单点：坐标（Bohr）与权重。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GridPoint {
    /// 直角坐标，Bohr。
    pub position_bohr: [f64; 3],
    /// 权重（已含 `4πr²` 型 Jacobian，单位 Bohr³）。
    pub weight: f64,
}

/// 一组网格点。
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Grid {
    /// 点。
    pub points: Vec<GridPoint>,
}

impl Grid {
    /// 点数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// 是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// 权重之和。
    #[must_use]
    pub fn weight_sum(&self) -> f64 {
        self.points.iter().map(|point| point.weight).sum()
    }
}

/// Gauss–Legendre 节点/权重（`[-1, 1]`），用 I-00 的对称特征分解（Golub–Welsch）。
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jacobi = Matrix::zeros(n, n);
    for i in 1..n {
        let beta = f64::from(i as u32) / ((4.0 * f64::from(i as u32).powi(2) - 1.0).sqrt());
        jacobi[(i - 1, i)] = beta;
        jacobi[(i, i - 1)] = beta;
    }
    let (values, vectors) = symmetric_eigen(&jacobi).expect("symmetric eigendecomposition");
    let nodes: Vec<f64> = (0..n).map(|i| values.get(i)).collect();
    let weights: Vec<f64> = (0..n).map(|i| 2.0 * vectors[(0, i)].powi(2)).collect();
    (nodes, weights)
}

/// 角向规则：返回单位球上的方向与权重，`Σw = 4π`。
///
/// # Errors
///
/// `angular_rule: lebedev`（本构建无点集表）或点数太小。
pub fn angular_rule(level: &GridLevelDef) -> Result<(Vec<[f64; 3]>, Vec<f64>), GridError> {
    match level.angular_rule {
        AngularRule::Lebedev => Err(GridError::LebedevUnavailable {
            order: level.lebedev_order,
        }),
        AngularRule::GaussProduct => {
            let n = (f64::from(level.lebedev_order).sqrt().ceil() as usize).max(2);
            let (nodes, weights) = gauss_legendre(n);
            let phi_step = 2.0 * std::f64::consts::PI / n as f64;
            let mut directions = Vec::with_capacity(n * n);
            let mut angular_weights = Vec::with_capacity(n * n);
            for (node, weight) in nodes.iter().zip(&weights) {
                // x = cosθ
                let sin_theta = (1.0 - node * node).max(0.0).sqrt();
                for k in 0..n {
                    let phi = phi_step * (k as f64 + 0.5);
                    directions.push([sin_theta * phi.cos(), sin_theta * phi.sin(), *node]);
                    angular_weights.push(weight * phi_step);
                }
            }
            Ok((directions, angular_weights))
        }
        // `AngularRule` 是 non_exhaustive：不认识就报错，不猜。
        _ => Err(GridError::Invalid(
            "grid",
            "unknown angular_rule value".to_string(),
        )),
    }
}

/// 径向规则：返回 `(r, w)`，满足 `Σ w·f(r) ≈ ∫₀^∞ r² f dr`（**不含 4π**：\\n/// 角向权重之和已经是 4π，两者相乘才是完整的体积元）。
///
/// # Errors
///
/// 点数太小或 `cutoff` 非正。
pub fn radial_rule(level: &GridLevelDef) -> Result<(Vec<f64>, Vec<f64>), GridError> {
    if level.radial_points < 10 {
        return Err(GridError::Invalid(
            "radial",
            format!("radial_points = {} is too small", level.radial_points),
        ));
    }
    if !(level.cutoff.is_finite() && level.cutoff > 0.0) {
        return Err(GridError::Invalid(
            "radial",
            format!("cutoff = {} must be positive", level.cutoff),
        ));
    }
    let n = level.radial_points as usize;
    let a = level.cutoff / 8.0;
    // t ∈ [0, 1) 用 GL 映射：t = (x+1)/2
    let (nodes, weights) = gauss_legendre(n);
    let mut radii = Vec::with_capacity(n);
    let mut radial_weights = Vec::with_capacity(n);
    for (node, weight) in nodes.iter().zip(&weights) {
        // t 必须严格小于 1：否则 r = a t²/(1−t²) 发散，坐标变成 inf/NaN，
        // 而 inf×0 会污染后续所有积分（S-02 就是被这个坑住的）。
        let t = (0.5 * (node + 1.0)).min(1.0 - 1.0e-3);
        let one_minus = 1.0 - t * t;
        if one_minus <= 0.0 {
            continue;
        }
        let r = a * t * t / one_minus;
        // dr/dt = a·2t/(1−t²)²；Jacobian 4πr²·dr/dt·dt/dx，dt/dx = 1/2
        let dr_dt = a * 2.0 * t / (one_minus * one_minus);
        // **不要在这里乘 4π**：角向权重之和已经是 4π（∫dΩ），再乘一次就是 4π²。
        // 这里只给 ∫ r² f dr 的径向权重。
        let jacobian = r * r * dr_dt * 0.5;
        radii.push(r);
        radial_weights.push(weight * jacobian);
    }
    Ok((radii, radial_weights))
}

/// 三次步进函数 `s(ν) = ½(3ν − ν³)`，迭代 `k` 次（Becke 的模糊单元）。
fn smoothed_step(mut nu: f64, iterations: u32) -> f64 {
    for _ in 0..iterations {
        nu = 1.5 * nu - 0.5 * nu * nu * nu;
    }
    0.5 * (1.0 + nu)
}

/// 单中心网格（不分区）：`Σ w = Σ_radial w · 4π`。
///
/// # Errors
///
/// 见 [`angular_rule`] / [`radial_rule`]。
pub fn atomic_grid(level: &GridLevelDef, center_bohr: [f64; 3]) -> Result<Grid, GridError> {
    let (directions, angular_weights) = angular_rule(level)?;
    let (radii, radial_weights) = radial_rule(level)?;
    let mut points = Vec::with_capacity(directions.len() * radii.len());
    for (radius, radial_weight) in radii.iter().zip(&radial_weights) {
        for (direction, angular_weight) in directions.iter().zip(&angular_weights) {
            points.push(GridPoint {
                position_bohr: [
                    center_bohr[0] + radius * direction[0],
                    center_bohr[1] + radius * direction[1],
                    center_bohr[2] + radius * direction[2],
                ],
                weight: radial_weight * angular_weight,
            });
        }
    }
    Ok(Grid { points })
}

/// 分子网格：各原子网格取并集，权重用 Becke 分区（**每点 `Σ_i W_i = 1`**）。
///
/// 每个原子的模糊单元权重是 `W_i = Π_{j≠i} s(ν_ij)`，其中
/// `ν_ij = (d_i − d_j)/R_ij`（`d_i` 是点到核心 i 的距离，`R_ij` 是两核间距）。
/// 先取对数做累加再归一化，避免连乘下溢。
///
/// # Errors
///
/// 见 [`atomic_grid`]。
pub fn molecular_grid(level: &GridLevelDef, molecule: &Molecule) -> Result<Grid, GridError> {
    let centers: Vec<[f64; 3]> = molecule.atoms.iter().map(|a| a.position_bohr()).collect();
    if centers.is_empty() {
        return Ok(Grid::default());
    }
    // 锐度 → 迭代次数：becke_sharpness 越大越"硬"。
    let iterations = ((level.becke_sharpness * 5.0).round().clamp(1.0, 8.0)) as u32;

    let distance = |point: [f64; 3], center: [f64; 3]| -> f64 {
        (0..3)
            .map(|axis| (point[axis] - center[axis]).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut points = Vec::new();
    for (atom_index, center) in centers.iter().enumerate() {
        let atomic = atomic_grid(level, *center)?;
        for point in atomic.points {
            // 每个原子各自的 ln W_i = Σ_{j≠i} ln s(ν_ij)
            let mut log_weights = vec![0.0_f64; centers.len()];
            let mut distance_to = vec![0.0_f64; centers.len()];
            for (index, other) in centers.iter().enumerate() {
                distance_to[index] = distance(point.position_bohr, *other);
            }
            for i in 0..centers.len() {
                for j in 0..centers.len() {
                    if i == j {
                        continue;
                    }
                    let separation = distance(centers[i], centers[j]);
                    if separation == 0.0 {
                        continue;
                    }
                    let nu = (distance_to[i] - distance_to[j]) / separation;
                    log_weights[i] += smoothed_step(nu, iterations).max(1e-300).ln();
                }
            }
            // 归一化（先减最大值，数值稳定）
            let max = log_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = log_weights.iter().map(|log| (log - max).exp()).sum();
            let partition = if denominator > 0.0 {
                (log_weights[atom_index] - max).exp() / denominator
            } else {
                1.0
            };
            points.push(GridPoint {
                position_bohr: point.position_bohr,
                weight: point.weight * partition,
            });
        }
    }
    Ok(Grid { points })
}
