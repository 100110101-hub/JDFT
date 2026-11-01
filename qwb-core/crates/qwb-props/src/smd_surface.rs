//! SV-03 的第二半：**多原子** CDS 表面分解（每原子暴露面积）。
//!
//! 单原子极限（`A = 4π(R+0.4)²`）已经在 [`crate::smd`] 里对上游验证过；多原子的面积
//! 必须**从头算**，因为相互遮蔽会让"把每个原子的球面积相加"重复计数。
//!
//! # 切换函数
//!
//! 球 `i` 上的一个采样点 `p` 相对球 `j` 的**暴露比例**取
//!
//! ```text
//! θ_ij(p) = S(u),  u = clamp((|p − c_j| − (R_j − w)) / w, 0, 1)
//! S(u)    = u²(3 − 2u)                      （C¹ 平滑阶跃）
//! ```
//!
//! 于是 `w → 0` 时 `θ` 退化成"点在球 `j` 内 → 0，否则 1"，正是 [`crate::cavity::cavity`]
//! 用的二值遮蔽判定。`w > 0` 时**只有球面内侧那层** `R_j − w < d < R_j` 被按比例计入，
//! 球外的点仍然整份计入。
//!
//! 带宽选在球面内侧（而不是以 `R_j` 为中心的对称带）是为了让 `θ` 对 `w`
//! **逐点单调不减**：这样面积随带宽单调、且永远夹在二值结果与"互不遮蔽"的和之间；
//! 对称带会让球外一点点的地方随 `w` 变大反而被切掉，面积不再单调（第一版就是这样，
//! 被测试抓到）。面积
//!
//! ```text
//! A_i = (4πR_i²/N) · Σ_k Π_{j≠i} θ_ij(p_k)
//! ```
//!
//! # 为什么这层的判据是"自洽性"而不是对上游的死差
//!
//! 上游 `mnsol.F` 的 `SMXCDS` 用它自己的点集与切换宽度，而且是 Fortran 里逐对累加的
//! 近似；两边"点集 + 切换半径"不同就没有逐位可比性。所以这层的验收用四条**与实现
//! 无关**的性质：
//!
//! 1. 单原子极限等于闭式 `4π(R+0.4)²`（与已验证的 σ 层对接）；
//! 2. `w → 0` 复现二值遮蔽（与 SV-01 已验收的空腔几何对接）；
//! 3. 二体极限收敛到**解析**球帽面积（`analytic_area_volume`，对双原子精确）；
//! 4. 随距离/带宽单调、分离时严格可加；转动不变性只到 **采样误差**（点集钉在全局
//!    坐标系里，这是所有点采样表面的共同缺陷，`O(1/√N)`，见测试的实测表）。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::smd_atomic::SmdAtomic;

use crate::cavity::fibonacci_point;

/// 表面分解的结果。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SmdSurface {
    /// 每个原子的暴露面积，单位 Å²，顺序与 `molecule.atoms` 一致。
    pub areas_angstrom2: Vec<f64>,
    /// 总面积，单位 Å²（`CSSIGM` 项乘的就是它）。
    pub total_area_angstrom2: f64,
}

impl SmdSurface {
    /// 由每原子面积构造（总和在这里一次算好，避免两处求和顺序不同）。
    #[must_use]
    pub fn new(areas_angstrom2: Vec<f64>) -> Self {
        let total_area_angstrom2 = areas_angstrom2.iter().sum();
        Self {
            areas_angstrom2,
            total_area_angstrom2,
        }
    }
}

/// 每元素的 **CDS 有效半径**：`vdW + area_offset`（Å）。
///
/// 注意这**不是** `water.yaml` 里的 Bondi 半径，也不是 SMD 的第二套参数化半径
/// （C 1.85 / N 1.89 / O 走 Abraham α）—— 三套家族不能混用。
///
/// # Errors
///
/// 元素不在 `smd-atomic` 数据范围内。
pub fn cds_effective_radii(data: &SmdAtomic, molecule: &Molecule) -> Result<Vec<f64>, EngineError> {
    molecule
        .atoms
        .iter()
        .map(|atom| {
            data.cds
                .vdw_radii_angstrom
                .get(&atom.symbol)
                .map(|radius| radius + data.cds.area_offset_angstrom)
                .ok_or_else(|| {
                    EngineError::Data(format!(
                        "smd-surface: no CDS vdW radius for element `{}`",
                        atom.symbol
                    ))
                })
        })
        .collect()
}

/// 平滑阶跃 `S(u) = u²(3−2u)`，并把 `w = 0` 特判成二值判定。
///
/// `distance` 是采样点到球 `j` 球心的距离，`radius` 是球 `j` 的有效半径，
/// `width` 是切换带宽度 `w`（Å）。带宽落在球面**内侧** `[radius − width, radius]`。
#[must_use]
fn switch_fraction(distance: f64, radius: f64, width: f64) -> f64 {
    if width <= 0.0 {
        return if distance < radius { 0.0 } else { 1.0 };
    }
    let inner = radius - width;
    let u = ((distance - inner) / width).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// 多原子 CDS 表面分解。
///
/// `points_per_atom` 是每球采样点数（沿用 [`crate::cavity::cavity`] 的下限 12），
/// `switch_width_angstrom` 是切换带宽 `w`（Å）；传 `0.0` 得到二值遮蔽极限。
///
/// # 已知精度
///
/// Fibonacci 点集不是 Lebedev 求积，二值极限下双球暴露面积的误差是 `O(1/√N)`
/// 的随机量级（见测试里实测的收敛表），不是逐位可复现的。要更紧的精度需要换
/// Lebedev 点集，那是 SV-01 第二半的前置，不影响这里的模型层。
///
/// # Errors
///
/// 点数太少、切换带宽为负或非有限、元素不在数据范围内 → 显式错误（不静默退化）。
pub fn smd_surface(
    data: &SmdAtomic,
    molecule: &Molecule,
    points_per_atom: usize,
    switch_width_angstrom: f64,
) -> Result<SmdSurface, EngineError> {
    if points_per_atom < 12 {
        return Err(EngineError::InvalidInput(format!(
            "smd-surface: {points_per_atom} points per atom is too few (need >= 12)"
        )));
    }
    if !switch_width_angstrom.is_finite() || switch_width_angstrom < 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "smd-surface: switch width must be finite and non-negative, got {switch_width_angstrom}"
        )));
    }
    let radii = cds_effective_radii(data, molecule)?;
    let n_atoms = molecule.n_atoms();
    let mut areas = vec![0.0_f64; n_atoms];
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let radius = radii[index];
        let area_per_point = 4.0 * std::f64::consts::PI * radius * radius / points_per_atom as f64;
        let center = atom.position_angstrom;
        for k in 0..points_per_atom {
            let normal = fibonacci_point(k, points_per_atom);
            let point = [
                center[0] + radius * normal[0],
                center[1] + radius * normal[1],
                center[2] + radius * normal[2],
            ];
            let mut exposed = 1.0_f64;
            for (other_index, other) in molecule.atoms.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let dx = point[0] - other.position_angstrom[0];
                let dy = point[1] - other.position_angstrom[1];
                let dz = point[2] - other.position_angstrom[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                exposed *= switch_fraction(distance, radii[other_index], switch_width_angstrom);
                if exposed == 0.0 {
                    break;
                }
            }
            areas[index] += area_per_point * exposed;
        }
    }
    Ok(SmdSurface::new(areas))
}
