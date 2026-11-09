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
//! 上游 `mnsol.F` 里 CDS 的表面是 `CDSCALC` → `CDS_EG` → `DAREAL` 算的（它用自己的点集与
//! 遮蔽判定，`NCROSS`/`DAREA` 一整套，我们还没复刻），**不是** `SMXCDS` —— 后者是另一件事：
//! 成键接触权重 `COT` 与每原子 σ 的组装。能量形状是 `Σ_k (STS_k + CSSIGM)·A_k·1e-3`，
//! 其中 `A_k` 来自 `DAREAL`、`STS_k` 来自 `SMXCDS`。本模块只做 `A_k`；`STS_k` 目前只用
//! 零阶 σ 表（缺 H 与成对修正，见 `choices` 一百六十五），所以水（那些修正全为 0）没问题，
//! 含 C–O / C–N / C–H 的分子会系统性偏。
//!
//! 两边"点集 + 遮蔽判定"不同，逐位可比不成立。所以这层的验收用四条**与实现无关**的性质：
//!
//! 1. 单原子极限等于闭式 `4π(R+0.4)²`（与已验证的 σ 层对接）；
//! 2. `w → 0` 复现二值遮蔽（与 SV-01 已验收的空腔几何对接）；
//! 3. 二体极限收敛到**解析**球帽面积（`analytic_area_volume`，对双原子精确）；
//! 4. 随距离/带宽单调、分离时严格可加；转动不变性见下。
//!
//! # 坐标架：默认与 `cavity()` 同一个（转动协变）
//!
//! 采样点在哪个坐标系里生成，决定了"转一下分子面积会不会变"：
//!
//! * [`CavityFrame::Covariant`]（**默认**）：点放在**每原子的局部坐标架**里
//!   （[`crate::cavity::atom_frame`]，由邻居几何构造），坐标架随分子一起转，于是面积为刚性
//!   转动精确不变（只剩浮点舍入）。这与 [`crate::cavity::cavity`] 的默认**必须是同一个**：
//!   COSMO 的空腔和 CDS 的面描述的是同一个溶剂边界，一个协变一个全局就等于把
//!   `O(1/√N)` 的转动漂移留在了半条路径上。
//! * [`CavityFrame::Global`]：SV-03 的原始做法，点钉在全局坐标轴上，面积对转动只到采样
//!   误差（`O(1/√N)`）。保留它是因为历史数值按位钉着，而且比较"两种坐标架收敛到同一个
//!   面积"需要一个参照。
//!
//! 显式要旧行为用 [`smd_surface_with_frame`] 与 [`CavityFrame::Global`]。

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::smd_atomic::SmdAtomic;

use crate::cavity::{CavityFrame, atom_frame, compose_direction, fibonacci_point};

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

/// 多原子 CDS 表面分解（[`CavityFrame::default()`] = 转动协变）。
///
/// `points_per_atom` 是每球采样点数（沿用 [`crate::cavity::cavity`] 的下限 12），
/// `switch_width_angstrom` 是切换带宽 `w`（Å）；传 `0.0` 得到二值遮蔽极限。
///
/// 这个函数**逐位**等于 [`smd_surface_with_frame`] 传 [`CavityFrame::Covariant`]。
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
    smd_surface_with_frame(
        data,
        molecule,
        points_per_atom,
        switch_width_angstrom,
        CavityFrame::default(),
    )
}

/// 多原子 CDS 表面分解，**可选坐标架**（与 [`crate::cavity::cavity_with_frame`] 同形状）。
///
/// 除坐标架外与 [`smd_surface`] 完全相同：同一批 Fibonacci 点、同一个面积权重
/// `4πR²/N`、同一个切换函数、同一个每原子归属。
///
/// * [`CavityFrame::Covariant`]：点在每原子的局部坐标架（[`atom_frame`]）里生成，
///   坐标架随分子刚性转动而转动，所以面积为刚性转动精确不变。
/// * [`CavityFrame::Global`]：点在全局坐标轴下生成（SV-03 原始行为，逐位不变，
///   但转动漂移 `O(1/√N)`）。
///
/// # Errors
///
/// 同 [`smd_surface`]。
pub fn smd_surface_with_frame(
    data: &SmdAtomic,
    molecule: &Molecule,
    points_per_atom: usize,
    switch_width_angstrom: f64,
    frame: CavityFrame,
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
    // 协变模式下先把每原子的坐标架算好（只依赖几何，一次 O(n²)）；全局模式保持 `None`，
    // 走的是与原始实现**逐位相同**的那条分支。
    let frames: Option<Vec<[[f64; 3]; 3]>> = match frame {
        CavityFrame::Global => None,
        CavityFrame::Covariant => Some(
            (0..n_atoms)
                .map(|index| atom_frame(molecule, index))
                .collect(),
        ),
    };
    let mut areas = vec![0.0_f64; n_atoms];
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let radius = radii[index];
        let area_per_point = 4.0 * std::f64::consts::PI * radius * radius / points_per_atom as f64;
        let center = atom.position_angstrom;
        for k in 0..points_per_atom {
            let local = fibonacci_point(k, points_per_atom);
            let normal = match &frames {
                Some(frames) => compose_direction(&frames[index], local),
                None => local,
            };
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
