//! SV-01 的第一半：**空腔几何**（球面镶嵌 + 暴露面积/体积）。
//!
//! # 为什么先做几何
//!
//! SV-01 的判据里有一条与介电常数无关、完全自洽的：「表面面积/体积对**转动**不变
//! （< 1e-6）」。这条能先把"镶嵌 + 遮蔽判定 + 面积权重"三件事钉死，而不依赖还没转录的
//! SMD 半径表（用 `qwb-data/solvent/water.yaml` 里的 Bondi 半径就够）。COSMO 的表面
//! 电荷迭代（SV-01 的另一半）要在这套 tessera 上做，所以几何错了后面全错。
//!
//! # 镶嵌方案与其局限
//!
//! 用 **Fibonacci 球**（等面积近似均匀取点），每个 tessera 的权重取 `4πR²/N`。
//! 这不是 Lebedev 求积：它不做球面精确多项式积分，所以"单个原子球的面积恰好 4πR²"
//! 是**按构造**成立的，而"两个球相交后的暴露面积"只有 `O(1/N)` 精度。这一点在
//! [`cavity`] 的文档里写清，也写在账本里 —— 换成 Lebedev（`qwb-grid` 里已有球面点）
//! 是 SV-01 第二半的前置。
//!
//! # 遮蔽判定与转动不变性
//!
//! 点在球 `i` 的局部坐标系里生成，再放到空间里；若该点落在**另一个**球内部，就判为被
//! 遮蔽并丢掉。这条性质正好能抓住"遮蔽判定用了错的坐标系"这类错误（例如漏做旋转、
//! 或用原子间距离却比较了错的半径）。
//!
//! 但**"局部坐标系"是相对什么**决定了转动不变性：
//!
//! * [`CavityFrame::Global`]（默认、SV-01 原始行为）：点钉在**全局**坐标轴上。分子整体
//!   转动时点集不跟着转，遮蔽边界上的碎片会翻转，面积/体积漂移 `O(1/√N)`（水 + Bondi
//!   半径、250 点/原子实测 4.76e-3 / 3.56e-3）。这一模式**保持逐位不变**（已有钉住的数值
//!   不许动），本模块只把它命名清楚。
//! * [`CavityFrame::Covariant`]：点钉在**每原子的局部坐标架**上（[`atom_frame`]，由邻居
//!   几何构造），坐标架随分子一起转动，于是同一批点转过同一个旋转、遮蔽判定只看距离 ——
//!   面积与体积对刚性转动精确不变（只有浮点舍入），实测见 `tests/sv04_covariant_cavity.rs`。
//!
//! 退化情形（孤立原子、双原子/直线分子）在 [`atom_frame`] 里逐条写明：那些几何不存在
//! 转动协变的坐标架（绕轴转动是对称操作），所以显式退到确定的全局兜底。

use std::collections::BTreeMap;

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::elements::atomic_number;

/// 镶嵌方向所用的**坐标架**。
///
/// [`CavityFrame::Global`] 是 SV-01 的原始做法：每个原子的 Fibonacci 点在全局坐标轴下生成。
/// 分子整体转动时这些点**不**跟着转，于是"哪些点被别的球遮住"会变，面积/体积对刚性转动
/// 有 `O(1/√N)` 的漂移（水 + Bondi 半径、250 点/原子实测 4.76e-3 / 3.56e-3）—— 与
/// SV-01 的判据（< 1e-6）不符，是 SV-02 的解析梯度上不去的前置缺陷。
///
/// [`CavityFrame::Covariant`] 在**每原子的局部坐标架**里生成同一批点，坐标架由邻居几何
/// 构造（见 [`atom_frame`]）因而随分子一起转动：同一批点转过同一个旋转，遮蔽判定只看距离，
/// 所以面积与体积对刚性转动精确不变（只剩浮点舍入，实测见
/// `tests/sv04_covariant_cavity.rs`）。这一模式**不是**默认值，默认路径逐位不变。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CavityFrame {
    /// 全局坐标轴（SV-01 原始行为，默认；数值逐位不变）。
    #[default]
    Global,
    /// 每原子由邻居几何构造的局部坐标架（刚性转动协变）。
    Covariant,
}

/// 一个表面碎片（tessera）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Tessera {
    /// 中心，单位 Å。
    pub center_angstrom: [f64; 3],
    /// 外法向（单位向量）。
    pub normal: [f64; 3],
    /// 面积，单位 Å²。
    pub area_angstrom2: f64,
    /// 所属原子序号。
    pub atom_index: usize,
}

impl Tessera {
    /// 构造一个表面碎片。
    ///
    /// 面积由调用方给出（解析 caps 的精确值），这里不做几何自洽检查 ——
    /// 面积与中心不自洽会让 COSMO 的方程失去物理意义，所以生产代码一律走
    /// [`cavity`] / [`analytic_area_volume`]。
    #[must_use]
    pub fn new(
        center_angstrom: [f64; 3],
        normal: [f64; 3],
        area_angstrom2: f64,
        atom_index: usize,
    ) -> Self {
        Self {
            center_angstrom,
            normal,
            area_angstrom2,
            atom_index,
        }
    }
}

/// 空腔几何。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Cavity {
    /// 表面碎片（只含**未被遮蔽**的那些）。
    pub tesserae: Vec<Tessera>,
    /// 表面积，单位 Å²。
    pub surface_area_angstrom2: f64,
    /// 体积，单位 Å³（对每个碎片用 `R·A/3` 求和，即散度定理）。
    pub volume_angstrom3: f64,
}

impl Cavity {
    /// 校验：面积/体积为正且有限、法向为单位向量。
    ///
    /// # Errors
    ///
    /// 出现非有限值、面积/体积非正、或法向未归一 → [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.surface_area_angstrom2.is_finite() || self.surface_area_angstrom2 <= 0.0 {
            return Err(EngineError::Internal(format!(
                "cavity: surface area {} is not positive and finite",
                self.surface_area_angstrom2
            )));
        }
        if !self.volume_angstrom3.is_finite() || self.volume_angstrom3 <= 0.0 {
            return Err(EngineError::Internal(format!(
                "cavity: volume {} is not positive and finite",
                self.volume_angstrom3
            )));
        }
        for (index, tessera) in self.tesserae.iter().enumerate() {
            let norm2 = tessera
                .normal
                .iter()
                .map(|component| component * component)
                .sum::<f64>();
            if (norm2 - 1.0).abs() > 1.0e-10 {
                return Err(EngineError::Internal(format!(
                    "cavity: tessera {index} normal is not a unit vector (|n|² = {norm2})"
                )));
            }
        }
        Ok(())
    }
}

/// 第 `k` 个 Fibonacci 球面点（`n` 个点，单位球）。
pub(crate) fn fibonacci_point(k: usize, n: usize) -> [f64; 3] {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let z = 1.0 - (2.0 * k as f64 + 1.0) / n as f64;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * std::f64::consts::PI * (k as f64 / golden).rem_euclid(1.0);
    [radius * phi.cos(), radius * phi.sin(), z]
}

/// 共线判定的阈值：两方向夹角的正弦小于它就算共线。
///
/// 阈值不能取到舍入级别：`e2` 是由"去掉 `e1` 分量后归一化"得到的，它的相对误差约
/// `ε/sin θ`（`ε ≈ 1e-16`）。`sin θ ≥ 1e-3` 时坐标架的误差 ≤ `1e-13`，转动不变性稳在
/// 这个量级；若把阈值压到 `1e-6`，接近共线的体系会得到 `1e-10` 量级的噪声坐标架。
const NON_COLLINEAR_SINE: f64 = 1.0e-3;

/// 距离并列的**相对**容差：两个候选的距离相差不到 `1e-9` 就当作并列、按原子序号取小者。
///
/// 这一条是转动不变性的关键：对称分子里"数学上精确相等"的距离，在用转动后的坐标算出来时
/// 会差 `~1e-16`；若直接比较这两个数，分支会在转动下翻转（实测甲醇的 H2/H3 就是这样把
/// 面积抖动到 4.9e-3）。把并列判定的窗口放大到 `1e-9`（远大于 `1e-16`、又远小于任何真实
/// 的化学距离差）之后，分支在转动下就是稳定的。
const DISTANCE_TIE_RELATIVE: f64 = 1.0e-9;

/// 三个三维向量的基础运算（只在本模块用，避免为一个坐标架拖进线性代数 crate）。
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// 单位化；长度非正/非有限时返回 `None`（调用方负责退化分支）。
fn unit3(a: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm3(a);
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    Some([a[0] / length, a[1] / length, a[2] / length])
}

/// 给定一根单位轴 `axis`，返回一根**确定但不协变**的垂直单位向量。
///
/// 取绝对分量最小的全局坐标轴做参考向量。这条分支只在"其余所有原子都与 `axis` 共线"
/// 时用到（双原子、直线分子、孤立原子之外的 0/1 方向体系）—— 那种几何**不存在**
/// 转动协变的垂直向量（绕轴转动是对称操作，任何协变构造都只能给出与轴平行的向量，
/// 矛盾），所以这里是显式的兜底而不是疏漏。
fn fallback_perpendicular(axis: [f64; 3]) -> [f64; 3] {
    let mut reference = [0.0_f64; 3];
    let components = [axis[0].abs(), axis[1].abs(), axis[2].abs()];
    let mut smallest = 0;
    for candidate in 1..3 {
        if components[candidate] < components[smallest] {
            smallest = candidate;
        }
    }
    reference[smallest] = 1.0;
    let projection = dot3(reference, axis);
    // 参考轴与 `axis` 的夹角 ≤ 54.7°（最小分量 ≤ 1/√3），去掉轴分量后长度 ≥ 0.816，非零。
    let orthogonal = [
        reference[0] - projection * axis[0],
        reference[1] - projection * axis[1],
        reference[2] - projection * axis[2],
    ];
    // 上面的界保证 `unit3` 不会返回 None；真要出现非有限值就退回全局 z 的垂直（+x）。
    unit3(orthogonal).unwrap_or([1.0, 0.0, 0.0])
}

/// 按（距离，序号）排序候选方向，**并列判定带容差**。
///
/// 实现成一串贪心抽取而不是一次 `sort_by`：`sort_by` 的比较器要求传递性，而"相差不到
/// `DISTANCE_TIE_RELATIVE` 就算并列"不满足传递性；贪心抽取每次只问"谁的序号最小且距离
/// 在最小值的容差窗内"，结果唯一、确定，且在 `~1e-16` 的转动噪声下稳定。
fn order_candidates(mut candidates: Vec<(f64, usize, [f64; 3])>) -> Vec<(f64, usize, [f64; 3])> {
    // 先按原子序号排好（`remove` 保持顺序），于是"窗内序号最小"= 窗内第一个。
    candidates.sort_by_key(|candidate| candidate.1);
    let mut ordered = Vec::with_capacity(candidates.len());
    while !candidates.is_empty() {
        let nearest = candidates
            .iter()
            .map(|candidate| candidate.0)
            .fold(f64::INFINITY, f64::min);
        let window = nearest * (1.0 + DISTANCE_TIE_RELATIVE);
        // `position` 不会失败：`nearest` 自己就在窗内。
        let picked = candidates
            .iter()
            .position(|candidate| candidate.0 <= window)
            .unwrap_or(0);
        ordered.push(candidates.remove(picked));
    }
    ordered
}

/// 原子 `index` 的**局部坐标架**（三行 `e1`/`e2`/`e3`，单位正交、右手）。
///
/// # 定义
///
/// 1. `e1`：指向**最近**邻居的方向。距离并列（相对差 < `1e-9`）时按原子序号取小者，
///    保证确定性与转动稳定性 —— 见本文件里的 `DISTANCE_TIE_RELATIVE` 常量及其说明。
/// 2. `e2`：在其余原子中按（距离，序号）顺序找到**第一个与 `e1` 不共线**的方向
///    （`sin θ ≥ 1e-3`），去掉 `e1` 分量后归一化。
/// 3. `e3 = e1 × e2`，再把 `e2` 重算成 `e3 × e1`，让三根轴严格正交（Gram–Schmidt）。
///
/// 取"最近邻居 + 第一根不共线方向"而不是惯量主轴，是为了让坐标架对**任意配位数**
/// 都良定义：惯量主轴在球形陀螺（甲烷）上三重简并，需要额外的确定性破缺规则；
/// 而上面两条规则对所有非共线环境都直接给出唯一答案。
///
/// # 协变性
///
/// 输入只用到相对位置向量，且所有分支判定都带容差（远大于 `~1e-16` 的转动噪声），
/// 所以对刚性旋转 `Q`：`frame(Q·r) = Q·frame(r)`（逐轴）—— 这就是
/// [`CavityFrame::Covariant`] 精确转动不变的全部来源。
///
/// # 退化情形（显式、确定）
///
/// * **0 个邻居**（孤立原子、或与所有其它原子重合）：返回全局单位架。孤立原子没有任何
///   取向信息，它的面积仍然精确是 `4πR²`（所有碎片都暴露），但体积的散度定理式
///   `(1/3)Σ(r·n)A` 带一个**预先存在**的原点依赖，因此孤立原子**不可能**做到转动不变 ——
///   这是 `cavity()` 体积公式本身的性质，不在坐标架这一层能修。
/// * **其余原子都与 `e1` 共线**（双原子、直线分子）：退到本文件里 `fallback_perpendicular`
///   的确定性兜底（取绝对分量最小的全局轴做参考），确定但**不**协变，同样是因为那种几何
///   不存在协变的垂直向量。实测（`tests/sv04_covariant_cavity.rs`）：这种体系的面积/体积
///   在三次刚性转动下**仍然逐位不变** —— 共线体系的遮蔽判据只依赖 `n·e1`，而 `n·e1` 是
///   本地坐标里的固定量（`fibonacci_point` 的第一个分量），换一根垂直参考向量只是绕对称轴
///   转，改不了"哪些点被遮住"。也就是说兜底只影响坐标架，不影响观测量。
#[must_use]
pub fn atom_frame(molecule: &Molecule, index: usize) -> [[f64; 3]; 3] {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let Some(origin) = molecule.atoms.get(index).map(|atom| atom.position_angstrom) else {
        return identity;
    };
    // 候选方向只依赖相对几何，所以随分子一起转动。
    let mut candidates: Vec<(f64, usize, [f64; 3])> = Vec::new();
    for (other_index, other) in molecule.atoms.iter().enumerate() {
        if other_index == index {
            continue;
        }
        let difference = sub3(other.position_angstrom, origin);
        let distance = norm3(difference);
        if !distance.is_finite() || distance <= 0.0 {
            continue;
        }
        candidates.push((
            distance,
            other_index,
            [
                difference[0] / distance,
                difference[1] / distance,
                difference[2] / distance,
            ],
        ));
    }
    let candidates = order_candidates(candidates);
    let Some(&(_, _, e1)) = candidates.first() else {
        return identity;
    };
    let mut secondary: Option<[f64; 3]> = None;
    for &(_, _, direction) in &candidates {
        let projection = dot3(direction, e1);
        let orthogonal = [
            direction[0] - projection * e1[0],
            direction[1] - projection * e1[1],
            direction[2] - projection * e1[2],
        ];
        if norm3(orthogonal) > NON_COLLINEAR_SINE
            && let Some(unit) = unit3(orthogonal)
        {
            secondary = Some(unit);
            break;
        }
    }
    let e2 = secondary.unwrap_or_else(|| fallback_perpendicular(e1));
    let Some(e3) = unit3(cross3(e1, e2)) else {
        // `e1 ⟂ e2` 且都单位化，叉积长度应为 1；走到这里说明输入是非有限值。
        return identity;
    };
    let e2 = cross3(e3, e1);
    [e1, e2, e3]
}

/// 把局部坐标下的单位方向 `local` 组合成全局方向 `Σ_k local_k · frame_k`。
fn compose_direction(frame: &[[f64; 3]; 3], local: [f64; 3]) -> [f64; 3] {
    let mut direction = [0.0_f64; 3];
    for (axis, coefficient) in frame.iter().zip(local) {
        for (component, value) in axis.iter().enumerate() {
            direction[component] += coefficient * value;
        }
    }
    direction
}

/// 由分子与每元素半径构建空腔（[`CavityFrame::Global`]，SV-01 的原始行为）。
///
/// `radii_angstrom` 是**每元素**的 vdW 半径（Å），`scale` 是半径缩放因子，
/// `points_per_atom` 是每球的点数。
///
/// 这个函数**逐位**等于 [`cavity_with_frame`] 传 [`CavityFrame::Global`]；要转动协变的
/// 空腔用 [`cavity_with_frame`] 与 [`CavityFrame::Covariant`]。
///
/// # Errors
///
/// 元素缺半径、点数太少、出现非有限值 → [`EngineError::Data`] / [`EngineError::InvalidInput`]。
pub fn cavity(
    molecule: &Molecule,
    radii_angstrom: &BTreeMap<String, f64>,
    scale: f64,
    points_per_atom: usize,
) -> Result<Cavity, EngineError> {
    cavity_with_frame(
        molecule,
        radii_angstrom,
        scale,
        points_per_atom,
        CavityFrame::Global,
    )
}

/// 由分子与每元素半径构建空腔，**可选坐标架**。
///
/// 除坐标架外与 [`cavity`] 完全相同：同一批 Fibonacci 点、同一批面积权重
/// `4πR²/N`、同一个二值遮蔽判定、同一个散度定理体积。
///
/// # 两种模式的区别
///
/// * [`CavityFrame::Global`]：点在全局坐标轴下生成（原始行为，逐位不变）。
/// * [`CavityFrame::Covariant`]：点在每原子的局部坐标架（[`atom_frame`]）里生成，
///   坐标架随分子刚性转动而转动，所以面积/体积对转动精确不变。
///
/// # Errors
///
/// 同 [`cavity`]。
pub fn cavity_with_frame(
    molecule: &Molecule,
    radii_angstrom: &BTreeMap<String, f64>,
    scale: f64,
    points_per_atom: usize,
    frame: CavityFrame,
) -> Result<Cavity, EngineError> {
    if points_per_atom < 12 {
        return Err(EngineError::InvalidInput(format!(
            "cavity: {points_per_atom} points per atom is too few (need >= 12)"
        )));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "cavity: scale must be positive and finite, got {scale}"
        )));
    }
    let mut tesserae = Vec::new();
    let mut area_total = 0.0;
    let mut volume_total = 0.0;
    // 协变模式下先把每原子的坐标架算好（只依赖几何，一次 O(n²)）；
    // 全局模式保持 `None`，走的是与原始实现**逐位相同**的那条分支。
    let frames: Option<Vec<[[f64; 3]; 3]>> = match frame {
        CavityFrame::Global => None,
        CavityFrame::Covariant => Some(
            (0..molecule.atoms.len())
                .map(|index| atom_frame(molecule, index))
                .collect(),
        ),
    };
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let radius = radii_angstrom.get(&atom.symbol).copied().ok_or_else(|| {
            EngineError::Data(format!("cavity: no radius for element `{}`", atom.symbol))
        })?;
        if atomic_number(&atom.symbol).is_none() {
            return Err(EngineError::Data(format!(
                "cavity: unknown element `{}`",
                atom.symbol
            )));
        }
        let weighted = scale * radius;
        if !weighted.is_finite() || weighted <= 0.0 {
            return Err(EngineError::Data(format!(
                "cavity: element `{}` has a non-positive scaled radius {weighted}",
                atom.symbol
            )));
        }
        let center = atom.position_angstrom;
        let area_per_point =
            4.0 * std::f64::consts::PI * weighted * weighted / points_per_atom as f64;
        for k in 0..points_per_atom {
            let local = fibonacci_point(k, points_per_atom);
            let normal = match &frames {
                Some(frames) => compose_direction(&frames[index], local),
                None => local,
            };
            let point = [
                center[0] + weighted * normal[0],
                center[1] + weighted * normal[1],
                center[2] + weighted * normal[2],
            ];
            // 遮蔽判定：落在**另一个**球内部就丢掉（严格小于，边界点保留）。
            let mut exposed = true;
            for (other_index, other) in molecule.atoms.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let other_radius = radii_angstrom.get(&other.symbol).copied().ok_or_else(|| {
                    EngineError::Data(format!("cavity: no radius for element `{}`", other.symbol))
                })?;
                let other_weighted = scale * other_radius;
                let dx = point[0] - other.position_angstrom[0];
                let dy = point[1] - other.position_angstrom[1];
                let dz = point[2] - other.position_angstrom[2];
                let distance2 = dx * dx + dy * dy + dz * dz;
                if distance2 < other_weighted * other_weighted - 1.0e-12 {
                    exposed = false;
                    break;
                }
            }
            if !exposed {
                continue;
            }
            tesserae.push(Tessera {
                center_angstrom: point,
                normal,
                area_angstrom2: area_per_point,
                atom_index: index,
            });
            area_total += area_per_point;
            // 散度定理：ΔV = (1/3)·(r·n)·A，r 从球心量起 → (1/3)·R·A
            // 散度定理：V = (1/3)∮(r·n)dA，r 是**全局**坐标（从球心量起会漏掉 center·n 项）。
            let radial = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
            volume_total += radial * area_per_point / 3.0;
        }
    }
    let cavity = Cavity {
        tesserae,
        surface_area_angstrom2: area_total,
        volume_angstrom3: volume_total,
    };
    cavity.validate()?;
    Ok(cavity)
}

/// 绕 `axis`（单位向量）转 `angle`（弧度）—— 测试与"转动不变性"用。
#[must_use]
pub fn rotate_molecule(molecule: &Molecule, axis: [f64; 3], angle: f64) -> Molecule {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
    let (c, s) = (angle.cos(), angle.sin());
    // Rodrigues
    let rotate = |p: [f64; 3]| -> [f64; 3] {
        let dot = x * p[0] + y * p[1] + z * p[2];
        [
            p[0] * c + (y * p[2] - z * p[1]) * s + x * dot * (1.0 - c),
            p[1] * c + (z * p[0] - x * p[2]) * s + y * dot * (1.0 - c),
            p[2] * c + (x * p[1] - y * p[0]) * s + z * dot * (1.0 - c),
        ]
    };
    let atoms = molecule
        .atoms
        .iter()
        .map(|atom| {
            let mut rotated = atom.clone();
            rotated.position_angstrom = rotate(atom.position_angstrom);
            rotated
        })
        .collect();
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

/// **解析**（球帽）版本的暴露面积与体积 —— 只依赖球心距离，因此对转动**精确**不变。
///
/// 对每对球 `(i, j)`（`d < R_i + R_j` 且互不包含），截平面到球心 `i` 的距离是
///
/// ```text
/// z_ij = (d² + R_i² − R_j²) / (2d)
/// ```
///
/// 被球 `j` 切掉的**球帽高度**是 `h_ij = R_i − z_ij`（`z < 0` 时球心 `i` 落在球 `j`
/// 内、切掉超过半球，公式仍成立；`h` 夹到 `[0, 2R_i]` 处理包含关系）。
/// 面积 `A_cap = 2πR_i h`，体积 `V_cap = πh²(3R_i − h)/3`。
///
/// # 曾经的错与现在的口径
///
/// 这里的 `h` 一度被写成 `z` 本身（"帽高 = 切片位置"），单原子与不交叠的体系都看不出来，
/// 一交叠就把球帽面积算成 `2πR_i z`：H₂（`R = 1.6 Å`、`d = 1.2 Å`）上偏大 15 %，
/// 是 SV-03 用双原子解析面积做锚点时才暴露的。**只有两个球**时上式精确；
/// 三个以上球时逐对相减会**重复扣除**三重交叠（同一块被减两次）——那部分误差不是
/// 舍入，而是这个近似本身的口径，见 [`analytic_area_volume`] 的测试与账本。
///
/// # 已知近似
///
/// 逐对相减会**重复扣除**三重交叠。水（Bondi 半径）实测比稠密采样**小 18 %**，
/// 方向恒定（总是偏小）、对转动仍精确不变。精确处理三重交叠需要球面多边形求交，
/// 留到后面；因此这个函数**只**在"两个球"时当解析锚点用。
///
/// # Errors
///
/// 元素缺半径、`scale` 非正 → [`EngineError::Data`] / [`EngineError::InvalidInput`]。
#[must_use = "返回面积与体积"]
pub fn analytic_area_volume(
    molecule: &Molecule,
    radii_angstrom: &BTreeMap<String, f64>,
    scale: f64,
) -> Result<(f64, f64), EngineError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "cavity: scale must be positive and finite, got {scale}"
        )));
    }
    let n = molecule.n_atoms();
    let mut radii = Vec::with_capacity(n);
    for atom in &molecule.atoms {
        let radius = radii_angstrom.get(&atom.symbol).copied().ok_or_else(|| {
            EngineError::Data(format!("cavity: no radius for element `{}`", atom.symbol))
        })?;
        radii.push(scale * radius);
    }
    let mut area = 0.0;
    let mut volume = 0.0;
    for i in 0..n {
        let r_i = radii[i];
        let mut removed_area = 0.0;
        let mut removed_volume = 0.0;
        // 逐对球算球帽；两个下标都要用（面积扣在 i 上、半径取 j），用下标更清楚。
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_j = radii[j];
            let dx =
                molecule.atoms[i].position_angstrom[0] - molecule.atoms[j].position_angstrom[0];
            let dy =
                molecule.atoms[i].position_angstrom[1] - molecule.atoms[j].position_angstrom[1];
            let dz =
                molecule.atoms[i].position_angstrom[2] - molecule.atoms[j].position_angstrom[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance >= r_i + r_j || distance <= (r_i - r_j).abs() {
                // 不相交，或一个球完全含在另一个里（后者按"小球的球面全被遮住"处理）
                if distance <= (r_i - r_j).abs() && r_i <= r_j + 1.0e-12 {
                    removed_area += 4.0 * std::f64::consts::PI * r_i * r_i;
                    removed_volume += 4.0 * std::f64::consts::PI * r_i * r_i * r_i / 3.0;
                }
                continue;
            }
            // 截平面到球心 i 的距离 z；帽高是 R_i − z（不是 z 本身）。
            let z = (distance * distance + r_i * r_i - r_j * r_j) / (2.0 * distance);
            let h = (r_i - z).clamp(0.0, 2.0 * r_i);
            removed_area += 2.0 * std::f64::consts::PI * r_i * h;
            removed_volume += std::f64::consts::PI * h * h * (3.0 * r_i - h) / 3.0;
        }
        area += (4.0 * std::f64::consts::PI * r_i * r_i - removed_area).max(0.0);
        volume += (4.0 * std::f64::consts::PI * r_i * r_i * r_i / 3.0 - removed_volume).max(0.0);
    }
    Ok((area, volume))
}
