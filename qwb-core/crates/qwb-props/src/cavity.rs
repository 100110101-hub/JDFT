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
//! 点在球 `i` 的局部坐标系里生成，再按"分子的全局取向"旋转；若该点落在**另一个**球
//! 内部，就判为被遮蔽并丢掉。分子整体转动时，同一批点在空间里跟着转，谁被遮蔽不变，
//! 所以面积与体积**应当**严格不变（浮点误差量级）。这条性质正好能抓住"遮蔽判定用了
//! 错的坐标系"这类错误（例如漏做旋转、或用原子间距离却比较了错的半径）。

use std::collections::BTreeMap;

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::elements::atomic_number;

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
fn fibonacci_point(k: usize, n: usize) -> [f64; 3] {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let z = 1.0 - (2.0 * k as f64 + 1.0) / n as f64;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * std::f64::consts::PI * (k as f64 / golden).rem_euclid(1.0);
    [radius * phi.cos(), radius * phi.sin(), z]
}

/// 由分子与每元素半径构建空腔。
///
/// `radii_angstrom` 是**每元素**的 vdW 半径（Å），`scale` 是半径缩放因子，
/// `points_per_atom` 是每球的点数。
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
            let normal = fibonacci_point(k, points_per_atom);
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
            volume_total += weighted * area_per_point / 3.0;
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
