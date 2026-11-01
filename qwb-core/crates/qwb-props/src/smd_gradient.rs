//! SV-03 的第四层：CDS 的**解析梯度**（`∂E_CDS/∂R`，kcal/(mol·Å)）。
//!
//! `E_CDS = Σ_i (STS_i + CSSIGM)·A_i·1e-3`，所以
//!
//! ```text
//! ∂E/∂R_m = 1e-3 · [ Σ_i (STS_i + CSSIGM)·∂A_i/∂R_m + Σ_i A_i·∂STS_i/∂R_m ]
//! ```
//!
//! # 为什么梯度**必须**用带切换函数的面（`w > 0`）
//!
//! `w = 0` 时面积是"采样点在不在别的球里"的阶跃函数：它对核坐标的导数几乎处处为 0，
//! 真正的变化集中在遮挡边界那层测度零的集合上 —— 解析导数与有限差分**都不存在**。
//! 所以能量可以（也确实）用 `w = 0` 对上游，而**梯度**必须用 `w > 0` 的平滑面；函数在
//! `w <= 0` 时显式报错，而不是悄悄给 0。两处口径在账本里分开记。
//!
//! # 导数都是显式的
//!
//! * 切换 `θ = S(u)`、`u = (d − (R − w))/w`、`S(u) = u²(3 − 2u)` → `dθ/dd = 6u(1 − u)/w`
//!   （`u` 被夹住时导数为 0）；
//! * `COT = exp(Δ/(d − (R + Δ)))` → `dCOT/dd = −COT·Δ/(d − (R + Δ))²`；
//! * 软核 `W = exp(1 − 1/(1 − x))/e`、`x = (d − R)/Δ` → `dW/dd = −W/(Δ(1 − x)²)`。
//!
//! 面积那半：`A_i = (4πR_i²/N)·Σ_k Π_{j≠i} θ_ij(p_k)`、`p_k = c_i + R_i n_k`，于是
//! `m = i` 走 `∂p_k/∂c_i = 1`、`m ≠ i` 只经过 `θ_im`（符号相反）。
//!
//! `STS` 那半按 [`crate::smd_sigma`] 的分支逐条求导（H/C/N/O/S），N 的三链
//! （`i`–`C`–`k`）也在内。判据是 Richardson 有限差分 —— 专抓链式漏项。
//!
//! # 不实装就报错
//!
//! 上游还有七条成对分支（`SIGMA(107/111/112/113/115/117/118)`）本仓库数据全为 0，
//! [`crate::smd_sigma`] 没写它们、这里也没写。入口会**先检查**这七项确实为 0，否则报错 ——
//! 而不是悄悄漏掉它们对梯度的贡献。
#![deny(unsafe_code)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::elements::atomic_number;
use qwb_registry::smd_atomic::SmdAtomic;

use crate::cavity::fibonacci_point;
use crate::smd::SmdDescriptors;
use crate::smd_sigma::{
    DEFAULT_DELTA, ISTS, cot_weight, hsigma_at, pair_sigma_at, smd_sts, soft_weight,
};
use crate::smd_surface::cds_effective_radii;

/// 成对分支里"上游有、本仓库数据为 0、因此没实装"的七项（见模块文档）。
const UNIMPLEMENTED_PAIRS: [usize; 7] = [7, 11, 12, 13, 15, 17, 18];

/// `derivatives[i][m] = ∂A_i/∂c_m`。
type AreaDerivatives = Vec<Vec<[f64; 3]>>;

/// CDS 的能量与梯度。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CdsEnergyGradient {
    /// `E_CDS`，kcal/mol。
    pub energy_kcal: f64,
    /// `∂E_CDS/∂R`，kcal/(mol·Å)，每个原子一个三分量向量。
    pub gradient_kcal_per_angstrom: Vec<[f64; 3]>,
}

/// 切换函数的值与 `dθ/dd`。
fn switch_and_derivative(distance: f64, radius: f64, width: f64) -> (f64, f64) {
    if width <= 0.0 {
        return (if distance < radius { 0.0 } else { 1.0 }, 0.0);
    }
    let inner = radius - width;
    let u = ((distance - inner) / width).clamp(0.0, 1.0);
    let value = u * u * (3.0 - 2.0 * u);
    let derivative = if u <= 0.0 || u >= 1.0 {
        0.0
    } else {
        6.0 * u * (1.0 - u) / width
    };
    (value, derivative)
}

/// `COT` 的值与 `dCOT/dd`。
fn cot_and_derivative(distance: f64, reference: f64, delta: f64) -> (f64, f64) {
    let value = cot_weight(distance, reference, delta);
    if value == 0.0 {
        return (0.0, 0.0);
    }
    let gap = distance - (reference + delta);
    (value, -value * delta / (gap * gap))
}

/// 软核 `W` 的值与 `dW/dd`。
fn soft_and_derivative(distance: f64, reference: f64, delta: f64) -> (f64, f64) {
    let value = soft_weight(distance, reference, delta);
    if value == 0.0 {
        return (0.0, 0.0);
    }
    let x = (distance - reference) / delta;
    let one_minus_x = 1.0 - x;
    (value, -value / (delta * one_minus_x * one_minus_x))
}

/// 原子之间的几何量、`RKKVAL` 参考键长（含 O–C / O–O 特例）与成键权重。
struct Geometry<'a> {
    data: &'a SmdAtomic,
    molecule: &'a Molecule,
    symbols: Vec<&'a str>,
    atomic_numbers: Vec<usize>,
    natoms: usize,
}

impl<'a> Geometry<'a> {
    fn new(data: &'a SmdAtomic, molecule: &'a Molecule) -> Result<Self, EngineError> {
        let symbols: Vec<&'a str> = molecule
            .atoms
            .iter()
            .map(|atom| atom.symbol.as_str())
            .collect();
        let mut atomic_numbers = Vec::with_capacity(symbols.len());
        for symbol in &symbols {
            let number = atomic_number(symbol)
                .ok_or_else(|| EngineError::Data(format!("smd: unknown element `{symbol}`")))?;
            atomic_numbers.push(number as usize);
        }
        Ok(Self {
            data,
            molecule,
            natoms: symbols.len(),
            symbols,
            atomic_numbers,
        })
    }

    fn position(&self, index: usize) -> [f64; 3] {
        self.molecule.atoms[index].position_angstrom
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (self.position(i), self.position(j));
        let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 从 `j` 指向 `i` 的单位向量（等于 `∂d_ij/∂c_i`）。
    fn unit(&self, i: usize, j: usize) -> [f64; 3] {
        let (a, b) = (self.position(i), self.position(j));
        let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d == 0.0 {
            return [0.0; 3];
        }
        [dx / d, dy / d, dz / d]
    }

    fn element_type(&self, symbol: &str) -> usize {
        self.data
            .cds
            .bonds
            .element_types
            .get(symbol)
            .copied()
            .unwrap_or(0)
    }

    fn reference(&self, i: usize, j: usize) -> Option<(f64, f64)> {
        let (zi, zj) = (self.atomic_numbers[i], self.atomic_numbers[j]);
        if (zi == 8 && zj == 6) || (zi == 6 && zj == 8) {
            return Some((1.330, 0.10));
        }
        if ISTS == 6 && zi == 8 && zj == 8 {
            return Some((1.800, 0.30));
        }
        let ti = self.element_type(self.symbols[i]);
        let tj = self.element_type(self.symbols[j]);
        if ti == 0 || tj == 0 {
            return None;
        }
        let reference = self.data.cds.bonds.rkkval_angstrom[ti - 1][tj - 1];
        if reference <= 0.0 {
            return None;
        }
        Some((reference, DEFAULT_DELTA))
    }

    fn cot_pair(&self, i: usize, j: usize) -> (f64, f64) {
        match self.reference(i, j) {
            Some((reference, delta)) => cot_and_derivative(self.distance(i, j), reference, delta),
            None => (0.0, 0.0),
        }
    }

    /// `RKKVAL(type_k, type_j)`（N 的软核里用；`j` 是碳）。
    fn soft_reference(&self, k: usize, carbon: usize) -> Option<f64> {
        let kind = self.element_type(self.symbols[k]);
        let carbon_type = self.element_type(self.symbols[carbon]);
        if kind == 0 || carbon_type == 0 {
            return None;
        }
        let reference = self.data.cds.bonds.rkkval_angstrom[kind - 1][carbon_type - 1];
        if reference <= 0.0 {
            None
        } else {
            Some(reference)
        }
    }
}

fn add_scaled(target: &mut [f64; 3], vector: [f64; 3], scale: f64) {
    for axis in 0..3 {
        target[axis] += scale * vector[axis];
    }
}

/// `(A_i, ∂A_i/∂c_m)`：`derivatives[i][m]`。
fn surface_areas_and_gradients(
    data: &SmdAtomic,
    geometry: &Geometry<'_>,
    points_per_atom: usize,
    width: f64,
    // 每原子面积 + 每个 (i, m) 的偏导：`derivatives[i][m]`。
) -> Result<(Vec<f64>, AreaDerivatives), EngineError> {
    let radii = cds_effective_radii(data, geometry.molecule)?;
    let n = geometry.natoms;
    let mut areas = vec![0.0_f64; n];
    let mut derivatives = vec![vec![[0.0_f64; 3]; n]; n];
    for i in 0..n {
        let radius = radii[i];
        let area_per_point = 4.0 * std::f64::consts::PI * radius * radius / points_per_atom as f64;
        let center = geometry.position(i);
        for k in 0..points_per_atom {
            let normal = fibonacci_point(k, points_per_atom);
            let point = [
                center[0] + radius * normal[0],
                center[1] + radius * normal[1],
                center[2] + radius * normal[2],
            ];
            let mut theta = vec![1.0_f64; n];
            let mut dtheta = vec![0.0_f64; n];
            let mut unit = vec![[0.0_f64; 3]; n];
            let mut exposed = 1.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let other = geometry.position(j);
                let (dx, dy, dz) = (
                    point[0] - other[0],
                    point[1] - other[1],
                    point[2] - other[2],
                );
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                let (value, derivative) = switch_and_derivative(distance, radii[j], width);
                theta[j] = value;
                exposed *= value;
                if derivative != 0.0 && distance > 0.0 {
                    dtheta[j] = derivative;
                    unit[j] = [dx / distance, dy / distance, dz / distance];
                }
            }
            areas[i] += area_per_point * exposed;
            for j in 0..n {
                if j == i || dtheta[j] == 0.0 {
                    continue;
                }
                let mut product_without_j = 1.0_f64;
                for (l, value) in theta.iter().enumerate() {
                    if l != i && l != j {
                        product_without_j *= value;
                    }
                }
                let scale = area_per_point * product_without_j * dtheta[j];
                add_scaled(&mut derivatives[i][i], unit[j], scale);
                add_scaled(&mut derivatives[i][j], unit[j], -scale);
            }
        }
    }
    Ok((areas, derivatives))
}

/// `Σ_i A_i·∂STS_i/∂c_m`（已按面积加权，单位 cal/(mol·Å)）。
fn sts_derivatives(
    data: &SmdAtomic,
    geometry: &Geometry<'_>,
    areas: &[f64],
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<Vec<[f64; 3]>, EngineError> {
    for pair in UNIMPLEMENTED_PAIRS {
        let value = pair_sigma_at(data, pair, water, descriptors)?;
        if value != 0.0 {
            return Err(EngineError::Data(format!(
                "smd: the CDS gradient does not implement pair {pair} (SIGMA({})) but its \
                 value is {value}; the chain rule for that branch is missing",
                100 + pair
            )));
        }
    }
    let n = geometry.natoms;
    let mut derivatives = vec![[0.0_f64; 3]; n];
    for i in 0..n {
        let weight = areas[i];
        match geometry.atomic_numbers[i] {
            1 => {
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let (cot, dcot) = geometry.cot_pair(i, j);
                    if cot == 0.0 && dcot == 0.0 {
                        continue;
                    }
                    let value = hsigma_at(data, geometry.symbols[j], water, descriptors)?;
                    let scale = weight * dcot * value;
                    add_scaled(&mut derivatives[i], geometry.unit(i, j), scale);
                    add_scaled(&mut derivatives[j], geometry.unit(i, j), -scale);
                }
            }
            8 => {
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let pair = match geometry.atomic_numbers[j] {
                        6 => Some(3),
                        7 => Some(6),
                        16 => Some(18),
                        14 => Some(17),
                        15 => Some(14),
                        8 => {
                            if ISTS == 6 {
                                Some(4)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    let Some(pair) = pair else { continue };
                    let (cot, dcot) = geometry.cot_pair(i, j);
                    if cot == 0.0 && dcot == 0.0 {
                        continue;
                    }
                    let value = pair_sigma_at(data, pair, water, descriptors)?;
                    let scale = weight * dcot * value;
                    add_scaled(&mut derivatives[i], geometry.unit(i, j), scale);
                    add_scaled(&mut derivatives[j], geometry.unit(i, j), -scale);
                }
            }
            6 => {
                let carbon_type = geometry.element_type(geometry.symbols[i]);
                if carbon_type == 0 {
                    continue;
                }
                let reference1 = data.cds.bonds.rkkval_angstrom[carbon_type - 1][carbon_type - 1];
                let p1 = pair_sigma_at(data, 1, water, descriptors)?;
                let p2 = pair_sigma_at(data, 2, water, descriptors)?;
                let p10 = pair_sigma_at(data, 10, water, descriptors)?;
                let mut contacts = 0.0_f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    if geometry.atomic_numbers[j] == 7 {
                        contacts += geometry.cot_pair(i, j).0;
                    }
                }
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let unit = geometry.unit(i, j);
                    let distance = geometry.distance(i, j);
                    let mut scale = 0.0;
                    if geometry.atomic_numbers[j] == 6 {
                        let (_, d1) = soft_and_derivative(distance, reference1, DEFAULT_DELTA);
                        let (_, d2) = soft_and_derivative(distance, 1.27, 0.07);
                        scale += d1 * p1 + d2 * p2;
                    }
                    if geometry.atomic_numbers[j] == 7 {
                        let (_, dcot) = geometry.cot_pair(i, j);
                        scale += 2.0 * contacts * dcot * p10;
                    }
                    if scale != 0.0 {
                        add_scaled(&mut derivatives[i], unit, weight * scale);
                        add_scaled(&mut derivatives[j], unit, -weight * scale);
                    }
                }
            }
            7 => {
                let p5 = pair_sigma_at(data, 5, water, descriptors)?;
                let p16 = pair_sigma_at(data, 16, water, descriptors)?;
                // 第一遍：只算 RTKKS（`∂(RTKKS^1.3)` 需要它的值），并缓存链式项。
                let mut rtkks = 0.0_f64;
                struct Term {
                    j: usize,
                    cot: f64,
                    dcot: f64,
                    unit: [f64; 3],
                    soft_sum: f64,
                    soft_terms: Vec<(usize, f64, [f64; 3])>,
                }
                let mut terms: Vec<Term> = Vec::new();
                for j in 0..n {
                    if j == i || geometry.atomic_numbers[j] != 6 {
                        continue;
                    }
                    let (cot, dcot) = geometry.cot_pair(i, j);
                    let mut soft_sum = 0.0_f64;
                    let mut soft_terms = Vec::new();
                    for k in 0..n {
                        if k == i || k == j {
                            continue;
                        }
                        let Some(reference) = geometry.soft_reference(k, j) else {
                            continue;
                        };
                        let (value, derivative) =
                            soft_and_derivative(geometry.distance(j, k), reference, DEFAULT_DELTA);
                        soft_sum += value;
                        if derivative != 0.0 {
                            soft_terms.push((k, derivative, geometry.unit(j, k)));
                        }
                    }
                    rtkks += cot * soft_sum * soft_sum;
                    terms.push(Term {
                        j,
                        cot,
                        dcot,
                        unit: geometry.unit(i, j),
                        soft_sum,
                        soft_terms,
                    });
                }
                let exponent_scale = 1.3 * rtkks.powf(0.3);
                for term in &terms {
                    // d(RTKKS^1.3)/d(c_i) = 1.3·RTKKS^0.3·dcot·S_j²
                    let scale = p5 * exponent_scale * term.dcot * term.soft_sum * term.soft_sum;
                    add_scaled(&mut derivatives[i], term.unit, weight * scale);
                    add_scaled(&mut derivatives[term.j], term.unit, -weight * scale);
                    // d/d(c_j) 与 d/d(c_k)：1.3·RTKKS^0.3·2·COT·S_j·dS_j
                    let chain = p5 * exponent_scale * 2.0 * term.cot * term.soft_sum;
                    for (k, derivative, unit) in &term.soft_terms {
                        let contribution = weight * chain * derivative;
                        add_scaled(&mut derivatives[term.j], *unit, contribution);
                        add_scaled(&mut derivatives[*k], *unit, -contribution);
                    }
                }
                // 三键：`Σ W(d_ij; 1.225, 0.065)·SIGMA(116)`
                for j in 0..n {
                    if j == i || geometry.atomic_numbers[j] != 6 {
                        continue;
                    }
                    let (_, derivative) =
                        soft_and_derivative(geometry.distance(i, j), 1.225, 0.065);
                    if derivative != 0.0 {
                        let scale = weight * derivative * p16;
                        add_scaled(&mut derivatives[i], geometry.unit(i, j), scale);
                        add_scaled(&mut derivatives[j], geometry.unit(i, j), -scale);
                    }
                }
                // `SIGMA(111)`（RTKKS2）在本仓库数据里为 0，入口已检查。
            }
            16 => {
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let pair = match geometry.atomic_numbers[j] {
                        16 => Some(7),
                        15 => Some(15),
                        _ => None,
                    };
                    let Some(pair) = pair else { continue };
                    let (cot, dcot) = geometry.cot_pair(i, j);
                    if cot == 0.0 && dcot == 0.0 {
                        continue;
                    }
                    let value = pair_sigma_at(data, pair, water, descriptors)?;
                    let scale = weight * dcot * value;
                    add_scaled(&mut derivatives[i], geometry.unit(i, j), scale);
                    add_scaled(&mut derivatives[j], geometry.unit(i, j), -scale);
                }
            }
            _ => {}
        }
    }
    Ok(derivatives)
}

/// CDS 的能量与解析梯度。
///
/// `switch_width_angstrom` **必须**大于 0（见模块文档：`w = 0` 的梯度不存在）。
///
/// # Errors
///
/// `switch_width <= 0`、点数太少、元素缺数据、成对分支与实现不符 → 显式错误。
pub fn smd_cds_energy_gradient(
    data: &SmdAtomic,
    molecule: &Molecule,
    points_per_atom: usize,
    switch_width_angstrom: f64,
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<CdsEnergyGradient, EngineError> {
    if !switch_width_angstrom.is_finite() || switch_width_angstrom <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "smd: the CDS gradient needs a positive switch width (the binary limit is not \
             differentiable), got {switch_width_angstrom}"
        )));
    }
    if points_per_atom < 12 {
        return Err(EngineError::InvalidInput(format!(
            "smd: {points_per_atom} points per atom is too few (need >= 12)"
        )));
    }
    let geometry = Geometry::new(data, molecule)?;
    let (areas, darea) =
        surface_areas_and_gradients(data, &geometry, points_per_atom, switch_width_angstrom)?;
    let sts = smd_sts(data, molecule, water, descriptors)?;
    let dsts = sts_derivatives(data, &geometry, &areas, water, descriptors)?;
    let cssigm = crate::smd::cds_constant_term(data, water, descriptors);
    let energy_kcal = crate::smd_sigma::cds_energy_kcal_from_sts(&sts, cssigm, &areas)
        .map_err(EngineError::Data)?;
    let mut gradient = vec![[0.0_f64; 3]; geometry.natoms];
    for m in 0..geometry.natoms {
        gradient[m] = dsts[m];
        for i in 0..geometry.natoms {
            let scale = sts[i] + cssigm;
            add_scaled(&mut gradient[m], darea[i][m], scale);
        }
        for component in gradient[m].iter_mut() {
            *component /= 1000.0;
        }
    }
    Ok(CdsEnergyGradient {
        energy_kcal,
        gradient_kcal_per_angstrom: gradient,
    })
}
