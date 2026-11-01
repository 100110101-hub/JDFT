//! DFT-D4 的**中间量**层：配位数、EEQ 电荷、参考权重与电荷相关的 C6（DA-06）。
//!
//! # 分层
//!
//! 上游（dftd4 v4.3.1）把 D4 拆成四步，本项目照这个顺序实现与验收，每一步都有独立的对照物：
//!
//! ```text
//! ① CN     电负性加权的误差函数 CN（纯几何量）
//! ② EEQ    χ/η/rad + 拉格朗日约束 Σq = Q 的稠密线性系统 → 原子电荷
//! ③ C6     每个参考体系一对 alpha(iω)，按参考权重加权后做 3/π·trapzd 的 Casimir–Polder 积分
//! ④ 能量   两体项 + ATM 三体项（ATM 用 q = 0 重建的 C6）——下一段
//! ```
//!
//! 本模块只做 ①②③：这样能量一旦对不上，可以先用 `d4_properties.csv` 的 CN/电荷/C6 三列把错
//! 定位到具体一层。D3 那轮只有一个总能量，错了只能二分。
//!
//! # 与上游的两处"看起来该一样、其实不一样"
//!
//! * **长度常数**：D4 内部的半径换算是 `rad_angstrom × aatoau`，`aatoau` 来自 mctc-lib 的
//!   CODATA-2018 常数（`1.8897261257783122`），**不是** K2 的 `BOHR_PER_ANGSTROM`
//!   （`1.889726089838572`）。两者相对差 1.9e-8，恰好等于验证余差；混用会让 C6 整体偏 1e-8。
//! * **ATM 的 C6** 用 `q = 0` 重建（上游 `disp.f90:139-141`），不是分子电荷；用分子电荷偏最多 30%。
#![deny(unsafe_code)]

use qwb_registry::d4::D4Dispersion;
use qwb_registry::elements;

/// EEQ 模型的 CN 截断（Bohr）。上游 `multicharge` 的 eeq 模型用它；**不是**
/// `coordination_number.cutoff_bohr`（那是 D4 CN 的 30 Bohr）。
const EEQ_CUTOFF_BOHR: f64 = 25.0;

/// EEQ 的 CN 上限（mctc 的 `cn_max = 8`，用软上限而不是硬截断）。
const EEQ_CN_MAX: f64 = 8.0;

/// `x/sqrt(x + ε)` 里的 ε，避免 `CN = 0` 时 0/0。
const CN_REGULARIZER: f64 = 1.0e-14;

/// `sqrt(2/π)`（EEQ 对角项的系数；从 `FRAC_2_PI` 算，不写魔数）。
fn sqrt_two_over_pi() -> f64 {
    std::f64::consts::FRAC_2_PI.sqrt()
}

/// 一层原子的 D4 中间量（能量与梯度都从它出发）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct D4Properties {
    /// 配位数（D4 的 EN 加权误差函数 CN）。
    pub coordination_numbers: Vec<f64>,
    /// EEQ 电荷（电子单位；总和 = 总电荷）。
    pub charges: Vec<f64>,
    /// 电荷与配位相关的 C6 矩阵（`c6[i][j]`，Ha·Bohr⁶）。
    pub c6: Vec<Vec<f64>>,
}

/// 数据文件里该元素的下标（`elements` 按 Z 升序，与 `eeq.*` / `reference` 同步）。
fn element_index(data: &D4Dispersion, symbol: &str) -> Option<usize> {
    data.elements
        .iter()
        .position(|candidate| candidate == symbol)
}

/// `nint`：四舍五入到最近整数（上游 Fortran 的 `nint`，.5 远离零）。
fn nint(value: f64) -> i32 {
    let rounded = (value.abs() + 0.5).floor();
    if value < 0.0 {
        -(rounded as i32)
    } else {
        rounded as i32
    }
}

/// `ζ(a, c, q_ref, q)`：上游 `utils.f90` 的电荷缩放函数。
fn zeta(a: f64, c: f64, q_ref: f64, q: f64) -> f64 {
    if q < 0.0 {
        a.exp()
    } else {
        (a * (1.0 - (c * (1.0 - q_ref / q)).exp())).exp()
    }
}

/// 每个参考体系的高斯权重个数 `ngw`（上游 `reference.f90` 的 `set_refgw_num`：
/// `ngw = icn·(icn+1)/2`，`icn` 是**同整数 CN** 的参考体系个数）。
fn reference_weight_counts(reference_counts: &[f64]) -> Vec<usize> {
    let mut histogram = [0usize; 20];
    histogram[0] = 1;
    for count in reference_counts {
        histogram[nint(*count).clamp(0, 19) as usize] += 1;
    }
    reference_counts
        .iter()
        .map(|count| {
            let n = histogram[nint(*count).clamp(0, 19) as usize];
            n * (n + 1) / 2
        })
        .collect()
}

fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    squared_distance(a, b).sqrt()
}

/// ① D4 的配位数：`½(1 + erf(−kcn(R−rc)/rc))` 乘电负性因子，
/// `rc = 4/3·(rcov_2009,i + rcov_2009,j) × aatoau`。
fn coordination_numbers(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
) -> Vec<f64> {
    let n = indices.len();
    let mut values = vec![0.0_f64; n];
    let cn = &data.coordination_number;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let left = &data.reference[indices[i]];
            let right = &data.reference[indices[j]];
            let rc = (left.covalent_rad_2009_angstrom + right.covalent_rad_2009_angstrom)
                * cn.radius_scaling
                * data.constants.aatoau;
            if rc <= 0.0 {
                continue;
            }
            let count =
                0.5 * (1.0 + erf(-cn.kcn * (distance(positions[i], positions[j]) - rc) / rc));
            let delta = (left.pauling_en - right.pauling_en).abs();
            let factor = cn.k4 * (-((delta + cn.k5).powi(2)) / cn.k6).exp();
            values[i] += factor * count;
        }
    }
    values
}

/// EEQ 的**原始**（未加软上限）误差函数 CN，以及每对的 `∂cn_p/∂R_pq`。
///
/// 第二项是给 [`charge_gradient`] 用的：`derivative[p][q]` 是 `cn_p` 对
/// `R_pq = |R_q − R_p|` 的偏导（软上限之后还要再乘 [`eeq_soft_cap_derivative`]）。
/// 截断 `R > 25 Bohr` 与 `rcov <= 0` 的处理必须与 `eeq_charges` 完全一致。
#[allow(clippy::needless_range_loop)]
fn eeq_raw_cn(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = indices.len();
    let kcn = data.coordination_number.kcn;
    let scaling = data.coordination_number.radius_scaling * data.constants.aatoau;
    let mut cn = vec![0.0_f64; n];
    let mut derivative = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let rcov = (data.reference[indices[i]].covalent_rad_2009_angstrom
                + data.reference[indices[j]].covalent_rad_2009_angstrom)
                * scaling;
            let r = distance(positions[i], positions[j]);
            if r > EEQ_CUTOFF_BOHR || rcov <= 0.0 {
                continue;
            }
            let u = -kcn * (r - rcov) / rcov;
            cn[i] += 0.5 * (1.0 + erf(u));
            // d/dR [½(1+erf(u))] = (1/√π)·e^{−u²}·du/dR，du/dR = −kcn/rcov。
            derivative[i][j] = -kcn / (rcov * std::f64::consts::PI.sqrt()) * (-(u * u)).exp();
        }
    }
    (cn, derivative)
}

/// EEQ CN 的软上限 `ln(1+e^M) − ln(1+e^{M−cn})`（`M = cn_max`）。
fn eeq_soft_cap(cn: f64) -> f64 {
    (1.0 + EEQ_CN_MAX.exp()).ln() - (1.0 + (EEQ_CN_MAX - cn).exp()).ln()
}

/// 软上限对原始 CN 的导数：`e^{M−cn}/(1+e^{M−cn})`（即 `σ(M−cn)`，恒在 (0,1)）。
fn eeq_soft_cap_derivative(cn: f64) -> f64 {
    let t = (EEQ_CN_MAX - cn).exp();
    t / (1.0 + t)
}

/// 单个原子的 EEQ `rad`（Bohr）。
///
/// # Errors
///
/// 元素符号不在周期表内。
fn eeq_radius(data: &D4Dispersion, index: usize) -> Result<f64, String> {
    let symbol = &data.elements[index];
    let z =
        elements::atomic_number(symbol).ok_or_else(|| format!("d4: unknown element `{symbol}`"))?;
    Ok(data.eeq.rad[(z as usize) - 1])
}

/// ② 的稠密系统：`A`（含拉格朗日约束行/列）与右端项 `b`。
///
/// [`charge_gradient`] 要同一套 `A`、`b`，所以从 [`eeq_charges`] 里提出来这里，
/// 保证电荷与电荷导数用的是逐位相同的矩阵（行为与提出前一致）。
#[allow(clippy::needless_range_loop)] // 拉格朗日约束要同时写第 n 行与第 n 列
fn eeq_system(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
    total_charge: f64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let n = indices.len();
    // EEQ 的 CN：**不带**电负性因子的纯误差函数 CN，截断 25 Bohr，软上限 8。
    let (raw, _) = eeq_raw_cn(data, indices, positions);
    let cn: Vec<f64> = raw.iter().map(|value| eeq_soft_cap(*value)).collect();

    let m = n + 1;
    let mut matrix = vec![vec![0.0_f64; m]; m];
    let mut rhs = vec![0.0_f64; m];
    for i in 0..n {
        let symbol = &data.elements[indices[i]];
        let z = elements::atomic_number(symbol)
            .ok_or_else(|| format!("d4: unknown element `{symbol}`"))?;
        let index = (z as usize) - 1;
        let rad = data.eeq.rad[index];
        matrix[i][i] = data.eeq.eta[index] + sqrt_two_over_pi() / rad;
        rhs[i] =
            -data.eeq.chi[index] + data.eeq.kcnchi[index] * cn[i] / (cn[i] + CN_REGULARIZER).sqrt();
        for j in 0..n {
            if i == j {
                continue;
            }
            let other_symbol = &data.elements[indices[j]];
            let other_z = elements::atomic_number(other_symbol)
                .ok_or_else(|| format!("d4: unknown element `{other_symbol}`"))?;
            let other_rad = data.eeq.rad[(other_z as usize) - 1];
            let squared = squared_distance(positions[i], positions[j]);
            let gamma = 1.0 / (rad * rad + other_rad * other_rad);
            matrix[i][j] = erf((squared * gamma).sqrt()) / squared.sqrt();
        }
    }
    for i in 0..n {
        matrix[n][i] = 1.0;
        matrix[i][n] = 1.0;
    }
    matrix[n][n] = 0.0;
    rhs[n] = total_charge;

    Ok((matrix, rhs))
}

/// ② EEQ(2019) 电荷：`A q = x`，最后一行/列是 `Σq = Q` 的拉格朗日约束。
fn eeq_charges(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
    total_charge: f64,
) -> Result<Vec<f64>, String> {
    let n = indices.len();
    let (matrix, rhs) = eeq_system(data, indices, positions, total_charge)?;
    let solution = solve_dense(matrix, rhs)?;
    Ok(solution[..n].to_vec())
}

/// ③ 每个原子的参考权重（CN 高斯 × 电荷缩放 ζ）。
///
/// 公开是为了让 `tests/da06_d4_derivs.rs` 能在**未扰动几何**上取一份权重喂给
/// [`c6_gradient`]（`D4Properties` 不带权重，而重算 `c6_matrix` 又需要它）。
pub fn reference_weights(
    data: &D4Dispersion,
    indices: &[usize],
    charges: &[f64],
    cn: &[f64],
) -> Vec<Vec<f64>> {
    let mut weights = Vec::with_capacity(indices.len());
    for (atom, index) in indices.iter().enumerate() {
        let element = &data.reference[*index];
        let counts: Vec<f64> = element
            .reference_systems
            .iter()
            .map(|system| system.refcovcn)
            .collect();
        let ngw = reference_weight_counts(&counts);
        let gi = element.chemical_hardness * data.constants.gc;
        let mut exponential = Vec::with_capacity(counts.len());
        let mut norm = 0.0;
        for (r, system) in element.reference_systems.iter().enumerate() {
            let mut sum = 0.0;
            for igw in 1..=ngw[r] {
                sum += (-(igw as f64) * data.constants.wf * (cn[atom] - system.refcovcn).powi(2))
                    .exp();
            }
            exponential.push(sum);
            norm += sum;
        }
        let inverse = if norm.abs() > 0.0 { 1.0 / norm } else { 0.0 };
        let mut atom_weights = Vec::with_capacity(counts.len());
        for (r, system) in element.reference_systems.iter().enumerate() {
            atom_weights.push(
                exponential[r]
                    * inverse
                    * zeta(
                        data.constants.ga,
                        gi,
                        system.clsq + element.zeff,
                        charges[atom] + element.zeff,
                    ),
            );
        }
        weights.push(atom_weights);
    }
    weights
}

/// ③ C6：对每个参考体系对做 `3/π·Σ_k w_k·α_i(ref,k)·α_j(ref',k)`，再按参考权重加权。
fn c6_matrix(
    data: &D4Dispersion,
    indices: &[usize],
    weights: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let n = indices.len();
    let grid = &data.frequency_grid;
    if grid.hartree.len() != grid.trapzd_weights.len() {
        return Err("d4: frequency grid and trapzd weights differ in length".into());
    }
    let mut c6 = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let left = &data.reference[indices[i]];
        for j in 0..n {
            let right = &data.reference[indices[j]];
            let mut total = 0.0;
            for (r, weight_r) in weights[i].iter().enumerate() {
                if *weight_r == 0.0 {
                    continue;
                }
                let alpha_r = &left.reference_systems[r].alpha_iw_eeq;
                for (s, weight_s) in weights[j].iter().enumerate() {
                    if *weight_s == 0.0 {
                        continue;
                    }
                    let alpha_s = &right.reference_systems[s].alpha_iw_eeq;
                    let mut integral = 0.0;
                    for (point, grid_weight) in grid.trapzd_weights.iter().enumerate() {
                        integral += grid_weight * alpha_r[point] * alpha_s[point];
                    }
                    total += weight_r * weight_s * data.constants.thopi * integral;
                }
            }
            c6[i][j] = total;
        }
    }
    Ok(c6)
}

/// 一层原子的 D4 中间量（CN、EEQ 电荷、C6）。
///
/// # Errors
///
/// 元素不在数据覆盖范围内、EEQ 线性系统奇异、坐标/电荷非有限。
pub fn properties(
    data: &D4Dispersion,
    symbols: &[&str],
    positions_bohr: &[[f64; 3]],
    total_charge: f64,
) -> Result<D4Properties, String> {
    if symbols.len() != positions_bohr.len() {
        return Err(format!(
            "d4: {} element symbols for {} positions",
            symbols.len(),
            positions_bohr.len()
        ));
    }
    if positions_bohr
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("d4: positions must be finite (Bohr)".into());
    }
    if !total_charge.is_finite() {
        return Err("d4: total charge must be finite".into());
    }
    let mut indices = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        indices.push(element_index(data, symbol).ok_or_else(|| {
            format!(
                "d4: element `{symbol}` outside the data range ({} elements: {}..{})",
                data.elements.len(),
                data.elements.first().map(String::as_str).unwrap_or("?"),
                data.elements.last().map(String::as_str).unwrap_or("?")
            )
        })?);
    }
    let coordination_numbers = coordination_numbers(data, &indices, positions_bohr);
    let charges = eeq_charges(data, &indices, positions_bohr, total_charge)?;
    let weights = reference_weights(data, &indices, &charges, &coordination_numbers);
    let c6 = c6_matrix(data, &indices, &weights)?;
    Ok(D4Properties {
        coordination_numbers,
        charges,
        c6,
    })
}

/// 一层原子的 D4 色散能（两体 + ATM）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct D4Result {
    /// 总色散能（两体 + ATM），Hartree（负值）。
    pub energy_hartree: f64,
    /// 两体项，Hartree。
    pub two_body_hartree: f64,
    /// ATM 三体项，Hartree。
    pub atm_hartree: f64,
}

/// 计算 D4 的色散能（两体 + ATM）。**梯度还没做**（见模块文档）。
///
/// # 与 D3 的区别（都在能量里体现）
///
/// * C6 由 `alpha(iω)` 现算、且随**电荷**与 CN 变（见 [`properties`]）；
/// * 阻尼是 `s6/(R^6+r0^6) + s8*rrij/(R^8+r0^8)`，**没有**零阻尼的 `alp` 指数；
/// * ATM 三体项默认开（`s9 = 1`），且它的 C6 必须用 **q = 0** 重建 —— 用分子电荷偏最多 30%。
///
/// # Errors
///
/// 该泛函在上游没有 D4 参数（`m06-2x` 就是这种）、元素超范围、EEQ 系统奇异。
pub fn dispersion_energy(
    data: &D4Dispersion,
    functional: &str,
    symbols: &[&str],
    positions_bohr: &[[f64; 3]],
    total_charge: f64,
) -> Result<D4Result, String> {
    let parameters = data
        .damping_parameters
        .get(functional)
        .ok_or_else(|| missing_functional_message(data, functional))?;

    let mut indices = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        indices.push(
            element_index(data, symbol)
                .ok_or_else(|| format!("d4: element `{symbol}` outside the data range"))?,
        );
    }
    let intermediates = properties(data, symbols, positions_bohr, total_charge)?;
    let n = symbols.len();
    let mut two_body = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i >= j {
                continue;
            }
            two_body -= pair_energy(
                data,
                parameters,
                &intermediates,
                symbols,
                positions_bohr,
                i,
                j,
            );
        }
    }

    let atm = if data.dispersion.atm.s9.abs() < 1.0e-16 {
        0.0
    } else {
        // ATM 的 C6 用 **原子电荷 = 0** 重建（上游 disp.f90:139-141）。注意"电荷 = 0"指的是
        // 每个原子的**部分电荷**为零，不是"把体系总电荷设成 0 再解一遍 EEQ"——后者仍然给出
        // ±0.6 的分电荷，实测会让水二聚体的 ATM 从 −3.2866e-7 变成 −9.6076e-8（差 3.42 倍，
        // 这正是 `choices` 一百五十七里那个"常数 2.4"的真身）。
        let atm_c6 = zero_charge_c6(data, &indices, symbols, positions_bohr)?;
        atm_energy(data, parameters, &atm_c6, symbols, positions_bohr)
    };

    Ok(D4Result {
        energy_hartree: two_body + atm,
        two_body_hartree: two_body,
        atm_hartree: atm,
    })
}

/// 泛函缺参数时的报错（`m06-2x` 会带上上游缺失的原因）。
fn missing_functional_message(data: &D4Dispersion, functional: &str) -> String {
    let available: Vec<&str> = data.damping_parameters.keys().map(String::as_str).collect();
    match data.unsupported_functionals.get(functional) {
        Some(reason) => format!(
            "no D4 parameters for `{functional}` in `{}` (available: {}): {}",
            data.name,
            available.join(", "),
            reason.reason
        ),
        None => format!(
            "no D4 parameters for `{functional}` in `{}` (available: {})",
            data.name,
            available.join(", ")
        ),
    }
}

/// 一对原子的两体能量（正数；调用方减去它）。
fn pair_energy(
    data: &D4Dispersion,
    parameters: &qwb_registry::dispersion::DampingParameters,
    properties: &D4Properties,
    symbols: &[&str],
    positions: &[[f64; 3]],
    i: usize,
    j: usize,
) -> f64 {
    let c6 = properties.c6[i][j];
    let r = distance(positions[i], positions[j]);
    let rrij = 3.0 * r4r2(data, symbols[i]) * r4r2(data, symbols[j]);
    let r0 = parameters.a1 * rrij.sqrt() + parameters.a2;
    let t6 = 1.0 / (r.powi(6) + r0.powi(6));
    let t8 = 1.0 / (r.powi(8) + r0.powi(8));
    c6 * (parameters.s6 * t6 + parameters.s8 * rrij * t8)
}

/// 数据的逐元素 `r4r2`（`sqrt(0.5*<r4>/<r2>*sqrt(Z))`）。
fn r4r2(data: &D4Dispersion, symbol: &str) -> f64 {
    element_index(data, symbol)
        .map(|index| data.reference[index].r4r2)
        .unwrap_or(0.0)
}

/// ATM 三体项（上游 `damping/atm.f90`）：`E = -sum_{i<j<k} ang*fdmp*C9`。
/// ATM 用的 C6：参考权重的电荷缩放按 **原子部分电荷 = 0** 求值（上游 `disp.f90:139-141`）。
fn zero_charge_c6(
    data: &D4Dispersion,
    indices: &[usize],
    symbols: &[&str],
    positions: &[[f64; 3]],
) -> Result<Vec<Vec<f64>>, String> {
    if symbols.len() != positions.len() {
        return Err("d4: element count does not match positions".into());
    }
    let cn = coordination_numbers(data, indices, positions);
    let zero_charges = vec![0.0_f64; indices.len()];
    let weights = reference_weights(data, indices, &zero_charges, &cn);
    c6_matrix(data, indices, &weights)
}

fn atm_energy(
    data: &D4Dispersion,
    parameters: &qwb_registry::dispersion::DampingParameters,
    c6: &[Vec<f64>],
    symbols: &[&str],
    positions: &[[f64; 3]],
) -> f64 {
    let n = symbols.len();
    let alp3 = data.dispersion.atm.alp / 3.0;
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if j <= i {
                continue;
            }
            for k in 0..n {
                if k <= j {
                    continue;
                }
                let rij = distance(positions[i], positions[j]);
                let rik = distance(positions[i], positions[k]);
                let rjk = distance(positions[j], positions[k]);
                let r2ij = rij * rij;
                let r2ik = rik * rik;
                let r2jk = rjk * rjk;
                let c9 = -(data.dispersion.atm.s9) * (c6[i][j] * c6[i][k] * c6[j][k]).abs().sqrt();
                let r0 = (parameters.a1
                    * (3.0 * r4r2(data, symbols[i]) * r4r2(data, symbols[j])).sqrt()
                    + parameters.a2)
                    * (parameters.a1
                        * (3.0 * r4r2(data, symbols[i]) * r4r2(data, symbols[k])).sqrt()
                        + parameters.a2)
                    * (parameters.a1
                        * (3.0 * r4r2(data, symbols[j]) * r4r2(data, symbols[k])).sqrt()
                        + parameters.a2);
                let r1 = rij * rik * rjk;
                let r3 = r1.powi(3);
                let r5 = r3 * r1 * r1;
                let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp3));
                let ang =
                    0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
                        / r5
                        + 1.0 / r3;
                total += -(ang * fdmp) * c9;
            }
        }
    }
    total
}

/// `erf(x)`：全正项幂级数 `(2/√π)·e^{−x²}·Σ (2x²)^n·x/(1·3···(2n+1))`。
///
/// 为什么不用 Abramowitz & Stegun 7.1.26（`|ε| ≤ 1.5e-7`）：CN 进入参考权重是
/// `exp(−igw·wf·(CN−CN_ref)²)`（`wf = 6`），1e-7 的逐点 erf 误差会放大成 1e-7 量级的 CN
/// 误差、再放大到 C6 —— 实测 CN 差 2.3e-7、C6 差 1.8e-5（Ha·Bohr⁶），而被对照的官方值
/// 自身只有 7 位小数（可达精度 ~1e-8 相对）。所以这里要的是"比参考值更准"，
/// 全正项求和不抵消、`|x| ≤ 6` 上到 1e-14；`|x| > 6` 时 `erfc < 2.2e-17`，直接饱和到 ±1。
fn erf(x: f64) -> f64 {
    if x >= 6.0 {
        return 1.0;
    }
    if x <= -6.0 {
        return -1.0;
    }
    let square = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0u32;
    while n < 200 {
        n += 1;
        // t_{n+1} = t_n · 2x² / (2n+1)
        term *= 2.0 * square / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term.abs() <= 1.0e-18 * sum.abs() {
            break;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * (-square).exp() * sum
}

/// 稠密线性方程组 `A x = b`（部分主元的高斯消元）。
///
/// 规模是"原子数 + 1"，分子体系 n ≤ 几百，稠密 O(n³) 够用。不引第三方线代库：这一层
/// 只需要一般（非对称）实数矩阵，而 `qwb-linalg` 的求解器面向对称/Hermite 矩阵。
#[allow(clippy::needless_range_loop)] // 消元本身就是"按行列下标原地改"的算法
fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err("d4: EEQ matrix is not square".into());
    }
    for column in 0..n {
        let mut pivot = column;
        let mut best = matrix[column][column].abs();
        for row in (column + 1)..n {
            let value = matrix[row][column].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best <= f64::EPSILON {
            return Err(format!("d4: EEQ matrix is singular at column {column}"));
        }
        if pivot != column {
            matrix.swap(column, pivot);
            rhs.swap(column, pivot);
        }
        let diagonal = matrix[column][column];
        for row in (column + 1)..n {
            let factor = matrix[row][column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for column_index in column..n {
                matrix[row][column_index] -= factor * matrix[column][column_index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for column_index in (row + 1)..n {
            sum -= matrix[row][column_index] * solution[column_index];
        }
        solution[row] = sum / matrix[row][row];
    }
    Ok(solution)
}

// ---------------------------------------------------------------------------
// 导数层：三个中间量（CN、EEQ 电荷、C6）对笛卡尔坐标的解析一阶导。
//
// # 符号约定
//
// 对一对原子 `(p, q)`，记
//
// ```text
// R  = |R_q − R_p|,        R̂ = (R_q − R_p)/R
// ∂R/∂R_p = −R̂            ∂R/∂R_q = +R̂
// ```
//
// 所以**任何**只通过 `R_pq` 依赖几何的成对量 `T(R)`，对两端的贡献都是
// `∓(∂T/∂R)·R̂`：大小相同、方向相反。这正是平移不变性
// （`Σ_a ∂CN_i/∂R_a = 0`）在本模块里自动成立的原因，也是
// `tests/da06_d4_derivs.rs` 里那条平移检查的来历。
//
// # 展平约定
//
// 张量返回成 `Vec<[f64; 3]>`（与验收测试的签名一致）：
//
// * [`cn_gradient`] / [`charge_gradient`]：`out[i*n + a] = ∂X_i/∂R_a`；
// * [`c6_gradient`]：`out[(i*n + j)*n + a] = ∂C6_ij/∂R_a`。
//
// 也就是说第 0 层下标 = 被求导的量，第 1 层下标 = 求导方向（原子），`[f64;3]` 是
// 那个原子的 x/y/z 分量。`n` 一律是 `indices.len()`。

/// D4 CN 的成对导数：`derivative[p][q] = ∂CN_p/∂R_pq`（取 `rc`、`kcn`、EN 因子与
/// [`coordination_numbers`] 完全一致的定义）。与 EEQ 的 CN 不同：**没有** 25 Bohr
/// 截断、**没有** EN 之外的其他截断，但有电负性因子。
///
/// `count = ½(1 + erf(u))`，`u = −kcn(R − rc)/rc`，故
/// `dcount/dR = −kcn/(rc√π)·e^{−u²}`。
#[allow(clippy::needless_range_loop)]
fn d4_cn_pair_derivatives(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
) -> Vec<Vec<f64>> {
    let n = indices.len();
    let mut derivative = vec![vec![0.0_f64; n]; n];
    let cn = &data.coordination_number;
    let root_pi = std::f64::consts::PI.sqrt();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let left = &data.reference[indices[i]];
            let right = &data.reference[indices[j]];
            let rc = (left.covalent_rad_2009_angstrom + right.covalent_rad_2009_angstrom)
                * cn.radius_scaling
                * data.constants.aatoau;
            if rc <= 0.0 {
                continue;
            }
            let r = distance(positions[i], positions[j]);
            let u = -cn.kcn * (r - rc) / rc;
            let delta = (left.pauling_en - right.pauling_en).abs();
            let factor = cn.k4 * (-((delta + cn.k5).powi(2)) / cn.k6).exp();
            derivative[i][j] = -factor * cn.kcn / (rc * root_pi) * (-(u * u)).exp();
        }
    }
    derivative
}

/// ① 的梯度：`out[i*n + a] = ∂CN_i/∂R_a`（返回长度 `n*n`）。
///
/// CN_i 只经由成对距离依赖几何，所以
///
/// ```text
/// ∂CN_i/∂R_i = Σ_{j≠i} (∂CN_i/∂R_ij)·(−R̂_ij)
/// ∂CN_i/∂R_j = (∂CN_i/∂R_ij)·(+R̂_ij)
/// ∂CN_i/∂R_a = 0                       (a ∉ {i, j})
/// ```
///
/// 于是每一行满足 `Σ_a ∂CN_i/∂R_a = 0`（平移不变），见验收测试。
pub fn cn_gradient(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    let pair = d4_cn_pair_derivatives(data, indices, positions);
    pair_derivative_jacobian(&pair, positions)
}

/// 把成对距离上的导数搬成笛卡尔 Jacobian。
///
/// 输入 `pair[p][q] = ∂F_p/∂R_pq`（`p == q` 或 `R = 0` 时忽略），输出
/// `out[p*n + a] = ∂F_p/∂R_a`。逐**有序**对 `(i, j)` 累积：
///
/// ```text
/// ∂F_i/∂R_i ∋ (∂F_i/∂R_ij)·(−R̂_ij),     ∂F_i/∂R_j ∋ (∂F_i/∂R_ij)·(+R̂_ij)
/// ```
///
/// 遍历有序对而不是 `i < j` 的好处：`F` 不必对称（EEQ 的 CN 也一样），且
/// `pair[j][i]` 自动给出 `∂F_j/∂R_ij`。这也是 `Σ_a ∂F_i/∂R_a = 0` 的来源：
/// 同一个 `d·R̂` 在一行里以 `−d·R̂` 与 `+d·R̂` 各出现一次。
#[allow(clippy::needless_range_loop)]
fn pair_derivative_jacobian(pair: &[Vec<f64>], positions: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = positions.len();
    let mut jacobian = vec![[0.0_f64; 3]; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r = distance(positions[i], positions[j]);
            if r <= 0.0 {
                continue;
            }
            let unit = [
                (positions[j][0] - positions[i][0]) / r,
                (positions[j][1] - positions[i][1]) / r,
                (positions[j][2] - positions[i][2]) / r,
            ];
            let d = pair[i][j];
            for alpha in 0..3 {
                // ∂R/∂R_i = −R̂，∂R/∂R_j = +R̂（R = |R_j − R_i|）。
                jacobian[i * n + i][alpha] -= d * unit[alpha];
                jacobian[i * n + j][alpha] += d * unit[alpha];
            }
        }
    }
    jacobian
}

/// ② 的梯度：`out[i*n + a] = ∂q_i/∂R_a`（返回长度 `n*n`）。
///
/// 直接对 `A q = b` 求全微分（`Q` 固定，故约束行右端为 0）：
///
/// ```text
/// A·dq = db − (dA)·q,      db_i = kcnchi_i · f'(CN_i) · ∂CN_i/∂R_a,
/// ```
///
/// 其中 `f(c) = c/√(c+ε)`，`f'(c) = (c/2 + ε)/(c+ε)^{3/2}`，`CN_i` 是**EEQ** 的
/// （软上限后的）CN —— 由 [`eeq_raw_cn`] 的原始 CN 乘 [`eeq_soft_cap_derivative`] 得到。
///
/// `A_ij = erf(R√γ)/R`（`γ = 1/(rad_i² + rad_j²)`）对**距离**的导数：
///
/// ```text
/// dA/dR = (2√γ/√π)·e^{−R²γ}/R − erf(R√γ)/R²
/// ```
///
/// 再按 `∂R/∂R_i = −R̂`、`∂R/∂R_j = +R̂` 搬到笛卡尔。对角项
/// `A_ii = η_i + √(2/π)/rad_i` 与几何无关，不产生 `dA`。
///
/// # Errors
///
/// 元素超出数据范围、EEQ 矩阵奇异。
#[allow(clippy::needless_range_loop)]
pub fn charge_gradient(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
    total_charge: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let n = indices.len();
    if positions.len() != n {
        return Err(format!(
            "d4: {} positions for {n} atoms in the charge gradient",
            positions.len()
        ));
    }
    let (matrix, rhs) = eeq_system(data, indices, positions, total_charge)?;
    let solution = solve_dense(matrix.clone(), rhs)?;
    let charges = &solution[..n];

    let (raw_cn, cn_pair) = eeq_raw_cn(data, indices, positions);
    let cn: Vec<f64> = raw_cn.iter().map(|value| eeq_soft_cap(*value)).collect();
    let mut radii = Vec::with_capacity(n);
    let mut kcnchi = Vec::with_capacity(n);
    for index in indices {
        radii.push(eeq_radius(data, *index)?);
        let symbol = &data.elements[*index];
        let z = elements::atomic_number(symbol)
            .ok_or_else(|| format!("d4: unknown element `{symbol}`"))?;
        kcnchi.push(data.eeq.kcnchi[(z as usize) - 1]);
    }

    // 成对距离上的 dA/dR（`gamma` 只依赖元素，不依赖几何）。
    let mut dmatrix_dr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r2 = squared_distance(positions[i], positions[j]);
            if r2 <= 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let gamma = 1.0 / (radii[i] * radii[i] + radii[j] * radii[j]);
            let root_gamma = gamma.sqrt();
            dmatrix_dr[i][j] = 2.0 * root_gamma / std::f64::consts::PI.sqrt() * (-r2 * gamma).exp()
                / r
                - erf(r * root_gamma) / r2;
        }
    }

    // EEQ 原始 CN 的笛卡尔 Jacobian：`raw_jacobian[i*n + a][α] = ∂cn_i/∂R_{a,α}`。
    // **不能**直接用 `cn_pair[i][a]`：当 `a == i` 时 `cn_pair[i][i] = 0`，而
    // `∂cn_i/∂R_i = Σ_j (∂cn_i/∂R_ij)·(−R̂_ij)` 一般不为零（与 D4 CN 同理）。
    let raw_jacobian = pair_derivative_jacobian(&cn_pair, positions);

    // 每对 (i, j) 的单位向量 R̂_ij = (R_j − R_i)/R，供 dA 的搬运使用。
    let mut unit = vec![vec![[0.0_f64; 3]; n]; n];
    for i in 0..n {
        for j in 0..n {
            let r = distance(positions[i], positions[j]);
            if i == j || r <= 0.0 {
                continue;
            }
            unit[i][j] = [
                (positions[j][0] - positions[i][0]) / r,
                (positions[j][1] - positions[i][1]) / r,
                (positions[j][2] - positions[i][2]) / r,
            ];
        }
    }

    let mut gradient = vec![[0.0_f64; 3]; n * n];
    let mut perturbed = vec![0.0_f64; n + 1];
    for a in 0..n {
        for alpha in 0..3 {
            for i in 0..n {
                // db_i/dR_{a,α}：链式 raw CN → 软上限 → f(c) = c/√(c+ε)。
                let cap = eeq_soft_cap_derivative(raw_cn[i]);
                let dcn = cap * raw_jacobian[i * n + a][alpha];
                let c = cn[i];
                let fprime = (0.5 * c + CN_REGULARIZER) / (c + CN_REGULARIZER).powf(1.5);
                let mut value = kcnchi[i] * fprime * dcn;
                // −(dA·q)_i：行 i 上每个 j ≠ i 的 ∂A_ij/∂R_{a,α}·q_j。
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let d = dmatrix_dr[i][j];
                    // R = |R_j − R_i| ⇒ ∂R/∂R_i = −R̂，∂R/∂R_j = +R̂。
                    let direction = if a == i {
                        -d * unit[i][j][alpha]
                    } else if a == j {
                        d * unit[i][j][alpha]
                    } else {
                        continue;
                    };
                    value -= direction * charges[j];
                }
                perturbed[i] = value;
            }
            // Σq = Q 固定 ⇒ 约束行右端为 0。
            perturbed[n] = 0.0;
            let dcharges = solve_dense(matrix.clone(), perturbed.clone())?;
            for i in 0..n {
                gradient[i * n + a][alpha] = dcharges[i];
            }
        }
    }
    Ok(gradient)
}

/// ζ 对**电荷**的导数（`q` 这里已经是 `q_i + zeff_i`，对 `q_i` 的导数相同）：
///
/// ```text
/// q < 0:  ζ = e^a            ⇒ dζ/dq = 0
/// q ≥ 0:  u = c(1 − q_ref/q)
///         dζ/dq = −ζ·a·e^u·c·q_ref/q²
/// ```
///
/// 照抄 [`zeta`] 的分支：`q < 0` 时 ζ 是常数。`q → 0⁺` 时 `e^u` 指数级地趋于 0，
/// 而 `q_ref/q²` 只代数发散，乘积仍趋于 0 —— 所以 `q = 0` 处取 0 是连续延拓。
fn zeta_charge_derivative(a: f64, c: f64, q_ref: f64, q: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    let u = c * (1.0 - q_ref / q);
    let value = zeta(a, c, q_ref, q);
    -value * a * u.exp() * c * q_ref / (q * q)
}

/// ③ 的梯度：`out[(i*n + j)*n + a] = ∂C6_ij/∂R_a`（返回长度 `n*n*n`）。
///
/// `C6` 对几何的依赖**只**经由参考权重 `w_r(i)`，而权重只经由 `CN_i` 与 `q_i`：
///
/// ```text
/// ∂C6_ij/∂R_a = Σ_r (∂C6_ij/∂w_r(i))·[∂w_r(i)/∂CN_i·∂CN_i/∂R_a + ∂w_r(i)/∂q_i·∂q_i/∂R_a]
///             + Σ_s (∂C6_ij/∂w_s(j))·[∂w_s(j)/∂CN_j·∂CN_j/∂R_a + ∂w_s(j)/∂q_j·∂q_j/∂R_a]
/// ```
///
/// 其中 `∂C6_ij/∂w_r(i) = Σ_s w_s(j)·(3/π)·Σ_k trapzd_k α_iw[r][k]α_jw[s][k]`
/// （`∂C6/∂w_s(j)` 对称），`∂CN/∂R` 与 `∂q/∂R` 来自 [`cn_gradient`] 与
/// [`charge_gradient`]。
///
/// 权重的两项导数：
///
/// ```text
/// w_r = (e_r/Z)·ζ_r,   e_r = Σ_{m=1..ngw} exp(−m·wf·(CN − CN_ref_r)²)
/// de_r/dCN = −2·wf·(CN − CN_ref_r)·Σ_m m·exp(−m·wf·(CN − CN_ref_r)²)
/// ∂w_r/∂CN = [de_r/dCN·Z − e_r·dZ/dCN]/Z² · ζ_r
/// ∂w_r/∂q  = (e_r/Z)·dζ_r/dq                      （见 [`zeta_charge_derivative`]）
/// ```
///
/// **这是"冻结权重"导数**：权重取未扰动几何上的值（`weights` 参数），链式到
/// `CN`/`q` 后就是复合映射 `R → CN, q → w → C6` 的全导数；但 `C6` 本身没有、
/// 也不该有对几何的直接依赖。验收测试的有限差分参考由 `properties` 在**扰动后**
/// 的几何上重算 `CN`/`q`/权重再算 `c6_matrix`，两者相等即说明这条链没漏项。
///
/// # Errors
///
/// 元素超范围、EEQ 矩阵奇异、`weights`/`positions` 与 `indices` 长度不符。
#[allow(clippy::needless_range_loop)]
pub fn c6_gradient(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
    total_charge: f64,
    weights: &[Vec<f64>],
) -> Result<Vec<[f64; 3]>, String> {
    let n = indices.len();
    if weights.len() != n {
        return Err(format!(
            "d4: {} weight rows for {n} atoms in the C6 gradient",
            weights.len()
        ));
    }
    if positions.len() != n {
        return Err(format!(
            "d4: {} positions for {n} atoms in the C6 gradient",
            positions.len()
        ));
    }
    let grid = &data.frequency_grid;
    if grid.hartree.len() != grid.trapzd_weights.len() {
        return Err("d4: frequency grid and trapzd weights differ in length".into());
    }

    let cn = coordination_numbers(data, indices, positions);
    let charges = eeq_charges(data, indices, positions, total_charge)?;
    let cn_jacobian = cn_gradient(data, indices, positions);
    let charge_jacobian = charge_gradient(data, indices, positions, total_charge)?;

    // ∂w_r(i)/∂CN_i 与 ∂w_r(i)/∂q_i（逐原子、逐参考体系）。
    let mut weight_by_cn = Vec::with_capacity(n);
    let mut weight_by_charge = Vec::with_capacity(n);
    for (atom, index) in indices.iter().enumerate() {
        let element = &data.reference[*index];
        let counts: Vec<f64> = element
            .reference_systems
            .iter()
            .map(|system| system.refcovcn)
            .collect();
        let ngw = reference_weight_counts(&counts);
        let hardness = element.chemical_hardness * data.constants.gc;
        let mut exponential = vec![0.0_f64; counts.len()];
        let mut derivative = vec![0.0_f64; counts.len()];
        let mut norm = 0.0_f64;
        let mut norm_derivative = 0.0_f64;
        for (r, system) in element.reference_systems.iter().enumerate() {
            let delta = cn[atom] - system.refcovcn;
            let mut sum = 0.0_f64;
            let mut dsum = 0.0_f64;
            for igw in 1..=ngw[r] {
                let order = igw as f64;
                let term = (-order * data.constants.wf * delta * delta).exp();
                sum += term;
                dsum -= 2.0 * order * data.constants.wf * delta * term;
            }
            exponential[r] = sum;
            derivative[r] = dsum;
            norm += sum;
            norm_derivative += dsum;
        }
        let inverse = if norm.abs() > 0.0 { 1.0 / norm } else { 0.0 };
        let mut by_cn = Vec::with_capacity(counts.len());
        let mut by_charge = Vec::with_capacity(counts.len());
        for (r, system) in element.reference_systems.iter().enumerate() {
            let charge = charges[atom] + element.zeff;
            let scale = zeta(
                data.constants.ga,
                hardness,
                system.clsq + element.zeff,
                charge,
            );
            by_cn.push(
                (derivative[r] * inverse - exponential[r] * norm_derivative * inverse * inverse)
                    * scale,
            );
            by_charge.push(
                exponential[r]
                    * inverse
                    * zeta_charge_derivative(
                        data.constants.ga,
                        hardness,
                        system.clsq + element.zeff,
                        charge,
                    ),
            );
        }
        weight_by_cn.push(by_cn);
        weight_by_charge.push(by_charge);
    }

    let mut gradient = vec![[0.0_f64; 3]; n * n * n];
    // ∂w_r(i)/∂R_a：逐原子 (i)、逐参考体系 (r)、逐原子方向 (a)。
    let mut weight_derivative = Vec::with_capacity(n);
    for i in 0..n {
        let mut per_reference = Vec::with_capacity(weights[i].len());
        for r in 0..weights[i].len() {
            let mut per_direction = Vec::with_capacity(n);
            for a in 0..n {
                let base = i * n + a;
                let mut value = [0.0_f64; 3];
                for alpha in 0..3 {
                    value[alpha] = weight_by_cn[i][r] * cn_jacobian[base][alpha]
                        + weight_by_charge[i][r] * charge_jacobian[base][alpha];
                }
                per_direction.push(value);
            }
            per_reference.push(per_direction);
        }
        weight_derivative.push(per_reference);
    }

    for i in 0..n {
        let left = &data.reference[indices[i]];
        for j in 0..n {
            let right = &data.reference[indices[j]];
            // ∂C6_ij/∂w_r(i) 与 ∂C6_ij/∂w_s(j)。
            let mut by_weight_i = vec![0.0_f64; weights[i].len()];
            let mut by_weight_j = vec![0.0_f64; weights[j].len()];
            for (r, weight_r) in weights[i].iter().enumerate() {
                let alpha_r = &left.reference_systems[r].alpha_iw_eeq;
                for (s, weight_s) in weights[j].iter().enumerate() {
                    let alpha_s = &right.reference_systems[s].alpha_iw_eeq;
                    let mut integral = 0.0;
                    for (point, grid_weight) in grid.trapzd_weights.iter().enumerate() {
                        integral += grid_weight * alpha_r[point] * alpha_s[point];
                    }
                    let term = data.constants.thopi * integral;
                    by_weight_i[r] += weight_s * term;
                    by_weight_j[s] += weight_r * term;
                }
            }
            for a in 0..n {
                let mut value = [0.0_f64; 3];
                for alpha in 0..3 {
                    let mut total = 0.0;
                    for r in 0..by_weight_i.len() {
                        total += by_weight_i[r] * weight_derivative[i][r][a][alpha];
                    }
                    for s in 0..by_weight_j.len() {
                        total += by_weight_j[s] * weight_derivative[j][s][a][alpha];
                    }
                    value[alpha] = total;
                }
                gradient[(i * n + j) * n + a] = value;
            }
        }
    }
    Ok(gradient)
}
