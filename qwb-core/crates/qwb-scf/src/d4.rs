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

/// ② EEQ(2019) 电荷：`A q = x`，最后一行/列是 `Σq = Q` 的拉格朗日约束。
#[allow(clippy::needless_range_loop)] // 拉格朗日约束要同时写第 n 行与第 n 列
fn eeq_charges(
    data: &D4Dispersion,
    indices: &[usize],
    positions: &[[f64; 3]],
    total_charge: f64,
) -> Result<Vec<f64>, String> {
    let n = indices.len();
    let kcn = data.coordination_number.kcn;
    let scaling = data.coordination_number.radius_scaling * data.constants.aatoau;
    // EEQ 的 CN：**不带**电负性因子的纯误差函数 CN，截断 25 Bohr，软上限 8。
    let mut cn = vec![0.0_f64; n];
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
            cn[i] += 0.5 * (1.0 + erf(-kcn * (r - rcov) / rcov));
        }
    }
    for value in &mut cn {
        *value = (1.0 + EEQ_CN_MAX.exp()).ln() - (1.0 + (EEQ_CN_MAX - *value).exp()).ln();
    }

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

    let solution = solve_dense(matrix, rhs)?;
    Ok(solution[..n].to_vec())
}

/// ③ 每个原子的参考权重（CN 高斯 × 电荷缩放 ζ）。
fn reference_weights(
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
