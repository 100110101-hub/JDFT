//! D3(BJ) 色散校正（FX-04）：作为可加项的能量与**解析梯度**。
//!
//! # 公式（与 `qwb-data/dispersion/d3bj.yaml` 的字段一一对应）
//!
//! ```text
//! CN_i = Σ_{j≠i, r<cutoff} 1/(1 + exp(−kcn·(rc_i+rc_j)/r − kcn))      rc = scale·r_Å·bohr
//! w_i(k) ∝ exp(−wf·(CN_ref(k) − CN_i)²)                              （log-sum-exp 归一化）
//! C6_ij = Σ_{k,l} w_i(k)·w_j(l)·C6_ref(k, l; 高Z-低Z)
//! (C8/C6)_ij = 3·r4r2_i·r4r2_j
//! R0 = a1·sqrt(C8/C6) + a2
//! E = Σ_{i<j} −C6_ij·[ s6/(r⁶ + R0⁶) + s8·(C8/C6)/(r⁸ + R0⁸) ]
//! ```
//!
//! # 两个约定（都写在数据文件的字段文档里，加载器也会核对）
//!
//! 1. `r4r2` 是**派生值** `sqrt(0.5·published·sqrt(Z))`，不是公布的 `<r⁴>/<r²>`；
//! 2. 参考块内**第一维是高 Z 元素**（`k` 属于 Z 较大的那个原子）。
//!
//! 这两条都是拿官方实现反解验证的（docs/choices.md 一百四十一），搞反任何一条都会让
//! 能量偏 2–16 倍。
//!
//! # 梯度
//!
//! 两条链：① 直接项 `dE/dr = −C6·dp/dr`；② CN 链式项
//! `dE/dCN_i = −Σ_j p_ij·(∂C6_ij/∂CN_i)`，再乘 `dCN_i/dr_ij`（计数函数的解析导数）。
//! 判据是「与中心差分差 < 1e-6 Ha/Bohr」，实测 5e-9 量级（见 `fx04_dispersion.rs`）。

use qwb_registry::dispersion::{Dispersion, PAIR_C6_LEN, REFERENCE_POINTS};

/// 一个体系的色散能与其对坐标的解析梯度。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DispersionResult {
    /// 总色散能，Hartree（负值）。
    pub energy_hartree: f64,
    /// 每个原子上的梯度，Hartree/Bohr（与输入坐标同序）。
    pub gradient: Vec<[f64; 3]>,
    /// 逐原子对的能量（`(i, j, E_ij)`，i < j，只含截断内的对；仅供诊断）。
    pub pairs: Vec<(usize, usize, f64)>,
}

/// 计算 D3(BJ) 色散能（含解析梯度）。
///
/// # Errors
///
/// - 该泛函没有 BJ 参数（例如 M06-L / M06-2X：上游只有零阻尼）；
/// - 元素符号不在数据里，或体系里出现非有限坐标。
pub fn dispersion_energy(
    dispersion: &Dispersion,
    functional: &str,
    symbols: &[&str],
    positions_bohr: &[[f64; 3]],
) -> Result<DispersionResult, String> {
    let params = dispersion.damping(functional).ok_or_else(|| {
        let available: Vec<&str> = dispersion
            .damping_parameters
            .keys()
            .map(String::as_str)
            .collect();
        format!(
            "no D3(BJ) parameters for `{functional}` in `{}` (available: {}); \
             M06-L and M06-2X have no published BJ parameters, only zero damping",
            dispersion.name,
            available.join(", ")
        )
    })?;
    if symbols.len() != positions_bohr.len() {
        return Err(format!(
            "symbols ({}) and positions ({}) must have the same length",
            symbols.len(),
            positions_bohr.len()
        ));
    }
    let n = symbols.len();
    if positions_bohr
        .iter()
        .any(|p| p.iter().any(|c| !c.is_finite()))
    {
        return Err("positions must be finite".into());
    }

    // ── 每原子的静态量 ──
    let mut radii = Vec::with_capacity(n);
    let mut r4r2 = Vec::with_capacity(n);
    let mut reference_cn: Vec<&[f64]> = Vec::with_capacity(n);
    for symbol in symbols {
        radii.push(dispersion.covalent_radius_bohr(symbol)?);
        r4r2.push(
            *dispersion
                .r4r2
                .get(*symbol)
                .ok_or_else(|| format!("r4r2 has no entry for `{symbol}`"))?,
        );
        reference_cn.push(
            dispersion
                .reference_cn
                .get(*symbol)
                .ok_or_else(|| format!("reference_cn has no entry for `{symbol}`"))?
                .as_slice(),
        );
    }

    let kcn = dispersion.coordination_number.kcn;
    let cn_cutoff = dispersion.coordination_number.cutoff_bohr;
    let pair_cutoff = dispersion.dispersion.cutoff_bohr;
    let wf = dispersion.reference.wf;

    // ── CN（两端都要用：权重与 CN 链式项的几何导数）──
    let mut cn = vec![0.0f64; n];
    for i in 0..n {
        for j in i..n {
            let (r, _) = separation(positions_bohr, i, j);
            if r < 1e-6 || r > cn_cutoff {
                continue;
            }
            let count = counting_function(kcn, radii[i] + radii[j], r);
            cn[i] += count;
            if i != j {
                cn[j] += count;
            }
        }
    }

    // ── 权重与它的一阶导 ──
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut weight_derivs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, reference) in reference_cn.iter().enumerate() {
        let (w, dw) = reference_weights(reference, cn[i], wf);
        weights.push(w);
        weight_derivs.push(dw);
    }

    // ── 两点项 ──
    let mut energy = 0.0f64;
    let mut gradient = vec![[0.0f64; 3]; n];
    let mut dedcn = vec![0.0f64; n];
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let (r, unit) = separation(positions_bohr, i, j);
            if r < 1e-6 || r > pair_cutoff {
                continue;
            }
            let (c6, dc6_i, dc6_j) = pair_c6(
                dispersion,
                symbols[i],
                symbols[j],
                &weights[i],
                &weights[j],
                &weight_derivs[i],
                &weight_derivs[j],
            )?;
            let rrij = 3.0 * r4r2[i] * r4r2[j];
            let (p, dp) = damping_polynomial(params, rrij, r);
            energy -= c6 * p;
            // 直接项：E_ij = -C6·p → dE/dr = -C6·dp/dr，方向 j→i。
            let magnitude = -c6 * dp;
            for axis in 0..3 {
                gradient[i][axis] += magnitude * unit[axis];
                gradient[j][axis] -= magnitude * unit[axis];
            }
            // CN 链式项：dE/dCN = -p·dC6/dCN。
            dedcn[i] -= p * dc6_i;
            dedcn[j] -= p * dc6_j;
            pairs.push((i, j, -c6 * p));
        }
    }

    // ── CN 对坐标的导数的链式项 ──
    for i in 0..n {
        for j in (i + 1)..n {
            let (r, unit) = separation(positions_bohr, i, j);
            if r < 1e-6 || r > cn_cutoff {
                continue;
            }
            let count = counting_function(kcn, radii[i] + radii[j], r);
            // countf = σ(x)，x = kcn·(rc/r − 1) → dcountf/dr = −kcn·rc/r²·σ(1−σ)。
            let derivative = -kcn * (radii[i] + radii[j]) / (r * r) * count * (1.0 - count);
            let magnitude = (dedcn[i] + dedcn[j]) * derivative;
            for axis in 0..3 {
                gradient[i][axis] += magnitude * unit[axis];
                gradient[j][axis] -= magnitude * unit[axis];
            }
        }
    }

    Ok(DispersionResult {
        energy_hartree: energy,
        gradient,
        pairs,
    })
}

/// 两原子间距离与其单位向量（从 `j` 指向 `i`）。
fn separation(positions: &[[f64; 3]], i: usize, j: usize) -> (f64, [f64; 3]) {
    let dx = positions[i][0] - positions[j][0];
    let dy = positions[i][1] - positions[j][1];
    let dz = positions[i][2] - positions[j][2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < 1e-12 {
        return (0.0, [0.0; 3]);
    }
    (r, [dx / r, dy / r, dz / r])
}

/// 指数计数函数 `1/(1 + exp(−kcn·(rc/r − 1)))`（`ncoord.f90`）。
fn counting_function(kcn: f64, rc: f64, r: f64) -> f64 {
    let exponent = (-kcn * (rc / r - 1.0)).exp();
    1.0 / (1.0 + exponent)
}

/// 参考体系权重（log-sum-exp 归一化）与它对 CN 的一阶导（`model.f90`）。
fn reference_weights(reference_cn: &[f64], cn: f64, wf: f64) -> (Vec<f64>, Vec<f64>) {
    let dcn: Vec<f64> = reference_cn.iter().map(|value| value - cn).collect();
    let dmin = dcn
        .iter()
        .map(|value| value * value)
        .fold(f64::INFINITY, f64::min);
    let mut gw: Vec<f64> = dcn
        .iter()
        .map(|value| (-wf * (value * value - dmin)).exp())
        .collect();
    let norm: f64 = gw.iter().sum();
    for value in &mut gw {
        *value /= norm;
    }
    let slope: Vec<f64> = dcn.iter().map(|value| 2.0 * wf * value).collect();
    let mean: f64 = gw.iter().zip(slope.iter()).map(|(w, s)| w * s).sum::<f64>();
    let dgw: Vec<f64> = gw
        .iter()
        .zip(slope.iter())
        .map(|(w, s)| w * (s - mean))
        .collect();
    (gw, dgw)
}

/// 插值出 C6 与它对两个 CN 的一阶导。
///
/// 块内第一维是**高 Z 元素**（数据文件的字段文档与加载器的检查都钉住这一点）；
/// 两个元素的实际参考点数可能少于 7，所以用的是 7×7 块的左上角子块。
fn pair_c6(
    dispersion: &Dispersion,
    symbol_i: &str,
    symbol_j: &str,
    gw_i: &[f64],
    gw_j: &[f64],
    dgw_i: &[f64],
    dgw_j: &[f64],
) -> Result<(f64, f64, f64), String> {
    let block = dispersion.pair_c6(symbol_i, symbol_j)?;
    if block.len() != PAIR_C6_LEN {
        return Err(format!(
            "reference_c6 block for ({symbol_i}, {symbol_j}) has {} values, expected {PAIR_C6_LEN}",
            block.len()
        ));
    }
    // `pair_c6` 已经按「高 Z 在前」选好了块；这里判断 i 是不是高 Z 的那个，
    // 以决定块内第一维对应哪一组权重。
    let high_first = qwb_registry::elements::atomic_number(symbol_i)
        >= qwb_registry::elements::atomic_number(symbol_j);
    let mut c6 = 0.0;
    let mut dc6_i = 0.0;
    let mut dc6_j = 0.0;
    for (i_ref, w_i) in gw_i.iter().enumerate() {
        for (j_ref, w_j) in gw_j.iter().enumerate() {
            let offset = if high_first {
                i_ref + REFERENCE_POINTS * j_ref
            } else {
                j_ref + REFERENCE_POINTS * i_ref
            };
            let reference = block[offset];
            c6 += w_i * w_j * reference;
            dc6_i += dgw_i[i_ref] * w_j * reference;
            dc6_j += w_i * dgw_j[j_ref] * reference;
        }
    }
    Ok((c6, dc6_i, dc6_j))
}

/// 阻尼多项式 `p = s6/(r⁶+R0⁶) + s8·(C8/C6)/(r⁸+R0⁸)` 与它对 `r` 的导数。
fn damping_polynomial(
    params: &qwb_registry::dispersion::DampingParameters,
    rrij: f64,
    r: f64,
) -> (f64, f64) {
    let r0 = params.a1 * rrij.sqrt() + params.a2;
    let r0_6 = r0.powi(6);
    let r0_8 = r0.powi(8);
    let r2 = r * r;
    let r3 = r2 * r;
    let r5 = r3 * r2;
    let r6 = r3 * r3;
    let r7 = r6 * r;
    let r8 = r6 * r2;
    let t6 = 1.0 / (r6 + r0_6);
    let t8 = 1.0 / (r8 + r0_8);
    let p = params.s6 * t6 + params.s8 * rrij * t8;
    let dp = -params.s6 * 6.0 * r5 * t6 * t6 - params.s8 * rrij * 8.0 * r7 * t8 * t8;
    (p, dp)
}

#[cfg(test)]
mod tests {
    use super::{counting_function, reference_weights};

    #[test]
    fn counting_function_matches_the_upstream_form() {
        // rc/r = 1 → 0.5；rc/r 很大 → 1；rc/r 很小 → 0。
        assert!((counting_function(16.0, 2.0, 2.0) - 0.5).abs() < 1e-15);
        assert!(counting_function(16.0, 2.0, 0.5) > 0.999_999);
        assert!(counting_function(16.0, 2.0, 20.0) < 1e-6);
    }

    #[test]
    fn weights_are_normalized_and_centered() {
        let reference = [0.0, 0.9925, 1.9887];
        let (gw, dgw) = reference_weights(&reference, 0.9, 4.0);
        assert!((gw.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // 权重对 CN 的导数之和必须为 0（归一化的必然结果）。
        assert!(dgw.iter().sum::<f64>().abs() < 1e-12);
        // CN 落在中间参考点上时，权重集中在它附近。
        let (gw, _) = reference_weights(&reference, 0.9925, 4.0);
        assert!(gw[1] > 0.999, "{gw:?}");
    }
}
