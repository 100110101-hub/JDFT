//! SMD 的 **CDS 项**（cavity dispersion solvent structure）：`E_CDS = Σ_i σ_i·A_i + CSSIGM·A_total`。
//!
//! 数据在 `qwb-data/solvent/smd_atomic.yaml`（`kind: smd-atomic`），已对 PySCF 2.14.0 的
//! `libsolvent` 验证：水相 σ 1.6e-5 cal/(mol·Å²)、五个非水溶剂 E_CDS 3.8e-7 kcal/mol。
//!
//! # 两件"看起来能用、实际会错"的事
//!
//! 1. **面积不是几何球面**：`A_i = 4π(R_i + 0.4 Å)²`。那个 `0.4` 是从官方能量**反推**出来的
//!    （用 `4πR²` 时 σ 逐元素偏大 1.48~1.78 倍），所以它作为数据字段存在，不写死在这里。
//! 2. **非水的 σ 不是查表**：`σ_i = σ_N,i·SOLN + σ_A,i·SOLA + σ_B,i·SOLB`，其中
//!    `SOLA/SOLB` 是溶剂的 Abraham α/β、`SOLN` 是折射率；另有与元素无关的
//!    `CSSIGM = c1·SOLG + c2·SOLB² + c3·SOLC² + c4·SOLH²`。
//!
//! 本模块**只做表达式**：面积由调用方给（单原子时就是 `4π(R+0.4)²`，多原子时来自表面分解）。
//! 这样"σ 与拼装规则是否正确"可以先用单原子单独验证，与表面积分解算法解耦。
#![deny(unsafe_code)]

use qwb_registry::smd_atomic::SmdAtomic;

/// 非水 CDS 需要的六个溶剂描述符（与上游 `mnsol.F` 的顺序一致）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SmdDescriptors {
    /// `SOLA`：Abraham 氢键酸性 α。
    pub sola: f64,
    /// `SOLB`：Abraham 氢键碱性 β。
    pub solb: f64,
    /// `SOLC`：碳芳香性。
    pub solc: f64,
    /// `SOLG`：宏观表面张力（cal/(mol·Å²)）。
    pub solg: f64,
    /// `SOLH`：电负性卤素性。
    pub solh: f64,
    /// `SOLN`：折射率。
    pub soln: f64,
}

impl SmdDescriptors {
    /// 由六个描述符构造（顺序与上游 `mnsol.F` 一致：SOLA/SOLB/SOLC/SOLG/SOLH/SOLN）。
    #[must_use]
    pub fn new(sola: f64, solb: f64, solc: f64, solg: f64, solh: f64, soln: f64) -> Self {
        Self {
            sola,
            solb,
            solc,
            solg,
            solh,
            soln,
        }
    }
}

/// 单元素的 CDS 表面张力 σ（cal/(mol·Å²)）。
///
/// # Errors
///
/// 元素不在数据覆盖范围内。
pub fn cds_sigma(
    data: &SmdAtomic,
    symbol: &str,
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<f64, String> {
    let index = data
        .elements
        .iter()
        .position(|candidate| candidate == symbol)
        .ok_or_else(|| {
            format!(
                "smd: element `{symbol}` outside the data range ({} elements)",
                data.elements.len()
            )
        })?;
    if water {
        // 水相：直接查表（`CSSIGM = 0`，上游 `SMD_CDS_AQ` 里没有常数项）。
        return data
            .cds
            .water
            .sigma_cal_per_mol_angstrom2
            .get(symbol)
            .copied()
            .ok_or_else(|| format!("smd: no aqueous sigma for `{symbol}`"));
    }
    let descriptors = descriptors.ok_or_else(|| {
        "smd: nonaqueous CDS needs the solvent Abraham descriptors \
         (SOLA/SOLB/SOLC/SOLG/SOLH/SOLN)"
            .to_string()
    })?;
    let nonaqueous = &data.cds.nonaqueous;
    Ok(nonaqueous.sigma_n[index] * descriptors.soln
        + nonaqueous.sigma_a[index] * descriptors.sola
        + nonaqueous.sigma_b[index] * descriptors.solb)
}

/// 与元素无关的常数表面张力 `CSSIGM`（cal/(mol·Å²)）；水相恒为 0。
#[must_use]
pub fn cds_constant_term(
    data: &SmdAtomic,
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> f64 {
    if water {
        return 0.0;
    }
    let Some(d) = descriptors else {
        return 0.0;
    };
    let coefficients = &data.cds.nonaqueous.sigma_mol;
    coefficients[0] * d.solg
        + coefficients[1] * d.solb * d.solb
        + coefficients[2] * d.solc * d.solc
        + coefficients[3] * d.solh * d.solh
}

/// CDS 项的能量（kcal/mol）。
///
/// `areas_angstrom2` 是与 `symbols` 同序的**每原子表面积**；`water` 决定用哪一套 σ。
///
/// # Errors
///
/// 符号数与面积数不一致、元素超范围、非水溶剂缺描述符。
pub fn cds_energy_kcal(
    data: &SmdAtomic,
    symbols: &[&str],
    areas_angstrom2: &[f64],
    water: bool,
    descriptors: Option<&SmdDescriptors>,
) -> Result<f64, String> {
    if symbols.len() != areas_angstrom2.len() {
        return Err(format!(
            "smd: {} symbols for {} surface areas",
            symbols.len(),
            areas_angstrom2.len()
        ));
    }
    if !water && descriptors.is_none() {
        return Err("smd: nonaqueous CDS needs the solvent Abraham descriptors".to_string());
    }
    let mut energy = 0.0;
    let mut total_area = 0.0;
    for (symbol, area) in symbols.iter().zip(areas_angstrom2) {
        if !area.is_finite() || *area < 0.0 {
            return Err(format!(
                "smd: surface area for `{symbol}` must be non-negative"
            ));
        }
        let sigma = cds_sigma(data, symbol, water, descriptors)?;
        energy += sigma * area;
        total_area += area;
    }
    energy += cds_constant_term(data, water, descriptors) * total_area;
    // σ 是 cal/(mol·Å²)、面积是 Å² → cal/mol；返回 kcal/mol。
    Ok(energy / 1000.0)
}

/// 单原子的 CDS 面积 `4π(R + offset)²`（Å²）—— 单原子极限的闭式参考。
///
/// 多原子体系的面积必须来自表面分解（切换函数、球面点集），不能用这个式子相加：
/// 那样等于把重叠部分重复计两次。
///
/// # Errors
///
/// 元素不在数据覆盖范围内。
pub fn single_atom_area_angstrom2(data: &SmdAtomic, symbol: &str) -> Result<f64, String> {
    let radius = data
        .cds
        .vdw_radii_angstrom
        .get(symbol)
        .copied()
        .ok_or_else(|| format!("smd: no CDS vdW radius for `{symbol}`"))?;
    let effective = radius + data.cds.area_offset_angstrom;
    Ok(4.0 * std::f64::consts::PI * effective * effective)
}
