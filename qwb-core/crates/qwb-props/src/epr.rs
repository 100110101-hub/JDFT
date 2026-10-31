//! ST-05：**各向同性超精细耦合（EPR/ENDOR 的 A 张量各向同性分量）**。
//!
//! # 物理
//!
//! 各向同性超精细耦合（Fermi 接触项）来自核位置上未配对电子密度：
//!
//! ```text
//! A_iso(K) = (8π/3) (μ0/4π) g_e μ_B g_K μ_N ρ_s(R_K)
//! ρ_s(R_K) = Σ_{μν} P^{α−β}_{μν} φ_μ(R_K) φ_ν(R_K)      （a.u.，即 a0^-3）
//! ```
//!
//! 两个容易错的地方，本模块都用可执行检查钉住：
//!
//! 1. **常数**。`(8π/3)(μ0/4π) g_e μ_B g_N μ_N / h` 这一串必须一次写对；用
//!    **氢原子 1s**（ρ_s = 1/π a0^-3，实验 1420.405751 MHz = 21 cm 线）当锚点，
//!    算出来 1422.4 MHz（差 0.14%，正是有限核质量/约化质量修正的量级）。差的不是常数。
//! 2. **符号与自旋约定**。`P^{α−β}` 是 α 减 β，不是总密度；H 原子的 1s 单电子给出正
//!    spin 密度，而 CH₃ 的 β 位质子是**负**的（自旋极化），符号由密度矩阵直接决定，
//!    代码里不做任何取绝对值。
//!
//! 单位：内部一律 a.u.，出口同时给 MHz 与 Gauss（`1 G = 2.802_49 MHz`，自由电子；
//! 实验 EPR 文献常用 Gauss，计算文献常用 MHz，两个都给出可以省掉换算错误）。
use qwb_engine_api::{EngineError, Molecule};
use qwb_linalg::Matrix;

/// 一个核的各向同性超精细耦合。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HyperfineCoupling {
    /// 核（原子）下标。
    pub nucleus_index: usize,
    /// 元素符号。
    pub symbol: String,
    /// 核位置上的自旋密度 `ρ_s = Σ P^{α−β}_{μν} φ_μ φ_ν`，单位 `a0^-3`。
    pub spin_density_au: f64,
    /// 各向同性耦合，MHz。
    pub a_iso_mhz: f64,
    /// 各向同性耦合，Gauss。
    pub a_iso_gauss: f64,
}

/// 核的 g 因子（自由核）。`g_N = μ_N / (nuclear magneton from the proton)`。
///
/// 只列本模块支持的核；其它元素返回 `None`（宁可报缺失，也不猜一个 g 因子 ——
/// 猜错就是"自算值冒充文献值"的典型形态）。
#[must_use]
pub fn nuclear_g_factor(symbol: &str) -> Option<f64> {
    Some(match symbol {
        // ¹H：g_N = 5.5856946893
        "H" => 5.585_694_689_3,
        // ¹³C（天然丰度领头同位素）：g_N = 1.4048236
        "C" => 1.404_823_6,
        // ¹⁴N：g_N = 0.4037607
        "N" => 0.403_760_7,
        // ¹⁶O：g_N = -0.757516（负 g，耦合符号会翻）
        "O" => -0.757_516,
        _ => return None,
    })
}

/// `a_iso[MHz] = C · g_N · ρ_s[a0^-3]` 里的 `C`。
///
/// 由**氢原子锚点**定标：`ρ_s(1s) = 1/π`、`g_H = 5.5856946893`、实验
/// `a_iso = 1420.405751 MHz`（21 cm 线）⇒
/// `C = 1420.405751 · π / 5.5856946893 = 798.8865344 MHz·a0³`。
///
/// # 为什么"由锚点定标"而不是直接从 SI 常数直算
///
/// 直算（`(8π/3)(μ0/4π) g_e μ_B g_N μ_N / h`，CODATA 2018）给 **1422.807579 MHz**，
/// 与 21 cm 线差 **+0.169%** —— 那是氢原子的有限核质量/约化质量修正，不是常数错。
/// 锚点定标让"常数组合"这一层不可出错：锚点本身就是一条实验事实，它一次覆盖了
/// `(8π/3)`、`μ0/4π`、`μ_B`、`μ_N`、`g_e`、`h` 六个量。直算值仍作为一条**测试**保留
/// （≤ 0.5% 一致），两者不一致就说明常数表或公式形状有一处写错。
const HYPERFINE_MHZ: f64 = 798.886_534_4;

/// `a_iso[MHz] = C·g_N·ρ_s`（`ρ_s` 单位 `a0^-3`）。
#[must_use]
pub fn a_iso_mhz(symbol: &str, spin_density_au: f64) -> Option<f64> {
    let g = nuclear_g_factor(symbol)?;
    Some(HYPERFINE_MHZ * g * spin_density_au)
}

/// 锚点定标的常数 `C = 798.8865344 MHz·a0³`（见 [`a_iso_mhz`] 的文档）。
#[must_use]
pub fn hyperfine_constant_mhz() -> f64 {
    HYPERFINE_MHZ
}

/// Gauss ↔ MHz（自由电子）：`1 G = 2.802_49 MHz`。
pub const MHZ_PER_GAUSS: f64 = 2.802_490_0;

/// 计算每个核的各向同性超精细耦合。
///
/// `spin_density` 必须是 `P^α − P^β`（开壳层）；`ao_values[k]` 是**第 k 个核位置上**的
/// AO 值（长度 = 基函数数）。AO 求值由调用方做：`qwb-scf` 依赖 `qwb-props` 的**上游**，
/// 这里不反向依赖它（`qwb-props` 只把 `qwb-scf` 当 dev-dependency 用于测试）。
///
/// # Errors
///
/// 维度不匹配、元素没有核 g 因子 → [`EngineError::Internal`] / [`EngineError::Unsupported`]。
pub fn isotropic_hyperfine(
    molecule: &Molecule,
    ao_values: &[Vec<f64>],
    spin_density: &Matrix,
) -> Result<Vec<HyperfineCoupling>, EngineError> {
    let n = spin_density.nrows();
    if spin_density.ncols() != n {
        return Err(EngineError::Internal(format!(
            "EPR: spin density is not square ({}x{})",
            spin_density.nrows(),
            spin_density.ncols()
        )));
    }
    if ao_values.len() != molecule.atoms.len() {
        return Err(EngineError::Internal(format!(
            "EPR: {} AO vectors for {} atoms",
            ao_values.len(),
            molecule.atoms.len()
        )));
    }
    if ao_values.iter().any(|values| values.len() != n) {
        return Err(EngineError::Internal(
            "EPR: an AO vector length does not match the spin density dimension".into(),
        ));
    }
    let mut couplings = Vec::with_capacity(molecule.atoms.len());
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let values = &ao_values[index];
        let mut density = 0.0;
        for mu in 0..values.len() {
            let mut row = 0.0;
            for nu in 0..values.len() {
                row += spin_density[(mu, nu)] * values[nu];
            }
            density += values[mu] * row;
        }
        let mhz = a_iso_mhz(&atom.symbol, density).ok_or_else(|| {
            EngineError::Unsupported(format!(
                "EPR: no nuclear g-factor for `{}` (ST-05 covers H/C/N/O only)",
                atom.symbol
            ))
        })?;
        couplings.push(HyperfineCoupling {
            nucleus_index: index,
            symbol: atom.symbol.clone(),
            spin_density_au: density,
            a_iso_mhz: mhz,
            a_iso_gauss: mhz / MHZ_PER_GAUSS,
        });
    }
    Ok(couplings)
}
