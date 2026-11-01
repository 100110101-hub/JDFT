//! SV-03 第五半：**SMD 参数化半径**（Marenich–Cramer–Truhlar 2009, eq. 16）。
//!
//! SMD 有两套半径，混用会静默错数：
//!
//! * **参数化半径**（本模块）：逐元素覆盖表 → 基础表（`radii.VDW`）；O 走
//!   `R = 1.52`（`α ≥ 0.43`）或 `R = 1.52 + 1.8·(0.43 − α)`。静电项用它。
//! * **CDS vdW 半径**（[`crate::smd_surface`]）：Bondi/Mantina 表 + 面积约定里的 0.4 Å。
//!   CDS 项用它。
//!
//! 两套数据在同一个 `kind: smd-atomic` 文件里并列（`parameterised_radii` 与
//! `cds.vdw_radii_angstrom`），各有名字与出处；测试里有一条断言专门钉住"C 1.85 ≠ 1.70、
//! N 1.89 ≠ 1.55"。
#![deny(unsafe_code)]

use qwb_engine_api::{EngineError, Molecule};
use qwb_registry::smd_atomic::SmdAtomic;

/// 每个原子的 SMD 参数化半径（Å），顺序与 `molecule.atoms` 一致。
///
/// `alpha` 是溶剂的 Abraham 氢键酸性 `α`（`solvent/<name>.yaml` 的 `smd.abraham_alpha`）。
///
/// # Errors
///
/// 元素既不在覆盖表、也不在基础表 → [`EngineError::Data`]。
pub fn smd_parameterised_radii(
    data: &SmdAtomic,
    molecule: &Molecule,
    alpha: f64,
) -> Result<Vec<f64>, EngineError> {
    if !alpha.is_finite() {
        return Err(EngineError::InvalidInput(format!(
            "smd: the solvent Abraham alpha must be finite, got {alpha}"
        )));
    }
    let rule = &data.parameterised_radii;
    let oxygen = &rule.oxygen;
    let oxygen_radius = if alpha >= oxygen.threshold_alpha {
        oxygen.radius_high_angstrom
    } else {
        oxygen.radius_high_angstrom + oxygen.slope_angstrom * (oxygen.offset_alpha - alpha)
    };
    let mut radii = Vec::with_capacity(molecule.n_atoms());
    for atom in &molecule.atoms {
        let radius = if atom.symbol == "O" {
            Some(oxygen_radius)
        } else {
            rule.overrides_angstrom
                .get(&atom.symbol)
                .copied()
                .or_else(|| rule.base_angstrom.get(&atom.symbol).copied())
        };
        let radius = radius.ok_or_else(|| {
            EngineError::Data(format!(
                "smd: no parameterised radius for element `{}`",
                atom.symbol
            ))
        })?;
        radii.push(radius);
    }
    Ok(radii)
}
