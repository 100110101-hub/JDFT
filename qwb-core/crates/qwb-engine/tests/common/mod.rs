//! RP-04 的共用检查：**解析梯度 vs 数值梯度**（`< 1e-5 Ha/Bohr`）。
//!
//! RP-04 的判据原文是「每个 RP-01/02/03 的测试里都有『解析 vs 数值梯度 < 1e-5』断言
//! （不允许只有能量对）」。这条断言在三个文件里都要出现，所以比对逻辑放在这里一份 ——
//! 三份拷贝迟早会演变成三个不同的容差。
//!
//! # 为什么要在**离开驻点**的几何上比
//!
//! 在极小点/鞍点上两个梯度都接近零，"0 vs 0"的对照通过了却什么也没验证（本项目在 IRC 的
//! 切空间测试里已经踩过一次 0 vs 0）。所以调用方应当把几何沿路径/本征向量挪开一点再比。
#![allow(dead_code)]

use qwb_engine::gradient::{GradientOptions, numerical_gradient};
use qwb_engine::opt::OptOptions;
use qwb_engine::opt::analytic_energy_and_gradient;
use qwb_engine_api::EngineError;
use qwb_engine_api::Molecule;
use qwb_registry::Registry;

/// 解析梯度与数值梯度的**逐分量最大差**（Ha/Bohr），并把两个梯度的最大分量一并返回。
///
/// 返回 `(max_abs_difference, analytic_max, numeric_max)`：调用方要用后两个确认"这不是
/// 0 vs 0"。
///
/// # Errors
///
/// 任何一侧的梯度计算失败，或维度不一致。
pub fn compare_gradients(
    registry: &Registry,
    molecule: &Molecule,
    opt: &OptOptions,
    delta_bohr: f64,
) -> Result<(f64, f64, f64), EngineError> {
    let (_, analytic) = analytic_energy_and_gradient(registry, molecule, opt)?;
    let mut numeric_options =
        GradientOptions::new(opt.basis.clone(), opt.functional.clone(), opt.grid.clone());
    numeric_options.delta_bohr = delta_bohr;
    let numeric = numerical_gradient(registry, molecule, &numeric_options)?;
    if analytic.len() != numeric.len() {
        return Err(EngineError::Internal(format!(
            "gradient check: analytic has {} rows, numeric has {}",
            analytic.len(),
            numeric.len()
        )));
    }
    let mut worst = 0.0_f64;
    let mut analytic_max = 0.0_f64;
    let mut numeric_max = 0.0_f64;
    for (left, right) in analytic.iter().zip(&numeric) {
        for axis in 0..3 {
            worst = worst.max((left[axis] - right[axis]).abs());
            analytic_max = analytic_max.max(left[axis].abs());
            numeric_max = numeric_max.max(right[axis].abs());
        }
    }
    Ok((worst, analytic_max, numeric_max))
}

/// 摆一个「非 0 vs 0」的断言：两个梯度都必须足够大（> `floor`），否则对照没有信息量。
///
/// # Panics
///
/// 差值超过 `tolerance`，或任一梯度小于 `floor` 时 panic（带完整诊断）。
#[allow(clippy::too_many_arguments)]
pub fn assert_gradient_matches(
    label: &str,
    registry: &Registry,
    molecule: &Molecule,
    opt: &OptOptions,
    delta_bohr: f64,
    tolerance: f64,
    floor: f64,
) {
    let (difference, analytic_max, numeric_max) =
        compare_gradients(registry, molecule, opt, delta_bohr).expect("gradient check");
    println!(
        "{label}: 解析 vs 数值 最大差 {difference:.3e} Ha/Bohr（解析 max = {analytic_max:.3e}, \
         数值 max = {numeric_max:.3e}）"
    );
    assert!(
        analytic_max > floor && numeric_max > floor,
        "{label}: 两个梯度都太小（{analytic_max:.3e} / {numeric_max:.3e}），这个对照是 0 vs 0，没有信息量"
    );
    assert!(
        difference < tolerance,
        "{label}: 解析 vs 数值梯度差 {difference:.3e} Ha/Bohr 超过 {tolerance:.1e}"
    );
}
