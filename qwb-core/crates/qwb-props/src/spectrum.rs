//! UV/Vis 谱（PR-04）：把 TDA/CIS 的跃迁（能量 + 振子强度）展宽成一条可画的曲线。
//!
//! # 展宽核与两条恒等式
//!
//! 用高斯核、按**半高半宽**（HWHM）参数化：
//!
//! ```text
//! g(E) = exp(−ln2·(E/HWHM)²)          （峰值 1，E = ±HWHM 处正好 0.5）
//! ∫_{-∞}^{∞} g dE = HWHM·√(π/ln2)
//! ```
//!
//! 于是采样曲线必须满足两条**恒等式级别**的性质（不是"看起来对"）：
//!
//! * **积分**：`Σ_i ε(E_i)·ΔE = Σ_k f_k·HWHM·√(π/ln2)`；
//! * **重心**：`Σ_i E_i·ε(E_i) / Σ_i ε(E_i) = Σ_k f_k·E_k / Σ_k f_k`
//!   （对称核的一阶矩精确等于中心位置）。
//!
//! 这两条是 `tests/spectrum.rs` 里钉住的判据；它们同时能抓住「核写错」「网格写错」
//! 「忘了乘 f」这三类最常见的错。
//!
//! # 展宽参数来自数据
//!
//! 默认值（HWHM、网格步长、窗口边距）来自 `qwb-data/numerics/numerics.yaml` 的
//! `spectrum_defaults`（K10 的**加性**扩展），不是代码里的常量 —— 见
//! [`SpectrumOptions::from_registry`]。

use qwb_engine_api::EngineError;
use qwb_registry::Registry;

/// `ln 2`：高斯按 HWHM 参数化时出现的常数。
const LN_2: f64 = std::f64::consts::LN_2;
/// 单位 HWHM 高斯的面积 √(π/ln2)。**由常数算**而不是抄一个数：1.7724... 是 √π，
/// 抄错了会让归一化判据差 20%（实测踩过）。
fn unit_area() -> f64 {
    (std::f64::consts::PI / LN_2).sqrt()
}

/// 展宽与网格设置（单位都是 eV）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SpectrumOptions {
    /// 高斯展宽的半高半宽。
    pub gaussian_hwhm_ev: f64,
    /// 能量网格步长。
    pub grid_step_ev: f64,
    /// 绘图窗口在最低/最高跃迁之外各留多少个 HWHM。
    pub padding_hwhm: f64,
}

impl Default for SpectrumOptions {
    fn default() -> Self {
        // 与 `qwb-data/numerics` 的 `spectrum_defaults` 一致；生产路径应当用
        // [`SpectrumOptions::from_registry`]，这条只是给不接数据的单元测试兜底。
        Self {
            gaussian_hwhm_ev: 0.4,
            grid_step_ev: 0.05,
            padding_hwhm: 6.0,
        }
    }
}

impl SpectrumOptions {
    /// 构造。
    #[must_use]
    pub fn new(gaussian_hwhm_ev: f64, grid_step_ev: f64, padding_hwhm: f64) -> Self {
        Self {
            gaussian_hwhm_ev,
            grid_step_ev,
            padding_hwhm,
        }
    }

    /// 从数据注册表读默认值（`numerics` 的 `spectrum_defaults`）。
    ///
    /// # Errors
    ///
    /// 数据里没有 `spectrum_defaults`（阶段二的旧数据）或取值非法。
    pub fn from_registry(registry: &Registry) -> Result<Self, EngineError> {
        let numerics = registry
            .get_numerics("numerics")
            .map_err(|err| EngineError::Data(format!("numerics data: {err}")))?;
        let defaults = numerics.spectrum_defaults.ok_or_else(|| {
            EngineError::Data(
                "numerics data has no `spectrum_defaults` block (PR-04 needs it; bump the data)"
                    .into(),
            )
        })?;
        let options = Self::new(
            defaults.gaussian_hwhm_ev,
            defaults.grid_step_ev,
            defaults.padding_hwhm,
        );
        options.validate()?;
        Ok(options)
    }

    /// 自检。
    ///
    /// # Errors
    ///
    /// HWHM / 步长 / 边距非法，或步长大于 HWHM（会漏掉峰）。
    pub fn validate(&self) -> Result<(), EngineError> {
        for (label, value) in [
            ("gaussian_hwhm_ev", self.gaussian_hwhm_ev),
            ("grid_step_ev", self.grid_step_ev),
            ("padding_hwhm", self.padding_hwhm),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(EngineError::InvalidInput(format!(
                    "spectrum {label} must be finite and > 0 (got {value})"
                )));
            }
        }
        if self.grid_step_ev > self.gaussian_hwhm_ev {
            return Err(EngineError::InvalidInput(format!(
                "spectrum grid_step_ev ({} eV) must not exceed gaussian_hwhm_ev ({} eV)",
                self.grid_step_ev, self.gaussian_hwhm_ev
            )));
        }
        Ok(())
    }
}

/// 一条展宽后的谱。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Spectrum {
    /// 高斯半高半宽，单位 eV（写进结果块，便于复现）。
    pub gaussian_hwhm_ev: f64,
    /// 网格步长，单位 eV。
    pub grid_step_ev: f64,
    /// `(能量 eV, 强度)` 采样点，能量递增。
    pub points: Vec<(f64, f64)>,
}

impl Spectrum {
    /// 解析归一化：整条曲线的面积（由恒等式给出，不依赖网格）。
    #[must_use]
    pub fn analytic_area(oscillator_sum: f64, gaussian_hwhm_ev: f64) -> f64 {
        oscillator_sum * gaussian_hwhm_ev * unit_area()
    }

    /// 采样曲线的数值面积（梯形）。
    #[must_use]
    pub fn area(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for pair in self.points.windows(2) {
            let (e0, i0) = pair[0];
            let (e1, i1) = pair[1];
            total += 0.5 * (i0 + i1) * (e1 - e0);
        }
        total
    }

    /// 一阶矩（强度重心的能量）。空谱返回 `None`。
    #[must_use]
    pub fn centroid_ev(&self) -> Option<f64> {
        let mut weight = 0.0;
        let mut first = 0.0;
        for (energy, intensity) in &self.points {
            weight += intensity;
            first += energy * intensity;
        }
        if weight.abs() < f64::EPSILON {
            None
        } else {
            Some(first / weight)
        }
    }

    /// 局部极大值的位置（能量 eV），按能量递增。
    #[must_use]
    pub fn peak_energies_ev(&self) -> Vec<f64> {
        let mut peaks = Vec::new();
        for window in self.points.windows(3) {
            let (e_left, i_left) = window[0];
            let (e_mid, i_mid) = window[1];
            let (_, i_right) = window[2];
            if i_mid > i_left && i_mid >= i_right {
                peaks.push(e_mid);
            }
            let _ = e_left;
        }
        peaks
    }

    /// CSV 文本（两列：`energy_ev,intensity`，LF 行尾，末尾带换行）。
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::from("energy_ev,intensity\n");
        for (energy, intensity) in &self.points {
            out.push_str(&format!("{energy:.6},{intensity:.8e}\n"));
        }
        out
    }
}

/// 单条高斯（峰值 1）。
#[must_use]
pub fn gaussian_kernel(offset_ev: f64, gaussian_hwhm_ev: f64) -> f64 {
    let scaled = offset_ev / gaussian_hwhm_ev;
    (-LN_2 * scaled * scaled).exp()
}

/// 把跃迁（`(能量 eV, 振子强度)`）展宽成采样谱。
///
/// # Errors
///
/// 没有跃迁、选项非法、出现非有限值。
pub fn broadened_spectrum(
    transitions: &[(f64, f64)],
    options: &SpectrumOptions,
) -> Result<Spectrum, EngineError> {
    options.validate()?;
    if transitions.is_empty() {
        return Err(EngineError::InvalidInput(
            "spectrum needs at least one transition".into(),
        ));
    }
    for (energy, strength) in transitions {
        if !(energy.is_finite() && strength.is_finite()) {
            return Err(EngineError::Internal(
                "spectrum input has a non-finite transition".into(),
            ));
        }
    }
    let lowest = transitions
        .iter()
        .map(|(energy, _)| *energy)
        .fold(f64::INFINITY, f64::min);
    let highest = transitions
        .iter()
        .map(|(energy, _)| *energy)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = options.padding_hwhm * options.gaussian_hwhm_ev;
    let start = lowest - padding;
    let end = highest + padding;
    let n_steps = ((end - start) / options.grid_step_ev).ceil().max(1.0) as usize;
    let mut points = Vec::with_capacity(n_steps + 1);
    for index in 0..=n_steps {
        let energy = start + index as f64 * options.grid_step_ev;
        let mut intensity = 0.0;
        for (centre, strength) in transitions {
            intensity += strength * gaussian_kernel(energy - centre, options.gaussian_hwhm_ev);
        }
        points.push((energy, intensity));
    }
    Ok(Spectrum {
        gaussian_hwhm_ev: options.gaussian_hwhm_ev,
        grid_step_ev: options.grid_step_ev,
        points,
    })
}
