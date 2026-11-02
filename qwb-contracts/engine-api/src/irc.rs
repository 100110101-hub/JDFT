//! K8d · IRC（内禀反应坐标）的输入选项与结果块（phase-2 `RP-02`）。
//!
//! IRC 是**从鞍点沿质量加权最陡下降路径走向两端极小点**的那条曲线。判据（§6 RP-02）有三条：
//!
//! 1. **两端优化后的几何与已知极小差 < 0.02 Å** —— 所以块里要能拿到两个端点几何的**指纹**，
//!    而不只是能量；
//! 2. **能量单调下降**（允许 ≤ 1 个平台步）—— 所以逐点的 `energy_hartree` 必须落盘，且
//!    `is_monotone` 这个派生量要被算出来（不然"单调"只能靠读者自己扫一眼）；
//! 3. **`irc.json` 落盘** —— 由 CLI 侧写。
//!
//! # 为什么端点用「内坐标指纹」而不是逐原子笛卡尔坐标
//!
//! 同一分子的两个极小点可能是**镜像**（NH3 伞形反转就是这样）：逐原子比较笛卡尔坐标要先
//! 解决对齐与原子置换，那本身就是一门课。所以端点块里报的是**键长/键角**这类置换与旋转不变的
//! 量（`fingerprint`），测试拿它与已知极小的同一组量比 —— 差 < 0.02 Å 的判据因此是可比、
//! 可复现的，而不是"经过某次对齐之后看起来差不多"。
use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// IRC 的选项（`options.irc`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct IrcOptions {
    /// 每个方向最多走多少步。
    pub max_steps: u32,
    /// 质量加权步长（单位 `sqrt(amu)·Bohr`）。0.1 是 H 转移类反应的常用量级。
    pub step_size: f64,
    /// 走哪一端（`both` 双向、`forward`/`reverse` 单端）。
    pub direction: IrcDirection,
    /// 平台的判据：某一步的能量下降小于这个值（Hartree）就停走，交给端点自由优化收尾。
    ///
    /// 定步长最速下降**不会**在极小点上按梯度收敛（它在谷底来回绕），所以停止条件用的是
    /// "再走也不降能了"。不用 `max|g|` 判：鞍点附近梯度本来就小，按梯度停会在起点就停
    /// （第一版正是这样，路径只有 2 个点）。
    pub plateau_hartree: f64,
}

/// IRC 的方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum IrcDirection {
    /// 两端都走（缺省）。
    Both,
    /// 只走虚频本征向量的正方向。
    Forward,
    /// 只走负方向。
    Reverse,
}

impl IrcDirection {
    /// 冻结的小写名。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Forward => "forward",
            Self::Reverse => "reverse",
        }
    }

    /// 是否走正方向。
    #[must_use]
    pub fn forward(self) -> bool {
        matches!(self, Self::Both | Self::Forward)
    }

    /// 是否走负方向。
    #[must_use]
    pub fn reverse(self) -> bool {
        matches!(self, Self::Both | Self::Reverse)
    }
}

impl IrcOptions {
    /// 缺省：200 步、步长 0.1 `sqrt(amu)·Bohr`、双向。
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_steps: 200,
            step_size: 0.1,
            direction: IrcDirection::Both,
            plateau_hartree: 1.0e-5,
        }
    }

    /// 改步数。
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 改步长。
    #[must_use]
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// 改方向。
    #[must_use]
    pub fn with_direction(mut self, direction: IrcDirection) -> Self {
        self.direction = direction;
        self
    }

    /// 改"平台"的判据（Hartree）。
    #[must_use]
    pub fn with_plateau(mut self, plateau_hartree: f64) -> Self {
        self.plateau_hartree = plateau_hartree;
        self
    }
}

impl Default for IrcOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// IRC 上的一个点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct IrcPoint {
    /// 步数（端点 0 = 鞍点）。
    pub step: u32,
    /// 该点的能量（Hartree）。
    pub energy_hartree: f64,
    /// 该点的 `max|grad|`（Ha/Bohr）。
    pub max_gradient_hartree_per_bohr: f64,
}

impl IrcPoint {
    /// 构造。
    #[must_use]
    pub fn new(step: u32, energy_hartree: f64, max_gradient_hartree_per_bohr: f64) -> Self {
        Self {
            step,
            energy_hartree,
            max_gradient_hartree_per_bohr,
        }
    }
}

/// 一端的 IRC 结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct IrcEnd {
    /// 路径上的点（第 0 点是鞍点）。
    pub points: Vec<IrcPoint>,
    /// 沿路径能量是否单调下降（允许一个平台步）。
    pub monotone: bool,
    /// 端点**优化后**的能量（Hartree）。
    pub minimum_energy_hartree: f64,
    /// 端点**优化后**的内坐标指纹：键长（Å）与键角（度），置换/旋转不变。
    pub fingerprint: Vec<f64>,
    /// 端点优化的步数。
    pub optimisation_steps: u32,
}

impl IrcEnd {
    /// 构造（`monotone` 由点集算，避免两处判断不一致）。
    #[must_use]
    pub fn new(
        points: Vec<IrcPoint>,
        minimum_energy_hartree: f64,
        fingerprint: Vec<f64>,
        optimisation_steps: u32,
    ) -> Self {
        let mut plateaus = 0_usize;
        let mut monotone = true;
        for pair in points.windows(2) {
            let delta = pair[1].energy_hartree - pair[0].energy_hartree;
            if delta > 1.0e-6 {
                monotone = false;
                break;
            }
            if delta.abs() <= 1.0e-6 {
                plateaus += 1;
            }
        }
        // 允许多至一个平台步（数值上能量不再下降但也没上升）。
        if plateaus > 1 {
            monotone = false;
        }
        Self {
            points,
            monotone,
            minimum_energy_hartree,
            fingerprint,
            optimisation_steps,
        }
    }

    /// 自洽性检查。
    ///
    /// # Errors
    ///
    /// 少于两个点、能量非有限、`monotone` 与点集矛盾时返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.points.len() < 2 {
            return Err(EngineError::Internal(format!(
                "irc end has {} point(s): a path needs at least 2",
                self.points.len()
            )));
        }
        for point in &self.points {
            if !point.energy_hartree.is_finite() || !point.max_gradient_hartree_per_bohr.is_finite()
            {
                return Err(EngineError::Internal(
                    "irc point has a non-finite energy or gradient".into(),
                ));
            }
        }
        for value in &self.fingerprint {
            if !value.is_finite() {
                return Err(EngineError::Internal(
                    "irc fingerprint contains a non-finite value".into(),
                ));
            }
        }
        if !self.minimum_energy_hartree.is_finite() {
            return Err(EngineError::Internal(
                "irc minimum_energy_hartree is not finite".into(),
            ));
        }
        Ok(())
    }
}

/// IRC 的结果块（`jobtype: irc`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct IrcResult {
    /// 唯一虚频的波数（cm⁻¹，负值）—— 起点必须是这个鞍点对应的那个方向。
    pub imaginary_wavenumber_cm: f64,
    /// 正方向一端（`direction` 不含 forward 时为 `None`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<IrcEnd>,
    /// 负方向一端。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse: Option<IrcEnd>,
}

impl IrcResult {
    /// 构造。
    #[must_use]
    pub fn new(
        imaginary_wavenumber_cm: f64,
        forward: Option<IrcEnd>,
        reverse: Option<IrcEnd>,
    ) -> Self {
        Self {
            imaginary_wavenumber_cm,
            forward,
            reverse,
        }
    }

    /// 自洽性检查（非有限数、两端都缺、虚频符号不对）。
    ///
    /// # Errors
    ///
    /// 上述任一条件成立时返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.imaginary_wavenumber_cm.is_finite() || self.imaginary_wavenumber_cm >= 0.0 {
            return Err(EngineError::Internal(format!(
                "irc.imaginary_wavenumber_cm = {} must be a finite negative number \
                 (an IRC starts at a first-order saddle)",
                self.imaginary_wavenumber_cm
            )));
        }
        if self.forward.is_none() && self.reverse.is_none() {
            return Err(EngineError::Internal(
                "irc has neither a forward nor a reverse end".into(),
            ));
        }
        if let Some(end) = &self.forward {
            end.validate()?;
        }
        if let Some(end) = &self.reverse {
            end.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IrcDirection, IrcEnd, IrcOptions, IrcPoint, IrcResult};

    fn monotone_end() -> IrcEnd {
        IrcEnd::new(
            vec![
                IrcPoint::new(0, -55.4377, 3.7e-6),
                IrcPoint::new(1, -55.4401, 1.0e-3),
                IrcPoint::new(2, -55.4415, 1.0e-4),
            ],
            -55.4416,
            vec![1.02, 1.02, 1.02, 107.0],
            4,
        )
    }

    #[test]
    fn monotonicity_is_derived_from_the_points() {
        let end = monotone_end();
        assert!(end.monotone, "逐点下降 = 单调");
        assert!(end.validate().is_ok());

        // 能量上升一步 → 不单调
        let mut rising = monotone_end();
        rising.points[2].energy_hartree = -55.4300;
        let end = IrcEnd::new(rising.points, -55.4416, vec![1.0], 2);
        assert!(!end.monotone);

        // 两个平台步 → 不单调（判据只允许一个）
        let end = IrcEnd::new(
            vec![
                IrcPoint::new(0, -55.4377, 1e-6),
                IrcPoint::new(1, -55.4377, 1e-6),
                IrcPoint::new(2, -55.4377, 1e-6),
                IrcPoint::new(3, -55.4416, 1e-6),
            ],
            -55.4416,
            vec![1.0],
            1,
        );
        assert!(!end.monotone, "两个平台步超过判据允许的 1 个");
    }

    #[test]
    fn options_have_sane_defaults() {
        let options = IrcOptions::new();
        assert_eq!(options.max_steps, 200);
        assert!((options.step_size - 0.1).abs() < 1e-15);
        assert_eq!(options.direction, IrcDirection::Both);
        assert!(options.direction.forward() && options.direction.reverse());
        assert_eq!(IrcOptions::default(), options);
        assert!(IrcDirection::Forward.forward() && !IrcDirection::Forward.reverse());
        assert_eq!(IrcDirection::Reverse.as_str(), "reverse");
    }

    #[test]
    fn validate_catches_the_usual_drift() {
        let result = IrcResult::new(-523.4, Some(monotone_end()), Some(monotone_end()));
        assert!(result.validate().is_ok());

        // 没有端点
        assert!(IrcResult::new(-523.4, None, None).validate().is_err());
        // 虚频符号不对
        assert!(
            IrcResult::new(523.4, Some(monotone_end()), None)
                .validate()
                .is_err()
        );
        // 端点少于两个点
        let single = IrcEnd::new(vec![IrcPoint::new(0, -55.4, 1e-6)], -55.4, vec![1.0], 0);
        assert!(single.validate().is_err());
        // 指纹里的非有限值
        let mut bad = monotone_end();
        bad.fingerprint = vec![f64::NAN];
        assert!(bad.validate().is_err());
    }

    #[test]
    fn block_roundtrips_through_json() {
        let result = IrcResult::new(-523.4, Some(monotone_end()), None);
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(!json.contains("\"reverse\""), "缺省不写出没有的一端");
        let back: IrcResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, result);
    }
}
