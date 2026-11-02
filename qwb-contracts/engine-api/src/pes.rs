//! K8b · PES 扫描的输入选项与结果块（phase-2 `RP-03`）。
//!
//! 扫描是**一组几何上的能量**，所以它既不是 `opt` 也不是 `sp`：输入要说明扫哪个内坐标，
//! 结果要说明每个点的坐标值与能量、以及"点与点之间有多陡"。
//!
//! # 三条约定
//!
//! 1. **坐标种类是闭集合**（距离/键角/二面角），单位写进字段名：`_angstrom` / `_degree`。
//! 2. **`relaxed` 是逐点的、不是全局的**：刚性扫描的每个点只是"把坐标挪过去再算能量"，
//!    弛豫扫描的每个点是"从上一个点出发做一次自由优化"。两者混在一个 `PesScan` 里会让人
//!    读错，所以整个块只有一个 `relaxed` 布尔，混着跑是调用方的错。
//! 3. **`max_adjacent_delta_hartree` 是派生量但**必须**报**：判据「扫描点能量连续」靠它，
//!    而它也是"扫描跨过了一个不连续（换解/掉进另一个 SCF 盆地）"的唯一信号 —— 藏在测试里
//!    不报，读结果的人就看不到。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 扫描的内坐标种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ScanKind {
    /// 两个原子之间的距离（Å）。
    Distance,
    /// 三个原子的键角（度）。
    Angle,
    /// 四个原子的二面角（度）。
    Dihedral,
}

impl ScanKind {
    /// 冻结的小写名（进 `input.yaml` / `result.json`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Distance => "distance",
            Self::Angle => "angle",
            Self::Dihedral => "dihedral",
        }
    }

    /// 这个种类需要几个原子。
    #[must_use]
    pub fn atom_count(self) -> usize {
        match self {
            Self::Distance => 2,
            Self::Angle => 3,
            Self::Dihedral => 4,
        }
    }

    /// 这个种类的单位。
    #[must_use]
    pub fn unit(self) -> &'static str {
        match self {
            Self::Distance => "angstrom",
            _ => "degree",
        }
    }
}

/// 扫描的坐标与范围（`options.scan`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ScanOptions {
    /// 内坐标种类。
    pub kind: ScanKind,
    /// 参与该内坐标的原子序号（0 基，按内坐标的定义顺序）。
    pub atoms: Vec<usize>,
    /// 起点（Å 或度）。
    pub start: f64,
    /// 终点（Å 或度）。
    pub stop: f64,
    /// 点数（含两端；`points >= 2`）。
    pub points: u32,
    /// `true` = 每个点做一次自由优化（弛豫扫描）；`false` = 刚性扫描。
    pub relax: bool,
}

impl ScanOptions {
    /// 构造（`relax` 缺省 `false` = 刚性扫描）。
    #[must_use]
    pub fn new(kind: ScanKind, atoms: Vec<usize>, start: f64, stop: f64, points: u32) -> Self {
        Self {
            kind,
            atoms,
            start,
            stop,
            points,
            relax: false,
        }
    }

    /// 改成弛豫扫描。
    #[must_use]
    pub fn with_relax(mut self, relax: bool) -> Self {
        self.relax = relax;
        self
    }

    /// 各扫描点的坐标值（线性等距，含两端）。
    ///
    /// # Errors
    ///
    /// 点数 < 2、端点为非有限值、原子数与种类不符、原子序号越界。
    pub fn values(&self, n_atoms: usize) -> Result<Vec<f64>, EngineError> {
        if self.points < 2 {
            return Err(EngineError::InvalidInput(format!(
                "options.scan.points = {} needs at least 2 points",
                self.points
            )));
        }
        if !self.start.is_finite() || !self.stop.is_finite() {
            return Err(EngineError::InvalidInput(
                "options.scan.start/stop must be finite".into(),
            ));
        }
        if self.atoms.len() != self.kind.atom_count() {
            return Err(EngineError::InvalidInput(format!(
                "options.scan.kind = `{}` needs {} atoms, got {}",
                self.kind.as_str(),
                self.kind.atom_count(),
                self.atoms.len()
            )));
        }
        for atom in &self.atoms {
            if *atom >= n_atoms {
                return Err(EngineError::InvalidInput(format!(
                    "options.scan.atoms refers to atom {atom} but the molecule has {n_atoms}"
                )));
            }
        }
        let steps = f64::from(self.points - 1);
        Ok((0..self.points)
            .map(|index| self.start + (self.stop - self.start) * f64::from(index) / steps)
            .collect())
    }
}

/// 扫描的一个点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PesPoint {
    /// 内坐标的值（Å 或度）。
    pub value: f64,
    /// 该点上的能量（Hartree）。
    pub energy_hartree: f64,
    /// 该点是否做过弛豫（刚性扫描恒 `false`）。
    pub optimised: bool,
    /// 弛豫用掉的迭代步数（刚性与未跑优化的点为 0）。
    pub iterations: u32,
}

impl PesPoint {
    /// 构造。
    #[must_use]
    pub fn new(value: f64, energy_hartree: f64, optimised: bool, iterations: u32) -> Self {
        Self {
            value,
            energy_hartree,
            optimised,
            iterations,
        }
    }
}

/// 扫描结果块（`jobtype: scan`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PesScan {
    /// 内坐标种类的小写名。
    pub kind: String,
    /// 单位（`angstrom` / `degree`）。
    pub unit: String,
    /// 参与内坐标的原子序号。
    pub atoms: Vec<usize>,
    /// 整个扫描是否是弛豫扫描。
    pub relaxed: bool,
    /// 各点（按坐标值升序）。
    pub points: Vec<PesPoint>,
    /// 相邻点能量差的绝对值最大值（Hartree）——「扫描点能量连续」的判据。
    pub max_adjacent_delta_hartree: f64,
}

impl PesScan {
    /// 由点集构造（`max_adjacent_delta_hartree` 在这里算，避免两处求和顺序不同）。
    #[must_use]
    pub fn new(
        kind: impl Into<String>,
        unit: impl Into<String>,
        atoms: Vec<usize>,
        relaxed: bool,
        points: Vec<PesPoint>,
    ) -> Self {
        let max_adjacent = points
            .windows(2)
            .map(|pair| (pair[1].energy_hartree - pair[0].energy_hartree).abs())
            .fold(0.0_f64, f64::max);
        Self {
            kind: kind.into(),
            unit: unit.into(),
            atoms,
            relaxed,
            points,
            max_adjacent_delta_hartree: max_adjacent,
        }
    }

    /// 能量最低的点（同值时取先出现的）。
    #[must_use]
    pub fn minimum(&self) -> Option<&PesPoint> {
        self.points
            .iter()
            .fold(None::<&PesPoint>, |best, point| match best {
                Some(current) if current.energy_hartree <= point.energy_hartree => Some(current),
                _ => Some(point),
            })
    }

    /// 自洽性检查（写 `result.json` / `pes.json` 之前跑）。
    ///
    /// # Errors
    ///
    /// 点数 < 2、值非单调、能量非有限、`optimised` 与块的 `relaxed` 矛盾、
    /// 或 `max_adjacent_delta_hartree` 与实际点集不一致时返回 [`EngineError::Internal`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.points.len() < 2 {
            return Err(EngineError::Internal(format!(
                "pes_scan has {} point(s): a scan needs at least 2",
                self.points.len()
            )));
        }
        for pair in self.points.windows(2) {
            if pair[1].value.partial_cmp(&pair[0].value) != Some(std::cmp::Ordering::Greater) {
                return Err(EngineError::Internal(
                    "pes_scan points are not strictly increasing in the scan coordinate".into(),
                ));
            }
        }
        for point in &self.points {
            if !point.value.is_finite() || !point.energy_hartree.is_finite() {
                return Err(EngineError::Internal(
                    "pes_scan point has a non-finite value or energy".into(),
                ));
            }
            if !self.relaxed && point.optimised {
                return Err(EngineError::Internal(
                    "pes_scan: a rigid scan cannot contain optimised points".into(),
                ));
            }
        }
        let recomputed = PesScan::new(
            self.kind.clone(),
            self.unit.clone(),
            self.atoms.clone(),
            self.relaxed,
            self.points.clone(),
        )
        .max_adjacent_delta_hartree;
        if (self.max_adjacent_delta_hartree - recomputed).abs() > 1.0e-12 {
            return Err(EngineError::Internal(format!(
                "pes_scan.max_adjacent_delta_hartree = {} contradicts the point set ({recomputed})",
                self.max_adjacent_delta_hartree
            )));
        }
        if !self.max_adjacent_delta_hartree.is_finite() {
            return Err(EngineError::Internal(
                "pes_scan.max_adjacent_delta_hartree is not finite".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PesPoint, PesScan, ScanKind, ScanOptions};

    fn options() -> ScanOptions {
        ScanOptions::new(ScanKind::Distance, vec![0, 1], 0.8, 1.2, 5)
    }

    #[test]
    fn scan_values_are_inclusive_and_linear() {
        let values = options().values(3).expect("values");
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.8).abs() < 1e-15);
        assert!((values[4] - 1.2).abs() < 1e-15);
        assert!((values[2] - 1.0).abs() < 1e-15);
        assert!((values[1] - values[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn scan_options_reject_bad_ranges_and_atoms() {
        let mut bad = options();
        bad.points = 1;
        assert!(bad.values(3).is_err(), "单点不是扫描");

        let mut bad = options();
        bad.atoms = vec![0];
        assert!(bad.values(3).is_err(), "距离要两个原子");

        let mut bad = options();
        bad.atoms = vec![0, 7];
        assert!(bad.values(3).is_err(), "原子序号越界");

        assert_eq!(ScanKind::Angle.atom_count(), 3);
        assert_eq!(ScanKind::Dihedral.unit(), "degree");
        assert_eq!(ScanKind::Distance.unit(), "angstrom");
    }

    #[test]
    fn pes_scan_reports_the_steepest_adjacent_step() {
        let block = PesScan::new(
            "distance",
            "angstrom",
            vec![0, 1],
            false,
            vec![
                PesPoint::new(0.8, -74.9, false, 0),
                PesPoint::new(0.9, -74.94, false, 0),
                PesPoint::new(1.0, -74.93, false, 0),
            ],
        );
        assert!((block.max_adjacent_delta_hartree - 0.04).abs() < 1e-12);
        assert_eq!(block.minimum().expect("min").value, 0.9);
        assert!(block.validate().is_ok());
    }

    #[test]
    fn pes_scan_validate_catches_the_usual_drift() {
        let points = vec![
            PesPoint::new(0.9, -74.94, false, 0),
            PesPoint::new(1.0, -74.93, false, 0),
        ];
        let block = PesScan::new("distance", "angstrom", vec![0, 1], false, points.clone());
        assert!(block.validate().is_ok());

        // 非单调的坐标
        let mut bad = block.clone();
        bad.points.reverse();
        bad.max_adjacent_delta_hartree = 0.01;
        assert!(bad.validate().is_err());

        // 刚性扫描里不能有"优化过的点"
        let mut bad = block.clone();
        bad.points[0].optimised = true;
        assert!(bad.validate().is_err());

        // 派生量与点集不一致
        let mut bad = block.clone();
        bad.max_adjacent_delta_hartree = 0.5;
        assert!(bad.validate().is_err());

        // 少于两个点
        let bad = PesScan::new(
            "distance",
            "angstrom",
            vec![0, 1],
            false,
            vec![PesPoint::new(0.9, -74.94, false, 0)],
        );
        assert!(bad.validate().is_err());
    }
}
