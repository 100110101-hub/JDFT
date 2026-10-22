//! 分子与原子（K2）。
//!
//! 边界单位是 Å：输入 YAML、Python API、GUI 全部用 Å。引擎内部按
//! `BOHR_PER_ANGSTROM` 转 Bohr，转换只允许发生在 `Engine::run` 入口一次。

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// 1 Å 等于多少 Bohr（CODATA 2018 精确值，A.1.2）。
pub const BOHR_PER_ANGSTROM: f64 = 1.889_726_089_838_572;

/// 单个原子。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Atom {
    /// 元素符号，区分大小写（`"H"`、`"Cl"`）。
    pub symbol: String,
    /// 直角坐标，单位 Å，顺序为 `[x, y, z]`。
    pub position_angstrom: [f64; 3],
}

impl Atom {
    /// 用元素符号与 Å 坐标构造原子。
    #[must_use]
    pub fn new(symbol: impl Into<String>, position_angstrom: [f64; 3]) -> Self {
        Self {
            symbol: symbol.into(),
            position_angstrom,
        }
    }

    /// 坐标转换为 Bohr。
    #[must_use]
    pub fn position_bohr(&self) -> [f64; 3] {
        self.position_angstrom.map(|v| v * BOHR_PER_ANGSTROM)
    }
}

/// 分子：原子列表 + 总电荷 + 自旋多重度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Molecule {
    /// 原子列表，顺序即外部索引顺序（梯度/Hessian 的行号依赖它）。
    pub atoms: Vec<Atom>,
    /// 总电荷（整数，含核电荷求和）。
    pub charge: i32,
    /// 自旋多重度 `2S+1`，必须 >= 1。
    pub multiplicity: u32,
}

impl Molecule {
    /// 构造分子。
    #[must_use]
    pub fn new(atoms: Vec<Atom>, charge: i32, multiplicity: u32) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// 原子数 `N`。
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// 校验结构自洽性（非空、多重度合法、坐标有限）。
    ///
    /// # Errors
    ///
    /// 原子为空、`multiplicity == 0`、坐标含 NaN/Inf 时返回 [`EngineError::InvalidInput`]。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.atoms.is_empty() {
            return Err(EngineError::InvalidInput("molecule has no atoms".into()));
        }
        if self.multiplicity == 0 {
            return Err(EngineError::InvalidInput(
                "multiplicity must be >= 1".into(),
            ));
        }
        for (i, atom) in self.atoms.iter().enumerate() {
            if atom.symbol.trim().is_empty() {
                return Err(EngineError::InvalidInput(format!(
                    "atom {i} has empty symbol"
                )));
            }
            if atom.position_angstrom.iter().any(|v| !v.is_finite()) {
                return Err(EngineError::InvalidInput(format!(
                    "atom {i} ({}) has non-finite coordinates",
                    atom.symbol
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Atom, BOHR_PER_ANGSTROM, Molecule};

    fn h2() -> Molecule {
        Molecule::new(
            vec![
                Atom::new("H", [0.0, 0.0, -0.37]),
                Atom::new("H", [0.0, 0.0, 0.37]),
            ],
            0,
            1,
        )
    }

    #[test]
    fn angstrom_to_bohr_conversion() {
        let atom = Atom::new("H", [1.0, 0.0, 0.0]);
        assert!((atom.position_bohr()[0] - BOHR_PER_ANGSTROM).abs() < 1e-15);
    }

    #[test]
    fn validate_rejects_broken_molecules() {
        assert!(Molecule::new(vec![], 0, 1).validate().is_err());
        assert!(
            Molecule::new(vec![Atom::new("H", [0.0; 3])], 0, 0)
                .validate()
                .is_err()
        );
        assert!(
            Molecule::new(vec![Atom::new("H", [f64::NAN, 0.0, 0.0])], 0, 1)
                .validate()
                .is_err()
        );
        assert!(h2().validate().is_ok());
        assert_eq!(h2().n_atoms(), 2);
    }
}
