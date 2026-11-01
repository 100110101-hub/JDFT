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
    /// **幽灵原子**（ghost）：只提供基函数、不提供核电荷与电子。
    ///
    /// 这是 counterpoise（BSSE）校正要用的：把伙伴分子放在原位置但换成 ghost，
    /// 单体的能量里就只剩"伙伴的基函数"这一项基组不完备效应。默认 `false`，
    /// 且序列化时**不输出**（旧输出逐字节不变）。
    #[serde(default, skip_serializing_if = "is_false")]
    pub ghost: bool,
}

/// `skip_serializing_if` 的辅助：`false` 不写出来。
fn is_false(value: &bool) -> bool {
    !*value
}

impl Atom {
    /// 用元素符号与 Å 坐标构造原子。
    #[must_use]
    pub fn new(symbol: impl Into<String>, position_angstrom: [f64; 3]) -> Self {
        Self {
            symbol: symbol.into(),
            position_angstrom,
            ghost: false,
        }
    }

    /// 构造一个**幽灵原子**：保留该元素的基函数，但核电荷与电子数为 0。
    #[must_use]
    pub fn new_ghost(symbol: impl Into<String>, position_angstrom: [f64; 3]) -> Self {
        Self {
            symbol: symbol.into(),
            position_angstrom,
            ghost: true,
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
        if self.atoms.iter().all(|atom| atom.ghost) {
            return Err(EngineError::InvalidInput(
                "a molecule made only of ghost atoms has no electrons".into(),
            ));
        }
        Ok(())
    }

    /// 是否含幽灵原子（counterpoise 校正）。
    #[must_use]
    pub fn has_ghosts(&self) -> bool {
        self.atoms.iter().any(|atom| atom.ghost)
    }

    /// 真实原子（非 ghost）的个数。
    #[must_use]
    pub fn n_real_atoms(&self) -> usize {
        self.atoms.iter().filter(|atom| !atom.ghost).count()
    }
}

#[cfg(test)]
mod tests {
    use super::{Atom, BOHR_PER_ANGSTROM, Molecule};

    #[test]
    fn ghost_atoms_are_opt_in_and_never_serialized_when_false() {
        // 幽灵原子：默认 false、`false` 不写进 JSON（旧输出逐字节不变）、`new_ghost` 置 true。
        let plain = Atom::new("O", [0.0, 0.0, 0.0]);
        let text = serde_json::to_string(&plain).expect("json");
        assert!(
            !text.contains("ghost"),
            "false 的 ghost 不该出现在 JSON 里：{text}"
        );
        let restored: Atom = serde_json::from_str(&text).expect("back");
        assert!(!restored.ghost);

        let ghost = Atom::new_ghost("O", [0.0, 0.0, 1.5]);
        let text = serde_json::to_string(&ghost).expect("json");
        assert!(text.contains("ghost"), "{text}");
        let restored: Atom = serde_json::from_str(&text).expect("back");
        assert!(restored.ghost);

        // 分子层面：`has_ghosts` / `n_real_atoms`，以及"全是 ghost"被拒（没有电子）。
        let molecule = Molecule::new(vec![restored, Atom::new("H", [0.0, 0.0, -1.0])], 0, 1);
        assert!(molecule.has_ghosts());
        assert_eq!(molecule.n_real_atoms(), 1);
        molecule.validate().expect("一个真实原子就够");

        let all_ghosts = Molecule::new(vec![Atom::new_ghost("O", [0.0, 0.0, 0.0])], 0, 1);
        assert!(all_ghosts.validate().is_err(), "全是 ghost 的分子没有电子");
    }

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
