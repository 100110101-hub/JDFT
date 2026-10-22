//! 分子文件（§4 K1 `molecule.schema.yaml`）。
//!
//! **与注册表数据的区别**：分子是**用户输入**，不是受许可约束的数据集，因此这里
//! 不要求 `license`，`citation` 也可空（用户自己画的分子没有 DOI）。G2 数据门禁
//! 只作用于 `qwb-data/{basis,ecp,functional,grid,solvent}`。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::common::Citation;
use crate::elements;

/// 单个原子（Å，与 `engine-api::Atom` 对齐）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct MoleculeAtom {
    /// 元素符号。
    pub symbol: String,
    /// 直角坐标，单位 Å。
    pub position_angstrom: [f64; 3],
}

/// 分子文件（`qwb-data/molecule/*.yaml`，也用于 GUI 保存的分子）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct MoleculeFile {
    /// 分子名。
    pub name: String,
    /// 备注。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// 原子列表。
    pub atoms: Vec<MoleculeAtom>,
    /// 总电荷。
    pub charge: i32,
    /// 自旋多重度。
    pub multiplicity: u32,
    /// 结构来源引用（可空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citation: Vec<Citation>,
    /// 创建时间，ISO 8601 UTC。
    pub created: String,
    /// 内容摘要（工具生成，见 [`crate::sha256`]）。
    ///
    /// 分子不是 G2 数据集，所以**不要求**它存在；但 `qwb-stamp-data` 仍会为它打戳，
    /// 便于“这个几何文件被人改过没有”这类检查。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl MoleculeFile {
    /// 电子数（`Σ Z - charge`）。
    ///
    /// # Errors
    ///
    /// 含未知元素符号时返回错误说明。
    pub fn n_electrons(&self) -> Result<i64, String> {
        let mut z_sum: i64 = 0;
        for atom in &self.atoms {
            let z = elements::atomic_number(&atom.symbol)
                .ok_or_else(|| format!("unknown element `{}`", atom.symbol))?;
            z_sum += i64::from(z);
        }
        Ok(z_sum - i64::from(self.charge))
    }

    /// 语义校验。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.name.trim().is_empty() {
            issues.push("name must not be empty".into());
        }
        if self.created.trim().is_empty() {
            issues.push("created must not be empty".into());
        }
        if self.atoms.is_empty() {
            issues.push("atoms must not be empty".into());
        }
        if self.multiplicity == 0 {
            issues.push("multiplicity must be >= 1".into());
        }
        for (i, atom) in self.atoms.iter().enumerate() {
            if elements::atomic_number(&atom.symbol).is_none() {
                issues.push(format!(
                    "atoms[{i}].symbol `{}` is not a first-phase element (H..Kr)",
                    atom.symbol
                ));
            }
            if atom.position_angstrom.iter().any(|v| !v.is_finite()) {
                issues.push(format!("atoms[{i}] has non-finite coordinates"));
            }
        }
        for (i, cite) in self.citation.iter().enumerate() {
            if !cite.is_complete() {
                issues.push(format!("citation[{i}] needs title and doi/url"));
            }
        }

        // 电子数必须非负，且与多重度奇偶一致：多重度 2S+1 与电子数同奇偶。
        match self.n_electrons() {
            Ok(n) => {
                if n < 0 {
                    issues.push(format!(
                        "charge {} leaves {n} electrons: nuclear charge is smaller than the charge",
                        self.charge
                    ));
                } else if self.multiplicity >= 1 {
                    let parity_ok = n % 2 == i64::from(self.multiplicity - 1) % 2;
                    if !parity_ok {
                        issues.push(format!(
                            "multiplicity {m} is inconsistent with {n} electrons (parity mismatch)",
                            m = self.multiplicity
                        ));
                    }
                }
            }
            Err(msg) => issues.push(msg),
        }

        // 完全重合的原子是输入错误，不是化学。
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let (a, b) = (&self.atoms[i], &self.atoms[j]);
                if a.position_angstrom == b.position_angstrom && a.symbol == b.symbol {
                    issues.push(format!("atoms[{i}] and atoms[{j}] are identical"));
                }
            }
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::{MoleculeAtom, MoleculeFile};

    pub(crate) fn h2o() -> MoleculeFile {
        MoleculeFile {
            name: "water".into(),
            notes: None,
            atoms: vec![
                MoleculeAtom {
                    symbol: "O".into(),
                    position_angstrom: [0.0, 0.0, 0.0],
                },
                MoleculeAtom {
                    symbol: "H".into(),
                    position_angstrom: [0.0, 0.757, 0.586],
                },
                MoleculeAtom {
                    symbol: "H".into(),
                    position_angstrom: [0.0, -0.757, 0.586],
                },
            ],
            charge: 0,
            multiplicity: 1,
            citation: Vec::new(),
            created: "2025-01-10T08:00:00Z".into(),
            sha256: None,
        }
    }

    #[test]
    fn water_is_valid() {
        let m = h2o();
        assert_eq!(m.n_electrons().unwrap(), 10);
        assert!(m.check().is_empty(), "{:?}", m.check());
    }

    #[test]
    fn multiplicity_parity_is_enforced() {
        // 10 个电子（偶数）→ 未成对电子数必为偶数 → 多重度必为奇数。
        let mut m = h2o();
        m.multiplicity = 2; // S = 1/2（1 个未成对）→ 非法
        assert!(m.check().iter().any(|s| s.contains("parity mismatch")));

        let mut m = h2o();
        m.multiplicity = 4; // S = 3/2（3 个未成对）→ 非法
        assert!(m.check().iter().any(|s| s.contains("parity mismatch")));

        let mut m = h2o();
        m.multiplicity = 3; // S = 1（2 个未成对）→ 合法
        assert!(m.check().is_empty(), "{:?}", m.check());

        let mut m = h2o();
        m.charge = 1; // 9 电子（奇数）
        m.multiplicity = 2;
        assert!(m.check().is_empty(), "{:?}", m.check());
    }

    #[test]
    fn check_catches_broken_molecules() {
        let mut m = h2o();
        m.atoms.clear();
        assert!(
            m.check()
                .iter()
                .any(|s| s.contains("atoms must not be empty"))
        );

        let mut m = h2o();
        m.atoms[0].position_angstrom = [f64::NAN, 0.0, 0.0];
        assert!(m.check().iter().any(|s| s.contains("non-finite")));

        let mut m = h2o();
        m.atoms[0].symbol = "U".into();
        assert!(
            m.check()
                .iter()
                .any(|s| s.contains("not a first-phase element"))
        );

        let mut m = h2o();
        m.charge = 99;
        assert!(m.check().iter().any(|s| s.contains("nuclear charge")));

        let mut m = h2o();
        m.atoms[1] = m.atoms[0].clone();
        assert!(m.check().iter().any(|s| s.contains("identical")));
    }
}
