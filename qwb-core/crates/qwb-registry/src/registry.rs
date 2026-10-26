//! 注册表清单（§4 K1 `registry.schema.yaml`，§12.1 `qwb-data/registry.yaml`）。
//!
//! 清单是**数据的索引与提交快照**：每个条目带 `sha256`，与文件内部声明的 `sha256`
//! 必须一致（两处独立记录，任一处被改都会被发现 —— G2）。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::sha256::is_sha256_hex;

/// 一条数据条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RegistryEntry {
    /// 数据键名（必须与文件内的 `name` 一致）。
    pub name: String,
    /// 相对 `registry.yaml` 所在目录的路径。
    pub path: String,
    /// 该文件内容的 sha256（与文件内 `sha256` 字段一致）。
    pub sha256: String,
}

/// 注册表清单。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RegistryManifest {
    /// K1 版本（`crate::K1_VERSION`）。
    pub schema_version: String,
    /// 注册表名（如 `qwb-data`）。
    pub name: String,
    /// 数据仓版本（独立 semver，§8.6）。
    pub version: String,
    /// 创建时间，ISO 8601 UTC。
    pub created: String,
    /// 本数据组合向引擎声明的能力集合（见 `engine-api` 的 feature 词汇表）。
    #[serde(default)]
    pub features: Vec<String>,
    /// 基组条目。
    #[serde(default)]
    pub bases: Vec<RegistryEntry>,
    /// ECP 条目。
    #[serde(default)]
    pub ecps: Vec<RegistryEntry>,
    /// 泛函条目。
    #[serde(default)]
    pub functionals: Vec<RegistryEntry>,
    /// 网格条目。
    #[serde(default)]
    pub grids: Vec<RegistryEntry>,
    /// 溶剂条目。
    #[serde(default)]
    pub solvents: Vec<RegistryEntry>,
    /// 数值路径的精度等级与等价容差条目（阶段二 K10）。
    #[serde(default)]
    pub numerics: Vec<RegistryEntry>,
}

impl RegistryManifest {
    /// 全部条目，附所属域（域名字与 `qwb-data` 子目录一致）。
    pub fn all_entries(&self) -> impl Iterator<Item = (&'static str, &RegistryEntry)> {
        let groups: [(&'static str, &Vec<RegistryEntry>); 6] = [
            ("basis", &self.bases),
            ("ecp", &self.ecps),
            ("functional", &self.functionals),
            ("grid", &self.grids),
            ("solvent", &self.solvents),
            ("numerics", &self.numerics),
        ];
        groups
            .into_iter()
            .flat_map(|(domain, entries)| entries.iter().map(move |e| (domain, e)))
    }

    /// 语义校验。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.schema_version != crate::K1_VERSION {
            issues.push(format!(
                "schema_version = {} but this build implements K1 {}",
                self.schema_version,
                crate::K1_VERSION
            ));
        }
        if self.name.trim().is_empty() {
            issues.push("name must not be empty".into());
        }
        if self.version.trim().is_empty() {
            issues.push("version must not be empty".into());
        }
        if self.created.trim().is_empty() {
            issues.push("created must not be empty".into());
        }
        for (i, feature) in self.features.iter().enumerate() {
            if feature.trim().is_empty() {
                issues.push(format!("features[{i}] must not be empty"));
            }
        }

        let mut seen = std::collections::BTreeSet::new();
        for (domain, entry) in self.all_entries() {
            let key = format!("{domain}/{}", entry.name);
            if entry.name.trim().is_empty() {
                issues.push(format!("{domain} entry has an empty name"));
            }
            if !seen.insert(key.clone()) {
                issues.push(format!("duplicate entry {key}"));
            }
            if entry.path.trim().is_empty() {
                issues.push(format!("{key} has an empty path"));
            }
            if !is_sha256_hex(&entry.sha256) {
                issues.push(format!("{key} has a malformed sha256: {:?}", entry.sha256));
            }
        }

        let total = self.all_entries().count();
        if total == 0 {
            issues.push("registry contains no data entries".into());
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::{RegistryEntry, RegistryManifest};

    fn entry(name: &str) -> RegistryEntry {
        RegistryEntry {
            name: name.into(),
            path: format!("basis/{name}.yaml"),
            sha256: "a".repeat(64),
        }
    }

    pub(crate) fn manifest() -> RegistryManifest {
        RegistryManifest {
            schema_version: crate::K1_VERSION.into(),
            name: "qwb-data".into(),
            version: "0.1.0".into(),
            created: "2025-01-10T08:00:00Z".into(),
            features: vec!["jobtype:sp".into()],
            bases: vec![entry("sto-3g")],
            ecps: Vec::new(),
            functionals: Vec::new(),
            grids: Vec::new(),
            solvents: Vec::new(),
            numerics: Vec::new(),
        }
    }

    #[test]
    fn manifest_is_valid() {
        let m = manifest();
        assert!(m.check().is_empty(), "{:?}", m.check());
        assert_eq!(m.all_entries().count(), 1);
    }

    #[test]
    fn check_catches_broken_manifests() {
        let mut m = manifest();
        m.schema_version = "0.9.0".into();
        assert!(m.check().iter().any(|s| s.contains("schema_version")));

        let mut m = manifest();
        m.bases.push(entry("sto-3g"));
        assert!(m.check().iter().any(|s| s.contains("duplicate entry")));

        let mut m = manifest();
        m.bases[0].sha256 = "zz".into();
        assert!(m.check().iter().any(|s| s.contains("malformed sha256")));

        let mut m = manifest();
        m.bases.clear();
        assert!(m.check().iter().any(|s| s.contains("no data entries")));
    }
}
