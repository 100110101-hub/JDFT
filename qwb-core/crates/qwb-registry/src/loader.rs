//! D-01：数据加载器（`Registry`）。**load ≠ verify**（§8.6）。
//!
//! ```text
//! Registry::open(root)
//!   ├─ 读 <root>/registry.yaml        ← 清单（索引 + 版本 + feature 声明）
//!   └─ 逐个加载条目文件 → typed 值    ← 语法 + 结构 + 语义自洽
//! ```
//!
//! # load 与 verify 的分工（§8.6，v2.1 明确）
//!
//! | | `Registry::open`（load） | `qwb data verify`（verify） |
//! |---|---|---|
//! | 目的 | 让引擎/工具链能拿到数据 | G2 数据门禁 |
//! | 空注册表 | **`Ok`**（0 条目也合法 —— 冷启动必须可用） | 失败（`registry contains no data entries`） |
//! | 缺 `citation`/`license` | 不管 | 失败（`g2.citation` / `g2.license`） |
//! | `sha256` 不符 | 不管（加载不重算摘要） | 失败（`sha256.mismatch`） |
//! | YAML 语法 / 未知字段 / 语义不自洽 | **失败**（`Data`） | 失败 |
//!
//! 加载阶段额外做的**索引完整性**检查（属于结构，不是 G2）：
//! 文件内的 `name` 必须与 `registry.yaml` 索引它的名字一致；重名条目被拒。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qwb_engine_api::{DataSnapshot, EngineError};
use serde::de::DeserializeOwned;

use crate::basis::BasisSet;
use crate::common::DataCommon;
use crate::ecp::Ecp;
use crate::functional::Functional;
use crate::grid::{GridLevelDef, GridLevelName};
use crate::registry::{RegistryEntry, RegistryManifest};
use crate::solvent::Solvent;

/// 清单文件名（§12.1）。
pub const MANIFEST_FILE: &str = "registry.yaml";

/// 已加载的数据注册表：数据根 + 清单 + 全部数据的解析结果。
///
/// 不可变、`Send + Sync`：加载完成后可以随意共享给多个任务线程（K2 要求）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Registry {
    root: PathBuf,
    manifest: RegistryManifest,
    bases: BTreeMap<String, BasisSet>,
    ecps: BTreeMap<String, Ecp>,
    functionals: BTreeMap<String, Functional>,
    grids: BTreeMap<String, GridLevelDef>,
    solvents: BTreeMap<String, Solvent>,
}

impl Registry {
    /// 打开数据根并加载清单索引的全部数据。
    ///
    /// # Errors
    ///
    /// - [`EngineError::Data`]：数据根不是目录；`registry.yaml` 缺失或解析失败；
    ///   任一条目文件读不出、语法/结构/语义不合法；索引与文件内的 `name` 不一致；
    ///   同名条目重复。
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, EngineError> {
        let root = root.into();
        if !root.is_dir() {
            return Err(EngineError::Data(format!(
                "registry root is not a directory: {}",
                root.display()
            )));
        }

        let manifest_path = root.join(MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path).map_err(|err| {
            EngineError::Data(format!("cannot read {}: {err}", manifest_path.display()))
        })?;
        let manifest: RegistryManifest = serde_yaml::from_slice(&bytes)
            .map_err(|err| EngineError::Data(format!("{}: {err}", manifest_path.display())))?;

        let bases =
            load_all::<BasisSet>(&root, "basis", &manifest.bases, BasisSet::check_structure)?;
        let ecps = load_all::<Ecp>(&root, "ecp", &manifest.ecps, Ecp::check_structure)?;
        let functionals = load_all::<Functional>(
            &root,
            "functional",
            &manifest.functionals,
            Functional::check_structure,
        )?;
        let grids = load_all::<GridLevelDef>(
            &root,
            "grid",
            &manifest.grids,
            GridLevelDef::check_structure,
        )?;
        let solvents = load_all::<Solvent>(
            &root,
            "solvent",
            &manifest.solvents,
            Solvent::check_structure,
        )?;

        Ok(Self {
            root,
            manifest,
            bases,
            ecps,
            functionals,
            grids,
            solvents,
        })
    }

    /// 数据根。
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 清单（版本、feature 声明、sha256 索引）。
    #[must_use]
    pub fn manifest(&self) -> &RegistryManifest {
        &self.manifest
    }

    /// 本数据组合声明的 feature（喂给 `engine-api` 的 `Registry` 句柄）。
    pub fn features(&self) -> impl Iterator<Item = &str> {
        self.manifest.features.iter().map(String::as_str)
    }

    /// 基组键名（字典序）。
    pub fn basis_names(&self) -> impl Iterator<Item = &str> {
        self.bases.keys().map(String::as_str)
    }

    /// 泛函键名。
    pub fn functional_names(&self) -> impl Iterator<Item = &str> {
        self.functionals.keys().map(String::as_str)
    }

    /// ECP 键名。
    pub fn ecp_names(&self) -> impl Iterator<Item = &str> {
        self.ecps.keys().map(String::as_str)
    }

    /// 网格键名。
    pub fn grid_names(&self) -> impl Iterator<Item = &str> {
        self.grids.keys().map(String::as_str)
    }

    /// 溶剂键名。
    pub fn solvent_names(&self) -> impl Iterator<Item = &str> {
        self.solvents.keys().map(String::as_str)
    }

    /// 取基组。
    ///
    /// # Errors
    ///
    /// [`EngineError::Data`]：未索引该名字（消息里列出可用名字）。
    pub fn get_basis(&self, name: &str) -> Result<&BasisSet, EngineError> {
        lookup(&self.bases, "basis", name)
    }

    /// 取泛函。
    ///
    /// # Errors
    ///
    /// 同 [`Registry::get_basis`]。
    pub fn get_functional(&self, name: &str) -> Result<&Functional, EngineError> {
        lookup(&self.functionals, "functional", name)
    }

    /// 取 ECP。
    ///
    /// # Errors
    ///
    /// 同 [`Registry::get_basis`]。
    pub fn get_ecp(&self, name: &str) -> Result<&Ecp, EngineError> {
        lookup(&self.ecps, "ecp", name)
    }

    /// 取网格定义（按名字）。
    ///
    /// # Errors
    ///
    /// 同 [`Registry::get_basis`]。
    pub fn get_grid(&self, name: &str) -> Result<&GridLevelDef, EngineError> {
        lookup(&self.grids, "grid", name)
    }

    /// 取溶剂。
    ///
    /// # Errors
    ///
    /// 同 [`Registry::get_basis`]。
    pub fn get_solvent(&self, name: &str) -> Result<&Solvent, EngineError> {
        lookup(&self.solvents, "solvent", name)
    }

    /// 按等级取网格定义（引擎入口：`JobInput.grid_level` → 参数）。
    ///
    /// # Errors
    ///
    /// - [`EngineError::Data`]：没有任何 grid 条目的 `level` 等于它；
    /// - [`EngineError::Data`]：**多于一个**条目声明了同一等级（注册表有歧义，
    ///   引擎不能靠“先到先得”猜参数）。
    pub fn get_grid_by_level(&self, level: GridLevelName) -> Result<&GridLevelDef, EngineError> {
        let matches: Vec<(&str, &GridLevelDef)> = self
            .grids
            .iter()
            .filter(|(_, grid)| grid.level == level)
            .map(|(name, grid)| (name.as_str(), grid))
            .collect();

        match matches.as_slice() {
            [(_, grid)] => Ok(grid),
            [] => Err(EngineError::Data(format!(
                "no grid definition for level `{}`; available levels: {}",
                level_dir_name(level),
                {
                    let mut levels: Vec<&str> = self
                        .grids
                        .values()
                        .map(|grid| level_dir_name(grid.level))
                        .collect();
                    levels.sort_unstable();
                    levels.dedup();
                    if levels.is_empty() {
                        "none".to_string()
                    } else {
                        levels.join(", ")
                    }
                }
            ))),
            many => Err(EngineError::Data(format!(
                "ambiguous grid level `{}`: {} definitions declared ({}); \
                 registry.yaml must declare exactly one grid per level",
                level_dir_name(level),
                many.len(),
                many.iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    /// **唯一**的基组/ECP 配对入口（§8.2）。
    ///
    /// 校验两件事：① ECP 的元素在基组里有对应的壳层块；② `z_eff == Z - n_core` 自洽。
    ///
    /// # Errors
    ///
    /// [`EngineError::Data`]：任一名字不存在、元素不匹配、`z_eff` 不自洽。
    pub fn load_pair(
        &self,
        basis_name: &str,
        ecp_name: &str,
    ) -> Result<(&BasisSet, &Ecp), EngineError> {
        let basis = self.get_basis(basis_name)?;
        let ecp = self.get_ecp(ecp_name)?;

        if !basis.elements.iter().any(|element| element == &ecp.element) {
            return Err(EngineError::Data(format!(
                "ecp `{ecp_name}` is for {} but basis `{basis_name}` covers [{}]; \
                 no valence basis block for {}",
                ecp.element,
                basis.elements.join(", "),
                ecp.element
            )));
        }
        ecp.z_eff_consistent()
            .map_err(|msg| EngineError::Data(format!("ecp `{ecp_name}`: {msg}")))?;
        Ok((basis, ecp))
    }

    /// 全量数据快照（`path -> sha256`）：直接来自清单索引。
    ///
    /// 写进结果的 `data_snapshot.json`（K3），与文件内 `sha256` 字段、
    /// `registry.yaml` 条目三处一致（§8.6）。
    #[must_use]
    pub fn snapshot(&self) -> DataSnapshot {
        let mut snapshot = DataSnapshot::new();
        for (_, entry) in self.manifest.all_entries() {
            snapshot.insert(&entry.path, &entry.sha256);
        }
        snapshot
    }

    /// 构造 K2 的引擎句柄（数据根 + 本次声明的 feature）。
    #[must_use]
    pub fn engine_handle(&self) -> qwb_engine_api::Registry {
        qwb_engine_api::Registry::new(&self.root).with_declared_features(self.features())
    }
}

/// 加载一个域的全部条目。
fn load_all<T>(
    root: &Path,
    domain: &'static str,
    entries: &[RegistryEntry],
    check: fn(&T) -> Vec<String>,
) -> Result<BTreeMap<String, T>, EngineError>
where
    T: DeserializeOwned + DataCommon,
{
    let mut map = BTreeMap::new();
    for entry in entries {
        let path = root.join(&entry.path);
        let bytes = std::fs::read(&path).map_err(|err| {
            EngineError::Data(format!(
                "{domain}/{}: cannot read {}: {err}",
                entry.name,
                path.display()
            ))
        })?;
        let value: T = serde_yaml::from_slice(&bytes).map_err(|err| {
            EngineError::Data(format!(
                "{domain}/{} ({}): {err}",
                entry.name,
                path.display()
            ))
        })?;

        if value.name() != entry.name {
            return Err(EngineError::Data(format!(
                "{domain}/{}: file {} declares name `{}` but the manifest indexes it as `{}`",
                entry.name,
                path.display(),
                value.name(),
                entry.name
            )));
        }

        let issues = check(&value);
        if !issues.is_empty() {
            return Err(EngineError::Data(format!(
                "{domain}/{} ({}): {}",
                entry.name,
                path.display(),
                issues.join("; ")
            )));
        }

        if map.insert(entry.name.clone(), value).is_some() {
            return Err(EngineError::Data(format!(
                "{domain}: duplicate data entry `{}` in registry.yaml",
                entry.name
            )));
        }
    }
    Ok(map)
}

/// 按名字取数据，错误消息里带上可用名字（避免“not found”无从下手）。
fn lookup<'a, T>(
    map: &'a BTreeMap<String, T>,
    domain: &str,
    name: &str,
) -> Result<&'a T, EngineError> {
    map.get(name).ok_or_else(|| {
        let available = map.keys().cloned().collect::<Vec<_>>().join(", ");
        EngineError::Data(format!(
            "unknown {domain} `{name}`; available: [{}]",
            if available.is_empty() {
                "none".to_string()
            } else {
                available
            }
        ))
    })
}

/// 等级的小写名（与 `grid.schema.yaml` 的 `level` 取值一致）。
fn level_dir_name(level: GridLevelName) -> &'static str {
    match level {
        GridLevelName::Coarse => "coarse",
        GridLevelName::Normal => "normal",
        GridLevelName::Fine => "fine",
        GridLevelName::VeryFine => "veryfine",
    }
}
