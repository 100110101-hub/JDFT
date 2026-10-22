//! 数据注册表的**契约侧句柄**（K2 只写了 `Engine::new(&Registry)`，未定义其内容）。
//!
//! 设计取舍（见 `docs/choices.md` → “K2 formalization”）：
//!
//! - 契约 crate **不解析数据文件**。数据加载是 `qwb-registry`（D-01）的职责。
//! - `Registry` 在这里是**数据根 + 声明的能力集合**的句柄：谁加载了数据，谁把
//!   `features` 填进来；`Engine::new` 只做“我这一版引擎能不能吃下这些 feature”的判断。
//! - 这样 K2 保持纯契约：没有文件格式、没有 `serde` 依赖、没有 IO 逻辑泄漏进 API crate。
//!
//! # feature 词汇表
//!
//! feature 是 `"<域>:<值>"` 形式的字符串，用来表达“本次计算需要引擎具备什么能力”，
//! 例如 `jobtype:opt`、`functional:hybrid`、`solvent:implicit`、`ecp`。
//! 引擎拿 [`crate::SUPPORTED_FEATURES`] 比对，缺一个就返回
//! [`crate::EngineError::Unsupported`] —— 这是 §8.5 要求的“loader 不报错，
//! `Engine::run` 返回 Unsupported”的落地点。

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// 数据注册表句柄：数据根目录 + 本次计算声明的能力集合。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Registry {
    root: PathBuf,
    declared_features: BTreeSet<String>,
}

impl Registry {
    /// 以数据根目录构造（不校验、不 IO）。
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            declared_features: BTreeSet::new(),
        }
    }

    /// 声明本次计算需要的 feature（重复项自动去重，保持字典序）。
    #[must_use]
    pub fn with_declared_features<I, S>(mut self, features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.declared_features
            .extend(features.into_iter().map(Into::into));
        self
    }

    /// 追加单个 feature。
    pub fn declare_feature(&mut self, feature: impl Into<String>) {
        self.declared_features.insert(feature.into());
    }

    /// 数据根目录。
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 已声明的 feature，字典序。
    pub fn declared_features(&self) -> impl Iterator<Item = &str> {
        self.declared_features.iter().map(String::as_str)
    }

    /// 是否声明了某 feature。
    #[must_use]
    pub fn declares(&self, feature: &str) -> bool {
        self.declared_features.contains(feature)
    }
}

#[cfg(test)]
mod tests {
    use super::Registry;

    #[test]
    fn features_are_deduplicated_and_sorted() {
        let r = Registry::new("qwb-data").with_declared_features([
            "functional:hybrid",
            "ecp",
            "functional:hybrid",
        ]);
        assert_eq!(
            r.declared_features().collect::<Vec<_>>(),
            vec!["ecp", "functional:hybrid"]
        );
        assert!(r.declares("ecp"));
        assert!(!r.declares("functional:meta-gga"));
        assert_eq!(r.root().to_string_lossy(), "qwb-data");
    }

    #[test]
    fn default_registry_has_no_features() {
        let r = Registry::default();
        assert_eq!(r.declared_features().count(), 0);
    }
}
