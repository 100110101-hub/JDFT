//! 第三方数据的「配方清单」：`qwb-data/sources.json`。
//!
//! # 为什么数据不在仓库里
//!
//! `qwb-data/basis/` 与 `qwb-data/ecp/` 的部分文件是从 Basis Set Exchange 导出的第三方
//! 数据。LANL2DZ 的 ECP 原始论文没有明确再分发条款（`redistribute_ok: false`，
//! project.md §14 #20），把文件提交到公开仓库就等于再分发。所以仓库commit 的是**配方**：
//!
//! - `sources.json`：每条数据的 `fetch` 段（BSE 的 basis 名 + 元素）、`transcribe` 段
//!   （名字/版本/引用/许可/形态），以及**摘要钉** `sha256`；
//! - `registry.yaml`：同一条数据的 `path` + `sha256`（K1 要求，不变）。
//!
//! 数据由 `python -m qwb_studio.datafetch` 现场取回并核对摘要钉；核对不通过就拒绝落盘。
//! 摘要规则只有一份实现（[`crate::sha256`]），所以"取回的文件"与"当初提交的文件"
//! 内容逐字节等价 —— `sources.json` 的钉与 `registry.yaml` 的钉必须相等，本模块下面
//! 的单元测试会检查这一点。
//!
//! # 这份清单的用途
//!
//! 1. 加载器据此**跳过本地缺失的第三方文件**（而不是让整个注册表打不开）：缺 LANL2DZ 不该
//!    挡住一个 STO-3G 的作业；
//! 2. 取这个名字的人会拿到"文件没装 + 怎么装"的错误，而不是"未知基组"。
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::path::Path;

use qwb_engine_api::EngineError;
use serde::Deserialize;

/// 清单文件名（与 `qwb-data/sources.json` 一致）。
pub const SOURCES_FILE: &str = "sources.json";

/// 一条第三方数据条目（只取本模块需要的字段：路径与摘要钉）。
///
/// 清单里还有 `kind` / `fetch` / `transcribe`，那是取数脚本的输入，加载器不关心；
/// 因此这里**不用** `deny_unknown_fields`（否则取数一侧一加字段就把加载器弄坏）。
#[derive(Debug, Clone, Deserialize)]
struct SourcesEntry {
    /// 相对数据根的路径（如 `basis/lanl2dz.yaml`）。
    path: String,
    /// 内容摘要钉（与 `registry.yaml` 里的同一条相等）。
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Sources {
    #[serde(default)]
    entries: Vec<SourcesEntry>,
}

/// 一条本地缺失的第三方数据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingData {
    /// 数据域（`basis` / `ecp`），即 `path` 的第一级目录。
    pub domain: String,
    /// 数据键名（路径去掉目录与 `.yaml`）。
    pub name: String,
    /// 相对数据根的路径。
    pub path: String,
}

impl MissingData {
    fn from_path(path: &str) -> Self {
        let normalized = normalize(path);
        let mut parts = normalized.split('/');
        let domain = parts.next().unwrap_or("data").to_string();
        let name = Path::new(&normalized)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .to_string();
        Self {
            domain,
            name,
            path: normalized,
        }
    }
}

/// 路径归一化：`\` → `/`、去掉前导 `./`。清单与 `registry.yaml` 都写 `/`，
/// 但手工编辑时可能混进反斜杠，两处比较必须同形。
#[must_use]
pub fn normalize(path: &str) -> String {
    let mut text = path.replace('\\', "/");
    while let Some(rest) = text.strip_prefix("./") {
        text = rest.to_string();
    }
    text
}

/// 读清单；文件不存在时返回空集（老数据根 / 全部数据随仓库走的情形）。
///
/// # Errors
///
/// [`EngineError::Data`]：`sources.json` 存在但读不出或不是合法 JSON。
pub fn third_party_paths(root: &Path) -> Result<BTreeSet<String>, EngineError> {
    let path = root.join(SOURCES_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(err) => {
            return Err(EngineError::Data(format!(
                "cannot read {}: {err}",
                path.display()
            )));
        }
    };
    let sources: Sources = serde_json::from_slice(&bytes).map_err(|err| {
        EngineError::Data(format!(
            "{} is not a valid sources.json: {err}",
            path.display()
        ))
    })?;
    Ok(sources
        .entries
        .into_iter()
        .map(|entry| normalize(&entry.path))
        .collect())
}

/// 清单里声明、但本地没有的文件（按路径排序）。
///
/// # Errors
///
/// 同 [`third_party_paths`]。
pub fn missing_data(root: &Path) -> Result<Vec<MissingData>, EngineError> {
    let mut missing = Vec::new();
    for relative in third_party_paths(root)? {
        if !root.join(&relative).is_file() {
            missing.push(MissingData::from_path(&relative));
        }
    }
    Ok(missing)
}

/// 清单与 `registry.yaml` 的摘要钉是否一致（`sources.json` 自洽性检查）。
///
/// # Errors
///
/// 清单读不出、不是合法 JSON，或 `registry.yaml` 读不出。
pub fn check_pins(
    root: &Path,
    manifest: &crate::RegistryManifest,
) -> Result<Vec<String>, EngineError> {
    let path = root.join(SOURCES_FILE);
    let bytes = std::fs::read(&path)
        .map_err(|err| EngineError::Data(format!("cannot read {}: {err}", path.display())))?;
    let sources: Sources = serde_json::from_slice(&bytes).map_err(|err| {
        EngineError::Data(format!(
            "{} is not a valid sources.json: {err}",
            path.display()
        ))
    })?;

    let index: std::collections::BTreeMap<String, String> = manifest
        .all_entries()
        .map(|(_, entry)| (normalize(&entry.path), entry.sha256.clone()))
        .collect();

    let mut issues = Vec::new();
    for entry in &sources.entries {
        let relative = normalize(&entry.path);
        match index.get(&relative) {
            None => issues.push(format!(
                "sources.json 列了 {relative}，但 registry.yaml 没有这个条目"
            )),
            Some(known) if known != &entry.sha256 => issues.push(format!(
                "{}: sources.json 的钉 {} 与 registry.yaml 的 {} 不一致",
                relative, entry.sha256, known
            )),
            Some(_) => {}
        }
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::{MissingData, check_pins, missing_data, normalize, third_party_paths};
    use std::path::{Path, PathBuf};

    fn data_root() -> PathBuf {
        // `<repo>/qwb-core/crates/qwb-registry` → 上三级 = 仓库根
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        here.parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .expect("workspace root")
            .join("qwb-data")
    }

    #[test]
    fn normalize_folds_separators() {
        assert_eq!(normalize("basis/sto-3g.yaml"), "basis/sto-3g.yaml");
        assert_eq!(normalize("basis\\sto-3g.yaml"), "basis/sto-3g.yaml");
        assert_eq!(normalize("./basis/sto-3g.yaml"), "basis/sto-3g.yaml");
    }

    #[test]
    fn missing_data_derives_domain_and_name() {
        let missing = MissingData::from_path("ecp/lanl2dz-cu.yaml");
        assert_eq!(missing.domain, "ecp");
        assert_eq!(missing.name, "lanl2dz-cu");
        assert_eq!(missing.path, "ecp/lanl2dz-cu.yaml");
    }

    #[test]
    fn sources_json_is_self_consistent() {
        let root = data_root();
        let paths = third_party_paths(&root).expect("sources.json");
        assert!(
            !paths.is_empty(),
            "仓库里的 qwb-data/sources.json 应当列出第三方数据（本仓库不提交这些文件）"
        );
        assert!(paths.iter().all(|path| path.ends_with(".yaml")));

        // 清单与 registry.yaml 的钉必须一致：不一致意味着"取回的文件"与"索引里的文件"
        // 不是同一份内容，而这正是本机制要防的事。
        let manifest_text = std::fs::read_to_string(root.join("registry.yaml")).expect("registry");
        let manifest: crate::RegistryManifest =
            serde_yaml::from_str(&manifest_text).expect("parse registry.yaml");
        let issues = check_pins(&root, &manifest).expect("check pins");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn absent_sources_file_is_not_an_error() {
        let empty = std::env::temp_dir().join("qwb-thirdparty-empty");
        std::fs::create_dir_all(&empty).expect("tmp");
        assert!(third_party_paths(&empty).expect("ok").is_empty());
        assert!(missing_data(&empty).expect("ok").is_empty());
    }
}
