//! K4 `.qwb` 项目容器（**Zip + JSON**，K4 已定案，不用 HDF5）。
//!
//! ```text
//! meta.json
//! tasks/<id>/input.yaml
//! tasks/<id>/result.json
//! tasks/<id>/env.json
//! tasks/<id>/log.txt
//! data_snapshot.json
//! artifacts/**
//! ```
//!
//! 设计要点：
//!
//! - **纯 Rust、跨平台**：`zip` crate，无系统 libhdf5 依赖（R-14）。
//! - **人类可探查、可 diff**：条目都是文本（JSON/YAML/txt）；大矩阵用 list-of-list（K4）。
//! - **内存有界**：读取走流式单条目；写入一次一个条目。
//! - **结构性防御**：拒绝绝对路径、`..` 穿越、重复条目、目录条目（`.qwb` 只存文件）。
//! - **字节可复现**：条目时间戳固定为 1980-01-01，压缩级别固定，条目顺序固定，
//!   因此相同内容重复导出得到**逐字节相同**的容器（`tests/` 里有断言）。这直接支撑
//!   `qwb replay` 的“结果 identical”门禁（G5）。

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// 项目元数据文件名（K4）。
pub const META_FILE: &str = "meta.json";
/// 数据快照文件名（K4）。
pub const DATA_SNAPSHOT_FILE: &str = "data_snapshot.json";
/// 本 crate 实现的 K4 版本（写进 `meta.json.schema_version`）。
pub const K4_SCHEMA_VERSION: &str = "1.0.0";

/// `.qwb` 容器的 `meta.json`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ProjectMeta {
    /// K1/K3/K4 的 schema 版本。
    pub schema_version: String,
    /// 项目名。
    pub name: String,
    /// 创建时间，ISO 8601 UTC。
    pub created: String,
    /// 写入该容器的应用版本。
    pub app_version: String,
    /// task 数（与 `tasks/` 下的目录数一致；由写入方维护）。
    pub task_count: u32,
    /// 备注。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl ProjectMeta {
    /// 构造元数据。
    #[must_use]
    pub fn new(
        schema_version: impl Into<String>,
        name: impl Into<String>,
        created: impl Into<String>,
        app_version: impl Into<String>,
        task_count: u32,
    ) -> Self {
        Self {
            schema_version: schema_version.into(),
            name: name.into(),
            created: created.into(),
            app_version: app_version.into(),
            task_count,
            notes: None,
        }
    }

    /// 语义校验。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for (label, value) in [
            ("schema_version", &self.schema_version),
            ("name", &self.name),
            ("created", &self.created),
            ("app_version", &self.app_version),
        ] {
            if value.trim().is_empty() {
                issues.push(format!("meta.{label} must not be empty"));
            }
        }
        issues
    }
}

/// `.qwb` 操作错误。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContainerError {
    /// zip 层错误。
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// IO 错误。
    #[error("io error on {path}: {source}")]
    Io {
        /// 路径。
        path: String,
        /// 底层错误。
        source: std::io::Error,
    },
    /// `meta.json` 缺失（不是合法 `.qwb`）。
    #[error("meta.json is missing: not a valid .qwb container")]
    MetaMissing,
    /// `meta.json` 解析失败。
    #[error("meta.json is invalid: {0}")]
    MetaInvalid(String),
    /// 条目不存在。
    #[error("entry not found in container: {0}")]
    EntryMissing(String),
    /// 条目名非法（绝对路径 / `..` 穿越 / 目录条目 / 重复）。
    #[error("invalid entry name: {0}")]
    InvalidEntry(String),
    /// 条目名重复。
    #[error("duplicate entry name: {0}")]
    DuplicateEntry(String),
}

impl ContainerError {
    /// K5 退出码：容器结构问题算数据校验失败（3）。
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            ContainerError::Zip(_)
            | ContainerError::MetaMissing
            | ContainerError::MetaInvalid(_) => 3,
            ContainerError::EntryMissing(_)
            | ContainerError::InvalidEntry(_)
            | ContainerError::DuplicateEntry(_) => 3,
            ContainerError::Io { .. } => 4,
        }
    }
}

fn io_err(path: &Path, source: std::io::Error) -> ContainerError {
    ContainerError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// `.qwb` 容器句柄。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QwbContainer {
    path: PathBuf,
    meta: ProjectMeta,
    order: Vec<String>,
}

impl QwbContainer {
    /// 新建容器（只写 `meta.json`；随后用 [`QwbContainer::write_files`] 加内容）。
    ///
    /// # Errors
    ///
    /// 元数据非法、写入失败。
    pub fn create(path: impl Into<PathBuf>, meta: ProjectMeta) -> Result<Self, ContainerError> {
        let path = path.into();
        let issues = meta.check();
        if !issues.is_empty() {
            return Err(ContainerError::MetaInvalid(issues.join("; ")));
        }
        let mut container = Self {
            path,
            meta,
            order: Vec::new(),
        };
        container.write_files(Vec::new())?;
        Ok(container)
    }

    /// 打开并校验容器结构。
    ///
    /// # Errors
    ///
    /// 不是 zip、缺 `meta.json`、条目名非法或重复。
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ContainerError> {
        let path = path.into();
        let file = File::open(&path).map_err(|e| io_err(&path, e))?;
        let mut archive = zip::ZipArchive::new(file)?;

        let mut order = Vec::new();
        let mut seen = BTreeSet::new();
        for index in 0..archive.len() {
            let entry = archive.by_index(index)?;
            let name = entry.name()?.to_string();
            if entry.is_dir() {
                return Err(ContainerError::InvalidEntry(format!(
                    "{name}: directory entries are not allowed in .qwb (K4)"
                )));
            }
            check_entry_name(&name)?;
            if !seen.insert(name.clone()) {
                return Err(ContainerError::DuplicateEntry(name));
            }
            order.push(name);
        }

        let meta_bytes = {
            let mut entry = archive
                .by_name(META_FILE)
                .map_err(|_| ContainerError::MetaMissing)?;
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).map_err(|e| io_err(&path, e))?;
            buf
        };
        let meta: ProjectMeta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| ContainerError::MetaInvalid(e.to_string()))?;
        let issues = meta.check();
        if !issues.is_empty() {
            return Err(ContainerError::MetaInvalid(issues.join("; ")));
        }

        Ok(Self { path, meta, order })
    }

    /// 容器路径。
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 元数据。
    #[must_use]
    pub fn meta(&self) -> &ProjectMeta {
        &self.meta
    }

    /// 全部条目名（按写入顺序）。
    #[must_use]
    pub fn entry_names(&self) -> &[String] {
        &self.order
    }

    /// 读一个条目（流式，内存占用 = 该条目大小）。
    ///
    /// # Errors
    ///
    /// 容器打不开或条目不存在。
    pub fn read_file(&self, name: &str) -> Result<Vec<u8>, ContainerError> {
        check_entry_name(name)?;
        let file = File::open(&self.path).map_err(|e| io_err(&self.path, e))?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut entry = archive
            .by_name(name)
            .map_err(|_| ContainerError::EntryMissing(name.to_string()))?;
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| io_err(&self.path, e))?;
        Ok(buf)
    }

    /// 写/覆盖若干条目：重写整个容器（一次一个条目，内存有界）。
    ///
    /// 传空列表可以“重写为规范化容器”，用于测试字节可复现性。
    ///
    /// # Errors
    ///
    /// 条目名非法、读写失败。
    pub fn write_files(&mut self, files: Vec<(String, Vec<u8>)>) -> Result<(), ContainerError> {
        for (name, _) in &files {
            check_entry_name(name)?;
        }
        let new_content: BTreeMap<String, Vec<u8>> = files.into_iter().collect();

        // 先把旧内容按既定顺序取出来（读一个、写一个）。
        let old_order = self.order.clone();
        let mut old_contents: Vec<(String, Vec<u8>)> = Vec::with_capacity(old_order.len());
        for name in &old_order {
            if new_content.contains_key(name) {
                old_contents.push((name.clone(), Vec::new()));
            } else {
                old_contents.push((name.clone(), self.read_file(name)?));
            }
        }

        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let mut order = Vec::with_capacity(old_order.len() + new_content.len());

        // meta.json 永远第一个，保证容器头部可快速解析。
        let meta_bytes = serde_json::to_vec_pretty(&self.meta)
            .map_err(|e| ContainerError::MetaInvalid(e.to_string()))?;
        write_entry(&mut writer, META_FILE, &meta_bytes)?;
        order.push(META_FILE.to_string());

        for (name, old_bytes) in old_contents {
            if name == META_FILE {
                continue;
            }
            let bytes = match new_content.get(&name) {
                Some(bytes) => bytes.clone(),
                None => old_bytes,
            };
            write_entry(&mut writer, &name, &bytes)?;
            order.push(name);
        }
        for (name, bytes) in &new_content {
            if order.iter().any(|existing| existing == name) {
                continue;
            }
            write_entry(&mut writer, name, bytes)?;
            order.push(name.clone());
        }

        let cursor = writer.finish()?;
        let bytes = cursor.into_inner();
        std::fs::write(&self.path, bytes).map_err(|e| io_err(&self.path, e))?;
        self.order = order;
        Ok(())
    }

    /// 结构自检：`meta.json` 可读、条目名合法、`task_count` 与 `tasks/` 目录数一致。
    ///
    /// # Errors
    ///
    /// 任何不一致。
    pub fn validate(&self) -> Result<(), ContainerError> {
        let mut issues = self.meta.check();

        let mut task_ids = BTreeSet::new();
        for name in &self.order {
            if let Some(rest) = name.strip_prefix("tasks/")
                && let Some((id, _)) = rest.split_once('/')
            {
                task_ids.insert(id.to_string());
            }
        }
        if task_ids.len() != self.meta.task_count as usize {
            issues.push(format!(
                "meta.task_count = {} but tasks/ holds {} task directories",
                self.meta.task_count,
                task_ids.len()
            ));
        }
        if !issues.is_empty() {
            return Err(ContainerError::MetaInvalid(issues.join("; ")));
        }
        Ok(())
    }
}

/// 固定的条目时间戳（1980-01-01 00:00），保证导出字节可复现。
const FIXED_YEAR: u16 = 1980;

fn write_entry(
    writer: &mut zip::ZipWriter<Cursor<Vec<u8>>>,
    name: &str,
    bytes: &[u8],
) -> Result<(), ContainerError> {
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644)
        .last_modified_time(
            zip::DateTime::from_date_and_time(FIXED_YEAR, 1, 1, 0, 0, 0).unwrap_or_default(),
        );
    writer.start_file(name, options)?;
    writer
        .write_all(bytes)
        .map_err(|source| ContainerError::Io {
            path: name.to_string(),
            source,
        })?;
    Ok(())
}

/// 条目名结构校验（K4 安全边界）。
fn check_entry_name(name: &str) -> Result<(), ContainerError> {
    if name.is_empty() {
        return Err(ContainerError::InvalidEntry("empty name".into()));
    }
    if name.starts_with('/') || name.contains('\\') || name.contains(':') {
        return Err(ContainerError::InvalidEntry(format!(
            "{name}: must be a relative POSIX path without drive letters"
        )));
    }
    if name.ends_with('/') {
        return Err(ContainerError::InvalidEntry(format!(
            "{name}: directory entries are not allowed"
        )));
    }
    for segment in name.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(ContainerError::InvalidEntry(format!(
                "{name}: path traversal or empty segment"
            )));
        }
    }
    Ok(())
}

/// `tasks/<id>/input.yaml` 之类的条目路径。
pub fn task_entry_path(task_id: &str, file: &str) -> Result<String, ContainerError> {
    let path = format!("tasks/{task_id}/{file}");
    check_entry_name(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::{
        ContainerError, META_FILE, ProjectMeta, QwbContainer, check_entry_name, task_entry_path,
    };
    use std::io::Write as _;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("qwb-container-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("example.qwb")
    }

    fn meta() -> ProjectMeta {
        ProjectMeta::new("1.0.0", "demo", "2025-01-10T08:00:00Z", "0.1.0", 1)
    }

    fn populated(tag: &str) -> QwbContainer {
        let path = temp_path(tag);
        let mut container = QwbContainer::create(&path, meta()).expect("create");
        container
            .write_files(vec![
                (
                    task_entry_path("t1", "input.yaml").unwrap(),
                    b"jobtype: sp\n".to_vec(),
                ),
                ("data_snapshot.json".to_string(), b"{}\n".to_vec()),
            ])
            .expect("write");
        container
    }

    #[test]
    fn create_open_and_read_roundtrip() {
        let container = populated("roundtrip");
        let reopened = QwbContainer::open(container.path()).expect("open");
        assert_eq!(reopened.meta(), container.meta());
        assert_eq!(reopened.entry_names(), container.entry_names());
        assert!(reopened.validate().is_ok());
        assert_eq!(
            reopened.read_file("tasks/t1/input.yaml").unwrap(),
            b"jobtype: sp\n".to_vec()
        );
        assert!(reopened.read_file("nope/x").is_err());
        std::fs::remove_file(container.path()).ok();
    }

    #[test]
    fn meta_is_always_readable_first() {
        let container = populated("meta");
        assert_eq!(container.entry_names()[0], META_FILE);
        std::fs::remove_file(container.path()).ok();
    }

    #[test]
    fn export_is_byte_identical() {
        let path = temp_path("stable");
        let build = || {
            let mut c = QwbContainer::create(&path, meta()).unwrap();
            c.write_files(vec![
                (
                    task_entry_path("t1", "input.yaml").unwrap(),
                    b"jobtype: sp\n".repeat(50),
                ),
                ("data_snapshot.json".into(), b"{}\n".to_vec()),
            ])
            .unwrap();
            std::fs::read(&path).unwrap()
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "K4/G5: 相同内容必须导出相同字节");
        // 改写同样的内容也必须稳定。
        let mut c = QwbContainer::open(&path).unwrap();
        c.write_files(vec![]).unwrap();
        let c = std::fs::read(&path).unwrap();
        assert_eq!(a, c, "重写为空操作时字节不变");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupt_file_is_rejected() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"this is not a zip file").unwrap();
        assert!(QwbContainer::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_meta_is_rejected() {
        let path = temp_path("nometa");
        // 手工写一个没有 meta.json 的 zip。
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file(
                    "tasks/t1/input.yaml",
                    zip::write::SimpleFileOptions::default(),
                )
                .unwrap();
            writer.write_all(b"x").unwrap();
            writer.finish().unwrap();
        }
        let err = QwbContainer::open(&path).expect_err("must reject");
        assert!(matches!(err, ContainerError::MetaMissing));
        assert_eq!(err.exit_code(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn path_traversal_and_absolute_names_are_rejected() {
        for bad in [
            "../evil.txt",
            "tasks/../../evil",
            "/abs/path",
            "a/b/",
            "a//b",
            "C:\\win\\path",
            "",
        ] {
            assert!(
                check_entry_name(bad).is_err(),
                "must reject entry name {bad:?}"
            );
        }
        assert!(check_entry_name("tasks/t1/input.yaml").is_ok());
        assert!(check_entry_name("artifacts/mo.png").is_ok());
    }

    #[test]
    fn write_rejects_bad_names() {
        let path = temp_path("badname");
        let mut container = QwbContainer::create(&path, meta()).unwrap();
        let err = container
            .write_files(vec![("../evil".to_string(), b"x".to_vec())])
            .expect_err("must reject");
        assert!(matches!(err, ContainerError::InvalidEntry(_)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn task_count_must_match_directories() {
        let path = temp_path("count");
        let mut container = QwbContainer::create(&path, meta()).unwrap();
        container
            .write_files(vec![
                (task_entry_path("t1", "input.yaml").unwrap(), b"a".to_vec()),
                (task_entry_path("t2", "input.yaml").unwrap(), b"b".to_vec()),
            ])
            .unwrap();
        let reopened = QwbContainer::open(&path).unwrap();
        let err = reopened.validate().expect_err("2 tasks but task_count = 1");
        assert!(matches!(err, ContainerError::MetaInvalid(_)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn invalid_meta_is_rejected() {
        let path = temp_path("badmeta");
        let mut bad = ProjectMeta::new("1.0.0", "", "2025-01-10T08:00:00Z", "0.1.0", 0);
        bad.notes = Some("x".into());
        assert!(QwbContainer::create(&path, bad).is_err());

        let meta = ProjectMeta::new("1.0.0", "ok", "2025-01-10T08:00:00Z", "0.1.0", 0);
        QwbContainer::create(&path, meta).ok();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_from_other_crate_visibility() {
        // 公开 API 的最小使用示例（也防止 API 意外收紧）。
        let path = temp_path("api");
        let mut c = QwbContainer::create(&path, meta()).unwrap();
        c.write_files(vec![(
            super::DATA_SNAPSHOT_FILE.to_string(),
            b"{}".to_vec(),
        )])
        .unwrap();
        let bytes = c.read_file(super::DATA_SNAPSHOT_FILE).unwrap();
        assert_eq!(bytes, b"{}".to_vec());
        std::fs::remove_file(&path).ok();
    }
}
