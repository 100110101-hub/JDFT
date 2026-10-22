//! K3 结果目录布局 + G5 复现门禁。
//!
//! ```text
//! <task_id>/
//!   input.yaml
//!   env.json            # 缺它 = 不完整结果，UI 标红（K3 / G5）
//!   log.txt
//!   result.json
//!   data_snapshot.json
//!   artifacts/
//! ```

use std::path::{Path, PathBuf};

use qwb_engine_api::EnvInfo;

/// 结果目录里的必需文件（K3 冻结）。
pub const REQUIRED_FILES: [&str; 4] =
    ["input.yaml", "result.json", "log.txt", "data_snapshot.json"];

/// 复现环境文件（G5：没有它 → 结果不完整）。
pub const ENV_FILE: &str = "env.json";

/// 产物目录。
pub const ARTIFACTS_DIR: &str = "artifacts";

/// 布局检查错误。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// 目录不存在或不是目录。
    #[error("task directory is missing: {0}")]
    MissingDir(String),
    /// 缺少 `env.json`（K3：视为不完整结果；G5 退出码 3）。
    #[error("env.json is missing: {0} (result is not reproducible, K3/G5)")]
    MissingEnv(String),
    /// 缺少其它必需文件。
    #[error("missing required file(s) in {dir}: {files}")]
    MissingFiles {
        /// 目录。
        dir: String,
        /// 缺失文件名，逗号分隔。
        files: String,
    },
    /// `env.json` 内容不完整。
    #[error("env.json is incomplete: {0}")]
    IncompleteEnv(String),
    /// 读取失败。
    #[error("cannot read {path}: {source}")]
    Io {
        /// 路径。
        path: String,
        /// 底层错误。
        source: std::io::Error,
    },
    /// JSON 解析失败。
    #[error("cannot parse {path}: {source}")]
    Json {
        /// 路径。
        path: String,
        /// 底层错误。
        source: serde_json::Error,
    },
}

impl LayoutError {
    /// K5 退出码映射：布局/复现问题一律算“数据校验失败”（3）。
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        3
    }
}

/// 一个 task 结果目录。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResultDir {
    root: PathBuf,
}

impl ResultDir {
    /// 绑定一个目录（不做 IO）。
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// 目录路径。
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 文件是否存在。
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.root.join(name).is_file()
    }

    /// 是否有 `env.json`（G5 的最小判据）。
    #[must_use]
    pub fn has_env_json(&self) -> bool {
        self.has(ENV_FILE)
    }

    /// 缺失的必需文件（不含 `env.json`，它单独成条以对应 G5 文案）。
    #[must_use]
    pub fn missing_required(&self) -> Vec<&'static str> {
        REQUIRED_FILES
            .into_iter()
            .filter(|name| !self.has(name))
            .collect()
    }

    /// 读取并校验 `env.json`。
    ///
    /// # Errors
    ///
    /// 文件缺失、解析失败、[`EnvInfo::validate`] 不通过。
    pub fn read_env(&self) -> Result<EnvInfo, LayoutError> {
        let path = self.root.join(ENV_FILE);
        if !path.is_file() {
            return Err(LayoutError::MissingEnv(self.root.display().to_string()));
        }
        let bytes = std::fs::read(&path).map_err(|source| LayoutError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let env: EnvInfo = serde_json::from_slice(&bytes).map_err(|source| LayoutError::Json {
            path: path.display().to_string(),
            source,
        })?;
        env.validate()
            .map_err(|err| LayoutError::IncompleteEnv(err.to_string()))?;
        Ok(env)
    }

    /// 完整布局检查（`qwb inspect` 的后端；G5）。
    ///
    /// # Errors
    ///
    /// 目录缺失、缺 `env.json`、缺其它必需文件、`env.json` 不完整。
    pub fn check(&self) -> Result<(), LayoutError> {
        if !self.root.is_dir() {
            return Err(LayoutError::MissingDir(self.root.display().to_string()));
        }
        if !self.has_env_json() {
            return Err(LayoutError::MissingEnv(self.root.display().to_string()));
        }
        let missing = self.missing_required();
        if !missing.is_empty() {
            return Err(LayoutError::MissingFiles {
                dir: self.root.display().to_string(),
                files: missing.join(", "),
            });
        }
        self.read_env()?;
        Ok(())
    }

    /// `artifacts/` 目录（不存在时返回路径，不创建）。
    #[must_use]
    pub fn artifacts_dir(&self) -> PathBuf {
        self.root.join(ARTIFACTS_DIR)
    }
}

#[cfg(test)]
mod tests {
    use super::{ENV_FILE, LayoutError, ResultDir};
    use qwb_engine_api::{DataSnapshot, EnvInfo};

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "qwb-layout-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    use std::path::PathBuf;

    fn complete_env() -> EnvInfo {
        EnvInfo::new(
            "0.1.0",
            "rustc 1.97.1",
            "a".repeat(64),
            DataSnapshot::new(),
            4,
            "windows 11 x86_64",
            "2025-01-10T08:00:00Z",
        )
    }

    fn populate(dir: &std::path::Path, with_env: bool) {
        for name in ["input.yaml", "result.json", "log.txt", "data_snapshot.json"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        if with_env {
            let json = serde_json::to_vec(&complete_env()).unwrap();
            std::fs::write(dir.join(ENV_FILE), json).unwrap();
        }
    }

    #[test]
    fn complete_result_dir_passes() {
        let dir = temp_dir("ok");
        populate(&dir, true);
        let layout = ResultDir::new(&dir);
        assert!(layout.has_env_json());
        assert!(layout.missing_required().is_empty());
        assert!(layout.check().is_ok());
        assert_eq!(layout.read_env().unwrap().threads, 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_env_json_fails_with_g5_error() {
        let dir = temp_dir("noenv");
        populate(&dir, false);
        let layout = ResultDir::new(&dir);
        let err = layout.check().expect_err("must fail without env.json");
        assert!(matches!(err, LayoutError::MissingEnv(_)));
        assert_eq!(err.exit_code(), 3, "K5/T7: 无 env.json → 退出码 3");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_other_files_are_reported() {
        let dir = temp_dir("partial");
        populate(&dir, true);
        std::fs::remove_file(dir.join("result.json")).unwrap();
        let err = ResultDir::new(&dir).check().expect_err("must fail");
        match err {
            LayoutError::MissingFiles { files, .. } => assert!(files.contains("result.json")),
            other => panic!("unexpected {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn incomplete_env_is_rejected() {
        let dir = temp_dir("badenv");
        populate(&dir, true);
        let mut env = complete_env();
        env.threads = 0;
        std::fs::write(dir.join(ENV_FILE), serde_json::to_vec(&env).unwrap()).unwrap();
        let err = ResultDir::new(&dir).check().expect_err("must fail");
        assert!(matches!(err, LayoutError::IncompleteEnv(_)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_directory_is_reported() {
        let layout = ResultDir::new(std::env::temp_dir().join("qwb-layout-nope-xyz"));
        assert!(matches!(layout.check(), Err(LayoutError::MissingDir(_))));
    }
}
