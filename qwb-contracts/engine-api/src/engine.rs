//! [`Engine`] 门面与后端注入（K2）。
//!
//! K2 把 `Engine` 放在契约 crate，但引擎实现（积分/XC/SCF）在 `qwb-core` 各 crate。
//! 因此这里把 `Engine` 定义成**门面**：它持有 `Box<dyn EngineBackend>`，
//! 由顶层 `qwb-engine` 通过 [`Engine::with_backend`] 注入。契约 crate 不含任何
//! 数值实现，也不解析数据文件。

use std::fmt;
use std::path::PathBuf;

use crate::error::EngineError;
use crate::input::JobInput;
use crate::registry::Registry;
use crate::reporter::Reporter;
use crate::result::JobResult;

/// 第一阶段契约声明支持的 feature 词汇表（`project.md` §2.1 范围，§2.2 非目标之外）。
///
/// `Registry` 声明了不在此表中的 feature（如 `functional:meta-gga`、`tddft`、
/// `solvent:smd`）时，[`Engine::new`] 返回 [`EngineError::Unsupported`]。
///
/// 注意：**声明支持 ≠ 本构建实现了**。后端是否真的能算，由 [`EngineBackend::run`]
/// 再判一次（返回 `Unsupported` 或 `Data`）。
pub const SUPPORTED_FEATURES: &[&str] = &[
    "jobtype:sp",
    "jobtype:opt",
    "jobtype:freq",
    "basis:orbital",
    "basis:ri-j",
    "basis:ri-jk",
    "basis:ecp-fit",
    "functional:lda",
    "functional:gga",
    "functional:hybrid",
    "grid",
    "solvent:vacuum",
    "solvent:implicit",
    "ecp",
];

/// 引擎后端：真正的数值实现。
///
/// 由 `qwb-engine`（顶层实现）提供。`Send + Sync` 是 K2 冻结要求：
/// 同一 `Engine` 可被多个任务线程共享，取消走 `&self`。
pub trait EngineBackend: Send + Sync {
    /// 执行一次作业。
    ///
    /// # Errors
    ///
    /// 数据问题返回 [`EngineError::Data`]，不收敛返回
    /// [`EngineError::ScfNotConverged`]，未实现返回 [`EngineError::Unsupported`]。
    fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError>;

    /// 请求取消正在跑的作业（异步意图，不阻塞）。
    fn request_cancel(&self);
}

/// 引擎门面（字段不透明，K2）。
pub struct Engine {
    registry_root: PathBuf,
    backend: Option<Box<dyn EngineBackend>>,
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("registry_root", &self.registry_root)
            .field("has_backend", &self.backend.is_some())
            .finish()
    }
}

impl Engine {
    /// 用数据注册表构造引擎。
    ///
    /// 本 crate 是**契约 crate**，不含后端实现，因此本方法在数据根合法、feature
    /// 合法之后**必然返回** [`EngineError::Unsupported`]（提示用
    /// [`Engine::with_backend`] 注入实现）。这正是 C-02 验收要求的
    /// “`Engine::new` 可返回 `Unsupported` 错”。
    ///
    /// # Errors
    ///
    /// - 数据根目录不存在 → [`EngineError::Data`]
    /// - 声明了 [`SUPPORTED_FEATURES`] 之外的 feature → [`EngineError::Unsupported`]
    /// - 无后端（恒定） → [`EngineError::Unsupported`]
    pub fn new(registry: &Registry) -> Result<Self, EngineError> {
        let registry_root = check_registry(registry)?;
        Err(EngineError::Unsupported(format!(
            "no engine backend linked in this build (K2 contract crate) for registry root {}; \
             link qwb-engine with Engine::with_backend",
            registry_root.display()
        )))
    }

    /// 用注册表 + 后端构造引擎（`qwb-engine` 的入口）。
    ///
    /// # Errors
    ///
    /// 同 [`Engine::new`] 的前两项。
    pub fn with_backend(
        registry: &Registry,
        backend: Box<dyn EngineBackend>,
    ) -> Result<Self, EngineError> {
        let registry_root = check_registry(registry)?;
        Ok(Self {
            registry_root,
            backend: Some(backend),
        })
    }

    /// 数据根目录。
    #[must_use]
    pub fn registry_root(&self) -> &std::path::Path {
        &self.registry_root
    }

    /// 执行一次作业。
    ///
    /// 门面只做输入校验与转发；数值逻辑全在 [`EngineBackend`]。
    ///
    /// # Errors
    ///
    /// - 输入非法 → [`EngineError::InvalidInput`]（`Reporter::is_cancelled` 为真时
    ///   不进入后端）
    /// - 无后端 → [`EngineError::Internal`]
    /// - 其余由后端决定
    pub fn run(&self, input: &JobInput, reporter: &dyn Reporter) -> Result<JobResult, EngineError> {
        input.validate()?;
        if reporter.is_cancelled() {
            return Err(EngineError::InvalidInput(
                "job cancelled before start".into(),
            ));
        }
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| EngineError::Internal("engine has no backend".into()))?;
        backend.run(input, reporter)
    }

    /// 请求取消。
    pub fn request_cancel(&self) {
        if let Some(backend) = &self.backend {
            backend.request_cancel();
        }
    }
}

/// 校验注册表句柄，返回数据根。
fn check_registry(registry: &Registry) -> Result<PathBuf, EngineError> {
    let root = registry.root().to_path_buf();
    if !root.is_dir() {
        return Err(EngineError::Data(format!(
            "registry root is not a directory: {}",
            root.display()
        )));
    }
    for feature in registry.declared_features() {
        if !SUPPORTED_FEATURES.contains(&feature) {
            return Err(EngineError::Unsupported(format!(
                "registry declares feature `{feature}` outside the first-phase scope"
            )));
        }
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::{Engine, EngineBackend, SUPPORTED_FEATURES};
    use crate::error::EngineError;
    use crate::input::{GridLevel, JobInput, JobType, TaskOptions};
    use crate::molecule::{Atom, Molecule};
    use crate::registry::Registry;
    use crate::reporter::{NullReporter, Reporter};
    use crate::result::JobResult;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("qwb-k2-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp registry root");
        dir
    }

    fn input() -> JobInput {
        JobInput::new(
            Molecule::new(
                vec![
                    Atom::new("H", [0.0, 0.0, -0.37]),
                    Atom::new("H", [0.0, 0.0, 0.37]),
                ],
                0,
                1,
            ),
            "sto-3g",
            "svwn5",
            GridLevel::Normal,
            "vacuum",
            JobType::Sp,
            TaskOptions::default(),
        )
    }

    #[derive(Default)]
    struct CancelledReporter(AtomicBool);

    impl Reporter for CancelledReporter {
        fn progress(&self, _stage: &str, _pct: f64) {}
        fn log(&self, _line: &str) {}
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }

    struct DummyBackend {
        cancelled: Arc<AtomicBool>,
    }

    impl EngineBackend for DummyBackend {
        fn run(
            &self,
            _input: &JobInput,
            reporter: &dyn Reporter,
        ) -> Result<JobResult, EngineError> {
            reporter.progress("scf", 100.0);
            Err(EngineError::Internal("dummy backend".into()))
        }

        fn request_cancel(&self) {
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn engine_new_rejects_missing_registry_root() {
        let missing = std::env::temp_dir().join("qwb-k2-does-not-exist-xyz");
        let registry = Registry::new(missing);
        let err = Engine::new(&registry).expect_err("must fail");
        assert!(matches!(err, EngineError::Data(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 3);
    }

    #[test]
    fn engine_new_returns_unsupported_for_out_of_scope_feature() {
        let registry =
            Registry::new(temp_root("scope")).with_declared_features(["functional:meta-gga"]);
        let err = Engine::new(&registry).expect_err("must fail");
        assert!(matches!(err, EngineError::Unsupported(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 4);
        assert!(!SUPPORTED_FEATURES.contains(&"functional:meta-gga"));
    }

    #[test]
    fn engine_new_without_backend_is_unsupported() {
        let registry = Registry::new(temp_root("nobackend")).with_declared_features(["ecp"]);
        let err = Engine::new(&registry).expect_err("must fail");
        match err {
            EngineError::Unsupported(msg) => assert!(msg.contains("no engine backend")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn engine_with_backend_delegates_and_propagates_cancel() {
        let registry = Registry::new(temp_root("backend"));
        let flag = Arc::new(AtomicBool::new(false));
        let engine = Engine::with_backend(
            &registry,
            Box::new(DummyBackend {
                cancelled: Arc::clone(&flag),
            }),
        )
        .expect("backend links");
        assert_eq!(engine.registry_root(), registry.root());

        let err = engine
            .run(&input(), &NullReporter)
            .expect_err("dummy backend errors");
        assert!(matches!(err, EngineError::Internal(_)));

        engine.request_cancel();
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn run_validates_input_before_touching_backend() {
        let registry = Registry::new(temp_root("validate"));
        let engine = Engine::with_backend(
            &registry,
            Box::new(DummyBackend {
                cancelled: Arc::new(AtomicBool::new(false)),
            }),
        )
        .expect("backend links");

        let mut bad = input();
        bad.basis_name = String::new();
        let err = engine.run(&bad, &NullReporter).expect_err("invalid input");
        assert!(matches!(err, EngineError::InvalidInput(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn cancelled_before_start_does_not_run() {
        let registry = Registry::new(temp_root("cancel"));
        let engine = Engine::with_backend(
            &registry,
            Box::new(DummyBackend {
                cancelled: Arc::new(AtomicBool::new(false)),
            }),
        )
        .expect("backend links");

        let reporter = CancelledReporter(AtomicBool::new(true));
        let err = engine.run(&input(), &reporter).expect_err("cancelled");
        assert!(matches!(err, EngineError::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
        assert_send_sync::<Registry>();
        assert_send_sync::<JobResult>();
        assert_send_sync::<JobInput>();
    }
}
