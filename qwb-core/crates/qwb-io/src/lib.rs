//! QWB 文件层：K3（Task 状态机 + 结果目录布局）与 K4（`.qwb` Zip 容器）。
//!
//! 这一层只做**结构**与**契约**，不做数值：`result.json` 的内容由 `qwb-engine-api`
//! 的 [`qwb_engine_api::JobResult`] 决定，这里只负责“它在哪个文件、缺了算不算完整”。
#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod container;
pub mod layout;
pub mod task;

pub use container::{ContainerError, ProjectMeta, QwbContainer};
pub use layout::{LayoutError, ResultDir};
pub use task::{TaskRecord, TaskState, TransitionError};
