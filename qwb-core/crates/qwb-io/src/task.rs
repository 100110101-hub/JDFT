//! K3 Task 状态机（冻结）。
//!
//! ```text
//! draft → queued → running → converged
//!                    │
//!                    ├→ failed
//!                    └→ cancelled
//! ```
//!
//! **本模块只实现 K3 图上有的 5 条转移。** 图上没有的转移（如 `queued → cancelled`、
//! `failed → queued` 重试）一律拒绝，并在 `docs/choices.md` 里作为待人类裁决项登记
//! —— 按 §10.2.2，agent 不得自行扩展 K3。

use serde::{Deserialize, Serialize};

use qwb_engine_api::JobInput;

use crate::task::TransitionError::IllegalTransition;

/// Task 状态（K3 冻结；序列化为小写字符串）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum TaskState {
    /// 已创建、参数未定稿。
    Draft,
    /// 已入队。
    Queued,
    /// 正在计算。
    Running,
    /// 已收敛（终态）。
    Converged,
    /// 失败（终态）。
    Failed,
    /// 已取消（终态）。
    Cancelled,
}

impl TaskState {
    /// 全部状态。
    pub const ALL: [TaskState; 6] = [
        TaskState::Draft,
        TaskState::Queued,
        TaskState::Running,
        TaskState::Converged,
        TaskState::Failed,
        TaskState::Cancelled,
    ];

    /// 冻结的合法转移集合（与 K3 图逐条对应）。
    pub const LEGAL: [(TaskState, TaskState); 5] = [
        (TaskState::Draft, TaskState::Queued),
        (TaskState::Queued, TaskState::Running),
        (TaskState::Running, TaskState::Converged),
        (TaskState::Running, TaskState::Failed),
        (TaskState::Running, TaskState::Cancelled),
    ];

    /// 小写名（与 `task.schema.yaml` 的取值一致）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::Draft => "draft",
            TaskState::Queued => "queued",
            TaskState::Running => "running",
            TaskState::Converged => "converged",
            TaskState::Failed => "failed",
            TaskState::Cancelled => "cancelled",
        }
    }

    /// 是否终态。
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Converged | TaskState::Failed | TaskState::Cancelled
        )
    }

    /// 是否允许转移到 `next`。
    #[must_use]
    pub fn can_transition_to(self, next: TaskState) -> bool {
        Self::LEGAL.contains(&(self, next))
    }

    /// 执行转移。
    ///
    /// # Errors
    ///
    /// K3 图上没有的转移返回 [`TransitionError::IllegalTransition`]。
    pub fn transition_to(self, next: TaskState) -> Result<TaskState, TransitionError> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(IllegalTransition {
                from: self,
                to: next,
            })
        }
    }
}

/// 状态转移错误。
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TransitionError {
    /// 非法转移。
    #[error("illegal task transition: {} -> {}", from.as_str(), to.as_str())]
    IllegalTransition {
        /// 起始状态。
        from: TaskState,
        /// 目标状态。
        to: TaskState,
    },
    /// 终态不可再变（与上者等价，但语义更明确，供 UI 提示）。
    #[error("task is already in terminal state {}", state.as_str())]
    AlreadyTerminal {
        /// 当前（终态）状态。
        state: TaskState,
    },
}

/// Task 记录（`tasks/<id>/` 的元数据；K3 `task.schema.yaml` 的 Rust 对应物）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TaskRecord {
    /// Task id（`tasks/<id>/` 的目录名）。
    pub id: String,
    /// 显示名。
    pub name: String,
    /// 当前状态。
    pub state: TaskState,
    /// 创建时间，ISO 8601 UTC。
    pub created: String,
    /// 最近一次状态变更时间，ISO 8601 UTC。
    pub updated: String,
    /// 作业输入（`input.yaml` 的内容）。
    pub input: JobInput,
}

impl TaskRecord {
    /// 新建 draft 任务。
    #[must_use]
    pub fn draft(
        id: impl Into<String>,
        name: impl Into<String>,
        now: impl Into<String>,
        input: JobInput,
    ) -> Self {
        let now = now.into();
        Self {
            id: id.into(),
            name: name.into(),
            state: TaskState::Draft,
            created: now.clone(),
            updated: now,
            input,
        }
    }

    /// 按 K3 状态机推进状态并刷新 `updated`。
    ///
    /// # Errors
    ///
    /// 非法转移。
    pub fn set_state(
        &mut self,
        next: TaskState,
        now: impl Into<String>,
    ) -> Result<(), TransitionError> {
        if self.state.is_terminal() {
            return Err(TransitionError::AlreadyTerminal { state: self.state });
        }
        self.state = self.state.transition_to(next)?;
        self.updated = now.into();
        Ok(())
    }

    /// 语义校验。
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.id.trim().is_empty() {
            issues.push("id must not be empty".into());
        }
        if self.id.contains(['/', '\\', '.']) {
            issues.push("id must not contain path separators or dots".into());
        }
        if self.name.trim().is_empty() {
            issues.push("name must not be empty".into());
        }
        if self.created.trim().is_empty() || self.updated.trim().is_empty() {
            issues.push("created/updated must not be empty".into());
        }
        if let Err(err) = self.input.validate() {
            issues.push(format!("input: {err}"));
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::{TaskRecord, TaskState, TransitionError};
    use qwb_engine_api::{Atom, GridLevel, JobInput, JobType, Molecule, TaskOptions};

    fn input() -> JobInput {
        JobInput::new(
            Molecule::new(vec![Atom::new("H", [0.0, 0.0, 0.0])], 0, 2),
            "sto-3g",
            "svwn5",
            GridLevel::Normal,
            "vacuum",
            JobType::Sp,
            TaskOptions::default(),
        )
    }

    #[test]
    fn k3_legal_transitions_only() {
        assert!(TaskState::Draft.transition_to(TaskState::Queued).is_ok());
        assert!(TaskState::Queued.transition_to(TaskState::Running).is_ok());
        assert!(
            TaskState::Running
                .transition_to(TaskState::Converged)
                .is_ok()
        );
        assert!(TaskState::Running.transition_to(TaskState::Failed).is_ok());
        assert!(
            TaskState::Running
                .transition_to(TaskState::Cancelled)
                .is_ok()
        );

        // K3 图上没有的：
        for (from, to) in [
            (TaskState::Queued, TaskState::Cancelled),
            (TaskState::Draft, TaskState::Running),
            (TaskState::Failed, TaskState::Queued),
            (TaskState::Converged, TaskState::Running),
            (TaskState::Cancelled, TaskState::Draft),
        ] {
            assert!(
                !from.can_transition_to(to),
                "{} -> {} must be illegal",
                from.as_str(),
                to.as_str()
            );
        }
    }

    #[test]
    fn terminal_states_are_sticky() {
        for state in [
            TaskState::Converged,
            TaskState::Failed,
            TaskState::Cancelled,
        ] {
            assert!(state.is_terminal());
            let mut record = TaskRecord::draft("t1", "n", "2025-01-10T08:00:00Z", input());
            record.state = state;
            let err = record
                .set_state(TaskState::Running, "2025-01-10T09:00:00Z")
                .expect_err("terminal is sticky");
            assert!(matches!(err, TransitionError::AlreadyTerminal { .. }));
        }
    }

    #[test]
    fn record_walks_the_happy_path() {
        let mut record = TaskRecord::draft("t1", "H2 sp", "2025-01-10T08:00:00Z", input());
        record
            .set_state(TaskState::Queued, "2025-01-10T08:00:01Z")
            .unwrap();
        record
            .set_state(TaskState::Running, "2025-01-10T08:00:02Z")
            .unwrap();
        record
            .set_state(TaskState::Converged, "2025-01-10T08:00:03Z")
            .unwrap();
        assert_eq!(record.state, TaskState::Converged);
        assert_eq!(record.updated, "2025-01-10T08:00:03Z");
        assert!(record.check().is_empty(), "{:?}", record.check());
    }

    #[test]
    fn record_check_rejects_path_injection() {
        let mut record = TaskRecord::draft("../x", "n", "2025-01-10T08:00:00Z", input());
        record.state = TaskState::Running;
        assert!(record.check().iter().any(|m| m.contains("path separators")));
    }

    #[test]
    fn task_state_serde_is_lowercase() {
        assert_eq!(
            serde_json::to_string(&TaskState::Converged).unwrap(),
            "\"converged\""
        );
        assert_eq!(TaskState::Draft.as_str(), "draft");
    }
}
