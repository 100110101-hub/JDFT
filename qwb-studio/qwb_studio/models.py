"""K3 契约的 pydantic v2 模型（P-01）。

字段与 `qwb-contracts/studio/{task,result}.schema.yaml` 一一对应；状态机与
`qwb-core/crates/qwb-io/src/task.rs` 的 `TaskState::LEGAL` 一致（6 条合法转移）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IllegalTransitionError(ValueError):
    """非法状态转移（P-01 验收：必须抛 `ValueError`）。"""


class TaskState(str, Enum):
    """K3 状态机的状态（小写名与 `task.schema.yaml` 的 enum 逐字一致）。"""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def terminal(cls) -> frozenset[TaskState]:
        """终态：没有出边。"""
        return frozenset({cls.CONVERGED, cls.FAILED, cls.CANCELLED})

    def is_terminal(self) -> bool:
        return self in self.terminal()

    def can_transition_to(self, nxt: TaskState) -> bool:
        return (self, nxt) in LEGAL_TRANSITIONS

    def transition_to(self, nxt: TaskState) -> TaskState:
        """执行转移；非法转移抛 :class:`IllegalTransitionError`（`ValueError` 的子类）。"""
        if not self.can_transition_to(nxt):
            raise IllegalTransitionError(f"illegal transition {self.value} -> {nxt.value}")
        return nxt


# 与 Rust `TaskState::LEGAL` 逐条对应（v2.1 起 6 条，含两条 cancelled 边）。
LEGAL_TRANSITIONS: frozenset[tuple[TaskState, TaskState]] = frozenset(
    {
        (TaskState.DRAFT, TaskState.QUEUED),
        (TaskState.QUEUED, TaskState.RUNNING),
        (TaskState.QUEUED, TaskState.CANCELLED),
        (TaskState.RUNNING, TaskState.CONVERGED),
        (TaskState.RUNNING, TaskState.FAILED),
        (TaskState.RUNNING, TaskState.CANCELLED),
    }
)


def utc_now_iso() -> str:
    """ISO 8601 UTC（与 Rust 侧 `created`/`updated` 的写法一致）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Atom(BaseModel):
    """一个原子（与 `molecule.schema.yaml` 对应）。"""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(description="元素符号，区分大小写")
    position_angstrom: tuple[float, float, float] = Field(description="Å 坐标")

    @field_validator("symbol")
    @classmethod
    def _symbol_not_empty(cls, value: str) -> str:
        if not value or not value[0].isalpha():
            raise ValueError("atom symbol must start with a letter")
        return value


class Molecule(BaseModel):
    """分子（K1 molecule 契约的最小形态 + K2 的 JobInput 所需字段）。"""

    model_config = ConfigDict(extra="forbid")

    atoms: list[Atom] = Field(min_length=1)
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)


class Task(BaseModel):
    """一个 task 的元数据（`tasks/<id>/task.json`，对应 `task.schema.yaml`）。

    六个字段全部必填 —— 与契约的 `required` 列表严格一致；新建任务请用
    :meth:`Task.new`，它会填上 `draft` 与当前时间戳。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str
    state: TaskState
    created: str
    updated: str
    input: dict[str, Any]

    @classmethod
    def new(cls, id: str, name: str, input: dict[str, Any]) -> "Task":  # noqa: A002
        """新建一个 `draft` 任务；`created` 与 `updated` 取同一时刻。"""
        stamp = utc_now_iso()
        return cls(id=id, name=name, state=TaskState.DRAFT, created=stamp, updated=stamp, input=input)

    def transition_to(self, nxt: TaskState) -> "Task":
        """转移并把 `updated` 刷新到当前时间（返回新对象，保持不可变语义）。"""
        state = self.state.transition_to(nxt)
        return self.model_copy(update={"state": state, "updated": utc_now_iso()})


class EnvInfo(BaseModel):
    """复现环境（`result.schema.yaml` 的 `EnvInfo`；G5：缺它结果视为不完整）。"""

    model_config = ConfigDict(extra="forbid")

    engine_version: str
    rustc_version: str
    cargo_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_snapshot: dict[str, str]
    threads: int = Field(ge=1)
    os: str
    created_at_utc: str
    toolchain_channel: str | None = None
    crate_versions: dict[str, str] = Field(default_factory=dict)

    def check(self) -> list[str]:
        """G5 复现门禁，判据与 Rust `EnvInfo::validate` 逐条相同（返回问题清单，不抛）。"""
        issues: list[str] = []
        for label in ("engine_version", "rustc_version", "os", "created_at_utc"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"env.{label} is empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.cargo_lock_sha256 or ""):
            issues.append("env.cargo_lock_sha256 is not a sha256 hex")
        if not isinstance(self.threads, int) or self.threads < 1:
            issues.append("env.threads must be >= 1")
        if not isinstance(self.data_snapshot, dict):
            issues.append("env.data_snapshot must be a mapping")
        else:
            for path, digest in self.data_snapshot.items():
                if not isinstance(path, str) or not path:
                    issues.append("env.data_snapshot has an empty path")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    issues.append(f"env.data_snapshot[{path}] is not a 64-char lowercase sha256 hex")
        if not isinstance(self.crate_versions, dict):
            issues.append("env.crate_versions must be a mapping")
        return issues

    def validate(self) -> None:
        issues = self.check()
        if issues:
            raise ValueError("env.json is incomplete: " + "; ".join(issues))


class Result(BaseModel):
    """结果（`result.json` + `env.json` 的合并视图，对应 `result.schema.yaml` 的 `JobResult`）。"""

    model_config = ConfigDict(extra="forbid")

    energy_hartree: float
    scf_converged: bool
    data_snapshot: dict[str, str]
    log: str
    env: EnvInfo
    gradient_hartree_per_bohr: list[list[float]] | None = None
    hessian_hartree_per_bohr2: list[list[float]] | None = None
    dipole_debye: list[float] | None = None
    mo_energies_hartree: list[float] | None = None
    mo_coeffs: list[list[float]] | None = None


class Project(BaseModel):
    """一个 Studio 工程（P-01 的顶层对象；`.qwb` 容器的元数据见 P-02）。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = "0.1.0"
    tasks: list[Task] = Field(default_factory=list)

    def add_task(self, task: Task) -> None:
        if any(existing.id == task.id for existing in self.tasks):
            raise ValueError(f"duplicate task id {task.id!r}")
        self.tasks.append(task)
