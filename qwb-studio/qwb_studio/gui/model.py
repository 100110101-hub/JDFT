"""GUI 的项目模型：把 P-03 的 `Scheduler` 包成 Qt 可观察对象。

GUI 只做三件事：登记任务（`submit`）、发取消（`cancel`）、轮询队列状态并广播。
真正的计算在调度器的子进程里，所以界面线程永远是空闲的。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..models import TaskState
from ..scheduler import Scheduler


@dataclass(frozen=True)
class TaskView:
    """界面上展示一个 task 需要的全部字段。"""

    task_id: str
    state: str
    reason: str | None
    seconds: float


class GuiProject(QObject):
    """一个工程（out 目录 + queue.json）的界面侧模型。"""

    taskAdded = Signal(str)
    taskStateChanged = Signal(str, str)  # task_id, state
    projectChanged = Signal()

    def __init__(
        self,
        out_root: str | Path,
        *,
        workers: int = 2,
        runner: str = "mock",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.out_root = Path(out_root)
        self.workers = workers
        self.runner = runner
        self.scheduler = Scheduler(
            queue_path=self.out_root / "queue.json",
            out_root=self.out_root,
            workers=workers,
            runner_name=runner,
            grace_seconds=0.5,
        )
        self.scheduler.load()
        self._payloads: dict[str, float] = {}

    # -- 查询 ---------------------------------------------------------------

    def tasks(self) -> list[TaskView]:
        return [
            TaskView(
                task_id=entry.task_id,
                state=entry.state.value,
                reason=entry.reason,
                seconds=self._payloads.get(entry.task_id, 0.0),
            )
            for entry in self.scheduler.entries
        ]

    def state_of(self, task_id: str) -> TaskState:
        return self.scheduler.state_of(task_id)

    # -- 变更 ---------------------------------------------------------------

    def submit(self, task_id: str, seconds: float = 0.2, label: str | None = None) -> TaskView:
        self.scheduler.submit(task_id, f"{task_id}.mock", {"seconds": seconds, "label": label})
        self._payloads[task_id] = seconds
        self.taskAdded.emit(task_id)
        self.projectChanged.emit()
        return TaskView(task_id, "queued", None, seconds)

    def cancel(self, task_id: str) -> None:
        try:
            self.scheduler.request_cancel(task_id)
        finally:
            self.projectChanged.emit()

    def run(self) -> dict[str, Any]:
        """在**调用线程**里跑完所有 queued（由 runner 放到工作线程里调）。"""
        return self.scheduler.run()

    def sync(self) -> None:
        """把磁盘上的状态同步到界面（工作线程跑完之后调用）。"""
        self.scheduler.load()
        for view in self.tasks():
            self.taskStateChanged.emit(view.task_id, view.state)
        self.projectChanged.emit()
