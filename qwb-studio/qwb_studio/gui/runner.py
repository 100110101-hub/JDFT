"""把调度器放到工作线程里跑，界面线程只收信号。

`SchedulerBridge` 本身是 `QObject`（住在界面线程），它持有一个 `QThread`，
工作线程里跑 `GuiProject.run()`；运行期间用一个 `QTimer` 轮询队列文件并广播状态变化。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal


class _Runner(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self._project = project

    def run(self) -> None:
        try:
            outcome = self._project.run()
        except BaseException as exc:  # noqa: BLE001 - 线程里的任何异常都要报出去
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(outcome)


class SchedulerBridge(QObject):
    """界面侧的执行器：`start()` 后 UI 立即返回，结果通过信号到达。"""

    stateChanged = Signal(str, str)
    runFinished = Signal(dict)
    runFailed = Signal(str)

    def __init__(self, project, parent: QObject | None = None, poll_ms: int = 50) -> None:
        super().__init__(parent)
        self.project = project
        self._thread: QThread | None = None
        self._runner: _Runner | None = None
        self._poller = QTimer(self)
        self._poller.setInterval(poll_ms)
        self._poller.timeout.connect(self._poll)
        self._last_states: dict[str, str] = {}

    # -- 生命周期 -----------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> None:
        if self.is_running():
            return
        self._last_states = {view.task_id: view.state for view in self.project.tasks()}
        self._thread = QThread()
        self._runner = _Runner(self.project)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.finished.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)
        self._poller.start()
        self._thread.start()

    def wait(self, timeout_ms: int = 600_000) -> bool:
        """测试用：阻塞等到工作线程结束。真实界面不要调用它。"""
        if self._thread is None:
            return True
        return self._thread.wait(timeout_ms)

    # -- 内部 ---------------------------------------------------------------

    def _poll(self) -> None:
        self.project.scheduler.load()
        for view in self.project.tasks():
            previous = self._last_states.get(view.task_id)
            if previous != view.state:
                self._last_states[view.task_id] = view.state
                self.stateChanged.emit(view.task_id, view.state)

    def _stop_polling(self) -> None:
        self._poller.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)
            self._thread = None
            self._runner = None

    def _on_finished(self, outcome: dict) -> None:
        self._poll()
        self._stop_polling()
        self.runFinished.emit(outcome)

    def _on_failed(self, message: str) -> None:
        self._stop_polling()
        self.runFailed.emit(message)
