"""把调度器放到工作线程里跑，界面线程只收信号。

`SchedulerBridge` 本身是 `QObject`（住在界面线程），它持有一个 `QThread`，
工作线程里跑 `GuiProject.run()`；运行期间用一个 `QTimer` 轮询队列文件并广播状态变化。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..scheduler import SchedulerError, TaskQueue


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
        #: `start()` 之后到 `_on_finished` 处理完之前都为 True。
        #:
        #: 为什么不能只看 `QThread.isRunning()`：工作线程一结束 `isRunning()` 就变 False，
        #: 但"把最终状态同步进界面"的 `_on_finished` 是**排队信号**，还要等事件循环转一圈。
        #: 只看线程状态的话，调用方（测试、状态栏）会读到"线程停了但界面还是 running"的
        #: 中间态 —— 这正是一开始 gate 里偶发
        #: `['converged', 'running', 'converged']` 的原因。
        self._active = False

    # -- 生命周期 -----------------------------------------------------------

    def is_running(self) -> bool:
        if self._active:
            return True
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> None:
        if self.is_running():
            return
        self._last_states = {view.task_id: view.state for view in self.project.tasks()}
        self._active = True
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
        """把队列**文件**当成只读快照来轮询，不碰 scheduler 的内存状态。

        工作线程正在 `set_state` + `save`，界面线程如果调 `scheduler.load()` 就是两边
        同时改同一个列表（`load` 整个替换 `_entries`）—— 那不只是读到旧值，还可能让
        工作线程把状态写到错的位置上。所以这里每次单独读一份队列文件。
        """
        try:
            entries = TaskQueue(self.project.scheduler.queue_path).load()
        except SchedulerError:
            return
        for entry in entries:
            state = entry.state.value
            previous = self._last_states.get(entry.task_id)
            if previous != state:
                self._last_states[entry.task_id] = state
                self.stateChanged.emit(entry.task_id, state)

    def _stop_polling(self) -> None:
        self._poller.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)
            self._thread = None
            self._runner = None

    def _on_finished(self, outcome: dict) -> None:
        self._stop_polling()
        # 工作线程已经退出，这时再把 scheduler 的内存状态与磁盘对齐是安全的。
        try:
            self.project.sync()
        finally:
            self._poll()
            self._active = False
        self.runFinished.emit(outcome)

    def _on_failed(self, message: str) -> None:
        self._stop_polling()
        self._active = False
        self.runFailed.emit(message)
