"""数据面板（P-09）：本地数据状态 + 第三方数据下载。

为什么需要这个面板
==================

`qwb-data/basis/` 与 `qwb-data/ecp/` 的部分文件是从 Basis Set Exchange 导出的第三方
数据，**仓库不提交**它们（再分发条款，见 `qwb-data/sources.json` 与 project.md §14 #20）。
所以"刚 clone 下来的仓库"是**故意**缺这批文件的：

* 提交计算时不用管 —— worker 是 Rust CLI，`run` 前会自动取回缺的那几个
  （`QWB_AUTO_FETCH=0` 可关闭，见 `qwb-cli`）；
* 但如果网络刚好不通、或者想提前把数据备好（离线机器、打包分发），就需要一个**手动**入口，
  也就是本面板：刷新状态 / 下载缺失 / 全部重下。

界面线程只做展示：下载在工作线程里跑，进度行通过信号回到 UI（`SchedulerBridge` 同样的做法）。
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..datafetch import DataFetchError, EntryStatus, check_entries, fetch_entries, find_data_root

__all__ = ["DataPanel", "STATE_LABELS"]

#: `check_entries` 的状态 → 界面文字。
STATE_LABELS = {"ok": "已就绪", "missing": "缺失", "stale": "摘要不符"}


class _FetchSignals(QObject):
    """工作线程 → 界面线程的信号载体（Qt 负责排队投递）。"""

    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)


class DataPanel(QWidget):
    """数据状态表 + 下载按钮。"""

    def __init__(
        self,
        data_root: str | Path | None = None,
        parent: QWidget | None = None,
        *,
        fetcher=fetch_entries,
    ) -> None:
        super().__init__(parent)
        self._data_root_arg = data_root
        self._fetcher = fetcher
        self._signals = _FetchSignals()
        self._signals.progress.connect(self._append_log)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._thread: threading.Thread | None = None
        self._statuses: list[EntryStatus] = []

        layout = QVBoxLayout(self)
        self.title = QLabel("数据：—")
        layout.addWidget(self.title)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["文件", "状态", "摘要钉"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self.refresh)
        self.fetch_missing_button = QPushButton("下载缺失")
        self.fetch_missing_button.clicked.connect(lambda: self.start_fetch(force=False))
        self.fetch_all_button = QPushButton("全部重下")
        self.fetch_all_button.clicked.connect(lambda: self.start_fetch(force=True))
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.fetch_missing_button)
        buttons.addWidget(self.fetch_all_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("下载日志（来源：Basis Set Exchange）")
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log)

        self.refresh()

    # -- 状态 ---------------------------------------------------------------

    def data_root(self) -> Path | None:
        """数据根；找不到时返回 None（按钮会禁用，而不是抛出去）。"""
        try:
            return find_data_root(self._data_root_arg)
        except DataFetchError as exc:
            self._append_log(f"找不到数据根：{exc}")
            return None

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def statuses(self) -> list[EntryStatus]:
        return list(self._statuses)

    # -- 交互 ---------------------------------------------------------------

    def refresh(self) -> list[EntryStatus]:
        """重读每条清单的状态（纯本地读文件，不联网）。"""
        root = self.data_root()
        if root is None:
            self.title.setText("数据：找不到 qwb-data/")
            for button in (self.fetch_missing_button, self.fetch_all_button):
                button.setEnabled(False)
            return []
        try:
            self._statuses = check_entries(root)
        except DataFetchError as exc:
            self.title.setText(f"数据：{exc}")
            return []
        self._fill_table()
        ready = sum(1 for status in self._statuses if status.ok)
        pending = len(self._statuses) - ready
        self.title.setText(
            f"数据：{ready}/{len(self._statuses)} 已就绪"
            + (f"（{pending} 个需下载）" if pending else "")
            + f"  —  {root}"
        )
        enabled = bool(pending)
        self.fetch_missing_button.setEnabled(enabled)
        self.fetch_all_button.setEnabled(True)
        return list(self._statuses)

    def start_fetch(self, *, force: bool = False) -> bool:
        """在后台线程里下载；返回是否真的启动了。"""
        if self.is_busy():
            self._append_log("下载已在进行中")
            return False
        root = self.data_root()
        if root is None:
            return False
        self._append_log(
            "开始下载：Basis Set Exchange → 转录 → 核对摘要钉" + ("（强制重下）" if force else "")
        )
        for button in (self.refresh_button, self.fetch_missing_button, self.fetch_all_button):
            button.setEnabled(False)

        signals = self._signals
        fetcher = self._fetcher

        def work() -> None:
            try:
                fetched = fetcher(root, force=force, log=signals.progress.emit)
            except BaseException as exc:  # noqa: BLE001 - 线程里的异常都要报回界面
                signals.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            signals.finished.emit(dict(fetched))

        self._thread = threading.Thread(target=work, name="qwb-datafetch", daemon=True)
        self._thread.start()
        return True

    def wait(self, timeout: float = 600.0) -> bool:
        """测试用：等后台下载结束。真实界面不要调用。"""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- 内部 ---------------------------------------------------------------

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self._statuses))
        for row, status in enumerate(self._statuses):
            cells = (
                status.path,
                STATE_LABELS.get(status.state, status.state),
                status.expected[:16] + "…",
            )
            for column, text in enumerate(cells):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                item.setText(text)
            if not status.ok:
                for column in range(self.table.columnCount()):
                    self.table.item(row, column).setForeground(Qt.GlobalColor.darkYellow)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_finished(self, fetched: dict) -> None:
        self._append_log(f"完成：{len(fetched)} 条")
        for button in (self.refresh_button, self.fetch_missing_button, self.fetch_all_button):
            button.setEnabled(True)
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self._append_log(f"失败：{message}")
        for button in (self.refresh_button, self.fetch_missing_button, self.fetch_all_button):
            button.setEnabled(True)
        self.refresh()
