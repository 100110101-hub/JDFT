"""主窗口：项目树 + Task Panel + Job Monitor + 日志 + 结果（P-09）。

布局
====

```
┌──────────────┬───────────────────────────────────────────────┐
│ 项目树        │ Task Panel（新建/提交/取消）                    │
│ project      │ Job Monitor（表格：id / 状态 / 原因）           │
│ ├ t0 queued  │ 结果（能量 / 偶极 / 收敛）                      │
│ └ t1 running │ 日志（LogView，流式）                          │
└──────────────┴───────────────────────────────────────────────┘
```

界面线程不做计算：提交只是往队列里登记，然后让 `SchedulerBridge` 在工作线程里跑。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..resultdir import ResultDirError, open_result_dir
from .logview import LogView
from .scene_panel import Scene3DPanel
from .model import GuiProject
from .runner import SchedulerBridge

#: 状态 → 颜色（终态给个明确底色，一眼能看出失败/取消）。
STATE_COLORS = {
    "queued": "#8a6d00",
    "running": "#0b5fa5",
    "converged": "#1b7f3b",
    "failed": "#a51a1a",
    "cancelled": "#5a5a5a",
}


class MainWindow(QMainWindow):
    """一个工程窗口。"""

    def __init__(self, out_root: str | Path, *, workers: int = 2, runner: str = "mock") -> None:
        super().__init__()
        self.setWindowTitle("QuantumWorkbench Studio")
        self.resize(1100, 700)
        self.runner = runner
        self.project = GuiProject(out_root, workers=workers, runner=runner)
        self.bridge = SchedulerBridge(self.project, self)
        self._rows: dict[str, int] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

        self._status = QLabel("就绪")
        self.statusBar().addWidget(self._status)

        self.bridge.stateChanged.connect(self._on_state_changed)
        self.bridge.runFinished.connect(self._on_run_finished)
        self.bridge.runFailed.connect(self._on_run_failed)
        self._refresh_monitor()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    # -- 构建 ---------------------------------------------------------------

    def _build_tree(self) -> QWidget:
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["项目 / 任务", "状态"])
        self.tree.setColumnWidth(0, 200)
        root = QTreeWidgetItem(["project", ""])
        root.setExpanded(True)
        self.tree.addTopLevelItem(root)
        self._tree_root = root
        return self.tree

    def _build_right(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_task_panel(), "Task")
        tabs.addTab(self._build_monitor(), "Job Monitor")
        tabs.addTab(self._build_result_panel(), "Result")
        self.log_view = LogView()
        tabs.addTab(self.log_view, "Log")
        self.scene_panel = Scene3DPanel()
        tabs.addTab(self.scene_panel, "3D")
        self.tabs = tabs
        return tabs

    def _build_task_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self.id_edit = QLineEdit("t0")
        self.seconds_spin = QDoubleSpinBox()
        self.seconds_spin.setRange(0.01, 60.0)
        self.seconds_spin.setValue(0.2)
        self.seconds_spin.setSuffix(" s")
        self.input_edit = QLineEdit("")
        self.input_edit.setPlaceholderText("input.yaml（engine runner 用）")
        form.addRow("任务 id", self.id_edit)
        form.addRow("mock 时长", self.seconds_spin)
        form.addRow("作业输入", self.input_edit)

        buttons = QHBoxLayout()
        self.submit_button = QPushButton("提交")
        self.submit_button.clicked.connect(self.submit_task)
        self.submit_engine_button = QPushButton("提交真作业")
        self.submit_engine_button.clicked.connect(self.submit_engine_task)
        self.run_button = QPushButton("运行队列")
        self.run_button.clicked.connect(self.run_queue)
        self.cancel_button = QPushButton("取消选中")
        self.cancel_button.clicked.connect(self.cancel_selected)
        buttons.addWidget(self.submit_button)
        buttons.addWidget(self.submit_engine_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.addWidget(panel)
        row = QWidget()
        row.setLayout(buttons)
        outer.addWidget(row)
        outer.addStretch(1)
        return holder

    def _build_monitor(self) -> QWidget:
        self.monitor = QTableWidget(0, 3)
        self.monitor.setHorizontalHeaderLabels(["task", "状态", "原因"])
        self.monitor.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.monitor.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.monitor.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return self.monitor

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.energy_label = QLabel("能量：—")
        self.dipole_label = QLabel("偶极：—")
        self.converged_label = QLabel("SCF：—")
        self.env_label = QLabel("env.json：—")
        for label in (self.energy_label, self.dipole_label, self.converged_label, self.env_label):
            layout.addWidget(label)
        self.load_result_button = QPushButton("打开结果目录…")
        self.load_result_button.clicked.connect(self.reload_result_dir)
        layout.addWidget(self.load_result_button)
        layout.addStretch(1)
        self._result_root: Path | None = None
        return panel

    # -- 槽 ---------------------------------------------------------------

    def submit_task(self) -> str:
        task_id = self.id_edit.text().strip() or f"t{len(self._rows)}"
        view = self.project.submit(task_id, self.seconds_spin.value())
        self._sync_tree()
        self._refresh_monitor()
        if not self.id_edit.text().strip():
            self.id_edit.setText(f"t{len(self._rows)}")
        self._status.setText(f"已登记 {view.task_id}（{view.state}）")
        return task_id

    def submit_engine_task(self) -> str | None:
        """登记一个真作业（`engine` runner）：input.yaml → 队列 → `qwb run`。"""
        source = self.input_edit.text().strip()
        if not source:
            self._status.setText("先填 input.yaml 路径")
            return None
        task_id = self.id_edit.text().strip() or "task"
        view = self.project.submit_engine(task_id, source)
        self._sync_tree()
        self._refresh_monitor()
        self._status.setText(f"已登记真作业 {view.task_id}")
        return view.task_id

    def run_queue(self) -> None:
        if self.bridge.is_running():
            self._status.setText("已经在跑")
            return
        self.bridge.start()
        self._status.setText("运行中…")

    def cancel_selected(self) -> None:
        task_id = self.selected_task_id()
        if task_id is None:
            self._status.setText("先选中一行")
            return
        self.project.cancel(task_id)
        self._refresh_monitor()
        self._status.setText(f"已请求取消 {task_id}")

    def selected_task_id(self) -> str | None:
        items = self.monitor.selectionModel().selectedRows() if self.monitor.selectionModel() else []
        if not items:
            return None
        row = items[0].row()
        item = self.monitor.item(row, 0)
        return item.text() if item is not None else None

    def _on_state_changed(self, task_id: str, state: str) -> None:
        self._refresh_monitor()
        row = self._rows.get(task_id)
        if row is not None:
            color = STATE_COLORS.get(state)
            if color:
                for column in range(self.monitor.columnCount()):
                    cell = self.monitor.item(row, column)
                    if cell is not None:
                        cell.setForeground(QColor(color))
        self._status.setText(f"{task_id} → {state}")
        if state == "converged":
            self.select_task(task_id)
            # 真作业的结果目录是 `<out>/<task_id>/`，直接打开给用户看
            candidate = self.project.out_root / task_id
            if (candidate / "result.json").is_file():
                self.set_result_root(candidate)
            else:
                self.reload_result_dir()

    def _on_run_finished(self, outcome: dict) -> None:
        self._refresh_monitor()
        counts = outcome.get("counts", {})
        self._status.setText(
            "完成：{converged} 收敛 / {failed} 失败 / {cancelled} 取消".format(
                converged=counts.get("converged", 0),
                failed=counts.get("failed", 0),
                cancelled=counts.get("cancelled", 0),
            )
        )

    def _on_run_failed(self, message: str) -> None:
        self._status.setText(f"运行失败：{message}")

    # -- 刷新 ---------------------------------------------------------------

    def _refresh_monitor(self) -> None:
        views = self.project.tasks()
        self._rows = {view.task_id: row for row, view in enumerate(views)}
        self.monitor.setRowCount(len(views))
        for row, view in enumerate(views):
            for column, text in enumerate((view.task_id, view.state, view.reason or "")):
                item = self.monitor.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.monitor.setItem(row, column, item)
                item.setText(text)
            color = STATE_COLORS.get(view.state)
            if color:
                for column in range(self.monitor.columnCount()):
                    self.monitor.item(row, column).setForeground(QColor(color))
        self._sync_tree()

    def _sync_tree(self) -> None:
        wanted = {view.task_id: view.state for view in self.project.tasks()}
        existing = {}
        for index in range(self._tree_root.childCount()):
            child = self._tree_root.child(index)
            existing[child.text(0)] = child
        for task_id, state in wanted.items():
            child = existing.get(task_id)
            if child is None:
                child = QTreeWidgetItem([task_id, state])
                self._tree_root.addChild(child)
            else:
                child.setText(1, state)
            color = STATE_COLORS.get(state)
            if color:
                child.setForeground(1, QColor(color))

    def select_task(self, task_id: str) -> None:
        row = self._rows.get(task_id)
        if row is not None:
            self.monitor.selectRow(row)

    # -- 结果 ---------------------------------------------------------------

    def set_result_root(self, root: str | Path) -> None:
        self._result_root = Path(root)
        self.reload_result_dir()
        # P-10：同一个结果目录顺手渲染 3D（离屏 VTK，几十毫秒，不阻塞 UI）
        self.scene_panel.show_result(root)

    def reload_result_dir(self) -> None:
        if self._result_root is None:
            return
        try:
            directory = open_result_dir(self._result_root)
        except ResultDirError as exc:
            self.energy_label.setText("能量：—（结果目录不可用）")
            self.env_label.setText(f"env.json：{exc}")
            return
        dipole = ""
        if directory.result.dipole_debye is not None:
            dipole = ", ".join(f"{value:+.4f}" for value in directory.result.dipole_debye)
        self.energy_label.setText(f"能量：{directory.result.energy_hartree:.10f} Ha")
        self.dipole_label.setText(f"偶极：{dipole or '—'} D")
        self.converged_label.setText(f"SCF：{'收敛' if directory.result.scf_converged else '未收敛'}")
        self.env_label.setText(
            f"env.json：rustc {directory.env.rustc_version}，threads {directory.env.threads}"
        )

    # -- 生命周期 -----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self.bridge.is_running():
            self.bridge.wait(30_000)
        super().closeEvent(event)


def build_application(argv: list[str] | None = None):
    """创建 `QApplication`（测试与 `python -m qwb_studio.gui` 共用）。"""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    return app


def main(argv: list[str] | None = None) -> int:
    """入口：`python -m qwb_studio.gui --out <dir>`。"""
    import argparse

    parser = argparse.ArgumentParser(prog="qwb-studio-gui", description="QuantumWorkbench Studio GUI（P-09 骨架）")
    parser.add_argument("--out", default="studio-out", help="工程输出目录（queue.json 与任务子目录）")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    app = build_application([sys_argv0()])
    window = MainWindow(args.out, workers=args.workers)
    window.show()
    return app.exec()


def sys_argv0() -> str:
    import sys

    return sys.argv[0]
