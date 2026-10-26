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
from ..runner import find_data_root
from .builder import MoleculeBuilder
from .datapanel import DataPanel
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

    def __init__(
        self,
        out_root: str | Path,
        *,
        workers: int = 2,
        runner: str = "mock",
        data_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("QuantumWorkbench Studio")
        self.resize(1100, 700)
        self.runner = runner
        self.project = GuiProject(out_root, workers=workers, runner=runner)
        self.bridge = SchedulerBridge(self.project, self)
        self._rows: dict[str, int] = {}
        self._data_root = data_root

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)
        self._build_menu()

        self._status = QLabel("就绪")
        self.statusBar().addWidget(self._status)

        self.bridge.stateChanged.connect(self._on_state_changed)
        self.bridge.runFinished.connect(self._on_run_finished)
        self.bridge.runFailed.connect(self._on_run_failed)
        # T8 第三条判据：崩溃后重启 → 从磁盘队列重建视图（running 落 failed/interrupted，
        # queued 留着可以直接接着跑）
        self._restore_unfinished_view()
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
        # "建分子 → 提交" 是 T8 的主流程，所以 Build 放第一个。
        self.builder = MoleculeBuilder(self._data_root)
        self.builder.submitRequested.connect(self._on_builder_submit)
        self.builder.statusChanged.connect(self._on_builder_status)
        tabs.addTab(self.builder, "Build")
        tabs.addTab(self._build_task_panel(), "Task")
        tabs.addTab(self._build_monitor(), "Job Monitor")
        tabs.addTab(self._build_result_panel(), "Result")
        self.log_view = LogView()
        tabs.addTab(self.log_view, "Log")
        self.scene_panel = Scene3DPanel()
        tabs.addTab(self.scene_panel, "3D")
        # 第三方数据（基组/ECP）不在仓库里：这里给一个手动下载/校验的入口。
        self.data_panel = DataPanel(self._data_root)
        tabs.addTab(self.data_panel, "Data")
        self.tabs = tabs
        return tabs

    def _build_menu(self) -> None:
        """菜单栏：数据动作也放到菜单里（和 Data 页是同一批槽，状态因此一致）。"""
        menu = self.menuBar().addMenu("数据")
        for label, slot in (
            ("刷新状态", self.data_panel.refresh),
            ("下载缺失", lambda: self.data_panel.start_fetch(force=False)),
            ("全部重下", lambda: self.data_panel.start_fetch(force=True)),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            menu.addAction(action)

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

    def _on_builder_status(self, message: str) -> None:
        self._status.setText(message)

    def _on_builder_submit(self, input_yaml: str) -> str | None:
        """Build 页点"提交真作业"：登记成 engine 任务（worker = `qwb run`）。"""
        task_id = self.id_edit.text().strip() or f"build{len(self._rows)}"
        try:
            view = self.project.submit_engine(task_id, input_yaml)
        except (FileNotFoundError, OSError) as exc:
            self._status.setText(f"提交失败：{exc}")
            self.log_view.appendPlainText(f"提交失败：{exc}")
            return None
        self.id_edit.setText(_next_task_id(task_id))
        self.input_edit.setText(input_yaml)
        self._sync_tree()
        self._refresh_monitor()
        self.log_view.appendPlainText(f"已登记 {view.task_id}（{view.state}）：{input_yaml}")
        self._status.setText(f"已登记 {view.task_id}（Build 页）")
        return view.task_id

    def _restore_unfinished_view(self) -> None:
        """启动时把磁盘上的任务表读回来（T8：崩溃后重启恢复未完成视图）。"""
        try:
            outcome = self.project.restore()
        except Exception as exc:  # noqa: BLE001 - 队列文件坏了不该让窗口起不来
            self.log_view.appendPlainText(f"恢复失败：{exc}")
            self._status.setText(f"恢复失败：{exc}")
            return
        restored = outcome["restored"]
        interrupted = outcome["interrupted"]
        if restored:
            self.log_view.appendPlainText(
                f"恢复 {len(restored)} 个任务：" + ", ".join(f"{view.task_id}={view.state}" for view in restored)
            )
        for task_id in interrupted:
            self.log_view.appendPlainText(
                f"{task_id} 上次停在 running（进程被杀）：K3 没有 running→queued，已落 failed，"
                "请用新的 task id 重提"
            )
        if restored:
            queued = sum(1 for view in restored if view.state == "queued")
            self._status.setText(
                f"恢复了 {len(restored)} 个任务（{queued} 个 queued 可继续跑，{len(interrupted)} 个被打断）"
            )

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


def _next_task_id(previous: str) -> str:
    """`t3` → `t4`；带前缀的（`build3`）只递增末尾的数字。"""
    digits = ""
    for character in reversed(previous):
        if not character.isdigit():
            break
        digits = character + digits
    if not digits:
        return f"{previous}1"
    stem = previous[: len(previous) - len(digits)]
    return f"{stem}{int(digits) + 1}"


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
    parser.add_argument(
        "--data-root",
        default=None,
        help="qwb-data 目录（默认 QWB_DATA_ROOT 或从当前目录向上找）",
    )
    args = parser.parse_args(argv)

    app = build_application([sys_argv0()])
    window = MainWindow(args.out, workers=args.workers, data_root=args.data_root)
    window.show()
    return app.exec()


def sys_argv0() -> str:
    import sys

    return sys.argv[0]
