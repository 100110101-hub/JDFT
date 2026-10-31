"""结果对比表面板（PR-07）：一批结果目录 → 一张表（能量 / ΔE / 谱峰）+ 导出 CSV。

数值全部来自 [`..comparison`]（纯数据层，Qt 无关），面板只负责"显示"和"点按钮"：

* `add_task_dir()` / `set_task_dirs()`：装载（缺结果目录会进状态行，不抛异常）；
* `export_csv()`：写 CSV（与 `qwb compare --csv` 同一列序、同一小数位）；
* `status_text`：一句话说清"几个任务、参考任务是哪个"。

为什么不自己算：判据是"对比值与单看一致（1e-10）"，所以面板不许有第二套算术 ——
数字只有一处来源，测试才能用同一个来源核对。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..comparison import (
    ComparisonError,
    ComparisonRow,
    build_comparison,
    write_comparison_csv,
)

HEADERS = ("task", "method", "basis", "energy/Ha", "ΔE/(kcal/mol)", "SCF", "peaks(eV)")


class ComparisonPanel(QWidget):
    """多任务对比表 + 导出按钮。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._task_dirs: list[Path] = []
        self._rows: tuple[ComparisonRow, ...] = ()
        self._status = "还没有结果目录"

        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("添加结果目录…")
        self.add_button.setObjectName("comparison_add")
        self.clear_button = QPushButton("清空")
        self.clear_button.setObjectName("comparison_clear")
        self.export_button = QPushButton("导出 CSV…")
        self.export_button.setObjectName("comparison_export")
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QLabel(self._status)
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setObjectName("comparison_table")
        self.table.setHorizontalHeaderLabels(list(HEADERS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table, 1)

        self.add_button.clicked.connect(self._on_add_clicked)
        self.clear_button.clicked.connect(self.clear)
        self.export_button.clicked.connect(self._on_export_clicked)

    # ── 数据 ────────────────────────────────────────────────────────────────
    @property
    def task_dirs(self) -> tuple[Path, ...]:
        return tuple(self._task_dirs)

    @property
    def rows(self) -> tuple[ComparisonRow, ...]:
        return self._rows

    @property
    def status_text(self) -> str:
        return self._status

    def add_task_dir(self, task_dir: Path | str) -> bool:
        """追加一个结果目录；返回它是否进了表。"""
        self.set_task_dirs([*self._task_dirs, Path(task_dir)])
        return Path(task_dir) in self._task_dirs

    def set_task_dirs(self, task_dirs) -> None:
        """整体替换任务列表并重算表格（少于两个任务时表是空的，状态行说明原因）。"""
        self._task_dirs = [Path(directory) for directory in task_dirs]
        self._rebuild()

    def clear(self) -> None:
        self.set_task_dirs([])

    def _rebuild(self) -> None:
        self.table.setRowCount(0)
        self._rows = ()
        if not self._task_dirs:
            self._status = "还没有结果目录"
            self.status_label.setText(self._status)
            return
        if len(self._task_dirs) < 2:
            # 对比表至少两个任务才有 ΔE 可言（与 `qwb compare` 的退出码 1 同一条规矩）。
            self._status = f"{len(self._task_dirs)} 个任务；对比需要至少 2 个结果目录"
            self.status_label.setText(self._status)
            return
        try:
            rows = build_comparison(self._task_dirs)
        except ComparisonError as exc:
            self._status = f"对比表不可用：{exc}"
            self.status_label.setText(self._status)
            return
        self._rows = rows
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            peaks = ", ".join(f"{peak:.2f}" for peak in row.spectrum_peaks_ev) or "—"
            cells = (
                row.task,
                row.method,
                row.basis,
                f"{row.energy_hartree:.10f}",
                f"{row.delta_e_kcal_per_mol:+.4f}",
                "yes" if row.converged else "no",
                peaks,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column >= 3:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(index, column, item)
        reference = rows[0].task
        self._status = f"{len(rows)} 个任务，参考（ΔE = 0）是 {reference}"
        self.status_label.setText(self._status)

    # ── 导出 ────────────────────────────────────────────────────────────────
    def export_csv(self, path: Path | str) -> Path:
        """把当前表格写成 CSV（列序/小数位与 `qwb compare --csv` 相同）。"""
        if not self._rows:
            raise ComparisonError("没有可导出的对比表（先装载至少两个结果目录）")
        return write_comparison_csv(path, self._rows)

    # ── 对话框（测试直接调上面的 API，不碰这两条） ────────────────────────
    def _on_add_clicked(self) -> None:  # pragma: no cover - 需要真实窗口系统
        from PySide6.QtWidgets import QFileDialog

        chosen = QFileDialog.getExistingDirectory(self, "选择结果目录")
        if chosen:
            self.add_task_dir(chosen)

    def _on_export_clicked(self) -> None:  # pragma: no cover - 需要真实窗口系统
        from PySide6.QtWidgets import QFileDialog

        target, _ = QFileDialog.getSaveFileName(
            self, "导出对比表", "comparison.csv", "CSV (*.csv)"
        )
        if target:
            self.export_csv(target)
