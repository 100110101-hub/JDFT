"""T8 的"建分子"：把坐标、电荷、基组、泛函、作业类型填成一份 K2/K4 的 `input.yaml`。

为什么要有这一块：T8 的判据是"打开项目 → **建分子** → 选 PBE/def2-SVP/opt → 提交 →
Job Monitor → 3D 结果"。在这之前 GUI 只能提交**已经存在**的 `input.yaml`，
"建分子"是缺的一环。

设计边界
========

* 这里只做**输入组装**，不做任何电子结构判断：合法性由引擎那边判（K2 `JobInput::validate`
  与数据层）。界面能挡的先挡（元素必须是该基组覆盖的、坐标必须有限），挡不住的
  就让它失败并显示原因 —— 不假装校验过。
* 下拉框的取值来自 `qwb_studio.data_catalog`（`qwb-data/registry.yaml`），
  不写死数据名。
* 生成的 `input.yaml` 用 `qwb_studio.yaml_min.dumps`：与 Rust 的 `serde_yaml`
  读得懂的块式 YAML 同一子集（K4 的 `input.yaml` 形态），不引入 PyYAML（§2.4）。
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data_catalog import GRID_LEVELS, JOB_TYPES, Catalog, CatalogError
from ..models import Atom, Molecule
from ..yaml_min import dumps as yaml_dumps

__all__ = ["MoleculeBuilder", "build_job_input"]


def build_job_input(
    molecule: Molecule,
    *,
    basis: str,
    method: str,
    grid: str,
    jobtype: str,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    """组装 K2 `JobInput` 的字典形态（顺序与 `examples/*.yaml` 一致，便于人读）。"""
    if jobtype not in JOB_TYPES:
        raise ValueError(f"jobtype 只能是 {'/'.join(JOB_TYPES)}，收到 {jobtype!r}")
    if grid not in GRID_LEVELS:
        raise ValueError(f"grid_level 只能是 {'/'.join(GRID_LEVELS)}，收到 {grid!r}")
    if not molecule.atoms:
        raise ValueError("分子至少有 1 个原子")
    base_options: dict[str, object] = {
        "scf_max_iter": 128,
        "scf_energy_threshold_hartree": 1.0e-8,
        "scf_density_threshold": 1.0e-6,
        "diis_history": 8,
        "threads": 0,
        "max_opt_steps": 100,
        "opt_gradient_threshold_hartree_per_bohr": 4.5e-4,
        "freq_step_angstrom": 0.01,
        "ecp_name": None,
    }
    if options:
        base_options.update(options)
    return {
        "molecule": {
            "atoms": [
                {"symbol": atom.symbol, "position_angstrom": list(atom.position_angstrom)}
                for atom in molecule.atoms
            ],
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
        },
        "basis_name": basis,
        "method_name": method,
        "grid_level": grid,
        "solvent": "vacuum",
        "jobtype": jobtype,
        "options": base_options,
    }


@dataclass(frozen=True)
class _AtomRow:
    symbol: str
    position: tuple[float, float, float]


class MoleculeBuilder(QWidget):
    """建分子 + 生成 input.yaml。"""

    inputWritten = Signal(str)  # 生成的 input.yaml 路径
    submitRequested = Signal(str)  # 要提交的 input.yaml 路径
    statusChanged = Signal(str)

    def __init__(
        self,
        data_root: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog: Catalog | None = None
        self._catalog_error: str | None = None
        self._last_path: Path | None = None

        # -- 原子输入 ------------------------------------------------------
        self.element_combo = QComboBox()
        self.element_combo.setEditable(True)
        self.x_spin = _coordinate_spin()
        self.y_spin = _coordinate_spin()
        self.z_spin = _coordinate_spin()
        self.add_button = QPushButton("添加原子")
        self.add_button.clicked.connect(self.add_atom)
        self.remove_button = QPushButton("删除选中")
        self.remove_button.clicked.connect(self.remove_selected)

        self.atom_table = QTableWidget(0, 4)
        self.atom_table.setHorizontalHeaderLabels(["元素", "x (Å)", "y (Å)", "z (Å)"])
        self.atom_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.atom_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.atom_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.atom_count_label = QLabel("原子数：0")

        # -- 体系属性 ------------------------------------------------------
        self.charge_spin = QSpinBox()
        self.charge_spin.setRange(-3, 3)
        self.multiplicity_spin = QSpinBox()
        self.multiplicity_spin.setRange(1, 5)

        # -- 方法 ----------------------------------------------------------
        self.basis_combo = QComboBox()
        self.method_combo = QComboBox()
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(GRID_LEVELS)
        self.grid_combo.setCurrentText("normal")
        self.jobtype_combo = QComboBox()
        self.jobtype_combo.addItems(JOB_TYPES)
        self.basis_combo.currentTextChanged.connect(self._on_basis_changed)

        self.path_edit = QLineEdit("")
        self.path_edit.setPlaceholderText("生成的 input.yaml 路径（留空 = <out>/.staging/build.yaml）")
        self.write_button = QPushButton("生成 input.yaml")
        self.write_button.clicked.connect(self.write_input)
        self.submit_button = QPushButton("提交真作业")
        self.submit_button.setToolTip("把这份 input.yaml 交给 engine runner（qwb run）")
        self.submit_button.clicked.connect(self.submit)
        self.info_label = QLabel("")

        self._compose()
        self._load_catalog(data_root)

    # -- 布局 ---------------------------------------------------------------

    def _compose(self) -> None:
        coords = QHBoxLayout()
        coords.addWidget(QLabel("元素"))
        coords.addWidget(self.element_combo)
        for label, spin in (("x", self.x_spin), ("y", self.y_spin), ("z", self.z_spin)):
            coords.addWidget(QLabel(label))
            coords.addWidget(spin)
        coords.addWidget(self.add_button)
        coords.addWidget(self.remove_button)
        coords.addStretch(1)

        form = QFormLayout()
        form.addRow("电荷", self.charge_spin)
        form.addRow("多重度", self.multiplicity_spin)
        form.addRow("基组", self.basis_combo)
        form.addRow("方法", self.method_combo)
        form.addRow("网格", self.grid_combo)
        form.addRow("作业类型", self.jobtype_combo)
        form.addRow("输出", self.path_edit)

        buttons = QHBoxLayout()
        buttons.addWidget(self.write_button)
        buttons.addWidget(self.submit_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(coords)
        layout.addWidget(self.atom_table)
        layout.addWidget(self.atom_count_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.info_label)
        layout.addStretch(1)

    # -- 数据目录 -----------------------------------------------------------

    def _load_catalog(self, data_root: str | Path | None) -> None:
        try:
            catalog = Catalog.open(data_root)
        except CatalogError as exc:
            self._catalog_error = str(exc)
            self.info_label.setText(f"数据目录不可用：{exc}")
            self.write_button.setEnabled(False)
            self.submit_button.setEnabled(False)
            return
        self._catalog = catalog
        self._catalog_error = None
        self.basis_combo.clear()
        self.basis_combo.addItems(catalog.bases)
        self.method_combo.clear()
        self.method_combo.addItems(catalog.methods)
        self._on_basis_changed(self.basis_combo.currentText())
        self.info_label.setText(
            f"数据目录 {catalog.root}：{len(catalog.bases)} 个基组、"
            f"{len(catalog.functionals)} 个泛函（+ hf）"
        )

    def _on_basis_changed(self, basis: str) -> None:
        if self._catalog is None:
            return
        try:
            elements = self._catalog.elements(basis)
        except CatalogError as exc:
            self.element_combo.clear()
            self.info_label.setText(f"{basis} 的元素读不出来：{exc}")
            return
        current = self.element_combo.currentText()
        self.element_combo.clear()
        self.element_combo.addItems(elements)
        if current in elements:
            self.element_combo.setCurrentText(current)

    # -- 原子表 -------------------------------------------------------------

    def add_atom(self) -> bool:
        symbol = self.element_combo.currentText().strip()
        if not symbol:
            self.statusChanged.emit("先选元素")
            return False
        position = (self.x_spin.value(), self.y_spin.value(), self.z_spin.value())
        if not all(math.isfinite(value) for value in position):
            self.statusChanged.emit("坐标必须是有限数")
            return False
        row = self.atom_table.rowCount()
        self.atom_table.insertRow(row)
        for column, text in enumerate(
            (symbol, f"{position[0]:.6f}", f"{position[1]:.6f}", f"{position[2]:.6f}")
        ):
            self.atom_table.setItem(row, column, QTableWidgetItem(text))
        self.atom_count_label.setText(f"原子数：{self.atom_table.rowCount()}")
        self.statusChanged.emit(f"已添加 {symbol} ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) Å")
        return True

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.atom_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.atom_table.removeRow(row)
        self.atom_count_label.setText(f"原子数：{self.atom_table.rowCount()}")

    def _rows(self) -> list[_AtomRow]:
        out: list[_AtomRow] = []
        for row in range(self.atom_table.rowCount()):
            cells = [self.atom_table.item(row, column) for column in range(4)]
            if any(cell is None for cell in cells):
                continue
            symbol = cells[0].text().strip()  # type: ignore[union-attr]
            values = tuple(float(cells[column].text()) for column in (1, 2, 3))  # type: ignore[union-attr]
            out.append(_AtomRow(symbol, (values[0], values[1], values[2])))
        return out

    def molecule(self) -> Molecule:
        """当前的分子（K1/K2 的 `Atom`/`Molecule` 模型；非法时抛 `ValueError`）。"""
        rows = self._rows()
        if not rows:
            raise ValueError("分子至少要有 1 个原子")
        return Molecule(
            atoms=[Atom(symbol=row.symbol, position_angstrom=row.position) for row in rows],
            charge=self.charge_spin.value(),
            multiplicity=self.multiplicity_spin.value(),
        )

    # -- 生成 / 提交 --------------------------------------------------------

    def job_input(self) -> dict[str, object]:
        return build_job_input(
            self.molecule(),
            basis=self.basis_combo.currentText(),
            method=self.method_combo.currentText(),
            grid=self.grid_combo.currentText(),
            jobtype=self.jobtype_combo.currentText(),
        )

    def write_input(self, path: str | Path | None = None) -> Path | None:
        """生成 input.yaml 并返回路径；失败时把原因写进 info_label 并返回 None。"""
        target = Path(path) if path is not None else (
            Path(self.path_edit.text().strip()) if self.path_edit.text().strip() else None
        )
        if target is None:
            self.info_label.setText("先在“输出”里填一个路径，或用 submit() 走默认暂存目录")
            self.statusChanged.emit("缺少输出路径")
            return None
        try:
            payload = self.job_input()
        except ValueError as exc:
            self.info_label.setText(f"输入不完整：{exc}")
            self.statusChanged.emit(str(exc))
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml_dumps(payload), encoding="utf-8")
        self._last_path = target
        self.path_edit.setText(str(target))
        self.info_label.setText(f"已写 {target}")
        self.statusChanged.emit(f"已写 {target}")
        self.inputWritten.emit(str(target))
        return target

    def generate_into(self, task_id: str, staging_root: str | Path) -> Path:
        """把 input.yaml 写进 `<staging_root>/<task_id>.yaml` 并返回路径。

        文件名必须等于 task id：`qwb run` 用输入文件的 stem 命名结果目录，
        只有这样才能落进 `<out>/<task_id>/`（与队列里的 task 对上）。
        """
        return self.write_input(Path(staging_root) / f"{task_id}.yaml")  # type: ignore[return-value]

    def submit(self) -> str | None:
        """写一份临时 input.yaml 并发 `submitRequested`（由主窗口登记成真作业）。"""
        if self.atom_table.rowCount() == 0:
            self.info_label.setText("先添加原子")
            self.statusChanged.emit("先添加原子")
            return None
        if self._last_path is None:
            # 没填输出路径时落到**临时目录**，不落到进程的 CWD：
            # 从桌面快捷方式启动 GUI 时 CWD 可能是任何地方，往那儿写文件是没礼貌的。
            fallback = Path(tempfile.gettempdir()) / "qwb-studio-build" / "build-input.yaml"
            written = self.write_input(fallback)
            if written is None:
                return None
        path = self._last_path
        assert path is not None
        self.submitRequested.emit(str(path))
        return str(path)

    # -- 供主窗口/测试读 -----------------------------------------------------

    @property
    def catalog(self) -> Catalog | None:
        return self._catalog

    @property
    def catalog_error(self) -> str | None:
        return self._catalog_error

    @property
    def last_path(self) -> Path | None:
        return self._last_path


def _coordinate_spin() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-50.0, 50.0)
    spin.setDecimals(6)
    spin.setSingleStep(0.05)
    spin.setSuffix(" Å")
    return spin
