"""PR-07 的 GUI 一半：对比表面板装载、显示、导出，且**不重算**。

判据：

1. 装载三个结果目录后，表格行数 = 任务数，能量单元格保留 10 位小数，
   且与该任务 `result.json` 的差 ≤ 1e-10（面板不许有第二套算术）；
2. 参考任务（第一行）的 ΔE 显示为 `+0.0000`；
3. `export_csv` 写出的文件与 [`qwb_studio.comparison.comparison_csv`] 逐字节相同；
4. 退化路径：0 个 / 1 个目录、缺 `result.json` 的目录，都不抛异常，状态行说清原因。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from qwb_studio.comparison import comparison_csv  # noqa: E402
from qwb_studio.gui.comparison_panel import HEADERS, ComparisonPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _task(root: Path, name: str, energy: float, *, method: str, spectrum: bool) -> Path:
    task = root / name
    (task / "artifacts").mkdir(parents=True)
    payload: dict = {"energy_hartree": energy, "scf_converged": True}
    if spectrum:
        payload["uv_vis_spectrum"] = {
            "gaussian_hwhm_ev": 0.4,
            "grid_step_ev": 0.05,
            "points": [[6.0 + 0.05 * index, 1.0 / (1.0 + (index - 10) ** 2)] for index in range(40)],
        }
    (task / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (task / "input.yaml").write_text(
        f"method_name: {method}\nbasis_name: sto-3g\n", encoding="utf-8"
    )
    return task


@pytest.fixture()
def tasks(tmp_path: Path) -> list[Path]:
    return [
        _task(tmp_path, "water_hf", -76.0266326514, method="hf", spectrum=True),
        _task(tmp_path, "water_b3lyp", -76.4182371723, method="b3lyp", spectrum=False),
        _task(tmp_path, "water_svwn5", -75.8736120931, method="svwn5", spectrum=False),
    ]


def test_panel_shows_the_same_values_as_the_result_files(tasks: list[Path], app) -> None:
    panel = ComparisonPanel()
    panel.set_task_dirs(tasks)
    assert panel.table.rowCount() == 3
    assert panel.table.columnCount() == len(HEADERS)
    for row_index, task_dir in enumerate(tasks):
        payload = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
        cell = panel.table.item(row_index, 3).text()
        assert abs(float(cell) - payload["energy_hartree"]) <= 1e-10, cell
        # 10 位小数：人看到的位数不能比 CSV 少。
        assert len(cell.split(".")[1]) == 10, cell
        assert panel.table.item(row_index, 0).text() == task_dir.name
        assert "参考（ΔE = 0）" in panel.status_text
    assert panel.table.item(0, 4).text() == "+0.0000"


def test_export_matches_the_data_layer(tasks: list[Path], tmp_path: Path, app) -> None:
    panel = ComparisonPanel()
    panel.set_task_dirs(tasks)
    target = panel.export_csv(tmp_path / "comparison.csv")
    assert target.is_file()
    assert target.read_bytes().decode("utf-8") == comparison_csv(panel.rows)
    assert b"\r" not in target.read_bytes()


def test_adding_one_by_one_works(tasks: list[Path], app) -> None:
    panel = ComparisonPanel()
    assert panel.add_task_dir(tasks[0]) is True
    # 只有一个任务时不出表，但要说清为什么。
    assert panel.table.rowCount() == 0
    assert "至少 2 个" in panel.status_text
    assert panel.add_task_dir(tasks[1]) is True
    assert panel.table.rowCount() == 2
    panel.clear()
    assert panel.table.rowCount() == 0
    assert "还没有结果目录" in panel.status_text


def test_degenerate_inputs_do_not_raise(tmp_path: Path, app) -> None:
    panel = ComparisonPanel()
    # 完全没有目录。
    panel.set_task_dirs([])
    assert "还没有结果目录" in panel.status_text
    # 目录存在但缺 result.json：进状态行，不抛异常。
    empty = tmp_path / "empty_task"
    empty.mkdir()
    good = _task(tmp_path, "good", -1.5, method="hf", spectrum=False)
    panel.set_task_dirs([empty, good])
    assert panel.table.rowCount() == 0
    assert "对比表不可用" in panel.status_text and "result.json" in panel.status_text
    # 没有可导出的表时，导出必须报错而不是写一个空文件。
    with pytest.raises(Exception):
        panel.export_csv(tmp_path / "nope.csv")
