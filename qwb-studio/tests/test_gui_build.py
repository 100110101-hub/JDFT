"""T8 的三条判据（GUI 侧）：

1. **建分子 → 选 PBE/def2-SVP/opt → 提交 → Job Monitor → 3D 结果，不崩溃不阻塞**
   —— 在这条测试里用 STO-3G 代替 def2-SVP（PBE/def2-SVP/opt 是十几分钟的慢档，
   实测数字与跑法记在 `docs/status.yaml` 的 T8 里）；链路本身**完全相同**：
   Build 页建分子 → 生成 input.yaml → engine runner（`qwb run`）→ K3 结果目录 →
   Result 面板 + 3D 面板（几何取自 `artifacts/optimized.xyz`，不是初值）。
2. **大日志不卡** —— 见 `test_gui.py::test_big_log_streams_without_freezing_the_ui`
   （102.8 MB / 154 万行）。
3. **崩溃后重启恢复未完成视图** —— 直接构造"被杀的现场"（队列里一条 running、
   一条 queued，running 的结果目录不存在），再起一个新窗口，验证 running 落
   `failed`（reason 带 interrupted）、queued 仍可继续跑。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qwb_studio.gui.builder import build_job_input  # noqa: E402
from qwb_studio.gui.mainwindow import MainWindow  # noqa: E402
from qwb_studio.models import Atom, Molecule  # noqa: E402
from qwb_studio.resultdir import open_result_dir  # noqa: E402
from qwb_studio.yaml_min import loads as yaml_loads  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def spin_until(app: QApplication, predicate, timeout: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _h2_builder(window: MainWindow) -> None:
    """在 Build 页建一个 H2（0.80 Å），方法 PBE、基组 STO-3G、作业类型 opt。"""
    builder = window.builder
    builder.element_combo.setCurrentText("H")
    builder.z_spin.setValue(-0.40)
    assert builder.add_atom()
    builder.z_spin.setValue(0.40)
    assert builder.add_atom()
    builder.basis_combo.setCurrentText("sto-3g")
    builder.method_combo.setCurrentText("pbe")
    builder.grid_combo.setCurrentText("normal")
    builder.jobtype_combo.setCurrentText("opt")
    builder.charge_spin.setValue(0)
    builder.multiplicity_spin.setValue(1)


# ---------------------------------------------------------------------------
# 1. 建分子 → 提交 → 监控 → 3D
# ---------------------------------------------------------------------------


def test_build_then_submit_opt_job_then_3d(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(tmp_path / "project", workers=1, runner="engine", data_root=DATA_ROOT)
    rendered: list[int] = []
    window.scene_panel.rendered.connect(lambda _path, painted: rendered.append(painted))
    try:
        # Build 页的下拉框是数据驱动的：只有 qwb-data 里真有的东西
        assert window.builder.catalog is not None, window.builder.catalog_error
        assert "sto-3g" in window.builder.catalog.bases
        assert "pbe" in window.builder.catalog.methods
        # 元素下拉框跟着基组走（STO-3G 覆盖 H/C/N/O/F）
        assert "H" in window.builder.catalog.elements("sto-3g")

        _h2_builder(window)
        assert window.builder.atom_table.rowCount() == 2
        window.id_edit.setText("h2opt")

        # 点"提交真作业"：Build 页写一份 input.yaml 并登记成 engine 任务
        path = window.builder.submit()
        assert path is not None
        written = yaml_loads(Path(path).read_text(encoding="utf-8"))
        assert written["jobtype"] == "opt"
        assert written["basis_name"] == "sto-3g"
        assert written["method_name"] == "pbe"
        assert [atom["symbol"] for atom in written["molecule"]["atoms"]] == ["H", "H"]

        states = [view.state for view in window.project.tasks()]
        assert states == ["queued"], states
        assert window.monitor.rowCount() == 1

        window.run_queue()
        assert spin_until(app, lambda: not window.bridge.is_running(), timeout=600), "opt 作业没在 600 s 内结束"
        states = [view.state for view in window.project.tasks()]
        assert states == ["converged"], [view.state for view in window.project.tasks()]

        # Job Monitor 的 Task 面板/结果面板都是真数据
        root = tmp_path / "project" / "h2opt"
        directory = open_result_dir(root)
        assert directory.result.scf_converged is True
        assert directory.result.gradient_hartree_per_bohr is not None, "opt 结果必须带终点梯度"
        assert "Ha" in window.energy_label.text()

        # K3 自由形态产物：优化后的几何 + 轨迹
        assert (root / "artifacts" / "optimized.xyz").is_file()
        optimization = json.loads((root / "artifacts" / "optimization.json").read_text(encoding="utf-8"))
        assert optimization["converged"] is True
        assert optimization["used_analytic_gradient"] is True, optimization["analytic_fallback"]

        # 3D：几何取自 optimized.xyz（0.80 Å 的初值已经被优化到约 0.735 Å）
        assert window.scene_panel.pixmap() is not None, "3D 面板没有出图"
        assert rendered and rendered[-1] > 1000, rendered
        atoms = window.scene_panel.scene_atoms()
        assert len(atoms) == 2
        distance = abs(atoms[1].position_angstrom[2] - atoms[0].position_angstrom[2])
        assert 0.70 < distance < 0.78, f"3D 画的是 {distance:.4f} Å，不是优化后的几何"
    finally:
        window.close()


def test_build_job_input_rejects_bad_choices() -> None:
    molecule = Molecule(atoms=[Atom(symbol="H", position_angstrom=(0.0, 0.0, 0.0))])
    try:
        build_job_input(molecule, basis="sto-3g", method="pbe", grid="normal", jobtype="freq")
    except ValueError as exc:
        assert "jobtype" in str(exc)
    else:  # pragma: no cover - 走不到
        raise AssertionError("freq 不该被 GUI 接受")
    try:
        build_job_input(molecule, basis="sto-3g", method="pbe", grid="ultrafine", jobtype="sp")
    except ValueError as exc:
        assert "grid_level" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("超纲的 grid_level 不该被接受")


# ---------------------------------------------------------------------------
# 3. 崩溃后重启恢复未完成视图
# ---------------------------------------------------------------------------


def test_restores_unfinished_view_after_a_crash(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "queue.json").write_text(
        json.dumps(
            [
                {"input": "task-r.yaml", "task_id": "killed", "state": "running"},
                {"input": "task-q.yaml", "task_id": "later", "state": "queued"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    app = _app()
    window = MainWindow(project, workers=1)
    try:
        # 视图被重建：两个任务都在 Task Panel / Job Monitor / 项目树里
        assert window.monitor.rowCount() == 2, window.monitor.rowCount()
        assert window.tree.topLevelItem(0).childCount() == 2
        states = {view.task_id: view for view in window.project.tasks()}
        assert set(states) == {"killed", "later"}

        # 停在 running 的那条：上一次进程没写终态 → K3 唯一合法出路是 failed
        killed = states["killed"]
        assert killed.state == "failed", killed.state
        assert killed.reason and "interrupted" in killed.reason, killed.reason
        # queued 的仍然 queued（可以直接接着跑）
        assert states["later"].state == "queued", states["later"].state
        assert "任务" in window._status.text()
        assert "killed" in window.log_view.toPlainText()

        # 点"运行队列"：queued 的那个真的跑完（mock runner，秒级）
        window.run_queue()
        assert spin_until(app, lambda: not window.bridge.is_running(), timeout=120)
        assert window.project.state_of("later").value == "converged"
        # 被打断的仍然保持 failed，不会被"顺手重跑"（K3 没有 running→queued）
        assert window.project.state_of("killed").value == "failed"
    finally:
        window.close()


def test_restore_keeps_a_finished_task_finished(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "queue.json").write_text(
        json.dumps([{"input": "task-done.yaml", "task_id": "done", "state": "converged"}]),
        encoding="utf-8",
    )
    app = _app()
    window = MainWindow(project, workers=1)
    try:
        assert window.project.state_of("done").value == "converged"
        # 没有 queued 的任务 → run_queue 立刻返回，不会把 converged 改成别的
        window.run_queue()
        assert spin_until(app, lambda: not window.bridge.is_running(), timeout=60)
        assert window.project.state_of("done").value == "converged"
    finally:
        window.close()
