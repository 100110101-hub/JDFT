"""P-09 验收：GUI 骨架（PySide6，`QT_QPA_PLATFORM=offscreen` 下 headless 跑）。

判据（project.md §7 P-09 / T8）：
* **创建 / 提交 / 查看 mock 任务不阻塞 UI** —— 不是"看起来不卡"，而是用 QTimer 计数：
  任务运行期间事件循环必须持续推进；
* **大日志（> 100 MB）不卡** —— `LogView` 流式追加，事件循环在流式过程中仍在转，
  且文档块数被上限截住（内存有界）；
* 结果面板能显示能量 / 偶极 / SCF 状态（喂一个真实的 K3 结果目录）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

# 硬导入：缺 PySide6 就让测试**失败**而不是 skip —— 否则 P-09 的验收在缺依赖的机器上
# 会"静悄悄地通过"。依赖装法见 README 的 .venv 一节。
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import qwb_engine as qe  # noqa: E402
from qwb_studio.gui.logview import LogView  # noqa: E402
from qwb_studio.gui.mainwindow import MainWindow  # noqa: E402
from qwb_studio.runner import run_job_to_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


class Ticker:
    """数事件循环转了多少圈（QTimer 0 ms）。"""

    def __init__(self, app: QApplication, interval: int = 0) -> None:
        self.count = 0
        self._timer = QTimer()
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._tick)
        self._app = app

    def _tick(self) -> None:
        self.count += 1

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()


def spin_until(app: QApplication, predicate, timeout: float = 120.0) -> bool:
    """转事件循环直到 predicate() 为真（每圈只 mutate 一点点，不 sleep 整块）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_submit_and_monitor_do_not_block_the_ui(app, tmp_path: Path) -> None:
    window = MainWindow(tmp_path / "project", workers=2)
    try:
        for index in range(3):
            window.id_edit.setText(f"t{index}")
            window.seconds_spin.setValue(0.3)
            window.submit_task()
        assert len(window.project.tasks()) == 3
        assert window.monitor.rowCount() == 3

        ticker = Ticker(app)
        ticker.start()
        window.run_queue()
        assert window.bridge.is_running()
        finished = spin_until(app, lambda: not window.bridge.is_running(), timeout=120)
        ticker.stop()

        assert finished, "运行没在 120 s 内结束"
        assert ticker.count > 50, f"任务运行期间事件循环只转了 {ticker.count} 圈：UI 被阻塞了"
        states = [view.state for view in window.project.tasks()]
        assert states == ["converged"] * 3, states
        # 结果面板能跟着刷新（mock 结果目录里有 job_spec/worker.log）
        assert window.monitor.rowCount() == 3
        assert window.tree.topLevelItem(0).childCount() == 3
    finally:
        window.close()


def test_cancel_from_ui(app, tmp_path: Path) -> None:
    window = MainWindow(tmp_path / "project", workers=1)
    try:
        window.id_edit.setText("long")
        window.seconds_spin.setValue(20.0)
        window.submit_task()
        window.run_queue()
        assert spin_until(app, lambda: window.project.state_of("long").value == "running", timeout=60)
        window.monitor.selectRow(0)
        window.cancel_selected()
        cancelled = spin_until(
            app, lambda: window.project.state_of("long").value == "cancelled", timeout=60
        )
        assert cancelled, [view.state for view in window.project.tasks()]
        # 注意：这里必须继续转事件循环，不能直接 `bridge.wait()` ——
        # `finished` 是从工作线程排到界面线程的队列信号，主线程一旦阻塞就永远收不到。
        assert spin_until(app, lambda: not window.bridge.is_running(), timeout=60), (
            "取消之后工作线程没有退出"
        )
    finally:
        window.close()


def test_big_log_streams_without_freezing_the_ui(app, tmp_path: Path) -> None:
    log_path = tmp_path / "big.log"
    line = "scf: iteration 000  E = -76.0000000000  dE = 1.0e-09  |dP| = 2.0e-08\n"
    target_bytes = 101 * 1024 * 1024
    repeats = target_bytes // len(line) + 1
    with open(log_path, "w", encoding="utf-8") as handle:
        block = line * 8192
        written = 0
        while written < target_bytes:
            handle.write(block)
            written += len(block)
    size_mb = log_path.stat().st_size / (1024 * 1024)
    assert size_mb > 100, f"日志只有 {size_mb:.1f} MB"

    view = LogView()
    ticker = Ticker(app)
    ticker.start()
    began = time.monotonic()
    view.feed_file(log_path)
    done = spin_until(app, lambda: not view.is_streaming(), timeout=180)
    elapsed = time.monotonic() - began
    ticker.stop()

    assert done, "日志流式追加没有在 180 s 内结束"
    assert view.total_lines > 1_000_000, view.total_lines
    assert ticker.count > 20, f"流式追加期间事件循环只转了 {ticker.count} 圈"
    assert view.document().blockCount() <= LogView.DEFAULT_MAX_BLOCKS + 2
    assert elapsed < 180
    print(f"{size_mb:.1f} MB / {view.total_lines} 行：{elapsed:.1f} s，事件循环 {ticker.count} 圈")
    try:
        log_path.unlink()
    except OSError:
        pass


def test_result_panel_shows_a_real_result(app, tmp_path: Path) -> None:
    molecule = qe.Molecule([qe.Atom("H", (0.0, 0.0, -0.37)), qe.Atom("H", (0.0, 0.0, 0.37))])
    job = qe.JobInput(molecule, "sto-3g", "svwn5")
    outcome = run_job_to_dir(job, tmp_path / "runs", "h2", data_root=DATA_ROOT)

    window = MainWindow(tmp_path / "project", workers=1)
    try:
        window.set_result_root(outcome.root)
        text = window.energy_label.text()
        assert "能量：" in text and "Ha" in text
        assert "-1.157" in text, text
        assert "偶极：" in window.dipole_label.text()
        assert "收敛" in window.converged_label.text()
        assert "env.json：rustc" in window.env_label.text()
    finally:
        window.close()


def test_gui_reports_a_broken_result_directory(app, tmp_path: Path) -> None:
    root = tmp_path / "no-env"
    root.mkdir()
    for name in ("input.yaml", "result.json", "log.txt", "data_snapshot.json"):
        (root / name).write_text("{}", encoding="utf-8")
    window = MainWindow(tmp_path / "project", workers=1)
    try:
        window.set_result_root(root)
        assert "不可用" in window.energy_label.text()
        assert "env.json" in window.env_label.text()
    finally:
        window.close()
