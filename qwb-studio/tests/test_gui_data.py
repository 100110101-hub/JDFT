"""Data 页（P-09 扩展）：第三方数据的**手动**下载入口。

背景：`qwb-data/basis|ecp/` 的部分文件是 BSE 导出的第三方数据，仓库不提交它们
（再分发条款）。计算时 Rust CLI 会自动取回；本页是给"想提前备好 / 网络受限"的场景
留的手动入口，因此它必须满足三条：

* 状态表能反映 `sources.json` 的每条清单（缺失/不符都要看得见）；
* 下载在**工作线程**里跑，事件循环不阻塞（用 QTimer 计数验证，不看"感觉"）；
* 失败要显示在界面上（不是静默 nothing happened）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qwb_studio.gui.datapanel import STATE_LABELS, DataPanel  # noqa: E402
from qwb_studio.gui.mainwindow import MainWindow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def spin_until(app: QApplication, predicate, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


class Ticker:
    def __init__(self) -> None:
        self.count = 0
        self._timer = QTimer()
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self.count += 1

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()


def test_panel_lists_every_manifest_entry(app) -> None:
    panel = DataPanel(DATA_ROOT)
    try:
        statuses = panel.statuses()
        assert statuses, "sources.json 应当至少有一条第三方数据"
        assert panel.table.rowCount() == len(statuses)
        for row, status in enumerate(statuses):
            assert panel.table.item(row, 0).text() == status.path
            assert panel.table.item(row, 1).text() == STATE_LABELS[status.state]
    finally:
        panel.deleteLater()


def test_panel_finds_missing_and_offers_download(app, tmp_path) -> None:
    """把清单指向一个空目录：所有条目都该显示为缺失，且"下载缺失"可用。"""
    root = tmp_path / "qwb-data"
    root.mkdir()
    (root / "sources.json").write_text(
        (DATA_ROOT / "sources.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    panel = DataPanel(root)
    try:
        statuses = panel.statuses()
        assert statuses and all(status.state == "missing" for status in statuses)
        assert panel.fetch_missing_button.isEnabled()
        assert panel.fetch_all_button.isEnabled()
    finally:
        panel.deleteLater()


def test_download_runs_off_the_ui_thread(app, tmp_path) -> None:
    """假 fetcher：睡 200 ms 再返回。期间事件循环必须继续转（ticker 在涨）。"""
    root = tmp_path / "qwb-data"
    root.mkdir()
    (root / "sources.json").write_text(
        (DATA_ROOT / "sources.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    def fake_fetcher(data_root, *, only=None, force=False, log=print):
        log("fake fetch 开始")
        time.sleep(0.2)
        log("fake fetch 结束")
        return {"basis/sto-3g.yaml": "0" * 64}

    panel = DataPanel(root, fetcher=fake_fetcher)
    ticker = Ticker()
    ticker.start()
    try:
        assert panel.start_fetch(force=False) is True
        assert panel.is_busy() is True, "下载应当在后台线程里跑"
        assert spin_until(app, lambda: not panel.is_busy(), timeout=30.0)
        assert ticker.count > 0, "下载期间事件循环被阻塞了"
        assert spin_until(app, lambda: "完成" in panel.log.toPlainText(), timeout=5.0)
        assert "fake fetch 结束" in panel.log.toPlainText()
    finally:
        ticker.stop()
        panel.deleteLater()


def test_download_failure_is_shown(app, tmp_path) -> None:
    root = tmp_path / "qwb-data"
    root.mkdir()
    (root / "sources.json").write_text(
        (DATA_ROOT / "sources.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    def broken_fetcher(*args, **kwargs):
        raise RuntimeError("网络不通")

    panel = DataPanel(root, fetcher=broken_fetcher)
    try:
        assert panel.start_fetch(force=True) is True
        assert spin_until(app, lambda: "失败" in panel.log.toPlainText(), timeout=30.0)
        assert "网络不通" in panel.log.toPlainText()
    finally:
        panel.deleteLater()


def test_main_window_exposes_the_data_tab(app, tmp_path) -> None:
    window = MainWindow(tmp_path / "project", workers=1, data_root=DATA_ROOT)
    try:
        titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert "Data" in titles
        assert window.data_panel.table.rowCount() == len(window.data_panel.statuses())
    finally:
        window.close()
        window.deleteLater()
