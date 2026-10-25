"""真作业走调度器（`engine` runner）：CLI 当 worker，结果目录是真正的 K3 目录。

这一组把"调度器 + 真引擎"接起来：
* 成功：`qwb run` 退出 0、`<out>/<task_id>/result.json` 存在、`open_result_dir` 能读；
* 失败：坏 input.yaml → CLI 退出 1 → 任务 `failed`，`reason` 里带 worker 日志的尾巴。

`engine` runner 的临时文件（worker.log / 取消标志）放在 `<out>/.staging/`，
不污染 K3 结果目录。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from qwb_studio.resultdir import open_result_dir
from qwb_studio.scheduler import ENGINE_RUNNER, QUEUE_FILE, Scheduler

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"
EXAMPLE = QWB_CORE / "examples" / "h2_sto3g_svwn5.yaml"


def _scheduler(tmp_path: Path) -> Scheduler:
    scheduler = Scheduler(
        queue_path=tmp_path / QUEUE_FILE,
        out_root=tmp_path / "out",
        workers=1,
        runner_name=ENGINE_RUNNER,
        grace_seconds=1.0,
    )
    scheduler.load()
    return scheduler


def test_engine_runner_produces_a_real_k3_directory(tmp_path: Path) -> None:
    assert EXAMPLE.is_file(), EXAMPLE
    staged = tmp_path / "h2.yaml"
    staged.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    scheduler = _scheduler(tmp_path)
    scheduler.submit("h2", str(staged))

    results = scheduler.run()
    assert results["outcomes"]["h2"] == "converged", results
    root = tmp_path / "out" / "h2"
    assert (root / "result.json").is_file()
    directory = open_result_dir(root)
    assert directory.result.scf_converged is True
    assert directory.result.energy_hartree == pytest.approx(-1.1570005110, abs=1e-9)
    assert directory.env.rustc_version.startswith("rustc ")
    # 临时文件不落在结果目录里
    assert not (root / "worker.log").exists()
    assert (tmp_path / "out" / ".staging" / "h2" / "worker.log").is_file()


def test_engine_runner_failure_carries_the_worker_log(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("molecule: {atoms: 3}\n", encoding="utf-8")
    scheduler = _scheduler(tmp_path)
    scheduler.submit("broken", str(broken))

    started = time.monotonic()
    results = scheduler.run()
    assert results["outcomes"]["broken"] == "failed", results
    assert time.monotonic() - started < 120
    entry = scheduler._queue.get("broken")
    assert entry.reason, "失败必须带原因"
    assert "code 1" in entry.reason or "input" in entry.reason.lower(), entry.reason


def test_engine_runner_cancel_queued_task_never_runs(tmp_path: Path) -> None:
    staged = tmp_path / "h2.yaml"
    staged.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    scheduler = _scheduler(tmp_path)
    scheduler.submit("h2", str(staged))
    assert scheduler.request_cancel("h2").value == "cancelled"
    results = scheduler.run()
    assert results["outcomes"]["h2"] == "cancelled"
    assert not (tmp_path / "out" / "h2").exists()


def test_engine_runner_env_override_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`QWB_ENGINE_BIN` 指错时，任务必须以 failed 结束（不是静默 converged）。"""
    staged = tmp_path / "h2.yaml"
    staged.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("QWB_ENGINE_BIN", str(tmp_path / "no-such-engine"))
    scheduler = _scheduler(tmp_path)
    scheduler.submit("h2", str(staged))
    results = scheduler.run()
    assert results["outcomes"]["h2"] == "failed", results
    shutil.rmtree(tmp_path / "out", ignore_errors=True)
    monkeypatch.delenv("QWB_ENGINE_BIN", raising=False)
    assert os.environ.get("QWB_ENGINE_BIN") is None
