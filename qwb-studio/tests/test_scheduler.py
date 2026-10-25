"""P-03 验收：本地调度器（队列 + 并发 + 取消 + `kill -9` 恢复）。

判据来自 project.md 的 P-03 行：
* 10 个 mock 任务并发跑完；
* 取消 `queued` 与 `running` 都生效（对应 K3 的两条 cancelled 边）；
* 进程被杀后重启能恢复 `queued`。

并发不是靠"跑得快所以应该是并发的"来论证：每个 mock worker 在结果里写下自己的 pid 与
起止时间戳，测试用这些数字算真实的重叠度与并行加速。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from qwb_studio._fs import read_text_retry
from qwb_studio.models import IllegalTransitionError, TaskState
from qwb_studio.scheduler import (
    MOCK_RESULT_FILE,
    QUEUE_FILE,
    WORKER_LOG,
    QueueEntry,
    Scheduler,
    TaskQueue,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def _scheduler(tmp_path: Path, workers: int = 2, runner: str = "mock", grace: float = 1.0) -> Scheduler:
    scheduler = Scheduler(
        queue_path=tmp_path / "queue.json",
        out_root=tmp_path / "out",
        workers=workers,
        runner_name=runner,
        grace_seconds=grace,
    )
    scheduler.load()
    return scheduler


def _result(scheduler: Scheduler, task_id: str) -> dict:
    return json.loads((scheduler.out_root / task_id / MOCK_RESULT_FILE).read_text(encoding="utf-8"))


def _wait_for(predicate, timeout: float, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 并发
# ---------------------------------------------------------------------------


def test_ten_mock_tasks_run_concurrently(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=4)
    for index in range(10):
        scheduler.submit(f"t{index:02d}", f"input-{index:02d}.yaml", {"seconds": 0.4, "label": str(index)})

    started = time.monotonic()
    results = scheduler.run()
    elapsed = time.monotonic() - started

    assert results["counts"]["converged"] == 10, results
    assert results["counts"]["failed"] == 0

    values = [_result(scheduler, f"t{index:02d}")["value"] for index in range(10)]
    # 每个任务一个独立进程
    assert len({value["pid"] for value in values}) == 10
    # 每个任务确实各跑了约 0.4 s
    for value in values:
        assert value["finished_unix"] - value["started_unix"] >= 0.35

    # 真实并发度：按区间重叠算，同一时刻至少 3 个 worker 在跑（4 个槽 + 启动抖动）
    events = []
    for value in values:
        events.append((value["started_unix"], 1))
        events.append((value["finished_unix"], -1))
    events.sort()
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    assert peak >= 3, f"peak concurrency = {peak}, expected >= 3"

    # 也算一下并行加速：串行至少 4 s，4 槽应当明显更快
    assert elapsed < 3.0, f"elapsed {elapsed:.2f}s suggests it ran serially"

    queue = json.loads((tmp_path / QUEUE_FILE).read_text(encoding="utf-8"))
    assert [entry["state"] for entry in queue] == ["converged"] * 10


# ---------------------------------------------------------------------------
# 取消
# ---------------------------------------------------------------------------


def test_cancel_queued_task_never_starts(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=1)
    for index in range(3):
        scheduler.submit(f"t{index}", f"input-{index}.yaml", {"seconds": 0.3})

    worker = threading.Thread(target=scheduler.run)
    worker.start()
    try:
        assert _wait_for(lambda: scheduler.state_of("t0") is TaskState.RUNNING, timeout=10.0)
        assert scheduler.request_cancel("t2") is TaskState.CANCELLED
        assert scheduler.state_of("t2") is TaskState.CANCELLED
    finally:
        worker.join(timeout=60)
    assert not worker.is_alive()

    assert scheduler.state_of("t0") is TaskState.CONVERGED
    assert scheduler.state_of("t1") is TaskState.CONVERGED
    assert scheduler.state_of("t2") is TaskState.CANCELLED
    # 被取消的 queued 任务根本没启动：没有日志、没有结果文件
    assert not (scheduler.out_root / "t2" / WORKER_LOG).exists()
    assert not (scheduler.out_root / "t2" / MOCK_RESULT_FILE).exists()


def test_cancel_running_task_stops_cooperatively(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=1)
    scheduler.submit("long", "long.yaml", {"seconds": 30.0})

    worker = threading.Thread(target=scheduler.run)
    worker.start()
    try:
        assert _wait_for(lambda: scheduler.state_of("long") is TaskState.RUNNING, timeout=10.0)
        assert _wait_for(lambda: (tmp_path / "out" / "long" / WORKER_LOG).exists(), timeout=10.0)
        started = time.monotonic()
        scheduler.request_cancel("long")
        assert _wait_for(
            lambda: scheduler.state_of("long") is TaskState.CANCELLED, timeout=10.0
        ), "cancel must land in the K3 terminal state"
        assert time.monotonic() - started < 6.0
    finally:
        worker.join(timeout=60)
    assert not worker.is_alive()
    assert not (scheduler.out_root / "long" / MOCK_RESULT_FILE).exists()


def test_cancel_running_task_that_ignores_the_flag_is_killed(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=1, runner="stubborn", grace=0.5)
    scheduler.submit("stubborn", "stubborn.yaml", {"seconds": 60.0})

    worker = threading.Thread(target=scheduler.run)
    worker.start()
    try:
        assert _wait_for(lambda: scheduler.state_of("stubborn") is TaskState.RUNNING, timeout=10.0)
        scheduler.request_cancel("stubborn")
        started = time.monotonic()
        assert _wait_for(
            lambda: scheduler.state_of("stubborn") is TaskState.CANCELLED, timeout=15.0
        ), "a worker that ignores the cancel flag must be killed"
        # 协作取消失败 → 走 kill 路径，至少要把 grace 等满
        assert time.monotonic() - started >= 0.4
    finally:
        worker.join(timeout=60)
    assert not worker.is_alive()
    assert (scheduler.out_root / "stubborn" / WORKER_LOG).exists()


def test_cancel_after_terminal_state_is_illegal(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=1)
    scheduler.submit("done", "done.yaml", {"seconds": 0.05})
    scheduler.run()
    assert scheduler.state_of("done") is TaskState.CONVERGED
    with pytest.raises(IllegalTransitionError):
        scheduler.request_cancel("done")


# ---------------------------------------------------------------------------
# kill -9 恢复
# ---------------------------------------------------------------------------


def _spawn_scheduler_cli(tmp_path: Path, count: int, hold: float) -> tuple[subprocess.Popen, Path]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable,
        "-m",
        "qwb_studio.scheduler",
        "--queue",
        str(tmp_path / QUEUE_FILE),
        "--out",
        str(tmp_path / "out"),
        "--count",
        str(count),
        "--hold",
        str(hold),
        "--workers",
        "1",
    ]
    kwargs = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True  # 让整组进程可以被一次 SIGKILL 干掉
    log_path = tmp_path / "scheduler-cli.log"
    log = open(log_path, "wb")  # noqa: SIM115 - 交给子进程，测试结束时由进程退出关闭
    process = subprocess.Popen(  # noqa: S603 - 参数全由本测试构造
        command,
        cwd=str(PKG_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    return process, log_path


def _read_queue(path: Path) -> list[dict]:
    for _ in range(100):
        try:
            return json.loads(read_text_retry(path))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    raise AssertionError(f"cannot read {path}")


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], check=False, capture_output=True)
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait(timeout=30)


def test_kill9_then_restart_recovers_queued_and_fails_running(tmp_path: Path) -> None:
    queue_path = tmp_path / QUEUE_FILE
    process, log_path = _spawn_scheduler_cli(tmp_path, count=4, hold=5.0)
    try:
        # 等第一个任务进入 running（队列已落盘），其余仍 queued
        def one_running_and_one_queued() -> bool:
            entries = _read_queue(queue_path)
            states = [entry["state"] for entry in entries]
            return states.count("running") >= 1 and states.count("queued") >= 1

        ready = _wait_for(one_running_and_one_queued, timeout=120.0)
        assert ready, (
            f"scheduler CLI never reached running+queued: {_read_queue(queue_path)}\n"
            f"returncode={process.poll()}\n"
            f"log:\n{log_path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
        )
        _kill_tree(process)
    finally:
        if process.poll() is None:  # pragma: no cover - 兜底
            process.kill()
        process.wait(timeout=30)

    after_kill = {entry["task_id"]: entry["state"] for entry in _read_queue(queue_path)}
    assert after_kill["t00"] == "running", after_kill
    assert [after_kill[f"t{index:02d}"] for index in range(1, 4)] == ["queued"] * 3

    # 重启：running 落 failed（K3 没有 running→queued 边），queued 继续跑完
    scheduler = _scheduler(tmp_path, workers=1)
    results = scheduler.run()
    assert results["interrupted"] == ["t00"]
    assert results["outcomes"]["t00"] == "failed"
    for index in range(1, 4):
        assert results["outcomes"][f"t{index:02d}"] == "converged"
    interrupted = queue_entry(queue_path, "t00")
    assert interrupted["state"] == "failed"
    assert "interrupted" in interrupted["reason"]
    # 被中断的任务没有结果文件（没假装它成功了）
    assert not (tmp_path / "out" / "t00" / MOCK_RESULT_FILE).exists()


def queue_entry(path: Path, task_id: str) -> dict:
    for entry in _read_queue(path):
        if entry["task_id"] == task_id:
            return entry
    raise AssertionError(f"{task_id} not in {path}")


def test_recover_marks_only_running_entries(tmp_path: Path) -> None:
    queue_path = tmp_path / QUEUE_FILE
    queue_path.write_text(
        json.dumps(
            [
                {"input": "a.yaml", "task_id": "a", "state": "converged"},
                {"input": "b.yaml", "task_id": "b", "state": "running"},
                {"input": "c.yaml", "task_id": "c", "state": "queued"},
                {"input": "d.yaml", "task_id": "d", "state": "cancelled"},
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    scheduler = _scheduler(tmp_path)
    interrupted = scheduler.recover()
    assert [entry.task_id for entry in interrupted] == ["b"]
    states = {entry.task_id: entry.state for entry in scheduler.entries}
    assert states["a"] is TaskState.CONVERGED
    assert states["b"] is TaskState.FAILED
    assert states["c"] is TaskState.QUEUED
    assert states["d"] is TaskState.CANCELLED
    assert "interrupted" in queue_entry(queue_path, "b")["reason"]


# ---------------------------------------------------------------------------
# 队列格式（与 Rust `qwb batch` 互通）
# ---------------------------------------------------------------------------


def test_queue_json_uses_the_rust_batch_field_names(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, workers=1)
    scheduler.submit("one", "one.yaml", {"seconds": 0.05})
    scheduler.run()

    raw = json.loads((tmp_path / QUEUE_FILE).read_text(encoding="utf-8"))
    assert len(raw) == 1
    assert set(raw[0]) == {"input", "task_id", "state"}
    assert raw[0] == {"input": "one.yaml", "task_id": "one", "state": "converged"}
    # 原子写：不留 .tmp 残骸
    assert not (tmp_path / "queue.json.tmp").exists()


def test_accepts_a_rust_written_queue_file(tmp_path: Path) -> None:
    queue_path = tmp_path / QUEUE_FILE
    queue_path.write_text(
        json.dumps(
            [{"input": "in-a.yaml", "task_id": "a", "state": "queued"}],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    scheduler = _scheduler(tmp_path, workers=1)
    results = scheduler.run()
    assert results["outcomes"] == {"a": "converged"}


def test_bad_queue_file_is_a_clear_error(tmp_path: Path) -> None:
    queue_path = tmp_path / QUEUE_FILE
    queue_path.write_text("{ not a list }", encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        TaskQueue(queue_path).load()
    assert "queue" in str(excinfo.value).lower()

    queue_path.write_text(json.dumps([{"input": "a", "task_id": "a", "state": "nope"}]), encoding="utf-8")
    with pytest.raises(Exception):
        TaskQueue(queue_path).load()


def test_queue_entry_transition_is_checked(tmp_path: Path) -> None:
    entry = QueueEntry(input="x.yaml", task_id="x", state=TaskState.QUEUED)
    assert entry.transition(TaskState.RUNNING).state is TaskState.RUNNING
    with pytest.raises(IllegalTransitionError):
        entry.transition(TaskState.CONVERGED)
