"""原子替换在 Windows 上必须容忍"目标文件正被读"（P-03 的真实回归）。

背景：`os.replace` 在 Windows 上是 `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`，
当目标文件被别的进程打开（Python 的 `open()` 不带 `FILE_SHARE_DELETE`）时会抛
`PermissionError`。这个 bug 是在全量测试 + 门禁同时跑机器时被 P-03 的
`kill -9` 测试踩出来的：测试每 10 ms 读一次 `queue.json`，调度器恰好那时替换文件。

这里用一个"前几次 `os.replace` 假装失败"的探针，把重试逻辑钉死，
再加一个**真并发**用例（读线程 + 写线程）作为端到端证据。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from qwb_studio._fs import atomic_replace, read_text_retry
from qwb_studio.scheduler import QUEUE_FILE, Scheduler


def test_retries_through_a_transient_sharing_violation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(source, destination):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "The process cannot access the file", str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky)
    tmp = tmp_path / "queue.json.tmp"
    target = tmp_path / "queue.json"
    tmp.write_text("new\n", encoding="utf-8")
    atomic_replace(tmp, target)
    assert calls["n"] == 4
    assert target.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()


def test_gives_up_loudly_when_the_target_never_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def always_busy(source, destination):  # type: ignore[no-untyped-def]
        raise PermissionError(13, "still open", str(destination))

    monkeypatch.setattr(os, "replace", always_busy)
    tmp = tmp_path / "x.tmp"
    tmp.write_text("x", encoding="utf-8")
    with pytest.raises(PermissionError):
        atomic_replace(tmp, tmp_path / "x", attempts=5)


def test_queue_saves_survive_a_concurrent_reader(tmp_path: Path) -> None:
    """读线程一直读 queue.json，写线程一直原子替换：两边都不许炸。"""
    scheduler = Scheduler(queue_path=tmp_path / QUEUE_FILE, out_root=tmp_path / "out", workers=1)
    scheduler.load()
    scheduler.submit("t0", "input-t0.yaml", {"seconds": 0.01})

    errors: list[BaseException] = []
    stop = threading.Event()
    reads = {"n": 0}

    def reader() -> None:
        path = tmp_path / QUEUE_FILE
        while not stop.is_set():
            try:
                text = read_text_retry(path)
                json.loads(text)
                reads["n"] += 1
            except FileNotFoundError:
                continue
            except BaseException as exc:  # noqa: BLE001 - 记下来后面断言
                errors.append(exc)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(60):
            scheduler.save()
            time.sleep(0.002)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors, f"concurrent reader failed: {errors!r}"
    assert reads["n"] > 0
    assert json.loads((tmp_path / QUEUE_FILE).read_text(encoding="utf-8"))
