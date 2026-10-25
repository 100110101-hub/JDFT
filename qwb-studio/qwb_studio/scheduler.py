"""本地调度器（P-03）：队列 + 子进程 worker + K3 状态机 + 取消 + `kill -9` 恢复。

设计要点
========

* **队列文件与 Rust `qwb batch` 同格式**：`<out>/queue.json` 是一个 JSON 数组，每项
  `{input, task_id, state}`（`state` 用 K3 的小写状态名）。原子写（`.tmp` + `os.replace`），
  所以被 `kill -9` 也不会留下半截文件。
* **状态转移只走 K3 的 6 条边**：`queued → running` **先落盘再干活**；`running → converged/failed/cancelled`；
  `queued → cancelled`。任何非法转移都抛 `IllegalTransitionError`（`ValueError`），不靠约定。
* **worker 是真子进程**（不是线程、也不是 multiprocessing）：每个任务一个进程，
  spec 与结果都走磁盘文件，worker 的 stdout/stderr 进 `worker.log`。
  这样 Windows 上不需要 spawn 的 pickle 规则，取消/杀进程的语义也和 P-07 真正调用
  `qwb run` 的形态一致。
* **取消**：
  - `queued`：直接 `queued → cancelled`（K3 v2.1 的边），任务根本不启动；
  - `running`：先写协作取消标志文件（worker 在轮询点自己退出），超过 `grace_seconds`
    还不退出就 `kill()` 进程 —— **然后**才写 `cancelled`，不会出现"标记取消但进程还在跑"。
* **`kill -9` 恢复**：进程被杀时队列里会留下 `running`。重启后按 K3 处理：`running → failed`
  （`reason` 记 `interrupted`），`queued` 的继续跑。Rust 的 `qwb batch` 对同一情形是
  **报告但不改状态**（它没有 reason 字段，等人用新 task id 重提）；Python 落到终态 `failed`
  是合法边，也如实报告，两者都符合 K3，差异记在 docs/status.yaml。
* **mock 任务**：P-03 的验收是调度语义，不是电子结构，所以 `mock` / `stubborn` 两个 runner
  只做可观测的"耗时 + 取消响应"，产物叫 `mock_result.json`（刻意不叫 `result.json`，
  免得被当成 K3 的真实结果）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ._fs import atomic_replace, read_text_retry
from .models import TaskState, utc_now_iso

__all__ = [
    "MOCK_RUNNERS",
    "CancelToken",
    "QueueEntry",
    "Scheduler",
    "SchedulerError",
    "TaskQueue",
    "cancel_queued_task",
    "mock_runner",
    "stubborn_runner",
]

QUEUE_FILE = "queue.json"
MOCK_RESULT_FILE = "mock_result.json"
SPEC_FILE = "job_spec.json"
WORKER_LOG = "worker.log"
CANCEL_FLAG = "cancel.requested"
CANCEL_ACK_FILE = "cancel_ack.json"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class SchedulerError(Exception):
    """调度器用不了的状态（队列文件损坏、worker 崩溃等）。"""


# ---------------------------------------------------------------------------
# 队列
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueEntry:
    """队列里的一条记录；三个主字段名与 Rust `QueueEntry` 逐字一致。"""

    input: str
    task_id: str
    state: TaskState = TaskState.QUEUED
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "input": self.input,
            "task_id": self.task_id,
            "state": self.state.value,
        }
        if self.reason is not None:
            obj["reason"] = self.reason
        return obj

    @classmethod
    def from_json(cls, obj: Any) -> "QueueEntry":
        if not isinstance(obj, dict):
            raise SchedulerError(f"queue entry must be an object, got {type(obj).__name__}")
        for key in ("input", "task_id", "state"):
            if key not in obj:
                raise SchedulerError(f"queue entry is missing {key!r}: {obj!r}")
        try:
            state = TaskState(obj["state"])
        except ValueError as exc:
            raise SchedulerError(f"unknown task state {obj['state']!r}") from exc
        reason = obj.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise SchedulerError(f"queue entry reason must be a string: {obj!r}")
        return cls(input=str(obj["input"]), task_id=str(obj["task_id"]), state=state, reason=reason)

    def transition(self, nxt: TaskState, reason: str | None = None) -> "QueueEntry":
        """按 K3 转移（非法转移抛 `IllegalTransitionError`）。"""
        return replace(self, state=self.state.transition_to(nxt), reason=reason)


class TaskQueue:
    """`<out>/queue.json` 的读写（原子写，`kill -9` 安全）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[QueueEntry] = []

    def load(self) -> list[QueueEntry]:
        if not self.path.is_file():
            self._entries = []
            return list(self._entries)
        try:
            raw = json.loads(read_text_retry(self.path))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchedulerError(f"{self.path} is not a valid queue file: {exc}") from exc
        if not isinstance(raw, list):
            raise SchedulerError(f"{self.path} must contain a JSON array, got {type(raw).__name__}")
        self._entries = [QueueEntry.from_json(item) for item in raw]
        ids = [entry.task_id for entry in self._entries]
        if len(set(ids)) != len(ids):
            raise SchedulerError(f"{self.path} has duplicate task ids: {ids}")
        return list(self._entries)

    def save(self) -> None:
        payload = json.dumps([entry.to_json() for entry in self._entries], indent=2, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        # Windows：读者（GUI / kill -9 测试）可能正开着 queue.json，直接 os.replace 会
        # 抛共享冲突。atomic_replace 会短暂重试，POSIX 上就是一次 rename。
        atomic_replace(tmp, self.path)

    @property
    def entries(self) -> list[QueueEntry]:
        return list(self._entries)

    def get(self, task_id: str) -> QueueEntry:
        for entry in self._entries:
            if entry.task_id == task_id:
                return entry
        raise SchedulerError(f"task {task_id!r} is not in the queue")

    def add(self, entry: QueueEntry) -> None:
        if any(existing.task_id == entry.task_id for existing in self._entries):
            raise SchedulerError(f"duplicate task id {entry.task_id!r}")
        if entry.state is not TaskState.QUEUED:
            raise SchedulerError(f"new tasks must be queued, got {entry.state.value}")
        self._entries.append(entry)

    def set_state(self, task_id: str, nxt: TaskState, reason: str | None = None) -> QueueEntry:
        for index, entry in enumerate(self._entries):
            if entry.task_id == task_id:
                updated = entry.transition(nxt, reason)
                self._entries[index] = updated
                return updated
        raise SchedulerError(f"task {task_id!r} is not in the queue")


# ---------------------------------------------------------------------------
# 取消令牌（子进程侧读标志文件；父进程侧写标志文件）
# ---------------------------------------------------------------------------


class CancelToken:
    """跨进程的协作取消标志：`wait(t)` 轮询标志文件，返回 True 表示要停。"""

    def __init__(self, path: str | Path | None, poll_interval: float = 0.01) -> None:
        self.path = Path(path) if path is not None else None
        self.poll_interval = poll_interval
        self._set = False

    def is_set(self) -> bool:
        if self._set:
            return True
        if self.path is not None and self.path.exists():
            self._set = True
        return self._set

    def set(self) -> None:
        self._set = True
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("cancel\n", encoding="utf-8")

    def wait(self, timeout: float | None = None) -> bool:
        """最多等 `timeout` 秒；期间被取消则立即返回 True（模仿 `threading.Event.wait`）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.is_set():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(min(self.poll_interval, 0.05))


# ---------------------------------------------------------------------------
# runner（子进程里跑的东西；模块级函数，便于 `python -m` 直接调用）
# ---------------------------------------------------------------------------


def mock_runner(payload: Mapping[str, Any], cancel: CancelToken, poll_interval: float) -> dict[str, Any]:
    """Mock 任务：睡 `seconds` 秒，可被协作取消；记录自己的 pid 与起止时间。"""
    started = time.time()
    deadline = started + float(payload.get("seconds", 0.1))
    ticks = 0
    while time.time() < deadline:
        if cancel.wait(poll_interval):
            return {"cancelled": True, "cancelled_after": time.time() - started, "ticks": ticks}
        ticks += 1
    return {
        "cancelled": False,
        "pid": os.getpid(),
        "started_unix": started,
        "finished_unix": time.time(),
        "ticks": ticks,
        "label": payload.get("label"),
    }


def stubborn_runner(payload: Mapping[str, Any], cancel: CancelToken, poll_interval: float) -> dict[str, Any]:
    """故意**不检查**取消标志的 worker：用来验证调度器会 kill 它。"""
    started = time.time()
    end = started + float(payload.get("seconds", 30.0))
    while time.time() < end:
        time.sleep(0.05)
    return {"cancelled": False, "pid": os.getpid(), "started_unix": started, "finished_unix": time.time()}


MOCK_RUNNERS: dict[str, Callable[[Mapping[str, Any], CancelToken, float], Any]] = {
    "mock": mock_runner,
    "stubborn": stubborn_runner,
}

Runner = Callable[[Mapping[str, Any], CancelToken, float], Any]


def _worker_main(args: argparse.Namespace) -> int:
    """`--worker` 模式：读 spec，跑 runner，把结果写进 `--result`（被取消则写 `--cancel-ack`）。

    `mock_result.json` 的语义是"这个任务真的产出了结果"，所以取消**不写**它 ——
    否则"有结果文件"这个判据会被取消污染。
    """
    spec = json.loads(Path(args.worker).read_text(encoding="utf-8"))
    runner = MOCK_RUNNERS[args.runner]
    token = CancelToken(args.cancel, args.poll)
    outcome = runner(spec.get("payload", {}), token, args.poll)
    if token.is_set():
        Path(args.cancel_ack).write_text(
            json.dumps({"state": "cancelled", "value": outcome}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0
    Path(args.result).write_text(
        json.dumps({"state": "ok", "value": outcome}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------


@dataclass
class Scheduler:
    """一个最小可用的本地调度器：N 个 worker 槽、K3 状态机、可取消、可恢复。"""

    queue_path: str | Path
    out_root: str | Path
    workers: int = 2
    runner_name: str = "mock"
    poll_interval: float = 0.01
    grace_seconds: float = 1.0
    poll: float = 0.005
    payloads: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    _queue: TaskQueue = field(init=False, repr=False)
    _cancel_requests: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _children: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict, init=False, repr=False)
    _deadlines: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.queue_path = Path(self.queue_path)
        self.out_root = Path(self.out_root)
        self._queue = TaskQueue(self.queue_path)
        if self.runner_name not in MOCK_RUNNERS:
            raise SchedulerError(f"unknown runner {self.runner_name!r}; have {sorted(MOCK_RUNNERS)}")

    # -- 队列 ---------------------------------------------------------------

    def load(self) -> list[QueueEntry]:
        return self._queue.load()

    def save(self) -> None:
        self._queue.save()

    @property
    def entries(self) -> list[QueueEntry]:
        return self._queue.entries

    def submit(self, task_id: str, input_path: str, payload: Mapping[str, Any] | None = None) -> QueueEntry:
        """登记一个 `queued` 任务（同名任务不会重复登记）。"""
        for entry in self._queue.entries:
            if entry.task_id == task_id:
                return entry
        entry = QueueEntry(input=input_path, task_id=task_id, state=TaskState.QUEUED)
        self._queue.add(entry)
        if payload is not None:
            self.payloads[task_id] = dict(payload)
        self._queue.save()
        return entry

    def submit_many(self, specs: Iterable[tuple[str, Any, Mapping[str, Any] | None]]) -> list[QueueEntry]:
        return [self.submit(task_id, input_path, payload) for task_id, input_path, payload in specs]

    def state_of(self, task_id: str) -> TaskState:
        return self._queue.get(task_id).state

    # -- 取消 ---------------------------------------------------------------

    def request_cancel(self, task_id: str, reason: str = "cancelled by user") -> TaskState:
        """请求取消：`queued` 立即落盘；`running` 置协作标志，由 :meth:`run` 收尾。"""
        entry = self._queue.get(task_id)
        if entry.state is TaskState.QUEUED:
            updated = self._queue.set_state(task_id, TaskState.CANCELLED, reason)
            self._queue.save()
            return updated.state
        if entry.state is TaskState.RUNNING:
            self._cancel_requests[task_id] = reason
            self._write_cancel_flag(task_id)
            self._deadlines.setdefault(task_id, time.monotonic() + self.grace_seconds)
            return TaskState.RUNNING
        return entry.state.transition_to(TaskState.CANCELLED)  # 终态 → 抛 IllegalTransitionError

    def cancel_queued_task(self, task_id: str, reason: str = "cancelled by user") -> TaskState:
        queue = TaskQueue(self.queue_path)
        queue.load()
        updated = queue.set_state(task_id, TaskState.CANCELLED, reason)
        queue.save()
        return updated.state

    # -- 执行 ---------------------------------------------------------------

    def recover(self) -> list[QueueEntry]:
        """`kill -9` 之后：`running` 一律落 `failed`（reason = interrupted）。"""
        self._queue.load()
        interrupted: list[QueueEntry] = []
        for entry in self._queue.entries:
            if entry.state is TaskState.RUNNING:
                interrupted.append(
                    self._queue.set_state(
                        entry.task_id,
                        TaskState.FAILED,
                        "interrupted: the previous scheduler process died while this task was running",
                    )
                )
        if interrupted:
            self._queue.save()
        return interrupted

    def run(self) -> dict[str, Any]:
        """把 `queued` 的跑完，返回汇总。"""
        self.out_root.mkdir(parents=True, exist_ok=True)
        interrupted = self.recover()
        for entry in interrupted:
            print(
                f"qwb-scheduler: task {entry.task_id} 上次停在 running（进程被杀）；"
                "K3 没有 running→queued 边，已落 failed，请用新 task id 重提",
                file=sys.stderr,
            )

        pending = [entry.task_id for entry in self._queue.entries if entry.state is TaskState.QUEUED]
        while pending or self._children:
            while pending and len(self._children) < max(1, self.workers):
                self._start(pending.pop(0))
            if not self._children:
                continue
            self._reap()
            time.sleep(self.poll)

        counts = {state.value: 0 for state in TaskState}
        outcomes: dict[str, str] = {}
        for entry in self._queue.entries:
            counts[entry.state.value] += 1
            outcomes[entry.task_id] = entry.state.value
        return {
            "interrupted": [entry.task_id for entry in interrupted],
            "outcomes": outcomes,
            "counts": counts,
            "total": len(self._queue.entries),
        }

    # -- 内部：子进程 -------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return self.out_root / task_id

    def _write_cancel_flag(self, task_id: str) -> None:
        target = self._task_dir(task_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / CANCEL_FLAG).write_text("cancel\n", encoding="utf-8")

    def _start(self, task_id: str) -> None:
        if self._queue.get(task_id).state is not TaskState.QUEUED:
            return  # 在 pending 里被别处取消掉了
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "task_id": task_id,
            "input": self._queue.get(task_id).input,
            "payload": dict(self.payloads.get(task_id, {})),
        }
        (task_dir / SPEC_FILE).write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        self._queue.set_state(task_id, TaskState.RUNNING)
        self._queue.save()  # 先落盘：被 kill -9 时下一轮才能看出它停在 running

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PACKAGE_ROOT), env.get("PYTHONPATH", "")] if env.get("PYTHONPATH") else [str(PACKAGE_ROOT)]
        )
        command = [
            sys.executable,
            "-m",
            "qwb_studio.scheduler",
            "--worker",
            str(task_dir / SPEC_FILE),
            "--cancel",
            str(task_dir / CANCEL_FLAG),
            "--result",
            str(task_dir / MOCK_RESULT_FILE),
            "--cancel-ack",
            str(task_dir / CANCEL_ACK_FILE),
            "--runner",
            self.runner_name,
            "--poll",
            str(self.poll_interval),
        ]
        with open(task_dir / WORKER_LOG, "wb") as log:
            self._children[task_id] = subprocess.Popen(  # noqa: S603 - 参数全由本模块构造
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(PACKAGE_ROOT),
            )

    def _reap(self) -> None:
        for task_id in list(self._children):
            process = self._children[task_id]
            cancelled = task_id in self._cancel_requests
            if cancelled and process.poll() is None:
                deadline = self._deadlines.setdefault(task_id, time.monotonic() + self.grace_seconds)
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait(timeout=10)
                    self._finish(task_id, TaskState.CANCELLED, self._cancel_requests[task_id])
                    continue
            code = process.poll()
            if code is None:
                continue
            if cancelled:
                self._finish(task_id, TaskState.CANCELLED, self._cancel_requests[task_id])
                continue
            if code != 0:
                self._finish(task_id, TaskState.FAILED, f"worker exited with code {code}")
                continue
            result = self._read_result(task_id)
            acked = (self._task_dir(task_id) / CANCEL_ACK_FILE).exists()
            if result is not None:
                self._finish(task_id, TaskState.CONVERGED, None)
            elif acked:
                self._finish(task_id, TaskState.CANCELLED, "cancelled by user")
            else:
                self._finish(task_id, TaskState.FAILED, "worker exited without writing a result file")

    def _read_result(self, task_id: str) -> dict[str, Any] | None:
        path = self._task_dir(task_id) / MOCK_RESULT_FILE
        for _ in range(200):  # 子进程刚退出时文件系统可能还没对父进程可见
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            time.sleep(0.005)
        return None

    def _finish(self, task_id: str, state: TaskState, reason: str | None) -> None:
        process = self._children.pop(task_id, None)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - 不该发生
                process.kill()
                process.wait(timeout=10)
        flag = self._task_dir(task_id) / CANCEL_FLAG
        if flag.exists():
            flag.unlink()
        self._queue.set_state(task_id, state, reason)
        self._queue.save()
        self._cancel_requests.pop(task_id, None)
        self._deadlines.pop(task_id, None)


def cancel_queued_task(queue_path: str | Path, task_id: str, reason: str = "cancelled by user") -> TaskState:
    """便利函数：对一个**不在跑**的队列做取消（`queued → cancelled`）。"""
    queue = TaskQueue(queue_path)
    queue.load()
    updated = queue.set_state(task_id, TaskState.CANCELLED, reason)
    queue.save()
    return updated.state


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwb-scheduler", description="QWB 本地调度器（P-03）")
    parser.add_argument("--queue", help="queue.json 路径")
    parser.add_argument("--out", help="结果根目录")
    parser.add_argument("--workers", type=int, default=2, help="并发 worker 数")
    parser.add_argument("--runner", choices=sorted(MOCK_RUNNERS), default="mock")
    parser.add_argument("--hold", type=float, default=0.2, help="每个 mock 任务的耗时（秒）")
    parser.add_argument("--count", type=int, default=0, help="队列为空时自动登记 count 个任务")
    # worker 模式（父进程内部使用）
    parser.add_argument("--worker", help="(内部) 任务 spec.json")
    parser.add_argument("--cancel", help="(内部) 取消标志文件")
    parser.add_argument("--result", help="(内部) 结果文件")
    parser.add_argument("--cancel-ack", help="(内部) 取消回执文件")
    parser.add_argument("--poll", type=float, default=0.01, help="(内部) 轮询间隔（秒）")
    args = parser.parse_args(argv)

    if args.worker:
        if not args.result or not args.cancel_ack:
            parser.error("--worker requires --result and --cancel-ack")
        return _worker_main(args)

    if not args.queue or not args.out:
        parser.error("--queue and --out are required outside worker mode")

    scheduler = Scheduler(
        queue_path=args.queue,
        out_root=args.out,
        workers=args.workers,
        runner_name=args.runner,
        grace_seconds=1.0,
    )
    scheduler.load()
    if not scheduler.entries and args.count > 0:
        for index in range(args.count):
            scheduler.submit(f"t{index:02d}", f"input-{index:02d}.yaml", {"seconds": args.hold, "index": index})
    results = scheduler.run()
    counts = results["counts"]
    print(
        "scheduler summary: "
        f"{counts['converged']} converged, {counts['failed']} failed, "
        f"{counts['cancelled']} cancelled, {counts['running']} interrupted, {results['total']} total"
    )
    if counts["failed"] or counts["running"]:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
