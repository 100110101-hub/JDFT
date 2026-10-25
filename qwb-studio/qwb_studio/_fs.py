"""小工具：Windows 上真正原子、且**容忍并发读**的文件替换。

`os.replace` 在 POSIX 上是原子的 rename，永远成功；在 Windows 上它是
`MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`，当目标文件正被别的进程打开时（没有
`FILE_SHARE_DELETE`）会抛 `PermissionError`(ERROR_SHARING_VIOLATION)。Python 的
`open()`/`Path.read_text()` 恰好不带 `FILE_SHARE_DELETE`，所以"另一个进程正在读
queue.json 或 .qwb"这种完全正常的情形会让原子写失败。

这不是理论问题：P-03 的 `kill -9` 恢复测试在全量测试 + 门禁并发跑机器时踩到过 ——
测试每 10 ms 读一次 `queue.json`，调度器恰好那时 `os.replace`，于是写失败、子进程崩掉。

所以这里统一用"短暂重试 + 最后再抛"的写法：POSIX 上等价于一次 `os.replace`。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

__all__ = ["atomic_replace", "read_text_retry"]

#: 总重试预算 ≈ 200 × 平均 12 ms ≈ 2.4 s：比任何一次正常写入都长得多，
#: 又足以让"读者马上关掉文件"的情形自愈。
_DEFAULT_ATTEMPTS = 200


def atomic_replace(tmp: str | Path, target: str | Path, attempts: int = _DEFAULT_ATTEMPTS) -> None:
    """把 `tmp` 替换成 `target`；遇到 Windows 的共享冲突就稍等重试。"""
    source = Path(tmp)
    destination = Path(target)
    delay = 0.001
    last: PermissionError | None = None
    for _ in range(max(1, attempts)):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:  # Windows: 目标正被读取
            last = exc
            time.sleep(delay)
            delay = min(delay * 1.5, 0.05)
    assert last is not None
    raise last


def read_text_retry(
    path: str | Path,
    attempts: int = _DEFAULT_ATTEMPTS,
    encoding: str = "utf-8",
) -> str:
    """读一个**会被原子替换**的文本文件；遇到 Windows 共享冲突就稍等重试。

    这不是多余的：写方 `os.replace` 期间，读者 `open()` 同样可能拿到
    `PermissionError`（ERROR_SHARING_VIOLATION）。文件不存在仍然是 `FileNotFoundError`，
    不会被吞掉。
    """
    target = Path(path)
    delay = 0.001
    last: PermissionError | None = None
    for _ in range(max(1, attempts)):
        try:
            return target.read_text(encoding=encoding)
        except PermissionError as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 1.5, 0.05)
    assert last is not None
    raise last
