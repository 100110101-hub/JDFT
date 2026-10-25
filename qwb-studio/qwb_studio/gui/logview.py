"""日志视图：大文件流式追加，内存有界。

`setMaximumBlockCount` 让 QPlainTextEdit 只保留最近 N 段；`feed` 每次只追加一小批，
由 `QTimer` 反复调度，保证事件循环每批之间都能处理别的输入事件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QPlainTextEdit


class LogView(QPlainTextEdit):
    """只读、按块追加、块数有上限的日志窗口。"""

    streamFinished = Signal(int)  # 已追加的总行数

    #: 保留的最大块数（≈ 行数）：100 MB 日志也只占这么多。
    DEFAULT_MAX_BLOCKS = 5000
    #: 每次事件循环追加的行数。
    DEFAULT_CHUNK = 500

    def __init__(self, parent=None, max_blocks: int = DEFAULT_MAX_BLOCKS, chunk: int = DEFAULT_CHUNK) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_blocks)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._chunk = max(1, chunk)
        self._lines: Iterator[str] | None = None
        self._chunks = 0
        self._total_lines = 0
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._pump)

    # -- 流式接口 -----------------------------------------------------------

    def feed(self, lines: Iterable[str]) -> None:
        """开始流式追加；返回时**还没**追加完（由事件循环推进）。"""
        self._lines = iter(lines)
        self._chunks = 0
        self._total_lines = 0
        if not self._timer.isActive():
            self._timer.start()

    def feed_file(self, path: str | Path, encoding: str = "utf-8") -> None:
        """流式读取一个日志文件（逐行，不在内存里整体载入）。"""

        def generator() -> Iterator[str]:
            with open(path, encoding=encoding, errors="replace") as handle:
                for line in handle:
                    yield line.rstrip("\n")

        self.feed(generator())

    @property
    def chunks_delivered(self) -> int:
        return self._chunks

    @property
    def total_lines(self) -> int:
        return self._total_lines

    def is_streaming(self) -> bool:
        return self._timer.isActive()

    # -- 内部 ---------------------------------------------------------------

    def _pump(self) -> None:
        if self._lines is None:
            self._timer.stop()
            return
        batch: list[str] = []
        try:
            for _ in range(self._chunk):
                batch.append(next(self._lines))
        except StopIteration:
            self._timer.stop()
            if batch:
                self.appendPlainText("\n".join(batch))
                self._total_lines += len(batch)
            self._lines = None
            self.streamFinished.emit(self._total_lines)
            return
        self.appendPlainText("\n".join(batch))
        self._total_lines += len(batch)
        self._chunks += 1
