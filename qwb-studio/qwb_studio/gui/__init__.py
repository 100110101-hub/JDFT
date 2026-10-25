"""P-09：GUI 骨架（PySide6）——项目树 / Task Panel / Job Monitor / 日志 / 结果面板。

设计要点
========

* **UI 不碰计算**：任务的执行交给 P-03 的 `Scheduler`（子进程 worker），GUI 只读队列状态、
  发取消请求。这样"不阻塞 UI"是结构性的，不是靠到处 `processEvents()` 硬撑。
* **日志流式追加**：`LogView` 用 `setMaximumBlockCount` 限制内存，并且按块从文件读取、
  每次追加少量行（由 `QTimer` 驱动），所以 100 MB 日志也不会把事件循环占死。
* **headless 可测**：测试用 `QT_QPA_PLATFORM=offscreen` 跑，断言的是"事件循环在任务运行
  期间仍在转"（QTimer 计数），不是"看起来不卡"。
"""

from .logview import LogView
from .mainwindow import MainWindow
from .model import GuiProject
from .runner import SchedulerBridge

__all__ = ["GuiProject", "LogView", "MainWindow", "SchedulerBridge"]
