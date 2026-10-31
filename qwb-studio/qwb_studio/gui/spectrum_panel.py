"""UV/Vis 谱面板（PR-04）：把结果目录里的展宽曲线画出来，不阻塞事件循环。

为什么自己画（不用 matplotlib）：产品的第三方依赖集是刻意保持最小的
（`project.md` §2.4：PySide6 / VTK / pydantic / pytest / hypothesis / mypy），
一条 `QPainter` 折线就够用，没必要为了画一条曲线引入绘图库。

判据（与 `docs/status-v2.yaml` 的 PR-04 一致）：

* 读 `result.json` 的 `uv_vis_spectrum` 块（K2 加性扩展），峰位从点集里算；
* 画一帧要快（不阻塞）：测试里直接对 `QPixmap` 渲染并计时；
* 旧结果目录（没有那个块）不能炸 —— 面板要能明确显示"这份结果没有谱"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


@dataclass(frozen=True)
class Spectrum:
    """一份展宽谱：`(能量 eV, 强度)` 采样点 + 实际使用的展宽参数。"""

    gaussian_hwhm_ev: float
    grid_step_ev: float
    points: tuple[tuple[float, float], ...]

    @property
    def peak_energies_ev(self) -> list[float]:
        """局部极大值的能量，按能量递增（与 Rust 侧 `UvVisSpectrum::peak_energies_ev` 同义）。"""
        peaks: list[float] = []
        for index in range(1, len(self.points) - 1):
            _, left = self.points[index - 1]
            energy, middle = self.points[index]
            _, right = self.points[index + 1]
            if middle > left and middle >= right:
                peaks.append(energy)
        return peaks

    @property
    def max_intensity(self) -> float:
        return max((intensity for _, intensity in self.points), default=0.0)


def load_spectrum(task_dir: Path) -> Spectrum | None:
    """从结果目录读谱；没有 `uv_vis_spectrum` 块时返回 `None`（旧结果目录）。"""
    result_path = Path(task_dir) / "result.json"
    if not result_path.is_file():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    block = payload.get("uv_vis_spectrum")
    if not isinstance(block, dict):
        return None
    points = tuple(
        (float(point[0]), float(point[1])) for point in block.get("points", [])
    )
    if len(points) < 3:
        return None
    return Spectrum(
        gaussian_hwhm_ev=float(block.get("gaussian_hwhm_ev", 0.0)),
        grid_step_ev=float(block.get("grid_step_ev", 0.0)),
        points=points,
    )


class SpectrumPanel(QWidget):
    """一条展宽曲线：坐标框 + 折线 + 峰位刻度。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._spectrum: Spectrum | None = None
        self._status = "no spectrum loaded"

    # ── 数据 ────────────────────────────────────────────────────────────────
    def load_task_dir(self, task_dir: Path) -> bool:
        """装载一个结果目录；返回是否真的读到谱。"""
        spectrum = load_spectrum(task_dir)
        self.set_spectrum(spectrum)
        return spectrum is not None

    def set_spectrum(self, spectrum: Spectrum | None) -> None:
        self._spectrum = spectrum
        self._status = (
            "no spectrum in this result (old result dir or no excited states)"
            if spectrum is None
            else f"{len(spectrum.points)} points, HWHM = {spectrum.gaussian_hwhm_ev:.2f} eV"
        )
        self.update()

    @property
    def spectrum(self) -> Spectrum | None:
        return self._spectrum

    @property
    def status_text(self) -> str:
        return self._status

    def peak_energies_ev(self) -> list[float]:
        return [] if self._spectrum is None else self._spectrum.peak_energies_ev

    # ── 绘制 ────────────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101418"))

        frame = QRectF(44.0, 12.0, max(10.0, self.width() - 56.0), max(10.0, self.height() - 44.0))
        painter.setPen(QPen(QColor("#4a5568"), 1.0))
        painter.drawRect(frame)

        painter.setPen(QPen(QColor("#8fa3b8"), 1.0))
        painter.drawText(8, int(frame.top()) + 8, "I")
        painter.drawText(int(frame.right()) - 60, int(frame.bottom()) + 18, "energy (eV)")

        spectrum = self._spectrum
        if spectrum is None or not spectrum.points:
            painter.setPen(QPen(QColor("#8fa3b8"), 1.0))
            painter.drawText(frame.toRect(), Qt.AlignmentFlag.AlignCenter, self._status)
            painter.end()
            return

        energies = [energy for energy, _ in spectrum.points]
        low, high = min(energies), max(energies)
        span = max(high - low, 1e-9)
        peak = max(spectrum.max_intensity, 1e-30)

        def to_point(energy: float, intensity: float) -> QPointF:
            x = frame.left() + (energy - low) / span * frame.width()
            y = frame.bottom() - intensity / peak * frame.height()
            return QPointF(x, y)

        polyline = [to_point(energy, intensity) for energy, intensity in spectrum.points]
        painter.setPen(QPen(QColor("#63b3ed"), 1.6))
        painter.drawPolyline(polyline)

        painter.setPen(QPen(QColor("#f6ad55"), 1.0))
        for energy in spectrum.peak_energies_ev:
            x = to_point(energy, 0.0).x()
            painter.drawLine(QPointF(x, frame.bottom()), QPointF(x, frame.top()))
            painter.drawText(QPointF(x - 14.0, frame.top() - 2.0), f"{energy:.2f}")

        painter.setPen(QPen(QColor("#8fa3b8"), 1.0))
        painter.drawText(
            QPointF(frame.left(), frame.bottom() + 18.0),
            f"{low:.2f} eV  →  {high:.2f} eV",
        )
        painter.end()

