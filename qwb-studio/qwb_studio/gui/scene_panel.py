"""P-10 的 Qt 面板：把 `scene3d` 的离屏渲染结果显示成图片。

用「VTK 离屏渲染 → PNG → QLabel」而不是 `QVTKRenderWindowInteractor`：
后者要求 PySide6 + VTK 的 Qt 绑定对齐（版本稍不合就 import 失败），
而我们的验收判据是"3D 结果可视化正确"，用像素说话就够了。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .scene3d import SceneError, load_result_scene, MoleculeScene

__all__ = ["Scene3DPanel"]


class Scene3DPanel(QWidget):
    """分子 / ESP / 轨道 / 振动帧的 3D 面板。"""

    rendered = Signal(str, int)  # 图片路径, 非背景像素数

    def __init__(self, parent: QWidget | None = None, size: tuple[int, int] = (520, 400)) -> None:
        super().__init__(parent)
        self.size_hint = size
        self.image_label = QLabel("（还没有结果）")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(*size)
        self.image_label.setStyleSheet("background: #101216; color: #9aa4b2;")

        self.esp_button = QPushButton("ESP 着色")
        self.orbital_button = QPushButton("轨道等值面")
        self.vibration_button = QPushButton("振动帧 ±")
        for button in (self.esp_button, self.orbital_button, self.vibration_button):
            button.setEnabled(False)
        self.esp_button.clicked.connect(self.toggle_esp)
        self.orbital_button.clicked.connect(self.toggle_orbital)
        self.vibration_button.clicked.connect(self.step_vibration)

        buttons = QHBoxLayout()
        for button in (self.esp_button, self.orbital_button, self.vibration_button):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addLayout(buttons)
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        self._result_dir: Path | None = None
        self._scene: MoleculeScene | None = None
        self._esp = False
        self._orbital = False
        self._vibration_step = 0
        self._last_image = Path(tempfile.gettempdir()) / "qwb-scene3d.png"

    # -- 对外 ---------------------------------------------------------------

    def show_result(self, result_dir: str | Path) -> bool:
        """打开一个 K3 结果目录并渲染分子。返回是否成功。"""
        root = Path(result_dir)
        self._result_dir = root
        try:
            molecule = load_result_scene(root)
            if self._scene is not None:
                self._scene.close()
            self._scene = MoleculeScene(molecule, size=self.size_hint)
        except SceneError as exc:
            self._reset(f"无法渲染：{exc}")
            return False
        self._esp = False
        self._orbital = False
        self._vibration_step = 0
        for button in (self.esp_button, self.orbital_button, self.vibration_button):
            button.setEnabled(True)
        self.vibration_button.setEnabled(False)
        has_cube = (root / "artifacts" / "homo.cube").is_file()
        self.orbital_button.setEnabled(has_cube)
        self._refresh()
        return True

    def toggle_esp(self) -> None:
        if self._scene is None:
            return
        self._esp = not self._esp
        self._scene.set_esp(self._esp)
        self._refresh()

    def toggle_orbital(self) -> None:
        if self._scene is None or self._result_dir is None:
            return
        cube = self._result_dir / "artifacts" / "homo.cube"
        if not cube.is_file():
            return
        if self._orbital:
            # 关掉等值面：重建分子场景（最省事、也不会漏删 actor）
            self._orbital = False
            self._scene.close()
            self._scene = MoleculeScene(load_result_scene(self._result_dir), size=self.size_hint)
            self._scene.set_esp(self._esp)
        else:
            self._scene.add_orbital(cube, isovalue=0.02)
            self._orbital = True
        self._refresh()

    def step_vibration(self) -> None:
        """在 ±δ 两个振动帧之间切换（位移向量取自当前分子的一个"拉伸"模式）。"""
        if self._scene is None or self._result_dir is None:
            return
        molecule = load_result_scene(self._result_dir)
        # 一条演示用的对称伸缩：每个原子沿"远离质心"的方向位移
        center = [
            sum(atom.position_angstrom[axis] for atom in molecule.atoms) / len(molecule.atoms)
            for axis in range(3)
        ]
        displacement = []
        for atom in molecule.atoms:
            vector = [atom.position_angstrom[axis] - center[axis] for axis in range(3)]
            length = sum(value * value for value in vector) ** 0.5 or 1.0
            displacement.append(tuple(0.15 * value / length for value in vector))
        self._vibration_step = 0 if self._vibration_step else 1
        scale = 1.0 if self._vibration_step else -1.0
        moved = molecule.displaced(displacement, scale=scale)
        self._scene.close()
        self._scene = MoleculeScene(moved, size=self.size_hint)
        self._scene.set_esp(self._esp)
        if self._orbital:
            self._scene.add_orbital(self._result_dir / "artifacts" / "homo.cube", isovalue=0.02)
        self._refresh()

    def pixmap(self) -> QPixmap | None:
        return self.image_label.pixmap()

    def scene_atoms(self):
        """当前场景里的原子（测试与"3D 画的是不是优化后的几何"用）。"""
        return self._scene.molecule.atoms if self._scene is not None else ()

    # -- 内部 ---------------------------------------------------------------

    def _refresh(self) -> None:
        if self._scene is None:
            return
        pixels = self._scene.pixels()
        painted = _non_background(pixels)
        path = self._scene.render(self._last_image)
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.info_label.setText(
            f"{self._result_dir.name}：ESP {'开' if self._esp else '关'}，"
            f"轨道 {'开' if self._orbital else '关'}，振动帧 {self._vibration_step}（非背景像素 {painted}）"
        )
        self.rendered.emit(str(path), painted)

    def _reset(self, message: str) -> None:
        if self._scene is not None:
            self._scene.close()
            self._scene = None
        self.image_label.setText(message)
        self.image_label.setPixmap(QPixmap())
        for button in (self.esp_button, self.orbital_button, self.vibration_button):
            button.setEnabled(False)
        self.info_label.setText("")


def _non_background(pixels) -> int:
    background = (15, 18, 23)
    red, green, blue = (pixels[..., index].astype(int) for index in range(3))
    mask = (
        (abs(red - background[0]) > 20)
        | (abs(green - background[1]) > 20)
        | (abs(blue - background[2]) > 20)
    )
    return int(mask.sum())
