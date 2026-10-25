"""P-10：VTK 3D 视图 —— 分子、轨道等值面、ESP（S-06 电荷着色）、振动帧。

数据来源全部是**契约里的产物**，不重算电子结构：

| 元素 | 来源 |
| --- | --- |
| 原子/键 | 结果目录的 `input.yaml`（K4 输入） |
| ESP 着色 | `artifacts/populations.json`（S-06 的 Mulliken/Löwdin 布居） |
| 轨道等值面 | `artifacts/homo.cube` / `lumo.cube`（Gaussian cube，引擎写出） |
| 振动帧 | 调用方给的**位移向量**（笛卡尔，Bohr）；freq 作业的模式来自 Hessian |

渲染走 VTK 的 offscreen 路径（`SetOffScreenRendering(1)` + `vtkWindowToImageFilter`），
所以 gate 里不需要显示器，也能把像素读回来做断言。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..resultdir import ARTIFACTS_DIR, ResultDirError, open_result_dir
from ..yaml_min import loads as yaml_loads

__all__ = ["Atom3D", "Molecule3D", "SceneError", "read_cube"]

#: 元素 → (颜色, 半径 Å)。颜色取常见的 CPK 近似。
ELEMENT_STYLE: dict[str, tuple[tuple[float, float, float], float]] = {
    "H": ((0.95, 0.95, 0.95), 0.31),
    "C": ((0.28, 0.28, 0.28), 0.76),
    "N": ((0.19, 0.31, 0.97), 0.71),
    "O": ((0.90, 0.19, 0.19), 0.66),
    "F": ((0.19, 0.85, 0.31), 0.57),
    "Cu": ((0.78, 0.50, 0.20), 1.32),
    "Zn": ((0.49, 0.50, 0.69), 1.22),
}

BOHR_PER_ANGSTROM = 1.0 / 0.529_177_210_903


class SceneError(Exception):
    """结果目录里缺少 3D 视图需要的东西。"""


@dataclass(frozen=True)
class Atom3D:
    """一个原子（Å 坐标 + 可选的部分电荷）。"""

    symbol: str
    position_angstrom: tuple[float, float, float]
    charge: float | None = None


@dataclass(frozen=True)
class Molecule3D:
    """一个可渲染的分子。"""

    atoms: tuple[Atom3D, ...]
    bonds: tuple[tuple[int, int], ...]
    source: Path

    def displaced(self, displacement_bohr: tuple[tuple[float, float, float], ...], scale: float) -> "Molecule3D":
        """沿一条振动模的位移向量移动原子（`displacement_bohr` 单位 Bohr）。"""
        if len(displacement_bohr) != len(self.atoms):
            raise SceneError(
                f"位移向量长度 {len(displacement_bohr)} 与原子数 {len(self.atoms)} 不一致"
            )
        moved = []
        for atom, delta in zip(self.atoms, displacement_bohr):
            angstrom = [delta[axis] * scale / BOHR_PER_ANGSTROM for axis in range(3)]
            moved.append(
                Atom3D(
                    symbol=atom.symbol,
                    position_angstrom=(
                        atom.position_angstrom[0] + angstrom[0],
                        atom.position_angstrom[1] + angstrom[1],
                        atom.position_angstrom[2] + angstrom[2],
                    ),
                    charge=atom.charge,
                )
            )
        return Molecule3D(atoms=tuple(moved), bonds=self.bonds, source=self.source)


def read_xyz(path: str | Path) -> list[Atom3D]:
    """读 XYZ（Å）。`opt` 作业的 `artifacts/optimized.xyz` 就是它。"""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneError(f"读不了 {source}: {exc}") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise SceneError(f"{source} 不像 XYZ（少于 2 行）")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise SceneError(f"{source} 第一行不是原子数：{lines[0]!r}") from exc
    atoms: list[Atom3D] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise SceneError(f"{source} 的原子行格式不对：{line!r}")
        atoms.append(
            Atom3D(
                symbol=fields[0],
                position_angstrom=(float(fields[1]), float(fields[2]), float(fields[3])),
            )
        )
    if len(atoms) != count:
        raise SceneError(f"{source} 声明 {count} 个原子，实际读到 {len(atoms)} 个")
    return atoms


def load_result_scene(result_dir: str | Path) -> Molecule3D:
    """从 K3 结果目录读分子 + S-06 电荷（ESP 着色要用）。

    `opt` 作业的几何在 `artifacts/optimized.xyz`（K3 允许的自由形态产物）：**优先**用它。
    `input.yaml` 里是**初值**几何 —— 画初值就等于把"优化过的分子"画错。
    """
    root = Path(result_dir)
    try:
        directory = open_result_dir(root)  # 缺 env.json 等会被这里挡住
    except ResultDirError as exc:
        raise SceneError(str(exc)) from exc
    charges: list[float] | None = None
    populations = root / ARTIFACTS_DIR / "populations.json"
    if populations.is_file():
        payload = json.loads(populations.read_text(encoding="utf-8"))
        charges = payload.get("lowdin_charges") or payload.get("mulliken_charges")

    optimized = root / ARTIFACTS_DIR / "optimized.xyz"
    if optimized.is_file():
        atoms = read_xyz(optimized)
        if charges is not None and len(charges) == len(atoms):
            atoms = [
                Atom3D(atom.symbol, atom.position_angstrom, float(charges[index]))
                for index, atom in enumerate(atoms)
            ]
        return Molecule3D(atoms=tuple(atoms), bonds=guess_bonds(atoms), source=root)

    document = yaml_loads(directory.input_yaml)
    if not isinstance(document, dict) or "molecule" not in document:
        raise SceneError(f"{root}/input.yaml 里没有 molecule")
    raw_atoms = document["molecule"].get("atoms") or []
    atoms = []
    for index, raw in enumerate(raw_atoms):
        symbol = str(raw.get("symbol", "?"))
        position = raw.get("position_angstrom") or [0.0, 0.0, 0.0]
        if len(position) != 3:
            raise SceneError(f"atom {index} 的 position_angstrom 不是 3 个数")
        charge = charges[index] if charges is not None and index < len(charges) else None
        atoms.append(Atom3D(symbol, (float(position[0]), float(position[1]), float(position[2])), charge))
    return Molecule3D(atoms=tuple(atoms), bonds=guess_bonds(atoms), source=root)


def guess_bonds(atoms: list[Atom3D], tolerance: float = 1.25) -> tuple[tuple[int, int], ...]:
    """按共价半径和猜键（只用于可视化；不作任何化学判断）。"""
    radii = {symbol: style[1] for symbol, style in ELEMENT_STYLE.items()}
    bonds: list[tuple[int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            distance = sum((a.position_angstrom[k] - b.position_angstrom[k]) ** 2 for k in range(3)) ** 0.5
            limit = (radii.get(a.symbol, 0.8) + radii.get(b.symbol, 0.8)) * tolerance
            if 0.2 < distance <= limit:
                bonds.append((i, j))
    return tuple(bonds)


def read_cube(path: str | Path) -> tuple[list[Atom3D], tuple[int, int, int], tuple[float, float, float], tuple[float, float, float], Any]:
    """读 Gaussian cube → (原子, 维度, 原点, 步长, 数值数组 nz×ny×nx)。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneError(f"读不到 cube 文件：{path}（{exc}）") from exc
    lines = text.splitlines()
    if len(lines) < 6:
        raise SceneError(f"{path} 不像 cube 文件（行数 {len(lines)}）")
    header = lines[2].split()
    natoms = int(header[0])
    origin = tuple(float(value) for value in header[1:4])
    counts = []
    spacing = []
    for axis in range(3):
        fields = lines[3 + axis].split()
        counts.append(int(fields[0]))
        spacing.append(float(fields[axis + 1]))
    atoms: list[Atom3D] = []
    for index in range(natoms):
        fields = lines[6 + index].split()
        z = int(fields[0])
        symbol = _symbol_of(z)
        atoms.append(
            Atom3D(symbol, (float(fields[2]) / BOHR_PER_ANGSTROM, float(fields[3]) / BOHR_PER_ANGSTROM, float(fields[4]) / BOHR_PER_ANGSTROM))
        )
    import numpy as np

    values = np.array([float(line) for line in lines[6 + natoms :]], dtype=float)
    expected = counts[0] * counts[1] * counts[2]
    if values.size != expected:
        raise SceneError(f"{path}: 数据点 {values.size} != {expected}")
    return atoms, (counts[0], counts[1], counts[2]), origin, (spacing[0], spacing[1], spacing[2]), values


def _symbol_of(z: int) -> str:
    table = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 29: "Cu", 30: "Zn"}
    return table.get(z, f"Z{z}")


def charge_color(charge: float, scale: float = 0.6) -> tuple[float, float, float]:
    """ESP 着色：负电荷偏红、正电荷偏蓝、近零偏白（线性插值）。"""
    fraction = max(-1.0, min(1.0, charge / scale))
    if fraction < 0:
        # 负 → 红
        return (1.0, 1.0 + fraction, 1.0 + fraction)
    # 正 → 蓝
    return (1.0 - fraction, 1.0 - fraction, 1.0)


class MoleculeScene:
    """一个可离屏渲染的 VTK 场景（分子 + 可选轨道等值面 + 可选振动帧）。"""

    def __init__(self, molecule: Molecule3D, *, size: tuple[int, int] = (640, 480)) -> None:
        self.molecule = molecule
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.06, 0.07, 0.09)
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*size)
        self.atom_actors: list[vtk.vtkActor] = []
        self.orbital_actors: list[vtk.vtkActor] = []
        self._add_molecule()

    # -- 构建 ---------------------------------------------------------------

    def _add_molecule(self, *, esp: bool = False) -> None:
        for actor in self.atom_actors:
            self.renderer.RemoveActor(actor)
        self.atom_actors.clear()
        for atom in self.molecule.atoms:
            radius, rgb = self._style(atom, esp=esp)
            source = vtk.vtkSphereSource()
            source.SetCenter(*atom.position_angstrom)
            source.SetRadius(radius)
            source.SetThetaResolution(24)
            source.SetPhiResolution(24)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*rgb)
            actor.GetProperty().SetSpecular(0.35)
            actor.GetProperty().SetSpecularPower(20)
            self.renderer.AddActor(actor)
            self.atom_actors.append(actor)
        for i, j in self.molecule.bonds:
            a = self.molecule.atoms[i].position_angstrom
            b = self.molecule.atoms[j].position_angstrom
            line = vtk.vtkLineSource()
            line.SetPoint1(*a)
            line.SetPoint2(*b)
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(0.12)
            tube.SetNumberOfSides(12)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.75, 0.75, 0.78)
            self.renderer.AddActor(actor)
            self.atom_actors.append(actor)

    def _style(self, atom: Atom3D, *, esp: bool) -> tuple[float, tuple[float, float, float]]:
        color, radius = ELEMENT_STYLE.get(atom.symbol, ((0.6, 0.6, 0.6), 0.7))
        if esp and atom.charge is not None:
            return radius, charge_color(atom.charge)
        return radius, color

    def set_esp(self, enabled: bool = True) -> None:
        """切换 ESP 着色（S-06 的部分电荷）。"""
        self._add_molecule(esp=enabled)

    def add_orbital(self, cube_path: str | Path, isovalue: float = 0.02) -> int:
        """把 cube 里的轨道画成 ±isovalue 两张等值面；返回新增的 actor 数。"""
        _atoms, counts, origin, spacing, values = read_cube(cube_path)
        import numpy as np

        grid = vtk.vtkImageData()
        grid.SetDimensions(*counts)
        grid.SetOrigin(*origin)
        grid.SetSpacing(*spacing)
        # cube 的 x 变化最快；VTK 的结构化点也是 x 最快，顺序一致
        scalars = numpy_to_vtk(np.ascontiguousarray(values, dtype=float), deep=True)
        scalars.SetName("orbital")
        grid.GetPointData().SetScalars(scalars)

        added = 0
        for sign, color in ((1.0, (0.85, 0.25, 0.25)), (-1.0, (0.25, 0.45, 0.85))):
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(grid)
            contour.SetNumberOfContours(1)
            contour.SetValue(0, sign * isovalue)
            contour.ComputeNormalsOn()
            contour.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(0.55)
            self.renderer.AddActor(actor)
            self.orbital_actors.append(actor)
            added += 1
        return added

    def orbital_points(self) -> int:
        """当前轨道等值面的总点数（0 表示没画/画空）。"""
        total = 0
        for actor in self.orbital_actors:
            mapper = actor.GetMapper()
            polydata = mapper.GetInput() if mapper is not None else None
            if polydata is not None:
                total += polydata.GetNumberOfPoints()
        return total

    # -- 渲染 ---------------------------------------------------------------

    def render(self, path: str | Path, *, azimuth: float = 30.0, elevation: float = 20.0) -> Path:
        """离屏渲染成 PNG，返回路径。"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(azimuth)
        camera.Elevation(elevation)
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.SetScale(1)
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        return Path(path)

    def pixels(self) -> Any:
        """渲染并返回 RGB 数组（height×width×3，uint8），用于断言。"""
        self.window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.ReadFrontBufferOff()
        capture.Update()
        image = capture.GetOutput()
        width, height, _ = image.GetDimensions()
        scalars = image.GetPointData().GetScalars()
        array = vtk_to_numpy(scalars).reshape(height, width, -1)
        return array

    def close(self) -> None:
        self.window.Finalize()
