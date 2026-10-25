"""P-10 验收：VTK 3D 视图（分子 / 轨道 / ESP / 振动），全部离屏渲染后**读回像素**断言。

判据（project.md §7 P-10 / T8）：打开 H2O 结果可视化正确；ESP 用 S-06 的电荷着色。

做法：
1. 用 `qwb run`（真 CLI）产出一个**真结果目录** —— 它现在会写 `artifacts/populations.json`
   （S-06 布居）与 `artifacts/{homo,lumo}.cube`（轨道等值面输入）；
2. 用 VTK 离屏渲染：分子球棍、±isovalue 等值面、ESP 着色、振动帧；
3. 断言落在**像素**上：背景之外有足够多的彩色像素、ESP 打开后画面确实变了、
   等值面有非零点、振动帧与平衡帧不同。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("vtk")

from qwb_studio.gui.scene3d import (  # noqa: E402
    MoleculeScene,
    SceneError,
    charge_color,
    load_result_scene,
    read_cube,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"
DATA_ROOT = REPO_ROOT / "qwb-data"
WATER = QWB_CORE / "examples" / "h2o_sto3g_svwn5.yaml"


def _binary() -> Path:
    name = "qwb.exe" if sys.platform.startswith("win") else "qwb"
    candidate = QWB_CORE / "target" / "debug" / name
    if candidate.is_file():
        return candidate
    completed = subprocess.run(  # noqa: S603
        ["cargo", "build", "-q", "-p", "qwb-cli", "--bin", "qwb"],
        cwd=str(QWB_CORE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return candidate


@pytest.fixture(scope="module")
def water_result(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """跑一次真的 H2O/STO-3G/SVWN5，拿到带 artifacts 的结果目录。"""
    out = tmp_path_factory.mktemp("p10")
    completed = subprocess.run(  # noqa: S603
        [str(_binary()), "run", str(WATER), "--out", str(out)],
        cwd=str(QWB_CORE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root = out / "h2o_sto3g_svwn5"
    assert (root / "artifacts" / "populations.json").is_file(), "CLI 应该写 S-06 布居"
    assert (root / "artifacts" / "homo.cube").is_file(), "CLI 应该写 HOMO cube"
    return root


def _non_background(pixels) -> int:
    """数一数"不是背景色"的像素（背景是 0.06/0.07/0.09 → 约 15/18/23）。"""
    background = (15, 18, 23)
    red, green, blue = pixels[..., 0].astype(int), pixels[..., 1].astype(int), pixels[..., 2].astype(int)
    mask = (
        (abs(red - background[0]) > 20)
        | (abs(green - background[1]) > 20)
        | (abs(blue - background[2]) > 20)
    )
    return int(mask.sum())


def test_scene_reads_a_real_result_directory(water_result: Path) -> None:
    molecule = load_result_scene(water_result)
    assert len(molecule.atoms) == 3
    assert [atom.symbol for atom in molecule.atoms] == ["O", "H", "H"]
    # S-06 的 Löwdin 电荷：O 负、两个 H 正
    charges = [atom.charge for atom in molecule.atoms]
    assert charges[0] is not None and charges[0] < -0.1, charges
    assert all(charge is not None and charge > 0.05 for charge in charges[1:]), charges
    assert len(molecule.bonds) == 2, molecule.bonds


def test_molecule_renders_to_png(water_result: Path, tmp_path: Path) -> None:
    molecule = load_result_scene(water_result)
    scene = MoleculeScene(molecule, size=(480, 360))
    try:
        pixels = scene.pixels()
        assert pixels.shape[:2] == (360, 480)
        painted = _non_background(pixels)
        assert painted > 2000, f"只画出了 {painted} 个非背景像素，分子没渲染出来"
        path = scene.render(tmp_path / "water.png")
        assert path.is_file() and path.stat().st_size > 5_000
    finally:
        scene.close()


def test_esp_colouring_changes_the_picture(water_result: Path) -> None:
    molecule = load_result_scene(water_result)
    scene = MoleculeScene(molecule, size=(320, 240))
    try:
        before = scene.pixels().astype(int)
        scene.set_esp(True)
        after = scene.pixels().astype(int)
        difference = abs(before[..., :3] - after[..., :3]).sum()
        assert difference > 10_000, f"ESP 着色几乎没改变画面（差 {difference}）"
    finally:
        scene.close()


def test_esp_colour_mapping_is_monotone() -> None:
    negative = charge_color(-0.5)
    neutral = charge_color(0.0)
    positive = charge_color(0.5)
    assert negative[0] > negative[2], negative  # 负电荷偏红
    assert positive[2] > positive[0], positive  # 正电荷偏蓝
    assert neutral == (1.0, 1.0, 1.0), neutral


def test_orbital_isosurface_has_points(water_result: Path) -> None:
    molecule = load_result_scene(water_result)
    scene = MoleculeScene(molecule, size=(320, 240))
    try:
        _atoms, counts, _origin, _spacing, values = read_cube(water_result / "artifacts" / "homo.cube")
        assert counts[0] * counts[1] * counts[2] == values.size
        assert float(values.max()) > 0.02 and float(values.min()) < -0.02, "cube 里没有可画的幅值"
        added = scene.add_orbital(water_result / "artifacts" / "homo.cube", isovalue=0.02)
        assert added == 2, "应当有 ±isovalue 两张等值面"
        assert scene.orbital_points() > 500, scene.orbital_points()
        before = scene.pixels().astype(int)
        scene.render.__self__  # 保持接口检查
        assert _non_background(before) > 2000
    finally:
        scene.close()


def test_vibration_frame_moves_the_atoms(water_result: Path) -> None:
    molecule = load_result_scene(water_result)
    # 一条"对称伸缩"式的位移向量（Bohr）：O 不动、两个 H 沿 O–H 方向反向
    displacement = [
        (0.0, 0.0, 0.0),
        (0.0, -0.1, 0.1),
        (0.0, 0.1, 0.1),
    ]
    moved = molecule.displaced(displacement, scale=1.0)
    assert moved.atoms[0].position_angstrom == molecule.atoms[0].position_angstrom
    assert moved.atoms[1].position_angstrom != molecule.atoms[1].position_angstrom
    scene_rest = MoleculeScene(molecule, size=(320, 240))
    scene_moved = MoleculeScene(moved, size=(320, 240))
    try:
        difference = abs(scene_rest.pixels().astype(int) - scene_moved.pixels().astype(int)).sum()
        assert difference > 10_000, f"振动帧与平衡帧一样（差 {difference}）"
    finally:
        scene_rest.close()
        scene_moved.close()


def test_missing_artifacts_are_reported(water_result: Path, tmp_path: Path) -> None:
    # 复制结果目录但删掉 artifacts → ESP 没有电荷，`charge` 应该全是 None（不炸）
    import shutil

    copy = tmp_path / "no-artifacts"
    shutil.copytree(water_result, copy)
    shutil.rmtree(copy / "artifacts")
    (copy / "artifacts").mkdir()
    molecule = load_result_scene(copy)
    assert all(atom.charge is None for atom in molecule.atoms)
    # 而 cube 缺失时必须明确报错
    with pytest.raises(SceneError):
        read_cube(copy / "artifacts" / "homo.cube")
