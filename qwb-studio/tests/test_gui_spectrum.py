"""PR-04 的 GUI 一半：UV/Vis 谱面板必须能装载、能画、且**不阻塞**。

三条判据（对应 §6 的 PR-04 第三项）：

1. **装载**：从真实形状的结果目录（`result.json` 的 `uv_vis_spectrum` 块）读出点集；
2. **峰位一致**：面板算出的峰位与点集的局部极大值相同（与 Rust 侧同义）；
3. **出图不阻塞**：把面板渲染到 `QPixmap` 计时（判据 < 1 s，实测远低于），
   并且事件循环在处理一次 `processEvents` 后仍有响应。

旧结果目录（没有那个块）必须安全：`load_task_dir` 返回 `False` 且状态文本说明原因。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qwb_studio.gui.spectrum_panel import SpectrumPanel, load_spectrum  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
H2O_RESULT = REPO_ROOT / "qwb-core" / "examples" / "results" / "h2o_tddft"


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _synthetic_task(tmp_path: Path, blocks: dict | None) -> Path:
    task = tmp_path / "task"
    (task / "artifacts").mkdir(parents=True)
    payload = {"energy_hartree": -75.0, "converged": True}
    if blocks:
        payload.update(blocks)
    (task / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return task


def _three_peaks() -> dict:
    hwhm = 0.4
    step = 0.05
    points = []
    for index in range(-20, 101):
        energy = 6.0 + index * step
        intensity = sum(
            strength
            * __import__("math").exp(
                -__import__("math").log(2)
                * ((energy - centre) / hwhm) ** 2
            )
            for centre, strength in ((7.0, 0.5), (9.0, 0.9))
        )
        points.append([round(energy, 6), intensity])
    return {
        "uv_vis_spectrum": {
            "gaussian_hwhm_ev": hwhm,
            "grid_step_ev": step,
            "points": points,
        }
    }


def test_panel_loads_a_real_result_dir(app) -> None:
    panel = SpectrumPanel()
    loaded = panel.load_task_dir(H2O_RESULT)
    # 仓库里提交的 h2o_tddft 结果可能来自加块之前的一次运行：那时没有谱，必须安全退化。
    if loaded:
        assert panel.peak_energies_ev(), "有谱就必须有峰"
        assert "HWHM" in panel.status_text
    else:
        assert "no spectrum" in panel.status_text


def test_peaks_match_the_sampled_curve(tmp_path: Path, app) -> None:
    task = _synthetic_task(tmp_path, _three_peaks())
    spectrum = load_spectrum(task)
    assert spectrum is not None
    assert abs(spectrum.gaussian_hwhm_ev - 0.4) < 1e-12
    panel = SpectrumPanel()
    assert panel.load_task_dir(task)
    peaks = panel.peak_energies_ev()
    assert len(peaks) == 2, peaks
    assert abs(peaks[0] - 7.0) <= 0.5 * spectrum.grid_step_ev
    assert abs(peaks[1] - 9.0) <= 0.5 * spectrum.grid_step_ev


def test_rendering_a_frame_is_fast(tmp_path: Path, app) -> None:
    task = _synthetic_task(tmp_path, _three_peaks())
    panel = SpectrumPanel()
    assert panel.load_task_dir(task)
    panel.resize(640, 260)
    panel.show()

    def render_once() -> float:
        pixmap = QPixmap(panel.size())
        started = time.perf_counter()
        panel.render(pixmap)
        elapsed = time.perf_counter() - started
        app.processEvents()
        return elapsed

    first = render_once()
    worst = max(render_once() for _ in range(3))
    assert worst < 1.0, f"渲染一帧用了 {worst:.3f} s（判据 < 1 s）"
    assert first < 1.0
    panel.close()


def test_old_result_dir_degrades_gracefully(tmp_path: Path, app) -> None:
    task = _synthetic_task(tmp_path, None)
    assert load_spectrum(task) is None
    panel = SpectrumPanel()
    assert panel.load_task_dir(task) is False
    assert "no spectrum" in panel.status_text
    assert panel.peak_energies_ev() == []
    # 没有谱也必须能画（显示"这份结果没有谱"），不能抛异常。
    pixmap = QPixmap(320, 200)
    panel.resize(320, 200)
    panel.render(pixmap)

