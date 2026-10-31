"""PR-07 的 Python 一半：对比表的值必须与「单看一个结果目录」一致（1e-10）。

判据（`project-phase2.md` §6.9 PR-07）：

1. **值一致**：表里每一行的 `energy_hartree` 与它自己 `result.json` 的差 ≤ 1e-10；
   CSV 里写的是同一批数字（能量 10 位小数、ΔE 8 位小数），读回来与表内值一致；
2. **ΔE 自洽**：`delta_e_kcal_per_mol == (E - E_ref) * 627.5094740634593`；
3. **可被 `pandas` 读**：`pandas.read_csv` 得到 7 列、行数 = 任务数、数值列真的是数字；
   同样的文件用 stdlib `csv` 也读得出来（`pandas` 只是**测试工具**：不在
   `pyproject.toml` 里、不在产品依赖里，缺了就 `skip`，gate 不会因此变红）；
4. **列序冻结**：`COMPARISON_COLUMNS` 与 Rust 侧 `qwb compare` 的表头逐字相同 ——
   这条字面量在两边各钉一次，任何一侧漂移都会红。

另外覆盖"旧结果目录"（没有 `uv_vis_spectrum` 块）：谱峰列为空，而不是整张表失败。
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import pytest

from qwb_studio.comparison import (
    COMPARISON_COLUMNS,
    HARTREE_TO_KCAL_PER_MOL,
    ComparisonError,
    build_comparison,
    comparison_csv,
    read_task_summary,
    write_comparison_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_RESULT = REPO_ROOT / "qwb-core" / "examples" / "results" / "h2o_tddft"


def _spectrum_block(centres: tuple[tuple[float, float], ...]) -> dict:
    """两三条高斯叠出来的采样点（与 Rust 侧展宽同形状：等步长、成对浮点）。"""
    hwhm, step = 0.4, 0.05
    points = []
    for index in range(-40, 141):
        energy = 5.0 + index * step
        intensity = sum(
            strength * math.exp(-math.log(2.0) * ((energy - centre) / hwhm) ** 2)
            for centre, strength in centres
        )
        points.append([round(energy, 6), intensity])
    return {"gaussian_hwhm_ev": hwhm, "grid_step_ev": step, "points": points}


def _task(
    root: Path,
    name: str,
    energy: float,
    *,
    method: str = "b3lyp",
    basis: str = "def2-svp",
    converged: bool = True,
    spectrum: dict | None = None,
) -> Path:
    task = root / name
    (task / "artifacts").mkdir(parents=True)
    payload: dict = {"energy_hartree": energy, "scf_converged": converged}
    if spectrum is not None:
        payload["uv_vis_spectrum"] = spectrum
    (task / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (task / "input.yaml").write_text(
        f"method_name: {method}\nbasis_name: {basis}\ngrid_level: fine\n",
        encoding="utf-8",
    )
    return task


@pytest.fixture()
def three_tasks(tmp_path: Path) -> list[Path]:
    # 能量刻意选成「不整齐」的十进制，1e-10 判据才有意义（整齐的数测不出舍入）。
    return [
        _task(tmp_path, "water_hf", -76.0266326514, method="hf", spectrum=_spectrum_block(((7.0, 0.4), (9.0, 0.9)))),
        _task(tmp_path, "water_b3lyp", -76.4182371723, method="b3lyp"),
        _task(tmp_path, "water_svwn5", -75.8736120931, method="svwn5", converged=False),
    ]


def test_values_match_reading_one_result_dir(three_tasks: list[Path]) -> None:
    rows = build_comparison(three_tasks)
    assert len(rows) == 3
    reference = rows[0].energy_hartree
    for row, task_dir in zip(rows, three_tasks):
        payload = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
        assert row.task == task_dir.name
        assert abs(row.energy_hartree - payload["energy_hartree"]) <= 1e-10, row
        assert row.converged is payload["scf_converged"]
        assert abs(
            row.delta_e_kcal_per_mol
            - (payload["energy_hartree"] - reference) * HARTREE_TO_KCAL_PER_MOL
        ) <= 1e-10
    # 参考任务的 ΔE 必须是精确的 0（不是 1e-17 那种"看起来像 0"）。
    assert rows[0].delta_e_kcal_per_mol == 0.0
    # 方法/基组来自 input.yaml。
    assert [row.method for row in rows] == ["hf", "b3lyp", "svwn5"]
    assert {row.basis for row in rows} == {"def2-svp"}


def test_peak_energies_come_from_the_sampled_curve(three_tasks: list[Path]) -> None:
    rows = build_comparison(three_tasks)
    peaks = rows[0].spectrum_peaks_ev
    assert len(peaks) == 2, peaks
    assert abs(peaks[0] - 7.0) <= 0.5 * 0.05
    assert abs(peaks[1] - 9.0) <= 0.5 * 0.05
    # 没有 `uv_vis_spectrum` 块（旧结果目录）：谱峰为空，行还在。
    assert rows[1].spectrum_peaks_ev == ()
    assert rows[1].energy_hartree == pytest.approx(-76.4182371723, abs=1e-12)


def test_csv_columns_are_frozen() -> None:
    # 与 Rust `qwb-cli/src/main.rs` 的 `COMPARISON_COLUMNS` 同一串字面量；
    # Rust 侧 `tests/t17_compare.rs` 里也钉着这一串。
    assert COMPARISON_COLUMNS == (
        "task,method,basis,energy_hartree,delta_e_kcal_per_mol,converged,spectrum_peaks_ev"
    )


def test_csv_round_trips_through_pandas(three_tasks: list[Path], tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas", reason="CSV 可读性判据需要 pandas（测试工具，非产品依赖）")
    rows = build_comparison(three_tasks)
    path = write_comparison_csv(tmp_path / "comparison.csv", rows)

    raw = path.read_bytes()
    assert b"\r" not in raw, "CSV 必须是 LF，不能有 CR（Windows 换行翻译的尾巴）"

    frame = pd.read_csv(path)
    assert list(frame.columns) == COMPARISON_COLUMNS.split(",")
    assert len(frame) == len(rows)
    # 数值列真的被读成数字，且与表内值一致（pandas 的默认 float 精度足够 1e-10）。
    for index, row in enumerate(rows):
        assert frame.loc[index, "task"] == row.task
        assert abs(float(frame.loc[index, "energy_hartree"]) - row.energy_hartree) <= 1e-10
        assert (
            abs(float(frame.loc[index, "delta_e_kcal_per_mol"]) - row.delta_e_kcal_per_mol)
            <= 1e-8
        )
        assert bool(frame.loc[index, "converged"]) is row.converged
    assert frame["energy_hartree"].dtype.kind == "f"
    assert frame["delta_e_kcal_per_mol"].dtype.kind == "f"


def test_csv_round_trips_through_the_stdlib(three_tasks: list[Path]) -> None:
    # `pandas` 不在产品依赖里，也不在 gate 声明的工具链里 —— 所以「最基本的可读性」
    # 必须用 stdlib 也测一遍：这条永远跑，不 skip。
    rows = build_comparison(three_tasks)
    text = comparison_csv(rows)
    assert text.endswith("\n")
    reader = csv.DictReader(io.StringIO(text))
    assert reader.fieldnames == COMPARISON_COLUMNS.split(",")
    records = list(reader)
    assert len(records) == len(rows)
    for record, row in zip(records, rows):
        assert record["task"] == row.task
        assert float(record["energy_hartree"]) == pytest.approx(row.energy_hartree, abs=1e-10)
        assert record["converged"] in ("true", "false")
        assert record["converged"] == ("true" if row.converged else "false")
    # 谱峰列用 `;` 分隔（逗号已经是列分隔符）。
    assert records[0]["spectrum_peaks_ev"].count(";") == len(rows[0].spectrum_peaks_ev) - 1
    assert records[1]["spectrum_peaks_ev"] == ""


def test_a_real_result_dir_still_reads() -> None:
    """仓库里提交的真实结果目录（不是合成夹具）也必须能进对比表。"""
    if not (REAL_RESULT / "result.json").is_file():
        pytest.skip("examples/results/h2o_tddft 不在仓库里")
    summary = read_task_summary(REAL_RESULT)
    payload = json.loads((REAL_RESULT / "result.json").read_text(encoding="utf-8"))
    assert summary.task == "h2o_tddft"
    assert abs(summary.energy_hartree - payload["energy_hartree"]) <= 1e-10
    assert summary.method and summary.method != "?"
    assert summary.basis and summary.basis != "?"


def test_errors_are_input_errors(tmp_path: Path) -> None:
    with pytest.raises(ComparisonError, match="at least two"):
        build_comparison([_task(tmp_path, "only", -1.0)])
    missing = tmp_path / "nothing_here"
    missing.mkdir()
    with pytest.raises(ComparisonError, match="result.json"):
        read_task_summary(missing)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ComparisonError, match="JSON"):
        read_task_summary(broken)
    no_energy = tmp_path / "no_energy"
    no_energy.mkdir()
    (no_energy / "result.json").write_text('{"scf_converged": true}', encoding="utf-8")
    with pytest.raises(ComparisonError, match="energy_hartree"):
        read_task_summary(no_energy)
