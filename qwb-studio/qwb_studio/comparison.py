"""多任务结果对比表（PR-07）：能量 / ΔE / 谱峰，可导出 CSV。

判据（`project-phase2.md` §6.9 的 PR-07 第一条）：

* **对比值与单看一致（1e-10）**：这里的每个数字都直接来自各任务目录的
  `result.json`，不做重算、不做二次舍入；写进 CSV 时能量保留 10 位小数、
  ΔE 保留 8 位小数（与 Rust 侧 `qwb compare` 的 `comparison_csv` 同格式）。
* **导出 CSV 可被 `pandas` 读**：列序是冻结的
  [`COMPARISON_COLUMNS`]，每行 7 列、无 CR、数值列是裸数字 —— 所以
  `pandas.read_csv` 与 stdlib `csv` 都能直接读。测试里两条路都测
  （`pandas` 只是**测试工具**，不在产品依赖里，也不在 `pyproject.toml` 里）。

为什么 Python 侧也实现一遍（而不是读 Rust 写出的 CSV）：GUI 可以在没有
`qwb` 可执行文件的环境里打开一批结果目录；两边的**列序与小数位是同一个冻结
契约**，各自都有测试钉住同一串字面量，任何一侧漂移都会红。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "COMPARISON_COLUMNS",
    "HARTREE_TO_KCAL_PER_MOL",
    "ComparisonError",
    "ComparisonRow",
    "TaskSummary",
    "build_comparison",
    "comparison_csv",
    "read_task_summary",
    "write_comparison_csv",
]

#: Hartree → kcal/mol（与全仓库其它地方同一个常数）。
HARTREE_TO_KCAL_PER_MOL = 627.5094740634593

#: CSV 列序（**冻结**）：必须与 Rust `qwb_cli::COMPARISON_COLUMNS` 逐字相同。
COMPARISON_COLUMNS = (
    "task,method,basis,energy_hartree,delta_e_kcal_per_mol,converged,spectrum_peaks_ev"
)


class ComparisonError(Exception):
    """结果目录不能进对比表（缺文件 / 缺字段 / JSON 坏了）。"""


@dataclass(frozen=True)
class TaskSummary:
    """一个结果目录在对比表里需要的全部信息（只读、无派生值）。"""

    task: str
    method: str
    basis: str
    energy_hartree: float
    converged: bool
    spectrum_peaks_ev: tuple[float, ...] = ()


@dataclass(frozen=True)
class ComparisonRow:
    """对比表的一行：`TaskSummary` + 参考任务的能量差。"""

    task: str
    method: str
    basis: str
    energy_hartree: float
    delta_e_kcal_per_mol: float
    converged: bool
    spectrum_peaks_ev: tuple[float, ...]


def _local_maxima(points: list[tuple[float, float]]) -> tuple[float, ...]:
    """展宽谱的局部极大值（与 Rust `UvVisSpectrum::peak_energies_ev` 同义）。"""
    peaks: list[float] = []
    for index in range(1, len(points) - 1):
        _, left = points[index - 1]
        energy, middle = points[index]
        _, right = points[index + 1]
        if middle > left and middle >= right:
            peaks.append(energy)
    return tuple(peaks)


def read_task_summary(task_dir: Path | str) -> TaskSummary:
    """从一个结果目录读出对比表需要的字段。

    与 GUI 谱面板同样**宽松**：`result.json` 必须有 `energy_hartree`；
    `input.yaml`（方法/基组）与 `uv_vis_spectrum`（谱峰）读不到就退化，不抛错 ——
    对比表的价值在于「能一起看」，不该因为一份旧结果没有谱就整张表不出。
    """
    directory = Path(task_dir)
    result_path = directory / "result.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ComparisonError(f"cannot read {result_path}: 文件不存在") from exc
    except OSError as exc:
        raise ComparisonError(f"cannot read {result_path}: {exc}") from exc
    except ValueError as exc:
        raise ComparisonError(f"{result_path} 不是合法 JSON：{exc}") from exc
    energy = payload.get("energy_hartree")
    if not isinstance(energy, (int, float)):
        raise ComparisonError(f"{result_path}: 缺少 energy_hartree")

    points: list[tuple[float, float]] = []
    block = payload.get("uv_vis_spectrum")
    if isinstance(block, dict):
        for point in block.get("points") or []:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, IndexError, ValueError):
                continue

    method = "?"
    basis = "?"
    input_path = directory / "input.yaml"
    try:
        # 只为了两个标量字段：这里用最小解析（`yaml_min`）而不是整个模型校验，
        # 因为对比表要能读"手工改坏了一点的旧目录"。
        for line in input_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            value = value.strip().strip("'\"")
            if key == "method_name" and value:
                method = value
            elif key == "basis_name" and value:
                basis = value
    except (OSError, ValueError):
        pass

    return TaskSummary(
        task=directory.name or "?",
        method=method,
        basis=basis,
        energy_hartree=float(energy),
        converged=bool(payload.get("scf_converged", False)),
        spectrum_peaks_ev=_local_maxima(points),
    )


def build_comparison(task_dirs: list[Path | str] | tuple[Path | str, ...]) -> tuple[ComparisonRow, ...]:
    """把若干结果目录变成对比表的行；**第一行是参考任务**（ΔE = 0）。"""
    if len(task_dirs) < 2:
        raise ComparisonError("compare needs at least two result directories")
    summaries = [read_task_summary(directory) for directory in task_dirs]
    reference = summaries[0].energy_hartree
    return tuple(
        ComparisonRow(
            task=summary.task,
            method=summary.method,
            basis=summary.basis,
            energy_hartree=summary.energy_hartree,
            delta_e_kcal_per_mol=(summary.energy_hartree - reference)
            * HARTREE_TO_KCAL_PER_MOL,
            converged=summary.converged,
            spectrum_peaks_ev=summary.spectrum_peaks_ev,
        )
        for summary in summaries
    )


def comparison_csv(rows: tuple[ComparisonRow, ...] | list[ComparisonRow]) -> str:
    """对比表的 CSV 文本：表头 = [`COMPARISON_COLUMNS`]，LF 换行，无 CR。"""
    lines = [COMPARISON_COLUMNS]
    for row in rows:
        peaks = ";".join(f"{peak:.4f}" for peak in row.spectrum_peaks_ev)
        flag = "true" if row.converged else "false"
        lines.append(
            f"{row.task},{row.method},{row.basis},"
            f"{row.energy_hartree:.10f},{row.delta_e_kcal_per_mol:.8f},{flag},{peaks}"
        )
    return "\n".join(lines) + "\n"


def write_comparison_csv(path: Path | str, rows: tuple[ComparisonRow, ...] | list[ComparisonRow]) -> Path:
    """把对比表写成 CSV；返回实际写入的路径。"""
    target = Path(path)
    # 用 `write_bytes` 而不是 `write_text`：要的是**字节级**的 LF，不能让平台
    # 换行翻译偷偷把 CR 塞进去（Windows 上 `write_text` 默认会）。
    target.write_bytes(comparison_csv(rows).encode("utf-8"))
    return target
