"""Python 侧端到端执行（P-07）：一次作业 → 一个 K3 结果目录。

链路：`qwb_engine`（P-06 的 pyo3 绑定）算 → 用 P-01 的模型 + P-04 的 `env.json` 写出
K3 冻结布局（`input.yaml` / `result.json` / `env.json` / `log.txt` / `data_snapshot.json` / `artifacts/`）。

**不复刻任何数值逻辑**：能量、轨道、偶极、env、数据快照全部来自引擎返回值；这一层只负责
"把引擎给的东西写成契约规定的文件"，写完立刻用 `open_result_dir` 自检一遍（写不出来的结果
不算结果）。

`data_root` 的定位：显式参数 > `QWB_DATA_ROOT` > 逐级向上找 `qwb-data/`（与 Rust `qwb-cli`
的 `find_data_root` 同规则），找不到就报错而不是猜。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .env import write_env_json
from .models import EnvInfo, Result
from .resultdir import ARTIFACTS_DIR, ENV_FILE, ResultDirectory, open_result_dir
from .yaml_min import dumps as yaml_dumps

__all__ = [
    "RunOutcome",
    "find_data_root",
    "run_job_to_dir",
]

#: 结果目录里除 env.json 之外的必需文件（K3）。
INPUT_FILE = "input.yaml"
RESULT_FILE = "result.json"
LOG_FILE = "log.txt"
SNAPSHOT_FILE = "data_snapshot.json"


class RunnerError(Exception):
    """Python 侧执行失败（数据根找不到、引擎报错等）。"""


def find_data_root(explicit: str | Path | None = None, start: str | Path | None = None) -> Path:
    """定位 `qwb-data/`：显式 > `QWB_DATA_ROOT` > 向上搜索。"""
    if explicit is not None:
        root = Path(explicit)
        if not root.is_dir():
            raise RunnerError(f"data root is not a directory: {root}")
        return root

    from_env = os.environ.get("QWB_DATA_ROOT")
    if from_env:
        root = Path(from_env)
        if not root.is_dir():
            raise RunnerError(f"QWB_DATA_ROOT is not a directory: {root}")
        return root

    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "qwb-data").is_dir():
            return candidate / "qwb-data"
    raise RunnerError(
        "cannot locate qwb-data/: pass data_root=..., set QWB_DATA_ROOT, or run from inside the repo"
    )


@dataclass(frozen=True)
class RunOutcome:
    """一次 Python 侧执行的产物。"""

    root: Path
    directory: ResultDirectory

    @property
    def energy_hartree(self) -> float:
        return self.directory.result.energy_hartree


def _input_yaml(job: Any) -> str:
    """把 `qwb_engine.JobInput` 写成 K4 的 `input.yaml` 文本。

    字段与 `qwb-contracts/studio/qwb-format.md` §1 一致；用 block 风格输出
    （`yaml_min` 只做这一种子集），Rust 的 `serde_yaml` 读得回来 —— 这一点由
    `tests/test_runner.py` 里"Python 写、Rust `qwb replay` 重算"的用例实证。
    """
    molecule = job.molecule
    atoms = []
    for atom in molecule.atoms:
        x, y, z = atom.position_angstrom
        atoms.append({"symbol": atom.symbol, "position_angstrom": [x, y, z]})
    options = {
        "scf_max_iter": job.options.scf_max_iter,
        "scf_energy_threshold_hartree": job.options.scf_energy_threshold_hartree,
        "scf_density_threshold": job.options.scf_density_threshold,
        "diis_history": job.options.diis_history,
        "threads": job.options.threads,
        "max_opt_steps": job.options.max_opt_steps,
        "opt_gradient_threshold_hartree_per_bohr": job.options.opt_gradient_threshold_hartree_per_bohr,
        "freq_step_angstrom": job.options.freq_step_angstrom,
        "ecp_name": job.options.ecp_name,
    }
    document: Mapping[str, Any] = {
        "molecule": {
            "atoms": atoms,
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
        },
        "basis_name": job.basis_name,
        "method_name": job.method_name,
        "grid_level": job.grid_level,
        "solvent": job.solvent,
        "jobtype": job.jobtype,
        "options": options,
    }
    return yaml_dumps(dict(document))


def _env_from_engine(engine_env: Any) -> EnvInfo:
    """`qwb_engine.EnvInfo` → P-01 的 `EnvInfo`（逐字段，不猜）。"""
    payload = {
        "engine_version": engine_env.engine_version,
        "rustc_version": engine_env.rustc_version,
        "cargo_lock_sha256": engine_env.cargo_lock_sha256,
        "data_snapshot": dict(engine_env.data_snapshot.items()),
        "threads": engine_env.threads,
        "os": engine_env.os,
        "created_at_utc": engine_env.created_at_utc,
        "toolchain_channel": engine_env.toolchain_channel,
        "crate_versions": dict(engine_env.crate_versions),
    }
    env = EnvInfo.model_validate(payload)
    env.validate()
    return env


def run_job_to_dir(
    job: Any,
    out_root: str | Path,
    task_id: str,
    *,
    data_root: str | Path | None = None,
    reporter: Any = None,
) -> RunOutcome:
    """跑一次作业并写出 K3 结果目录，返回 :class:`RunOutcome`（已自检过）。"""
    import qwb_engine

    root = Path(out_root) / task_id
    root.mkdir(parents=True, exist_ok=True)
    (root / ARTIFACTS_DIR).mkdir(exist_ok=True)

    engine = qwb_engine.Engine(str(find_data_root(data_root)))
    outcome = engine.run(job, reporter) if reporter is not None else engine.run(job)

    env = _env_from_engine(outcome.env)
    result = Result.model_validate(
        {
            "energy_hartree": outcome.energy_hartree,
            "scf_converged": outcome.scf_converged,
            "data_snapshot": dict(outcome.data_snapshot.items()),
            "log": outcome.log,
            "env": env.model_dump(mode="json"),
            "gradient_hartree_per_bohr": _matrix(outcome.gradient_hartree_per_bohr),
            "hessian_hartree_per_bohr2": _matrix(outcome.hessian_hartree_per_bohr2),
            "dipole_debye": _vector(outcome.dipole_debye),
            "mo_energies_hartree": _vector(outcome.mo_energies_hartree),
            "mo_coeffs": _matrix(outcome.mo_coeffs),
        }
    )

    (root / INPUT_FILE).write_text(_input_yaml(job), encoding="utf-8")
    (root / RESULT_FILE).write_text(outcome.to_json(), encoding="utf-8")
    (root / LOG_FILE).write_text(outcome.log, encoding="utf-8")
    (root / SNAPSHOT_FILE).write_text(
        json.dumps(result.data_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_env_json(root / ENV_FILE, env)

    directory = open_result_dir(root)  # 写不出来的结果不算结果
    return RunOutcome(root=root, directory=directory)


def _matrix(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(item) for item in row] for row in value.tolist()]


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.tolist()]
