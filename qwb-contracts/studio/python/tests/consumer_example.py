"""K6 验收：这份消费者代码必须能被 `mypy --strict` 通过。

它只使用 `qwb_engine.pyi` 声明的接口（C-05 的验收就是“桩文件能撑起真实用法”）。
P-06 交付真扩展后，同一个文件可以直接改成真跑（去掉 `TYPE_CHECKING` 假设即可）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import qwb_engine as qwb


class CollectingReporter(qwb.Reporter):
    """最朴素的 Reporter 实现。"""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.cancelled = False
        self.pct = 0.0

    def progress(self, stage: str, pct: float) -> None:
        self.pct = pct
        if not stage:
            raise ValueError("stage must not be empty")

    def log(self, line: str) -> None:
        self.lines.append(line)

    def is_cancelled(self) -> bool:
        return self.cancelled


def build_h2_input() -> qwb.JobInput:
    molecule = qwb.Molecule(
        atoms=[
            qwb.Atom("H", (0.0, 0.0, -0.37)),
            qwb.Atom("H", (0.0, 0.0, 0.37)),
        ],
        charge=0,
        multiplicity=1,
    )
    options = qwb.TaskOptions()
    options.scf_max_iter = 200
    options.threads = 4
    options.ecp_name = None
    return qwb.JobInput(
        molecule=molecule,
        basis_name="sto-3g",
        method_name="svwn5",
        grid_level="normal",
        solvent="vacuum",
        jobtype="sp",
        options=options,
    )


def run_and_summarize(registry_path: str) -> tuple[float, int]:
    job = build_h2_input()
    job.validate()

    engine = qwb.Engine(registry_path=registry_path, features=["jobtype:sp"])
    reporter = CollectingReporter()
    result = engine.run(job, reporter)

    energy: float = result.energy_hartree
    if not result.scf_converged:
        raise RuntimeError("scf did not converge")

    gradient: npt.NDArray[np.float64] | None = result.gradient_hartree_per_bohr
    n_dof = 0 if gradient is None else int(np.asarray(gradient).size)

    rustc: str = result.env.rustc_version
    if not rustc:
        raise RuntimeError("env.json lacks rustc_version")
    result.env.validate()

    snapshot_items: list[tuple[str, str]] = result.data_snapshot.items()
    for path, sha in snapshot_items:
        if len(sha) != 64:
            raise ValueError(f"{path} has a malformed sha256")

    return energy, n_dof


def error_exit_codes() -> dict[str, int]:
    return {
        "invalid_input": qwb.InvalidInputError().exit_code,
        "not_converged": qwb.ScfNotConvergedError().exit_code,
        "data": qwb.DataError().exit_code,
        "unsupported": qwb.UnsupportedError().exit_code,
    }


def first_feature() -> str:
    return qwb.BUILTIN_REGISTRY_FEATURES[0]
