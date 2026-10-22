"""qwb_engine — QuantumWorkbench 引擎 Python 绑定（**契约 K6，冻结**）。

实现：`pyo3` + `maturin`，ABI `abi3-py39`（R-12）。本文件是**类型桩**，P-06 交付
真正的二进制扩展；在 P-06 之前，`mypy --strict` 就是用这份桩来验收的。

硬约定（K6）：
- `Engine.run` 内部必须释放 GIL（`py.allow_threads`），否则 Python 端并发废掉。
- 矩阵零拷贝：初期全部拷贝（< 500×500 可接受），接口不变（R-18）。
- `Reporter` 回调必须由 Rust 侧**批量**推送，禁止逐行跨 GIL 回调。
"""

from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "Atom",
    "Molecule",
    "TaskOptions",
    "JobInput",
    "GridLevel",
    "JobType",
    "DataSnapshot",
    "EnvInfo",
    "JobResult",
    "Reporter",
    "EngineError",
    "DataError",
    "InvalidInputError",
    "ScfNotConvergedError",
    "UnsupportedError",
    "BUILTIN_REGISTRY_FEATURES",
    "Engine",
    "run_job",
]

#: 网格等级（与 `engine-api::GridLevel` 的 serde 拼写一致）。
GridLevel = Literal["coarse", "normal", "fine", "veryfine"]

#: 作业类型（小写形态；K2 的 `Sp`/`Opt`/`Freq` 拼写同样被接受）。
JobType = Literal["sp", "opt", "freq"]

#: 第一阶段契约声明的 feature 词汇表（`engine-api::SUPPORTED_FEATURES`）。
BUILTIN_REGISTRY_FEATURES: tuple[str, ...]

class Atom:
    """单个原子（坐标单位 Å）。"""

    symbol: str
    position_angstrom: tuple[float, float, float]
    def __init__(self, symbol: str, position_angstrom: Sequence[float]) -> None: ...
    def position_bohr(self) -> tuple[float, float, float]: ...

class Molecule:
    """分子：原子 + 总电荷 + 自旋多重度。"""

    atoms: list[Atom]
    charge: int
    multiplicity: int
    def __init__(
        self,
        atoms: Sequence[Atom],
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None: ...
    def n_atoms(self) -> int: ...
    def validate(self) -> None: ...

class TaskOptions:
    """数值控制参数（全部有默认值，未赋值的字段用默认）。"""

    scf_max_iter: int
    scf_energy_threshold_hartree: float
    scf_density_threshold: float
    diis_history: int
    threads: int
    max_opt_steps: int
    opt_gradient_threshold_hartree_per_bohr: float
    freq_step_angstrom: float
    ecp_name: str | None
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class JobInput:
    """一次作业的完整输入（= `input.yaml` 的形态）。"""

    molecule: Molecule
    basis_name: str
    method_name: str
    grid_level: GridLevel
    solvent: str
    jobtype: JobType
    options: TaskOptions
    def __init__(
        self,
        molecule: Molecule,
        basis_name: str,
        method_name: str,
        grid_level: GridLevel = "normal",
        solvent: str = "vacuum",
        jobtype: JobType = "sp",
        options: TaskOptions | None = None,
    ) -> None: ...
    def validate(self) -> None: ...

class DataSnapshot:
    """数据快照：相对路径 -> sha256（64 位小写十六进制）。"""

    def __init__(self) -> None: ...
    def insert(self, path: str, sha256: str) -> None: ...
    def get(self, path: str) -> str | None: ...
    def items(self) -> list[tuple[str, str]]: ...
    def __len__(self) -> int: ...
    def validate(self) -> None: ...

class EnvInfo:
    """复现环境（`env.json`；G5：缺它结果不是结果）。"""

    engine_version: str
    rustc_version: str
    cargo_lock_sha256: str
    data_snapshot: DataSnapshot
    threads: int
    os: str
    created_at_utc: str
    toolchain_channel: str | None
    crate_versions: dict[str, str]
    def validate(self) -> None: ...

class JobResult:
    """一次作业的结果（`result.json` 的形态；矩阵是 list-of-list，K4）。"""

    energy_hartree: float
    gradient_hartree_per_bohr: npt.NDArray[np.float64] | None
    hessian_hartree_per_bohr2: npt.NDArray[np.float64] | None
    dipole_debye: npt.NDArray[np.float64] | None
    mo_energies_hartree: npt.NDArray[np.float64] | None
    mo_coeffs: npt.NDArray[np.float64] | None
    scf_converged: bool
    data_snapshot: DataSnapshot
    log: str
    env: EnvInfo
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> JobResult: ...

class Reporter(Protocol):
    """引擎观测面。实现方必须线程安全。"""

    def progress(self, stage: str, pct: float) -> None: ...
    def log(self, line: str) -> None: ...
    def is_cancelled(self) -> bool: ...

class EngineError(Exception):
    """引擎错误基类；`exit_code` 与 K5 一致。"""

    exit_code: int

class InvalidInputError(EngineError):
    """输入语义非法（K5 退出码 1）。"""

class ScfNotConvergedError(EngineError):
    """SCF 未收敛（K5 退出码 2）。"""

    iters: int
    max_iters: int

class DataError(EngineError):
    """数据缺失/损坏/sha256 不符（K5 退出码 3）。"""

class UnsupportedError(EngineError):
    """未实现或超出第一阶段范围（K5 退出码 4）。"""

class Engine:
    """引擎门面。构造即校验数据根与 feature；数值执行在 `run`。"""

    def __init__(
        self,
        registry_path: str,
        *,
        features: Sequence[str] | None = None,
    ) -> None: ...
    @property
    def registry_root(self) -> str: ...
    def run(
        self,
        job_input: JobInput,
        reporter: Reporter | None = None,
    ) -> JobResult:
        """执行一次作业。**必须**在内部 `py.allow_threads` 运行。"""
        ...
    def request_cancel(self) -> None: ...

def run_job(job_input: JobInput, registry_path: str) -> JobResult:
    """一次性便捷入口（等价于 `Engine(registry_path).run(job_input)`）。"""
    ...
