"""QuantumWorkbench Studio：K3 契约的 pydantic v2 模型层 + K4 容器（P-01 / P-02）。

设计原则
========

* **契约优先**：`qwb_studio.models` 里的每个字段都与 `qwb-contracts/studio/*.schema.yaml`
  一一对应，测试会直接读那份 YAML 的文本做比对（不引入 PyYAML，见 project.md §2.4 的依赖白名单）。
* **状态机冻结**：`TaskState` 的 6 条合法转移与 `qwb-core/crates/qwb-io/src/task.rs`
  的 `TaskState::LEGAL` 完全一致；非法转移一律抛 `ValueError`（P-01 的验收要求）。
* **不做电子结构**：这一层只做项目/分子/任务/结果的建模与 IO，计算全部在 Rust 引擎里
  （project.md §2.2：Python 侧不许有第二套引擎）。
* **YAML 子集自持**：`qwb_studio.yaml_min` 提供 block 风格 YAML 的读写，因为依赖白名单
  （§2.4）里没有 PyYAML；它只覆盖模型层会写出的形态，越界时显式报错。
* **容器与 Rust 互通**：`qwb_studio.container.QwbContainer` 读写 K4 `.qwb`（`zipfile`），
  与 Rust 的 `qwb_io::container::QwbContainer` 双向互操作（P-02）。
"""

from .container import (
    DATA_SNAPSHOT_FILE,
    K4_SCHEMA_VERSION,
    META_FILE,
    ContainerError,
    ProjectMeta,
    QwbContainer,
    task_entry_path,
)
from .env import (
    EnvError,
    collect_env,
    read_env_json,
    write_env_json,
)
from .models import (
    LEGAL_TRANSITIONS,
    Atom,
    EnvInfo,
    IllegalTransitionError,
    Molecule,
    Project,
    Result,
    Task,
    TaskState,
)
from .resultdir import ResultDirError, ResultDirectory, open_result_dir
from .runner import RunOutcome, find_data_root, run_job_to_dir
from .yaml_min import YamlError
from .yaml_min import dumps as yaml_dumps
from .yaml_min import loads as yaml_loads

__all__ = [
    "DATA_SNAPSHOT_FILE",
    "K4_SCHEMA_VERSION",
    "LEGAL_TRANSITIONS",
    "META_FILE",
    "Atom",
    "ContainerError",
    "EnvError",
    "EnvInfo",
    "IllegalTransitionError",
    "Molecule",
    "Project",
    "ProjectMeta",
    "QwbContainer",
    "Result",
    "ResultDirError",
    "ResultDirectory",
    "RunOutcome",
    "Task",
    "TaskState",
    "YamlError",
    "collect_env",
    "find_data_root",
    "open_result_dir",
    "read_env_json",
    "run_job_to_dir",
    "task_entry_path",
    "write_env_json",
    "yaml_dumps",
    "yaml_loads",
]

__version__ = "0.1.0"
