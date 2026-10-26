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

为什么这里的导出是**惰性**的（PEP 562）
=====================================

数据取回（`qwb_studio.datafetch` / `qwb_studio.bse_data`）只用标准库，它必须能在
**只装了系统 Python 的机器**上跑 —— 用户可能只想下载几个基组，不该为此装 pydantic/numpy。
但 `import qwb_studio.datafetch` 会先执行这个 `__init__`，所以 `__init__` 里不能有
"eager import 模型层"这种要求。

于是公开 API 改成按需解析：`from qwb_studio import Result` 与 `qwb_studio.Result` 都照常
工作（第一次访问时才 import `models`），缺依赖时得到的是**真实**的 `ImportError`
（不是被吞掉的降级），而 `import qwb_studio.datafetch` 不会连带把 pydantic 拖进来。
回归测试钉住这一点：`qwb-studio/tests/test_datafetch.py::test_datafetch_import_is_stdlib_only`。
"""

from __future__ import annotations

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

#: 导出名 → (所在子模块, 子模块里的属性名)。除 `yaml_dumps`/`yaml_loads` 两个别名外，
#: 属性名与导出名相同（`yaml_min` 里叫 `dumps`/`loads`，包层面加了 `yaml_` 前缀）。
_LAZY: dict[str, tuple[str, str]] = {
    "DATA_SNAPSHOT_FILE": ("container", "DATA_SNAPSHOT_FILE"),
    "K4_SCHEMA_VERSION": ("container", "K4_SCHEMA_VERSION"),
    "META_FILE": ("container", "META_FILE"),
    "ContainerError": ("container", "ContainerError"),
    "ProjectMeta": ("container", "ProjectMeta"),
    "QwbContainer": ("container", "QwbContainer"),
    "task_entry_path": ("container", "task_entry_path"),
    "EnvError": ("env", "EnvError"),
    "EnvInfo": ("models", "EnvInfo"),
    "collect_env": ("env", "collect_env"),
    "read_env_json": ("env", "read_env_json"),
    "write_env_json": ("env", "write_env_json"),
    "LEGAL_TRANSITIONS": ("models", "LEGAL_TRANSITIONS"),
    "Atom": ("models", "Atom"),
    "IllegalTransitionError": ("models", "IllegalTransitionError"),
    "Molecule": ("models", "Molecule"),
    "Project": ("models", "Project"),
    "Result": ("models", "Result"),
    "Task": ("models", "Task"),
    "TaskState": ("models", "TaskState"),
    "ResultDirError": ("resultdir", "ResultDirError"),
    "ResultDirectory": ("resultdir", "ResultDirectory"),
    "open_result_dir": ("resultdir", "open_result_dir"),
    "RunOutcome": ("runner", "RunOutcome"),
    "find_data_root": ("runner", "find_data_root"),
    "run_job_to_dir": ("runner", "run_job_to_dir"),
    "YamlError": ("yaml_min", "YamlError"),
    "yaml_dumps": ("yaml_min", "dumps"),
    "yaml_loads": ("yaml_min", "loads"),
}


def __getattr__(name: str) -> object:
    """PEP 562：第一次访问导出名时才 import 对应的子模块（见模块文档）。"""
    import importlib

    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
