"""K3 结果目录的读取与 G5 复现门禁（P-04 的 `Result` 构造侧）。

布局（`qwb-format.md` §2，与 Rust `qwb_io::layout::ResultDir` 一致）：

```text
<task_id>/
  input.yaml
  env.json            # 缺它 = 不完整结果（G5，退出码 3）
  log.txt
  result.json
  data_snapshot.json
  artifacts/
```

比 Rust 的 `check()` 多两条**一致性**校验（Rust 的 `inspect` 只看存在性 + env 完整性）：

* `result.json` 里的 `env` 必须与 `env.json` 逐字段相同；
* `result.json` 里的 `data_snapshot` 必须与 `data_snapshot.json` 相同。

K3 明说 `data_snapshot.json` "与 result.json/env.json 里的 data_snapshot 同内容"，
所以这两条不是自找麻烦，而是把"同内容"真的当判据 —— 不一致就报错，不挑一个当真。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .env import EnvError, read_env_json
from .models import EnvInfo, Result

__all__ = [
    "ARTIFACTS_DIR",
    "ENV_FILE",
    "REQUIRED_FILES",
    "IncompleteEnv",
    "InconsistentResult",
    "MissingDir",
    "MissingEnvJson",
    "MissingResultFiles",
    "ResultDirError",
    "ResultDirectory",
    "ResultDirectory.check",
    "open_result_dir",
]

REQUIRED_FILES = ("input.yaml", "result.json", "log.txt", "data_snapshot.json")
ENV_FILE = "env.json"
ARTIFACTS_DIR = "artifacts"


class ResultDirError(Exception):
    """结果目录不可用；`code` 与 K5 一致（布局/复现问题 = 3）。"""

    code = 3


class MissingDir(ResultDirError):
    """task 目录不存在。"""


class MissingEnvJson(ResultDirError):
    """缺 `env.json`（K3/G5：结果不可复现）。"""


class MissingResultFiles(ResultDirError):
    """缺其它必需文件。"""


class IncompleteEnv(ResultDirError):
    """`env.json` 内容不完整。"""


class InconsistentResult(ResultDirError):
    """`result.json` 与 `env.json` / `data_snapshot.json` 不一致。"""


@dataclass(frozen=True)
class ResultDirectory:
    """一个通过校验的 K3 结果目录。"""

    root: Path
    result: Result
    env: EnvInfo
    input_yaml: str
    log: str
    has_artifacts: bool

    @classmethod
    def check(cls, path: str | Path) -> "ResultDirectory":
        root = Path(path)
        if not root.is_dir():
            raise MissingDir(f"task directory is missing: {root}")
        env_path = root / ENV_FILE
        if not env_path.is_file():
            raise MissingEnvJson(
                f"env.json is missing: {root} (result is not reproducible, K3/G5)"
            )
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise MissingResultFiles(f"missing required file(s) in {root}: {', '.join(missing)}")

        try:
            env = read_env_json(env_path)
        except EnvError as exc:
            raise IncompleteEnv(str(exc)) from exc

        try:
            raw = json.loads((root / "result.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InconsistentResult(f"cannot parse {root / 'result.json'}: {exc}") from exc
        try:
            result = Result.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - pydantic.ValidationError
            raise InconsistentResult(f"result.json does not match the K3 schema: {exc}") from exc

        if result.env != env:
            raise InconsistentResult(
                "result.json env does not match env.json: the directory is inconsistent (K3)"
            )
        try:
            snapshot = json.loads((root / "data_snapshot.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InconsistentResult(f"cannot parse {root / 'data_snapshot.json'}: {exc}") from exc
        if not isinstance(snapshot, dict):
            raise InconsistentResult(f"{root / 'data_snapshot.json'} must be a JSON object")
        if snapshot != result.data_snapshot:
            raise InconsistentResult(
                "data_snapshot.json does not match result.json.data_snapshot (K3)"
            )

        try:
            input_yaml = (root / "input.yaml").read_text(encoding="utf-8")
            log = (root / "log.txt").read_text(encoding="utf-8")
        except OSError as exc:
            raise InconsistentResult(f"cannot read the result directory {root}: {exc}") from exc

        return cls(
            root=root,
            result=result,
            env=env,
            input_yaml=input_yaml,
            log=log,
            has_artifacts=(root / ARTIFACTS_DIR).is_dir(),
        )


def open_result_dir(path: str | Path) -> ResultDirectory:
    """读取并校验一个 K3 结果目录（缺 `env.json` → :class:`MissingEnvJson`）。"""
    return ResultDirectory.check(path)
