"""Python 侧 `qwb` 入口（P-05）：薄壳，把一切转发给 Rust 的 `qwb` 二进制。

**不重新实现任何逻辑**：只做三件事 —— 找到引擎二进制、原样转发参数、原样返回退出码。
这样 K5 的冻结签名与退出码表只有一份实现（Rust），Python 侧不可能走出第二套语义。

定位顺序（可用 `QWB_ENGINE_BIN` 覆盖）：

1. 环境变量 `QWB_ENGINE_BIN` 指向的可执行文件（指向不存在的路径 → 退出码 4）；
2. 仓库里已构建的 `qwb-core/target/debug/qwb[.exe]`；
3. 都没有就用 `cargo run -q -p qwb-cli --bin qwb --`（在 `qwb-core` 里跑，慢但不会说谎）。

用法：

    python -m qwb_studio run examples/h2_sto3g_svwn5.yaml --out results
    python -m qwb_studio data verify
    qwb-studio --help
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = ["EngineNotFound", "engine_command", "main", "run_engine"]

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"

#: K5 的冻结退出码；本壳只转发，但测试用它做集合断言。
K5_EXIT_CODES = {
    0: "success",
    1: "input error",
    2: "SCF did not converge",
    3: "data validation failure",
    4: "internal error / unsupported",
}


class EngineNotFound(Exception):
    """找不到 Rust 引擎二进制（应映射为 K5 退出码 4）。"""


def _binary_name() -> str:
    return "qwb.exe" if sys.platform.startswith("win") else "qwb"


def engine_command(explicit: str | Path | None = None) -> tuple[list[str], Path | None]:
    """→ (命令前缀, 子进程 cwd)。`cwd` 只有 `cargo run` 分支需要。"""
    candidate = explicit if explicit is not None else os.environ.get("QWB_ENGINE_BIN")
    if candidate:
        path = Path(candidate)
        if not path.is_file():
            raise EngineNotFound(f"QWB_ENGINE_BIN points at a missing file: {path}")
        return [str(path)], None

    built = QWB_CORE / "target" / "debug" / _binary_name()
    if built.is_file():
        return [str(built)], None

    return ["cargo", "run", "-q", "-p", "qwb-cli", "--bin", "qwb", "--"], QWB_CORE


def run_engine(argv: list[str], *, explicit: str | Path | None = None) -> int:
    """把 `argv` 原样交给引擎，返回它的退出码（stdio 继承，不缓冲不重排）。"""
    command, cwd = engine_command(explicit)
    try:
        completed = subprocess.run(  # noqa: S603 - 参数由调用方原样给出
            [*command, *argv],
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EngineNotFound(f"cannot execute the engine ({command[0]}): {exc}") from exc
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return run_engine(args)
    except EngineNotFound as exc:
        print(f"qwb-studio: {exc}", file=sys.stderr)
        return 4  # K5: 内部错误 / 本构建不含引擎后端


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
