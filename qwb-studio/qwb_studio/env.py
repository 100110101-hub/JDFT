"""`env.json` 生成器（P-04）：把复现环境需要的每一项都**真的读出来**。

字段来源（没有占位符，取不到就明说 `xxx (unavailable)`，与 Rust `qwb-engine::env` 的约定一致）：

| 字段 | 来源 |
| --- | --- |
| `engine_version` | 本包的 `__version__`（与引擎同号） |
| `rustc_version` | 运行 `rustc --version`（第一行原文） |
| `cargo_lock_sha256` | `qwb-core/Cargo.lock` 字节的 sha256（Rust 侧是编译期嵌入同一份文件） |
| `toolchain_channel` | 根 `rust-toolchain.toml` 的 `channel` |
| `data_snapshot` | `qwb-data/registry.yaml` 里每个条目的 `path -> sha256`（注册表自己的摘要，不重算） |
| `crate_versions` | 从 `Cargo.lock` 解析出的 `name -> version` 全量清单 |
| `threads` | 调用方要传给引擎的线程数；引擎当前单线程（I-04），所以默认记 1 |
| `os` | `"{os} {arch}"`，与 Rust `env.rs` 的 `std::env::consts::{OS, ARCH}` 同格式 |
| `created_at_utc` | 生成时刻，ISO 8601 UTC |

`threads` 的口径写清楚：K3 要求的是"**实际生效**线程数"。引擎自己的 `env.json` 由
`qwb-engine::env` 写出（I-04 未并行 → 记 1），那是引擎运行时的权威值；本模块是给
"Python 侧要写一份 env.json" 的场景用的（GUI 预览、测试、外部工具），所以 `threads`
默认 1 以与当前引擎实际行为一致，调用方确实并行时才传真实值。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ._fs import atomic_replace, read_text_retry
from .models import EnvInfo, utc_now_iso
from .yaml_min import loads as yaml_loads

__all__ = [
    "CARGO_LOCK_RELATIVE",
    "REGISTRY_RELATIVE",
    "TOOLCHAIN_RELATIVE",
    "EnvError",
    "collect_env",
    "crate_versions",
    "data_snapshot",
    "os_description",
    "read_env_json",
    "sha256_file",
    "toolchain_channel",
    "write_env_json",
]

CARGO_LOCK_RELATIVE = Path("qwb-core") / "Cargo.lock"
REGISTRY_RELATIVE = Path("qwb-data") / "registry.yaml"
TOOLCHAIN_RELATIVE = Path("rust-toolchain.toml")


class EnvError(Exception):
    """复现环境无法生成（缺文件、命令不可用等）。"""


# ---------------------------------------------------------------------------
# 单项读取
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """文件的 sha256（64 位小写十六进制）。"""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EnvError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _command_first_line(command: list[str]) -> str:
    """跑一个命令取第一行；失败返回 `"<program> (unavailable)"`（与 Rust 侧文案一致）。"""
    program = command[0]
    try:
        completed = subprocess.run(  # noqa: S603 - 命令由本模块写死
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return f"{program} (unavailable)"
    text = (completed.stdout or "").strip()
    if completed.returncode != 0 or not text:
        return f"{program} (unavailable)"
    return text.splitlines()[0].strip()


def toolchain_channel(root: str | Path) -> str | None:
    """`rust-toolchain.toml` 的 channel（没有就是 None）。"""
    path = Path(root) / TOOLCHAIN_RELATIVE
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("channel"):
            rest = stripped[len("channel") :].lstrip()
            if not rest.startswith("="):
                continue
            return rest[1:].strip().strip('"')
    return None


def data_snapshot(registry_path: str | Path) -> dict[str, str]:
    """`path -> sha256`：取自注册表清单（`registry.yaml` 各分节的条目）。"""
    path = Path(registry_path)
    if not path.is_file():
        raise EnvError(f"registry manifest not found: {path}")
    try:
        manifest = yaml_loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - YAML 子集错误统一成 EnvError
        raise EnvError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EnvError(f"{path} must contain a mapping")
    snapshot: dict[str, str] = {}
    for section, entries in manifest.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
                continue
            entry_path = entry["path"]
            digest = entry["sha256"]
            if not isinstance(entry_path, str) or not isinstance(digest, str):
                raise EnvError(f"{path}: section {section!r} has a non-string path/sha256")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise EnvError(f"{path}: digest for {entry_path!r} is not a 64-char lowercase sha256")
            snapshot[entry_path] = digest
    if not snapshot:
        raise EnvError(f"{path} declares no data entries")
    return snapshot


def crate_versions(lock_path: str | Path) -> dict[str, str]:
    """从 `Cargo.lock` 解析 `name -> version`（`[[package]]` 块）。"""
    path = Path(lock_path)
    if not path.is_file():
        raise EnvError(f"Cargo.lock not found: {path}")
    versions: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            name = None
            continue
        if stripped.startswith("name = "):
            name = stripped[len("name = ") :].strip().strip('"')
        elif stripped.startswith("version = ") and name is not None:
            version = stripped[len("version = ") :].strip().strip('"')
            versions[name] = version
            name = None
    if not versions:
        raise EnvError(f"{path} contains no [[package]] entries")
    return versions


def os_description() -> str:
    """`"{os} {arch}"`，与 Rust `env.rs` 的 `std::env::consts::{OS, ARCH}` 同格式。"""
    system = {"windows": "windows", "darwin": "macos", "linux": "linux"}.get(
        platform.system().lower(), platform.system().lower() or "unknown"
    )
    machine = platform.machine().lower()
    arch = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "x86": "x86",
    }.get(machine, machine or "unknown")
    return f"{system} {arch}"


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def collect_env(
    root: str | Path,
    *,
    threads: int | None = None,
    engine_version: str | None = None,
    now: str | None = None,
) -> EnvInfo:
    """读仓库文件 + 环境，拼出合法的 `EnvInfo`。

    `threads` 默认 1（引擎当前单线程）；调用方确实要并行时传真实值。
    """
    root = Path(root)
    if engine_version is None:
        from . import __version__ as engine_version  # 延迟导入，避免循环

    resolved_threads = 1 if threads is None else int(threads)
    if resolved_threads < 1:
        raise EnvError(f"threads must be >= 1, got {resolved_threads}")

    env = EnvInfo(
        engine_version=str(engine_version),
        rustc_version=_command_first_line(["rustc", "--version"]),
        cargo_lock_sha256=sha256_file(root / CARGO_LOCK_RELATIVE),
        data_snapshot=data_snapshot(root / REGISTRY_RELATIVE),
        threads=resolved_threads,
        os=os_description(),
        created_at_utc=now or utc_now_iso(),
        toolchain_channel=toolchain_channel(root),
        crate_versions=crate_versions(root / CARGO_LOCK_RELATIVE),
    )
    issues = env.check()
    if issues:
        raise EnvError("generated env.json is incomplete: " + "; ".join(issues))
    return env


def write_env_json(path: str | Path, env: EnvInfo) -> Path:
    """写 `env.json`（`serde_json::to_vec_pretty` 风格：2 空格缩进、无尾换行）；原子替换。"""
    issues = env.check()
    if issues:
        raise EnvError("refusing to write an incomplete env.json: " + "; ".join(issues))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, Any] = env.model_dump(mode="json")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_replace(tmp, target)
    return target


def read_env_json(path: str | Path) -> EnvInfo:
    """读 `env.json`：缺字段/格式不对 → `EnvError`（对应 K3 的"不完整结果"）。"""
    target = Path(path)
    try:
        raw = json.loads(read_text_retry(target))
    except FileNotFoundError as exc:
        raise EnvError(f"env.json is missing: {target} (result is not reproducible, K3/G5)") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvError(f"cannot read {target}: {exc}") from exc
    try:
        env = EnvInfo.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - pydantic.ValidationError
        raise EnvError(f"env.json is incomplete: {target}: {exc}") from exc
    issues = env.check()
    if issues:
        raise EnvError("env.json is incomplete: " + "; ".join(issues))
    return env
