"""K4 `.qwb` 项目容器（Python 侧，`zipfile`）。

与 Rust 的 `qwb_io::container::QwbContainer` **互操作**（P-02 验收）：

| 方向 | 手段 |
| --- | --- |
| Rust 写 → Python 读 | `qwb-contracts/studio/examples/example.qwb`（Rust 生成、入库）由 Python 打开并逐条目读取 |
| Python 写 → Rust 读 | `qwb-studio/tests/fixtures/python_written.qwb`（本模块生成、入库）由 `qwb-io` 的 `tests/p02_interop.rs` 打开、校验、并把 `input.yaml` 反序列化成 `JobInput` |

结构性约束照抄 K4（`qwb-format.md` §3，加载器必须**拒绝**而不是"尽力打开"）：

1. 条目名必须是相对 POSIX 路径：禁止绝对路径、盘符、反斜杠、`..`、空段、目录条目；
2. 条目名不得重复；
3. `meta.json` 必须存在且可解析，并且是容器里的第一个条目；
4. `meta.task_count` 必须等于 `tasks/` 下的目录数；
5. 不是合法 zip / zip 被截断 → 报错。

导出时条目时间戳固定 1980-01-01 00:00:00、压缩方法固定 Deflate、`meta.json` 第一、
其余按"旧顺序 + 新增名字典序"排列，因此**同一份内容重复导出逐字节相同**（由本模块
自己保证；跨 zlib 实现与 Rust 的 Deflate 字节不保证相同，所以跨语言只承诺**语义**互操作）。
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ._fs import atomic_replace

__all__ = [
    "DATA_SNAPSHOT_FILE",
    "FIXED_DATE_TIME",
    "K4_SCHEMA_VERSION",
    "META_FILE",
    "ContainerError",
    "DuplicateEntry",
    "EntryMissing",
    "InvalidEntry",
    "IoFailure",
    "MetaInvalid",
    "MetaMissing",
    "ProjectMeta",
    "QwbContainer",
    "check_entry_name",
    "task_entry_path",
]

META_FILE = "meta.json"
DATA_SNAPSHOT_FILE = "data_snapshot.json"
K4_SCHEMA_VERSION = "1.0.0"
#: 固定条目时间戳（K4 字节可复现）。
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
#: 固定权限位（Rust 侧 `unix_permissions(0o644)`）。
FIXED_EXTERNAL_ATTR = 0o644 << 16


class ContainerError(Exception):
    """`.qwb` 操作错误；`code` 是 K5 退出码（结构性 3 / IO 4），与 Rust 侧一致。"""

    code = 3

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ZipFailure(ContainerError):
    """不是合法 zip / zip 被截断。"""


class MetaMissing(ContainerError):
    """`meta.json` 缺失 —— 不是合法 `.qwb`。"""


class MetaInvalid(ContainerError):
    """`meta.json` 解析失败或语义非法（空字段、`task_count` 不符）。"""


class EntryMissing(ContainerError):
    """条目不存在。"""


class InvalidEntry(ContainerError):
    """条目名非法（绝对路径 / `..` 穿越 / 目录条目 / 空段）。"""


class DuplicateEntry(ContainerError):
    """条目名重复。"""


class IoFailure(ContainerError):
    """IO 失败（K5 退出码 4）。"""

    code = 4


def check_entry_name(name: str) -> None:
    """K4 条目名校验；非法抛 :class:`InvalidEntry`。"""
    if not name:
        raise InvalidEntry("invalid entry name: empty name")
    if name.startswith("/") or "\\" in name or ":" in name:
        raise InvalidEntry(f"invalid entry name: {name}: must be a relative POSIX path without drive letters")
    if name.endswith("/"):
        raise InvalidEntry(f"invalid entry name: {name}: directory entries are not allowed")
    for segment in name.split("/"):
        if segment in ("", ".", ".."):
            raise InvalidEntry(f"invalid entry name: {name}: path traversal or empty segment")


def task_entry_path(task_id: str, file: str) -> str:
    """`tasks/<id>/input.yaml` 之类的条目路径（同样做名字校验）。"""
    path = f"tasks/{task_id}/{file}"
    check_entry_name(path)
    return path


@dataclass(frozen=True)
class ProjectMeta:
    """`.qwb` 的 `meta.json`（字段与 Rust `ProjectMeta` 一致；未知字段必须报错）。"""

    schema_version: str
    name: str
    created: str
    app_version: str
    task_count: int
    notes: str | None = None

    FIELDS = ("schema_version", "name", "created", "app_version", "task_count", "notes")
    REQUIRED = ("schema_version", "name", "created", "app_version", "task_count")

    def check(self) -> list[str]:
        """语义校验；返回问题清单（空 = 合法），不抛异常（与 Rust `ProjectMeta::check` 一致）。"""
        issues: list[str] = []
        for label in ("schema_version", "name", "created", "app_version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"meta.{label} must not be empty")
        if not isinstance(self.task_count, int) or isinstance(self.task_count, bool):
            issues.append("meta.task_count must be an integer")
        elif self.task_count < 0:
            issues.append("meta.task_count must not be negative")
        if self.notes is not None and not isinstance(self.notes, str):
            issues.append("meta.notes must be a string or null")
        return issues

    def validate(self) -> None:
        issues = self.check()
        if issues:
            raise MetaInvalid("meta.json is invalid: " + "; ".join(issues))

    def to_json_object(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "created": self.created,
            "app_version": self.app_version,
            "task_count": self.task_count,
        }
        if self.notes is not None:
            obj["notes"] = self.notes
        return obj

    def to_json_bytes(self) -> bytes:
        """与 Rust `serde_json::to_vec_pretty` 对齐：2 空格缩进、无尾换行、非 ASCII 原样。"""
        return json.dumps(self.to_json_object(), indent=2, ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json_object(cls, obj: Any) -> "ProjectMeta":
        if not isinstance(obj, dict):
            raise MetaInvalid(f"meta.json must be a JSON object, got {type(obj).__name__}")
        unknown = sorted(set(obj) - set(cls.FIELDS))
        if unknown:
            raise MetaInvalid(f"meta.json has unknown fields: {', '.join(unknown)}")
        missing = [name for name in cls.REQUIRED if name not in obj]
        if missing:
            raise MetaInvalid(f"meta.json is missing required fields: {', '.join(missing)}")
        for name in ("schema_version", "name", "created", "app_version"):
            if not isinstance(obj[name], str):
                raise MetaInvalid(f"meta.{name} must be a string")
        if not isinstance(obj["task_count"], int) or isinstance(obj["task_count"], bool):
            raise MetaInvalid("meta.task_count must be an integer")
        notes = obj.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise MetaInvalid("meta.notes must be a string or null")
        meta = cls(
            schema_version=obj["schema_version"],
            name=obj["name"],
            created=obj["created"],
            app_version=obj["app_version"],
            task_count=obj["task_count"],
            notes=notes,
        )
        meta.validate()
        return meta

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "ProjectMeta":
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetaInvalid(f"meta.json is not valid JSON: {exc}") from exc
        return cls.from_json_object(obj)


def _as_bytes(value: bytes | bytearray | memoryview | str, name: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"content for {name!r} must be bytes or str, got {type(value).__name__}")


class QwbContainer:
    """`.qwb` 容器句柄（打开时校验结构；写回时整容器重写）。"""

    def __init__(self, path: Path, meta: ProjectMeta, order: Sequence[str], archive: zipfile.ZipFile) -> None:
        self._path = Path(path)
        self._meta = meta
        self._order = list(order)
        self._archive: zipfile.ZipFile | None = archive

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path, meta: ProjectMeta) -> "QwbContainer":
        """新建容器（只写 `meta.json`；随后用 :meth:`write_files` 加内容）。"""
        meta.validate()
        target = Path(path)
        container = cls(target, meta, [META_FILE], archive=None)
        container.write_files({})
        assert container._archive is not None
        return container

    @classmethod
    def open(cls, path: str | Path) -> "QwbContainer":
        """打开并校验容器结构。"""
        target = Path(path)
        try:
            archive = zipfile.ZipFile(target, "r")
        except FileNotFoundError as exc:
            raise IoFailure(f"io error on {target}: {exc}") from exc
        except IsADirectoryError as exc:
            raise IoFailure(f"io error on {target}: {exc}") from exc
        except zipfile.BadZipFile as exc:
            raise ZipFailure(f"not a valid .qwb container ({target}): {exc}") from exc
        except OSError as exc:
            raise IoFailure(f"io error on {target}: {exc}") from exc

        try:
            order: list[str] = []
            seen: set[str] = set()
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or name.endswith("/"):
                    raise InvalidEntry(
                        f"invalid entry name: {name}: directory entries are not allowed in .qwb (K4)"
                    )
                check_entry_name(name)
                if name in seen:
                    raise DuplicateEntry(f"duplicate entry name: {name}")
                seen.add(name)
                order.append(name)
            if META_FILE not in seen:
                raise MetaMissing("meta.json is missing: not a valid .qwb container")
            if order[0] != META_FILE:
                raise MetaInvalid(
                    f"meta.json must be the first entry in a .qwb container (K4), found {order[0]!r} first"
                )
            meta = ProjectMeta.from_json_bytes(archive.read(META_FILE))
        except ContainerError:
            archive.close()
            raise
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            archive.close()
            raise ZipFailure(f"corrupt .qwb container ({target}): {exc}") from exc

        return cls(target, meta, order, archive)

    # -- 访问 ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> ProjectMeta:
        return self._meta

    @property
    def entry_names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def read_file(self, name: str) -> bytes:
        """读一个条目（流式，内存占用 = 该条目大小）。"""
        check_entry_name(name)
        archive = self._require_archive()
        try:
            return archive.read(name)
        except KeyError as exc:
            raise EntryMissing(f"entry not found in container: {name}") from exc
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            raise ZipFailure(f"corrupt .qwb container ({self._path}): {exc}") from exc

    def read_text(self, name: str, encoding: str = "utf-8") -> str:
        return self.read_file(name).decode(encoding)

    def files(self) -> Iterator[tuple[str, bytes]]:
        """按容器顺序产出 `(name, bytes)`；artifact 很大的话请用 :meth:`read_file` 逐个读。"""
        for name in self._order:
            yield name, self.read_file(name)

    # -- 写入 ---------------------------------------------------------------

    def write_files(self, files: Mapping[str, bytes | str] | Iterable[tuple[str, bytes | str]]) -> None:
        """写/覆盖若干条目：重写整个容器（`meta.json` 永远第一，新条目按名字典序追加）。"""
        if isinstance(files, Mapping):
            items: list[tuple[str, bytes | str]] = list(files.items())
        else:
            items = list(files)
        for name, _ in items:
            check_entry_name(name)
        new_content: dict[str, bytes] = {}
        for name, value in items:
            if name in new_content:
                raise DuplicateEntry(f"duplicate entry name: {name}")
            new_content[name] = _as_bytes(value, name)

        # 先按旧顺序读出来（读一个、写一个），再落盘。
        old_contents: list[tuple[str, bytes]] = []
        for name in self._order:
            if name == META_FILE or name in new_content:
                old_contents.append((name, b""))
            else:
                old_contents.append((name, self.read_file(name)))

        order: list[str] = [META_FILE]
        payloads: list[tuple[str, bytes]] = [(META_FILE, self._meta.to_json_bytes())]
        for name, old in old_contents:
            if name == META_FILE:
                continue
            payloads.append((name, new_content.get(name, old)))
            order.append(name)
        written = set(order)
        for name in sorted(new_content):
            if name in written:
                continue
            payloads.append((name, new_content[name]))
            order.append(name)

        self.close()
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for name, data in payloads:
                    info = zipfile.ZipInfo(name, date_time=FIXED_DATE_TIME)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = FIXED_EXTERNAL_ATTR
                    info.create_system = 3  # 3 = Unix，让权限位可移植
                    archive.writestr(info, data)
            atomic_replace(tmp, self._path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise IoFailure(f"io error on {self._path}: {exc}") from exc
        self._order = order
        self._archive = zipfile.ZipFile(self._path, "r")

    # -- 校验 ---------------------------------------------------------------

    def validate(self) -> None:
        """结构自检：`meta.json` 合法、`task_count` 与 `tasks/` 目录数一致。"""
        issues = self._meta.check()
        task_ids = {
            name[len("tasks/") :].split("/", 1)[0]
            for name in self._order
            if name.startswith("tasks/") and "/" in name[len("tasks/") :]
        }
        if len(task_ids) != self._meta.task_count:
            issues.append(
                f"meta.task_count = {self._meta.task_count} but tasks/ holds {len(task_ids)} task directories"
            )
        if issues:
            raise MetaInvalid("; ".join(issues))

    def task_ids(self) -> list[str]:
        """`tasks/` 下的 task id（字典序）。"""
        return sorted(
            {
                name[len("tasks/") :].split("/", 1)[0]
                for name in self._order
                if name.startswith("tasks/") and "/" in name[len("tasks/") :]
            }
        )

    # -- 生命周期 -----------------------------------------------------------

    def _require_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            raise ContainerError(f"container is closed: {self._path}")
        return self._archive

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __enter__(self) -> "QwbContainer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"QwbContainer(path={str(self._path)!r}, entries={len(self._order)}, task_count={self._meta.task_count})"
