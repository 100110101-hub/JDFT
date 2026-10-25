"""P-02 验收：`.qwb` 容器的 Python 读写 + 与 Rust 的互操作。

"Rust 写 → Python 读" 用入库的 `qwb-contracts/studio/examples/example.qwb`（由 Rust
`qwb_io::container` 生成）；"Python 写 → Rust 读" 由本目录的
`fixtures/python_written.qwb` 承载，验证在 `qwb-io/tests/p02_interop.rs`。
这里同时覆盖 K4 的每一条结构性约束：损坏 zip、截断、缺 meta、meta 不在首位、
重复条目、目录条目、`..` 穿越、`task_count` 不符。
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from qwb_studio.container import (
    DATA_SNAPSHOT_FILE,
    META_FILE,
    ContainerError,
    DuplicateEntry,
    EntryMissing,
    InvalidEntry,
    MetaInvalid,
    MetaMissing,
    ProjectMeta,
    QwbContainer,
    ZipFailure,
    check_entry_name,
    task_entry_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_QWB = REPO_ROOT / "qwb-contracts" / "studio" / "examples" / "example.qwb"
FIXTURE_QWB = Path(__file__).resolve().parent / "fixtures" / "python_written.qwb"


def _meta(task_count: int = 1, name: str = "demo") -> ProjectMeta:
    return ProjectMeta(
        schema_version="1.0.0",
        name=name,
        created="2026-10-25T00:00:00Z",
        app_version="0.1.0",
        task_count=task_count,
    )


def _raw_zip(path: Path, entries: list[tuple[str, bytes]], *, meta_first: bool = True) -> Path:
    """绕过 QwbContainer 手工造 zip（用来验证加载器的拒绝行为）。"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return path


def _load_generator():
    path = REPO_ROOT / "scripts" / "gen_p02_fixture.py"
    spec = importlib.util.spec_from_file_location("gen_p02_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Rust 写 → Python 读
# ---------------------------------------------------------------------------


def test_open_rust_written_example_container() -> None:
    assert EXAMPLE_QWB.is_file(), f"missing Rust-written example {EXAMPLE_QWB}"
    with QwbContainer.open(EXAMPLE_QWB) as container:
        assert container.meta.name == "example-project"
        assert container.meta.task_count == 1
        assert container.meta.schema_version == "1.0.0"
        assert container.entry_names[0] == META_FILE
        assert "tasks/demo-0001/input.yaml" in container.entry_names
        container.validate()
        assert container.task_ids() == ["demo-0001"]

        # meta.json 能被 Python 侧解析成同样的字段
        meta_json = json.loads(container.read_file(META_FILE).decode("utf-8"))
        assert meta_json["name"] == "example-project"
        assert meta_json["task_count"] == 1

        # Rust 的 serde_yaml 把 position_angstrom 写成了块状序列，也能读
        input_yaml = container.read_text("tasks/demo-0001/input.yaml")
        assert "basis_name: sto-3g" in input_yaml
        assert "position_angstrom" in input_yaml


def test_python_rewrite_of_rust_container_preserves_entries(tmp_path: Path) -> None:
    copy = tmp_path / "rewritten.qwb"
    copy.write_bytes(EXAMPLE_QWB.read_bytes())

    with QwbContainer.open(copy) as container:
        before = dict(container.files())
        meta = container.meta
        container.write_files({})
        assert container.meta == meta

    with QwbContainer.open(copy) as rewritten:
        after = dict(rewritten.files())
        rewritten.validate()

    assert set(before) == set(after)
    for name, data in before.items():
        assert after[name] == data, f"entry {name} changed across Python rewrite"


# ---------------------------------------------------------------------------
# Python 写 → （Rust 读）夹具保持最新
# ---------------------------------------------------------------------------


def test_committed_fixture_matches_current_writer() -> None:
    generator = _load_generator()
    ok, detail = generator.fixture_matches()
    assert ok, detail


def test_fixture_is_valid_and_has_two_tasks() -> None:
    with QwbContainer.open(FIXTURE_QWB) as container:
        container.validate()
        assert container.meta.name == "python-fixture"
        assert container.meta.task_count == 2
        assert container.task_ids() == ["t1", "t2"]
        assert container.entry_names[0] == META_FILE
        snapshot = json.loads(container.read_file(DATA_SNAPSHOT_FILE).decode("utf-8"))
        assert len(snapshot) == 3
        assert all(len(digest) == 64 for digest in snapshot.values())


# ---------------------------------------------------------------------------
# 基本读写
# ---------------------------------------------------------------------------


def test_create_write_reopen_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "project.qwb"
    container = QwbContainer.create(path, _meta(task_count=1))
    try:
        assert container.entry_names == (META_FILE,)
        container.write_files(
            {
                task_entry_path("t1", "input.yaml"): "jobtype: sp\n",
                DATA_SNAPSHOT_FILE: "{}\n",
            }
        )
    finally:
        container.close()

    with QwbContainer.open(path) as reopened:
        reopened.validate()
        assert reopened.meta.name == "demo"
        assert reopened.read_text(task_entry_path("t1", "input.yaml")) == "jobtype: sp\n"
        assert reopened.task_ids() == ["t1"]
        # meta.json 永远第一，其余按名字典序追加
        assert reopened.entry_names == (
            META_FILE,
            DATA_SNAPSHOT_FILE,
            task_entry_path("t1", "input.yaml"),
        )


def test_export_is_byte_stable(tmp_path: Path) -> None:
    payload = {
        task_entry_path("t1", "input.yaml"): b"jobtype: sp\n" * 20,
        DATA_SNAPSHOT_FILE: b"{}\n",
        "artifacts/notes.txt": b"hi\n",
    }

    def build(name: str) -> bytes:
        path = tmp_path / name
        container = QwbContainer.create(path, _meta())
        try:
            container.write_files(payload)
        finally:
            container.close()
        return path.read_bytes()

    first = build("a.qwb")
    second = build("b.qwb")
    assert first == second, "K4/G5: 相同内容必须导出相同的字节"


def test_overwrite_replaces_content_and_drops_nothing(tmp_path: Path) -> None:
    path = tmp_path / "project.qwb"
    container = QwbContainer.create(path, _meta())
    try:
        container.write_files({DATA_SNAPSHOT_FILE: b"{\"a\": 1}\n", "artifacts/x.txt": b"x"})
        container.write_files({DATA_SNAPSHOT_FILE: b"{\"a\": 2}\n"})
    finally:
        container.close()

    with QwbContainer.open(path) as reopened:
        assert reopened.read_file(DATA_SNAPSHOT_FILE) == b"{\"a\": 2}\n"
        assert reopened.read_file("artifacts/x.txt") == b"x"


def test_read_missing_entry_raises(tmp_path: Path) -> None:
    path = tmp_path / "project.qwb"
    container = QwbContainer.create(path, _meta())
    try:
        with pytest.raises(EntryMissing):
            container.read_file("tasks/nope/input.yaml")
    finally:
        container.close()


def test_closed_container_raises(tmp_path: Path) -> None:
    path = tmp_path / "project.qwb"
    container = QwbContainer.create(path, _meta())
    container.close()
    with pytest.raises(ContainerError):
        container.read_file(META_FILE)


def test_write_rejects_bad_entry_names(tmp_path: Path) -> None:
    path = tmp_path / "project.qwb"
    container = QwbContainer.create(path, _meta())
    try:
        for bad in ["../evil", "/abs/path", "a\\b", "a//b", "dir/"]:
            with pytest.raises(InvalidEntry):
                container.write_files({bad: b"x"})
    finally:
        container.close()


# ---------------------------------------------------------------------------
# K4 结构性约束（必须拒绝）
# ---------------------------------------------------------------------------


def test_entry_name_rules() -> None:
    for bad in ["", "/abs", "C:\\win", "a\\b", "a/../b", "a//b", "dir/", "../evil"]:
        with pytest.raises(InvalidEntry):
            check_entry_name(bad)
    for good in ["meta.json", "tasks/t1/input.yaml", "artifacts/mo.png", "a.b-c_d"]:
        check_entry_name(good)


def test_corrupt_zip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.qwb"
    path.write_bytes(b"this is not a zip file at all")
    with pytest.raises(ZipFailure) as excinfo:
        QwbContainer.open(path)
    assert excinfo.value.code == 3


def test_truncated_zip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.qwb"
    data = EXAMPLE_QWB.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ZipFailure) as excinfo:
        QwbContainer.open(path)
    assert excinfo.value.code == 3


def test_missing_file_is_io_failure(tmp_path: Path) -> None:
    with pytest.raises(ContainerError) as excinfo:
        QwbContainer.open(tmp_path / "nope.qwb")
    assert excinfo.value.code == 4


def test_missing_meta_is_rejected(tmp_path: Path) -> None:
    path = _raw_zip(tmp_path / "nometa.qwb", [("tasks/t1/input.yaml", b"jobtype: sp\n")])
    with pytest.raises(MetaMissing):
        QwbContainer.open(path)


def test_invalid_meta_json_is_rejected(tmp_path: Path) -> None:
    path = _raw_zip(
        tmp_path / "badmeta.qwb",
        [(META_FILE, b"{ not json }"), ("tasks/t1/input.yaml", b"x")],
    )
    with pytest.raises(MetaInvalid):
        QwbContainer.open(path)


def test_meta_must_be_first_entry(tmp_path: Path) -> None:
    meta = _meta().to_json_bytes()
    path = _raw_zip(
        tmp_path / "meta-last.qwb",
        [("tasks/t1/input.yaml", b"x"), (META_FILE, meta)],
    )
    with pytest.raises(MetaInvalid):
        QwbContainer.open(path)


def test_duplicate_entries_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.qwb"
    meta = _meta().to_json_bytes()
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(META_FILE, meta)
            archive.writestr(META_FILE, meta)
    with pytest.raises(DuplicateEntry):
        QwbContainer.open(path)


def test_directory_entries_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dirs.qwb"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(META_FILE, _meta().to_json_bytes())
        archive.writestr("artifacts/", b"")
    with pytest.raises(InvalidEntry):
        QwbContainer.open(path)


def test_path_traversal_entries_are_rejected(tmp_path: Path) -> None:
    path = _raw_zip(
        tmp_path / "traversal.qwb",
        [(META_FILE, _meta().to_json_bytes()), ("../evil.txt", b"x")],
    )
    with pytest.raises(InvalidEntry):
        QwbContainer.open(path)


def test_task_count_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "count.qwb"
    container = QwbContainer.create(path, _meta(task_count=2))
    try:
        container.write_files({task_entry_path("t1", "input.yaml"): b"x"})
    finally:
        container.close()
    with QwbContainer.open(path) as reopened:
        with pytest.raises(MetaInvalid):
            reopened.validate()


def test_meta_unknown_fields_rejected() -> None:
    payload = {
        "schema_version": "1.0.0",
        "name": "x",
        "created": "2026-10-25T00:00:00Z",
        "app_version": "0.1.0",
        "task_count": 0,
        "bogus": 1,
    }
    with pytest.raises(MetaInvalid):
        ProjectMeta.from_json_object(payload)


def test_meta_missing_fields_rejected() -> None:
    with pytest.raises(MetaInvalid):
        ProjectMeta.from_json_object({"schema_version": "1.0.0", "name": "x"})
    with pytest.raises(MetaInvalid):
        ProjectMeta.from_json_object([])


def test_meta_empty_name_rejected() -> None:
    meta = ProjectMeta(
        schema_version="1.0.0",
        name="   ",
        created="2026-10-25T00:00:00Z",
        app_version="0.1.0",
        task_count=0,
    )
    assert meta.check()
    with pytest.raises(MetaInvalid):
        meta.validate()


def test_meta_notes_roundtrip() -> None:
    meta = ProjectMeta(
        schema_version="1.0.0",
        name="demo",
        created="2026-10-25T00:00:00Z",
        app_version="0.1.0",
        task_count=0,
        notes="含中文与 emoji 🧪",
    )
    again = ProjectMeta.from_json_bytes(meta.to_json_bytes())
    assert again == meta
    assert json.loads(meta.to_json_bytes().decode("utf-8"))["notes"] == "含中文与 emoji 🧪"
