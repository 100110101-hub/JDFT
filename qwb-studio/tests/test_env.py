"""P-04 验收（生成侧）：`env.json` 生成器读到的都是**真值**。

判据来自 project.md 的 P-04 行：`rustc --version`、`Cargo.lock` sha256、数据快照、threads、os、
crate 版本；缺 `env.json` 时 `Result` 构造失败（在 `test_resultdir.py`）；
生成物能被 Rust `EnvInfo::validate` 接受（`test_resultdir.py` 里用 `qwb inspect` 端到端验证）。

这里的每个字段都与**独立重算**的值比对（不是"生成器自己跟自己对"）。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from qwb_studio.env import (
    EnvError,
    collect_env,
    crate_versions,
    data_snapshot,
    os_description,
    read_env_json,
    sha256_file,
    toolchain_channel,
    write_env_json,
)
from qwb_studio.models import EnvInfo

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_collect_env_reads_real_values() -> None:
    env = collect_env(REPO_ROOT)

    # rustc --version 的第一行原文
    assert env.rustc_version.startswith("rustc "), env.rustc_version
    assert re.match(r"^rustc \d+\.\d+\.\d+", env.rustc_version), env.rustc_version

    # Cargo.lock 的 sha256：独立用 hashlib 再算一遍
    lock = REPO_ROOT / "qwb-core" / "Cargo.lock"
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert env.cargo_lock_sha256 == expected
    assert re.fullmatch(r"[0-9a-f]{64}", env.cargo_lock_sha256)

    # 数据快照：独立用正则从 registry.yaml 抠出 path/sha256，再比字典
    registry = (REPO_ROOT / "qwb-data" / "registry.yaml").read_text(encoding="utf-8")
    pairs = re.findall(r"path:\s*(\S+)\s*\n\s*sha256:\s*([0-9a-f]{64})", registry)
    assert pairs, "registry.yaml should declare path/sha256 pairs"
    assert env.data_snapshot == dict(pairs)
    assert len(env.data_snapshot) >= 12

    # crate 版本清单：Cargo.lock 里 serde 的版本能对上
    serde = re.search(r'name = "serde"\nversion = "([^"]+)"', lock.read_text(encoding="utf-8"))
    assert serde is not None
    assert env.crate_versions["serde"] == serde.group(1)
    assert {"serde", "faer", "qwb-scf", "qwb-engine-api"} <= set(env.crate_versions)

    # os 与 toolchain channel
    assert re.fullmatch(r"(windows|linux|macos) (x86_64|aarch64|x86)", env.os), env.os
    assert env.toolchain_channel == "1.97.1"

    # 引擎当前单线程（I-04），默认必须记 1
    assert env.threads == 1
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", env.created_at_utc)

    # G5 判据自己也要过
    assert env.check() == []


def test_env_json_write_and_read_roundtrip(tmp_path: Path) -> None:
    env = collect_env(REPO_ROOT, threads=4, now="2026-10-25T00:00:00Z")
    path = write_env_json(tmp_path / "env.json", env)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{\n  \"engine_version\"")  # serde_json pretty 风格
    assert not text.endswith("\n")
    parsed = json.loads(text)
    assert parsed["threads"] == 4
    assert read_env_json(path) == env


def test_threads_must_be_positive() -> None:
    with pytest.raises(EnvError):
        collect_env(REPO_ROOT, threads=0)


def test_missing_registry_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EnvError) as excinfo:
        collect_env(tmp_path)
    assert "registry" in str(excinfo.value)


def test_bad_digest_is_refused_before_writing(tmp_path: Path) -> None:
    env = collect_env(REPO_ROOT)
    broken = env.model_copy(update={"data_snapshot": {"basis/x.yaml": "ZZZ"}})
    assert broken.check(), "a 3-char digest must be flagged"
    with pytest.raises(EnvError):
        write_env_json(tmp_path / "env.json", broken)


def test_checks_mirror_the_rust_g5_menu() -> None:
    good = collect_env(REPO_ROOT)
    assert good.check() == []
    for field, value in [
        ("engine_version", ""),
        ("rustc_version", "   "),
        ("os", ""),
        ("created_at_utc", ""),
        ("cargo_lock_sha256", "abc"),
        ("data_snapshot", {"basis/x.yaml": "not-hex"}),
    ]:
        broken = good.model_copy(update={field: value})
        assert broken.check(), f"{field} = {value!r} must be flagged"
    with pytest.raises(ValueError):
        good.model_copy(update={"engine_version": ""}).validate()


def test_helpers_are_independent_of_the_aggregator(tmp_path: Path) -> None:
    # 单独调每一个读取函数，行为与 collect_env 一致
    assert data_snapshot(REPO_ROOT / "qwb-data" / "registry.yaml")["basis/sto-3g.yaml"]
    assert crate_versions(REPO_ROOT / "qwb-core" / "Cargo.lock")["serde"]
    assert toolchain_channel(REPO_ROOT) == "1.97.1"
    assert toolchain_channel(tmp_path) is None
    assert sha256_file(REPO_ROOT / "qwb-core" / "Cargo.lock") == collect_env(REPO_ROOT).cargo_lock_sha256
    assert os_description().split()[0] in {"windows", "linux", "macos"}
    with pytest.raises(EnvError):
        data_snapshot(tmp_path / "nope.yaml")


def test_env_info_requires_the_k3_fields() -> None:
    with pytest.raises(Exception):
        EnvInfo.model_validate({"engine_version": "0.1.0"})  # type: ignore[arg-type]
