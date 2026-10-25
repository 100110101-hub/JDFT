"""P-05 验收：Python 侧 `qwb` 入口是**薄壳**，子命令 `--help` 全通、退出码与 K5 一致。

判据来自 project.md 的 P-05 行：全部子命令 `--help` 通过；退出码与 K5 一致。

"退出码一致"不是把期望值抄一遍：每个用例都同时跑
① `python -m qwb_studio <args>`（被测的壳）和 ② 直接跑 Rust 二进制，
断言两者退出码**相同**、stdout 逐字节相同，并且码值落在 K5 的集合里。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qwb_studio.cli import K5_EXIT_CODES, EngineNotFound, engine_command

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"
PKG_ROOT = Path(__file__).resolve().parents[1]


def _engine() -> list[str]:
    command, _ = engine_command()
    assert command[0] != "cargo", (
        "P-05 tests need a built qwb binary (cargo test already builds it); "
        "run `cargo build -p qwb-cli --bin qwb` in qwb-core"
    )
    return command


def _run_python(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("QWB_ENGINE_BIN", None)
    return subprocess.run(  # noqa: S603 - 参数由本测试构造
        [sys.executable, "-m", "qwb_studio", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
        **kwargs,
    )


def _run_rust(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 参数由本测试构造
        [*_engine(), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )


HELP_CASES = [
    ["--help"],
    ["run", "--help"],
    ["batch", "--help"],
    ["inspect", "--help"],
    ["replay", "--help"],
    ["data", "--help"],
    ["data", "list", "--help"],
    ["data", "show", "--help"],
    ["data", "verify", "--help"],
    ["data", "sign", "--help"],
]


@pytest.mark.parametrize("args", HELP_CASES, ids=lambda args: " ".join(args))
def test_every_subcommand_help_passes(args: list[str]) -> None:
    wrapped = _run_python(args)
    assert wrapped.returncode == 0, wrapped.stdout + wrapped.stderr
    direct = _run_rust(args)
    assert direct.returncode == 0
    # 壳不许改写输出
    assert wrapped.stdout == direct.stdout
    assert "qwb" in wrapped.stdout.lower()


def test_wrapper_reports_the_usage_of_run() -> None:
    wrapped = _run_python(["run", "--help"])
    assert "--out" in wrapped.stdout
    assert "--threads" in wrapped.stdout


def test_engine_not_found_is_exit_code_four(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["QWB_ENGINE_BIN"] = str(tmp_path / "does-not-exist")
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "qwb_studio", "inspect", "."],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert completed.returncode == 4
    assert "QWB_ENGINE_BIN" in completed.stderr or "qwb-studio:" in completed.stderr


def test_engine_command_prefers_the_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / ("qwb.exe" if sys.platform.startswith("win") else "qwb")
    fake.write_text("not really a binary", encoding="utf-8")
    monkeypatch.setenv("QWB_ENGINE_BIN", str(fake))
    command, cwd = engine_command()
    assert command == [str(fake)]
    assert cwd is None
    monkeypatch.setenv("QWB_ENGINE_BIN", str(tmp_path / "missing"))
    with pytest.raises(EngineNotFound):
        engine_command()


# ---------------------------------------------------------------------------
# 退出码：逐例与直接调用 Rust 二进制比对
# ---------------------------------------------------------------------------

EXPECTED_EXIT = {
    "valid-run": 0,
    "bad-input": 1,
    "missing-env": 3,
    "missing-dir": 3,
    "bad-data-file": 3,
    "data-list": 0,
}


def _make_bad_input(tmp_path: Path) -> Path:
    path = tmp_path / "bad.yaml"
    path.write_text("molecule: {atoms: 3}\n", encoding="utf-8")
    return path


def _make_result_dir_without_env(tmp_path: Path) -> Path:
    root = tmp_path / "no-env"
    root.mkdir()
    for name in ("input.yaml", "result.json", "log.txt", "data_snapshot.json"):
        (root / name).write_text("{}", encoding="utf-8")
    return root


def _case_args(label: str, tmp_path: Path) -> list[str]:
    good = QWB_CORE / "examples" / "h2_sto3g_svwn5.yaml"
    if label == "valid-run":
        return ["run", str(good), "--out", str(tmp_path / "runs")]
    if label == "bad-input":
        return ["run", str(_make_bad_input(tmp_path))]
    if label == "missing-env":
        return ["inspect", str(_make_result_dir_without_env(tmp_path))]
    if label == "missing-dir":
        return ["inspect", str(tmp_path / "nope")]
    if label == "bad-data-file":
        return ["data", "verify", str(_make_bad_input(tmp_path))]
    if label == "data-list":
        return ["data", "list"]
    raise AssertionError(f"unknown case {label}")


@pytest.mark.parametrize("label", sorted(EXPECTED_EXIT))
def test_exit_codes_match_the_rust_binary(label: str, tmp_path: Path) -> None:
    args = _case_args(label, tmp_path)
    expected = EXPECTED_EXIT[label]
    assert expected in K5_EXIT_CODES, f"{expected} is not a K5 exit code"

    wrapped = _run_python(args)
    direct = _run_rust(args)
    assert wrapped.returncode == direct.returncode, (
        f"{label}: wrapper {wrapped.returncode} != engine {direct.returncode}\n"
        + wrapped.stdout[-2000:]
        + wrapped.stderr[-2000:]
    )
    assert direct.returncode == expected, (
        f"{label}: expected exit {expected} ({K5_EXIT_CODES[expected]}), "
        f"got {direct.returncode}\n{direct.stdout[-1000:]}{direct.stderr[-1000:]}"
    )
