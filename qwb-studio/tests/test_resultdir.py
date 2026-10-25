"""P-04 验收（结果侧）：缺 `env.json` 时 `Result` 构造失败 + Rust 接受我们生成的 `env.json`。

两条证据：

1. `open_result_dir` 在缺 `env.json` 时抛 `MissingEnvJson`（K3/G5，退出码 3），
   而不是"读个差不多"；
2. 用 `qwb inspect`（Rust 侧 `qwb_io::layout::ResultDir::check` → `EnvInfo::validate`）
   检查一个**env.json 由 Python 生成**的结果目录，必须退出 0；
   删掉 env.json 后必须退出 3。

另外 `test_open_real_engine_result_dir` 跑一次真实的 `qwb run`（H2/STO-3G/SVWN5，约 1 秒），
用 Python 的模型读回结果目录 —— 证明我们的 `Result`/`EnvInfo` 能吃真实引擎输出，而不是只能吃自己写的。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qwb_studio.env import collect_env, write_env_json
from qwb_studio.models import EnvInfo, Result
from qwb_studio.resultdir import (
    IncompleteEnv,
    InconsistentResult,
    MissingDir,
    MissingEnvJson,
    MissingResultFiles,
    open_result_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"


# ---------------------------------------------------------------------------
# Rust CLI 定位（gate 里 cargo test 已经建好了 bin；没有就 build 一次）
# ---------------------------------------------------------------------------


def _qwb_binary() -> Path:
    name = "qwb.exe" if sys.platform.startswith("win") else "qwb"
    candidate = QWB_CORE / "target" / "debug" / name
    if candidate.is_file():
        return candidate
    completed = subprocess.run(  # noqa: S603 - 固定参数
        ["cargo", "build", "-q", "-p", "qwb-cli", "--bin", "qwb"],
        cwd=str(QWB_CORE),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, (
        "cannot build the qwb CLI (P-04 needs it to prove Rust accepts our env.json):\n"
        + completed.stderr[-4000:]
    )
    assert candidate.is_file(), f"expected {candidate} after a successful build"
    return candidate


def _run_qwb(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 固定参数
        [str(_qwb_binary()), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


# ---------------------------------------------------------------------------
# 造一个自洽的 K3 结果目录
# ---------------------------------------------------------------------------


def _build_result_dir(root: Path, env: EnvInfo, energy: float = -1.1570004991) -> Path:
    """写一个**自洽**的 K3 目录。

    `energy` 是 status.yaml 里记录的 H2/STO-3G/SVWN5 实测值；这个目录是**格式夹具**，
    不是"又算了一遍"的结果 —— 真跑引擎的验证在 `test_open_real_engine_result_dir`。
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    result = Result(
        energy_hartree=energy,
        scf_converged=True,
        data_snapshot=dict(env.data_snapshot),
        log="fixture: format only, see tests/test_resultdir.py\n",
        env=env,
    )
    (root / "input.yaml").write_text(
        "molecule:\n  atoms:\n    - symbol: H\n      position_angstrom: [0.0, 0.0, -0.37]\n"
        "    - symbol: H\n      position_angstrom: [0.0, 0.0, 0.37]\n  charge: 0\n  multiplicity: 1\n"
        "basis_name: sto-3g\nmethod_name: svwn5\n",
        encoding="utf-8",
    )
    (root / "log.txt").write_text(result.log, encoding="utf-8")
    (root / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "data_snapshot.json").write_text(
        json.dumps(result.data_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_env_json(root / "env.json", env)
    return root


@pytest.fixture(scope="module")
def generated_env() -> EnvInfo:
    return collect_env(REPO_ROOT)


# ---------------------------------------------------------------------------
# G5：缺 env.json = 结果不完整
# ---------------------------------------------------------------------------


def test_missing_env_json_is_the_g5_error(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    (root / "env.json").unlink()
    with pytest.raises(MissingEnvJson) as excinfo:
        open_result_dir(root)
    assert excinfo.value.code == 3
    assert "env.json is missing" in str(excinfo.value)


def test_result_construction_without_env_is_impossible(tmp_path: Path, generated_env: EnvInfo) -> None:
    # pydantic 层：Result 的 env 是必填字段
    payload = {
        "energy_hartree": -1.0,
        "scf_converged": True,
        "data_snapshot": dict(generated_env.data_snapshot),
        "log": "",
    }
    with pytest.raises(Exception):
        Result.model_validate(payload)
    # 结果目录层：没有 env.json 就不是一个结果目录
    root = _build_result_dir(tmp_path / "task", generated_env)
    (root / "env.json").unlink()
    with pytest.raises(MissingEnvJson):
        open_result_dir(root)


def test_missing_other_files_are_listed(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    (root / "log.txt").unlink()
    (root / "result.json").unlink()
    with pytest.raises(MissingResultFiles) as excinfo:
        open_result_dir(root)
    message = str(excinfo.value)
    assert "log.txt" in message and "result.json" in message


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(MissingDir):
        open_result_dir(tmp_path / "nope")


def test_incomplete_env_is_rejected(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    broken = json.loads((root / "env.json").read_text(encoding="utf-8"))
    broken["cargo_lock_sha256"] = "not-a-sha"
    broken["threads"] = 0
    (root / "env.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(IncompleteEnv):
        open_result_dir(root)


def test_inconsistent_env_is_rejected(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    raw = json.loads((root / "result.json").read_text(encoding="utf-8"))
    raw["env"]["threads"] = 99
    (root / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(InconsistentResult):
        open_result_dir(root)


def test_inconsistent_data_snapshot_is_rejected(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    (root / "data_snapshot.json").write_text('{"basis/other.yaml": "' + "a" * 64 + '"}', encoding="utf-8")
    with pytest.raises(InconsistentResult):
        open_result_dir(root)


def test_consistent_directory_reads_back(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    directory = open_result_dir(root)
    assert directory.result.energy_hartree == pytest.approx(-1.1570004991)
    assert directory.env == generated_env
    assert directory.has_artifacts is True
    assert "basis_name: sto-3g" in directory.input_yaml
    assert directory.log.startswith("fixture: format only")


# ---------------------------------------------------------------------------
# 跨语言：Rust 的 EnvInfo::validate 必须接受 Python 生成的 env.json
# ---------------------------------------------------------------------------


def test_rust_qwb_inspect_accepts_the_generated_env_json(tmp_path: Path, generated_env: EnvInfo) -> None:
    root = _build_result_dir(tmp_path / "task", generated_env)
    completed = _run_qwb("inspect", str(root))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = completed.stdout
    assert "status  ok" in text, text
    # Rust 打印的是 `env.rustc_version` 原文（本身以 "rustc " 开头）
    assert f"env     rustc {generated_env.rustc_version}" in text, text
    assert f"threads {generated_env.threads}" in text, text
    assert f"data snapshot: {len(generated_env.data_snapshot)} file(s)" in text, text

    # 对照：删掉 env.json → Rust 侧同样是退出码 3（G5）
    (root / "env.json").unlink()
    completed = _run_qwb("inspect", str(root))
    assert completed.returncode == 3, completed.stdout + completed.stderr
    assert "env.json is missing" in completed.stdout + completed.stderr


# ---------------------------------------------------------------------------
# 真实引擎结果：Python 模型能读 Rust 写的结果目录
# ---------------------------------------------------------------------------


def test_open_real_engine_result_dir(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    completed = _run_qwb(
        "run", "examples/h2_sto3g_svwn5.yaml", "--out", str(out), cwd=QWB_CORE
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    energy_line = [line for line in completed.stdout.splitlines() if line.startswith("energy")]
    assert energy_line, completed.stdout
    printed_energy = float(energy_line[0].split()[1])

    root = out / "h2_sto3g_svwn5"
    directory = open_result_dir(root)
    # `qwb run` 只打印 10 位小数，result.json 里是全精度，所以按 1e-9 比
    assert directory.result.energy_hartree == pytest.approx(printed_energy, abs=1e-9)
    assert directory.result.scf_converged is True
    # 真实引擎写的 env.json 自带 crate_versions（可能为空）与 toolchain channel
    assert directory.env.rustc_version.startswith("rustc ")
    assert directory.env.cargo_lock_sha256 == generated_env_sha()
    assert directory.has_artifacts is True
    shutil.rmtree(out, ignore_errors=True)


def generated_env_sha() -> str:
    return collect_env(REPO_ROOT).cargo_lock_sha256
