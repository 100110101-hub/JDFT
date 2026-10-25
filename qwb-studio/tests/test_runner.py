"""P-07 验收：Python 侧端到端跑通 T1，并且写出的结果目录 Rust 能复现。

判据来自 project.md 的 P-07 行（"跑通 T1"）。这里把它拆成可复跑的三件事：

1. `qwb_engine` 跑 H2/STO-3G/SVWN5，能量与 Rust CLI 写进 `result.json` 的值**逐位相同**；
2. 写出的 K3 结果目录通过 `qwb inspect`（Rust 的布局 + G5 门禁）；
3. 该目录能被 `qwb replay` **重算**并且 `delta = 0.000e0 Ha`（Python 写的 input.yaml /
   result.json / env.json 对 Rust 来说是真的可复现结果）。

第 3 条同时是 P-08 的判据（replay < 1e-10），这里是它的跨语言版本。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import qwb_engine as qe

from qwb_studio.runner import RunnerError, find_data_root, run_job_to_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
QWB_CORE = REPO_ROOT / "qwb-core"
DATA_ROOT = REPO_ROOT / "qwb-data"


def _qwb_binary() -> Path:
    name = "qwb.exe" if sys.platform.startswith("win") else "qwb"
    candidate = QWB_CORE / "target" / "debug" / name
    if candidate.is_file():
        return candidate
    completed = subprocess.run(  # noqa: S603
        ["cargo", "build", "-q", "-p", "qwb-cli", "--bin", "qwb"],
        cwd=str(QWB_CORE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return candidate


def _run_qwb(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(_qwb_binary()), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )


def _h2_job() -> qe.JobInput:
    molecule = qe.Molecule([qe.Atom("H", (0.0, 0.0, -0.37)), qe.Atom("H", (0.0, 0.0, 0.37))])
    return qe.JobInput(molecule, "sto-3g", "svwn5")


@pytest.fixture(scope="module")
def python_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("python-run")
    outcome = run_job_to_dir(_h2_job(), root, "py-h2", data_root=DATA_ROOT)
    yield outcome
    shutil.rmtree(root, ignore_errors=True)


def test_h2_runs_end_to_end(python_run) -> None:
    outcome = python_run
    assert outcome.energy_hartree == pytest.approx(-1.1570005110367827, abs=1e-12)
    directory = outcome.directory
    assert directory.result.scf_converged is True
    assert directory.has_artifacts is True
    # K3 的五个必需文件 + artifacts/
    names = sorted(path.name for path in outcome.root.iterdir())
    assert names == ["artifacts", "data_snapshot.json", "env.json", "input.yaml", "log.txt", "result.json"]
    # env 是引擎给的，不是手写的
    assert directory.env.rustc_version.startswith("rustc ")
    assert len(directory.env.data_snapshot) >= 12


def test_energy_matches_the_rust_cli(python_run, tmp_path: Path) -> None:
    example = QWB_CORE / "examples" / "h2_sto3g_svwn5.yaml"
    completed = _run_qwb("run", str(example), "--out", str(tmp_path / "cli"))
    assert completed.returncode == 0, completed.stderr
    recorded = json.loads(
        (tmp_path / "cli" / "h2_sto3g_svwn5" / "result.json").read_text(encoding="utf-8")
    )
    assert python_run.energy_hartree == recorded["energy_hartree"]


def test_input_yaml_roundtrips_through_the_engine(python_run) -> None:
    text = (python_run.root / "input.yaml").read_text(encoding="utf-8")
    assert "basis_name: sto-3g" in text
    assert "method_name: svwn5" in text
    assert "jobtype: sp" in text
    assert "ecp_name: null" in text


def test_rust_inspect_accepts_the_python_result_dir(python_run) -> None:
    completed = _run_qwb("inspect", str(python_run.root))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "status  ok" in completed.stdout
    assert "env.json" in completed.stdout


def test_rust_replay_reproduces_the_python_result(python_run) -> None:
    completed = _run_qwb("replay", str(python_run.root))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "delta   0.000e0 Ha" in completed.stdout
    assert "reproduced" in completed.stdout


def test_data_root_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWB_DATA_ROOT", raising=False)
    assert find_data_root(explicit=DATA_ROOT) == DATA_ROOT
    with pytest.raises(RunnerError):
        find_data_root(explicit=tmp_path / "nope")
    # 仓库里任意位置都该向上找到 qwb-data
    assert find_data_root(start=QWB_CORE) == DATA_ROOT
    # 一个没有任何 qwb-data 的目录树里必须报错，而不是猜
    isolated = tmp_path / "isolated" / "deep"
    isolated.mkdir(parents=True)
    with pytest.raises(RunnerError):
        find_data_root(start=isolated)
    # 环境变量优先
    monkeypatch.setenv("QWB_DATA_ROOT", str(DATA_ROOT))
    assert find_data_root(start=isolated) == DATA_ROOT
