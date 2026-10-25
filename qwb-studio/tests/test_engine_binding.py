"""P-06 验收：`qwb_engine`（pyo3 + maturin，abi3-py39）与 K6 冻结桩一致，且真的释放 GIL。

判据来自 project.md 的 P-06 行：与 `qwb_engine.pyi` 一致（`mypy --strict` 通过，K6 门禁已在跑）；
并发测试证明 GIL 被释放。

这里的做法：
* **接口一致性**：直接解析 `qwb-contracts/studio/python/qwb_engine.pyi` 的文本，取出每个
  `class` 的名字与方法名，逐个 `hasattr` 到运行时对象上；`__all__` 也逐项比对
  （只排除 `GridLevel`/`JobType`/`Reporter` 三个**纯类型层**名字：桩里它们是 `Literal`/`Protocol`，
  运行时不存在，硬塞进 `__all__` 会让 `import *` 直接炸）。
* **数值一致性**：同一个 H2/STO-3G/SVWN5 作业，Python 绑定与 Rust CLI 的能量必须**逐位相同**。
* **GIL**：① 计算期间另一个纯 Python 线程必须还能跑；② 两个作业并发时墙钟远小于串行和。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

import qwb_engine as qe

from qwb_studio.env import collect_env
from qwb_studio.models import Result

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"
STUB = REPO_ROOT / "qwb-contracts" / "studio" / "python" / "qwb_engine.pyi"

#: 桩里的纯类型名字（`Literal` / `Protocol`），运行时没有对象。
TYPING_ONLY = {"GridLevel", "JobType", "Reporter"}


def _h2() -> qe.JobInput:
    molecule = qe.Molecule(
        [
            qe.Atom("H", (0.0, 0.0, -0.37)),
            qe.Atom("H", (0.0, 0.0, 0.37)),
        ],
        charge=0,
        multiplicity=1,
    )
    return qe.JobInput(molecule, "sto-3g", "svwn5")


def _h2o() -> qe.JobInput:
    molecule = qe.Molecule(
        [
            qe.Atom("O", (0.0, 0.0, 0.0)),
            qe.Atom("H", (0.0, 0.757, 0.586)),
            qe.Atom("H", (0.0, -0.757, 0.586)),
        ]
    )
    return qe.JobInput(molecule, "sto-3g", "svwn5")


# ---------------------------------------------------------------------------
# 接口：与冻结桩一致
# ---------------------------------------------------------------------------


def _parse_stub() -> tuple[list[str], dict[str, list[str]]]:
    text = STUB.read_text(encoding="utf-8")
    all_match = re.search(r"__all__ = \[(.*?)\]", text, re.S)
    assert all_match is not None
    names = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', all_match.group(1))
    classes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        class_match = re.match(r"class ([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if class_match:
            current = class_match.group(1)
            classes[current] = []
            continue
        if current is not None:
            if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
                current = None
                continue
            method = re.match(r"    def ([A-Za-z_][A-Za-z0-9_]*)\(", line)
            if method and method.group(1) != "__init__":
                classes[current].append(method.group(1))
    return names, classes


def test_all_matches_the_frozen_stub() -> None:
    stub_names, _ = _parse_stub()
    runtime = set(qe.__all__)
    assert runtime == set(stub_names) - TYPING_ONLY
    # 反过来：运行时不许有关键名字不在桩里
    assert runtime <= set(stub_names)


def test_classes_and_methods_exist_at_runtime() -> None:
    _, classes = _parse_stub()
    assert classes, "stub parser found no classes"
    missing: list[str] = []
    for name, methods in classes.items():
        if name in TYPING_ONLY:
            # `Reporter` 是 `Protocol`：运行时没有对象，类型面由 K6 的 mypy 门禁保证。
            continue
        target = getattr(qe, name, None)
        if target is None:
            missing.append(name)
            continue
        for method in methods:
            if not hasattr(target, method):
                missing.append(f"{name}.{method}")
    assert not missing, f"runtime is missing stub members: {missing}"


def test_exception_hierarchy_and_exit_codes() -> None:
    assert issubclass(qe.InvalidInputError, qe.EngineError)
    assert issubclass(qe.ScfNotConvergedError, qe.EngineError)
    assert issubclass(qe.DataError, qe.EngineError)
    assert issubclass(qe.UnsupportedError, qe.EngineError)

    with pytest.raises(qe.DataError) as bad_data:
        qe.Engine(str(REPO_ROOT / "does-not-exist"))
    assert bad_data.value.exit_code == 3

    engine = qe.Engine(str(DATA_ROOT))
    job = _h2()
    job.basis_name = "not-a-basis"
    with pytest.raises(qe.DataError) as bad_basis:
        engine.run(job)
    assert bad_basis.value.exit_code == 3

    job = _h2()
    job.options.scf_max_iter = 0
    with pytest.raises(qe.InvalidInputError) as bad_options:
        engine.run(job)
    assert bad_options.value.exit_code == 1

    job = _h2()
    # `freq` 还没接到 EngineBackend 上（T8 只要求 opt）→ UnsupportedError / 退出码 4
    job.jobtype = "freq"
    with pytest.raises(qe.UnsupportedError) as unsupported:
        engine.run(job)
    assert unsupported.value.exit_code == 4


def test_scf_not_converged_carries_iters() -> None:
    engine = qe.Engine(str(DATA_ROOT))
    job = _h2o()
    job.options.scf_max_iter = 1
    with pytest.raises(qe.ScfNotConvergedError) as excinfo:
        engine.run(job)
    error = excinfo.value
    assert error.exit_code == 2
    assert error.iters >= 1
    assert error.max_iters == 1


# ---------------------------------------------------------------------------
# 数值：与 CLI 逐位一致
# ---------------------------------------------------------------------------


def test_run_job_matches_the_cli() -> None:
    import subprocess
    import sys

    result = qe.run_job(_h2(), str(DATA_ROOT))
    assert result.scf_converged is True
    assert result.energy_hartree == pytest.approx(-1.1570005110367827, abs=1e-12)

    name = "qwb.exe" if sys.platform.startswith("win") else "qwb"
    binary = REPO_ROOT / "qwb-core" / "target" / "debug" / name
    if binary.is_file():
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(  # noqa: S603
                [str(binary), "run", "examples/h2_sto3g_svwn5.yaml", "--out", tmp],
                cwd=str(REPO_ROOT / "qwb-core"),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            recorded = json.loads(
                (Path(tmp) / "h2_sto3g_svwn5" / "result.json").read_text(encoding="utf-8")
            )
        assert result.energy_hartree == recorded["energy_hartree"]


def test_result_json_is_accepted_by_the_studio_model() -> None:
    result = qe.run_job(_h2(), str(DATA_ROOT))
    payload = json.loads(result.to_json())
    model = Result.model_validate(payload)
    assert model.energy_hartree == result.energy_hartree
    assert model.env.rustc_version == result.env.rustc_version

    # from_json 往返
    again = qe.JobResult.from_json(result.to_json())
    assert again.energy_hartree == result.energy_hartree
    assert again.to_json() == result.to_json()


def test_engine_env_snapshot_matches_the_generator() -> None:
    result = qe.run_job(_h2(), str(DATA_ROOT))
    generated = collect_env(REPO_ROOT)
    assert dict(result.env.data_snapshot.items()) == generated.data_snapshot
    assert result.env.cargo_lock_sha256 == generated.cargo_lock_sha256
    assert result.env.os == generated.os


# ---------------------------------------------------------------------------
# GIL 释放
# ---------------------------------------------------------------------------


def test_python_threads_keep_running_during_a_job() -> None:
    engine = qe.Engine(str(DATA_ROOT))
    job = _h2o()

    ticks = {"n": 0}
    started = threading.Event()
    stop = threading.Event()

    def counter() -> None:
        started.set()
        while not stop.is_set():
            ticks["n"] += 1

    thread = threading.Thread(target=counter, daemon=True)
    thread.start()
    assert started.wait(timeout=10)
    time.sleep(0.05)  # 让计数器先跑起来
    before = ticks["n"]

    began = time.monotonic()
    result = engine.run(job)
    elapsed = time.monotonic() - began
    stop.set()
    thread.join(timeout=10)
    during = ticks["n"] - before

    assert result.scf_converged is True
    assert elapsed > 0.05, "the job finished suspiciously fast; cannot judge GIL behaviour"
    assert during > 1000, (
        f"only {during} Python iterations ran during a {elapsed:.2f}s job: the GIL was held"
    )


def test_two_concurrent_jobs_overlap() -> None:
    """并发跑作业时墙钟远小于串行和 ⇒ GIL 被释放（pyo3 的 `allow_threads`）。

    判据的**噪声处理**：单个 H2O/STO-3G/SVWN5 只要 0.8 s 左右，两个并发的墙钟很容易被
    调度抖动吃掉 5%（2026-10-26 gate 第 50 轮实测：0.844 s 单个 vs 1.297 s 两个，
    差 2.5% 就踩线了）。所以拉成**三个**并发、并把阈值放到 0.8×串行和：
    持锁时三个并发 ≈ 3×单个（2.5 s 量级），释放后 ≈ 1.5 s 量级，判据仍然只有"释放了 GIL"
    才能过，但不再靠 2.5% 的运气。
    """
    engine = qe.Engine(str(DATA_ROOT))

    def timed(job: qe.JobInput) -> tuple[float, float]:
        began = time.monotonic()
        result = engine.run(job)
        return time.monotonic() - began, result.energy_hartree

    single, single_energy = timed(_h2o())

    outcomes: list[tuple[float, float]] = []
    lock = threading.Lock()

    def worker() -> None:
        outcome = timed(_h2o())
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
    wall = time.monotonic() - began

    assert len(outcomes) == 3
    for duration, energy in outcomes:
        assert energy == pytest.approx(single_energy, abs=1e-10)
        assert duration > 0.0
    # 串行至少是三个单次之和；释放了 GIL 就该明显更快
    serial = single * 3
    assert wall < serial * 0.8, (
        f"three concurrent runs took {wall:.2f}s vs {serial:.2f}s serial: no overlap"
    )


# ---------------------------------------------------------------------------
# Reporter 批量推送
# ---------------------------------------------------------------------------


class _CollectingReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.progress: list[tuple[str, float]] = []
        self.cancelled = False

    def log(self, line: str) -> None:
        self.lines.append(line)

    def progress(self, stage: str, pct: float) -> None:
        self.progress.append((stage, pct))

    def is_cancelled(self) -> bool:
        return self.cancelled


def test_reporter_receives_the_batched_log() -> None:
    engine = qe.Engine(str(DATA_ROOT))
    reporter = _CollectingReporter()
    result = engine.run(_h2(), reporter)
    assert result.scf_converged is True
    # 引擎日志本来就在 result.log 里；Reporter 拿到的是同一批内容的分发
    assert isinstance(result.log, str)
    # 允许引擎这一轮没有逐行日志，但只要有，就必须已经到达 reporter
    if result.log.strip():
        assert reporter.lines, "engine produced log lines but the reporter saw none"


def test_data_snapshot_and_options_behave() -> None:
    snapshot = qe.DataSnapshot()
    snapshot.insert("basis/sto-3g.yaml", "a" * 64)
    assert len(snapshot) == 1
    assert snapshot.get("basis/sto-3g.yaml") == "a" * 64
    assert snapshot.items() == [("basis/sto-3g.yaml", "a" * 64)]
    snapshot.validate()

    broken = qe.DataSnapshot()
    broken.insert("basis/sto-3g.yaml", "zzz")
    with pytest.raises(qe.DataError):
        broken.validate()

    options = qe.TaskOptions()
    assert options.scf_max_iter >= 1
    options.validate()
    options.scf_max_iter = 0
    with pytest.raises(qe.InvalidInputError):
        options.validate()

    molecule = qe.Molecule([], 0, 1)
    assert molecule.n_atoms() == 0
    with pytest.raises(qe.EngineError):
        molecule.validate()


def test_builtin_features_is_a_tuple_of_the_frozen_vocabulary() -> None:
    assert isinstance(qe.BUILTIN_REGISTRY_FEATURES, tuple)
    assert "jobtype:sp" in qe.BUILTIN_REGISTRY_FEATURES
    assert all(isinstance(item, str) for item in qe.BUILTIN_REGISTRY_FEATURES)
