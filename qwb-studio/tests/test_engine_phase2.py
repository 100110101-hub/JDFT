"""PR-02 验收：Python 绑定（pyo3）必须暴露阶段二（KV-05）新增的选项与结果块。

判据（`project-phase2.md` §6.9 的 PR-02 一行，拆开写）：

1. **`mypy --strict` 通过**：桩 `qwb_engine.pyi` 已加性扩展（新类 + 新字段），
   gate 里的两条 `mypy --strict` 就是这条判据；
2. **与 CLI 能量逐位相同**：H2O/STO-3G/HF + TDA 的 Python 结果，与仓库里提交的
   CLI 产物 `qwb-core/examples/results/h2o_tddft/result.json` 的 `energy_hartree`
   **逐位相同**（`==`，不是 approx）；
3. **GIL 仍释放**：`test_engine_binding.py` 的三并发用例已经在跑；这里再补一条
   "带新选项的作业并发跑也互相重叠"，确保新代码路径没有在 Rust 里重新抓 GIL；
4. **非法取值在入口就报错**（`InvalidInputError` / K5 退出码 1），而不是拖到引擎里。

另外钉住"缺块 = `None`"：阶段一的旧作业不会有激发态/谱/相关能，读它们不能炸。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import qwb_engine as qe

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "qwb-data"
TDDT_FIXTURE = REPO_ROOT / "qwb-core" / "examples" / "results" / "h2o_tddft" / "result.json"

#: 与 `qwb-core/examples/h2o_tddft.yaml` 同一几何（这样能量才能与提交的产物逐位比）。
H2O = [
    ("O", (0.0, 0.0, 0.11779)),
    ("H", (0.0, 0.75545, -0.47116)),
    ("H", (0.0, -0.75545, -0.47116)),
]


def _h2o(options: qe.TaskOptions | None = None, method: str = "hf") -> qe.JobInput:
    molecule = qe.Molecule(
        [qe.Atom(symbol, position) for symbol, position in H2O], charge=0, multiplicity=1
    )
    return qe.JobInput(molecule, "sto-3g", method, options=options)


def _tda_options() -> qe.TaskOptions:
    options = qe.TaskOptions()
    options.scf_max_iter = 200
    options.scf_energy_threshold_hartree = 1.0e-11
    options.scf_density_threshold = 1.0e-9
    options.diis_history = 8
    options.excited_states = qe.ExcitedStatesOptions(n_states=5, method="tda")
    return options


# ---------------------------------------------------------------------------
# 1. 新选项类：存在、可构造、名字与 K2 的 serde 拼写一致
# ---------------------------------------------------------------------------


def test_new_option_classes_are_exported_and_constructible() -> None:
    for name in ("Mp2Options", "ExcitedStatesOptions", "NumericsOptions"):
        assert name in qe.__all__, f"{name} 必须出现在 __all__（import * 也要能用）"
        assert hasattr(qe, name)

    mp2 = qe.Mp2Options()
    assert mp2.kind == "mp2"
    assert mp2.aux_basis_name is None
    mp2.validate()

    ri = qe.Mp2Options(kind="ri-mp2", aux_basis_name="def2-universal-jkfit")
    assert ri.kind == "ri-mp2"
    ri.validate()

    excited = qe.ExcitedStatesOptions(n_states=3)
    assert excited.n_states == 3
    assert excited.method == "tda"  # 默认就是 phase-2 定的 TDA
    excited.validate()

    numerics = qe.NumericsOptions()
    assert numerics.path is None  # None = 解析路径（阶段一行为）
    assert 1 <= numerics.accuracy_level <= 5
    numerics.validate()

    options = qe.TaskOptions()
    assert options.scf_type is None
    assert options.mp2 is None
    assert options.excited_states is None
    assert options.solvent_model is None
    assert options.numerics is None


def test_invalid_option_values_fail_at_the_door() -> None:
    with pytest.raises(qe.InvalidInputError):
        qe.Mp2Options(kind="mp2-but-typo")
    with pytest.raises(qe.InvalidInputError):
        qe.ExcitedStatesOptions(n_states=5, method="adc3")
    with pytest.raises(qe.InvalidInputError):
        qe.NumericsOptions(accuracy_level=9)
    with pytest.raises(qe.InvalidInputError):
        qe.ExcitedStatesOptions(n_states=0).validate()

    options = qe.TaskOptions()
    options.scf_type = "rhf-but-typo"
    with pytest.raises(qe.InvalidInputError):
        options.validate()
    options.scf_type = None
    options.solvent_model = "water"
    with pytest.raises(qe.InvalidInputError):
        options.validate()


# ---------------------------------------------------------------------------
# 2. 数值：与 CLI 产物逐位相同
# ---------------------------------------------------------------------------


def test_excited_states_match_the_cli_product_bit_for_bit() -> None:
    if not TDDT_FIXTURE.is_file():
        pytest.skip("examples/results/h2o_tddft 不在仓库里")
    recorded = json.loads(TDDT_FIXTURE.read_text(encoding="utf-8"))

    result = qe.run_job(_h2o(_tda_options()), str(DATA_ROOT))
    assert result.scf_converged is True
    # 逐位相同：这里必须是 `==`，不能用 approx 掩盖差异
    assert result.energy_hartree == recorded["energy_hartree"]

    block = result.excited_states
    assert block is not None, "带 options.excited_states 的作业必须出 excited_states 块"
    transitions = block["transitions"]
    assert len(transitions) == 5
    # 块里的键名与 result.json 一致（同一份数据的两条通道）。
    # 容差 1e-12 Ha 而不是 `==`：仓库里这份 CLI 产物是 RS-03 收尾**之前**的一次快照，
    # 与今天的 TDA 特征分解差最后两位（相对 4e-15）；SCF 能量本身仍是逐位相同的
    # （上面那条 `==`）。若把这里也写成 `==`，就是在要求"历史快照不许有数值演进"。
    recorded_transitions = recorded["excited_states"]["transitions"]
    for ours, theirs in zip(transitions, recorded_transitions):
        assert ours["index"] == theirs["index"]
        assert ours["energy_hartree"] == pytest.approx(
            theirs["energy_hartree"], abs=1e-12
        )
        assert ours["oscillator_strength"] == pytest.approx(
            theirs["oscillator_strength"], abs=1e-12
        )
    # 展宽谱也是同一个块（PR-04 的通道经 pyo3 原样透出）。仓库里提交的那份产物是
    # PR-04 **之前**的快照，没有这个块 —— 所以只在它有时才比对点数。
    spectrum = result.uv_vis_spectrum
    assert spectrum is not None
    assert len(spectrum["points"]) > 0
    if "uv_vis_spectrum" in recorded:
        assert len(spectrum["points"]) == len(recorded["uv_vis_spectrum"]["points"])


def test_explicit_default_options_are_bit_identical_to_the_default() -> None:
    """把 `scf_type` / `solvent_model` 显式写成默认值，能量必须逐位不变（加性扩展）。"""
    default = qe.run_job(_h2o(), str(DATA_ROOT))

    options = qe.TaskOptions()
    options.scf_type = "rhf"
    options.solvent_model = "vacuum"
    options.validate()
    explicit = qe.run_job(_h2o(options), str(DATA_ROOT))

    assert explicit.energy_hartree == default.energy_hartree


def test_dispersion_is_an_additive_term_that_lands_in_the_result() -> None:
    """FX-04：`options.dispersion` 是**可加项**，两部分都落盘。

    判据要的是「`dispersion_energy_hartree` 进 result.json」，所以这里断言：
    ① 不给选项时是 `None`（阶段一回归）；② 给了之后总能量 = SCF 能量 + 色散能；
    ③ 色散能是负的且在合理量级（水分子 + STO-3G 是 −1e-4 Ha 量级）。
    """
    baseline = qe.run_job(_h2o(), str(DATA_ROOT))
    assert baseline.dispersion_energy_hartree is None

    options = qe.TaskOptions()
    options.dispersion = qe.DispersionOptions()
    options.validate()
    result = qe.run_job(_h2o(options), str(DATA_ROOT))
    dispersion = result.dispersion_energy_hartree
    assert dispersion is not None
    # 量级：水分子（含分子内 O-H/H-H 对）+ STO-3G + **HF** 的 D3(BJ) 实测 −4.51e-3 Ha
    # （≈ −2.83 kcal/mol）。绝对值由 `qwb-scf` 的 fx04 测试对官方实现钉住，
    # 这里只挡「符号反了」或「差了几个数量级」。
    assert -1.0e-2 < dispersion < 0.0
    assert abs(result.energy_hartree - (baseline.energy_hartree + dispersion)) < 1.0e-12


def test_numerics_options_are_declared_but_not_wired_yet() -> None:
    """`options.numerics` 已经在 K2/K6 里**声明**，但引擎还没接（IN-03/IN-04 未做）。

    这条测试存在的意义：不让"桩里有、引擎没接"变成沉默的谎言 —— 传了就报
    `UnsupportedError`（K5 退出码 4），而不是被悄悄忽略。
    """
    options = qe.TaskOptions()
    options.numerics = qe.NumericsOptions(accuracy_level=3, path="analytic")
    options.validate()
    with pytest.raises(qe.UnsupportedError) as excinfo:
        qe.run_job(_h2o(options), str(DATA_ROOT))
    assert excinfo.value.exit_code == 4
    assert "numerics" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. 缺块 = None（阶段一旧作业）
# ---------------------------------------------------------------------------


def test_absent_blocks_are_none() -> None:
    result = qe.run_job(_h2o(), str(DATA_ROOT))
    assert result.excited_states is None
    assert result.uv_vis_spectrum is None
    assert result.correlation is None
    assert result.solvation is None
    assert result.nmr_shieldings is None
    assert result.polarizability is None
    assert result.dispersion_energy_hartree is None


def test_blocks_survive_a_json_round_trip() -> None:
    """`result.json` 往返后块还在、量级不变。

    注意容差是 1e-13（相对）：`serde_json` 的浮点解析默认**不是**逐位精确的
    （要 `float_roundtrip` feature 才是；这里没开，K2 也没要求过），所以
    `to_json()` → `from_json()` → `to_json()` 会在最后一位上漂。这条测试钉的是
    "块的结构与数值都还在"，不是"文本逐字节相同"。
    """
    result = qe.run_job(_h2o(_tda_options()), str(DATA_ROOT))
    again = qe.JobResult.from_json(result.to_json())
    assert again.excited_states is not None
    assert again.uv_vis_spectrum is not None
    ours = again.excited_states["transitions"]
    theirs = result.excited_states["transitions"]
    assert [t["index"] for t in ours] == [t["index"] for t in theirs]
    for a, b in zip(ours, theirs):
        assert a["energy_hartree"] == pytest.approx(b["energy_hartree"], rel=1e-13)
    assert len(again.uv_vis_spectrum["points"]) == len(result.uv_vis_spectrum["points"])
    assert again.dispersion_energy_hartree is None


# ---------------------------------------------------------------------------
# 4. GIL：带新选项的作业并发跑也重叠
# ---------------------------------------------------------------------------


def test_concurrent_jobs_with_new_options_overlap() -> None:
    engine = qe.Engine(str(DATA_ROOT))

    def timed() -> tuple[float, float]:
        job = _h2o(_tda_options())
        began = time.monotonic()
        result = engine.run(job)
        return time.monotonic() - began, result.energy_hartree

    single, single_energy = timed()

    outcomes: list[tuple[float, float]] = []
    lock = threading.Lock()

    def worker() -> None:
        outcome = timed()
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
    wall = time.monotonic() - began

    assert len(outcomes) == 2
    for _, energy in outcomes:
        assert energy == pytest.approx(single_energy, abs=1e-12)
    serial = single * 2
    assert wall < serial * 0.95, (
        f"two concurrent TDA runs took {wall:.2f}s vs {serial:.2f}s serial: no overlap"
    )
