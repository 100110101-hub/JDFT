"""P-01 验收测试：pydantic 模型与 K3 冻结契约逐字段一致。

测试**直接解析** `qwb-contracts/studio/*.schema.yaml` 的文本（不引入 PyYAML：
project.md §2.4 的依赖白名单里没有它），断言三件事：

1. 模型字段名集合 == 契约 `properties` 键集合（两边都不许有对方没有的字段）；
2. 契约 `required` 里的每个字段在模型里都是必填；
3. `TaskState` 的取值集合、合法转移集合、初始态与终态集合，与
   `task.schema.yaml` 的 `enum` / `x-state-machine` 逐条一致。

另外覆盖 JSON 往返、非法转移抛 `ValueError`、`id` 与 `sha256` 正则等行为。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import qwb_studio
from qwb_studio import (
    LEGAL_TRANSITIONS,
    Atom,
    EnvInfo,
    IllegalTransitionError,
    Molecule,
    Project,
    Result,
    Task,
    TaskState,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "qwb-contracts" / "studio"
TASK_SCHEMA = SCHEMA_DIR / "task.schema.yaml"
RESULT_SCHEMA = SCHEMA_DIR / "result.schema.yaml"


# ---------------------------------------------------------------------------
# 极简 YAML 读取：只认本任务需要的四种形态（流式列表 / 标量 / 属性键 / 块）
# ---------------------------------------------------------------------------


def _lines(path: Path) -> list[str]:
    assert path.is_file(), f"frozen contract missing: {path}"
    text = path.read_text(encoding="utf-8")
    return [line.rstrip() for line in text.splitlines() if not line.lstrip().startswith("#")]


def _block(lines: list[str], header: str, indent: int) -> list[str]:
    """取 `indent` 缩进下 `header:` 的子块，直到出现缩进 <= indent 的非空行。"""
    prefix = " " * indent + header + ":"
    for start, line in enumerate(lines):
        if line == prefix:
            body: list[str] = []
            for candidate in lines[start + 1 :]:
                if not candidate.strip():
                    continue
                if len(candidate) - len(candidate.lstrip()) <= indent:
                    break
                body.append(candidate)
            return body
    raise AssertionError(f"block {header!r} at indent {indent} not found")


def _keys(lines: list[str], header: str, indent: int) -> set[str]:
    """`header:` 子层里 `indent+2` 缩进的键名 —— 即该层的属性名集合。"""
    pattern = re.compile("^" + " " * (indent + 2) + r"([A-Za-z_][A-Za-z0-9_]*):")
    return {m.group(1) for line in _block(lines, header, indent) if (m := pattern.match(line))}


def _flow_list(lines: list[str], header: str, indent: int) -> list[str]:
    """`header: [a, b, c]`（允许跨行）→ `['a', 'b', 'c']`。"""
    prefix = " " * indent + header + ":"
    for start, line in enumerate(lines):
        if line.startswith(prefix):
            buf = line[len(prefix) :]
            cursor = start + 1
            while (buf.count("[") == 0 or buf.count("[") > buf.count("]")) and cursor < len(lines):
                buf += " " + lines[cursor].strip()
                cursor += 1
            match = re.search(r"\[(.*?)\]", buf, re.S)
            assert match, f"{header}: [...] not found"
            return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]
    raise AssertionError(f"{header!r} at indent {indent} not found")


def _scalar(lines: list[str], header: str, indent: int) -> str:
    prefix = " " * indent + header + ":"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("\"'")
    raise AssertionError(f"{header!r} at indent {indent} not found")


# --- task.schema.yaml -------------------------------------------------------

TASK_LINES = _lines(TASK_SCHEMA)
TASK_PROPS = _keys(TASK_LINES, "properties", 0)
TASK_REQUIRED = set(_flow_list(TASK_LINES, "required", 0))
TASK_ENUM = _flow_list(TASK_LINES, "enum", 4)
TASK_INITIAL = _scalar(TASK_LINES, "initial", 2)
TASK_TERMINAL = set(_flow_list(TASK_LINES, "terminal", 2))
TASK_TRANSITIONS = {
    (m.group(1), m.group(2))
    for line in TASK_LINES
    for m in re.finditer(r"\{from:[ ]*([a-z_]+),[ ]*to:[ ]*([a-z_]+)\}", line)
}

# --- result.schema.yaml -----------------------------------------------------

RESULT_DEFS = _block(_lines(RESULT_SCHEMA), "$defs", 0)
JOB_RESULT = _block(RESULT_DEFS, "JobResult", 2)
ENV_INFO = _block(RESULT_DEFS, "EnvInfo", 2)
JOB_PROPS = _keys(JOB_RESULT, "properties", 4)
JOB_REQUIRED = set(_flow_list(JOB_RESULT, "required", 4))
ENV_PROPS = _keys(ENV_INFO, "properties", 4)
ENV_REQUIRED = set(_flow_list(ENV_INFO, "required", 4))


# ---------------------------------------------------------------------------
# 契约解析自检：解析器自身不能悄悄解析出空集合
# ---------------------------------------------------------------------------


def test_contract_parse_sanity() -> None:
    assert TASK_PROPS == {"id", "name", "state", "created", "updated", "input"}
    assert TASK_REQUIRED == TASK_PROPS
    assert len(TASK_ENUM) == 6
    assert len(TASK_TRANSITIONS) == 6
    assert len(JOB_PROPS) >= 10
    assert JOB_REQUIRED == {"energy_hartree", "scf_converged", "data_snapshot", "log", "env"}
    assert len(ENV_PROPS) == 9
    assert len(ENV_REQUIRED) == 7


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


def test_task_state_enum_matches_contract() -> None:
    assert {state.value for state in TaskState} == set(TASK_ENUM)


def test_task_state_machine_matches_contract() -> None:
    expected = {(TaskState(frm), TaskState(to)) for frm, to in TASK_TRANSITIONS}
    assert LEGAL_TRANSITIONS == expected
    assert {frm.value for frm, _ in LEGAL_TRANSITIONS} <= set(TASK_ENUM)


def test_initial_and_terminal_match_contract() -> None:
    assert TaskState(TASK_INITIAL) is TaskState.DRAFT
    assert {state.value for state in TaskState.terminal()} == TASK_TERMINAL
    for state in TaskState:
        assert state.is_terminal() == (state.value in TASK_TERMINAL)


def test_terminal_states_have_no_outgoing_edges() -> None:
    for frm, _ in LEGAL_TRANSITIONS:
        assert not frm.is_terminal(), f"{frm.value} must be terminal but has an outgoing edge"


def test_legal_transitions_succeed() -> None:
    for frm, to in sorted(LEGAL_TRANSITIONS, key=lambda pair: (pair[0].value, pair[1].value)):
        assert frm.transition_to(to) is to


def test_illegal_transitions_raise_value_error() -> None:
    illegal = [
        (frm, to)
        for frm in TaskState
        for to in TaskState
        if (frm, to) not in LEGAL_TRANSITIONS
    ]
    # 6 状态全对：36 对，其中 6 条合法，其余 30 条必须被拒。
    assert len(illegal) == 36 - len(LEGAL_TRANSITIONS)
    for frm, to in illegal:
        with pytest.raises(ValueError):
            frm.transition_to(to)
        with pytest.raises(IllegalTransitionError):
            frm.transition_to(to)


def test_retry_semantics_failed_does_not_return_to_queued() -> None:
    """冻结语义：失败重试 = 新 task id，`failed -> queued` 必须不存在。"""
    assert (TaskState.FAILED, TaskState.QUEUED) not in LEGAL_TRANSITIONS
    assert TaskState.FAILED.is_terminal()


# ---------------------------------------------------------------------------
# 字段一致性
# ---------------------------------------------------------------------------


def test_task_fields_match_contract() -> None:
    assert set(Task.model_fields) == TASK_PROPS
    for name in TASK_REQUIRED:
        assert Task.model_fields[name].is_required(), f"Task.{name} must be required"


def test_result_fields_match_contract() -> None:
    assert set(Result.model_fields) == JOB_PROPS
    for name in JOB_REQUIRED:
        assert Result.model_fields[name].is_required(), f"Result.{name} must be required"


def test_env_info_fields_match_contract() -> None:
    assert set(EnvInfo.model_fields) == ENV_PROPS
    for name in ENV_REQUIRED:
        assert EnvInfo.model_fields[name].is_required(), f"EnvInfo.{name} must be required"


def test_public_api_is_complete() -> None:
    assert set(qwb_studio.__all__) == {
        "DATA_SNAPSHOT_FILE",
        "K4_SCHEMA_VERSION",
        "LEGAL_TRANSITIONS",
        "META_FILE",
        "Atom",
        "ContainerError",
        "EnvError",
        "EnvInfo",
        "IllegalTransitionError",
        "Molecule",
        "Project",
        "ProjectMeta",
        "QwbContainer",
        "Result",
        "ResultDirError",
        "ResultDirectory",
        "RunOutcome",
        "Task",
        "TaskState",
        "YamlError",
        "collect_env",
        "find_data_root",
        "open_result_dir",
        "read_env_json",
        "run_job_to_dir",
        "task_entry_path",
        "write_env_json",
        "yaml_dumps",
        "yaml_loads",
    }
    assert qwb_studio.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# 往返与校验行为
# ---------------------------------------------------------------------------

SAMPLE_ENV = {
    "engine_version": "0.1.0",
    "rustc_version": "rustc 1.97.1 (0e0f9c2b9 2026-01-14)",
    "cargo_lock_sha256": "a" * 64,
    "data_snapshot": {"basis/sto-3g.json": "b" * 64},
    "threads": 4,
    "os": "windows",
    "created_at_utc": "2026-02-01T00:00:00Z",
}

SAMPLE_INPUT = {
    "molecule": {
        "atoms": [
            {"symbol": "O", "position_angstrom": [0.0, 0.0, 0.0]},
            {"symbol": "H", "position_angstrom": [0.0, 0.0, 0.9584]},
        ],
        "charge": 0,
        "multiplicity": 1,
    },
    "method": {"functional": "pbe", "basis": "def2-svp"},
    "task": "energy",
}


def _task() -> Task:
    return Task.new(id="t-001", name="water pbe", input=SAMPLE_INPUT)


def test_task_json_roundtrip() -> None:
    task = _task()
    again = Task.model_validate_json(task.model_dump_json())
    assert again == task
    assert again.state is TaskState.DRAFT


def test_task_accepts_plain_dict_from_json_document() -> None:
    document = {
        "id": "abc_123-X",
        "name": "x",
        "state": "queued",
        "created": "2026-02-01T00:00:00Z",
        "updated": "2026-02-01T00:00:01Z",
        "input": {},
    }
    task = Task.model_validate(document)
    assert task.state is TaskState.QUEUED


def test_task_transition_refreshes_state_and_timestamp() -> None:
    task = _task()
    queued = task.transition_to(TaskState.QUEUED)
    assert queued.state is TaskState.QUEUED
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", queued.updated)
    assert task.state is TaskState.DRAFT  # 原对象不变（返回新对象）
    with pytest.raises(IllegalTransitionError):
        queued.transition_to(TaskState.CONVERGED)


def test_task_new_fills_draft_and_timestamps() -> None:
    task = Task.new(id="t-002", name="x", input={})
    assert task.state is TaskState.DRAFT
    assert task.created == task.updated
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", task.created)
    assert Task.model_fields["state"].is_required()  # 默认值只在 Task.new 里给


def test_task_id_pattern() -> None:
    for bad in ["", "-leading", "has space", "a/b", "a\\b", "a.b", "ümlaut"]:
        with pytest.raises(ValidationError):
            Task.new(id=bad, name="x", input={})
    Task.new(id="abc", name="x", input={})
    Task.new(id="A0_-z", name="x", input={})


def test_extra_fields_rejected() -> None:
    valid = _task().model_dump(mode="json")
    with pytest.raises(ValidationError):
        Task.model_validate({**valid, "bogus": 1})
    with pytest.raises(ValidationError):
        EnvInfo(**SAMPLE_ENV, bogus=1)


def test_env_info_sha256_pattern() -> None:
    EnvInfo(**SAMPLE_ENV)
    for bad in ["A" * 64, "a" * 63, "g" * 64]:
        with pytest.raises(ValidationError):
            EnvInfo(**{**SAMPLE_ENV, "cargo_lock_sha256": bad})
    with pytest.raises(ValidationError):
        EnvInfo(**{**SAMPLE_ENV, "threads": 0})


def test_result_json_roundtrip() -> None:
    result = Result(
        energy_hartree=-76.0266,
        scf_converged=True,
        data_snapshot={"basis/sto-3g.json": "c" * 64},
        log="iteration 0 energy -75.0\n",
        env=EnvInfo(**SAMPLE_ENV),
        gradient_hartree_per_bohr=[[0.0, 0.0, 0.0]],
        dipole_debye=[0.0, 0.0, 1.9],
    )
    again = Result.model_validate_json(result.model_dump_json())
    assert again == result
    assert again.env.threads == 4
    assert again.hessian_hartree_per_bohr2 is None


def test_result_requires_the_five_fields() -> None:
    for missing in ["energy_hartree", "scf_converged", "data_snapshot", "log", "env"]:
        payload = {
            "energy_hartree": 0.0,
            "scf_converged": True,
            "data_snapshot": {},
            "log": "",
            "env": SAMPLE_ENV,
        }
        del payload[missing]
        with pytest.raises(ValidationError):
            Result.model_validate(payload)


def test_molecule_and_project_roundtrip() -> None:
    molecule = Molecule(
        atoms=[Atom(symbol="H", position_angstrom=(0.0, 0.0, 0.74))],
        charge=0,
        multiplicity=1,
    )
    assert molecule.atoms[0].symbol == "H"
    with pytest.raises(ValidationError):
        Molecule(atoms=[])

    project = Project(name="demo", tasks=[_task()])
    again = Project.model_validate_json(project.model_dump_json())
    assert again == project
    assert again.tasks[0].id == "t-001"

    with pytest.raises(ValueError):
        project.add_task(_task())
