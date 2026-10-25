"""P-01 验收：YAML 往返（`qwb_studio.yaml_min`）。

依赖白名单（project.md §2.4）里没有 PyYAML，所以往返必须是"自持"的：
模型 → `yaml_dumps` → `yaml_loads` → 模型，逐字段相等；且输出必须稳定
（再跑一遍得到逐字节相同的文本）。同时验证越界输入**显式报错**，
不许静默产出错误的树。
"""

from __future__ import annotations

import pytest

from qwb_studio import (
    Atom,
    EnvInfo,
    Molecule,
    Project,
    Result,
    Task,
    YamlError,
    yaml_dumps,
    yaml_loads,
)

SAMPLE_ENV = {
    "engine_version": "0.1.0",
    "rustc_version": "rustc 1.97.1 (0e0f9c2b9 2026-01-14)",
    "cargo_lock_sha256": "a" * 64,
    "data_snapshot": {"basis/sto-3g.json": "b" * 64, "xc/pbe.json": "c" * 64},
    "threads": 4,
    "os": "windows",
    "created_at_utc": "2026-02-01T00:00:00Z",
}

SAMPLE_INPUT = {
    "molecule": {
        "atoms": [
            {"symbol": "O", "position_angstrom": [0.0, 0.0, 0.0]},
            {"symbol": "H", "position_angstrom": [0.0, 0.0, 0.9584]},
            {"symbol": "H", "position_angstrom": [0.0, 0.9266, -0.2433]},
        ],
        "charge": 0,
        "multiplicity": 1,
    },
    "method": {"functional": "pbe", "basis": "def2-svp"},
    "scf": {"max_iter": 200, "energy_tol": 1e-8},
    "task": "energy",
}


def _task() -> Task:
    return Task.new(id="t-001", name="water pbe", input=SAMPLE_INPUT)


def _result() -> Result:
    return Result(
        energy_hartree=-76.02661168,
        scf_converged=True,
        data_snapshot={"basis/sto-3g.json": "d" * 64},
        log="iter 0 E=-75.000000000000\niter 1 E=-76.026600000000\n",
        env=EnvInfo(**SAMPLE_ENV),
        gradient_hartree_per_bohr=[[0.0, 0.0, 1e-9], [0.0, 0.0, -1e-9], [0.0, 0.0, 0.0]],
        dipole_debye=[0.0, 0.0, 1.9212],
        mo_energies_hartree=[-20.55, -1.32, -0.61, -0.44, 0.28],
    )


def test_task_yaml_roundtrip() -> None:
    task = _task()
    text = yaml_dumps(task.model_dump(mode="json"))
    again = Task.model_validate(yaml_loads(text))
    assert again == task
    assert again.input == SAMPLE_INPUT


def test_result_yaml_roundtrip() -> None:
    result = _result()
    text = yaml_dumps(result.model_dump(mode="json"))
    again = Result.model_validate(yaml_loads(text))
    assert again == result
    assert again.env.data_snapshot == SAMPLE_ENV["data_snapshot"]


def test_project_yaml_roundtrip() -> None:
    project = Project(name="demo", tasks=[_task()])
    text = yaml_dumps(project.model_dump(mode="json"))
    again = Project.model_validate(yaml_loads(text))
    assert again == project


def test_yaml_output_is_stable_and_idempotent() -> None:
    payload = _result().model_dump(mode="json")
    first = yaml_dumps(payload)
    assert yaml_dumps(yaml_loads(first)) == first


def test_yaml_preserves_scalar_types() -> None:
    tree = yaml_loads(yaml_dumps(_result().model_dump(mode="json")))
    assert isinstance(tree["energy_hartree"], float)
    assert isinstance(tree["scf_converged"], bool)
    assert isinstance(tree["env"]["threads"], int)
    assert tree["env"]["threads"] == 4
    assert tree["hessian_hartree_per_bohr2"] is None


def test_yaml_roundtrips_tricky_strings() -> None:
    payload = {
        "plain": "pbe",
        "spaces": "rustc 1.97.1 (0e0f9c2b9 2026-01-14)",
        "newline": "a\nb\n",
        "quotes": 'he said "hi"',
        "backslash": r"C:\tmp\x",
        "hash": "a # not a comment",
        "colon": "key: value",
        "empty": "",
        "looks_bool": "true",
        "looks_int": "42",
        "unicode": "水分子",
    }
    assert yaml_loads(yaml_dumps(payload)) == payload


def test_yaml_roundtrips_numbers() -> None:
    payload = {
        "zero": 0,
        "neg": -17,
        "big": 1234567890123,
        "tiny": 1e-9,
        "neg_exp": -2.5e-8,
        "pi": 3.141592653589793,
        "one": 1.0,
    }
    assert yaml_loads(yaml_dumps(payload)) == payload


def test_yaml_empty_containers() -> None:
    payload = {"empty_map": {}, "empty_list": [], "nested": {"a": {"b": []}}}
    assert yaml_loads(yaml_dumps(payload)) == payload


def test_yaml_reads_handwritten_document() -> None:
    text = """\
# 手写样例：缩进、注释、引号、空值、缩进式序列
id: t-001
name: 'water #1'
state: queued
created: "2026-02-01T00:00:00Z"
input:
  method:
    functional: pbe    # 行内注释
    basis: def2-svp
  atoms:
    -
      symbol: O
      z: 0.0
    -
      symbol: H
      z: 0.9584
  notes: null
  flags: []
"""
    tree = yaml_loads(text)
    assert tree["id"] == "t-001"
    assert tree["name"] == "water #1"
    assert tree["input"]["method"]["functional"] == "pbe"
    assert tree["input"]["atoms"] == [{"symbol": "O", "z": 0.0}, {"symbol": "H", "z": 0.9584}]
    assert tree["input"]["notes"] is None
    assert tree["input"]["flags"] == []


def test_yaml_reads_compact_sequence_items() -> None:
    tree = yaml_loads("atoms:\n  - O\n  - H\n")
    assert tree == {"atoms": ["O", "H"]}


def test_yaml_rejects_out_of_subset_input() -> None:
    cases = {
        "anchors": "a: &x 1\n",
        "aliases": "a: *x\n",
        "tags": "a: !!int 1\n",
        "flow_seq": "a: [1, 2]\n",
        "flow_map": "a: {b: 1}\n",
        "block_scalar": "a: |\n  text\n",
        "multi_document": "---\na: 1\n",
        "tabs": "\ta: 1\n",
        "duplicate_key": "a: 1\na: 2\n",
        "no_colon": "a\n",
        "unterminated_quote": 'a: "x\n',
        "bad_escape": 'a: "x\\q"\n',
    }
    for label, text in cases.items():
        with pytest.raises(YamlError):
            yaml_loads(text)


def test_yaml_tolerates_empty_document() -> None:
    assert yaml_loads("") is None
    assert yaml_loads("\n# only a comment\n") is None


def test_molecule_yaml_roundtrip() -> None:
    molecule = Molecule(
        atoms=[
            Atom(symbol="H", position_angstrom=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position_angstrom=(0.0, 0.0, 0.7414)),
        ],
        charge=0,
        multiplicity=1,
    )
    again = Molecule.model_validate(yaml_loads(yaml_dumps(molecule.model_dump(mode="json"))))
    assert again == molecule
    assert again.atoms[1].position_angstrom == (0.0, 0.0, 0.7414)


SERDE_STYLE_DOCUMENT = """\
molecule:
  atoms:
  - symbol: O
    position_angstrom:
    - 0.0
    - 0.0
    - 0.0
  - symbol: H
    position_angstrom:
    - 0.0
    - 0.757
    - 0.586
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: svwn5
grid_level: normal
options:
  scf_max_iter: 128
  ecp_name: null
"""


def test_parses_serde_yaml_style_documents() -> None:
    """Rust 侧 `serde_yaml` 写出的写法：紧凑序列项 + **indentless sequence**。

    引擎写进结果目录的 `input.yaml` 就是这种形态（`atoms:` 下面直接顶格写 `- symbol: O`），
    解析器不认它就读不了自己的产物 —— P-10 的 3D 视图正是踩在这里之后补上的。
    """
    document = yaml_loads(SERDE_STYLE_DOCUMENT)
    molecule = document["molecule"]
    assert molecule["charge"] == 0
    assert molecule["multiplicity"] == 1
    assert [atom["symbol"] for atom in molecule["atoms"]] == ["O", "H"]
    assert molecule["atoms"][1]["position_angstrom"] == [0.0, 0.757, 0.586]
    assert document["options"]["scf_max_iter"] == 128
    assert document["options"]["ecp_name"] is None
