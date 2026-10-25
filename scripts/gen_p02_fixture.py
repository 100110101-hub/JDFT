#!/usr/bin/env python
"""P-02 夹具生成器：用 **Python 侧** `qwb_studio.container` 写一个 `.qwb`。

生成的 `qwb-studio/tests/fixtures/python_written.qwb` 会被 Rust 侧的
`qwb-core/crates/qwb-io/tests/p02_interop.rs` 打开 —— 这构成 P-02 的
"Python 写 → Rust 读" 证据。

夹具内容（刻意保持**确定性**，不含时间戳/机器信息）：

    meta.json                        # K4 ProjectMeta（task_count = 2）
    data_snapshot.json               # 真实 sha256，取自 qwb-data/registry.yaml
    tasks/t1/input.yaml              # H2 / STO-3G / SVWN5（与 qwb-core/examples 一致）
    tasks/t1/log.txt
    tasks/t2/input.yaml              # H2O / STO-3G / SVWN5
    artifacts/mo-energies.txt        # artifacts/** 也进容器

不写 result.json / env.json：结果的 provenance 必须来自真实一次 `qwb run`，
不要为了"看起来完整"而手写环境信息（P-04/P-08 会补上真实结果）。

用法（在仓库根目录，用带 pydantic 的解释器 —— 即仓库的 `.venv`）：

    .venv\\Scripts\\python.exe scripts/gen_p02_fixture.py            # 重新生成夹具
    .venv\\Scripts\\python.exe scripts/gen_p02_fixture.py --check    # 只检查夹具是否与当前写入器一致

脚本只额外依赖 `qwb_studio`（容器层与 YAML 层都是纯标准库），不引入 PyYAML。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qwb-studio"))

from qwb_studio.container import (  # noqa: E402
    DATA_SNAPSHOT_FILE,
    K4_SCHEMA_VERSION,
    META_FILE,
    ProjectMeta,
    QwbContainer,
    task_entry_path,
)
from qwb_studio.yaml_min import loads as yaml_loads  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "qwb-studio" / "tests" / "fixtures" / "python_written.qwb"

#: H2 / STO-3G / SVWN5 单点（与 qwb-core/examples/h2_sto3g_svwn5.yaml 同内容，去掉注释）。
INPUT_H2 = """\
molecule:
  atoms:
    - symbol: H
      position_angstrom: [0.0, 0.0, -0.37]
    - symbol: H
      position_angstrom: [0.0, 0.0, 0.37]
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: svwn5
grid_level: normal
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 128
  scf_energy_threshold_hartree: 1.0e-8
  scf_density_threshold: 1.0e-6
  diis_history: 8
  threads: 0
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
"""

#: H2O / STO-3G / SVWN5 单点（与 qwb-core/examples/h2o_sto3g_svwn5.yaml 同内容，去掉注释）。
INPUT_H2O = """\
molecule:
  atoms:
    - symbol: O
      position_angstrom: [0.0, 0.0, 0.0]
    - symbol: H
      position_angstrom: [0.0, 0.757, 0.586]
    - symbol: H
      position_angstrom: [0.0, -0.757, 0.586]
  charge: 0
  multiplicity: 1
basis_name: sto-3g
method_name: svwn5
grid_level: normal
solvent: vacuum
jobtype: sp
options:
  scf_max_iter: 128
  scf_energy_threshold_hartree: 1.0e-8
  scf_density_threshold: 1.0e-6
  diis_history: 8
  threads: 0
  max_opt_steps: 100
  opt_gradient_threshold_hartree_per_bohr: 4.5e-4
  freq_step_angstrom: 0.01
  ecp_name: null
"""

LOG_T1 = """\
qwb 0.1.0 · H2 / sto-3g / svwn5 / sp
scf: 12 iterations, dE = 3.1e-11 Ha
energy = -1.1570004991 Ha
"""

ARTIFACT_MO = """\
# H2 / STO-3G / SVWN5 轨道能（Hartree）
-0.5786
+0.6701
"""

#: 夹具引用的数据文件（真实摘要来自 registry.yaml，不在脚本里抄）。
SNAPSHOT_PATHS = ("basis/sto-3g.yaml", "functional/svwn5.yaml", "grid/grid-normal.yaml")

FIXTURE_META = ProjectMeta(
    schema_version=K4_SCHEMA_VERSION,
    name="python-fixture",
    created="2026-10-25T00:00:00Z",
    app_version="0.1.0",
    task_count=2,
    notes="P-02 interop fixture, written by qwb_studio.container",
)


def registry_snapshot(paths=SNAPSHOT_PATHS) -> dict[str, str]:
    """从 `qwb-data/registry.yaml` 读真实的 `path -> sha256`（用自家 YAML 读取器，顺带回归它）。"""
    registry = yaml_loads((REPO_ROOT / "qwb-data" / "registry.yaml").read_text(encoding="utf-8"))
    digests: dict[str, str] = {}
    for section in ("bases", "functionals", "grids"):
        for entry in registry[section]:
            digests[entry["path"]] = entry["sha256"]
    snapshot: dict[str, str] = {}
    for path in paths:
        if path not in digests:
            raise SystemExit(f"registry.yaml has no entry for {path!r}")
        digest = digests[path]
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"registry digest for {path!r} is not a 64-char sha256: {digest!r}")
        snapshot[path] = digest
    return snapshot


def payload() -> tuple[ProjectMeta, list[tuple[str, bytes]]]:
    """夹具的确定性内容：`(meta, [(entry_name, bytes)])`。"""
    snapshot_json = "{" + ", ".join(f'"{p}": "{h}"' for p, h in registry_snapshot().items()) + "}\n"
    files: list[tuple[str, bytes]] = [
        (task_entry_path("t1", "input.yaml"), INPUT_H2.encode("utf-8")),
        (task_entry_path("t1", "log.txt"), LOG_T1.encode("utf-8")),
        (task_entry_path("t2", "input.yaml"), INPUT_H2O.encode("utf-8")),
        (DATA_SNAPSHOT_FILE, snapshot_json.encode("utf-8")),
        ("artifacts/mo-energies.txt", ARTIFACT_MO.encode("utf-8")),
    ]
    return FIXTURE_META, files


def write_fixture(path: Path = FIXTURE_PATH) -> Path:
    """按当前写入器重新生成夹具。"""
    meta, files = payload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    container = QwbContainer.create(path, meta)
    try:
        container.write_files(files)
    finally:
        container.close()
    return path


def fixture_matches(path: Path = FIXTURE_PATH) -> tuple[bool, str]:
    """夹具与当前写入器的输出是否一致（`--check` / 测试用）。"""
    path = Path(path)
    if not path.is_file():
        return False, f"missing fixture {path}: run `python scripts/gen_p02_fixture.py`"
    meta, files = payload()
    with QwbContainer.open(path) as committed:
        if committed.meta != meta:
            return False, f"fixture meta drifted: {committed.meta!r} != {meta!r}"
        expected = {META_FILE: meta.to_json_bytes(), **{name: data for name, data in files}}
        actual = dict(committed.files())
    if set(actual) != set(expected):
        return False, (
            "fixture entries drifted: "
            f"committed={sorted(actual)} expected={sorted(expected)}; "
            "run `python scripts/gen_p02_fixture.py`"
        )
    for name in expected:
        if actual[name] != expected[name]:
            return False, f"fixture entry {name!r} drifted; run `python scripts/gen_p02_fixture.py`"
    return True, "fixture matches the current writer"


def main(argv: list[str]) -> int:
    if "--check" in argv[1:]:
        ok, detail = fixture_matches()
        print(("ok: " if ok else "FAILED: ") + detail)
        return 0 if ok else 1
    path = write_fixture()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    with QwbContainer.open(path) as container:
        container.validate()
        print("entries: " + ", ".join(container.entry_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
