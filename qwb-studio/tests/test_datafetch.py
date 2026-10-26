"""第三方数据的按需取回（`qwb_studio.datafetch` / `bse_data`）。

判据（对应 §14 #20 的分发约束）：
  * **仓库不提交** BSE 导出的基组/ECP，只提交配方（`qwb-data/sources.json`）与摘要钉；
  * 取回的文件必须**逐项重现**摘要钉 —— 否则拒绝落盘（上游改了、或转录规则漂移了）；
  * 离线也能验：BSE 的导出夹具放在 `qwb-core/crates/qwb-registry/tests/fixtures/bse/`，
    `QWB_BSE_FIXTURES` 指向它时完全不联网。

本文件不复制任何转录规则：断言的对象是 `bse_data` 的**行为**（哪些 region 能过、
局域通道取哪个 l、`r^n` 约定、摘要是否与钉相等）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qwb_studio import datafetch
from qwb_studio.bse_data import (
    TranscriptionError,
    split_shells,
    transcribe_basis,
    transcribe_ecp,
)
from qwb_studio.datafetch import DataFetchError, check_entries, ensure_for_input, fetch_entries

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "qwb-data"
FIXTURES = REPO / "qwb-core" / "crates" / "qwb-registry" / "tests" / "fixtures" / "bse"


# ---------------------------------------------------------------------------
# 转录：region 标签与壳层拆分
# ---------------------------------------------------------------------------


def shell(region, angular=(0,), columns=None) -> dict:
    """一个最小 `electron_shell`（两列指数、每列两行）。"""
    columns = columns if columns is not None else [(1.0, 0.5)] * len(angular)
    return {
        "function_type": "gto",
        "region": region,
        "angular_momentum": list(angular),
        "exponents": ["3.0", "1.0"],
        "coefficients": [list(column) for column in columns],
    }


@pytest.mark.parametrize("region", [None, "", "valence", "polarization", "unknown"])
def test_split_shells_accepts_every_full_range_region(region):
    """`region` 只是标签，不是"被裁剪过"的证据。"""
    shells = split_shells(shell(region))
    assert len(shells) == 1
    assert shells[0]["l"] == 0


@pytest.mark.parametrize("region", ["core", "semicore"])
def test_split_shells_rejects_cropped_exports(region):
    with pytest.raises(TranscriptionError, match="unsupported region"):
        split_shells(shell(region))


def test_split_shells_rejects_non_gto():
    entry = shell("valence")
    entry["function_type"] = "sto"
    with pytest.raises(TranscriptionError, match="function_type"):
        split_shells(entry)


def test_split_shells_ragged_columns():
    entry = shell("valence", angular=(0,))
    entry["coefficients"] = [[1.0]]
    with pytest.raises(TranscriptionError, match="coefficient column length"):
        split_shells(entry)


def test_split_shells_unmappable_shape():
    """角动量两项、系数三列：既不是逐 l 一列，也不是通用收缩 → 拒绝而不是猜。"""
    entry = shell("valence", angular=(0, 1), columns=[(1.0, 0.5)] * 3)
    with pytest.raises(TranscriptionError, match="cannot map"):
        split_shells(entry)


def test_split_shells_general_contraction_keeps_columns():
    entry = shell("valence", angular=(0,), columns=[(1.0, 0.5), (0.3, 0.2)])
    shells = split_shells(entry)
    assert len(shells) == 1
    assert shells[0]["columns"] == [[1.0, 0.5], [0.3, 0.2]]


# ---------------------------------------------------------------------------
# 转录：基组 YAML（元数据只来自 spec）
# ---------------------------------------------------------------------------


def basis_spec(**overrides) -> dict:
    spec = {
        "name": "demo-basis",
        "version": "1.2.3",
        "notes": "demo",
        "citations": [{"title": "A Paper", "doi": "10.1000/x"}],
        "license": {"spdx": "Apache-2.0", "commercial_ok": True, "redistribute_ok": False},
        "created": "2026-01-01T00:00:00Z",
        "angular": "spherical",
        "normalization": "l2",
        "role": "orbital",
    }
    spec.update(overrides)
    return spec


def basis_payload() -> dict:
    return {
        "elements": {
            "1": {"electron_shells": [shell("valence", angular=(0,))]},
            "8": {"electron_shells": [shell("valence", angular=(0,)), shell("polarization", angular=(1,))]},
        }
    }


def test_transcribe_basis_uses_spec_metadata():
    text = transcribe_basis(basis_spec(), basis_payload())
    assert "name: demo-basis" in text
    assert "version: 1.2.3" in text
    assert "spdx: Apache-2.0" in text
    assert "redistribute_ok: false" in text
    assert 'doi: "10.1000/x"' in text
    # `elements:` 必须是块列表：`data_catalog` 只按顶层 `elements:` 的 `- X` 行扫，
    # GUI 的元素下拉框靠它。摘要按解析后的内容算，所以写错形态**不会**被钉拦住。
    assert "elements: [H, O]" not in text
    assert "elements:\n  - H\n  - O\n" in text
    # 逐壳层判定收缩类型：一列系数 = segmented
    assert text.count("contraction_type: segmented") == 3
    # 没有 sha256：摘要只由 Rust 的 `qwb data sign` 打
    assert "sha256:" not in text


def test_transcribe_basis_marks_general_contraction():
    payload = {"elements": {"1": {"electron_shells": [
        shell("valence", angular=(0,), columns=[(1.0, 0.5), (0.3, 0.2)])
    ]}}}
    text = transcribe_basis(basis_spec(), payload)
    assert "contraction_type: general" in text


def test_transcribe_basis_rejects_unknown_element():
    payload = {"elements": {"54": {"electron_shells": [shell("valence")]}}}
    with pytest.raises(TranscriptionError, match="H..Kr"):
        transcribe_basis(basis_spec(), payload)


# ---------------------------------------------------------------------------
# 转录：ECP
# ---------------------------------------------------------------------------


def ecp_payload() -> dict:
    return {
        "elements": {
            "29": {
                "ecp_electrons": 10,
                "ecp_potentials": [
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [0],
                        "r_exponents": [0, 1],
                        "gaussian_exponents": ["10.0", "2.0"],
                        "coefficients": [["3.0", "1.5"]],
                    },
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [2],
                        "r_exponents": [2],
                        "gaussian_exponents": ["1.0"],
                        "coefficients": [["-0.5"]],
                    },
                ],
            }
        }
    }


def test_transcribe_ecp_local_channel_is_highest_l():
    text = transcribe_ecp({"name": "demo-ecp", "element": "Cu", **basis_spec()}, ecp_payload())
    assert "element: Cu" in text
    assert "n_core_electrons: 10" in text
    assert "z_eff: 19.0" in text
    assert 'r_power_convention: "r^n"' in text
    # 局域通道是 l=2（最高 l），l=0 是投影
    assert 'l=2 (BSE local channel' in text
    assert "- l: 0" in text


def test_transcribe_ecp_rejects_unknown_element_symbol():
    with pytest.raises(TranscriptionError, match="不在 H–Kr"):
        transcribe_ecp({"name": "x", "element": "Xx", **basis_spec()}, ecp_payload())


# ---------------------------------------------------------------------------
# 清单状态：ok / missing / stale
# ---------------------------------------------------------------------------


def write_sources(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources.json").write_text(
        json.dumps({"schema_version": 1, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_check_entries_states(tmp_path):
    root = tmp_path / "qwb-data"
    pin_ok, pin_bad = "a" * 64, "b" * 64
    write_sources(
        root,
        [
            {"path": "basis/ok.yaml", "sha256": pin_ok},
            {"path": "basis/missing.yaml", "sha256": pin_ok},
            {"path": "basis/stale.yaml", "sha256": pin_bad},
        ],
    )
    (root / "basis").mkdir()
    (root / "basis" / "ok.yaml").write_text(f"name: ok\nsha256: {pin_ok}\n", encoding="utf-8")
    (root / "basis" / "stale.yaml").write_text(f"name: stale\nsha256: {pin_ok}\n", encoding="utf-8")

    states = {status.path: status.state for status in check_entries(root)}
    assert states == {
        "basis/ok.yaml": "ok",
        "basis/missing.yaml": "missing",
        "basis/stale.yaml": "stale",
    }
    only = check_entries(root, only=["basis/missing.yaml"])
    assert [status.path for status in only] == ["basis/missing.yaml"]


def test_find_data_root_prefers_explicit_then_env(tmp_path, monkeypatch):
    root = tmp_path / "qwb-data"
    write_sources(root, [{"path": "basis/x.yaml", "sha256": "a" * 64}])
    (root / "basis").mkdir()
    explicit = datafetch.find_data_root(explicit=root)
    assert explicit == root

    monkeypatch.setenv("QWB_DATA_ROOT", str(root))
    assert datafetch.find_data_root() == root

    monkeypatch.delenv("QWB_DATA_ROOT")
    monkeypatch.chdir(root / "basis")
    assert datafetch.find_data_root() == root


# ---------------------------------------------------------------------------
# 按 input.yaml 取数
# ---------------------------------------------------------------------------


def test_ensure_for_input_requests_only_listed_entries(tmp_path, monkeypatch):
    """CuH 的输入只需要 LANL2DZ 基组 + Cu 的 ECP；网格/泛函是自研数据，不该触发下载。"""
    root = tmp_path / "qwb-data"
    root.mkdir(parents=True)
    shutil.copy(DATA / "sources.json", root / "sources.json")
    input_yaml = REPO / "qwb-core" / "examples" / "cuh_lanl2dz_pbe.yaml"
    assert input_yaml.is_file()

    requested: list[list[str]] = []

    def fake_fetch(data_root, *, only=None, force=False, log=print):
        requested.append(list(only or []))
        log("fake fetch")
        return {path: "0" * 64 for path in only or []}

    monkeypatch.setattr(datafetch, "fetch_entries", fake_fetch)
    pending = ensure_for_input(input_yaml, data_root=root)
    assert sorted(pending) == ["basis/lanl2dz.yaml", "ecp/lanl2dz-cu.yaml"]
    assert requested == [pending]


def test_ensure_for_input_returns_empty_when_everything_is_present(tmp_path, monkeypatch):
    root = tmp_path / "qwb-data"
    root.mkdir(parents=True)
    shutil.copy(DATA / "sources.json", root / "sources.json")
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    for entry in sources["entries"]:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"name: x\nsha256: {entry['sha256']}\n", encoding="utf-8")

    def boom(*args, **kwargs):  # pragma: no cover - 不该被调用
        raise AssertionError("数据齐了还调了 fetch_entries")

    monkeypatch.setattr(datafetch, "fetch_entries", boom)
    input_yaml = REPO / "qwb-core" / "examples" / "cuh_lanl2dz_pbe.yaml"
    assert ensure_for_input(input_yaml, data_root=root) == []


# ---------------------------------------------------------------------------
# 端到端：离线夹具 → 摘要必须等于钉
# ---------------------------------------------------------------------------


def engine_binary() -> Path | None:
    """打摘要要 Rust 的 `qwb data sign`（K1 摘要规则的唯一实现）。"""
    explicit = os.environ.get("QWB_ENGINE_BIN")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for relative in ("target/debug/qwb.exe", "target/debug/qwb", "target/release/qwb.exe", "target/release/qwb"):
        candidate = REPO / "qwb-core" / relative
        if candidate.is_file():
            return candidate
    return None


def test_offline_fetch_reproduces_the_pinned_digest(tmp_path, monkeypatch):
    engine = engine_binary()
    if engine is None:
        pytest.skip("需要已构建的 qwb 二进制（cargo build -p qwb-cli）来打 K1 摘要")
    if not (FIXTURES / "sto-3g.json").is_file():
        pytest.skip("缺 BSE 导出夹具")

    pin = _pin_for("basis/sto-3g.yaml")
    root = tmp_path / "qwb-data"
    write_sources(root, [{"path": "basis/sto-3g.yaml", "sha256": pin, "kind": "basis",
                          "fetch": {"basis": "sto-3g", "elements": [1, 6, 7, 8, 9]},
                          "transcribe": _spec_for("basis/sto-3g.yaml")}])

    monkeypatch.setenv("QWB_BSE_FIXTURES", str(FIXTURES))
    monkeypatch.setenv("QWB_ENGINE_BIN", str(engine))
    fetched = fetch_entries(root, log=lambda line: None)
    assert fetched == {"basis/sto-3g.yaml": pin}
    assert check_entries(root)[0].state == "ok"


def test_offline_fetch_refuses_a_drifted_export(tmp_path, monkeypatch):
    """上游/夹具改了数字 → 摘要对不上 → 报错且**不落盘**（不能把坏数据写进数据根）。"""
    engine = engine_binary()
    if engine is None:
        pytest.skip("需要已构建的 qwb 二进制（cargo build -p qwb-cli）来打 K1 摘要")
    payload = json.loads((FIXTURES / "sto-3g.json").read_text(encoding="utf-8"))
    hydrogen = payload["elements"]["1"]["electron_shells"][0]
    hydrogen["exponents"][0] = "9.9999999"

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "sto-3g.json").write_text(json.dumps(payload), encoding="utf-8")

    pin = _pin_for("basis/sto-3g.yaml")
    root = tmp_path / "qwb-data"
    write_sources(root, [{"path": "basis/sto-3g.yaml", "sha256": pin, "kind": "basis",
                          "fetch": {"basis": "sto-3g", "elements": [1, 6, 7, 8, 9]},
                          "transcribe": _spec_for("basis/sto-3g.yaml")}])

    monkeypatch.setenv("QWB_BSE_FIXTURES", str(fixtures))
    monkeypatch.setenv("QWB_ENGINE_BIN", str(engine))
    with pytest.raises(DataFetchError, match="摘要"):
        fetch_entries(root, log=lambda line: None)
    assert not (root / "basis" / "sto-3g.yaml").exists()


def _pin_for(path: str) -> str:
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    for entry in sources["entries"]:
        if entry["path"] == path:
            return entry["sha256"]
    raise AssertionError(f"sources.json 里没有 {path}")


def test_datafetch_import_is_stdlib_only():
    """`import qwb_studio.datafetch` 不能把 pydantic 拖进来。

    下载基组的用户可能只装了系统 Python（没有 .venv）。包导出因此是惰性的（PEP 562），
    这条测试钉住它：一旦有人在 `__init__.py` 里恢复 eager import，这里就会红。
    """
    code = (
        "import sys; import qwb_studio.datafetch; "
        "print('pydantic' in sys.modules, 'numpy' in sys.modules)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "qwb-studio")
    completed = subprocess.run(  # noqa: S603 - 参数由本测试构造
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False False", completed.stdout


def _spec_for(path: str) -> dict:
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    for entry in sources["entries"]:
        if entry["path"] == path:
            return entry["transcribe"]
    raise AssertionError(f"sources.json 里没有 {path}")
