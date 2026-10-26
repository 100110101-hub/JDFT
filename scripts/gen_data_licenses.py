#!/usr/bin/env python3
"""生成 / 校验 `DATA_LICENSES.md`（D-15）。

判据（project.md §6.2）：「脚本可重跑且输出幂等；所有数据带 license/citation/sha256」。

做法：
  * 扫描 `qwb-data/` 下所有**生产数据**（`registry.yaml` 引用的 basis / functional /
    grid / ecp / solvent 文件，以及 registry 自身），逐个读出
    `name`、`version`、`license.spdx`、`citation`、`sha256`；
  * **第三方数据缺失时从 `qwb-data/sources.json` 补行**：仓库不提交那些文件
    （再分发条款，见 project.md §14 #20），但许可、引用与摘要钉都在清单里。
    所以"数据在不在本地"不改变本文件的内容 —— 否则这份文件会随下载状态漂移，
    而它恰恰是要回答"这份数据能不能用"的。
  * 输出一份按路径排序的 Markdown——**排序保证幂等**（同样的输入永远得到同样的字节）；
  * 任一文件缺 license / citation / sha256 时，脚本以非零退出码失败，并在文档里
    标出缺失项，所以它既是生成器也是门禁。

`qwb-data/examples/` 里的 SPECIMEN 不算生产数据（它们在 manifest 里没有条目），跳过。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "qwb-data"
OUTPUT = ROOT / "DATA_LICENSES.md"
SOURCES = DATA / "sources.json"

# 生产数据所在的子目录（与 registry.yaml 的引用一致）。
SUBDIRECTORIES = ("basis", "functional", "grid", "ecp", "solvent")


def parse_scalar(block: dict, key: str) -> str | None:
    """从极简的 YAML 行扫描里取一个标量（不引入 PyYAML：本仓库的数据是受控格式）。"""
    prefix = f"{key}:"
    for line in block["lines"]:
        # 只认**顶层**键（不缩进）：manifest 里嵌套着每条数据的 sha256，
        # 之前这里会把第一个嵌套值当成 manifest 自己的，属于静默取错。
        if line[:1].isspace() or not line:
            continue
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip("'\"")
    return None


def scoped_dict(block: dict, key: str) -> dict:
    """取一个二级映射（如 `license:` 下的 spdx 等）。"""
    out: dict = {}
    inside = False
    base_indent = 0
    for line in block["lines"]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if line.strip().startswith(f"{key}:"):
            inside = True
            base_indent = indent
            continue
        if inside:
            if indent <= base_indent:
                break
            stripped = line.strip()
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def scoped_list_titles(block: dict, key: str) -> list[str]:
    """取 `citation:` 列表里各条的 `title`。"""
    titles: list[str] = []
    inside = False
    base_indent = 0
    for line in block["lines"]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if line.strip().startswith(f"{key}:"):
            inside = True
            base_indent = indent
            continue
        if inside:
            if indent <= base_indent and not line.lstrip().startswith("-"):
                break
            stripped = line.lstrip("- ").strip()
            if stripped.startswith("title:"):
                titles.append(stripped[len("title:") :].strip().strip("'\"> "))
    return titles


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    problems: list[str] = []

    targets = [DATA / "registry.yaml"]
    for subdirectory in SUBDIRECTORIES:
        directory = DATA / subdirectory
        if directory.is_dir():
            targets.extend(sorted(directory.glob("*.yaml")))

    present: set[str] = set()
    for path in targets:
        lines = path.read_text(encoding="utf-8").splitlines()
        block = {"lines": lines}
        name = parse_scalar(block, "name") or "(未命名)"
        version = parse_scalar(block, "version") or "?"
        license_map = scoped_dict(block, "license")
        spdx = scoped_dict(block, "license").get("spdx", "?")
        citations = scoped_list_titles(block, "citation")
        recorded = parse_scalar(block, "sha256")
        relative = path.relative_to(ROOT).as_posix()
        present.add(relative)

        # `registry.yaml` 是 manifest（索引），不承载许可与引用——许可在各数据文件里。
        is_manifest = path.name == "registry.yaml"
        if not is_manifest and spdx in (None, "?", "", "null"):
            problems.append(f"{relative}: 缺 license.spdx")
        if not is_manifest and not citations:
            problems.append(f"{relative}: 缺 citation")
        if not is_manifest and recorded in (None, "", "null"):
            problems.append(f"{relative}: 缺 sha256（跑 `qwb-stamp-data`）")
        if not is_manifest and license_map.get("commercial_ok") not in ("true", "false"):
            problems.append(f"{relative}: license.commercial_ok 缺失或非布尔")

        records.append(
            {
                "path": relative,
                "name": name,
                "version": version,
                "spdx": spdx,
                "commercial_ok": license_map.get("commercial_ok", "?"),
                "redistribute_ok": license_map.get("redistribute_ok", "?"),
                "citations": citations or ([] if path.name == "registry.yaml" else ["(缺)"]),
                "is_manifest": path.name == "registry.yaml",
                "sha256": (recorded or "—") if not path.name == "registry.yaml" else "—（manifest，条目各自带摘要）",
            }
        )

    records.extend(declared_but_absent(present, problems))
    records.sort(key=lambda record: record["path"])
    return records, problems


def declared_but_absent(present: set[str], problems: list[str]) -> list[dict]:
    """`sources.json` 里声明、但本地没有的第三方数据 → 与文件在本地时**完全相同**的行。

    字段来自清单的 `transcribe` 段与摘要钉；渲染出来的表格和"文件已取回"时逐字节一致，
    所以 `--check` 不依赖网络、也不依赖本地下载状态。
    """
    if not SOURCES.is_file():
        return []
    try:
        payload = json.loads(SOURCES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"qwb-data/sources.json: 读不出（{error}）")
        return []

    rows: list[dict] = []
    for entry in payload.get("entries") or []:
        relative = "qwb-data/" + str(entry["path"]).replace("\\", "/")
        if relative in present:
            continue
        spec = entry.get("transcribe") or {}
        license_map = spec.get("license") or {}
        spdx = license_map.get("spdx") or "?"
        titles = [citation.get("title", "(缺)") for citation in spec.get("citations") or []]
        pin = entry.get("sha256") or "—"
        if spdx in ("?", "", "null"):
            problems.append(f"{relative}: sources.json 缺 license.spdx")
        if not titles:
            problems.append(f"{relative}: sources.json 缺 citation")
        if pin == "—":
            problems.append(f"{relative}: sources.json 缺 sha256 钉")
        rows.append(
            {
                "path": relative,
                "name": spec.get("name") or "(未命名)",
                "version": spec.get("version") or "?",
                "spdx": spdx,
                "commercial_ok": str(license_map.get("commercial_ok", "?")).lower(),
                "redistribute_ok": str(license_map.get("redistribute_ok", "?")).lower(),
                "citations": titles or ["(缺)"],
                "is_manifest": False,
                "sha256": pin,
            }
        )
    return rows


def render(records: list[dict], problems: list[str]) -> str:
    out = [
        "# 数据许可与出处（自动生成，请勿手改）",
        "",
        "本文件由 `scripts/gen_data_licenses.py` 生成。重跑必须得到**逐字节相同**的结果",
        "（排序固定 + 不含时间戳），所以它可以直接进 CI 比对。",
        "",
        f"共 {len(records)} 个生产数据文件。`qwb-data/examples/` 下的 SPECIMEN 不计入",
        "（它们是形状示例，不在 manifest 里）。",
        "",
        "其中 `qwb-data/basis/` 与 `qwb-data/ecp/` 的部分文件是从 Basis Set Exchange 导出的",
        "第三方数据：仓库**不提交**这些文件（再分发条款，见 `qwb-data/sources.json` 的",
        "`redistribute_ok`），按需用 `qwb data install`（或 `python -m qwb_studio.datafetch --all`）",
        "取回并核对摘要钉。下表因此不区分「已取回」与「未取回」——许可、引用与摘要都由清单",
        "钉住，与本机是否下载过无关。",
        "",
        "| 数据 | 版本 | 许可 | 商用 | 可再分发 | sha256（前 16 位） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        out.append(
            "| `{path}` | {version} | {spdx} | {commercial} | {redistribute} | `{sha}` |".format(
                path=record["path"],
                version=record["version"],
                spdx=record["spdx"],
                commercial=record["commercial_ok"],
                redistribute=record["redistribute_ok"],
                sha=record["sha256"][:16],
            )
        )
    out += ["", "## 引用", ""]
    for record in records:
        out.append(f"### `{record['path']}`（{record['name']} {record['version']}）")
        out.append("")
        for title in record["citations"]:
            out.append(f"- {title}")
        out.append("")
    out += ["## 门禁状态", ""]
    if problems:
        out.append("**不通过**：")
        out.append("")
        for problem in problems:
            out.append(f"- {problem}")
    else:
        out.append("所有数据都带 license / citation / sha256，且 sha256 与内容一致。")
    out.append("")
    return "\n".join(out)


def verify_with_rust_tool() -> list[str]:
    """用 `qwb-stamp-data --check` 校验摘要。

    摘要的规则是「YAML 去掉顶层 sha256 键后再规范化求 hash」（自引用），用 Python
    复刻规范化不现实，也不该有两份实现——所以这里直接调用权威工具。
    """
    import subprocess

    files = [str(DATA / "registry.yaml")]
    for subdirectory in SUBDIRECTORIES:
        directory = DATA / subdirectory
        if directory.is_dir():
            files.extend(str(path) for path in sorted(directory.glob("*.yaml")))
    try:
        result = subprocess.run(
            ["cargo", "run", "-q", "-p", "qwb-registry", "--bin", "qwb-stamp-data", "--"]
            + files
            + ["--check"],
            cwd=ROOT / "qwb-core",
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return [f"无法运行 qwb-stamp-data：{error}"]
    if result.returncode != 0:
        return [f"qwb-stamp-data --check 失败：{(result.stderr or result.stdout).strip()[:400]}"]
    return []


def main() -> int:
    check_only = "--check" in sys.argv
    records, problems = collect()
    problems.extend(verify_with_rust_tool())
    text = render(records, problems)
    if check_only:
        if not OUTPUT.is_file():
            print(f"FAIL {OUTPUT.name} 不存在；先不带 --check 跑一次", file=sys.stderr)
            return 1
        if OUTPUT.read_text(encoding="utf-8") != text:
            print(f"FAIL {OUTPUT.name} 与当前数据不一致；重跑脚本", file=sys.stderr)
            return 1
        print(f"ok: {OUTPUT.name} 与数据一致（{len(records)} 个文件）")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUTPUT.name}（{len(records)} 个文件）")
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
