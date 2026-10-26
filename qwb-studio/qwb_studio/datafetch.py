"""按 `qwb-data/sources.json` 把第三方数据取回本地（仓库**不存**这些文件）。

为什么要这样：`qwb-data/` 里的基组/ECP 是从 BSE 导出的第三方数据，其中 LANL2DZ 的 ECP
原始论文没有给出明确再分发条款（`redistribute_ok: false`，project.md §14 #20）。
把它们放进公开仓库 = 再分发。所以仓库只保留**配方**（`sources.json`）与**摘要钉**
（`registry.yaml` 里的 sha256），数据由使用者现场从上游取。

流程（每条清单）：

1. 按 `fetch` 段从 BSE 取 JSON（`QWB_BSE_FIXTURES` 指向目录时可离线用本地导出）；
2. 用 `qwb_studio.bse_data` 转录成 K1 YAML（**不联网**、规则只有一份）；
3. 调 Rust 的 `qwb data sign` 打 `sha256`（K1 的摘要规则只实现一次，不在 Python 里复刻）；
4. 核对摘要与 `sources.json` / `registry.yaml` 里的钉一致 —— 不一致就报错、不写坏文件。

用法：

```powershell
python -m qwb_studio.datafetch --list                # 哪些在、哪些缺、哪些摘要不对
python -m qwb_studio.datafetch --check               # 只检查（缺/坏 → 退出码 1）
python -m qwb_studio.datafetch --all                 # 全下（已正确的不动）
python -m qwb_studio.datafetch --need basis/sto-3g.yaml ecp/lanl2dz-cu.yaml
python -m qwb_studio.datafetch --for-input qwb-core/examples/cuh_lanl2dz_pbe.yaml
```

数据根：`--data-root` > `QWB_DATA_ROOT` > 向上搜索 `qwb-data/`（与 Rust CLI 同规则）。
打戳用的二进制：`QWB_ENGINE_BIN` > `qwb-core/target/{release,debug}/qwb[.exe]` > 报错。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .bse_data import (
    TranscriptionError,
    download_bse_json,
    transcribe,
    write_text,
)

__all__ = [
    "EntryStatus",
    "DataFetchError",
    "check_entries",
    "ensure_for_input",
    "fetch_entries",
    "load_sources",
]

SOURCES_FILE = "sources.json"


class DataFetchError(RuntimeError):
    """下载/转录/打戳失败（不写坏数据，不静默降级）。"""


@dataclass(frozen=True)
class EntryStatus:
    """一条清单的当前状态。"""

    path: str
    state: str  # "ok" | "missing" | "stale"
    expected: str
    actual: str | None

    @property
    def ok(self) -> bool:
        return self.state == "ok"


# ---------------------------------------------------------------------------
# 定位
# ---------------------------------------------------------------------------


def find_data_root(explicit: str | Path | None = None, start: str | Path | None = None) -> Path:
    """定位 `qwb-data/`：显式 > `QWB_DATA_ROOT` > 逐级向上找。

    与 `qwb_studio.runner.find_data_root` 的差别只有一处、而且是**故意的**：这里要求
    目录里有 `sources.json`。没有配方就没有"该下什么、钉是多少"的依据，此时硬要取数
    只能靠猜；而 `runner` 面对的是"自研数据 + 手工放好的第三方文件"这种合法场景。
    """
    if explicit is not None:
        root = Path(explicit)
        if not root.is_dir():
            raise DataFetchError(f"data root is not a directory: {root}")
        return root
    from_env = os.environ.get("QWB_DATA_ROOT")
    if from_env:
        root = Path(from_env)
        if not root.is_dir():
            raise DataFetchError(f"QWB_DATA_ROOT is not a directory: {root}")
        return root
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "qwb-data" / SOURCES_FILE).is_file():
            return candidate / "qwb-data"
    raise DataFetchError(
        f"找不到 qwb-data/{SOURCES_FILE}：用 --data-root 或 QWB_DATA_ROOT 指定"
    )


def load_sources(data_root: str | Path) -> dict:
    """读 `sources.json`（JSON，不需要 YAML 解析器）。"""
    path = Path(data_root) / SOURCES_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataFetchError(f"读不到 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DataFetchError(f"{path} 不是合法 JSON: {exc}") from exc
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DataFetchError(f"{path} 里没有 entries")
    return payload


def _sign(path: Path) -> str:
    """调 Rust 的 `qwb data sign` 打 sha256，返回摘要。

    为什么不在 Python 里算：K1 的摘要规则是"解析 YAML → 删 sha256 → 转 JSON（键排序）→
    紧凑 JSON 的 SHA-256"，与 YAML 排版无关。Python 侧没有 YAML 解析器（§2.4），
    而复刻一份规则就等于有两份真值 —— 交给 `qwb-registry` 的唯一实现。
    """
    from .cli import EngineNotFound, engine_command

    try:
        command, cargo_cwd = engine_command()
    except EngineNotFound as exc:
        raise DataFetchError(f"找不到 qwb 二进制：{exc}") from exc
    completed = subprocess.run(  # noqa: S603 - 命令由本模块构造
        [*command, "data", "sign", str(path)],
        cwd=str(cargo_cwd) if cargo_cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise DataFetchError(
            f"qwb data sign 失败（退出码 {completed.returncode}）：\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    # `qwb data sign` 的输出形如（每行一个文件，最后一行是汇总）：
    #   signed <64 位摘要>  <路径>
    #   1 file(s) signed
    for line in completed.stdout.splitlines():
        for token in line.split():
            if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
                return token
    raise DataFetchError(f"qwb data sign 没有打印摘要：\n{completed.stdout}")


# ---------------------------------------------------------------------------
# 检查 / 取数
# ---------------------------------------------------------------------------


def _actual_digest(path: Path) -> str | None:
    """读文件里声明的 `sha256`（不做规范化计算 —— 那是 `qwb data verify` 的事）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("sha256:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def check_entries(data_root: str | Path, *, only: list[str] | None = None) -> list[EntryStatus]:
    """检查每条清单：文件在不在、文件里声明的摘要等于不等于钉。"""
    root = Path(data_root)
    sources = load_sources(root)
    wanted = set(only) if only else None
    out: list[EntryStatus] = []
    for entry in sources["entries"]:
        rel = entry["path"]
        if wanted is not None and rel not in wanted:
            continue
        expected = entry["sha256"]
        target = root / rel
        if not target.is_file():
            out.append(EntryStatus(rel, "missing", expected, None))
            continue
        actual = _actual_digest(target)
        state = "ok" if actual == expected else "stale"
        out.append(EntryStatus(rel, state, expected, actual))
    return out


def _bse_payload(entry: dict, root: Path) -> dict:
    """取一条清单的 BSE JSON（优先本地夹具，便于离线测试/复现）。"""
    fetch = entry["fetch"]
    basis = fetch["basis"]
    elements = [int(z) for z in fetch["elements"]]

    fixture_dir = os.environ.get("QWB_BSE_FIXTURES")
    if fixture_dir:
        candidate = Path(fixture_dir) / f"{basis}.json"
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            have = set(int(k) for k in data.get("elements", {}))
            missing = [z for z in elements if z not in have]
            if missing:
                raise DataFetchError(
                    f"{candidate} 缺元素 {missing}；离线夹具要覆盖整条清单"
                )
            return data
        raise DataFetchError(f"QWB_BSE_FIXTURES 里没有 {candidate.name}")

    try:
        return download_bse_json(basis, elements)
    except TranscriptionError as exc:
        raise DataFetchError(str(exc)) from exc


def fetch_entries(
    data_root: str | Path,
    *,
    only: list[str] | None = None,
    force: bool = False,
    log=print,
) -> dict[str, str]:
    """下载/重建需要的数据文件；返回 `{path: digest}`。

    已经正确（摘要等于钉）的文件默认**不动**；`force=True` 时重下。
    写盘顺序：写到临时文件 → 打戳 → 核对摘要 → 原子替换目标。所以中断不会留下半个文件。
    """
    root = Path(data_root)
    sources = load_sources(root)
    wanted = set(only) if only else None
    statuses = {status.path: status for status in check_entries(root, only=only)}
    results: dict[str, str] = {}

    for entry in sources["entries"]:
        rel = entry["path"]
        if wanted is not None and rel not in wanted:
            continue
        status = statuses[rel]
        if status.ok and not force:
            log(f"ok      {rel}")
            results[rel] = status.expected
            continue

        log(f"fetch   {rel}（{status.state}）")
        data = _bse_payload(entry, root)
        try:
            text = transcribe(entry["kind"], entry["transcribe"], data)
        except TranscriptionError as exc:
            raise DataFetchError(f"{rel}: 转录失败：{exc}") from exc

        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qwb-datafetch-") as tmp:
            staged = Path(tmp) / Path(rel).name
            write_text(staged, text)
            digest = _sign(staged)
            if digest != entry["sha256"]:
                raise DataFetchError(
                    f"{rel}: 转录结果的摘要 {digest} 与清单里的钉 {entry['sha256']} 不一致。\n"
                    "  说明上游数据或转录规则变了：核对 BSE 修订号，或更新 sources.json 的 sha256"
                    "（并把 registry.yaml 里的同一条一起改）。"
                )
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), str(target))
            log(f"saved   {rel}  sha256={digest}")
            results[rel] = digest
    return results


# ---------------------------------------------------------------------------
# 按作业输入取数（GUI / Python 壳 / Rust CLI 的自动入口）
# ---------------------------------------------------------------------------


def _names_needed(input_yaml: str | Path) -> dict[str, str]:
    """从 `input.yaml` 里取作业需要的名字（不解析整份 YAML，只找那几行）。"""
    text = Path(input_yaml).read_text(encoding="utf-8")
    wanted: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for key, label in (
            ("basis_name:", "basis"),
            ("method_name:", "functional"),
            ("grid_level:", "grid"),
            ("ecp_name:", "ecp"),
        ):
            if stripped.startswith(key):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if value and value not in ("null", "~"):
                    wanted[label] = value
    return wanted


def ensure_for_input(
    input_yaml: str | Path,
    *,
    data_root: str | Path | None = None,
    log=print,
) -> list[str]:
    """保证这份 `input.yaml` 需要的第三方数据都在；缺了就下。返回补下的文件列表。

    只处理 `sources.json` 里列出的文件（自研数据随仓库走，不需要下）。
    """
    root = find_data_root(data_root)
    names = _names_needed(input_yaml)
    sources = load_sources(root)
    lookup: dict[tuple[str, str], str] = {}
    for entry in sources["entries"]:
        key = (entry["kind"], entry["transcribe"]["name"])
        lookup[key] = entry["path"]
    # `grid_level` 在数据里叫 `grid-normal` 这类名字
    if "grid" in names and not names["grid"].startswith("grid-"):
        names["grid"] = f"grid-{names['grid']}"

    needed: list[str] = []
    for kind, name in names.items():
        rel = lookup.get((kind, name))
        if rel is not None:
            needed.append(rel)
    if not needed:
        return []

    statuses = {status.path: status for status in check_entries(root, only=needed)}
    pending = [rel for rel in needed if not statuses[rel].ok]
    if not pending:
        return []
    log(f"数据缺失/过期：{', '.join(pending)}，正在从 BSE 取回…")
    fetch_entries(root, only=pending, log=log)
    return pending


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _force_utf8_streams() -> None:
    """把 stdout/stderr 切到 UTF-8。

    这里的中文提示（"缺失/过期"、"已取回"）在 Windows 上默认按控制台代码页（如 cp936）
    编码，遇到非该代码页的字符会变成乱码或直接 `UnicodeEncodeError`。下载脚本是给人的
    工具，输出不该因为控制台设置而不可读。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # pragma: no cover - 管道关闭等
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    parser = argparse.ArgumentParser(
        prog="qwb-datafetch",
        description="按 qwb-data/sources.json 从 BSE 取回第三方数据（仓库不存这些文件）",
    )
    parser.add_argument("--data-root", help="qwb-data 目录（默认 QWB_DATA_ROOT 或向上搜索）")
    parser.add_argument("--list", action="store_true", help="列出每条清单的状态")
    parser.add_argument("--check", action="store_true", help="只检查；有缺/坏则退出码 1")
    parser.add_argument("--all", action="store_true", help="取回全部清单（已正确的跳过）")
    parser.add_argument("--need", nargs="+", metavar="PATH", help="只取这些相对路径")
    parser.add_argument("--for-input", metavar="INPUT_YAML", help="按这份 input.yaml 取它需要的数据")
    parser.add_argument("--force", action="store_true", help="即使摘要已正确也重下")
    args = parser.parse_args(argv)

    if not any((args.list, args.check, args.all, args.need, args.for_input)):
        parser.error("至少要给 --list / --check / --all / --need / --for-input 之一")

    try:
        root = find_data_root(args.data_root)
        if args.list or args.check:
            statuses = check_entries(root, only=args.need)
            width = max(len(status.path) for status in statuses)
            lines = []
            for status in statuses:
                mark = {"ok": "ok     ", "missing": "MISSING", "stale": "STALE  "}[status.state]
                lines.append(f"{mark} {status.path:<{width}}  {status.expected[:16]}…")
            bad = [status for status in statuses if not status.ok]
            lines.append("")
            lines.append(f"{len(statuses) - len(bad)}/{len(statuses)} 条就绪")
            # 一次打印 + flush：否则提示行（stderr，无缓冲）会插到表格中间。
            print("\n".join(lines), flush=True)
            if args.check and bad:
                print(
                    "缺/坏的文件请跑：python -m qwb_studio.datafetch --all",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            return 0
        if args.for_input:
            fetched = ensure_for_input(args.for_input, data_root=root)
            print("无需下载" if not fetched else "已取回：" + ", ".join(fetched))
            return 0
        fetched = fetch_entries(root, only=args.need, force=args.force)
        print(f"\n完成：{len(fetched)} 条（数据根 {root}）")
        return 0
    except DataFetchError as exc:
        print(f"qwb-datafetch: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
