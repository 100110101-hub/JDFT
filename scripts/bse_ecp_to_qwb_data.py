#!/usr/bin/env python3
"""把 BSE 导出的 **ECP 参数**转成 K1 `ecp` YAML —— **薄封装**。

project.md §8.1/§8.2 允许「BSE 网站导出 + 一手文献核对」。转录规则（局域通道 =
导出里最大的 `l`、`r^n` 约定、`z_eff` 自洽）只在 `qwb-studio/qwb_studio/bse_data.py`
里有一份实现：规则一旦有两份就一定会漂移，而漂移的后果是**静默错数**。

日常用途：取回数据用 `python -m qwb_studio.datafetch --all`（或 `qwb data install`），
它读 `qwb-data/sources.json` 的配方；本脚本用于**离线已有导出**的手动转录。

用法（在仓库根）：

    python scripts/bse_ecp_to_qwb_data.py \\
      --input qwb-core/crates/qwb-registry/tests/fixtures/bse/lanl2dz.json \\
      --output qwb-data/ecp/lanl2dz-cu.yaml \\
      --name lanl2dz-cu --element Cu \\
      --title "..." --doi 10.1063/1.448975 --created 2026-10-26T14:00:00Z

输出**不含** `sha256`：打完 YAML 后跑 `qwb data sign <file>` 打戳。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwb-studio"))

from qwb_studio.bse_data import (  # noqa: E402 - 需要先改 sys.path
    TranscriptionError,
    atomic_number,
    transcribe_ecp,
    write_text,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="BSE JSON 导出文件（含 ecp_potentials）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--element", required=True, help="元素符号（一个文件一个元素，K1 的 Ecp 就是单元素）"
    )
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--title", required=True)
    parser.add_argument("--doi", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--spdx", default="NOASSERTION")
    parser.add_argument("--commercial-ok", dest="commercial_ok", action="store_true", default=False)
    parser.add_argument("--redistribute-ok", dest="redistribute_ok", action="store_true", default=False)
    parser.add_argument("--created", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    try:
        z = atomic_number(args.element)
    except TranscriptionError as error:
        print(str(error), file=sys.stderr)
        return 1
    entry = data["elements"].get(str(z)) or {}
    n_core = entry.get("ecp_electrons")
    if n_core is None:
        print(f"导出文件里没有 {args.element}（Z={z}）的 ecp_electrons", file=sys.stderr)
        return 1

    citation: dict = {"title": args.title}
    if args.doi:
        citation["doi"] = args.doi
    if args.url:
        citation["url"] = args.url
    spec: dict = {
        "name": args.name,
        "version": args.version,
        "notes": args.notes,
        "citations": [citation],
        "license": {
            "spdx": args.spdx,
            "commercial_ok": args.commercial_ok,
            "redistribute_ok": args.redistribute_ok,
        },
        "created": args.created,
        "element": args.element,
    }

    try:
        text = transcribe_ecp(spec, data)
    except TranscriptionError as error:
        print(f"转录失败：{error}", file=sys.stderr)
        return 1
    write_text(args.output, text)

    channels = sorted(
        int(block["angular_momentum"][0])
        for block in entry.get("ecp_potentials") or []
        if block.get("angular_momentum")
    )
    print(
        f"wrote {args.output}: {args.element} n_core={int(n_core)} "
        f"z_eff={z - int(n_core):g}, channels l={channels} "
        f"(local = {max(channels) if channels else '?'})"
    )
    print(f"next: qwb data sign {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
