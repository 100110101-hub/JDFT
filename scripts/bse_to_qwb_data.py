#!/usr/bin/env python3
"""把 BSE（Basis Set Exchange）导出的 JSON 转成 K1 基组 YAML —— **薄封装**。

project.md §8.1 允许「BSE 网站导出」，禁止「从其他项目代码仓库复制文件」。
转录规则（角动量拆分、系数列序、`region` 白名单、逐壳层收缩类型）只在
`qwb-studio/qwb_studio/bse_data.py` 里有一份实现：规则一旦有两份就一定会漂移，
而漂移的后果是**静默错数**。本脚本只做「命令行参数 → 转录元数据」的翻译。

日常用途：

* 用户/CI 取回第三方数据用 `python -m qwb_studio.datafetch --all`
  （或 `qwb data install`），它读 `qwb-data/sources.json` 的配方，不需要本脚本；
* 本脚本用于**离线已有导出**的手动转录（新增一个基组、或核对某个历史文件）。

用法（在仓库根）：

    python scripts/bse_to_qwb_data.py \\
      --input  qwb-core/crates/qwb-registry/tests/fixtures/bse/sto-3g.json \\
      --output qwb-data/basis/sto-3g.yaml \\
      --name sto-3g --version 1.0.0 \\
      --title "Self-Consistent Molecular-Orbital Methods. I. ..." \\
      --doi 10.1063/1.1672392 \\
      --created 2026-10-22T23:16:10Z \\
      --notes "..."

输出**不含** `sha256`：打完 YAML 后跑 `qwb data sign <file>` 打戳（K1 的摘要规则只有
Rust 那一份实现，Python 侧不复刻）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwb-studio"))

from qwb_studio.bse_data import (  # noqa: E402 - 需要先改 sys.path
    TranscriptionError,
    transcribe_basis,
    write_text,
)

# 只允许 gto 系的壳层；ECP 基组（LANL2DZ 等）的导出会同时带 `scalar_ecp`，
# 那是 ECP 参数（由 scripts/bse_ecp_to_qwb_data.py 转录），不影响基组转录。
ORBITAL_TYPES = {"gto", "gto_spherical"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="BSE JSON 导出文件")
    parser.add_argument("--output", required=True, help="输出的 K1 YAML 路径")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--title", required=True)
    parser.add_argument("--doi", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--spdx", default="Apache-2.0")
    parser.add_argument("--commercial-ok", dest="commercial_ok", action="store_true", default=True)
    parser.add_argument(
        "--no-commercial",
        dest="commercial_ok",
        action="store_false",
        help="原始条款没写清商用许可时用这个（§14 #20 的严格读法）",
    )
    parser.add_argument(
        "--no-redistribute",
        dest="redistribute_ok",
        action="store_false",
        default=True,
        help="外部来源条款未确认时用这个（§14 #20）；这类数据不进仓库，只进 sources.json",
    )
    parser.add_argument("--created", required=True)
    parser.add_argument("--angular", default="spherical", choices=["spherical", "cartesian"])
    parser.add_argument("--normalization", default="l2", choices=["l2", "none"])
    parser.add_argument("--role", default="orbital", choices=["orbital", "ri-j", "ri-jk", "ecp-fit"])
    parser.add_argument(
        "--contraction-type",
        dest="contraction_type",
        default="auto",
        choices=["auto", "segmented", "general"],
        help="auto（默认）= 逐壳层判定：一列系数=segmented，多列=general",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    seen = set(data.get("function_types") or [])
    unexpected = seen - ORBITAL_TYPES - {"scalar_ecp"}
    if unexpected:
        print(f"unsupported function_types: {sorted(unexpected)}", file=sys.stderr)
        return 1
    if "gto_spherical" in seen and args.angular != "spherical":
        print("BSE export contains spherical shells; use --angular spherical", file=sys.stderr)
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
        "angular": args.angular,
        "normalization": args.normalization,
        "role": args.role,
    }
    if args.contraction_type != "auto":
        spec["contraction_type"] = args.contraction_type

    try:
        text = transcribe_basis(spec, data)
    except TranscriptionError as error:
        print(f"转录失败：{error}", file=sys.stderr)
        return 1
    write_text(args.output, text)

    shell_count = text.count("      - l: ")
    print(f"wrote {args.output}（{len(data['elements'])} 个元素，{shell_count} 个壳层）")
    print(f"next: qwb data sign {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
