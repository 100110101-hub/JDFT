#!/usr/bin/env python3
"""取回第三方数据（基组/ECP）—— 仓库不提交这些文件，只提交配方与摘要钉。

这个脚本只是 `python -m qwb_studio.datafetch` 的薄入口，好处是不用装配
`PYTHONPATH`：`scripts/` 下的脚本自己把 `qwb-studio/` 加进 `sys.path`。

    python scripts/fetch_data.py --list                 # 哪些在、哪些缺、哪些摘要不对
    python scripts/fetch_data.py --check                # 只检查（缺/坏 → 退出码 1）
    python scripts/fetch_data.py --all                  # 全下（已正确的不动）
    python scripts/fetch_data.py --need basis/lanl2dz.yaml ecp/lanl2dz-cu.yaml
    python scripts/fetch_data.py --for-input qwb-core/examples/cuh_lanl2dz_pbe.yaml

等价入口（三选一，语义完全相同）：

    qwb data install [--all|--need ...|--for-input ...]   # Rust CLI 转发到同一个实现
    python -m qwb_studio.datafetch ...                    # 需要 qwb-studio 在 PYTHONPATH
    GUI → Data 页 → 「下载缺失」                           # PySide6 界面

数据写在 `qwb-data/basis/` 与 `qwb-data/ecp/`；每个文件写完前都会核对
`qwb-data/sources.json` 里的 `sha256` 钉，对不上就报错、不落盘。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwb-studio"))

from qwb_studio.datafetch import main  # noqa: E402 - 需要先改 sys.path

if __name__ == "__main__":
    # 中文日志在 Windows 控制台（cp936）下会乱码，先切 UTF-8（`main` 里也会切一次，
    # 这里是给"import 之前就打印"的情形兜底）。
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            try:
                _reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    raise SystemExit(main(sys.argv[1:]))
