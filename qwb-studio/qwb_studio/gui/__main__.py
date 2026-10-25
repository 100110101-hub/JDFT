"""`python -m qwb_studio.gui`（P-09 的 GUI 入口）。"""

from __future__ import annotations

import sys

from .mainwindow import main

if __name__ == "__main__":
    sys.exit(main())
