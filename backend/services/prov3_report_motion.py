"""Pro v3 Gemini 报告前的运动摘要：挥杆窗 + dense scan（实现复用 legacy 模块，对外仅暴露 prov3 命名）。"""

from __future__ import annotations

from services.pro_v2_dense_scan_service import DenseFrame, dense_scan_swing_region
from services.pro_v2_swing_window_service import find_swing_window_seconds

__all__ = ["DenseFrame", "dense_scan_swing_region", "find_swing_window_seconds"]
