"""数值层统一转型入口 — Decimal↔float 边界的唯一通道。

所有分析代码（report、charts、stability、analyzer 等）通过此模块获取
float64 numpy 数组，不再各自调用 float(bar.xxx) 或 float(p.total_value)。

设计原则：
    - 簿记层 (Bar, Portfolio) 保持 Decimal —— 结算精确
    - 分析层 (report, charts, risk) 使用 float64 —— 统计高效
    - 此模块是两层之间唯一的转型边界

未来方向：
    当此边界稳定后，可考虑将 Bar schema 的 Decimal 字段直接改为 float64，
    届时只需修改此模块内部实现，所有调用方不受影响。
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.data.schema import Bar
    from app.engine.backtest.models import EquityPoint

__all__ = [
    "equity_to_numpy",
    "bars_to_close_np",
    "bars_to_ohlcv",
    "to_float",
]


def to_float(val: Decimal | float | int | None) -> float:
    """安全的 Decimal/float 转型，统一处理 None。

    性能敏感路径已内联；此函数用于零散的单个转换场景。
    """
    if val is None:
        return np.nan
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


def equity_to_numpy(equity_curve: Sequence[EquityPoint]) -> np.ndarray:
    """将权益曲线转为 float64 numpy 数组。

    Args:
        equity_curve: EquityPoint 列表，每个点有 .total_value (Decimal)

    Returns:
        形状 (N,) 的 float64 数组，值为 total_value 的浮点表示

    Usage:
        values = equity_to_numpy(result.equity_curve)
        daily_rets = np.diff(values) / values[:-1]
    """
    return np.array([float(p.total_value) for p in equity_curve], dtype=np.float64)


def bars_to_close_np(bars: Sequence[Bar]) -> np.ndarray:
    """从 Bar 列表提取收盘价 float64 数组。

    Args:
        bars: Bar 对象列表

    Returns:
        形状 (N,) 的 float64 数组
    """
    return np.array([float(b.close) for b in bars], dtype=np.float64)


def bars_to_ohlcv(bars: Sequence[Bar]) -> dict[str, np.ndarray]:
    """将 Bar 列表转为 numpy OHLCV 数组字典。

    Args:
        bars: Bar 对象列表

    Returns:
        {"open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
        所有价格字段为 float64，volume 为 int64

    Usage:
        ohlcv = bars_to_ohlcv(bars)
        tr = np.maximum(ohlcv["high"] - ohlcv["low"],
                        np.maximum(np.abs(ohlcv["high"] - prev_close),
                                   np.abs(ohlcv["low"] - prev_close)))
    """
    n = len(bars)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.int64)

    for i, b in enumerate(bars):
        opens[i] = float(b.open)
        highs[i] = float(b.high)
        lows[i] = float(b.low)
        closes[i] = float(b.close)
        volumes[i] = b.volume

    return {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }
