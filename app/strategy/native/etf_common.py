"""ETF 策略共用常量与安全转换工具（无策略状态，可被任意 ETF 策略复用）。"""

from __future__ import annotations

import statistics
from typing import Any

LOT_SIZE = 100  # 场内基金最小交易单位：100 份
MIN_TRADE_RATIO = 0.002  # 小于组合市值 0.2% 的调仓不动，省摩擦
AMOUNT_WINDOW = 20  # 流动性统计窗（成交额）
VOL_WINDOW = 20  # 波动率统计窗
CORR_WINDOW = 60  # 同类去重的相关性统计窗
SPLIT_TOLERANCE = 0.005  # 与引擎 _apply_share_adjustments 同阈：视为无除权的偏差上限


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（None/NaN/decimal 异常时回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _int(value: Any, default: int = 0) -> int:
    """安全转 int（None/NaN 异常时回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _stdev(values: list[float]) -> float:
    """样本标准差；样本不足 2 个时返回 0。"""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _returns(prices: list[float]) -> list[float]:
    """相邻价格的简单收益率序列。"""
    return [curr / prev - 1.0 for prev, curr in zip(prices, prices[1:]) if prev > 0]


def _split_ratio(prev_close: float, pre_close: float) -> float | None:
    """拆分/合并/分红比例；无除权事件时返回 None。

    与引擎 ``_apply_share_adjustments`` 同口：昨日收盘 ≠ 今日参考价即视为除权，
    比例 = 昨收 / 参考价（拆分放大、合并缩小）。
    """
    if prev_close <= 0 or pre_close <= 0:
        return None
    ratio = prev_close / pre_close
    return None if abs(ratio - 1.0) < SPLIT_TOLERANCE else ratio
