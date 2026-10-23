"""技术指标预计算缓存。

为回测引擎提供 O(1) 查找的技术指标矩阵，避免逐 bar 重复计算。

用法:
    cache = build_precomputed_cache(all_data, config)
    indicators = cache.get("600519.SH")
    values = indicators.at(idx=500)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.data.schema import Bar

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedIndicators:
    """单只股票的预计算指标数组。

    所有数组索引对齐：索引 i 对应输入 bar 列表的第 i 根 K 线。
    """

    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray

    ema_fast: np.ndarray
    ema_slow: np.ndarray
    ma: np.ndarray

    atr: np.ndarray
    adx: np.ndarray

    bb_upper: np.ndarray
    bb_middle: np.ndarray
    bb_lower: np.ndarray

    rsi: np.ndarray

    @property
    def n(self) -> int:
        return len(self.closes)

    def at(self, idx: int) -> dict[str, float]:
        """返回指定 bar 索引处的指标值。O(1)。"""
        return {
            "price": float(self.closes[idx]),
            "ema_fast": float(self.ema_fast[idx]),
            "ema_slow": float(self.ema_slow[idx]),
            "ma": float(self.ma[idx]),
            "atr": float(self.atr[idx]),
            "adx": float(self.adx[idx]),
            "bb_upper": float(self.bb_upper[idx]),
            "bb_middle": float(self.bb_middle[idx]),
            "bb_lower": float(self.bb_lower[idx]),
            "rsi": float(self.rsi[idx]),
        }


@dataclass
class PrecomputedIndicatorCache:
    """多只股票的预计算指标缓存。"""

    _data: dict[str, PrecomputedIndicators] = field(default_factory=dict)

    def get(self, symbol: str) -> PrecomputedIndicators | None:
        return self._data.get(symbol)

    def set(self, symbol: str, data: PrecomputedIndicators) -> None:
        self._data[symbol] = data

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data


def _vectorized_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_period: int,
    adx_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化计算 ADX 和 ATR。

    返回 (atr, adx) 数组，形状为 (n,)。
    """
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.empty(n, dtype=np.float64)
    minus_dm = np.empty(n, dtype=np.float64)

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]

    tr = np.maximum(
        highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
    )

    plus_dm[1:] = np.where(
        (highs[1:] - prev_close[1:]) > (prev_close[1:] - lows[1:]),
        np.maximum(highs[1:] - prev_close[1:], 0.0),
        0.0,
    )
    plus_dm[0] = 0.0

    minus_dm[1:] = np.where(
        (prev_close[1:] - lows[1:]) > (highs[1:] - prev_close[1:]),
        np.maximum(prev_close[1:] - lows[1:], 0.0),
        0.0,
    )
    minus_dm[0] = 0.0

    def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
        """Wilder 平滑。"""
        result = np.empty_like(values)
        cumsum = np.cumsum(values)
        result[: period - 1] = np.nan
        result[period - 1] = cumsum[period - 1] / period
        for i in range(period, len(values)):
            result[i] = result[i - 1] - result[i - 1] / period + values[i]
        result[: period - 1] = result[period - 1]
        return result

    smoothed_tr = wilder_smooth(tr, atr_period)
    smoothed_plus_dm = wilder_smooth(plus_dm, adx_period)
    smoothed_minus_dm = wilder_smooth(minus_dm, adx_period)

    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)

    mask = smoothed_tr > 0
    plus_di[mask] = (smoothed_plus_dm[mask] / smoothed_tr[mask]) * 100
    minus_di[mask] = (smoothed_minus_dm[mask] / smoothed_tr[mask]) * 100
    di_sum = plus_di + minus_di
    nonzero = di_sum > 0
    dx[nonzero] = (np.abs(plus_di[nonzero] - minus_di[nonzero]) / di_sum[nonzero]) * 100

    adx_out = wilder_smooth(dx, adx_period)

    atr_out = np.empty(n, dtype=np.float64)
    atr_out[:] = smoothed_tr

    return atr_out, adx_out


def compute_for_bars(
    bars: list[Bar],
    ema_fast_period: int,
    ema_slow_period: int,
    ma_trend_period: int,
    atr_period: int,
    adx_period: int,
    bb_period: int,
    bb_std: float,
    rsi_period: int,
) -> PrecomputedIndicators:
    """为单只股票的完整历史 K 线计算所有指标。"""
    n = len(bars)
    if n == 0:
        raise ValueError("Cannot compute indicators for empty bar list")

    closes = np.array([b.close / 10_000 for b in bars], dtype=np.float64)
    highs = np.array([b.high / 10_000 for b in bars], dtype=np.float64)
    lows = np.array([b.low / 10_000 for b in bars], dtype=np.float64)

    alpha_fast = 2.0 / (ema_fast_period + 1)
    ema_fast = np.empty(n, dtype=np.float64)
    ema_fast[0] = closes[0]
    for i in range(1, n):
        ema_fast[i] = alpha_fast * closes[i] + (1 - alpha_fast) * ema_fast[i - 1]

    alpha_slow = 2.0 / (ema_slow_period + 1)
    ema_slow = np.empty(n, dtype=np.float64)
    ema_slow[0] = closes[0]
    for i in range(1, n):
        ema_slow[i] = alpha_slow * closes[i] + (1 - alpha_slow) * ema_slow[i - 1]

    ma = np.empty(n, dtype=np.float64)
    ma[: ma_trend_period - 1] = np.nan
    for i in range(ma_trend_period - 1, n):
        ma[i] = np.mean(closes[i - ma_trend_period + 1 : i + 1])

    atr, adx = _vectorized_adx(highs, lows, closes, atr_period, adx_period)

    bb_upper = np.empty(n, dtype=np.float64)
    bb_middle = np.empty(n, dtype=np.float64)
    bb_lower = np.empty(n, dtype=np.float64)
    bb_upper[: bb_period - 1] = np.nan
    bb_middle[: bb_period - 1] = np.nan
    bb_lower[: bb_period - 1] = np.nan

    for i in range(bb_period - 1, n):
        window = closes[i - bb_period + 1 : i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=0)
        bb_middle[i] = mid
        bb_upper[i] = mid + bb_std * std
        bb_lower[i] = mid - bb_std * std

    rsi = np.empty(n, dtype=np.float64)
    rsi[:1] = 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = np.mean(gains[:rsi_period])
    avg_loss[0] = np.mean(losses[:rsi_period])

    for i in range(1, n):
        if i >= rsi_period:
            avg_gain[i] = (
                avg_gain[i - 1] * (rsi_period - 1) + gains[i - 1]
            ) / rsi_period
            avg_loss[i] = (
                avg_loss[i - 1] * (rsi_period - 1) + losses[i - 1]
            ) / rsi_period
        else:
            avg_gain[i] = np.mean(gains[: i + 1])
            avg_loss[i] = np.mean(losses[: i + 1])

    rs = np.zeros(n, dtype=np.float64)
    nonzero = avg_loss > 0
    rs[nonzero] = avg_gain[nonzero] / avg_loss[nonzero]
    rsi = np.where(avg_loss > 0, 100.0 - 100.0 / (1.0 + rs), 100.0)

    return PrecomputedIndicators(
        closes=closes,
        highs=highs,
        lows=lows,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ma=ma,
        atr=atr,
        adx=adx,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        rsi=rsi,
    )


def build_precomputed_cache(
    all_data: dict[str, list[Bar]],
    ema_fast_period: int,
    ema_slow_period: int,
    ma_trend_period: int,
    atr_period: int,
    adx_period: int,
    bb_period: int,
    bb_std: float,
    rsi_period: int,
) -> PrecomputedIndicatorCache:
    """为所有股票构建预计算指标缓存。"""
    cache = PrecomputedIndicatorCache()
    for symbol, bars in all_data.items():
        if not bars:
            continue
        try:
            indicators = compute_for_bars(
                bars,
                ema_fast_period=ema_fast_period,
                ema_slow_period=ema_slow_period,
                ma_trend_period=ma_trend_period,
                atr_period=atr_period,
                adx_period=adx_period,
                bb_period=bb_period,
                bb_std=bb_std,
                rsi_period=rsi_period,
            )
            cache.set(symbol, indicators)
        except Exception as e:
            logger.warning("Precomputation failed for %s: %s", symbol, e)
    return cache
