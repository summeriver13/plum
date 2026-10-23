"""技术指标计算器 — 支持增量 EMA 和预计算缓存。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.factors.precomputed import PrecomputedIndicatorCache

if TYPE_CHECKING:
    from app.data.schema import Bar


@dataclass
class IndicatorCalculator:
    """技术指标计算器。

    功能：
    - 增量计算 EMA、MA、ATR、ADX、布林带、RSI
    - 支持预计算缓存 O(1) 查找
    - 支持状态保存/恢复
    """

    ema_fast_period: int = 14
    ema_slow_period: int = 25
    ma_trend_period: int = 200
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14

    _ema_fast_k: float = field(init=False)
    _ema_slow_k: float = field(init=False)
    _history: dict[str, deque] = field(default_factory=dict)
    _ema_fast: dict[str, float] = field(default_factory=dict)
    _ema_slow: dict[str, float] = field(default_factory=dict)
    _precomputed: PrecomputedIndicatorCache | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._ema_fast_k = 2 / (self.ema_fast_period + 1)
        self._ema_slow_k = 2 / (self.ema_slow_period + 1)

    def attach_precomputed(self, cache: PrecomputedIndicatorCache) -> None:
        """附加预计算指标缓存以实现 O(1) 查找。"""
        self._precomputed = cache

    def update(self, symbol: str, bar: Bar) -> None:
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=1000)

        history = self._history[symbol]
        history.append(bar)

        prices = [b.close / 10_000 for b in history]

        if symbol not in self._ema_fast:
            self._ema_fast[symbol] = 0.0
            self._ema_slow[symbol] = 0.0

        if len(prices) >= self.ema_fast_period:
            if self._ema_fast[symbol] == 0.0:
                self._ema_fast[symbol] = (
                    sum(prices[: self.ema_fast_period]) / self.ema_fast_period
                )
            else:
                self._ema_fast[symbol] = (
                    prices[-1] - self._ema_fast[symbol]
                ) * self._ema_fast_k + self._ema_fast[symbol]

        if len(prices) >= self.ema_slow_period:
            if self._ema_slow[symbol] == 0.0:
                self._ema_slow[symbol] = (
                    sum(prices[: self.ema_slow_period]) / self.ema_slow_period
                )
            else:
                self._ema_slow[symbol] = (
                    prices[-1] - self._ema_slow[symbol]
                ) * self._ema_slow_k + self._ema_slow[symbol]

    def calculate(self, symbol: str, warmup_bars: int) -> dict[str, float] | None:
        if symbol not in self._history:
            return None

        history = self._history[symbol]
        n = len(history)
        if n < warmup_bars:
            return None

        if self._precomputed is not None:
            precomp = self._precomputed.get(symbol)
            if precomp is not None:
                idx = n - 1
                if idx < 0 or idx >= precomp.n:
                    return None
                return {
                    "price": float(precomp.closes[idx]),
                    "ma": float(precomp.ma[idx]),
                    "ema_fast": float(precomp.ema_fast[idx]),
                    "ema_slow": float(precomp.ema_slow[idx]),
                    "atr": float(precomp.atr[idx]),
                    "adx": float(precomp.adx[idx]),
                    "bb_upper": float(precomp.bb_upper[idx]),
                    "bb_middle": float(precomp.bb_middle[idx]),
                    "bb_lower": float(precomp.bb_lower[idx]),
                    "rsi": float(precomp.rsi[idx]),
                }

        closes = [b.close / 10_000 for b in history]
        ma = sum(closes[-self.ma_trend_period :]) / self.ma_trend_period

        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, n):
            high = history[i].high / 10_000
            low = history[i].low / 10_000
            prev_close = history[i - 1].close / 10_000

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

            plus_dm = (
                max(high - prev_close, 0)
                if (high - prev_close) > (prev_close - low)
                else 0
            )
            minus_dm = (
                max(prev_close - low, 0)
                if (prev_close - low) > (high - prev_close)
                else 0
            )
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        atr = sum(tr_list[-self.atr_period :]) / self.atr_period
        adx = self._calculate_adx(tr_list, plus_dm_list, minus_dm_list)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(closes)
        rsi = self._calculate_rsi(closes)

        return {
            "price": closes[-1],
            "ma": ma,
            "ema_fast": self._ema_fast.get(symbol, 0.0),
            "ema_slow": self._ema_slow.get(symbol, 0.0),
            "atr": atr,
            "adx": adx,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "rsi": rsi,
        }

    def _calculate_adx(
        self,
        tr_list: list[float],
        plus_dm_list: list[float],
        minus_dm_list: list[float],
    ) -> float:
        if len(tr_list) < self.adx_period * 2:
            return 0.0

        smoothed_tr = sum(tr_list[: self.adx_period])
        smoothed_plus_dm = sum(plus_dm_list[: self.adx_period])
        smoothed_minus_dm = sum(minus_dm_list[: self.adx_period])

        dx_values = []
        for i in range(self.adx_period, len(tr_list)):
            smoothed_tr = smoothed_tr - smoothed_tr / self.adx_period + tr_list[i]
            smoothed_plus_dm = (
                smoothed_plus_dm - smoothed_plus_dm / self.adx_period + plus_dm_list[i]
            )
            smoothed_minus_dm = (
                smoothed_minus_dm
                - smoothed_minus_dm / self.adx_period
                + minus_dm_list[i]
            )

            if smoothed_tr > 0:
                plus_di = (smoothed_plus_dm / smoothed_tr) * 100
                minus_di = (smoothed_minus_dm / smoothed_tr) * 100
                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx = (abs(plus_di - minus_di) / di_sum) * 100
                    dx_values.append(dx)

        if len(dx_values) < self.adx_period:
            return 0.0

        return sum(dx_values[-self.adx_period :]) / self.adx_period

    def _calculate_bollinger(self, closes: list[float]) -> tuple[float, float, float]:
        if len(closes) < self.bb_period:
            return 0.0, 0.0, 0.0

        recent = closes[-self.bb_period :]
        middle = sum(recent) / len(recent)
        variance = sum((p - middle) ** 2 for p in recent) / len(recent)
        std = variance**0.5

        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std

        return upper, middle, lower

    def _calculate_rsi(self, closes: list[float]) -> float:
        if len(closes) < self.rsi_period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < self.rsi_period:
            return 50.0

        avg_gain = sum(gains[-self.rsi_period :]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period :]) / self.rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def get_state(self, symbol: str) -> dict | None:
        if symbol not in self._history:
            return None
        return {
            "history": list(self._history[symbol]),
            "ema_fast": self._ema_fast.get(symbol, 0.0),
            "ema_slow": self._ema_slow.get(symbol, 0.0),
        }

    def set_state(self, symbol: str, data: dict) -> None:
        self._history[symbol] = deque(data.get("history", []), maxlen=1000)
        self._ema_fast[symbol] = data.get("ema_fast", 0.0)
        self._ema_slow[symbol] = data.get("ema_slow", 0.0)
