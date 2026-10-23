"""可复用的策略组件。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from app.data.schema import Bar


@dataclass
class TimingComponent:
    """基于MA的通用择时模块。

    可与任何需要大盘择时的策略组合使用。
    被HybridSmallCapMAStrategy用于大盘择时。

    Example:
        timing = TimingComponent(ma_short=5, ma_long=20, window=20)
        timing.update_history(bars)
        if timing.is_ready:
            signals = timing.get_signals(universe)
    """

    ma_short: int = 5
    ma_long: int = 20
    window: int = 20

    _history: dict[str, deque[Bar]] = field(default_factory=dict)
    _last_signal: dict[str, int] = field(default_factory=dict)
    _is_ready: bool = False

    def update_history(self, bars: dict[str, Bar]) -> None:
        for symbol, bar in bars.items():
            if symbol not in self._history:
                self._history[symbol] = deque(maxlen=self.window)
                self._history[symbol].append(bar)
            else:
                self._history[symbol].append(bar)

        if len(self._history) > 0:
            sample_symbol = next(iter(self._history))
            if len(self._history[sample_symbol]) >= self.window:
                self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def ma(self, symbol: str, period: int) -> Decimal:
        history = self._history.get(symbol, deque())
        if len(history) < period:
            return Decimal("0")
        prices = [b.close for b in list(history)[-period:]]
        return sum(prices) / Decimal(period)

    def get_signals(self, symbols: set[str]) -> set[str]:
        signals: set[str] = set()
        for symbol in symbols:
            ma_short = self.ma(symbol, self.ma_short)
            ma_long = self.ma(symbol, self.ma_long)
            if ma_short > ma_long:
                signals.add(symbol)
                self._last_signal[symbol] = 1
            else:
                self._last_signal[symbol] = -1
        return signals

    def get_state(self) -> dict:
        return {
            "history": {k: list(v) for k, v in self._history.items()},
            "last_signal": dict(self._last_signal),
            "is_ready": self._is_ready,
        }

    def set_state(self, state: dict) -> None:
        self._history = {
            k: deque(v, maxlen=self.window) for k, v in state.get("history", {}).items()
        }
        self._last_signal = state.get("last_signal", {})
        self._is_ready = state.get("is_ready", False)
