from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.data.schema import Bar
from app.strategy.base import WindowedStrategy


@dataclass
class MACrossStrategy(WindowedStrategy):
    short_period: int = 5
    long_period: int = 20
    symbols: list[str] | None = None

    _last_signal: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        WindowedStrategy.__init__(self, window=self.long_period)

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        self.update_history(bars)

        if not self.is_ready:
            return []

        signals: list[Signal] = []
        targets = self.symbols or list(bars.keys())
        current_positions = {s: p.quantity for s, p in portfolio.positions.items()}

        for symbol in targets:
            if symbol not in bars:
                continue

            ma_short = self.ma(symbol, self.short_period)
            ma_long = self.ma(symbol, self.long_period)

            if ma_short == Decimal("0") or ma_long == Decimal("0"):
                continue

            current_qty = current_positions.get(symbol, 0)
            last_sig = self._last_signal.get(symbol, -1)

            if ma_short > ma_long and last_sig <= 0:
                if current_qty == 0:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            signal_type=SignalType.ENTRY,
                            strength=1.0,
                            reason="MA_CROSS_UP",
                            source=self.__class__.__name__,
                            timestamp=ts,
                        )
                    )
                self._last_signal[symbol] = 1

            elif ma_short < ma_long and last_sig >= 0:
                if current_qty > 0:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            direction=SignalDirection.FLAT,
                            signal_type=SignalType.EXIT,
                            reason="MA_CROSS_DOWN",
                            source=self.__class__.__name__,
                            timestamp=ts,
                        )
                    )
                self._last_signal[symbol] = -1

        return signals
