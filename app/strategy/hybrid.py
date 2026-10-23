from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.data.schema import Bar
from app.strategy.base import BaseStrategy, StatefulMixin
from app.strategy.components.timing import TimingComponent


@dataclass
class HybridSmallCapMAStrategy(BaseStrategy, StatefulMixin):
    ma_short: int = 5
    ma_long: int = 20
    min_cap_rank: int = 500
    max_cap_rank: int = 2000
    top_n: int = 50
    freq: str = "W"

    timing: TimingComponent | None = None

    def __post_init__(self) -> None:
        if self.timing is None:
            self.timing = TimingComponent(
                ma_short=self.ma_short,
                ma_long=self.ma_long,
                window=self.ma_long,
            )

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        if self.timing is None:
            return []
        self.timing.update_history(bars)

        if not self.timing.is_ready:
            return []

        match self.freq:
            case "W":
                if ts.weekday() != 0:
                    return []
            case "M":
                if ts.day != 1:
                    return []
            case "D":
                pass
            case _:
                pass

        # 获取候选股票（简化版，实际应使用因子排名）
        # 这里使用bars中的所有股票作为候选
        factor_selected = set(bars.keys())
        timing_signals = self.timing.get_signals(factor_selected)
        final_signals = factor_selected & timing_signals

        result: list[Signal] = []
        for symbol in final_signals:
            result.append(
                Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_type=SignalType.ENTRY,
                    strength=1.0,
                    reason="SMALL_CAP + MA_CROSS",
                    source=self.__class__.__name__,
                    timestamp=ts,
                )
            )

        return result

    def get_state(self) -> dict:
        state = StatefulMixin.get_state(self) if hasattr(self, "get_state") else {}
        if self.timing:
            state["timing"] = self.timing.get_state()
        return state

    def set_state(self, state: dict) -> None:
        if self.timing and "timing" in state:
            self.timing.set_state(state["timing"])
