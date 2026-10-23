from __future__ import annotations

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from app.engine.backtest.models import Signal
from app.engine.backtest.portfolio import Portfolio
from app.data.schema import Bar


class Strategy(Protocol):
    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]: ...


@dataclass(frozen=True)
class StrategyContext:
    ts: datetime
    bars: dict[str, Bar]
    portfolio: Portfolio

    @property
    def positions(self) -> dict[str, int]:
        return {s: p.quantity for s, p in self.portfolio.positions.items()}

    def get_price(self, symbol: str) -> Decimal | None:
        bar = self.bars.get(symbol)
        return bar.close if bar else None

    def get_bar(self, symbol: str) -> Bar | None:
        return self.bars.get(symbol)


@dataclass
class StatefulMixin:
    """维护跨bar状态的策略混入类。

    Attributes:
        watched_symbols: 策略关注的股票集合。
            当定义（非空）时，引擎将过滤current_bars仅包含这些股票+当前持仓。
            显著提升聚焦型策略的性能。
    """

    strategy_id: str = ""
    warmup_period: int = 0
    watched_symbols: set[str] = field(default_factory=set)

    _state: dict = field(default_factory=dict)
    _state_buffer: dict = field(default_factory=dict)
    _last_updated: datetime | None = None

    def get_state(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "warmup_period": self.warmup_period,
            "last_updated": self._last_updated.isoformat()
            if self._last_updated
            else None,
            "data": self._state,
        }

    def set_state(self, state: dict) -> None:
        self._state = state.get("data", {})
        self._state_buffer.clear()
        last_updated = state.get("last_updated")
        if last_updated:
            self._last_updated = datetime.fromisoformat(last_updated)

    def update_state(self, key: str, value: object) -> None:
        self._state_buffer[key] = value

    def flush_state(self) -> None:
        if self._state_buffer:
            self._state.update(self._state_buffer)
            self._state_buffer.clear()
            self._last_updated = datetime.now()

    def clear_buffer(self) -> None:
        self._state_buffer.clear()

    def get_state_value(self, key: str, default: object = None) -> object:
        if key in self._state_buffer:
            return self._state_buffer[key]
        return self._state.get(key, default)


@dataclass
class RebalanceMixin:
    """再平衡控制。

    支持 hold_period 指定持仓周期:
        D      — 每天
        3D     — 每 3 个交易日
        5D     — 每 5 个交易日
        W      — 每周一
        2W     — 每隔一周的周一
        M      — 每月 1 号

    offset 分批进场:
        某策略 hold_period="3D", offset_list=[0,1,2]
        → 拆成 3 个虚拟子策略
        → offset=0: day0,3,6... 调仓
        → offset=1: day1,4,7... 调仓
        → offset=2: day2,5,8... 调仓
    """

    hold_period: str = "W"
    offset: int = 0
    cap_weight: float = 1.0

    # 内部状态
    _business_day_count: int = field(default=0, init=False, repr=False)
    _period_start_count: int = field(default=-1, init=False, repr=False)
    _last_rebalance_ts: datetime | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _parse_period_days(period: str) -> int | None:
        """解析 hold_period 为天数。

        Returns:
            None 表示不用天数计数（如 W/M），int 表示周期天数。
        """
        period = period.strip().upper()
        if period == "D":
            return 1
        if period.endswith("D"):
            try:
                return int(period[:-1])
            except ValueError:
                pass
        if period.endswith("W"):
            try:
                return int(period[:-1]) * 5
            except ValueError:
                pass
        return None  # W, M, 2W 等非整数天的格式

    def is_rebalance_date(self, ts: datetime) -> bool:
        """判断当前是否再平衡日。"""
        period = self.hold_period.strip().upper()

        # 幂等保护：同一天多次调用不重复计数
        if self._last_rebalance_ts and self._last_rebalance_ts.date() == ts.date():
            return self._last_result
        self._last_rebalance_ts = ts

        days = self._parse_period_days(period)

        # ── 按天计数模式（D / 3D / 5D） ──
        if days is not None:
            self._business_day_count += 1
            effective_count = self._business_day_count - 1 - self.offset

            if effective_count < 0:
                self._last_result = False
                return False

            if self._period_start_count < 0:
                self._period_start_count = effective_count
                self._last_result = True
                return True

            since_last = effective_count - self._period_start_count
            if since_last >= days:
                self._period_start_count = effective_count
                self._last_result = True
                return True
            self._last_result = False
            return False

        # ── 每周模式 ──
        if period == "W":
            self._last_result = ts.weekday() == 0
            return self._last_result

        # ── 每 2 周模式 ──
        if period == "2W":
            self._last_result = ts.weekday() == 0 and ts.isocalendar()[1] % 2 == 0
            return self._last_result

        # ── 每月模式 ──
        if period == "M":
            self._last_result = ts.day == 1
            return self._last_result

        # ── 未知格式 → 默认每天 ──
        self._last_result = True
        return True


@dataclass
class BaseStrategy(Strategy, StatefulMixin):
    @abstractmethod
    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]: ...


@dataclass
class MultiFactorSignalStrategy(BaseStrategy, RebalanceMixin):
    factor_names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    universe: set[str] = field(default_factory=set)
    top_n: int = 50
    freq: str = "W"
    equal_weight: bool = True

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        if not self.is_rebalance_date(ts):
            return []

        # factor_store 由外部注入，此处返回空信号
        # 实际使用时应通过 app.factors.store.FactorStore 实现
        return []


@dataclass
class WindowedStrategy(BaseStrategy, StatefulMixin):
    window: int = 20
    forward_fill: bool = True

    _history: dict[str, deque[Bar]] = field(default_factory=dict)
    _last_bar: dict[str, Bar] = field(default_factory=dict)
    _is_ready: bool = False

    def update_history(self, bars: dict[str, Bar]) -> None:
        for symbol, bar in bars.items():
            if bar.close > Decimal("0"):
                self._last_bar[symbol] = bar

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

    def get_history(self, symbol: str) -> list[Bar]:
        return list(self._history.get(symbol, []))

    def get_last_price(self, symbol: str) -> Decimal:
        bar = self._last_bar.get(symbol)
        if bar is None:
            return Decimal("0")
        return bar.close

    def ma(self, symbol: str, period: int | None = None) -> Decimal:
        if period is None:
            period = self.window
        history = self.get_history(symbol)
        if len(history) < period:
            return Decimal("0")
        prices = [b.close for b in history[-period:]]
        return sum(prices) / Decimal(period)

    def ema(self, symbol: str, period: int = 12) -> Decimal:
        history = self.get_history(symbol)
        if len(history) < period:
            return Decimal("0")
        multiplier = Decimal("2") / Decimal(period + 1)
        ema = history[0].close
        for bar in history[1:]:
            ema = (bar.close - ema) * multiplier + ema
        return ema

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        raise NotImplementedError("Subclass must implement on_bar")


MultiFactorStrategy = MultiFactorSignalStrategy


@dataclass
class PositionOptimizer:
    max_positions: int = 8
    min_position_value: Decimal = Decimal("5000")
    rebalance_threshold: float = 0.05

    def optimize(
        self,
        candidate_scores: list[tuple[str, float]],
        current_holdings: dict[str, int],
        prices: dict[str, Decimal],
    ) -> dict[str, int]:
        if not candidate_scores:
            return {s: qty for s, qty in current_holdings.items() if qty > 0}

        top_candidates = candidate_scores[: self.max_positions]

        current_symbols = set(current_holdings.keys())
        new_symbols = {s for s, _ in top_candidates}

        keep_symbols = current_symbols & new_symbols
        sell_symbols = current_symbols - new_symbols
        buy_symbols = new_symbols - current_symbols

        result: dict[str, int] = {}

        for symbol in keep_symbols:
            result[symbol] = current_holdings[symbol]

        for symbol in sell_symbols:
            qty = current_holdings.get(symbol, 0)
            if qty > 0:
                result[symbol] = 0

        if not buy_symbols:
            return result

        buy_scores = [(s, score) for s, score in top_candidates if s in buy_symbols]
        if not buy_scores:
            return result

        new_scores = dict(buy_scores)
        avg_score = sum(new_scores.values()) / len(new_scores)

        for symbol in buy_symbols:
            score = new_scores.get(symbol, 0)
            if score < avg_score * (1 - self.rebalance_threshold):
                continue

            price = prices.get(symbol, Decimal("0"))
            if price <= Decimal("0"):
                continue

            position_value = Decimal(self.max_positions) * self.min_position_value
            qty_float = position_value / price
            qty = int(qty_float / 100) * 100

            if qty >= 100:
                result[symbol] = qty

        return result
