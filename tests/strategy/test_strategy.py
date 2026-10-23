"""Strategy module tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar
from app.engine.backtest.models import SignalType
from app.engine.backtest.portfolio import Portfolio
from app.strategy.base import (
    PositionOptimizer,
    RebalanceMixin,
    StatefulMixin,
    StrategyContext,
    WindowedStrategy,
)
from app.strategy.buy_and_hold import BuyAndHoldStrategy, PositionThesis
from app.strategy.components.timing import TimingComponent
from app.strategy.hybrid import HybridSmallCapMAStrategy
from app.strategy.macross import MACrossStrategy
from app.strategy.trend_following import MarketRegime, TrendFollowingStrategy


def _make_bar(
    close: float = 10.0,
    pre_close: float = 10.0,
    high: float = 10.5,
    low: float = 9.5,
    open_price: float = 10.0,
    symbol: str = "600519.SH",
    ts: datetime = datetime(2024, 1, 1),
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=Decimal(str(open_price)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        pre_close=Decimal(str(pre_close)),
        volume=1000000,
        amount=Decimal("10000000"),
    )


def _make_bars(
    prices: list[float],
    symbol: str = "600519.SH",
    start_date: datetime = datetime(2024, 1, 1),
) -> dict[str, Bar]:
    """创建单个Bar的字典（仅最后一个价格）。用于简单测试。"""
    if not prices:
        return {}
    price = prices[-1]
    pre = prices[-2] if len(prices) > 1 else price
    return {
        symbol: Bar(
            symbol=symbol,
            timestamp=start_date,
            open=Decimal(str(price)),
            high=Decimal(str(price * 1.02)),
            low=Decimal(str(price * 0.98)),
            close=Decimal(str(price)),
            pre_close=Decimal(str(pre)),
            volume=1000000,
            amount=Decimal("10000000"),
        )
    }


def _make_bars_series(
    prices: list[float],
    symbol: str = "600519.SH",
    start_date: datetime = datetime(2024, 1, 1),
) -> list[dict[str, Bar]]:
    """创建多个Bar字典列表，每个价格对应一个bar字典，使用不同日期。"""
    result = []
    for i, price in enumerate(prices):
        pre = prices[i - 1] if i > 0 else price
        bar = Bar(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1 + i),
            open=Decimal(str(price)),
            high=Decimal(str(price * 1.02)),
            low=Decimal(str(price * 0.98)),
            close=Decimal(str(price)),
            pre_close=Decimal(str(pre)),
            volume=1000000,
            amount=Decimal("10000000"),
        )
        result.append({symbol: bar})
    return result


class TestStatefulMixin:
    def test_get_state_empty(self) -> None:
        mixin = StatefulMixin(strategy_id="test")
        state = mixin.get_state()
        assert state["strategy_id"] == "test"
        assert state["data"] == {}

    def test_set_state(self) -> None:
        mixin = StatefulMixin()
        mixin.set_state(
            {"data": {"key": "value"}, "last_updated": "2024-01-01T00:00:00"}
        )
        assert mixin.get_state_value("key") == "value"

    def test_update_and_flush_state(self) -> None:
        mixin = StatefulMixin()
        mixin.update_state("counter", 42)
        assert mixin.get_state_value("counter") == 42
        mixin.flush_state()
        assert mixin._state["counter"] == 42
        assert mixin._state_buffer == {}

    def test_clear_buffer(self) -> None:
        mixin = StatefulMixin()
        mixin.update_state("temp", "data")
        mixin.clear_buffer()
        assert mixin.get_state_value("temp") is None


class TestRebalanceMixin:
    def test_weekly_rebalance_monday(self) -> None:
        mixin = RebalanceMixin(hold_period="W")
        monday = datetime(2024, 1, 1)  # Monday
        assert mixin.is_rebalance_date(monday) is True

    def test_weekly_rebalance_friday(self) -> None:
        mixin = RebalanceMixin(hold_period="W")
        friday = datetime(2024, 1, 5)  # Friday
        assert mixin.is_rebalance_date(friday) is False

    def test_monthly_rebalance_first(self) -> None:
        mixin = RebalanceMixin(hold_period="M")
        first = datetime(2024, 1, 1)
        assert mixin.is_rebalance_date(first) is True

    def test_monthly_rebalance_other(self) -> None:
        mixin = RebalanceMixin(hold_period="M")
        other = datetime(2024, 1, 15)
        assert mixin.is_rebalance_date(other) is False

    def test_daily_rebalance(self) -> None:
        mixin = RebalanceMixin(hold_period="D")
        assert mixin.is_rebalance_date(datetime(2024, 1, 1)) is True

    def test_period_days_rebalance_every_three_calls(self) -> None:
        """3D：第 1 次为起点，其后每 3 个计数日再平衡一次。"""
        mixin = RebalanceMixin(hold_period="3D")
        days = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
        ]
        assert [mixin.is_rebalance_date(day) for day in days] == [
            True,
            False,
            False,
            True,
        ]

    def test_same_day_call_is_idempotent(self) -> None:
        """同一天重复调用不重复计数（3D 第二次同日调用仍为起点）。"""
        mixin = RebalanceMixin(hold_period="3D")
        day = datetime(2024, 1, 1)
        assert mixin.is_rebalance_date(day) is True
        assert mixin.is_rebalance_date(day) is True


class TestWindowedStrategy:
    def test_update_history_fills_window(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            strategy.update_history(bars)
        assert strategy.is_ready is True

    def test_update_history_not_ready(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars = _make_bars([10.0])
        strategy.update_history(bars)
        assert strategy.is_ready is False

    def test_get_last_price(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars = _make_bars([15.5])
        strategy.update_history(bars)
        price = strategy.get_last_price("600519.SH")
        assert price == Decimal("15.5")

    def test_get_last_price_no_bar(self) -> None:
        strategy = WindowedStrategy(window=5)
        price = strategy.get_last_price("600519.SH")
        assert price == Decimal("0")

    def test_ma_calculation(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            strategy.update_history(bars)
        ma = strategy.ma("600519.SH", 5)
        assert ma == Decimal("12.0")

    def test_ma_insufficient_data(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars = _make_bars([10.0])
        strategy.update_history(bars)
        ma = strategy.ma("600519.SH", 5)
        assert ma == Decimal("0")

    def test_ema_calculation(self) -> None:
        strategy = WindowedStrategy(window=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            strategy.update_history(bars)
        ema = strategy.ema("600519.SH", 3)
        assert ema > Decimal("0")


class TestStrategyContext:
    def test_positions_property(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        portfolio.apply_trade("600519.SH", 100, Decimal("10.0"), Decimal("10"))
        bars = _make_bars([10.0])
        ctx = StrategyContext(
            ts=datetime(2024, 1, 1),
            bars=bars,
            portfolio=portfolio,
        )
        assert ctx.positions == {"600519.SH": 100}

    def test_get_price(self) -> None:
        bars = _make_bars([15.5])
        ctx = StrategyContext(
            ts=datetime(2024, 1, 1),
            bars=bars,
            portfolio=Portfolio(initial_cash=Decimal("100000")),
        )
        price = ctx.get_price("600519.SH")
        assert price == Decimal("15.5")

    def test_get_bar(self) -> None:
        bars = _make_bars([10.0])
        ctx = StrategyContext(
            ts=datetime(2024, 1, 1),
            bars=bars,
            portfolio=Portfolio(initial_cash=Decimal("100000")),
        )
        bar = ctx.get_bar("600519.SH")
        assert bar is not None
        assert bar.close == Decimal("10.0")


class TestPositionOptimizer:
    def test_optimize_new_positions(self) -> None:
        optimizer = PositionOptimizer(
            max_positions=3, min_position_value=Decimal("5000"), rebalance_threshold=0.5
        )
        candidates = [("600519.SH", 0.9), ("600036.SH", 0.8), ("000001.SH", 0.7)]
        result = optimizer.optimize(
            candidates,
            {},
            {
                "600519.SH": Decimal("10.0"),
                "600036.SH": Decimal("20.0"),
                "000001.SH": Decimal("30.0"),
            },
        )
        assert len(result) == 3

    def test_optimize_empty_candidates(self) -> None:
        optimizer = PositionOptimizer()
        result = optimizer.optimize([], {"600519.SH": 100}, {})
        assert result == {"600519.SH": 100}

    def test_optimize_keep_existing(self) -> None:
        optimizer = PositionOptimizer(
            max_positions=2, min_position_value=Decimal("5000")
        )
        candidates = [("600519.SH", 0.9), ("600036.SH", 0.8)]
        result = optimizer.optimize(
            candidates,
            {"600519.SH": 100},
            {"600519.SH": Decimal("10.0"), "600036.SH": Decimal("20.0")},
        )
        assert "600519.SH" in result


class TestBuyAndHoldStrategy:
    def test_entry_signal_generated(self) -> None:
        strategy = BuyAndHoldStrategy(min_entry_score=50, max_positions=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars = _make_bars([10.0])
        signals = strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) > 0

    def test_no_entry_when_full(self) -> None:
        strategy = BuyAndHoldStrategy(min_entry_score=50, max_positions=1)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        portfolio.apply_trade("600519.SH", 100, Decimal("10.0"), Decimal("10"))
        bars = _make_bars([10.0])
        signals = strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_exit_signal_on_zero_price(self) -> None:
        strategy = BuyAndHoldStrategy(max_positions=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        portfolio.apply_trade("600519.SH", 100, Decimal("10.0"), Decimal("10"))
        strategy._theses["600519.SH"] = PositionThesis(
            symbol="600519.SH",
            entry_date=datetime(2024, 1, 1),
            entry_price=Decimal("10.0"),
            entry_reason="TEST",
            entry_score=80,
        )
        bars = {"600519.SH": _make_bar(close=0.0)}
        signals = strategy.on_bar(datetime(2024, 1, 2), bars, portfolio)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) > 0

    def test_thesis_tracking(self) -> None:
        strategy = BuyAndHoldStrategy(min_entry_score=50, max_positions=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars = _make_bars([10.0])
        strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        theses = strategy.get_all_theses()
        assert len(theses) > 0

    def test_description(self) -> None:
        strategy = BuyAndHoldStrategy(min_entry_score=80, max_loss_tolerance=-0.5)
        desc = strategy.description
        assert "BuyAndHold" in desc
        assert "80" in desc


class TestMACrossStrategy:
    def test_no_signals_before_warmup(self) -> None:
        strategy = MACrossStrategy(short_period=5, long_period=10)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars = _make_bars([10.0])
        signals = strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        assert signals == []

    def test_entry_on_cross_up(self) -> None:
        strategy = MACrossStrategy(short_period=3, long_period=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        prices = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
        bars_list = _make_bars_series(prices)
        all_signals = []
        for i, bars in enumerate(bars_list):
            signals = strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)
            all_signals.extend(signals)
        assert any(s.signal_type == SignalType.ENTRY for s in all_signals)

    def test_exit_on_cross_down(self) -> None:
        strategy = MACrossStrategy(short_period=3, long_period=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        portfolio.apply_trade("600519.SH", 100, Decimal("10.0"), Decimal("10"))
        # First create uptrend to establish position
        up_prices = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]
        bars_list = _make_bars_series(up_prices)
        for i, bars in enumerate(bars_list):
            strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)

        # Now create downtrend
        down_prices = [14.0, 13.0, 12.0, 11.0, 10.0, 9.0]
        bars_list = _make_bars_series(down_prices)
        all_signals = []
        for i, bars in enumerate(bars_list):
            signals = strategy.on_bar(datetime(2024, 1, 11 + i), bars, portfolio)
            all_signals.extend(signals)
        assert any(s.signal_type == SignalType.EXIT for s in all_signals)


class TestTrendFollowingStrategy:
    def test_no_signals_before_warmup(self) -> None:
        strategy = TrendFollowingStrategy(ma_trend=5, bb_period=5, ema_slow=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars = _make_bars([10.0])
        signals = strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        assert signals == []

    def test_bull_regime_detection(self) -> None:
        strategy = TrendFollowingStrategy(
            ma_trend=5, bb_period=5, ema_slow=5, trend_buffer=0.02
        )
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        prices = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
        bars_list = _make_bars_series(prices)
        for i, bars in enumerate(bars_list):
            strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)
        regime = strategy.get_state_value("_prev_regime")
        assert regime == MarketRegime.BULL

    def test_bear_regime_detection(self) -> None:
        strategy = TrendFollowingStrategy(
            ma_trend=5, bb_period=5, ema_slow=5, trend_buffer=0.02
        )
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        prices = [10.0, 9.8, 9.6, 9.4, 9.2, 9.0]
        bars_list = _make_bars_series(prices)
        for i, bars in enumerate(bars_list):
            strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)
        regime = strategy.get_state_value("_prev_regime")
        assert regime == MarketRegime.BEAR


class TestTimingComponent:
    def test_update_history_fills_window(self) -> None:
        timing = TimingComponent(window=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            timing.update_history(bars)
        assert timing.is_ready is True

    def test_ma_calculation(self) -> None:
        timing = TimingComponent(window=5, ma_short=3, ma_long=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            timing.update_history(bars)
        ma = timing.ma("600519.SH", 5)
        assert ma == Decimal("12.0")

    def test_get_signals_bullish(self) -> None:
        timing = TimingComponent(window=5, ma_short=3, ma_long=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            timing.update_history(bars)
        signals = timing.get_signals({"600519.SH"})
        assert "600519.SH" in signals

    def test_state_serialization(self) -> None:
        timing = TimingComponent(window=5)
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for bars in bars_list:
            timing.update_history(bars)
        state = timing.get_state()
        new_timing = TimingComponent(window=5)
        new_timing.set_state(state)
        assert new_timing.is_ready == timing.is_ready


class TestHybridSmallCapMAStrategy:
    def test_no_signals_before_warmup(self) -> None:
        strategy = HybridSmallCapMAStrategy(ma_long=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars = _make_bars([10.0])
        signals = strategy.on_bar(datetime(2024, 1, 1), bars, portfolio)
        assert signals == []

    def test_signals_on_monday(self) -> None:
        from app.strategy.components.timing import TimingComponent

        timing = TimingComponent(ma_short=3, ma_long=5, window=5)
        strategy = HybridSmallCapMAStrategy(
            ma_short=3, ma_long=5, freq="W", timing=timing
        )
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        # Start from a Monday, with enough bars to warm up first
        # 2024-01-01 is Monday, 2024-01-08 is next Monday
        # First warmup with 5 bars (Mon-Fri)
        warmup_prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        bars_list = _make_bars_series(warmup_prices)
        for i, bars in enumerate(bars_list):
            strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)

        # Now test on next Monday (2024-01-08)
        monday_bars = {"600519.SH": _make_bar(close=15.0, ts=datetime(2024, 1, 8))}
        signals = strategy.on_bar(datetime(2024, 1, 8), monday_bars, portfolio)
        monday_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(monday_signals) > 0

    def test_state_serialization(self) -> None:
        strategy = HybridSmallCapMAStrategy(ma_long=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        bars_list = _make_bars_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for i, bars in enumerate(bars_list):
            strategy.on_bar(datetime(2024, 1, 1 + i), bars, portfolio)
        state = strategy.get_state()
        assert "timing" in state
