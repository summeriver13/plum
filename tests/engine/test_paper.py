"""Tests for paper trading module."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

import app.rule.trading.calculator as calc_module
from app.data.schema import Bar, Order, OrderStatus
from app.engine.backtest.portfolio import Portfolio
from app.engine.factory import EngineFactory
from app.engine.paper.broker import PaperBroker, PaperBrokerConfig
from app.engine.paper.clock import PaperClock, PaperClockConfig
from app.engine.paper.data_feed import PaperDataFeed
from app.engine.paper.environment import PaperEnvironment


def _make_bar(symbol: str, ts: datetime, close: Decimal = Decimal("10.0")) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        amount=close * 1000,
        pre_close=close,
    )


class TestPaperClock:
    def test_now_returns_datetime(self):
        clock = PaperClock()
        result = clock.now()
        assert isinstance(result, datetime)

    def test_is_market_open_trading_hours(self):
        config = PaperClockConfig(trading_hours_only=True)
        clock = PaperClock(config)

        ts = datetime(2024, 1, 15, 10, 30)
        with patch.object(
            calc_module.ASHARE_RULES, "is_trading_day", return_value=True
        ):
            assert clock.is_market_open(ts) is True

    def test_is_market_closed_weekend(self):
        config = PaperClockConfig(trading_hours_only=True)
        clock = PaperClock(config)

        ts = datetime(2024, 1, 13, 10, 30)
        with patch.object(
            calc_module.ASHARE_RULES, "is_trading_day", return_value=False
        ):
            assert clock.is_market_open(ts) is False

    def test_is_market_closed_lunch(self):
        config = PaperClockConfig(trading_hours_only=True)
        clock = PaperClock(config)

        ts = datetime(2024, 1, 15, 12, 0)
        with patch.object(
            calc_module.ASHARE_RULES, "is_trading_day", return_value=True
        ):
            assert clock.is_market_open(ts) is False

    def test_is_market_open_non_trading_hours(self):
        config = PaperClockConfig(trading_hours_only=True)
        clock = PaperClock(config)

        ts = datetime(2024, 1, 15, 20, 0)
        with patch.object(
            calc_module.ASHARE_RULES, "is_trading_day", return_value=True
        ):
            assert clock.is_market_open(ts) is False

    def test_trading_hours_always_open(self):
        config = PaperClockConfig(trading_hours_only=False)
        clock = PaperClock(config)

        ts = datetime(2024, 1, 15, 20, 0)
        with patch.object(
            calc_module.ASHARE_RULES, "is_trading_day", return_value=True
        ):
            assert clock.is_market_open(ts) is True

    def test_next_bar_time(self):
        clock = PaperClock()
        result = clock.next_bar_time()
        assert isinstance(result, datetime)


class TestPaperDataFeed:
    def test_subscribe(self):
        feed = PaperDataFeed()
        feed.subscribe(["600519.SH", "600036.SH"])
        assert "600519.SH" in feed.get_subscribed()
        assert "600036.SH" in feed.get_subscribed()

    def test_update_and_get_current_bars(self):
        feed = PaperDataFeed()
        ts = datetime(2024, 1, 1)
        bars = {
            "600519.SH": _make_bar("600519.SH", ts),
        }
        feed.update_bars(bars)
        result = feed.get_current_bars()
        assert "600519.SH" in result

    def test_get_current_bars_returns_copy(self):
        feed = PaperDataFeed()
        ts = datetime(2024, 1, 1)
        bars = {"600519.SH": _make_bar("600519.SH", ts)}
        feed.update_bars(bars)
        result1 = feed.get_current_bars()
        result1["000001.SH"] = _make_bar("000001.SH", ts)
        result2 = feed.get_current_bars()
        assert "000001.SH" not in result2

    def test_tick_callback(self):
        feed = PaperDataFeed()
        callback_called = []

        def cb(bars):
            callback_called.append(bars)

        feed.on_tick(cb)
        ts = datetime(2024, 1, 1)
        bars = {"600519.SH": _make_bar("600519.SH", ts)}
        feed.update_bars(bars)
        assert len(callback_called) == 1
        assert "600519.SH" in callback_called[0]

    def test_get_bars_with_catalog(self):
        mock_catalog = MagicMock()
        mock_catalog.load_bars.return_value = [
            _make_bar("600519.SH", datetime(2024, 1, 1))
        ]
        feed = PaperDataFeed(catalog=mock_catalog)
        result = feed.get_bars(
            ["600519.SH"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )
        assert "600519.SH" in result

    def test_get_bars_without_catalog(self):
        feed = PaperDataFeed()
        result = feed.get_bars(
            ["600519.SH"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )
        assert result == {}


class TestPaperBroker:
    def test_submit_order(self):
        config = PaperBrokerConfig(fill_delay_ms=(10, 50), fill_rate=1.0)
        broker = PaperBroker(config)
        ts = datetime(2024, 1, 1)
        broker.set_current_bars({"600519.SH": _make_bar("600519.SH", ts)})

        order = Order(
            order_id="test",
            symbol="600519.SH",
            qty=100,
            status=OrderStatus.PENDING,
            created_ts=ts,
        )
        order_id = broker.submit_order(order)
        assert order_id.startswith("PP_")

    def test_cancel_order(self):
        broker = PaperBroker()
        ts = datetime(2024, 1, 1)
        broker.set_current_bars({"600519.SH": _make_bar("600519.SH", ts)})

        order = Order(
            order_id="test",
            symbol="600519.SH",
            qty=100,
            status=OrderStatus.PENDING,
            created_ts=ts,
        )
        order_id = broker.submit_order(order)
        result = broker.cancel_order(order_id)
        assert result is True

    def test_cancel_nonexistent_order(self):
        broker = PaperBroker()
        result = broker.cancel_order("nonexistent")
        assert result is False

    def test_get_pending_orders(self):
        broker = PaperBroker()
        result = broker.get_pending_orders()
        assert isinstance(result, list)

    def test_on_fill_callback(self):
        config = PaperBrokerConfig(fill_delay_ms=(10, 50), fill_rate=1.0)
        broker = PaperBroker(config)
        ts = datetime(2024, 1, 1)
        broker.set_current_bars({"600519.SH": _make_bar("600519.SH", ts)})

        filled_orders = []
        broker.on_fill(lambda o: filled_orders.append(o))

        order = Order(
            order_id="test",
            symbol="600519.SH",
            qty=100,
            status=OrderStatus.PENDING,
            created_ts=ts,
        )
        broker.submit_order(order)

        import time

        time.sleep(0.2)

        assert len(filled_orders) == 1
        assert filled_orders[0].status == OrderStatus.FILLED

    def test_submit_order_no_bar_raises(self):
        broker = PaperBroker()
        ts = datetime(2024, 1, 1)
        order = Order(
            order_id="test",
            symbol="600519.SH",
            qty=100,
            status=OrderStatus.PENDING,
            created_ts=ts,
        )
        with patch.object(broker, "_simulate_fill_async"):
            with pytest.raises(ValueError, match="No bar data"):
                broker.submit_order(order)


class TestPaperEnvironment:
    def test_create_environment(self):
        env = PaperEnvironment()
        assert env.env_type == "paper"
        assert isinstance(env.clock, PaperClock)
        assert isinstance(env.data_feed, PaperDataFeed)
        assert isinstance(env.broker, PaperBroker)

    def test_initialize(self):
        env = PaperEnvironment()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        env.initialize(portfolio)
        assert env.sync_portfolio() is portfolio

    def test_subscribe(self):
        env = PaperEnvironment()
        env.subscribe(["600519.SH"])
        assert "600519.SH" in env.data_feed.get_subscribed()

    def test_on_bar_callback(self):
        env = PaperEnvironment()
        called = []

        def cb(ts, bars, portfolio):
            called.append((ts, bars))

        env.on_bar(cb)
        assert env._bar_callback is cb

    def test_sync_portfolio_not_initialized_raises(self):
        env = PaperEnvironment()
        with pytest.raises(RuntimeError, match="Portfolio not initialized"):
            env.sync_portfolio()

    def test_get_pending_orders(self):
        env = PaperEnvironment()
        result = env.get_pending_orders()
        assert isinstance(result, list)


class TestEngineFactory:
    def test_create_backtest(self):
        from app.engine.backtest.engine import Engine
        from app.engine.backtest.portfolio import Portfolio
        from app.engine.backtest.scenarios.commission import AShareCommissionModel

        env = EngineFactory.create(
            "backtest",
            portfolio=Portfolio(initial_cash=Decimal("1000000")),
            catalog=None,
            commission=AShareCommissionModel(),
            enable_delisting_protection=False,
        )
        assert isinstance(env, Engine)

    def test_create_paper(self):
        env = EngineFactory.create("paper")
        assert isinstance(env, PaperEnvironment)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown environment type"):
            EngineFactory.create("unknown")

    def test_available_environments(self):
        envs = EngineFactory.available_environments()
        assert "backtest" in envs
        assert "paper" in envs
        assert "live" in envs
