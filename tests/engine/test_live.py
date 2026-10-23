"""Tests for live trading module."""

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar, OrderStatus
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.live.gateway.base import (
    AccountInfo,
    Direction,
    OrderRequest,
    OrderType,
)
from app.engine.live.gateway.mock import MockGateway
from app.engine.live.risk import RiskConfig, RiskGuard
from app.engine.live.syncer import AccountSyncer, SyncResult
from app.engine.live.ledger import TradeLedger
from app.engine.live.engine import LiveEnvironment


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


class TestMockGateway:
    def test_connect_disconnect(self):
        gw = MockGateway()
        assert gw.connect() is True
        assert gw.is_connected is True
        gw.disconnect()
        assert gw.is_connected is False

    def test_subscribe_unsubscribe(self):
        gw = MockGateway()
        gw.subscribe(["600519.SH"])
        assert "600519.SH" in gw._subscribed
        gw.unsubscribe(["600519.SH"])
        assert "600519.SH" not in gw._subscribed

    def test_send_order_buy(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts, Decimal("10.0"))})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        resp = gw.send_order(order)
        assert resp.status == OrderStatus.FILLED
        assert resp.filled_quantity == 100

    def test_send_order_sell_no_position(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts)})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        resp = gw.send_order(order)
        assert resp.status == OrderStatus.REJECTED
        assert resp.reject_reason in ("no_position", "t1_restriction")

    def test_send_order_not_subscribed(self):
        gw = MockGateway()
        gw.connect()
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts)})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        resp = gw.send_order(order)
        assert resp.status == OrderStatus.REJECTED
        assert "not_subscribed" in resp.reject_reason

    def test_send_order_not_connected(self):
        gw = MockGateway()
        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        resp = gw.send_order(order)
        assert resp.status == OrderStatus.REJECTED
        assert "not_connected" in resp.reject_reason

    def test_cancel_order(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts)})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        gw.send_order(order)
        resp = gw.cancel_order("MOCK_1")
        assert resp.status == OrderStatus.CANCELLED

    def test_get_positions(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts, Decimal("10.0"))})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        gw.send_order(order)
        positions = gw.get_positions()
        assert positions["600519.SH"] == 100

    def test_get_account_info(self):
        gw = MockGateway()
        gw.connect()
        info = gw.get_account_info()
        assert isinstance(info, AccountInfo)
        assert info.cash == gw.initial_cash

    def test_on_trade_callback(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts, Decimal("10.0"))})

        trades = []
        gw.on_trade(lambda s, q, p, b: trades.append((s, q, p, b)))

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        gw.send_order(order)
        assert len(trades) == 1
        assert trades[0][0] == "600519.SH"
        assert trades[0][1] == 100

    def test_end_of_day(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts, Decimal("10.0"))})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        gw.send_order(order)
        assert gw._available["600519.SH"] == 100
        gw.end_of_day()
        assert "600519.SH" not in gw._available

    def test_reset(self):
        gw = MockGateway()
        gw.connect()
        gw.subscribe(["600519.SH"])
        ts = datetime(2024, 1, 1)
        gw.update_bars({"600519.SH": _make_bar("600519.SH", ts, Decimal("10.0"))})

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        gw.send_order(order)
        gw.reset()
        assert gw._cash == gw.initial_cash
        assert len(gw._positions) == 0


class TestRiskGuard:
    def test_pre_check_pass(self):
        guard = RiskGuard()
        allowed, reason = guard.pre_check(
            symbol="600519.SH",
            direction="BUY",
            quantity=100,
            price=100_000,
            current_positions={},
            portfolio_value=10_000_000_000,
        )
        assert allowed is True

    def test_pre_check_circuit_broken(self):
        guard = RiskGuard()
        guard._circuit_broken = True
        allowed, reason = guard.pre_check(
            symbol="600519.SH",
            direction="BUY",
            quantity=100,
            price=100_000,
            current_positions={},
            portfolio_value=10_000_000_000,
        )
        assert allowed is False
        assert "circuit_breaker" in reason

    def test_pre_check_position_limit(self):
        config = RiskConfig(max_position_pct=0.01)
        guard = RiskGuard(config=config)
        allowed, reason = guard.pre_check(
            symbol="600519.SH",
            direction="BUY",
            quantity=100_000_000,
            price=100_000,
            current_positions={},
            portfolio_value=1_000_000_000,
        )
        assert allowed is False

    def test_check_daily_loss(self):
        guard = RiskGuard()
        guard._peak_value = 10_000_000_000
        guard._daily_loss = -600_000_000
        allowed, limit = guard.check_daily_loss()
        assert allowed is False

    def test_reset_daily(self):
        guard = RiskGuard()
        guard._daily_loss = 100
        guard._peak_value = 200
        guard._order_timestamps.append(datetime.now())
        guard.reset_daily()
        assert guard._daily_loss == 0
        assert guard._peak_value == 0
        assert len(guard._order_timestamps) == 0

    def test_reset_circuit_breaker(self):
        guard = RiskGuard()
        guard._circuit_broken = True
        guard.reset_circuit_breaker()
        assert guard.is_circuit_broken is False


class TestAccountSyncer:
    def test_sync_no_diff(self):
        gw = MockGateway(initial_cash=4_000_000)
        gw.connect()
        portfolio = Portfolio(initial_cash=Decimal("4000000"))
        syncer = AccountSyncer(gateway=gw, portfolio=portfolio)

        result = syncer.sync()
        assert isinstance(result, SyncResult)
        assert result.needs_recovery is False

    def test_sync_with_diff(self):
        gw = MockGateway(initial_cash=40_000_0000)
        gw.connect()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        syncer = AccountSyncer(gateway=gw, portfolio=portfolio)

        result = syncer.sync()
        assert result.needs_recovery is True
        assert result.cash_diff != 0

    def test_recover_to_broker(self):
        gw = MockGateway(initial_cash=40_000_0000)
        gw.connect()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        syncer = AccountSyncer(gateway=gw, portfolio=portfolio)

        syncer.recover_to_broker()
        assert portfolio.cash == 40_000_0000


class TestTradeLedger:
    def test_write_and_scan(self, tmp_path):
        db = tmp_path / "test_ledger.sqlite"
        ledger = TradeLedger(db)

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        ledger.write_order(order)

        pending = ledger.scan_pending()
        assert len(pending) == 1
        assert pending[0]["symbol"] == "600519.SH"

    def test_update_status(self, tmp_path):
        db = tmp_path / "test_ledger.sqlite"
        ledger = TradeLedger(db)

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        ledger.write_order(order)
        ledger.update_status(order.order_id, "filled")

        pending = ledger.scan_pending()
        assert len(pending) == 0

    def test_list_all(self, tmp_path):
        db = tmp_path / "test_ledger.sqlite"
        ledger = TradeLedger(db)

        order = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        ledger.write_order(order)

        all_orders = ledger.list_all()
        assert len(all_orders) == 1


class TestLiveEnvironment:
    def test_create_environment(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)
        assert env.env_type == "live"
        assert env.gateway is gw
        assert env.portfolio is portfolio

    def test_set_strategy(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)

        strategy = object()
        env.set_strategy(strategy)
        assert env._strategy is strategy

    def test_set_risk_guard(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)

        guard = RiskGuard()
        env.set_risk_guard(guard)
        assert env._risk_guard is guard

    def test_start_stop(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)

        env.start(["600519.SH"])
        assert env.is_running is True
        env.stop()
        assert env.is_running is False

    def test_start_already_running(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)

        env.start(["600519.SH"])
        env.start(["600519.SH"])
        env.stop()

    def test_stop_not_running(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)
        env.stop()

    def test_notify_callback_error(self):
        gw = MockGateway()
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(gateway=gw, portfolio=portfolio, commission=commission)

        def bad_callback(event_type, **kwargs):
            raise ValueError("test error")

        env.set_notify_callback(bad_callback)
        env._notify(
            "trade", symbol="600519.SH", quantity=100, price_int=100000, is_buy=True
        )
