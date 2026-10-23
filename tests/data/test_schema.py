"""Tests for data schema (Decimal precision)."""

from datetime import datetime
from decimal import Decimal


from app.data.schema import (
    Bar,
    Quote,
    CostBreakdown,
    TradeRecord,
    Order,
    OrderStatus,
)


class TestBar:
    def test_create_bar_with_decimal(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.5"),
            high=Decimal("11.0"),
            low=Decimal("10.0"),
            close=Decimal("10.8"),
            volume=1000,
            amount=Decimal("10800"),
        )
        assert bar.close == Decimal("10.8")
        assert bar.close_price == Decimal("10.8")
        assert bar.volume == 1000

    def test_bar_from_dict(self):
        data = {
            "timestamp": "2024-01-01",
            "open": "10.5",
            "high": "11.0",
            "low": "10.0",
            "close": "10.8",
            "volume": "1000",
            "amount": "10800",
            "isST": "0",
        }
        bar = Bar.from_dict(data, "sh.600519")
        assert bar.close == Decimal("10.8")
        assert bar.volume == 1000
        assert bar.is_st is False

    def test_bar_from_dict_with_st(self):
        data = {
            "timestamp": "2024-01-01",
            "open": "10.5",
            "high": "11.0",
            "low": "10.0",
            "close": "10.8",
            "volume": "1000",
            "amount": "10800",
            "isST": "1",
        }
        bar = Bar.from_dict(data, "sz.000001")
        assert bar.is_st is True

    def test_bar_limit_up(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("11.0"),
            low=Decimal("10.0"),
            close=Decimal("11.0"),
            volume=1000,
            amount=Decimal("11000"),
            pre_close=Decimal("10.0"),
        )
        assert bar.is_limit_up(Decimal("0.10")) is True

    def test_bar_not_limit_up(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("10.5"),
            low=Decimal("10.0"),
            close=Decimal("10.5"),
            volume=1000,
            amount=Decimal("10500"),
            pre_close=Decimal("10.0"),
        )
        assert bar.is_limit_up(Decimal("0.10")) is False

    def test_bar_limit_down(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("10.0"),
            low=Decimal("9.0"),
            close=Decimal("9.0"),
            volume=1000,
            amount=Decimal("9000"),
            pre_close=Decimal("10.0"),
        )
        assert bar.is_limit_down(Decimal("0.10")) is True

    def test_bar_zero_pre_close(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("11.0"),
            low=Decimal("10.0"),
            close=Decimal("10.8"),
            volume=1000,
            amount=Decimal("10800"),
            pre_close=Decimal("0"),
        )
        assert bar.is_limit_up() is False
        assert bar.is_limit_down() is False

    def test_bar_adjusted_close(self):
        bar = Bar(
            symbol="sh.600519",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("11.0"),
            low=Decimal("10.0"),
            close=Decimal("10.0"),
            volume=1000,
            amount=Decimal("10000"),
            adj_factor=Decimal("1.5"),
        )
        assert bar.adjusted_close(Decimal("2.0")) == Decimal("10.0") * Decimal(
            "2.0"
        ) / Decimal("1.5")


class TestQuote:
    def test_create_quote(self):
        quote = Quote(
            symbol="sh.600519",
            timestamp=datetime.now(),
            last=Decimal("10.5"),
            open=Decimal("10.0"),
            high=Decimal("11.0"),
            low=Decimal("9.5"),
            volume=5000,
            amount=Decimal("52500"),
            bid1=Decimal("10.49"),
            ask1=Decimal("10.51"),
        )
        assert quote.last == Decimal("10.5")
        assert quote.bid1 == Decimal("10.49")


class TestCostBreakdown:
    def test_cost_total(self):
        cost = CostBreakdown(
            commission=Decimal("5.0"),
            transfer_fee=Decimal("0.1"),
            stamp_duty=Decimal("10.0"),
        )
        assert cost.total == Decimal("15.1")


class TestTradeRecord:
    def test_buy_trade(self):
        cost = CostBreakdown(
            commission=Decimal("5.0"),
            transfer_fee=Decimal("0.1"),
            stamp_duty=Decimal("0"),
        )
        record = TradeRecord(
            parent_order_id="order_1",
            slippage_cost=Decimal("0.5"),
            timestamp=datetime.now(),
            symbol="sh.600519",
            quantity=100,
            price=Decimal("10.0"),
            cost=cost,
            is_buy=True,
        )
        assert record.amount == Decimal("1000")
        assert record.net_amount == Decimal("-1005.6")

    def test_sell_trade(self):
        cost = CostBreakdown(
            commission=Decimal("5.0"),
            transfer_fee=Decimal("0.1"),
            stamp_duty=Decimal("10.0"),
        )
        record = TradeRecord(
            parent_order_id="order_2",
            slippage_cost=Decimal("0.5"),
            timestamp=datetime.now(),
            symbol="sh.600519",
            quantity=100,
            price=Decimal("12.0"),
            cost=cost,
            is_buy=False,
        )
        assert record.amount == Decimal("1200")
        assert record.net_amount == Decimal("1184.4")


class TestOrder:
    def test_order_properties(self):
        order = Order(
            order_id="o1", symbol="sh.600519", qty=100, created_ts=datetime.now()
        )
        assert order.is_buy is True
        assert order.is_sell is False
        assert order.status == OrderStatus.PENDING

    def test_order_sell(self):
        order = Order(
            order_id="o2", symbol="sh.600519", qty=-100, created_ts=datetime.now()
        )
        assert order.is_buy is False
        assert order.is_sell is True

    def test_order_execute(self):
        order = Order(
            order_id="o3", symbol="sh.600519", qty=100, created_ts=datetime.now()
        )
        order.execute(100)
        assert order.status == OrderStatus.EXECUTED
        assert order.filled_qty == 100

    def test_order_cancel(self):
        order = Order(
            order_id="o4", symbol="sh.600519", qty=100, created_ts=datetime.now()
        )
        order.cancel("User cancelled")
        assert order.status == OrderStatus.CANCELLED
        assert order.reject_reason == "User cancelled"

    def test_order_reject(self):
        order = Order(
            order_id="o5", symbol="sh.600519", qty=100, created_ts=datetime.now()
        )
        order.reject("Insufficient funds")
        assert order.status == OrderStatus.REJECTED
