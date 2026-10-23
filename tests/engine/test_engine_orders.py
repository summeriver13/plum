"""引擎级测试：订单执行、信号→目标持仓转换、现金守恒与涨跌停档位。"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from app.data.schema import Bar, Order, OrderStatus
from app.engine.backtest.engine import Engine
from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import (
    AShareCommissionModel,
    EtfCommissionModel,
)
from app.engine.backtest.scenarios.limit_up_down import LimitRate, MatchMode

if TYPE_CHECKING:
    from app.engine.backtest.engine import MarketCatalog


def _no_catalog() -> "MarketCatalog":
    """单测只调引擎内部方法（不跑 run），catalog 不参与，显式标注为 None。"""
    return cast("MarketCatalog", None)


class TestEngine:
    def _make_bar(
        self, symbol: str, ts: datetime, close: Decimal = Decimal("10.0")
    ) -> Bar:
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

    def _make_engine(self, portfolio: Portfolio, **kwargs) -> Engine:
        """构造只用于内部方法的引擎：关退市保护、关流动性限额、无滑点/冲击。

        默认用 ETF 佣金模型（0.5bp）：低于引擎预估的 2.2bp，正是历史上
        「买入凭空入账」暴露的场景；股票佣金模型有 5 元最低佣金会掩盖该缺陷。
        """
        kwargs.setdefault("commission", EtfCommissionModel())
        return Engine(
            portfolio=portfolio,
            catalog=_no_catalog(),
            participation_limit=Decimal("0"),
            enable_delisting_protection=False,
            **kwargs,
        )

    def test_buy_order_cash_conservation(self):
        """买入只扣「成交价 + 实际费用」：不得额外补入预估费差额（历史缺陷）。"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = self._make_engine(portfolio)
        ts = datetime(2024, 1, 1)
        bar = self._make_bar("sh.600519", ts, Decimal("10.0"))
        order = Order(
            order_id="ORD-TEST-1", symbol="sh.600519", qty=1000, created_ts=ts
        )
        assert engine._execute_single_order(order, bar) is True

        cost = engine.commission.calculate(bar.close, 1000, True)
        expected = Decimal("1000000") - Decimal(1000) * bar.close - cost.total
        assert portfolio.cash == expected
        assert portfolio.frozen_cash == Decimal("0")

    def test_pending_buy_freezes_then_releases_exactly(self):
        """挂单只冻结 available_cash（不动 cash），成交后按同一金额释放。"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = self._make_engine(portfolio, match_mode=MatchMode.NEXT_OPEN)
        ts = datetime(2024, 1, 1)
        bar = self._make_bar("sh.600519", ts, Decimal("10.0"))
        order = Order(
            order_id="ORD-TEST-2", symbol="sh.600519", qty=1000, created_ts=ts
        )
        engine._add_pending_order(order)
        assert order.frozen_amount > Decimal("0")
        assert portfolio.cash == Decimal("1000000")
        assert portfolio.available_cash == Decimal("1000000") - order.frozen_amount

        engine._process_pending_orders(ts, {"sh.600519": bar})
        cost = engine.commission.calculate(bar.close, 1000, True)
        assert portfolio.frozen_cash == Decimal("0")
        expected = Decimal("1000000") - Decimal(1000) * bar.close - cost.total
        assert portfolio.cash == expected

    def test_rejected_pending_buy_releases_reservation(self):
        """拒单也要释放挂单预留，否则 frozen_cash 泄漏、可用资金长期被压低。"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = self._make_engine(portfolio, match_mode=MatchMode.NEXT_OPEN)
        ts = datetime(2024, 1, 1)
        limit_up = Bar(
            symbol="sh.600519",
            timestamp=ts,
            open=Decimal("11.1"),
            high=Decimal("11.1"),
            low=Decimal("11.1"),
            close=Decimal("11.1"),
            volume=1000,
            amount=Decimal("11100"),
            pre_close=Decimal("10.0"),
        )
        order = Order(
            order_id="ORD-TEST-3", symbol="sh.600519", qty=1000, created_ts=ts
        )
        engine._add_pending_order(order)
        assert portfolio.frozen_cash > Decimal("0")

        engine._process_pending_orders(ts, {"sh.600519": limit_up})
        assert order.status == OrderStatus.REJECTED
        assert portfolio.frozen_cash == Decimal("0")
        assert portfolio.cash == Decimal("1000000")

    def test_engine_convert_signals_dict(self):
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = Engine(
            portfolio=portfolio,
            catalog=_no_catalog(),
            commission=AShareCommissionModel(),
            enable_delisting_protection=False,
        )
        ts = datetime(2024, 1, 1)
        bars = {"sh.600519": self._make_bar("sh.600519", ts)}
        result = engine._convert_signals(
            {"sh.600519": 100}, bars, portfolio.total_value
        )
        assert result == {"sh.600519": 100}

    def test_engine_convert_signals_list(self):
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = Engine(
            portfolio=portfolio,
            catalog=_no_catalog(),
            commission=AShareCommissionModel(),
            enable_delisting_protection=False,
        )
        ts = datetime(2024, 1, 1)
        bars = {"sh.600519": self._make_bar("sh.600519", ts, Decimal("10.0"))}
        signals = [
            Signal(
                symbol="sh.600519",
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                strength=1.0,
            )
        ]
        result = engine._convert_signals(signals, bars, portfolio.total_value)
        assert "sh.600519" in result
        assert result["sh.600519"] > 0

    def test_engine_build_timeline(self):
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = Engine(
            portfolio=portfolio,
            catalog=_no_catalog(),
            commission=AShareCommissionModel(),
            enable_delisting_protection=False,
        )
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        data = {
            "sh.600519": [
                self._make_bar("sh.600519", ts1),
                self._make_bar("sh.600519", ts2),
            ],
        }
        timeline = engine._build_timeline(data)
        assert len(timeline) == 2
        assert timeline[0] == ts1
        assert timeline[1] == ts2

    def test_engine_create_orders(self):
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        engine = Engine(
            portfolio=portfolio,
            catalog=_no_catalog(),
            commission=AShareCommissionModel(),
            enable_delisting_protection=False,
        )
        ts = datetime(2024, 1, 1)
        bars = {"sh.600519": self._make_bar("sh.600519", ts)}
        target = {"sh.600519": 100}
        orders = engine._create_orders(ts, target, bars)
        assert len(orders) == 1
        assert orders[0].symbol == "sh.600519"
        assert orders[0].qty == 100


class TestLimitRateBoardPriority:
    """涨跌停比例：板块优先于 ST。"""

    def test_st_on_chinext_keeps_twenty_percent(self):
        assert LimitRate.get_rate(is_st=True, is_chinext=True) == LimitRate.CHINEXT

    def test_st_on_kcb_keeps_twenty_percent(self):
        assert LimitRate.get_rate(is_st=True, is_kcb=True) == LimitRate.KCB

    def test_st_on_main_board_is_five_percent(self):
        assert LimitRate.get_rate(is_st=True) == LimitRate.ST
