"""Tests for backtest module (models, portfolio, scenarios, universe_filter)."""

from datetime import datetime
from decimal import Decimal

import pytest

from app.data.schema import Bar
from app.engine.backtest.models import (
    BacktestResult,
    EquityPoint,
    Signal,
    SignalDirection,
    SignalType,
    TargetPosition,
)
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.cash_guard import CashGuard
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.backtest.scenarios.limit_up_down import LimitUpDownGuard
from app.engine.backtest.scenarios.slippage import (
    BpsSlippageModel,
    TickSlippageModel,
    no_slippage,
)
from app.engine.backtest.scenarios.universe_filter import UniverseFilter

class TestSignal:
    def test_create_signal(self):
        sig = Signal(
            symbol="sh.600519",
            direction=SignalDirection.LONG,
            signal_type=SignalType.ENTRY,
            strength=0.8,
        )
        assert sig.symbol == "sh.600519"
        assert sig.direction == SignalDirection.LONG
        assert sig.strength == 0.8

    def test_strength_above_one_allowed(self):
        """多策略权重可 >1.0（实际仓位由引擎 Cash Guard 兜底），不再限制 0~1。"""
        sig = Signal(
            symbol="sh.600519",
            direction=SignalDirection.LONG,
            strength=1.5,
        )
        assert sig.strength == 1.5


class TestTargetPosition:
    def test_create_target(self):
        tp = TargetPosition(symbol="sh.600519", quantity=100)
        assert tp.symbol == "sh.600519"
        assert tp.quantity == 100

    def test_negative_quantity(self):
        with pytest.raises(ValueError, match="目标持仓数量必须"):
            TargetPosition(symbol="sh.600519", quantity=-100)


class TestPortfolio:
    def test_create_portfolio(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        assert p.cash == Decimal("1000000")
        assert p.total_value == Decimal("1000000")
        assert p.total_pnl == Decimal("0")

    def test_apply_buy_trade(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        p.apply_trade(
            symbol="sh.600519",
            quantity=100,
            price=Decimal("10.0"),
            fee=Decimal("5.0"),
        )
        assert p.positions["sh.600519"].quantity == 100
        assert p.cash == Decimal("998995")

    def test_apply_sell_trade(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        p.apply_trade(
            symbol="sh.600519",
            quantity=100,
            price=Decimal("10.0"),
            fee=Decimal("5.0"),
        )
        p.on_day_end()
        p.apply_trade(
            symbol="sh.600519",
            quantity=-100,
            price=Decimal("12.0"),
            fee=Decimal("5.0"),
        )
        assert "sh.600519" not in p.positions
        assert p.cash > Decimal("1000000")

    def test_freeze_unfreeze_cash(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        p.freeze_cash(Decimal("10000"))
        assert p.available_cash == Decimal("990000")
        p.unfreeze_cash(Decimal("10000"))
        assert p.available_cash == Decimal("1000000")

    def test_can_sell(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        p.apply_trade(
            symbol="sh.600519",
            quantity=100,
            price=Decimal("10.0"),
            fee=Decimal("5.0"),
        )
        assert p.can_sell("sh.600519", 100) is False
        p.on_day_end()
        assert p.can_sell("sh.600519", 100) is True

    def test_update_price(self):
        p = Portfolio(initial_cash=Decimal("1000000"))
        p.apply_trade(
            symbol="sh.600519",
            quantity=100,
            price=Decimal("10.0"),
            fee=Decimal("5.0"),
        )
        p.update_price("sh.600519", Decimal("11.0"))
        assert p.positions["sh.600519"].market_value == Decimal("1100")


class TestCommission:
    def test_buy_commission(self):
        model = AShareCommissionModel()
        cost = model.calculate(Decimal("10.0"), 100, is_buy=True)
        assert cost.commission == Decimal("5.0")
        assert cost.stamp_duty == Decimal("0")

    def test_sell_commission(self):
        model = AShareCommissionModel()
        cost = model.calculate(Decimal("10.0"), 100, is_buy=False)
        assert cost.stamp_duty > Decimal("0")

    def test_high_value_commission(self):
        model = AShareCommissionModel()
        cost = model.calculate(Decimal("100.0"), 1000, is_buy=True)
        assert cost.commission > Decimal("5.0")


class TestCashGuard:
    def test_affordable(self):
        guard = CashGuard()
        ok, qty = guard.is_affordable(Decimal("10.0"), 100, Decimal("100000"))
        assert ok is True
        assert qty == 100

    def test_not_affordable(self):
        guard = CashGuard()
        ok, qty = guard.is_affordable(Decimal("100.0"), 1000, Decimal("10000"))
        assert ok is False
        assert qty < 1000


class TestLimitUpDownGuard:
    def test_normal_limit_up(self):
        assert (
            LimitUpDownGuard.can_buy(Decimal("11.0"), Decimal("10.0"), is_st=False)
            is False
        )

    def test_normal_not_limit_up(self):
        assert (
            LimitUpDownGuard.can_buy(Decimal("10.5"), Decimal("10.0"), is_st=False)
            is True
        )

    def test_st_limit_up(self):
        assert (
            LimitUpDownGuard.can_buy(Decimal("10.5"), Decimal("10.0"), is_st=True)
            is False
        )

    def test_no_limit_day_never_blocks(self):
        """数据湖标注无涨跌幅限制（0）：上涨多少都不算涨停，不得拒单。"""
        assert (
            LimitUpDownGuard.can_buy(
                Decimal("13.0"), Decimal("10.0"), limit_rate=Decimal("0")
            )
            is True
        )
        assert (
            LimitUpDownGuard.can_sell(
                Decimal("7.0"), Decimal("10.0"), limit_rate=Decimal("0")
            )
            is True
        )

    def test_annotated_rate_overrides_board_rate(self):
        """标注比例优先：主板 ST 标注 5%，收盘 +8% 即算涨停。"""
        assert (
            LimitUpDownGuard.can_buy(
                Decimal("10.8"), Decimal("10.0"), limit_rate=Decimal("0.05")
            )
            is False
        )
        assert (
            LimitUpDownGuard.can_buy(
                Decimal("11.0"), Decimal("10.0"), limit_rate=Decimal("0.20")
            )
            is True
        )


class TestSlippage:
    def test_bps_slippage(self):
        model = BpsSlippageModel(bps=1.0)
        slip = model.calculate(Decimal("10.0"), 100, is_buy=True)
        assert slip == Decimal("0.1")

    def test_tick_slippage(self):
        model = TickSlippageModel(ticks=1)
        slip = model.calculate(Decimal("10.0"), 100, is_buy=True)
        assert slip == Decimal("1.0")

    def test_no_slippage(self):
        model = no_slippage()
        slip = model.calculate(Decimal("10.0"), 100, is_buy=True)
        assert slip == Decimal("0")


class TestEquityPoint:
    def test_create_equity_point(self):
        ts = datetime(2024, 1, 1)
        ep = EquityPoint(
            timestamp=ts,
            total_value=Decimal("1000000"),
            cash=Decimal("500000"),
            frozen_cash=Decimal("0"),
            positions_value=Decimal("500000"),
            positions_detail={"sh.600519": 100},
        )
        assert ep.total_value == Decimal("1000000")
        assert ep.positions_detail == {"sh.600519": 100}


class TestBacktestResult:
    def test_create_result(self):
        ts = datetime(2024, 1, 1)
        result = BacktestResult(
            symbol="sh.600519",
            start_date=ts,
            end_date=datetime(2024, 12, 31),
            initial_cash=Decimal("1000000"),
        )
        assert result.symbol == "sh.600519"
        assert result.initial_cash == Decimal("1000000")
        assert result.trade_log == []
        assert result.equity_curve == []


class TestUniverseFilter:
    def test_filter_allows_active_symbol(self):
        import os
        import tempfile

        from app.data.delisting_registry import DelistingRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "delisting.json")
            reg = DelistingRegistry(reg_path)
            uf = UniverseFilter(reg)

            ts = datetime(2024, 1, 1)
            bars: dict[str, Bar] = {}
            target = {"sh.600519": 100}
            allowed, blocked = uf.filter_orders(target, bars, ts)
            assert "sh.600519" in allowed
            assert len(blocked) == 0

    def test_filter_blocks_delisted_symbol(self):
        import os
        import tempfile

        from app.data.delisting_registry import DelistingRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "delisting.json")
            reg = DelistingRegistry(reg_path)
            reg.record_last_trading_day("sh.000001", datetime(2023, 6, 1))
            uf = UniverseFilter(reg)

            ts = datetime(2024, 1, 1)
            bars: dict[str, Bar] = {}
            target = {"sh.000001": 100}
            allowed, blocked = uf.filter_orders(target, bars, ts)
            assert "sh.000001" not in allowed
            assert len(blocked) == 1


