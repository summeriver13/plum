"""ETF 规则层测试：费用模型参数化 + T+0/T+1 交收制度。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import (
    ASHARE_RULES,
    ETF_RULES,
    ZERO_RULES,
    AShareCommissionModel,
    EtfCommissionModel,
)


class TestAShareCommissionUnchanged:
    """个股口径必须与既往回测完全一致（默认费率未变）。"""

    def test_default_rules_equal_legacy_constants(self) -> None:
        assert ASHARE_RULES.commission_rate == Decimal("0.0002")
        assert ASHARE_RULES.min_commission == Decimal("5.0")
        assert ASHARE_RULES.transfer_fee_rate == Decimal("0.00002")
        assert ASHARE_RULES.stamp_duty_rate == Decimal("0.0005")
        assert AShareCommissionModel.COMMISSION_RATE == ASHARE_RULES.commission_rate

    def test_buy_has_no_stamp_duty(self) -> None:
        cost = AShareCommissionModel().calculate(Decimal("10"), 10000, is_buy=True)
        assert cost.commission == Decimal("20.0")  # 10 万 × 万 2
        assert cost.transfer_fee == Decimal("2.0")
        assert cost.stamp_duty == Decimal("0")

    def test_sell_charges_stamp_duty(self) -> None:
        cost = AShareCommissionModel().calculate(Decimal("10"), 10000, is_buy=False)
        assert cost.stamp_duty == Decimal("50.0")  # 10 万 × 万 5
        assert cost.total == Decimal("72.0")

    def test_min_commission_binds_on_small_order(self) -> None:
        # 1 万元 × 万 2 = 2 元 < 5 元下限
        cost = AShareCommissionModel().calculate(Decimal("10"), 1000, is_buy=True)
        assert cost.commission == Decimal("5.0")


class TestEtfCommission:
    """ETF 口径：免印花税、免过户费、佣金万 0.5 且无最低收费。"""

    def test_rates(self) -> None:
        assert ETF_RULES.commission_rate == Decimal("0.00005")
        assert ETF_RULES.min_commission == Decimal("0")
        assert ETF_RULES.transfer_fee_rate == Decimal("0")
        assert ETF_RULES.stamp_duty_rate == Decimal("0")

    def test_sell_has_no_stamp_duty(self) -> None:
        cost = EtfCommissionModel().calculate(Decimal("3.5"), 10000, is_buy=False)
        assert cost.stamp_duty == Decimal("0")
        assert cost.transfer_fee == Decimal("0")
        assert cost.commission == Decimal("1.75000")  # 3.5 万 × 万 0.5

    def test_no_minimum_on_small_order(self) -> None:
        cost = EtfCommissionModel().calculate(Decimal("3.5"), 1000, is_buy=True)
        assert cost.commission == Decimal("0.17500")  # 远低于个股 5 元下限
        assert cost.total == Decimal("0.17500")

    def test_zero_rules_model(self) -> None:
        cost = AShareCommissionModel(ZERO_RULES).calculate(
            Decimal("3.5"), 10000, is_buy=False
        )
        assert cost.total == Decimal("0")

    def test_etf_is_cheaper_than_ashare_on_sell(self) -> None:
        ashare = AShareCommissionModel().calculate(
            Decimal("3.5"), 10000, is_buy=False
        ).total
        etf = EtfCommissionModel().calculate(Decimal("3.5"), 10000, is_buy=False).total
        assert etf < ashare


def _buy(portfolio: Portfolio, symbol: str = "SH.510300", quantity: int = 1000) -> None:
    """按给定数量买入（跳过费用，只验证交收逻辑）。"""
    portfolio.apply_trade(
        symbol,
        quantity,
        Decimal("3.5"),
        Decimal("0"),
        timestamp=datetime(2026, 1, 5),
    )


class TestSettlement:
    """T+1（股票默认）与 T+0（ETF）的可用量差异。"""

    def test_t1_buy_not_immediately_sellable(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        _buy(portfolio)
        assert portfolio.can_sell("SH.510300", 1000) is False
        portfolio.on_day_end()
        assert portfolio.can_sell("SH.510300", 1000) is True

    def test_t0_buy_immediately_sellable(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("100000"), settlement="t0")
        _buy(portfolio)
        assert portfolio.can_sell("SH.510300", 1000) is True

    def test_t0_day_end_is_idempotent(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("100000"), settlement="t0")
        _buy(portfolio)
        portfolio.on_day_end()
        portfolio.on_day_end()
        assert portfolio.can_sell("SH.510300", 1000) is True

    def test_t0_partial_buy_then_sell(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("100000"), settlement="t0")
        _buy(portfolio, quantity=1000)
        portfolio.apply_trade("SH.510300", -500, Decimal("3.6"), Decimal("0"))
        assert portfolio.can_sell("SH.510300", 500) is True
        assert portfolio.can_sell("SH.510300", 501) is False

    def test_invalid_settlement_rejected(self) -> None:
        with pytest.raises(ValueError, match="settlement"):
            Portfolio(initial_cash=Decimal("100000"), settlement="t2")

    def test_default_is_t1(self) -> None:
        assert Portfolio(initial_cash=Decimal("100000")).settlement == "t1"
