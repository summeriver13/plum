"""Tests for backtest analysis module."""

from datetime import datetime
from decimal import Decimal

import pytest

from app.data.schema import CostBreakdown, TradeRecord
from app.engine.backtest.analysis.analyzer import (
    Analyzer,
    FrictionReport,
    PerformanceReport,
    TradeSummary,
)
from app.engine.backtest.analysis.analyzer_ext import (
    AnalyzerExt,
    ExtendedReport,
)
from app.engine.backtest.analysis.buy_and_hold_analyzer import (
    BuyAndHoldAnalyzer,
    BuyAndHoldReport,
    DRIPReport,
    PaybackReport,
    SurvivalReport,
)
from app.engine.backtest.analysis.stability import (
    ParameterPlateauResult,
    PerformanceMetrics,
    SharpeGradientResult,
    StabilityAnalyzer,
)
from app.engine.backtest.models import EquityPoint


def _make_equity_point(
    ts: datetime,
    total_value: Decimal = Decimal("1000000"),
    cash: Decimal = Decimal("500000"),
    frozen_cash: Decimal = Decimal("0"),
    positions_value: Decimal = Decimal("500000"),
    positions_detail: dict | None = None,
) -> EquityPoint:
    return EquityPoint(
        timestamp=ts,
        total_value=total_value,
        cash=cash,
        frozen_cash=frozen_cash,
        positions_value=positions_value,
        positions_detail=positions_detail or {},
    )


def _make_trade(
    symbol: str,
    ts: datetime,
    quantity: int = 100,
    price: Decimal = Decimal("10.0"),
    is_buy: bool = True,
    cost: CostBreakdown | None = None,
) -> TradeRecord:
    if cost is None:
        cost = CostBreakdown(
            commission=Decimal("5.0"),
            transfer_fee=Decimal("0.2"),
            stamp_duty=Decimal("0") if is_buy else Decimal("5.0"),
        )
    return TradeRecord(
        parent_order_id="ORD-001",
        slippage_cost=Decimal("0"),
        timestamp=ts,
        symbol=symbol,
        quantity=quantity,
        price=price,
        cost=cost,
        is_buy=is_buy,
    )


class TestTradeSummary:
    def test_create_trade_summary(self):
        ts = TradeSummary(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            avg_profit=100.0,
            avg_loss=50.0,
            profit_loss_ratio=2.0,
            total_profit=Decimal("600"),
            total_loss=Decimal("200"),
            avg_holding_days=5.0,
        )
        assert ts.total_trades == 10
        assert ts.total_profit == Decimal("600")


class TestPerformanceReport:
    def test_create_report(self):
        report = PerformanceReport(
            initial_value=Decimal("1000000"),
            final_value=Decimal("1100000"),
            total_return=0.1,
            annualized_return=0.12,
            max_drawdown=-0.05,
            max_drawdown_start=None,
            max_drawdown_date=None,
            longest_drawdown=None,
            volatility=0.15,
            sharpe_ratio=0.8,
            calmar_ratio=2.4,
            drawdown_time_ratio=0.3,
            position_time_ratio=0.8,
            trading_days=100,
            risk_free_rate=0.02,
            trade_summary=None,
        )
        assert report.total_return == 0.1
        assert report.initial_value == Decimal("1000000")

    def test_to_dict(self):
        report = PerformanceReport(
            initial_value=Decimal("1000000"),
            final_value=Decimal("1100000"),
            total_return=0.1,
            annualized_return=0.12,
            max_drawdown=-0.05,
            max_drawdown_start=None,
            max_drawdown_date=None,
            longest_drawdown=None,
            volatility=0.15,
            sharpe_ratio=0.8,
            calmar_ratio=2.4,
            drawdown_time_ratio=0.3,
            position_time_ratio=0.8,
            trading_days=100,
            risk_free_rate=0.02,
            trade_summary=None,
        )
        d = report.to_dict()
        assert d["total_return"] == 0.1
        assert d["trading_days"] == 100


class TestAnalyzer:
    def test_calculate_basic(self):
        ts_start = datetime(2024, 1, 1)
        ts_end = datetime(2024, 6, 30)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(datetime(2024, 3, 31), Decimal("1050000")),
            _make_equity_point(ts_end, Decimal("1100000")),
        ]

        report = Analyzer.calculate(equity_curve)
        assert report.total_return > 0
        assert report.trading_days == 3
        assert report.initial_value == Decimal("1000000")
        assert report.final_value == Decimal("1100000")

    def test_calculate_with_trades(self):
        ts_start = datetime(2024, 1, 1)
        ts_end = datetime(2024, 1, 10)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(ts_end, Decimal("1010000")),
        ]
        trades = [
            _make_trade("sh.600519", datetime(2024, 1, 2), is_buy=True),
            _make_trade("sh.600519", datetime(2024, 1, 5), is_buy=False),
        ]

        report = Analyzer.calculate(equity_curve, trade_log=trades)
        assert report.trade_summary is not None
        assert report.trade_summary.total_trades == 2

    def test_calculate_drawdown(self):
        ts_start = datetime(2024, 1, 1)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(datetime(2024, 1, 2), Decimal("900000")),
            _make_equity_point(datetime(2024, 1, 3), Decimal("950000")),
            _make_equity_point(datetime(2024, 1, 4), Decimal("1050000")),
        ]

        report = Analyzer.calculate(equity_curve)
        assert report.max_drawdown < 0
        assert report.drawdown_time_ratio > 0

    def test_friction_audit(self):
        trades = [
            _make_trade("sh.600519", datetime(2024, 1, 1), is_buy=True),
            _make_trade("sh.600519", datetime(2024, 1, 5), is_buy=False),
        ]
        report = Analyzer.friction_audit(trades)
        assert isinstance(report, FrictionReport)
        assert report.total_commission > 0

    def test_empty_equity_curve_raises(self):
        with pytest.raises(ValueError, match="Equity curve"):
            Analyzer.calculate([])


class TestAnalyzerExt:
    def test_calculate_extended(self):
        ts_start = datetime(2024, 1, 1)
        ts_end = datetime(2024, 1, 10)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(ts_end, Decimal("1010000")),
        ]
        trades = [
            _make_trade("sh.600519", datetime(2024, 1, 2), is_buy=True),
            _make_trade("sh.600519", datetime(2024, 1, 5), is_buy=False),
        ]

        from app.engine.backtest.models import BacktestResult

        result = BacktestResult(
            symbol="sh.600519",
            start_date=ts_start,
            end_date=ts_end,
            initial_cash=Decimal("1000000"),
            trade_log=trades,
            equity_curve=equity_curve,
        )

        report = AnalyzerExt.calculate(result)
        assert isinstance(report, ExtendedReport)
        assert isinstance(report.base_performance, PerformanceReport)


class TestBuyAndHoldAnalyzer:
    def test_calculate_drip(self):
        ts_start = datetime(2024, 1, 1)
        ts_end = datetime(2024, 12, 31)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(ts_end, Decimal("1100000")),
        ]

        report = BuyAndHoldAnalyzer.calculate_drip(equity_curve, [], Decimal("1000000"))
        assert isinstance(report, DRIPReport)
        assert report.initial_value == Decimal("1000000")

    def test_calculate_payback(self):
        report = BuyAndHoldAnalyzer.calculate_payback(
            Decimal("100000"),
            [Decimal("5000"), Decimal("5000"), Decimal("5000")],
            Decimal("100000"),
        )
        assert isinstance(report, PaybackReport)
        assert report.initial_investment == Decimal("100000")

    def test_calculate_survival(self):
        report = BuyAndHoldAnalyzer.calculate_survival(
            Decimal("1000000"),
            current_dividend_yield=0.03,
        )
        assert isinstance(report, SurvivalReport)
        assert len(report.scenarios) > 0

    def test_calculate_full_report(self):
        ts_start = datetime(2024, 1, 1)
        ts_end = datetime(2025, 1, 1)
        equity_curve = [
            _make_equity_point(ts_start, Decimal("1000000")),
            _make_equity_point(ts_end, Decimal("1100000")),
        ]

        report = BuyAndHoldAnalyzer.calculate_full_report(
            symbol="sh.600519",
            period_start=ts_start,
            period_end=ts_end,
            initial_investment=Decimal("1000000"),
            initial_price=Decimal("10.0"),
            final_price=Decimal("11.0"),
            equity_curve=equity_curve,
        )
        assert isinstance(report, BuyAndHoldReport)
        assert report.symbol == "sh.600519"


class TestStabilityAnalyzer:
    def test_performance_metrics(self):
        metrics = PerformanceMetrics(
            total_return=0.1,
            annualized_return=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            max_drawdown=-0.05,
            win_rate=0.6,
            num_trades=50,
        )
        assert metrics.sharpe_ratio == 0.8

    def test_sharpe_gradient(self):
        analyzer = StabilityAnalyzer(lambda *a, **k: ([], []))
        result = analyzer.sharpe_gradient(
            "test_param",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 0.7, 0.8, 0.75, 0.6],
        )
        assert isinstance(result, SharpeGradientResult)
        assert result.best_sharpe == 0.8

    def test_parameter_plateau(self):
        analyzer = StabilityAnalyzer(lambda *a, **k: ([], []))
        result = analyzer.parameter_plateau(
            "test_param",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 0.7, 0.72, 0.71, 0.6],
        )
        assert isinstance(result, ParameterPlateauResult)
        assert len(result.regions) > 0

    def test_compute_metrics_empty(self):
        analyzer = StabilityAnalyzer(lambda *a, **k: ([], []))
        metrics = analyzer._compute_metrics([], [])
        assert metrics.total_return == 0.0
        assert metrics.num_trades == 0


def test_comprehensive_report_reads_exec_sanity():
    """成交价自检指标进入 ComprehensiveReport 固定输出。"""
    from app.engine.backtest.analysis.report import comprehensive_report
    from app.engine.backtest.models import BacktestResult, EquityPoint

    result = BacktestResult(
        symbol="test",
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 3),
        initial_cash=Decimal("1000000"),
        equity_curve=[
            EquityPoint(
                timestamp=datetime(2024, 1, 2),
                total_value=Decimal("1000000"),
                cash=Decimal("1000000"),
                frozen_cash=Decimal("0"),
                positions_value=Decimal("0"),
                positions_detail={},
            ),
            EquityPoint(
                timestamp=datetime(2024, 1, 3),
                total_value=Decimal("1010000"),
                cash=Decimal("1010000"),
                frozen_cash=Decimal("0"),
                positions_value=Decimal("0"),
                positions_detail={},
            ),
        ],
        metadata={
            "exec_sanity": {
                "exec_price_calls": 200,
                "clip_count": 6,
                "clip_ratio": 0.03,
                "clip_notional": 1234.5,
                "no_limit_calls": 42,
            }
        },
    )
    report = comprehensive_report(result)
    assert report.exec_price_calls == 200
    assert report.clip_count == 6
    assert report.clip_ratio == pytest.approx(0.03)
    assert report.clip_notional == pytest.approx(1234.5)
    assert report.no_limit_calls == 42
    assert report.to_dict()["成交价自检"]["clip_count"] == 6
    assert report.to_dict()["成交价自检"]["no_limit_calls"] == 42
