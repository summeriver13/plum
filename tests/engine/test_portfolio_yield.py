"""闲置资金停泊收益（国债逆回购口径）测试。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar
from app.engine.backtest.analysis.report import comprehensive_report
from app.engine.backtest.engine import Engine
from app.engine.backtest.models import BacktestResult, EquityPoint
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel

SYMBOL = "SH.600519"
RATE = 0.015
"""情景年化收益率 1.5%。"""

INITIAL = Decimal("1000000")
ONE_DAY_INTEREST = Decimal("59.5238")
"""100 万 × 1.5% ÷ 252 = 59.523809…，量化到 0.0001 元后为 59.5238。"""

TRADING_DAYS_IN_SAMPLE = 5


class _StubCatalog:
    """最小行情桩：按 symbol 返回预置日线。"""

    def __init__(self, data: dict[str, list[Bar]]) -> None:
        self._data = data

    def load_bars(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        **_kwargs: object,
    ) -> list[Bar]:
        bars = self._data.get(symbol, [])
        return [
            bar
            for bar in bars
            if (start is None or bar.timestamp >= start)
            and (end is None or bar.timestamp <= end)
        ]

    def get_stock_names(self) -> dict[str, str]:
        return {}


class _IdleStrategy:
    """不交易的策略：用于隔离利息对净值的影响。"""

    window = 0

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> dict[str, int]:
        return {}

    def flush_state(self) -> None:
        """无需维护状态。"""


def _make_bar(symbol: str, ts: datetime, close: str = "10.0") -> Bar:
    """构造一根日 K 线。"""
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000,
        amount=price * 1000,
        pre_close=price,
    )


class TestPortfolioParkingYield:
    def test_disabled_by_default(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL)
        assert portfolio.accrue_cash_interest() == Decimal("0")
        assert portfolio.cash == INITIAL

    def test_daily_interest_matches_annual_rate(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        interest = portfolio.accrue_cash_interest()
        assert interest == ONE_DAY_INTEREST
        assert portfolio.cash == INITIAL + ONE_DAY_INTEREST
        assert portfolio.cash_interest_total == ONE_DAY_INTEREST

    def test_interest_reaches_total_value(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        before = portfolio.total_value
        portfolio.accrue_cash_interest()
        assert portfolio.total_value - before == ONE_DAY_INTEREST

    def test_frozen_cash_excluded_from_parking_base(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        portfolio.freeze_cash(Decimal("400000"))
        # 只有 60 万可停泊：600000 × 1.5% ÷ 252 = 35.714285…
        assert portfolio.accrue_cash_interest() == Decimal("35.7143")

    def test_park_ratio_scales_interest(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE, park_ratio=0.4)
        # 40 万可停泊：400000 × 1.5% ÷ 252 = 23.809523…
        assert portfolio.accrue_cash_interest() == Decimal("23.8095")

    def test_park_ratio_above_one_clamped(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE, park_ratio=2.0)
        assert portfolio.accrue_cash_interest() == ONE_DAY_INTEREST

    def test_zero_park_ratio_disables(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE, park_ratio=0.0)
        assert portfolio.accrue_cash_interest() == Decimal("0")

    def test_negative_rate_disables(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=-0.01)
        assert portfolio.accrue_cash_interest() == Decimal("0")

    def test_no_cash_pays_nothing(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("0"), cash_yield=RATE)
        assert portfolio.accrue_cash_interest() == Decimal("0")

    def test_over_frozen_balance_pays_nothing(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        portfolio.freeze_cash(INITIAL * 2)
        assert portfolio.accrue_cash_interest() == Decimal("0")

    def test_interest_uses_available_cash_not_positions(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        portfolio.apply_trade(SYMBOL, 1000, Decimal("10.0"), Decimal("0"))
        # 花掉 1 万后现金 99 万：990000 × 1.5% ÷ 252 = 58.928571…
        assert portfolio.accrue_cash_interest() == Decimal("58.9286")

    def test_interest_is_not_lost_on_reset_of_daily_flow(self) -> None:
        portfolio = Portfolio(initial_cash=INITIAL, cash_yield=RATE)
        first = portfolio.accrue_cash_interest()
        second = portfolio.accrue_cash_interest()
        # 第二日基数含第一日利息，因此大于首日（复利）
        assert second > first
        assert portfolio.cash_interest_total == first + second


class TestEngineParkingYield:
    @staticmethod
    def _catalog() -> _StubCatalog:
        days = [datetime(2024, 1, day) for day in (2, 3, 4, 5, 8)]
        return _StubCatalog({SYMBOL: [_make_bar(SYMBOL, ts) for ts in days]})

    @staticmethod
    def _engine(**overrides: object) -> Engine:
        kwargs: dict[str, object] = {
            "portfolio": Portfolio(initial_cash=INITIAL),
            "catalog": TestEngineParkingYield._catalog(),
            "commission": AShareCommissionModel(),
            "enable_delisting_protection": False,
        }
        kwargs.update(overrides)
        return Engine(**kwargs)  # type: ignore[arg-type]

    def test_engine_accrues_over_trading_days(self) -> None:
        engine = self._engine(cash_yield=RATE)
        result = engine.run(
            [SYMBOL],
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            _IdleStrategy(),
            liquidate=False,
        )
        assert len(result.equity_curve) == TRADING_DAYS_IN_SAMPLE

        first_day = result.equity_curve[0]
        assert first_day.cash == INITIAL + ONE_DAY_INTEREST
        assert first_day.total_value == INITIAL + ONE_DAY_INTEREST

        # 5 个交易日复利：1_000_000 × ((1 + 1.5%/252)^5 − 1) ≈ 297.65
        total_interest = float(result.metadata["cash_interest_total"])
        expected = float(INITIAL) * ((1 + RATE / 252) ** TRADING_DAYS_IN_SAMPLE - 1)
        assert abs(total_interest - expected) < 0.01
        assert total_interest > float(ONE_DAY_INTEREST * TRADING_DAYS_IN_SAMPLE)

    def test_engine_records_config_in_metadata(self) -> None:
        engine = self._engine(cash_yield=RATE, park_ratio=0.4)
        result = engine.run(
            [SYMBOL],
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            _IdleStrategy(),
            liquidate=False,
        )
        assert result.metadata["cash_yield"] == RATE
        assert result.metadata["park_ratio"] == 0.4

    def test_engine_without_yield_keeps_cash_flat(self) -> None:
        engine = self._engine()
        result = engine.run(
            [SYMBOL],
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            _IdleStrategy(),
            liquidate=False,
        )
        assert result.metadata["cash_interest_total"] == 0.0
        assert result.equity_curve[-1].cash == INITIAL

    def test_equity_curve_carries_daily_interest(self) -> None:
        engine = self._engine(cash_yield=RATE)
        result = engine.run(
            [SYMBOL],
            datetime(2024, 1, 2),
            datetime(2024, 1, 8),
            _IdleStrategy(),
            liquidate=False,
        )
        daily = [point.cash_interest for point in result.equity_curve]
        assert daily[0] == ONE_DAY_INTEREST
        assert all(value > 0 for value in daily)
        assert sum(daily) == Decimal(str(result.metadata["cash_interest_total"]))


class TestReportCashInterest:
    """报告层：停泊收益分项与剥离对照。"""

    @staticmethod
    def _result(daily_interest: Decimal) -> BacktestResult:
        start = datetime(2024, 1, 2)
        curve = [
            EquityPoint(
                timestamp=start,
                total_value=INITIAL + daily_interest,
                cash=INITIAL + daily_interest,
                frozen_cash=Decimal("0"),
                positions_value=Decimal("0"),
                cash_interest=daily_interest,
            ),
            EquityPoint(
                timestamp=datetime(2024, 1, 3),
                total_value=INITIAL + daily_interest * 2,
                cash=INITIAL + daily_interest * 2,
                frozen_cash=Decimal("0"),
                positions_value=Decimal("0"),
                cash_interest=daily_interest,
            ),
        ]
        return BacktestResult(
            symbol=SYMBOL,
            start_date=start,
            end_date=datetime(2024, 1, 3),
            initial_cash=INITIAL,
            equity_curve=curve,
            metadata={"cash_yield": RATE, "park_ratio": 1.0},
        )

    def test_report_exposes_parking_section(self) -> None:
        report = comprehensive_report(self._result(Decimal("50")), run_id="t")
        assert report.cash_interest_total == 100.0
        assert report.cash_ratio == 1.0
        assert report.cash_yield_rate == RATE
        assert report.park_ratio == 1.0
        # 算术贡献：50/1e6 + 50/1_000_050 = 9.99975e-5，年化 ÷ (2/242)
        assert abs(report.cash_interest_annual - 0.01210) < 0.0001
        section = report.to_dict()["闲置资金收益"]
        assert section["cash_interest_total"] == 100.0
        assert "annual_return_ex" in section

    def test_report_omits_section_without_parking(self) -> None:
        report = comprehensive_report(self._result(Decimal("0")), run_id="t")
        assert report.cash_interest_total == 0.0
        assert report.cash_interest_annual == 0.0
        assert "闲置资金收益" not in report.to_dict()

    def test_stripped_metrics_reproduce_no_interest_curve(self) -> None:
        report = comprehensive_report(self._result(Decimal("50")), run_id="t")
        # 剥离后起始、末期净值都回到初始资金，年化对照值为 0
        assert report.annual_return_ex == 0.0
        assert report.max_drawdown_ex == 0.0
        assert report.calmar_ex == 0.0
