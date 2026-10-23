"""闲置资金停泊标的（真实持仓口径）测试。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar
from app.engine.backtest.engine import Engine
from app.engine.backtest.parking import (
    ParkingBook,
    plan_liquidation,
    plan_rebalance,
)
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import EtfCommissionModel

SYMBOL = "SH.510300"
PARKING = "SH.511010"
INITIAL = Decimal("1000000")
PRICE = Decimal("100")

DAYS = [datetime(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)]


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


def _make_bar(symbol: str, ts: datetime, close: Decimal) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
        amount=close * 1_000_000,
        pre_close=close,
    )


class TestParkingPlan:
    """停泊调仓决策（纯函数）。"""

    def test_rebalance_buys_lots_against_eligible_cash(self) -> None:
        trade = plan_rebalance(
            symbol=PARKING,
            price=PRICE,
            available_cash=Decimal("500000"),
            park_ratio=1.0,
            nav=INITIAL,
        )
        assert trade is not None
        assert trade.quantity == 5000  # 500,000 / (100 × 100) = 50 手
        assert trade.reason == "rebalance_buy"

    def test_rebalance_skips_below_threshold(self) -> None:
        trade = plan_rebalance(
            symbol=PARKING,
            price=PRICE,
            available_cash=Decimal("5000"),  # 低于 1 手（1 万）与 1% NAV
            park_ratio=1.0,
            nav=INITIAL,
        )
        assert trade is None

    def test_rebalance_parks_new_cash_regardless_of_holdings(self) -> None:
        """已有停泊仓位不影响补仓：新现金照停（只看手头现金，不做目标对齐）。"""
        trade = plan_rebalance(
            symbol=PARKING,
            price=PRICE,
            available_cash=Decimal("200000"),
            park_ratio=1.0,
            nav=INITIAL,
        )
        assert trade is not None
        assert trade.quantity == 2000

    def test_park_ratio_scales_target(self) -> None:
        trade = plan_rebalance(
            symbol=PARKING,
            price=PRICE,
            available_cash=Decimal("500000"),
            park_ratio=0.5,
            nav=INITIAL,
        )
        assert trade is not None
        assert trade.quantity == 2500

    def test_liquidation_rounds_lots_up(self) -> None:
        trade = plan_liquidation(
            symbol=PARKING,
            price=PRICE,
            shortfall=Decimal("15000"),
            current_quantity=1000,
        )
        assert trade is not None
        assert trade.quantity == -200  # ceil(15000 / 10000) = 2 手

    def test_liquidation_without_shortfall_or_holdings(self) -> None:
        assert (
            plan_liquidation(
                symbol=PARKING,
                price=PRICE,
                shortfall=Decimal("0"),
                current_quantity=500,
            )
            is None
        )
        assert (
            plan_liquidation(
                symbol=PARKING,
                price=PRICE,
                shortfall=Decimal("50000"),
                current_quantity=0,
            )
            is None
        )


class TestParkingBook:
    """停泊账本现金流与盈亏。"""

    def test_buy_moves_cash_into_book(self) -> None:
        book = ParkingBook(symbol=PARKING)
        delta = book.apply_trade(100, PRICE, Decimal("5"))
        assert delta == Decimal("-10005")
        assert book.quantity == 100
        assert book.cost_basis == Decimal("10005")
        book.update_price(Decimal("101"))
        assert book.market_value == Decimal("10100")

    def test_sell_realizes_pnl_net_of_fee(self) -> None:
        book = ParkingBook(symbol=PARKING)
        book.apply_trade(100, PRICE, Decimal("5"))
        delta = book.apply_trade(-100, Decimal("102"), Decimal("5"))
        assert delta == Decimal("10195")
        assert book.quantity == 0
        assert book.cost_basis == Decimal("0")
        assert book.realized_pnl == Decimal("190")
        assert book.sell_count == 1

    def test_sell_clamped_to_holdings(self) -> None:
        book = ParkingBook(symbol=PARKING)
        book.apply_trade(100, PRICE, Decimal("0"))
        delta = book.apply_trade(-500, PRICE, Decimal("0"))
        assert delta == Decimal("10000")
        assert book.quantity == 0

    def test_zero_price_trade_is_ignored(self) -> None:
        book = ParkingBook(symbol=PARKING)
        assert book.apply_trade(100, Decimal("0"), Decimal("0")) == Decimal("0")
        assert book.quantity == 0


class _IdleStrategy:
    """不交易策略，用于隔离停泊对净值的影响。"""

    window = 0

    def __init__(self) -> None:
        self.seen_symbols: set[str] = set()

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> dict[str, int]:
        self.seen_symbols.update(bars.keys())
        return {}

    def flush_state(self) -> None:
        """无需维护状态。"""


class _BuyOnceStrategy:
    """指定日买入固定金额，用于验证「缺现金时自动赎回停泊仓」。"""

    window = 0

    def __init__(self, buy_day: int, quantity: int) -> None:
        self._buy_day = buy_day
        self._quantity = quantity

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> dict[str, int]:
        if ts.day != self._buy_day:
            return {}
        return {SYMBOL: self._quantity}

    def flush_state(self) -> None:
        """无需维护状态。"""


def _catalog(
    parking_close: Decimal = PRICE, step: Decimal = Decimal("1")
) -> _StubCatalog:
    """策略标的恒价 100；停泊标的按日 +step（默认每日 +1 元）。"""
    parking_bars = [
        _make_bar(PARKING, ts, parking_close + step * index)
        for index, ts in enumerate(DAYS)
    ]
    return _StubCatalog(
        {
            SYMBOL: [_make_bar(SYMBOL, ts, PRICE) for ts in DAYS],
            PARKING: parking_bars,
        }
    )


def _engine(catalog: _StubCatalog, **overrides: object) -> Engine:
    kwargs: dict[str, object] = {
        "portfolio": Portfolio(initial_cash=INITIAL, settlement="t0"),
        "catalog": catalog,
        "commission": EtfCommissionModel(),
        "enable_delisting_protection": False,
        "parking_symbol": PARKING,
    }
    kwargs.update(overrides)
    return Engine(**kwargs)  # type: ignore[arg-type]


class TestEngineParking:
    """引擎接线：估值、补仓、赎回、隔离。"""

    def test_first_day_parks_idle_cash(self) -> None:
        engine = _engine(_catalog())
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], _IdleStrategy())
        first = result.equity_curve[0]
        assert first.parking_value == Decimal("990000")  # 99 手（预留费用缓冲）
        assert first.total_value == first.cash + first.parking_value
        assert result.metadata["parking_symbol"] == PARKING
        assert result.metadata["parking_trade_count"] >= 1

    def test_parking_position_is_marked_to_market(self) -> None:
        engine = _engine(_catalog())
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], _IdleStrategy())
        last = result.equity_curve[-1]
        # 持有 9,900 份；停泊价从 100 涨到 108 → 市值 1,069,200
        assert last.parking_value == Decimal("1069200")
        assert last.total_value > INITIAL

    def test_buy_funds_from_parking_without_rejection(self) -> None:
        engine = _engine(_catalog())
        strategy = _BuyOnceStrategy(buy_day=5, quantity=9500)
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], strategy)

        rejected = [
            o for o in engine.order_log if o.reject_reason == "insufficient_funds"
        ]
        assert rejected == []
        buys = [t for t in result.trade_log if t.is_buy]
        assert sum(t.quantity for t in buys) == 9500
        parking_sells = [t for t in result.parking_log if t.quantity < 0]
        assert parking_sells, "缺现金时应当赎回停泊仓"

    def test_strategy_bars_exclude_parking_symbol(self) -> None:
        engine = _engine(_catalog())
        strategy = _IdleStrategy()
        engine.run([SYMBOL], DAYS[0], DAYS[-1], strategy)
        assert PARKING not in strategy.seen_symbols
        assert SYMBOL in strategy.seen_symbols

    def test_cash_yield_is_disabled_when_parking_enabled(self) -> None:
        engine = _engine(_catalog(), cash_yield=0.015)
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], _IdleStrategy())
        assert result.metadata["cash_yield"] == 0.0
        assert result.metadata["cash_interest_total"] == 0.0

    def test_disabled_parking_keeps_cash_yield_path(self) -> None:
        engine = _engine(_catalog(), parking_symbol=None, cash_yield=0.015)
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], _IdleStrategy())
        assert result.metadata["parking_symbol"] == ""
        assert result.metadata["cash_interest_total"] > 0
        assert all(point.parking_value == 0 for point in result.equity_curve)

    def test_missing_parking_history_degrades_gracefully(self) -> None:
        catalog = _StubCatalog({SYMBOL: [_make_bar(SYMBOL, ts, PRICE) for ts in DAYS]})
        engine = _engine(catalog)
        result = engine.run([SYMBOL], DAYS[0], DAYS[-1], _IdleStrategy())
        assert all(point.parking_value == 0 for point in result.equity_curve)
        assert result.metadata["parking_trade_count"] == 0
