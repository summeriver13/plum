"""冲击成本与参与率约束测试。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar, Order, OrderStatus
from app.engine.backtest.engine import Engine
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.backtest.scenarios.impact import (
    DEFAULT_PARTICIPATION_LIMIT,
    SqrtImpactModel,
    daily_turnover,
    max_participating_quantity,
    no_impact,
    participation_rate,
)

AMOUNT = Decimal("10000000")  # 当日成交额 1000 万
PRICE = Decimal("10")


def _bar(amount: Decimal = AMOUNT, close: Decimal = PRICE) -> Bar:
    return Bar(
        symbol="600519.SH",
        timestamp=datetime(2024, 1, 2),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000000,
        amount=amount,
        pre_close=close,
    )


def _engine(participation_limit: Decimal = DEFAULT_PARTICIPATION_LIMIT) -> Engine:
    return Engine(
        portfolio=Portfolio(initial_cash=Decimal("1000000")),
        catalog=None,
        commission=AShareCommissionModel(),
        enable_delisting_protection=False,
        participation_limit=participation_limit,
        impact=SqrtImpactModel(coefficient=0.1),
    )


def test_participation_rate_basic_and_guards() -> None:
    """参与率 = 成交量 × 价格 / 成交额；成交额无效返回 0。"""
    assert participation_rate(PRICE, 10000, AMOUNT) == Decimal("0.01")
    assert participation_rate(PRICE, 0, AMOUNT) == Decimal("0")
    assert participation_rate(PRICE, 10000, Decimal("0")) == Decimal("0")


def test_participation_rate_is_capped_at_one() -> None:
    """成交量超过成交额时截断到 100%（避免开根放大）。"""
    assert participation_rate(PRICE, 10**9, AMOUNT) == Decimal("1")


def test_max_quantity_respects_limit_and_lot() -> None:
    """上限内最大股数：10% 成交额、向下取整到整手。"""
    # 1000 万 × 10% / 10 元 = 10 万股
    assert max_participating_quantity(PRICE, AMOUNT) == 100000
    # 1.05 万股 → 向下取整到 100 股
    assert max_participating_quantity(PRICE, Decimal("105000")) == 1000


def test_max_quantity_disabled_by_zero_limit_or_missing_amount() -> None:
    """限制关闭或成交额缺失时返回 None（不施加约束）。"""
    assert max_participating_quantity(PRICE, AMOUNT, limit=Decimal("0")) is None
    assert max_participating_quantity(PRICE, Decimal("0")) is None
    assert max_participating_quantity(Decimal("0"), AMOUNT) is None


def test_sqrt_impact_grows_with_size() -> None:
    """平方根冲击：让价随参与率单调递增，且与成交额反比。"""
    model = SqrtImpactModel(coefficient=0.1)
    small = model.price_impact(PRICE, 1000, AMOUNT)  # 参与率 0.1%
    mid = model.price_impact(PRICE, 10000, AMOUNT)  # 参与率 1%
    large = model.price_impact(PRICE, 100000, AMOUNT)  # 参与率 10%
    assert Decimal("0") < small < mid < large
    # 参与率 1% → 让价 = 10 元 × 0.1 × sqrt(1%) = 0.1 元
    assert mid == Decimal("0.1")


def test_no_impact_is_zero() -> None:
    """零冲击模型恒为 0（默认关闭冲击建模）。"""
    assert no_impact().price_impact(PRICE, 100000, AMOUNT) == Decimal("0")


def test_engine_caps_quantity_by_participation() -> None:
    """引擎按参与率上限削减下单量（10% × 1000 万 / 10 元 = 10 万股）。"""
    engine = _engine()
    order = Order(
        order_id="o1", symbol="600519.SH", qty=500000, created_ts=datetime(2024, 1, 1)
    )
    assert engine._apply_participation_limit(_bar(), order, 500000) == 100000


def test_engine_keeps_quantity_when_liquid() -> None:
    """成交额充足时不削减。"""
    engine = _engine()
    order = Order(
        order_id="o2", symbol="600519.SH", qty=1000, created_ts=datetime(2024, 1, 1)
    )
    assert engine._apply_participation_limit(_bar(), order, 1000) == 1000


def test_engine_ignores_constraint_without_amount() -> None:
    """成交额缺失（0）时不施加约束，避免把无数据当成无流动性。"""
    engine = _engine()
    order = Order(
        order_id="o2", symbol="600519.SH", qty=1000, created_ts=datetime(2024, 1, 1)
    )
    assert (
        engine._apply_participation_limit(_bar(amount=Decimal("0")), order, 1000)
        == 1000
    )


def test_engine_exec_price_includes_impact_for_buys() -> None:
    """买单成交价高于基准价（冲击方向对交易者不利）。"""
    engine = _engine()
    bar = _bar()
    assert engine._get_exec_price(bar, is_buy=True, quantity=100000) > bar.close
    assert engine._get_exec_price(bar, is_buy=False, quantity=100000) < bar.close


def test_engine_exec_price_impact_shrinks_with_smaller_size() -> None:
    """小单冲击更小：1000 股让价 < 10 万股让价。"""
    engine = _engine()
    bar = _bar()
    small = engine._get_exec_price(bar, is_buy=True, quantity=1000) - bar.close
    large = engine._get_exec_price(bar, is_buy=True, quantity=100000) - bar.close
    assert Decimal("0") < small < large


def test_order_rejected_when_liquidity_below_one_lot() -> None:
    """参与率上限不足一手时拒单（illiquid）。"""
    engine = _engine(participation_limit=Decimal("0.000001"))
    order = Order(
        order_id="o2", symbol="600519.SH", qty=1000, created_ts=datetime(2024, 1, 1)
    )
    bar = _bar(amount=Decimal("500"))
    assert engine._execute_single_order(order, bar) is False
    assert order.status == OrderStatus.REJECTED


def test_daily_turnover_falls_back_to_volume_times_close() -> None:
    """成交额缺失时用 成交量 × 收盘价 兜底（plum 时代覆盖率仅 45%）。"""
    assert daily_turnover(AMOUNT, 1000, PRICE) == AMOUNT
    assert daily_turnover(Decimal("0"), 1000, PRICE) == Decimal("10000")
    assert daily_turnover(Decimal("0"), 0, PRICE) == Decimal("0")  # 停牌不约束


def test_impact_applies_when_amount_missing_but_volume_present() -> None:
    """只有 volume 的历史数据也要吃到冲击成本（否则 P1 形同虚设）。"""
    engine = _engine()
    bar = _bar(amount=Decimal("0"))
    price = engine._get_exec_price(bar, is_buy=True, quantity=10000)
    assert price > bar.close


def test_limit_clip_skipped_on_inconsistent_pre_close() -> None:
    """pre_close 与当日价格严重不一致时放弃裁剪，避免幽灵成交价。"""
    engine = _engine()
    # pre_close=1.7 而 close=0.52（除权/复牌场景常见）：不得把买价顶到跌停价 1.53
    bar = Bar(
        symbol="600695.SH",
        timestamp=datetime(2022, 5, 16),
        open=Decimal("0.51"),
        high=Decimal("0.53"),
        low=Decimal("0.50"),
        close=Decimal("0.52"),
        volume=77719300,
        amount=Decimal("0"),
        pre_close=Decimal("1.7"),
    )
    price = engine._get_exec_price(bar, is_buy=True, quantity=10000)
    assert price < Decimal("0.6")  # 接近当日价，而非 1.53


def test_limit_clip_still_applies_on_consistent_data() -> None:
    """数据自洽时照旧裁剪到涨跌停带内。"""
    engine = _engine()
    bar = Bar(
        symbol="600519.SH",
        timestamp=datetime(2024, 1, 2),
        open=Decimal("11.0"),
        high=Decimal("11.0"),
        low=Decimal("11.0"),
        close=Decimal("11.0"),
        volume=1000000,
        amount=AMOUNT,
        pre_close=Decimal("10.0"),
    )
    # 基准价 11.0 高于涨停 11.0 以内的正常范围 → 裁剪生效；此处仅验证不越过涨停
    assert engine._get_exec_price(bar, is_buy=True, quantity=1000) <= Decimal("11.0")


def test_limit_rate_by_board_with_exchange_prefix() -> None:
    """创业板/科创板按 20%、主板 10%、ST 主板 5%（符号带交易所前缀）。"""
    engine = _engine()
    ts = datetime(2024, 1, 2)

    def _b(symbol: str, is_st: bool = False) -> Bar:
        return Bar(
            symbol=symbol,
            timestamp=ts,
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=1000,
            amount=AMOUNT,
            pre_close=Decimal("10"),
            is_st=is_st,
        )

    assert engine._get_limit_rate(_b("SZ.300912")) == Decimal("0.20")  # 创业板
    assert engine._get_limit_rate(_b("SH.688288")) == Decimal("0.20")  # 科创板
    assert engine._get_limit_rate(_b("BJ.830799")) == Decimal("0.30")  # 北交所
    assert engine._get_limit_rate(_b("SH.600519")) == Decimal("0.10")  # 主板
    assert engine._get_limit_rate(_b("SH.600519", is_st=True)) == Decimal("0.05")
    # ST 创业板仍是 20%（板块优先于 ST）
    assert engine._get_limit_rate(_b("SZ.300123", is_st=True)) == Decimal("0.20")


def test_chinext_price_above_ten_percent_is_not_clipped() -> None:
    """创业板 +20% 的成交价不得被 10% 带裁剪（幽灵滑点的直接成因）。"""
    engine = _engine()
    bar = Bar(
        symbol="SZ.300912",
        timestamp=datetime(2024, 11, 11),
        open=Decimal("15.84"),
        high=Decimal("15.84"),
        low=Decimal("15.0"),
        close=Decimal("15.84"),
        volume=1000000,
        amount=AMOUNT,
        pre_close=Decimal("13.2"),
    )
    price = engine._get_exec_price(bar, is_buy=True, quantity=1000)
    assert price > Decimal("15.0")  # 未被裁到 14.52


def test_exec_sanity_counts_clipped_calls() -> None:
    """自检元数据：裁剪次数 / 发生率 / 调整金额可追溯。"""
    engine = _engine()
    bar = Bar(
        symbol="SH.600519",
        timestamp=datetime(2024, 1, 2),
        open=Decimal("11.0"),
        high=Decimal("11.0"),
        low=Decimal("11.0"),
        close=Decimal("11.0"),
        volume=1000000,
        amount=AMOUNT,
        pre_close=Decimal("10.0"),
    )
    engine._get_exec_price(bar, is_buy=True, quantity=1000)
    sanity = engine._exec_sanity_metadata()
    assert sanity["exec_price_calls"] == 1
    assert sanity["clip_count"] == 1
    assert sanity["clip_ratio"] == 1.0
    assert float(sanity["clip_notional"]) >= 0.0


def test_exec_sanity_zero_when_price_within_band() -> None:
    """价格在涨跌停带内时不产生裁剪计数。"""
    engine = _engine()
    engine._get_exec_price(_bar(), is_buy=True, quantity=1000)
    sanity = engine._exec_sanity_metadata()
    assert sanity["exec_price_calls"] == 1
    assert sanity["clip_count"] == 0
    assert sanity["clip_ratio"] == 0.0


def test_exec_price_skips_clip_on_no_limit_day() -> None:
    """数据湖标注无涨跌幅限制（0）时不裁剪，并在自检中留痕。"""
    engine = _engine()
    bar = Bar(
        symbol="600519.SH",
        timestamp=datetime(2024, 1, 2),
        open=Decimal("13.0"),
        high=Decimal("13.0"),
        low=Decimal("13.0"),
        close=Decimal("13.0"),
        volume=1000000,
        amount=AMOUNT,
        pre_close=Decimal("10.0"),
        limit_rate=Decimal("0"),
        limit_reason="ipo_first_5d",
    )
    price = engine._get_exec_price(bar, is_buy=True, quantity=1000)
    assert price > Decimal("11.0")  # 未被裁到涨停价
    sanity = engine._exec_sanity_metadata()
    assert sanity["clip_count"] == 0
    assert sanity["no_limit_calls"] == 1


def test_exec_price_honors_annotated_rate() -> None:
    """标注比例优先于板块推得比例：标注 5%，+8% 被裁到 +5%。"""
    engine = _engine()
    bar = Bar(
        symbol="SH.600519",
        timestamp=datetime(2024, 1, 2),
        open=Decimal("10.8"),
        high=Decimal("10.8"),
        low=Decimal("10.8"),
        close=Decimal("10.8"),
        volume=1000000,
        amount=AMOUNT,
        pre_close=Decimal("10.0"),
        limit_rate=Decimal("0.05"),
        limit_reason="st",
    )
    price = engine._get_exec_price(bar, is_buy=True, quantity=1000)
    assert price == Decimal("10.5")
    assert engine._exec_sanity_metadata()["clip_count"] == 1
