"""除权除息/份额拆分的持仓复权测试。

背景：ETF 份额拆分（实测 159582 于 2026-07-12 做 5:1 拆分：因子 ×5、
原始价 4.991→0.929）与股票送股/转增都表现为复权因子跳变。
引擎若只按原始价标记持仓，持仓市值会凭空消失（拆分）或凭空出现（合并），
卖出时更会按错误数量成交。

判定优先级：adj_factor（权威：只含分红与拆分、累计值）→ 回退「昨收 / 参考价」。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar
from app.engine.backtest.engine import Engine
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel

SPLIT_DAY = datetime(2026, 7, 12)
QUANTITY = 37100
PRE_SPLIT_PRICE = Decimal("4.991")
POST_SPLIT_PRICE = Decimal("0.929")
POST_SPLIT_PRE_CLOSE = Decimal("0.998")  # 交易所拆分后参考价（≈ 4.991 / 5）


def _bar(pre_close: Decimal, close: Decimal) -> Bar:
    """构造简化 K 线（仅替换价格字段）。"""
    return Bar(
        symbol="SZ.159582",
        timestamp=SPLIT_DAY,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
        amount=Decimal("1000000"),
        pre_close=pre_close,
        adj_factor=Decimal("5.001002"),
    )


def _engine(share_adjustment: bool) -> Engine:
    """构造只带一个持仓的引擎（无需真实 catalog）。"""
    portfolio = Portfolio(initial_cash=Decimal("1000000"), settlement="t0")
    portfolio.apply_trade(
        "SZ.159582", QUANTITY, PRE_SPLIT_PRICE, Decimal("0"), timestamp=SPLIT_DAY
    )
    engine = Engine(
        portfolio=portfolio,
        catalog=object(),
        commission=AShareCommissionModel(),
        share_adjustment=share_adjustment,
    )
    if share_adjustment:
        # 处理器实例与注册都在 Engine.run() 的初始化里；直连测试需手动对齐
        from app.rule.corporate.actions import CorporateActionProcessor

        engine.action_processor = CorporateActionProcessor()
        engine._register_default_processors()
    return engine


class TestShareAdjustment:
    """拆分日持仓数量必须按参考价比例复权。"""

    def test_split_multiplies_quantity(self) -> None:
        engine = _engine(share_adjustment=True)
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY, {"SZ.159582": _bar(POST_SPLIT_PRE_CLOSE, POST_SPLIT_PRICE)}
        )
        position = engine.portfolio.positions["SZ.159582"]
        assert adjusted == 1
        assert position.quantity == int(QUANTITY * 5.001)  # 37100 → 185537
        # 拆分前后市值近似不变（按参考价 0.998 而非当日收盘 0.929 计算，
        # 后者含拆分当日的真实跌幅 -6.9%）
        assert abs(float(position.quantity) * 0.998 - QUANTITY * 4.991) < 20

    def test_available_scales_too(self) -> None:
        engine = _engine(share_adjustment=True)
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        engine._apply_share_adjustments(
            SPLIT_DAY, {"SZ.159582": _bar(POST_SPLIT_PRE_CLOSE, POST_SPLIT_PRICE)}
        )
        position = engine.portfolio.positions["SZ.159582"]
        assert position.available == position.quantity

    def test_flag_off_keeps_quantity(self) -> None:
        engine = _engine(share_adjustment=False)
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        engine._apply_share_adjustments(
            SPLIT_DAY, {"SZ.159582": _bar(POST_SPLIT_PRE_CLOSE, POST_SPLIT_PRICE)}
        )
        assert engine.portfolio.positions["SZ.159582"].quantity == QUANTITY

    def test_rounding_noise_ignored(self) -> None:
        engine = _engine(share_adjustment=True)
        engine.portfolio.update_price("SZ.159582", Decimal("4.991"))
        # 参考价与昨收相差 0.2%（最小报价单位舍入量级）→ 不应视为除权
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY, {"SZ.159582": _bar(Decimal("4.981"), Decimal("4.991"))}
        )
        assert adjusted == 0
        assert engine.portfolio.positions["SZ.159582"].quantity == QUANTITY

    def test_dividend_ratio_converts_to_shares(self) -> None:
        engine = _engine(share_adjustment=True)
        engine.portfolio.update_price("SZ.159582", Decimal("4.000"))
        # 2% 现金分红 → 参考价 3.922，数量按 4.000/3.922 放大（等价分红再投资）
        engine._apply_share_adjustments(
            SPLIT_DAY, {"SZ.159582": _bar(Decimal("3.922"), Decimal("3.922"))}
        )
        position = engine.portfolio.positions["SZ.159582"]
        assert position.quantity == int(QUANTITY * (4.000 / 3.922))


def _bar_with_factor(pre_close: Decimal, close: Decimal, factor: Decimal) -> Bar:
    """构造指定复权因子的简化 K 线。"""
    return Bar(
        symbol="SZ.159582",
        timestamp=SPLIT_DAY,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
        amount=Decimal("1000000"),
        pre_close=pre_close,
        adj_factor=factor,
    )


def _engine_with_base(base_factor: Decimal) -> Engine:
    """构造已记录因子基准的引擎（模拟建仓当日已写下因子）。"""
    engine = _engine(share_adjustment=True)
    engine.portfolio.positions["SZ.159582"].last_adj_factor = base_factor
    return engine


class TestFactorJudge:
    """主判据是复权因子：停牌跨期可补齐、小比例分红也能识别、不被脏价格误导。"""

    def test_factor_change_detected_when_pre_close_reset(self) -> None:
        """停牌复牌场景：参考价已复位，只有因子暴露了这次拆分。"""
        engine = _engine_with_base(Decimal("1.0"))
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY,
            {
                "SZ.159582": _bar_with_factor(
                    PRE_SPLIT_PRICE, POST_SPLIT_PRICE, Decimal("5.0")
                )
            },
        )
        position = engine.portfolio.positions["SZ.159582"]
        assert adjusted == 1
        assert position.quantity == QUANTITY * 5
        assert position.last_adj_factor == Decimal("5.0")

    def test_small_dividend_detected_by_factor(self) -> None:
        """因子判定无大容差：0.2% 分红也能识别（低于参考价判据的 0.5% 阈值）。"""
        engine = _engine_with_base(Decimal("1.0"))
        engine.portfolio.update_price("SZ.159582", Decimal("4.000"))
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY,
            {
                "SZ.159582": _bar_with_factor(
                    Decimal("3.992"), Decimal("3.992"), Decimal("1.002")
                )
            },
        )
        assert adjusted == 1
        assert engine.portfolio.positions["SZ.159582"].quantity == int(QUANTITY * 1.002)

    def test_fallback_when_factor_constant(self) -> None:
        """因子不可用（恒 1.0）时回退到参考价判据。"""
        engine = _engine_with_base(Decimal("1.0"))
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY,
            {
                "SZ.159582": _bar_with_factor(
                    POST_SPLIT_PRE_CLOSE, POST_SPLIT_PRICE, Decimal("1.0")
                )
            },
        )
        assert adjusted == 1
        assert engine.portfolio.positions["SZ.159582"].quantity == int(
            QUANTITY * (PRE_SPLIT_PRICE / POST_SPLIT_PRE_CLOSE)
        )

    def test_no_event_no_adjustment(self) -> None:
        """因子与参考价都不变时必须不动数量（防误触发）。"""
        engine = _engine_with_base(Decimal("1.0"))
        engine.portfolio.update_price("SZ.159582", PRE_SPLIT_PRICE)
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY,
            {
                "SZ.159582": _bar_with_factor(
                    PRE_SPLIT_PRICE, PRE_SPLIT_PRICE, Decimal("1.0")
                )
            },
        )
        assert adjusted == 0
        assert engine.portfolio.positions["SZ.159582"].quantity == QUANTITY


class TestEntrySync:
    """建仓当日必须记下因子基准，并把 last_price 对齐收盘价。"""

    def test_entry_records_factor_and_close(self) -> None:
        engine = _engine(share_adjustment=True)
        engine._sync_entry_state(
            "SZ.159582",
            _bar_with_factor(Decimal("4.900"), PRE_SPLIT_PRICE, Decimal("1.0")),
        )
        position = engine.portfolio.positions["SZ.159582"]
        assert position.last_adj_factor == Decimal("1.0")
        # 成交价 4.991 被收盘价覆盖，后续参考价判据不会把偏离误判成除权
        assert position.last_price == PRE_SPLIT_PRICE

    def test_split_next_day_after_entry(self) -> None:
        """建仓当日记基准 → 次日拆分仍能识别（若次日才记基准就会永久漏掉）。"""
        engine = _engine(share_adjustment=True)
        engine._sync_entry_state(
            "SZ.159582",
            _bar_with_factor(Decimal("4.900"), PRE_SPLIT_PRICE, Decimal("1.0")),
        )
        adjusted = engine._apply_share_adjustments(
            SPLIT_DAY,
            {
                "SZ.159582": _bar_with_factor(
                    POST_SPLIT_PRE_CLOSE, POST_SPLIT_PRICE, Decimal("5.0")
                )
            },
        )
        assert adjusted == 1
        assert engine.portfolio.positions["SZ.159582"].quantity == QUANTITY * 5


class TestDefaultOn:
    """正确性内建：开关默认开启，旧批次必须显式关闭。"""

    def test_default_share_adjustment_is_on(self) -> None:
        engine = Engine(
            portfolio=Portfolio(initial_cash=Decimal("1000000")),
            catalog=object(),
            commission=AShareCommissionModel(),
        )
        assert engine.share_adjustment is True

    def test_position_default_factor_is_unset(self) -> None:
        portfolio = Portfolio(initial_cash=Decimal("1000000"))
        portfolio.apply_trade("SZ.159582", 100, Decimal("10"), Decimal("0"), SPLIT_DAY)
        assert portfolio.positions["SZ.159582"].last_adj_factor == Decimal("0")
