"""涨跌停保护 — 检查涨跌停状态。"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum


class MatchMode(Enum):
    """撮合模式。"""

    CLOSE = "close"
    NEXT_OPEN = "next_open"


class LimitUpDownGuard:
    """涨跌停检查器。

    涨停时拒绝买入，跌停时拒绝卖出。
    """

    @staticmethod
    def can_buy(
        close: Decimal,
        pre_close: Decimal,
        is_st: bool = False,
        is_kcb: bool = False,
        is_chinext: bool = False,
        limit_rate: Decimal | None = None,
    ) -> bool:
        """检查是否可以买入（未涨停）。

        `limit_rate` 为数据湖标注的当日比例；0 表示无涨跌幅限制日，此时不存在
        「涨停不可买」，必须放行（否则任何上涨都会被当成涨停而拒单）。
        """
        if pre_close == Decimal("0"):
            return True

        rate = (
            LimitRate.get_rate(is_st, is_kcb, is_chinext)
            if limit_rate is None
            else limit_rate
        )
        if rate <= Decimal("0"):
            return True
        limit_price = pre_close * (Decimal("1") + rate)
        return close < limit_price

    @staticmethod
    def can_sell(
        close: Decimal,
        pre_close: Decimal,
        is_st: bool = False,
        is_kcb: bool = False,
        is_chinext: bool = False,
        limit_rate: Decimal | None = None,
    ) -> bool:
        """检查是否可以卖出（未跌停）。语义同 `can_buy`。"""
        if pre_close == Decimal("0"):
            return True

        rate = (
            LimitRate.get_rate(is_st, is_kcb, is_chinext)
            if limit_rate is None
            else limit_rate
        )
        if rate <= Decimal("0"):
            return True
        limit_price = pre_close * (Decimal("1") - rate)
        return close > limit_price


class LimitRate:
    """各板块涨跌停比例。"""

    NORMAL = Decimal("0.10")
    ST = Decimal("0.05")
    KCB = Decimal("0.20")
    CHINEXT = Decimal("0.20")
    BSE = Decimal("0.30")

    @staticmethod
    def get_rate(
        is_st: bool = False, is_kcb: bool = False, is_chinext: bool = False
    ) -> Decimal:
        """获取涨跌停比例。

        板块判定优先于 ST：ST 只在主板压到 5%，创业板/科创板的 ST 股仍是 20%
        （北交所 30% 同样不因 ST 变化，但当前数据湖无北交所标的，未建模）。
        """
        if is_kcb:
            return LimitRate.KCB
        if is_chinext:
            return LimitRate.CHINEXT
        if is_st:
            return LimitRate.ST
        return LimitRate.NORMAL
