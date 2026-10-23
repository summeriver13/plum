"""市场冲击成本与参与率约束。

回测若默认「想买多少就能成交多少」，小市值/低流动性标的的收益会被系统性高估。
本模块补两件事：

1. **参与率约束**：单笔成交股数 ≤ 当日成交额 × 参与率上限（默认 10%），
   向下取整到整手；成交额缺失时不施加约束（由调用方决定是否跳过）。
2. **平方根冲击**：冲击率 ≈ k × sqrt(参与率)（Almgren 型近似，即平方根定律
   ΔP/P = k·sqrt(Q/ADV)），以「每股价格让价」形式叠加到成交价上，亏损向交易者方向。
   k 的经济含义 ≈ 标的日均波动率：小盘股取 2%~3%。实测本组合单笔参与率中位 0.19%
   （p90 1.3%），k=0.025 对应单笔冲击中位 11bp / p90 29bp，k=0.1 则为 44bp/115bp
   （明显过重）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from math import sqrt

DEFAULT_PARTICIPATION_LIMIT = Decimal("0.10")
# 成交价与 pre_close 的最大允许偏差（涨跌停带 20% + 余量）；超出视为数据不自洽
MAX_LIMIT_DEVIATION = Decimal("0.22")
DEFAULT_IMPACT_COEFFICIENT = 0.025  # ≈ 小盘股日均波动率
LOT_SIZE = 100
MAX_PARTICIPATION = Decimal("1")


class ImpactModel(ABC):
    """冲击模型基类：返回每股价格让价（正数 = 对交易者不利）。"""

    @abstractmethod
    def price_impact(self, price: Decimal, quantity: int, amount: Decimal) -> Decimal:
        """计算冲击让价。"""


class _NoImpact(ImpactModel):
    """零冲击（默认；关闭冲击建模时使用）。"""

    def price_impact(self, price: Decimal, quantity: int, amount: Decimal) -> Decimal:
        return Decimal("0")


class SqrtImpactModel(ImpactModel):
    """平方根冲击模型：让价 = 价格 × k × sqrt(成交量 / 当日成交额)。

    coefficient（k）≈ 标的日均波动率；A 股小盘股经验值 0.02~0.03。
    """

    def __init__(self, coefficient: float = DEFAULT_IMPACT_COEFFICIENT) -> None:
        self.coefficient = Decimal(str(coefficient))

    def price_impact(self, price: Decimal, quantity: int, amount: Decimal) -> Decimal:
        participation = participation_rate(price, quantity, amount)
        if participation <= Decimal("0"):
            return Decimal("0")
        return price * self.coefficient * Decimal(str(sqrt(float(participation))))


def daily_turnover(amount: Decimal, volume: int, close: Decimal) -> Decimal:
    """当日成交额（元）；成交额缺失时用 成交量 × 收盘价 兜底。

    数据湖在 plum 时代（2025-05 之前）的 `amount` 仅覆盖约 45%，而 `volume`
    全覆盖；A 股 amount ≈ volume × VWAP ≈ volume × close，故兜底不会引入量级偏差。
    成交量为 0（停牌）时返回 0，即不施加流动性约束。
    """
    if amount is not None and amount > Decimal("0"):
        return amount
    if volume <= 0 or close <= Decimal("0"):
        return Decimal("0")
    return Decimal(volume) * close


def participation_rate(price: Decimal, quantity: int, amount: Decimal) -> Decimal:
    """成交量占当日成交额的比例（截断到 100%；成交额无效时返回 0）。"""
    turnover = Decimal(quantity) * price
    if turnover <= Decimal("0") or amount is None or amount <= Decimal("0"):
        return Decimal("0")
    return min(turnover / amount, MAX_PARTICIPATION)


def max_participating_quantity(
    price: Decimal,
    amount: Decimal,
    limit: Decimal = DEFAULT_PARTICIPATION_LIMIT,
    lot: int = LOT_SIZE,
) -> int | None:
    """参与率上限内的最大可成交股数（整手向下取整）。

    limit ≤ 0、价格或成交额无效时返回 None，表示「不施加约束」。
    """
    if (
        limit <= Decimal("0")
        or price <= Decimal("0")
        or amount is None
        or amount <= Decimal("0")
    ):
        return None
    quantity = int((amount * limit) / price)
    return (quantity // lot) * lot


_no_impact: ImpactModel | None = None


def no_impact() -> ImpactModel:
    """零冲击单例。"""
    global _no_impact
    if _no_impact is None:
        _no_impact = _NoImpact()
    return _no_impact
