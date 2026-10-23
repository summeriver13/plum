"""滑点模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal


class SlippageModel(ABC):
    """滑点计算基类。"""

    @abstractmethod
    def calculate(self, price: Decimal, quantity: int, is_buy: bool) -> Decimal:
        """计算滑点成本。"""


class BpsSlippageModel(SlippageModel):
    """基点滑点模型（1 BP = 0.01%）。"""

    def __init__(self, bps: float = 1.0) -> None:
        self.bps = Decimal(str(bps)) / Decimal("10000")

    def calculate(self, price: Decimal, quantity: int, is_buy: bool) -> Decimal:
        return Decimal(quantity) * price * self.bps


class TickSlippageModel(SlippageModel):
    """跳动滑点模型（tick_size = 0.01 元）。"""

    def __init__(self, ticks: int = 1) -> None:
        self.tick_size = Decimal("0.01") * Decimal(ticks)

    def calculate(self, price: Decimal, quantity: int, is_buy: bool) -> Decimal:
        return Decimal(quantity) * self.tick_size


class CombinedSlippageModel(SlippageModel):
    """组合滑点模型（BPS + Tick）。"""

    def __init__(self, bps: float = 1.0, ticks: int = 1) -> None:
        self._bps = BpsSlippageModel(bps)
        self._tick = TickSlippageModel(ticks)

    def calculate(self, price: Decimal, quantity: int, is_buy: bool) -> Decimal:
        return self._bps.calculate(price, quantity, is_buy) + self._tick.calculate(
            price, quantity, is_buy
        )


_no_slippage: SlippageModel | None = None


def no_slippage() -> SlippageModel:
    """零滑点单例。"""
    global _no_slippage
    if _no_slippage is None:
        _no_slippage = _NoSlippage()
    return _no_slippage


class _NoSlippage(SlippageModel):
    """零滑点。"""

    def calculate(self, price: Decimal, quantity: int, is_buy: bool) -> Decimal:
        return Decimal("0")
