from __future__ import annotations

from app.strategy.base import (
    BaseStrategy,
    MultiFactorSignalStrategy,
    MultiFactorStrategy,
    PositionOptimizer,
    RebalanceMixin,
    StatefulMixin,
    StrategyContext,
    WindowedStrategy,
)
from app.strategy.buy_and_hold import BuyAndHoldStrategy
from app.strategy.macross import MACrossStrategy
from app.strategy.trend_following import MarketRegime, TrendFollowingStrategy
from app.strategy.hybrid import HybridSmallCapMAStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "HybridSmallCapMAStrategy",
    "MACrossStrategy",
    "MarketRegime",
    "MultiFactorSignalStrategy",
    "MultiFactorStrategy",
    "PositionOptimizer",
    "RebalanceMixin",
    "StatefulMixin",
    "StrategyContext",
    "TrendFollowingStrategy",
    "WindowedStrategy",
]
