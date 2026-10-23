"""信号包 — 13 个择时信号适配器。"""

from app.signals.base import (
    TimingSignalAdapter,
    MarketTimingV1p5,
    MarketTimingV2,
    StockTimingMA,
    StockTimingMACD,
    StockTimingKDJ,
    StockTimingBollinger,
    StockTimingVolume,
    StockTimingTurtle,
    StockTimingMAEnhanced,
    StockTimingBollingerEnhanced,
    StockTimingGeneric,
    MarketTimingV3_1p5,
    MarketTimingV3_2,
)

__all__ = [
    "TimingSignalAdapter",
    "MarketTimingV1p5",
    "MarketTimingV2",
    "StockTimingMA",
    "StockTimingMACD",
    "StockTimingKDJ",
    "StockTimingBollinger",
    "StockTimingVolume",
    "StockTimingTurtle",
    "StockTimingMAEnhanced",
    "StockTimingBollingerEnhanced",
    "StockTimingGeneric",
    "MarketTimingV3_1p5",
    "MarketTimingV3_2",
]
