"""绩效分析模块 — 基于 Decimal 的回测结果分析器。

从 mvp/backtest/analysis 移植，适配 Decimal 金额体系。
"""

from .analyzer import (
    Analyzer,
    BenchmarkReport,
    DrawdownPeriod,
    FIFOReport,
    FIFOMatch,
    FrictionReport,
    MonteCarloResult,
    MonteCarloSimulator,
    ParameterSweep,
    PerformanceReport,
    SortinoReport,
    SweepResult,
    TradeSummary,
)
from .analyzer_ext import (
    AnalyzerExt,
    DrawdownRecoveryReport,
    ExtendedReport,
    PerSymbolReport,
    PositionAttributionReport,
    RegimeAwareReport,
)
from .buy_and_hold_analyzer import (
    BookValueReport,
    BuyAndHoldAnalyzer,
    BuyAndHoldReport,
    DRIPReport,
    EntryThesis,
    PaybackReport,
    SurvivalReport,
    SurvivalScenario,
)
from ._printer import (
    AnalyzerPrinter,
    BuyAndHoldPrinter,
    MonteCarloPrinter,
    StabilityPrinter,
    SweepPrinter,
)
from .stability import (
    GridPoint,
    GridSweepResult,
    ParameterPlateauResult,
    PerformanceMetrics,
    PlateauRegion,
    PrintReport,
    SharpeGradientResult,
    StabilityAnalyzer,
    WalkForwardResult,
)

__all__ = [
    # analyzer
    "Analyzer",
    "BenchmarkReport",
    "DrawdownPeriod",
    "FIFOReport",
    "FIFOMatch",
    "FrictionReport",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "ParameterSweep",
    "PerformanceReport",
    "SortinoReport",
    "SweepResult",
    "TradeSummary",
    # analyzer_ext
    "AnalyzerExt",
    "DrawdownRecoveryReport",
    "ExtendedReport",
    "PerSymbolReport",
    "PositionAttributionReport",
    "RegimeAwareReport",
    # buy_and_hold_analyzer
    "BookValueReport",
    "BuyAndHoldAnalyzer",
    "BuyAndHoldReport",
    "DRIPReport",
    "EntryThesis",
    "PaybackReport",
    "SurvivalReport",
    "SurvivalScenario",
    # stability
    "GridPoint",
    "GridSweepResult",
    "ParameterPlateauResult",
    "PerformanceMetrics",
    "PlateauRegion",
    "PrintReport",
    "SharpeGradientResult",
    "StabilityAnalyzer",
    "WalkForwardResult",
    # _printer
    "AnalyzerPrinter",
    "BuyAndHoldPrinter",
    "MonteCarloPrinter",
    "StabilityPrinter",
    "SweepPrinter",
]
