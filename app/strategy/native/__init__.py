"""Plum 原生策略 — 只依赖数据层与因子框架。

策略列表:
    LowValuationNative      — EP + BP 双因子低估值选股
    EtfRotationStrategy     — ETF 双动量轮动（相对动量排名 + 绝对趋势过滤）
    TimeSeriesTrendStrategy — ETF 时序趋势跟随（12 个月动量 + MA200 双确认）
"""

from app.strategy.native.etf_rotation import EtfRotationStrategy
from app.strategy.native.low_valuation import LowValuationNative
from app.strategy.native.tsmom import TimeSeriesTrendStrategy

__all__ = [
    "EtfRotationStrategy",
    "LowValuationNative",
    "TimeSeriesTrendStrategy",
]
