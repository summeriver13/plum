"""Plum 原生因子 — 只依赖数据层与因子框架。

从 Bar 数据模型直接计算，遵循 Plum Factor ABC 接口。
可被 FactorEngine 调度、FactorStore 持久化、策略直接消费。

内置因子:
    NativeEP         — 盈利收益率 (1 / PE_TTM)
    NativeBP         — 市净率倒数 (1 / PB_MRQ)
    NativeMarketCap  — 总市值
"""

from app.factors.native.ep import NativeEP
from app.factors.native.bp import NativeBP
from app.factors.native.market_cap import NativeMarketCap

__all__ = [
    "NativeEP",
    "NativeBP",
    "NativeMarketCap",
]
