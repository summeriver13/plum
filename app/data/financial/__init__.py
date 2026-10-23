"""财务数据模块 — QMT xtdata 财务数据获取、清洗、存储。

数据流：
  QMT xtdata → Fetcher（下载/获取） → Preprocessor（PIT 对齐 + TTM + 映射）
  → Store（Parquet 落盘） → DataBundle.attach_financials() → 因子计算
"""

from app.data.financial.fetcher import FinancialFetcher
from app.data.financial.preprocessor import FinancialPreprocessor
from app.data.financial.store import FinancialStore
from app.data.financial.integration import merge_financials_to_bars
from app.data.financial.schema import (
    FinancialTable,
    FieldMapping,
    FIELD_MAP,
    QMT_FIELD_DEFINITIONS,
    QMT_TABLE_LIST,
)

__all__ = [
    "FinancialFetcher",
    "FinancialPreprocessor",
    "FinancialStore",
    "merge_financials_to_bars",
    "FinancialTable",
    "FieldMapping",
    "FIELD_MAP",
    "QMT_FIELD_DEFINITIONS",
    "QMT_TABLE_LIST",
]
