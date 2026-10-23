"""数据层。

提供行情存储、标的元数据与退市注册表。

子模块：
- connectors: 数据源连接器（QMT、BaoStock——仅作存量兼容，数据重建走数据湖）
- store: 本地数据存储（Parquet + meta.json）
- schema: 核心数据模型
- delisting_registry: 退市股票注册表
- financial: 财务数据存取
"""

from app.data.connectors.base import BaseConnector, DataSource
from app.data.connectors.qmt import QMTConnector
from app.data.delisting_registry import (
    DelistingEvent,
    DelistingRegistry,
    StockLifecycle,
    get_registry,
)
from app.data.schema import (
    Bar,
    CostBreakdown,
    EquityPoint,
    Instrument,
    Order,
    OrderStatus,
    Quote,
    TradeRecord,
)
from app.data.store.base_provider import (
    BaseStorageProvider,
    exchange_code_to_symbol,
    symbol_to_exchange_code,
)
from app.data.store.market import MarketCatalog

__all__ = [
    # Schema
    "Bar",
    "Quote",
    "Instrument",
    "Order",
    "OrderStatus",
    "CostBreakdown",
    "TradeRecord",
    "EquityPoint",
    # Delisting
    "DelistingRegistry",
    "get_registry",
    "StockLifecycle",
    "DelistingEvent",
    # Connectors
    "BaseConnector",
    "DataSource",
    "BaoStockConnector",
    "QMTConnector",
    # Store
    "MarketCatalog",
    "BaseStorageProvider",
    "symbol_to_exchange_code",
    "exchange_code_to_symbol",
]


def __getattr__(name: str):
    """懒加载重型/可选依赖，仅在首次访问时导入。"""
    if name == "BaoStockConnector":
        from app.data.connectors.baostock import BaoStockConnector

        return BaoStockConnector
    raise AttributeError(f"module 'app.data' has no attribute '{name}'")
