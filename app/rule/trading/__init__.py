"""交易规则模块。

导出A股交易规则相关的类和函数。
"""

from app.rule.trading.board import (
    Board,
    BoardConfig,
    PriceLimit,
    get_board,
    get_board_config,
)
from app.rule.trading.calculator import ASHARE_RULES, TradingRules
from app.rule.trading.validators import (
    LotRule,
    MarketOrderRule,
    OrderQuantityRule,
    PriceCageRule,
    PriceLimitRule,
    PriceTickRule,
)

__all__ = [
    "TradingRules",
    "ASHARE_RULES",
    "Board",
    "BoardConfig",
    "PriceLimit",
    "get_board",
    "get_board_config",
    "LotRule",
    "OrderQuantityRule",
    "PriceTickRule",
    "PriceLimitRule",
    "PriceCageRule",
    "MarketOrderRule",
]
