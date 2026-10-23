"""规则模块 - A股交易规则和上市公司行为处理。

提供统一的规则验证接口：
- 交易规则：整手、涨跌停、价格笼子等
- 行为规则：分红、送股、配股、拆股等

使用示例：
    from app.rule import ASHARE_RULES, CorporateActionProcessor

    # 订单验证
    is_valid, msg = ASHARE_RULES.validate_order(symbol="600000.SH", quantity=100)

    # 行为处理
    processor = CorporateActionProcessor()
    result = processor.process(action, quantity=1000, cost_basis=105500, available_cash=50000)
"""

from app.rule.base import Rule, RuleChain, RuleResult
from app.rule.chain import (
    OrderSide,
    create_order_validation_chain,
    create_price_validation_chain,
)
from app.rule.corporate import (
    ActionResult,
    ActionType,
    BonusAction,
    CorporateAction,
    CorporateActionProcessor,
    DividendAction,
    DividendProcessor,
    RightsIssueAction,
    RightsIssueProcessor,
    SplitAction,
    SplitProcessor,
)
from app.rule.trading import (
    ASHARE_RULES,
    Board,
    BoardConfig,
    LotRule,
    MarketOrderRule,
    OrderQuantityRule,
    PriceCageRule,
    PriceLimit,
    PriceLimitRule,
    PriceTickRule,
    TradingRules,
)

__all__ = [
    "Rule",
    "RuleChain",
    "RuleResult",
    "OrderSide",
    "create_order_validation_chain",
    "create_price_validation_chain",
    "TradingRules",
    "ASHARE_RULES",
    "Board",
    "BoardConfig",
    "PriceLimit",
    "LotRule",
    "OrderQuantityRule",
    "PriceTickRule",
    "PriceLimitRule",
    "PriceCageRule",
    "MarketOrderRule",
    "CorporateActionProcessor",
    "ActionResult",
    "ActionType",
    "CorporateAction",
    "DividendAction",
    "BonusAction",
    "RightsIssueAction",
    "SplitAction",
    "DividendProcessor",
    "BonusProcessor",
    "RightsIssueProcessor",
    "SplitProcessor",
]
