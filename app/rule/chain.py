"""A股预定义规则链工厂。

提供A股常用的规则链构建函数，方便快速创建验证链。

使用示例：
    chain = create_order_validation_chain(
        rules=ASHARE_RULES,
        symbol="600000.SH",
        quantity=100,
        is_market_order=False,
        is_selling=False,
        current_holding=0,
    )
    all_valid, results = chain.validate(symbol="600000.SH", quantity=100)
"""

from typing import Literal

from app.rule.base import RuleChain
from app.rule.trading.validators import (
    LotRule,
    MarketOrderRule,
    OrderQuantityRule,
    PriceCageRule,
    PriceLimitRule,
    PriceTickRule,
)
from app.rule.trading.calculator import TradingRules

OrderSide = Literal["BUY", "SELL"]


def create_order_validation_chain(
    rules: TradingRules,
    symbol: str,
    quantity: int,
    is_market_order: bool = False,
    is_selling: bool = False,
    current_holding: int = 0,
) -> RuleChain:
    """创建A股委托数量验证规则链。

    链顺序：LotRule（整手） -> OrderQuantityRule（数量范围） -> MarketOrderRule（市价单限制）

    参数:
        rules: TradingRules 实例
        symbol: 股票代码
        quantity: 委托数量，必须大于0
        is_market_order: 是否为市价单
        is_selling: 是否为卖出
        current_holding: 当前持仓数量

    返回:
        配置好的 RuleChain 实例

    异常:
        ValueError: quantity <= 0 时抛出
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    chain = RuleChain()
    chain.add(LotRule(rules))
    chain.add(
        OrderQuantityRule(
            rules,
            symbol=symbol,
            is_market_order=is_market_order,
            is_selling=is_selling,
            current_holding=current_holding,
        )
    )
    if is_market_order:
        chain.add(MarketOrderRule(rules))
    return chain


def create_price_validation_chain(
    rules: TradingRules,
    symbol: str,
    price: int,
    pre_close: int,
    ref_price: int | None = None,
    is_st: bool = False,
    is_new_listing: bool = False,
    order_side: OrderSide = "BUY",
    trading_time: str | None = None,
) -> RuleChain:
    """创建A股价格验证规则链。

    链顺序：PriceTickRule（价格跳动） -> PriceLimitRule（涨跌停） -> PriceCageRule（价格笼子）

    参数:
        rules: TradingRules 实例
        symbol: 股票代码
        price: 委托价格，必须大于0
        pre_close: 前收盘价
        ref_price: 基准价（用于价格笼子计算）
        is_st: 是否为ST股票
        is_new_listing: 是否为新股上市首日
        order_side: 委托方向 "BUY" 或 "SELL"
        trading_time: 交易时段

    返回:
        配置好的 RuleChain 实例

    异常:
        ValueError: price <= 0 或 pre_close <= 0 时抛出
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if pre_close <= 0:
        raise ValueError("pre_close must be positive")

    chain = RuleChain()
    chain.add(PriceTickRule(rules))
    chain.add(
        PriceLimitRule(
            rules,
            symbol=symbol,
            pre_close=pre_close,
            is_st=is_st,
            is_new_listing=is_new_listing,
        )
    )
    if ref_price is not None:
        chain.add(
            PriceCageRule(
                rules,
                symbol=symbol,
                ref_price=ref_price,
                order_side=order_side,
                trading_time=trading_time,
            )
        )
    return chain
