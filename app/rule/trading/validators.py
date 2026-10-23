"""A股交易规则验证器（Rule适配器）。

将 TradingRules 核心计算适配为 Rule 接口，供 RuleChain 使用。
"""

from __future__ import annotations

from app.rule.base import Rule, RuleResult
from app.rule.trading.board import Board, get_board, get_board_config
from app.rule.trading.calculator import TradingRules


class LotRule(Rule):
    """整手规则验证器。"""

    def __init__(self, rules: TradingRules) -> None:
        self._rules = rules

    def validate(
        self, symbol: str, quantity: int, is_selling: bool = False
    ) -> RuleResult:
        config = get_board_config(symbol)
        if config.supports_odd_lot:
            return RuleResult(is_valid=True)
        if quantity % config.lot_size != 0:
            return RuleResult(
                is_valid=False,
                message=f"must be multiple of lot size {config.lot_size}",
            )
        return RuleResult(is_valid=True)

    def description(self) -> str:
        return "Lot size validation (整手规则)"


class OrderQuantityRule(Rule):
    """委托数量范围验证器。"""

    def __init__(
        self,
        rules: TradingRules,
        symbol: str,
        is_market_order: bool = False,
        is_selling: bool = False,
        current_holding: int = 0,
    ) -> None:
        self._rules = rules
        self._symbol = symbol
        self._is_market_order = is_market_order
        self._is_selling = is_selling
        self._current_holding = current_holding

    def validate(self, symbol: str, quantity: int) -> RuleResult:
        is_valid, msg = self._rules.validate_order(
            symbol,
            quantity,
            self._is_market_order,
            self._current_holding,
            self._is_selling,
        )
        return RuleResult(is_valid=is_valid, message=msg)

    def description(self) -> str:
        return "Order quantity range validation (委托数量范围)"


class PriceTickRule(Rule):
    """价格跳动规则验证器。"""

    def __init__(self, rules: TradingRules) -> None:
        self._rules = rules

    def validate(self, price: int) -> RuleResult:
        tick = self._rules.get_price_tick()
        if price % tick != 0:
            return RuleResult(
                is_valid=False,
                message=f"price must be multiple of tick {tick}",
            )
        return RuleResult(is_valid=True)

    def description(self) -> str:
        return "Price tick validation (价格跳动规则)"


class PriceLimitRule(Rule):
    """涨跌停规则验证器。"""

    def __init__(
        self,
        rules: TradingRules,
        symbol: str,
        pre_close: int,
        is_st: bool = False,
        is_new_listing: bool = False,
    ) -> None:
        self._rules = rules
        self._symbol = symbol
        self._pre_close = pre_close
        self._is_st = is_st
        self._is_new_listing = is_new_listing

    def validate(self, price: int) -> RuleResult:
        price_limit = self._rules.get_price_limit(
            self._symbol, self._pre_close, self._is_st, self._is_new_listing
        )
        if price < price_limit.low:
            return RuleResult(
                is_valid=False,
                message=f"价格低于跌停价 {price_limit.low / 10_000:.2f}",
            )
        if price > price_limit.high:
            return RuleResult(
                is_valid=False,
                message=f"价格高于涨停价 {price_limit.high / 10_000:.2f}",
            )
        return RuleResult(is_valid=True)

    def description(self) -> str:
        return "Price limit validation (涨跌停规则)"


class PriceCageRule(Rule):
    """价格笼子规则验证器。"""

    def __init__(
        self,
        rules: TradingRules,
        symbol: str,
        ref_price: int,
        order_side: str = "BUY",
        trading_time: str | None = None,
    ) -> None:
        self._rules = rules
        self._symbol = symbol
        self._ref_price = ref_price
        self._order_side = order_side
        self._trading_time = trading_time

    def validate(self, price: int) -> RuleResult:
        board = get_board(self._symbol)
        if board not in (Board.STAR, Board.BSE):
            return RuleResult(is_valid=True)

        cage = self._rules.get_price_cage(
            self._symbol, self._ref_price, self._order_side, self._trading_time
        )
        if price < cage.low:
            return RuleResult(
                is_valid=False,
                message=f"价格超出价格笼子下限 {cage.low / 10_000:.2f}",
            )
        if price > cage.high:
            return RuleResult(
                is_valid=False,
                message=f"价格超出价格笼子上限 {cage.high / 10_000:.2f}",
            )
        return RuleResult(is_valid=True)

    def description(self) -> str:
        return "Price cage validation (价格笼子规则)"


class MarketOrderRule(Rule):
    """市价单数量限制验证器。"""

    def __init__(self, rules: TradingRules) -> None:
        self._rules = rules

    def validate(self, symbol: str, quantity: int) -> RuleResult:
        config = get_board_config(symbol)
        if quantity > config.max_market_order:
            return RuleResult(
                is_valid=False,
                message=f"exceeds market order limit {config.max_market_order} shares",
            )
        return RuleResult(is_valid=True)

    def description(self) -> str:
        return "Market order quantity validation (市价单数量限制)"
