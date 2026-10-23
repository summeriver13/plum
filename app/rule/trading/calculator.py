"""A股交易规则核心计算。

提供所有交易规则计算方法。
使用单例 ASHARE_RULES 访问。
"""

from __future__ import annotations

from datetime import date

from app.rule.trading.board import Board, PriceLimit, get_board, get_board_config


class TradingRules:
    """A股交易规则核心类。

    提供所有交易规则计算方法。
    使用单例 ASHARE_RULES 访问。
    """

    def is_trading_day(self, dt: date) -> bool:
        """检查是否为交易日（排除周末，不含节假日）。"""
        return dt.weekday() < 5

    def floor_to_lot(self, symbol: str, quantity: int) -> int:
        """将数量向下取整到最近的手。"""
        config = get_board_config(symbol)
        if config.supports_odd_lot:
            return quantity
        return (quantity // config.lot_size) * config.lot_size

    def is_valid_lot(self, symbol: str, quantity: int) -> bool:
        """检查数量是否满足最小整手要求。"""
        config = get_board_config(symbol)
        return quantity >= config.min_order

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        is_market_order: bool = False,
        current_holding: int = 0,
        is_selling: bool = False,
    ) -> tuple[bool, str]:
        """验证委托数量是否合法。"""
        if quantity <= 0:
            return False, "quantity must be positive"

        config = get_board_config(symbol)

        max_qty = config.max_market_order if is_market_order else config.max_limit_order
        if quantity > max_qty:
            return False, f"exceeds maximum order {max_qty} shares"

        if is_selling and quantity == current_holding and current_holding > 0:
            return True, ""

        if quantity < config.min_order:
            return False, f"below minimum order {config.min_order} shares"

        if not config.supports_odd_lot and quantity % config.lot_size != 0:
            return False, f"must be multiple of lot size {config.lot_size}"

        return True, ""

    def round_order(
        self,
        symbol: str,
        quantity: int,
        is_market_order: bool = False,
        current_holding: int = 0,
        is_selling: bool = False,
    ) -> int:
        """对委托数量进行规则处理（取整、裁剪）。"""
        config = get_board_config(symbol)

        if is_selling and quantity == current_holding and current_holding > 0:
            return quantity

        if quantity < config.min_order:
            return 0
        max_qty = config.max_market_order if is_market_order else config.max_limit_order
        quantity = min(quantity, max_qty)

        return self.floor_to_lot(symbol, quantity)

    def get_price_tick(self, symbol: str | None = None) -> int:
        """获取最小价格跳动单位。A股股票统一为0.01元。"""
        return 100

    def get_min_order_value(self, symbol: str) -> int:
        """获取最小委托金额。"""
        config = get_board_config(symbol)
        tick = self.get_price_tick()
        return config.min_order * tick

    def round_price(self, price: int) -> int:
        """将价格取整到有效的价格跳动单位。"""
        tick = self.get_price_tick()
        return (price // tick) * tick

    def get_price_limit(
        self,
        symbol: str,
        pre_close: int,
        is_st: bool = False,
        is_new_listing: bool = False,
    ) -> PriceLimit:
        """计算涨跌停价格范围。"""
        board = get_board(symbol)
        config = get_board_config(symbol)

        if is_new_listing and board == Board.MAIN:
            return PriceLimit(
                low=pre_close
                - round(pre_close * 0.44 / self.get_price_tick())
                * self.get_price_tick(),
                high=pre_close
                + round(pre_close * 0.44 / self.get_price_tick())
                * self.get_price_tick(),
            )

        limit_pct = config.price_limit_pct

        if is_st:
            if board == Board.MAIN:
                limit_pct = 0.05
            elif board == Board.GEM:
                limit_pct = 0.05

        tick = self.get_price_tick()
        high = round(pre_close * (1 + limit_pct) / tick) * tick
        low = round(pre_close * (1 - limit_pct) / tick) * tick

        return PriceLimit(low=low, high=high)

    def validate_price(
        self,
        symbol: str,
        price: int,
        pre_close: int,
        ref_price: int | None = None,
        is_st: bool = False,
        is_new_listing: bool = False,
        order_side: str = "BUY",
    ) -> tuple[bool, str]:
        """验证委托价格是否合法。"""
        price_limit = self.get_price_limit(symbol, pre_close, is_st, is_new_listing)

        if price < price_limit.low:
            return False, f"价格低于跌停价 {price_limit.low / 10_000:.2f} 元"
        if price > price_limit.high:
            return False, f"价格高于涨停价 {price_limit.high / 10_000:.2f} 元"

        if ref_price is not None and ref_price > 0:
            cage = self.get_price_cage(symbol, ref_price, order_side)
            board = get_board(symbol)
            if board in (Board.STAR, Board.BSE):
                if price < cage.low:
                    return False, f"价格超出价格笼子下限 {cage.low / 10_000:.2f} 元"
                if price > cage.high:
                    return False, f"价格超出价格笼子上限 {cage.high / 10_000:.2f} 元"

        return True, ""

    def get_price_cage(
        self,
        symbol: str,
        ref_price: int,
        order_side: str = "BUY",
        trading_time: str | None = None,
    ) -> PriceLimit:
        """计算价格笼子（有效申报价格范围）。"""
        board = get_board(symbol)
        tick = self.get_price_tick()

        if board in (Board.MAIN, Board.GEM):
            if trading_time in ("opening_auction", "closing_auction"):
                return PriceLimit(low=0, high=int(1e9))

            cage_pct = 0.02
            cage_high = round(ref_price * (1 + cage_pct) / tick) * tick
            cage_low = round(ref_price * (1 - cage_pct) / tick) * tick

            alternative_high = ref_price + 1000
            alternative_low = max(0, ref_price - 1000)

            cage_high = max(cage_high, alternative_high)
            cage_low = min(cage_low, alternative_low)

            return PriceLimit(low=cage_low, high=cage_high)

        if board == Board.STAR:
            cage_high = round(ref_price * 1.02 / tick) * tick
            cage_low = round(ref_price * 0.97 / tick) * tick
            return PriceLimit(low=cage_low, high=cage_high)

        if board == Board.BSE:
            cage_high = round(ref_price * 1.05 / tick) * tick
            cage_low = round(ref_price * 0.95 / tick) * tick
            return PriceLimit(low=cage_low, high=cage_high)

        return PriceLimit(low=0, high=int(1e9))

    def get_valid_price_range(
        self,
        symbol: str,
        pre_close: int,
        ref_price: int | None = None,
        is_st: bool = False,
        is_new_listing: bool = False,
        order_side: str = "BUY",
        trading_time: str | None = None,
    ) -> PriceLimit:
        """获取有效价格范围（涨跌停 + 价格笼子的交集）。"""
        price_limit = self.get_price_limit(symbol, pre_close, is_st, is_new_listing)

        if ref_price is None:
            return price_limit

        cage = self.get_price_cage(symbol, ref_price, order_side, trading_time)

        combined_low = max(price_limit.low, cage.low)
        combined_high = min(price_limit.high, cage.high)

        return PriceLimit(low=combined_low, high=combined_high)

    def validate_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        protection_price: int,
    ) -> tuple[bool, str]:
        """验证市价单（保护限价）。"""
        if quantity <= 0:
            return False, "quantity must be positive"

        config = get_board_config(symbol)

        if quantity > config.max_market_order:
            return False, f"exceeds market order limit {config.max_market_order} shares"

        price_limit = self.get_price_limit(symbol, protection_price, is_st=False)
        if side == "BUY":
            if protection_price > price_limit.high:
                return (
                    False,
                    f"protection price exceeds limit high {price_limit.high / 10_000:.2f}",
                )
        else:
            if protection_price < price_limit.low:
                return (
                    False,
                    f"protection price below limit low {price_limit.low / 10_000:.2f}",
                )

        board = get_board(symbol)
        if board in (Board.STAR, Board.BSE):
            cage = self.get_price_cage(symbol, protection_price, side)
            if side == "BUY":
                if protection_price > cage.high:
                    return (
                        False,
                        f"protection price exceeds cage high {cage.high / 10_000:.2f}",
                    )
            else:
                if protection_price < cage.low:
                    return (
                        False,
                        f"protection price below cage low {cage.low / 10_000:.2f}",
                    )

        return True, ""


# 单例实例
ASHARE_RULES = TradingRules()
