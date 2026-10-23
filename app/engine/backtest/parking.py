"""闲置资金停泊标的（国债 / 货币 ETF）账本与调仓决策。

与 ``cash_yield``（年化利率口径、按日计息）不同，本模块按**真实标的持仓**停泊：
买入份额 → 逐日按收盘价估值 → 需要现金时赎回（ETF T+0）。摩擦（佣金 + 冲击）
由引擎按既有模型算好后传入；本模块只负责数量决策、账本与现金流。

口径约定：
- 只有 ``available_cash``（扣除挂单冻结）可停泊，且按 ``park_ratio`` 缩放；
- 目标停泊额与当前停泊市值的差额低于阈值 ``max(1 手, NAV × min_trade_ratio)`` 时不动；
- 买入手数向下取整（补不满 1 手就留现金）；赎回手数向上取整（宁多赎一点，
  避免次日又因现金不足二次赎回）；
- **只补仓、不因现金变少而减仓**：停泊仓只由「手头闲置现金」驱动（与已有持仓无关），
  只有「下单需要现金」时才赎回（``plan_liquidation``）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

from app.data.schema import Bar

logger = logging.getLogger(__name__)

ETF_LOT_SIZE = 100
"""ETF 最小交易单位（份）。"""

DEFAULT_MIN_TRADE_RATIO = 0.01
"""停泊调仓的最小差额（占组合净值比例）：低于该值不动，避免每日碎单。"""

DEFAULT_FEE_BUFFER = Decimal("0.001")
"""下单前预留的费用缓冲（占买入金额）：覆盖佣金/过户费，避免停在费用上。"""


@dataclass
class ParkingTrade:
    """停泊标的调仓记录（既用于决策，也用于成交留痕）。"""

    symbol: str
    quantity: int
    """正 = 买入，负 = 赎回。"""

    price: Decimal
    reason: str = ""
    fee: Decimal = Decimal("0")
    timestamp: datetime | None = None

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0

    @property
    def notional(self) -> Decimal:
        """成交金额（绝对值，元）。"""
        return Decimal(abs(self.quantity)) * self.price


@dataclass
class ParkingBook:
    """停泊持仓账本（数量 / 成本 / 估值）。"""

    symbol: str = ""
    quantity: int = 0
    cost_basis: Decimal = Decimal("0")
    last_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    buy_count: int = 0
    sell_count: int = 0

    @property
    def enabled(self) -> bool:
        """是否配置了停泊标的。"""
        return bool(self.symbol)

    @property
    def market_value(self) -> Decimal:
        """停泊仓市值（元）。"""
        return Decimal(self.quantity) * self.last_price

    def update_price(self, price: Decimal) -> None:
        """更新估值价（非正价格视为停牌，保留上一价）。"""
        if price > Decimal("0"):
            self.last_price = price

    def apply_trade(self, quantity: int, price: Decimal, fee: Decimal) -> Decimal:
        """应用成交并返回现金净变动（买入为负、赎回为正）。"""
        if quantity == 0 or price <= Decimal("0"):
            return Decimal("0")

        if quantity > 0:
            amount = Decimal(quantity) * price
            self.quantity += quantity
            self.cost_basis += amount + fee
            self.buy_count += 1
            self.last_price = price
            return -(amount + fee)

        sell_qty = min(abs(quantity), self.quantity)
        if sell_qty == 0:
            return Decimal("0")
        avg_cost = (
            self.cost_basis / Decimal(self.quantity) if self.quantity else Decimal("0")
        )
        cost_out = avg_cost * Decimal(sell_qty)
        proceeds = Decimal(sell_qty) * price
        self.quantity -= sell_qty
        self.cost_basis -= cost_out
        self.realized_pnl += proceeds - cost_out - fee
        self.sell_count += 1
        self.last_price = price
        return proceeds - fee


def _dec(value: float | Decimal) -> Decimal:
    """转 Decimal（float 经由 str，避免二进制误差）。"""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _eligible_cash(available_cash: Decimal, park_ratio: float | Decimal) -> Decimal:
    """可停泊现金：按 ``park_ratio`` 缩放并 clamp 到 [0, available_cash]。"""
    if available_cash <= Decimal("0"):
        return Decimal("0")
    ratio = _dec(park_ratio)
    if ratio <= Decimal("0"):
        return Decimal("0")
    if ratio >= Decimal("1"):
        return available_cash
    return available_cash * ratio


def floor_lots(amount: Decimal, lot_value: Decimal) -> int:
    """可买手数（向下取整）；非法输入（NaN/Inf）回退 0。"""
    try:
        return int((amount / lot_value).to_integral_value(rounding=ROUND_FLOOR))
    except (ArithmeticError, TypeError, ValueError):
        return 0


def _ceil_lots(amount: Decimal, lot_value: Decimal) -> int:
    """需赎回手数（向上取整）；非法输入（NaN/Inf）回退 0。

    注意：``Decimal`` 的 ``//`` 是**截断除法**（向零取整），不能用来实现 ceil，
    这里显式用 ``ROUND_CEILING``。
    """
    try:
        return int((amount / lot_value).to_integral_value(rounding=ROUND_CEILING))
    except (ArithmeticError, TypeError, ValueError):
        return 0


def plan_rebalance(
    *,
    symbol: str,
    price: Decimal,
    available_cash: Decimal,
    park_ratio: float | Decimal,
    nav: Decimal,
    lot_size: int = ETF_LOT_SIZE,
    min_trade_ratio: float | Decimal = DEFAULT_MIN_TRADE_RATIO,
) -> ParkingTrade | None:
    """把可用闲置现金补入停泊仓；可补金额小于阈值（或不足 1 手）时返回 None。

    停泊仓是闲置资金的存储，不是目标暴露：本函数**只看手头可用现金**
    （与已有停泊市值无关），因此永远不会减仓；赎回只能由 ``plan_liquidation``
    （下单需要现金）触发。
    """
    lot_value = Decimal(lot_size) * price
    if lot_value <= Decimal("0"):
        return None

    eligible = _eligible_cash(available_cash, park_ratio)
    if eligible < max(lot_value, nav * _dec(min_trade_ratio)):
        return None

    lots = floor_lots(eligible, lot_value)
    if lots <= 0:
        return None
    return ParkingTrade(
        symbol=symbol,
        quantity=lots * lot_size,
        price=price,
        reason="rebalance_buy",
    )


def load_parking_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    root: str | Path,
) -> list[Bar]:
    """从指定行情副本加载停泊标的日线（跨副本停泊用）。

    股票策略的行情副本里没有 ETF，要用国债/货币 ETF 停泊时必须从 ETF 副本读。

    参数:
        symbol: 停泊标的代码（如 ``SH.511010``）
        start: 起始时间（含）
        end: 结束时间（含）
        root: 行情副本根目录（形如 ``data/market/etf``）

    Returns:
        日线列表；副本缺失或读取失败时返回空列表（停泊降级为不生效）。
    """
    try:
        from app.data.store.market import MarketCatalog

        catalog = MarketCatalog(catalog_root=root)
        return list(catalog.load_bars(symbol, start=start, end=end))
    except Exception:  # noqa: BLE001 - 停泊为增强项，读取失败不应中断回测
        logger.warning(
            "停泊标的 %s 行情读取失败（root=%s），停泊不生效",
            symbol,
            root,
            exc_info=True,
        )
        return []


def plan_liquidation(
    *,
    symbol: str,
    price: Decimal,
    shortfall: Decimal,
    current_quantity: int,
    lot_size: int = ETF_LOT_SIZE,
) -> ParkingTrade | None:
    """为补足现金缺口而赎回（手数向上取整）；无缺口或无持仓时返回 None。"""
    if price <= Decimal("0") or lot_size <= 0 or current_quantity <= 0:
        return None
    if shortfall <= Decimal("0"):
        return None

    lot_value = Decimal(lot_size) * price
    lots = _ceil_lots(shortfall, lot_value)
    sell_qty = min(lots * lot_size, current_quantity)
    if sell_qty <= 0:
        return None
    return ParkingTrade(
        symbol=symbol,
        quantity=-sell_qty,
        price=price,
        reason="fund_orders",
    )
