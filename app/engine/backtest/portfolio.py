"""投资组合管理 — 回测引擎的账户账本。

功能：
- 持仓追踪（数量、成本、盈亏）
- 现金管理（含挂单冻结资金）
- 退市保护（自动清仓僵尸股）
- 公司行为处理（分红、送股、配股）

精度约定：
    所有金额字段使用 Decimal 存储，避免浮点精度问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from app.engine.backtest.parking import ParkingBook

if TYPE_CHECKING:
    from app.data.delisting_registry import DelistingRegistry
    from app.rule.corporate.actions import ActionResult


def _to_float(value: object, default: float = 0.0) -> float:
    """安全转 float（None/NaN/异常值回退默认值）。"""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _to_int(value: object, default: int = 0) -> int:
    """安全转 int（None/NaN/异常值回退默认值）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default


TRADING_DAYS_PER_YEAR = Decimal("252")
"""年交易日数（闲置资金收益按日折算用）。"""

CASH_INTEREST_QUANTUM = Decimal("0.0001")
"""利息计提精度（元）。"""


@dataclass
class Position:
    """单只股票持仓状态。"""

    symbol: str
    quantity: int = 0
    available: int = 0
    cost_basis: Decimal = Decimal("0")
    last_price: Decimal = Decimal("0")
    last_adj_factor: Decimal = Decimal("0")
    """上次已同步的复权因子；0 表示尚无基准（建仓当日由引擎写入）。"""
    entry_timestamp: datetime | None = None
    strategy_id: str = ""
    """哪个策略买入的（用于多策略归因）。"""

    @property
    def market_value(self) -> Decimal:
        return Decimal(self.quantity) * self.last_price

    @property
    def floating_pnl(self) -> Decimal:
        return self.market_value - self.cost_basis

    @property
    def avg_cost(self) -> Decimal:
        if self.quantity == 0:
            return Decimal("0")
        return self.cost_basis / Decimal(self.quantity)


@dataclass
class DelistingLoss:
    """退市股票清仓亏损记录。"""

    symbol: str
    quantity: int
    last_price: Decimal
    cost_basis: Decimal
    loss: Decimal
    delisting_date: datetime
    reason: str


@dataclass
class Portfolio:
    """账户资产管理，支持挂单冻结资金。

    功能特性：
    - T+1 结算的持仓追踪
    - 挂单冻结资金
    - 退市持仓自动清仓（僵尸股保护）
    """

    initial_cash: Decimal
    cash: Decimal = Decimal("0")
    frozen_cash: Decimal = Decimal("0")
    positions: dict[str, Position] = field(default_factory=dict)
    _delisting_losses: list[DelistingLoss] = field(default_factory=list)

    # 闲置资金停泊（国债逆回购口径）
    cash_yield: float = 0.0
    """年化收益率；≤ 0 表示关闭停泊收益。"""

    park_ratio: float = 1.0
    """可停泊资金比例（0-1），用于预留调仓所需的操作现金。"""

    settlement: str = "t1"
    """交收制度：``t1`` = 当日买入次日可卖（股票）；``t0`` = 当日可回转（ETF/可转债）。"""

    parking: ParkingBook = field(default_factory=ParkingBook)
    """停泊标的账本（真实持仓口径；``symbol`` 为空表示未启用）。"""

    _cash_interest_total: Decimal = field(default=Decimal("0"), repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.initial_cash, (int, float)):
            self.initial_cash = Decimal(str(self.initial_cash))
        if self.settlement not in ("t1", "t0"):
            raise ValueError(
                f"settlement 只能是 't1' 或 't0'，收到 {self.settlement!r}"
            )
        self.cash = self.initial_cash

    @property
    def total_value(self) -> Decimal:
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + self.frozen_cash + pos_value + self.parking.market_value

    @property
    def total_pnl(self) -> Decimal:
        return self.total_value - self.initial_cash

    @property
    def available_cash(self) -> Decimal:
        """可用资金（扣除冻结资金）。"""
        return self.cash - self.frozen_cash

    def freeze_cash(self, amount: Decimal) -> None:
        """冻结资金用于挂单买入。"""
        self.frozen_cash += amount

    def unfreeze_cash(self, amount: Decimal) -> None:
        """释放冻结资金（订单取消或拒绝时）。"""
        self.frozen_cash = max(Decimal("0"), self.frozen_cash - amount)

    def _parking_base(self) -> Decimal:
        """可停泊的闲置资金（扣除挂单冻结，再按停泊比例缩放）。"""
        if self.park_ratio <= 0:
            return Decimal("0")
        base = self.available_cash
        if base <= 0:
            return Decimal("0")
        if self.park_ratio >= 1:
            return base
        return base * Decimal(str(self.park_ratio))

    def accrue_cash_interest(self) -> Decimal:
        """按日计提闲置资金收益（国债逆回购 / 货基停泊口径）。

        基数用 ``available_cash`` 而非 ``cash``：挂单冻结期间该笔资金仍留在 ``cash``
        中（``freeze_cash`` 只累加 ``frozen_cash``），用 ``cash`` 会给已冻结资金重复计息。

        利息直接计入 ``cash``，因此 ``total_value``、后续仓位规模与净值曲线自动跟随，
        等价于真实账户中隔夜逆回购次日到期入账。

        Returns:
            当日计提金额（关闭或无可停泊资金时为 0）。
        """
        if self.cash_yield <= 0:
            return Decimal("0")
        base = self._parking_base()
        if base <= 0:
            return Decimal("0")
        daily_rate = Decimal(str(self.cash_yield)) / TRADING_DAYS_PER_YEAR
        interest = (base * daily_rate).quantize(CASH_INTEREST_QUANTUM)
        self.cash += interest
        self._cash_interest_total += interest
        return interest

    @property
    def cash_interest_total(self) -> Decimal:
        """累计闲置资金收益（元）。"""
        return self._cash_interest_total

    # ─── 停泊标的（真实持仓口径）─────────────────────────────

    @property
    def parking_value(self) -> Decimal:
        """停泊仓市值（元）。"""
        return self.parking.market_value

    @property
    def parking_quantity(self) -> int:
        """停泊仓数量（份）。"""
        return self.parking.quantity

    def update_parking_price(self, price: Decimal) -> None:
        """更新停泊标的估值价。"""
        self.parking.update_price(price)

    def apply_parking_trade(
        self, quantity: int, price: Decimal, fee: Decimal
    ) -> Decimal:
        """应用停泊标的成交，返回现金净变动（买入为负）。"""
        delta = self.parking.apply_trade(quantity, price, fee)
        self.cash += delta
        return delta

    def update_price(self, symbol: str, price: Decimal) -> None:
        """更新持仓最新价。"""
        if symbol in self.positions:
            self.positions[symbol].last_price = price

    def apply_trade(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        fee: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """应用交易到持仓。

        参数:
            symbol: 股票代码
            quantity: 交易数量（正=买入，负=卖出）
            price: 成交价格（Decimal 元）
            fee: 交易费用（Decimal 元）
            timestamp: 交易时间
        """
        if quantity == 0:
            return

        trade_amount = Decimal(quantity) * price
        self.cash -= trade_amount + fee

        pos = self.positions.get(symbol)
        was_zero = pos is None or pos.quantity == 0
        if pos is None:
            pos = Position(symbol=symbol)
            self.positions[symbol] = pos

        if quantity > 0:
            if was_zero and timestamp is not None:
                pos.entry_timestamp = timestamp
            pos.quantity += quantity
            pos.cost_basis += trade_amount + fee
            if self.settlement == "t0":
                # T+0 制度（ETF/可转债）：当日买入即可卖出，无需等到日终解锁
                pos.available += quantity
        else:
            pos.quantity += quantity
            pos.available += quantity
            avg_cost = pos.avg_cost
            pos.cost_basis += Decimal(quantity) * avg_cost

        pos.last_price = price

        if pos.quantity == 0:
            pos.entry_timestamp = None
            del self.positions[symbol]

    def on_day_end(self) -> None:
        """日终结算：T+1 可用更新（``t0`` 制度下为幂等操作）。"""
        for pos in self.positions.values():
            pos.available = pos.quantity

    def can_sell(self, symbol: str, quantity: int) -> bool:
        """检查是否可以卖出指定数量。"""
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        return pos.available >= quantity

    def get_position(self, symbol: str) -> Position | None:
        """获取持仓信息。"""
        return self.positions.get(symbol)

    def get_hold_days(self, symbol: str, current_ts: datetime) -> int:
        """获取持仓天数。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.entry_timestamp is None:
            return 0
        delta = current_ts - pos.entry_timestamp
        return delta.days

    # ─── 退市保护 ──────────────────────────────────────────────

    def check_delisting(
        self,
        current_ts: datetime,
        delisting_registry: DelistingRegistry,
    ) -> list[str]:
        """检查退市持仓，返回需要清仓的股票列表。"""
        to_liquidate = []
        for symbol in list(self.positions.keys()):
            if delisting_registry.is_delisted(symbol, current_ts):
                to_liquidate.append(symbol)
        return to_liquidate

    def liquidate_at_zero(
        self,
        symbol: str,
        current_ts: datetime,
        reason: str = "退市",
    ) -> DelistingLoss | None:
        """按零值强制清仓退市持仓。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return None

        loss = -pos.cost_basis

        delisting_loss = DelistingLoss(
            symbol=symbol,
            quantity=pos.quantity,
            last_price=Decimal("0"),
            cost_basis=pos.cost_basis,
            loss=loss,
            delisting_date=current_ts,
            reason=reason,
        )
        self._delisting_losses.append(delisting_loss)
        del self.positions[symbol]

        return delisting_loss

    def liquidate_delisted(
        self,
        current_ts: datetime,
        delisting_registry: DelistingRegistry,
        residual_value_ratio: float = 0.0,
    ) -> list[DelistingLoss]:
        """检查并清仓所有退市持仓。"""
        losses: list[DelistingLoss] = []
        to_liquidate = self.check_delisting(current_ts, delisting_registry)

        for symbol in to_liquidate:
            lifecycle = delisting_registry.get_lifecycle(symbol)
            reason = (lifecycle.reason if lifecycle else None) or "未知"
            delisting_date = (
                lifecycle.last_trading_day if lifecycle else None
            ) or current_ts

            pos = self.positions.get(symbol)
            if pos is None:
                continue

            if residual_value_ratio > 0:
                recovery_value = (
                    pos.last_price
                    * Decimal(pos.quantity)
                    * Decimal(str(residual_value_ratio))
                )
            else:
                recovery_value = Decimal("0")

            loss = recovery_value - pos.cost_basis

            delisting_loss = DelistingLoss(
                symbol=symbol,
                quantity=pos.quantity,
                last_price=pos.last_price,
                cost_basis=pos.cost_basis,
                loss=loss,
                delisting_date=delisting_date,
                reason=reason,
            )
            self._delisting_losses.append(delisting_loss)
            losses.append(delisting_loss)

            self.cash += recovery_value
            del self.positions[symbol]

        return losses

    def get_delisting_losses(self) -> list[DelistingLoss]:
        """获取所有退市亏损记录。"""
        return self._delisting_losses.copy()

    def get_total_delisting_loss(self) -> Decimal:
        """获取退市总亏损。"""
        return sum((loss.loss for loss in self._delisting_losses), Decimal("0"))

    # ─── ST 股票保护 ───────────────────────────────────────────

    def check_st(self, st_stocks: set[str]) -> list[str]:
        """检查持仓中的 ST 股票。"""
        return [s for s in self.positions if s in st_stocks]

    def liquidate_st(
        self,
        st_stocks: set[str],
        current_ts: datetime,
    ) -> list[DelistingLoss]:
        """清仓 ST 股票持仓。"""
        losses: list[DelistingLoss] = []
        to_liquidate = self.check_st(st_stocks)

        for symbol in to_liquidate:
            pos = self.positions.get(symbol)
            if pos is None or pos.quantity == 0:
                continue

            proceeds = pos.last_price * Decimal(pos.quantity)
            loss = proceeds - pos.cost_basis

            st_loss = DelistingLoss(
                symbol=symbol,
                quantity=pos.quantity,
                last_price=pos.last_price,
                cost_basis=pos.cost_basis,
                loss=loss,
                delisting_date=current_ts,
                reason="ST股票",
            )
            self._delisting_losses.append(st_loss)
            losses.append(st_loss)

            self.cash += proceeds
            del self.positions[symbol]

        return losses

    # ─── 事件处理 ──────────────────────────────────────────────

    def apply_action_result(self, result: ActionResult) -> None:
        """应用除权除息结果到持仓。"""
        action = result.action
        pos = self.positions.get(action.symbol)

        if pos is None and result.quantity_change > 0:
            pos = Position(symbol=action.symbol)
            self.positions[action.symbol] = pos

        if pos is None:
            return

        if result.cash_change != 0:
            self.cash += Decimal(result.cash_change)

        if result.quantity_change != 0:
            pos.quantity += result.quantity_change
            # 送转/拆分同步按比例放大可用量（T+1 锁定中的份额同样会拆分）
            from app.rule.corporate.actions import SplitAction

            if isinstance(action, SplitAction):
                pos.available = _to_int(pos.available * _to_float(action.split_ratio))

            if pos.quantity <= 0:
                pos.quantity = 0
                pos.cost_basis = Decimal("0")
                pos.available = 0
                del self.positions[action.symbol]

        if result.cost_basis_change != 0:
            pos.cost_basis += Decimal(result.cost_basis_change)
