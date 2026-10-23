"""订单执行器 — Kelly 仓位计算 + 摩擦成本过滤。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from app.engine.backtest.models import Signal, SignalDirection
from app.engine.backtest.scenarios.commission import AShareCommissionModel

if TYPE_CHECKING:
    pass


@dataclass
class ExecutorConfig:
    """执行器配置。"""

    initial_cash: Decimal = Decimal("4000000")
    max_positions: int = 8
    min_commission: Decimal = Decimal("5.0")
    friction_threshold: Decimal = Decimal("0.0015")
    min_position_value: Decimal = Decimal("50000")
    kelly_multiplier: Decimal = Decimal("1.0")
    can_short: bool = False


@dataclass
class Executor:
    """订单执行器。

    功能：
    - 将信号转换为目标持仓
    - Kelly 分数仓位计算
    - 摩擦成本过滤
    """

    config: ExecutorConfig = field(default_factory=ExecutorConfig)
    commission: AShareCommissionModel = field(default_factory=AShareCommissionModel)

    def execute(
        self,
        signals: list[Signal],
        current_holdings: dict[str, int],
        prices: dict[str, Decimal],
        portfolio_value: Decimal,
    ) -> dict[str, int]:
        """执行信号，返回目标持仓。

        参数:
            signals: 信号列表
            current_holdings: 当前持仓 {symbol: quantity}
            prices: 当前价格 {symbol: price}
            portfolio_value: 组合总价值

        返回:
            目标持仓 {symbol: quantity}
        """
        if not signals:
            return {s: qty for s, qty in current_holdings.items() if qty > 0}

        signals = sorted(
            signals, key=lambda s: (s.strength, s.signal_type.value), reverse=True
        )

        result: dict[str, int] = {}
        allocated_value = Decimal("0")
        allocated_positions = 0

        for signal in signals:
            if signal.direction == SignalDirection.FLAT:
                continue

            if signal.direction == SignalDirection.SHORT and not self.config.can_short:
                continue

            symbol = signal.symbol
            current_qty = current_holdings.get(symbol, 0)
            price = prices.get(symbol, Decimal("0"))

            if price <= Decimal("0"):
                continue

            if signal.direction == SignalDirection.SHORT:
                target_qty = 0
            else:
                if current_qty > 0:
                    target_qty = current_qty
                else:
                    if allocated_positions >= self.config.max_positions:
                        continue
                    target_qty = self._calculate_long_qty(
                        signal, price, portfolio_value, allocated_value
                    )

            if target_qty == current_qty:
                result[symbol] = target_qty
                continue

            if not self._check_friction(
                signal.direction, current_qty, target_qty, price
            ):
                continue

            result[symbol] = target_qty

            if target_qty > 0 and current_qty == 0:
                allocated_positions += 1
                allocated_value += Decimal(target_qty) * price

        for symbol, qty in current_holdings.items():
            if symbol not in result and qty > 0:
                result[symbol] = qty

        return result

    def _calculate_long_qty(
        self,
        signal: Signal,
        price: Decimal,
        portfolio_value: Decimal,
        allocated_value: Decimal,
    ) -> int:
        """计算买入数量（Kelly 分数仓位）。"""
        base_per_position = portfolio_value / Decimal(self.config.max_positions)
        kelly_fraction = min(
            Decimal(str(signal.strength)) * self.config.kelly_multiplier,
            Decimal("1.0"),
        )
        position_value = base_per_position * kelly_fraction

        qty_float = position_value / price
        qty = int(qty_float)
        qty = (qty // 100) * 100

        if Decimal(qty) * price < self.config.min_position_value:
            return 0

        return qty if qty >= 100 else 0

    def _check_friction(
        self,
        direction: SignalDirection,
        current_qty: int,
        target_qty: int,
        price: Decimal,
    ) -> bool:
        """检查摩擦成本是否超过阈值。"""
        if target_qty == current_qty:
            return True

        change_qty = abs(target_qty - current_qty)
        if change_qty <= 0:
            return True

        is_sell = direction == SignalDirection.SHORT or target_qty < current_qty
        cost = self.commission.calculate(price, change_qty, is_buy=not is_sell)
        total_cost = cost.total

        change_value = Decimal(change_qty) * price
        friction_ratio = (
            total_cost / change_value if change_value > Decimal("0") else Decimal("0")
        )
        return friction_ratio <= self.config.friction_threshold
