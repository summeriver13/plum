"""实盘交易风控 — 仓位限制、熔断、财务预警。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.rule.trading.calculator import ASHARE_RULES

if TYPE_CHECKING:
    pass


@dataclass
class RiskConfig:
    """风控配置。"""

    max_position_pct: float = 0.2
    max_order_pct: float = 0.1
    max_daily_loss_pct: float = 0.05
    min_position_value: int = 3_000_000
    enable_fat_finger_guard: bool = True
    max_orders_per_minute: int = 10


@dataclass
class FinancialRiskConfig:
    """财务风险配置。"""

    profit_decline_threshold: float = 0.15
    consecutive_quarters: int = 2
    enable_financial_warning: bool = True
    profit_cache_days: int = 7


class RiskGuard:
    """实盘交易风控守卫。

    功能：
    - 下单前风控检查（仓位限制、订单大小、频率限制）
    - 日损失熔断
    - 财务风险预警
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        financial_config: FinancialRiskConfig | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.financial_config = financial_config or FinancialRiskConfig()
        self._order_timestamps: list[datetime] = []
        self._daily_loss: int = 0
        self._peak_value: int = 0
        self._circuit_broken: bool = False

    def pre_check(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: int,
        current_positions: dict[str, int],
        portfolio_value: int,
    ) -> tuple[bool, str]:
        """下单前风控检查。

        返回:
            (是否允许, 拒绝原因)
        """
        if self._circuit_broken:
            return False, "circuit_breaker_open"

        order_value = quantity * price

        if self.config.enable_fat_finger_guard and not self._check_rate_limit():
            return False, "rate_limit_exceeded"

        current_holding = current_positions.get(symbol, 0)
        is_selling = direction == "SELL"
        valid, reason = ASHARE_RULES.validate_order(
            symbol,
            quantity,
            is_market_order=False,
            current_holding=current_holding,
            is_selling=is_selling,
        )
        if not valid:
            return False, reason

        if not self._check_position_limit(
            symbol, quantity, price, current_positions, portfolio_value
        ):
            return False, "position_limit_exceeded"

        if not self._check_order_size(order_value, portfolio_value):
            return False, "order_size_exceeded"

        if not self._check_min_position_value(order_value):
            return False, "position_value_too_small"

        self._order_timestamps.append(datetime.now())
        return True, ""

    def _check_rate_limit(self) -> bool:
        now = datetime.now()
        cutoff = now.replace(second=0, microsecond=0)
        recent = [t for t in self._order_timestamps if t >= cutoff]
        self._order_timestamps = recent
        return len(recent) < self.config.max_orders_per_minute

    def _check_position_limit(
        self,
        symbol: str,
        quantity: int,
        price: int,
        current_positions: dict[str, int],
        portfolio_value: int,
    ) -> bool:
        if portfolio_value <= 0:
            return True

        current_qty = current_positions.get(symbol, 0)
        new_position_value = (current_qty + quantity) * price
        max_position_value = int(portfolio_value * self.config.max_position_pct)

        return new_position_value <= max_position_value

    def _check_order_size(self, order_value: int, portfolio_value: int) -> bool:
        if portfolio_value <= 0:
            return True

        order_pct = order_value / portfolio_value
        return order_pct <= self.config.max_order_pct

    def _check_min_position_value(self, order_value: int) -> bool:
        return order_value >= self.config.min_position_value

    def record_trade(self, pnl: int = 0) -> None:
        self._daily_loss += pnl

    def update_peak(self, current_value: int) -> None:
        self._peak_value = max(self._peak_value, current_value)

    def check_daily_loss(self) -> tuple[bool, int]:
        if self._peak_value <= 0:
            return True, 0

        loss_pct = abs(self._daily_loss) / self._peak_value
        if loss_pct > self.config.max_daily_loss_pct:
            self._circuit_broken = True
            return False, int(self._peak_value * self.config.max_daily_loss_pct)

        return True, 0

    def reset_daily(self) -> None:
        self._daily_loss = 0
        self._peak_value = 0
        self._order_timestamps.clear()

    def reset_circuit_breaker(self) -> None:
        self._circuit_broken = False

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken
