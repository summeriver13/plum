"""账户对账器 — 本地账户 ↔ 柜台账户三方对账。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from app.engine.backtest.portfolio import Position

if TYPE_CHECKING:
    from app.engine.backtest.portfolio import Portfolio
    from app.engine.live.gateway.base import BaseGateway

CASH_TOLERANCE = 100


@dataclass
class SyncResult:
    local_cash: int
    broker_cash: int
    local_positions: dict[str, int]
    broker_positions: dict[str, int]
    cash_diff: int
    position_diffs: dict[str, tuple[int, int]]
    needs_recovery: bool
    synced_at: datetime = field(default_factory=datetime.now)


class AccountSyncer:
    """本地账户 ↔ 柜台账户对账。

    比较本地持仓和现金与券商账户的差异，
    检测偏差并支持恢复。
    """

    def __init__(
        self,
        gateway: BaseGateway,
        portfolio: Portfolio,
    ) -> None:
        self.gateway = gateway
        self.portfolio = portfolio
        self._last_sync: datetime | None = None
        self._sync_errors: list[str] = []

    def sync(self) -> SyncResult:
        account = self.gateway.get_account_info()

        broker_cash = int(account.cash)
        broker_positions = account.positions.copy()

        local_cash = self.portfolio.cash
        local_positions = {s: p.quantity for s, p in self.portfolio.positions.items()}

        cash_diff = local_cash - broker_cash

        position_diffs: dict[str, tuple[int, int]] = {}
        all_symbols = set(local_positions) | set(broker_positions)
        for symbol in all_symbols:
            local_q = local_positions.get(symbol, 0)
            broker_q = broker_positions.get(symbol, 0)
            if local_q != broker_q:
                position_diffs[symbol] = (local_q, broker_q)

        needs_recovery = abs(cash_diff) > CASH_TOLERANCE or len(position_diffs) > 0

        if needs_recovery:
            self._log_discrepancy(cash_diff, position_diffs)

        result = SyncResult(
            local_cash=local_cash,
            broker_cash=broker_cash,
            local_positions=local_positions,
            broker_positions=broker_positions,
            cash_diff=cash_diff,
            position_diffs=position_diffs,
            needs_recovery=needs_recovery,
        )

        self._last_sync = datetime.now()
        return result

    def _log_discrepancy(
        self,
        cash_diff: int,
        position_diffs: dict[str, tuple[int, int]],
    ) -> None:
        msg = f"Sync discrepancy at {datetime.now()}: cash_diff={cash_diff}"
        if position_diffs:
            msg += f", position_diffs={position_diffs}"
        self._sync_errors.append(msg)

    def recover_to_broker(self) -> None:
        """将本地账户恢复到券商账户状态。"""
        account = self.gateway.get_account_info()
        self.portfolio.cash = int(account.cash)
        self.portfolio.frozen_cash = int(account.frozen_cash)

        broker_positions = account.positions.copy()
        for symbol in list(self.portfolio.positions.keys()):
            local_pos = self.portfolio.positions[symbol]
            broker_qty = broker_positions.get(symbol, 0)
            if broker_qty != local_pos.quantity:
                if broker_qty == 0:
                    del self.portfolio.positions[symbol]
                else:
                    local_pos.quantity = broker_qty
                    local_pos.available = broker_qty

        for symbol, broker_qty in broker_positions.items():
            if symbol not in self.portfolio.positions:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol, quantity=broker_qty
                )

    @property
    def last_sync(self) -> datetime | None:
        return self._last_sync

    @property
    def sync_errors(self) -> list[str]:
        return list(self._sync_errors)
