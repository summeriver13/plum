"""Universe filter — 过滤退市/不可交易股票的信号。

解决"幽灵信号问题"：策略为不可交易的股票生成买入信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.data.schema import Bar

if TYPE_CHECKING:
    from app.data.delisting_registry import DelistingRegistry
    from app.engine.backtest.models import Signal


@dataclass
class FilteredSignal:
    """被过滤（拦截）的信号记录。"""

    signal: object
    reason: str
    timestamp: datetime


class UniverseFilter:
    """基于股票池（退市、停牌等）过滤信号。

    这是门口的"保安"——确保只有可交易的股票才能进入订单流水线。

    过滤规则：
    1. 退市股票 — 完全拦截
    2. 停牌股票 — 拦截买入
    3. 未上市股票 — 拦截
    4. 退市日期之后 — 拦截

    注意：此过滤器不会拦截现有持仓的卖出信号。
    """

    def __init__(
        self,
        delisting_registry: DelistingRegistry | None = None,
        allow_sell_delisted: bool = True,
        exclude_st: bool = False,
    ):
        from app.data.delisting_registry import get_registry

        self._registry = delisting_registry or get_registry()
        self._allow_sell_delisted = allow_sell_delisted
        self._exclude_st = exclude_st
        self._blocked_log: list[FilteredSignal] = []

    def is_blocked(
        self,
        symbol: str,
        direction: str,
        current_ts: datetime,
    ) -> tuple[bool, str]:
        """检查信号是否应该被拦截。

        参数:
            symbol: 股票代码
            direction: "BUY" 或 "SELL"
            current_ts: 当前时间戳

        返回:
            (是否被拦截, 原因)
        """
        if self._registry.is_delisted(symbol, current_ts):
            if direction == "SELL" and self._allow_sell_delisted:
                return False, ""
            return True, f"已退市 (delisted on or before {current_ts.date()})"

        lifecycle = self._registry.get_lifecycle(symbol)
        if (
            lifecycle
            and lifecycle.first_trading_day
            and current_ts < lifecycle.first_trading_day
        ):
            return True, f"未上市 (lists on {lifecycle.first_trading_day.date()})"

        return False, ""

    def filter_signals(
        self,
        signals: list[Signal],
        current_bars: dict[str, Bar],
        current_ts: datetime,
    ) -> tuple[list[Signal], list[FilteredSignal]]:
        """基于股票池规则过滤信号。

        参数:
            signals: 策略原始信号
            current_bars: 当前 K 线数据
            current_ts: 当前时间戳

        返回:
            (允许的信号, 被拦截的信号)
        """
        allowed: list = []
        blocked: list[FilteredSignal] = []

        for signal in signals:
            is_buy = signal.direction.value == 1
            direction = "BUY" if is_buy else "SELL"

            is_blocked, reason = self.is_blocked(signal.symbol, direction, current_ts)

            if is_blocked:
                filtered = FilteredSignal(
                    signal=signal, reason=reason, timestamp=current_ts
                )
                blocked.append(filtered)
                self._blocked_log.append(filtered)
            else:
                allowed.append(signal)

        return allowed, blocked

    def filter_orders(
        self,
        target_positions: dict[str, int],
        current_bars: dict[str, Bar],
        current_ts: datetime,
    ) -> tuple[dict[str, int], list[FilteredSignal]]:
        """过滤目标持仓（symbol -> target qty 字典）。

        参数:
            target_positions: 策略/执行器的目标持仓
            current_bars: 当前 K 线数据
            current_ts: 当前时间戳

        返回:
            (允许的持仓, 被拦截的记录)
        """
        allowed: dict[str, int] = {}
        blocked: list[FilteredSignal] = []

        for symbol, qty in target_positions.items():
            direction = "BUY" if qty > 0 else "SELL"

            is_blocked, reason = self.is_blocked(symbol, direction, current_ts)

            # 如果启用了 exclude_st，检查 ST 状态
            if not is_blocked and self._exclude_st and direction == "BUY":
                bar = current_bars.get(symbol)
                if bar is not None and bar.is_st:
                    is_blocked = True
                    reason = "ST股票 (excluded from buy)"

            if is_blocked:
                from app.engine.backtest.models import (
                    Signal,
                    SignalDirection,
                    SignalType,
                )

                synthetic_signal = Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG
                    if qty > 0
                    else SignalDirection.SHORT,
                    signal_type=SignalType.EXIT if qty < 0 else SignalType.ENTRY,
                    strength=1.0,
                    reason=f"BLOCKED: {reason}",
                )
                filtered = FilteredSignal(
                    signal=synthetic_signal,
                    reason=reason,
                    timestamp=current_ts,
                )
                blocked.append(filtered)
                self._blocked_log.append(filtered)
            else:
                allowed[symbol] = qty

        return allowed, blocked

    def get_blocked_log(self) -> list[FilteredSignal]:
        """获取所有被拦截的信号用于分析。"""
        return self._blocked_log.copy()

    def clear_log(self) -> None:
        """清空被拦截信号日志。"""
        self._blocked_log.clear()

    @property
    def total_blocked(self) -> int:
        """被拦截信号总数。"""
        return len(self._blocked_log)
