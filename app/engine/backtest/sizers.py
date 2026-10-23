"""仓位分配器 — 信号 → 目标股数的可替换策略。

将信号管线处理后的信号转换为具体的 target_positions dict。
内置实现：
    EqualRiskSizer     — 等风险分配（当前引擎默认行为）
    RankWeightedSizer  — 按信号强度非等权分配

用法:
    from app.engine.backtest.sizers import RankWeightedSizer

    engine = Engine(sizer=RankWeightedSizer(kelly_multiplier=0.8), ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from app.engine.backtest.models import Signal, SignalDirection


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/NaN/非数值返回 default，即不下单）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, InvalidOperation):
        return default


logger = logging.getLogger(__name__)


# ── 上下文 ──────────────────────────────────────────────


@dataclass
class SizerContext:
    """仓位分配上下文。"""

    portfolio_value: Decimal
    current_holdings: dict[str, int]
    prices: dict[str, Decimal]
    max_positions: int
    rules: Any | None = None
    kelly_multiplier: Decimal = Decimal("1.0")


# ── 协议 ────────────────────────────────────────────────


class PositionSizer(Protocol):
    """仓位分配器协议。"""

    def size(
        self,
        signals: list[Signal],
        context: SizerContext,
    ) -> dict[str, int]: ...


# ── 内置实现 ────────────────────────────────────────────


@dataclass
class EqualRiskSizer:
    """等风险分配 — 每个 LONG 信号获得均等的资金分配。

    这是引擎当前的默认行为，提取为独立组件以便替换。

    公式:
        per_position = portfolio_value × strength × kelly / max_positions
        qty = floor(per_position / price) → lot_size 取整
    """

    kelly_multiplier: Decimal = Decimal("1.0")
    """凯利乘数。1.0=满凯利，0.5=半凯利。"""

    def size(
        self,
        signals: list[Signal],
        context: SizerContext,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        allocated = 0
        max_pos = context.max_positions

        for sig in signals:
            if sig.direction == SignalDirection.FLAT:
                current_qty = context.current_holdings.get(sig.symbol, 0)
                if current_qty > 0:
                    result[sig.symbol] = 0
                continue

            if sig.direction == SignalDirection.LONG:
                current_qty = context.current_holdings.get(sig.symbol, 0)
                if current_qty > 0:
                    result[sig.symbol] = current_qty
                    continue

                if allocated >= max_pos:
                    continue

                price = context.prices.get(sig.symbol, Decimal("0"))
                if price <= Decimal("0"):
                    continue

                per_position = (
                    context.portfolio_value
                    * Decimal(str(sig.strength))
                    * (self.kelly_multiplier or context.kelly_multiplier)
                ) / Decimal(max_pos)

                qty_float = per_position / price
                qty = _int(qty_float)

                # 取整到整手
                if context.rules is not None:
                    qty = context.rules.floor_to_lot(sig.symbol, qty)
                else:
                    qty = (qty // 100) * 100

                if qty >= 100:
                    result[sig.symbol] = qty
                    allocated += 1

        return result


@dataclass
class RankWeightedSizer:
    """按信号强度非等权分配。

    信号 strength 越高，分配的资金比例越大。
    分配公式:
        weight_i = strength_i / Σ strength_j
        position_value_i = portfolio_value × weight_i × kelly
        qty_i = position_value_i / price_i

    适合因子分越高的股票拿越多仓位。
    """

    kelly_multiplier: Decimal = Decimal("1.0")
    max_positions: int = 20

    def size(
        self,
        signals: list[Signal],
        context: SizerContext,
    ) -> dict[str, int]:
        result: dict[str, int] = {}

        # ── 先处理 FLAT ──
        for sig in signals:
            if sig.direction == SignalDirection.FLAT:
                current_qty = context.current_holdings.get(sig.symbol, 0)
                if current_qty > 0:
                    result[sig.symbol] = 0

        # ── 收集所有 LONG 信号 ──
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        if not long_signals:
            return result

        # ── 按 strength 降序排列 ──
        long_signals.sort(key=lambda s: s.strength, reverse=True)

        # ── 只取前 max_positions ──
        selected = long_signals[: min(len(long_signals), self.max_positions)]
        total_strength = sum(s.strength for s in selected)
        if total_strength <= 0:
            return result

        for sig in selected:
            price = context.prices.get(sig.symbol, Decimal("0"))
            if price <= Decimal("0"):
                continue

            weight = Decimal(str(sig.strength / total_strength))
            position_value = context.portfolio_value * weight * self.kelly_multiplier

            qty_float = position_value / price
            qty = _int(qty_float)

            if context.rules is not None:
                qty = context.rules.floor_to_lot(sig.symbol, qty)
            else:
                qty = (qty // 100) * 100

            if qty >= 100:
                result[sig.symbol] = qty

        return result


__all__ = [
    "PositionSizer",
    "SizerContext",
    "EqualRiskSizer",
    "RankWeightedSizer",
]
