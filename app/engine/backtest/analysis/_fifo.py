"""统一 FIFO 买卖匹配引擎。

全部 FIFO 匹配依赖此模块，消除 analyzer / analyzer_ext / report / charts 中
四份独立实现的重复代码。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """一笔 FIFO 匹配的买卖对。"""

    symbol: str
    buy_timestamp: datetime
    sell_timestamp: datetime
    qty: int
    entry_price: Decimal  # 买入均价
    exit_price: Decimal  # 卖出均价
    realized_pnl: Decimal  # 已实现盈亏（含费用）
    gross_pnl: Decimal  # 买卖价差（不含费用）
    buy_fee: Decimal  # 买入端费用
    sell_fee: Decimal  # 卖出端费用
    holding_days: int  # 持仓天数

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


def fifo_match(
    trades: list,
    *,
    with_fees: bool = False,
) -> list[MatchedPair]:
    """FIFO 匹配买卖对 — 单次遍历，O(n)。

    Args:
        trades: TradeRecord 列表。
                每个对象需有: .symbol, .timestamp, .is_buy, .quantity, .price
                当 with_fees=True 时还需: .cost (含 .total 属性)
        with_fees: 是否按成交比例分摊买卖费用（佣金+过户费+印花税）

    Returns:
        MatchedPair 列表，按卖出时间排序
    """
    lots: dict[str, deque[dict]] = {}
    pairs: list[MatchedPair] = []

    # 按时间排序
    sorted_trades = sorted(trades, key=lambda t: t.timestamp)

    for t in sorted_trades:
        sym = getattr(t, "symbol", "UNKNOWN")
        qty = abs(getattr(t, "quantity", 0))
        is_buy = getattr(t, "is_buy", True)
        price_raw = getattr(t, "price", 0)
        price = price_raw if isinstance(price_raw, Decimal) else Decimal(str(price_raw))
        ts = t.timestamp

        if is_buy:
            fee = _cost_total(t) if with_fees else Decimal("0")
            lots.setdefault(sym, deque()).append(
                dict(qty=qty, price=price, ts=ts, fee=fee)
            )
        else:
            if sym not in lots or not lots[sym]:
                continue
            fee = _cost_total(t) if with_fees else Decimal("0")
            remaining = qty
            while remaining > 0 and lots[sym]:
                lot = lots[sym][0]
                matched = min(remaining, lot["qty"])
                lot_qty = max(lot["qty"], 1)
                ratio = Decimal(matched) / Decimal(lot_qty)

                buy_fee = lot["fee"] * ratio if with_fees else Decimal("0")
                sell_fee = (
                    fee * Decimal(matched) / Decimal(max(qty, 1))
                    if with_fees
                    else Decimal("0")
                )
                gross = (price - lot["price"]) * Decimal(matched)
                net = gross - buy_fee - sell_fee

                pairs.append(
                    MatchedPair(
                        symbol=sym,
                        buy_timestamp=lot["ts"],
                        sell_timestamp=ts,
                        qty=matched,
                        entry_price=lot["price"],
                        exit_price=price,
                        realized_pnl=net,
                        gross_pnl=gross,
                        buy_fee=buy_fee,
                        sell_fee=sell_fee,
                        holding_days=(ts - lot["ts"]).days,
                    )
                )

                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 0:
                    lots[sym].popleft()

    return pairs


def pairs_to_df(pairs: list[MatchedPair]) -> pd.DataFrame:
    """将 MatchedPair 列表转为 DataFrame（用于 charts 导出 CSV）。"""
    return pd.DataFrame(
        [
            {
                "代码": p.symbol,
                "持仓天数": p.holding_days,
                "盈亏": float(p.gross_pnl),
                "数量": p.qty,
                "买入日": p.buy_timestamp,
                "卖出日": p.sell_timestamp,
            }
            for p in pairs
        ]
    )


def pairs_summary(pairs: list[MatchedPair]) -> dict:
    """从匹配对计算汇总统计（用于 report.py）。"""
    if not pairs:
        return {"pnl": 0.0, "days": None}
    wins = [p for p in pairs if p.realized_pnl > 0]
    return {
        "total_pairs": len(pairs),
        "wins": len(wins),
        "win_rate": len(wins) / len(pairs) if pairs else 0.0,
        "total_pnl": float(sum(p.realized_pnl for p in pairs)),
        "avg_holding_days": float(sum(p.holding_days for p in pairs)) / len(pairs),
    }


# ── 内部辅助 ───────────────────────────────────────────


def _cost_total(trade) -> Decimal:
    """提取 trade.cost.total，失败返回 0。"""
    cost = getattr(trade, "cost", None)
    if cost is None:
        return Decimal("0")
    total = getattr(cost, "total", None)
    if total is None:
        return Decimal("0")
    return total if isinstance(total, Decimal) else Decimal(str(total))
