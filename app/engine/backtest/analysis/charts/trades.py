"""交易行为图表 — 交易散点图（持仓天数 vs 收益率）。"""

from __future__ import annotations

from collections import deque

import numpy as np

from app.engine.backtest.analysis.charts.core import (
    PALETTE,
    new_figure,
    save_and_close,
)


def trade_scatter(trade_log: list, out: str = "") -> None:
    """每笔交易一个点：横轴持仓天数，纵轴收益率，颜色=盈亏。"""
    pairs = _match_pairs(trade_log)
    if not pairs:
        return

    days = [p["days"] for p in pairs if p["days"] is not None]
    pnls = [p["pnl"] for p in pairs]
    sizes = [p["qty"] for p in pairs]

    if not days or not pnls:
        return

    fig, ax = new_figure(figsize=(10, 6))

    pos_days = [d for i, d in enumerate(days) if pnls[i] >= 0]
    pos_pnls = [pnls[i] for i in range(len(days)) if pnls[i] >= 0]
    pos_sizes = [sizes[i] for i in range(len(days)) if pnls[i] >= 0]

    neg_days = [d for i, d in enumerate(days) if pnls[i] < 0]
    neg_pnls = [pnls[i] for i in range(len(days)) if pnls[i] < 0]
    neg_sizes = [sizes[i] for i in range(len(days)) if pnls[i] < 0]

    size_factor = 50
    max_size = 200  # 限制最大圆圈，避免极端仓位撑破图

    # 裁剪极端 PnL（取 98 分位），持仓天数不截断
    pnl_limit = (
        np.percentile([abs(p) for p in pnls], 98)
        if len(pnls) > 5
        else max(abs(p) for p in pnls) * 1.2
    )
    pnl_outliers = sum(1 for p in pnls if abs(p) > pnl_limit)

    if pos_days:
        ax.scatter(
            pos_days,
            pos_pnls,
            s=[min(s * size_factor, max_size) for s in pos_sizes],
            c=PALETTE["scatter_profit"],
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
            label="Profit",
        )
    if neg_days:
        ax.scatter(
            neg_days,
            neg_pnls,
            s=[min(s * size_factor, max_size) for s in neg_sizes],
            c=PALETTE["scatter_loss"],
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
            label="Loss",
        )

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Holding Days")
    ax.set_ylabel("P&L")
    ax.set_title("Trade Scatter (size = quantity)")
    ax.legend(loc="upper right")

    # 坐标轴限制 — x 轴不截断，y 轴截断极端 PnL
    ax.set_ylim(-pnl_limit * 1.1, pnl_limit * 1.1)

    # 统计
    win_rate = len(pos_days) / len(days) if days else 0
    avg_hold = np.mean(days) if days else 0
    info = f"Trades={len(days)}  WinRate={win_rate:.1%}  AvgHold={avg_hold:.0f}d"
    if pnl_outliers:
        info += f"  [PnL clip: {pnl_outliers} outliers]"
    ax.text(
        0.02,
        0.95,
        info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if out:
        save_and_close(fig, out)


def _match_pairs(trade_log: list) -> list[dict]:
    """FIFO 匹配买卖对。"""
    lots: dict[str, deque] = {}
    pairs: list[dict] = []

    for t in sorted(trade_log, key=lambda x: x.timestamp):
        sym = getattr(t, "symbol", "UNKNOWN")
        qty = abs(t.quantity) if hasattr(t, "quantity") else 0
        is_buy = t.is_buy if hasattr(t, "is_buy") else True
        price = float(t.price) if hasattr(t, "price") else 0
        ts = t.timestamp

        if is_buy:
            lots.setdefault(sym, deque()).append({"qty": qty, "price": price, "ts": ts})
        else:
            q = lots.get(sym, deque())
            remaining = qty
            cost = 0.0
            first_ts = ts
            while remaining > 0 and q:
                lot = q.popleft()
                used = min(remaining, lot["qty"])
                cost += used * lot["price"]
                first_ts = lot["ts"]
                remaining -= used
                if lot["qty"] > used:
                    q.appendleft(
                        {
                            "qty": lot["qty"] - used,
                            "price": lot["price"],
                            "ts": lot["ts"],
                        }
                    )
            if remaining < qty:
                pnl = qty * price - cost
                days_val = (ts - first_ts).days if first_ts else None
                pairs.append({"pnl": pnl, "days": days_val, "qty": qty, "symbol": sym})

    return pairs
