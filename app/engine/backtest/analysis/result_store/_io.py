"""I/O 辅助 — DataFrame 转换与 JSON 序列化。"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from app.engine.backtest.analysis._numeric import to_float


def _safe_float(value: object, default: float = 0.0) -> float:
    """容错转 float（None / 缺失 → default）。

    CSV 回写路径不用 ``to_float``：它的 None 语义是 NaN，
    而交易字段缺失时应回落到 0 而不是污染下游金额汇总。
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _equity_to_df(curve: list) -> pd.DataFrame:
    rows = []
    for p in curve:
        rows.append(
            {
                "timestamp": p.timestamp,
                "total_value": to_float(p.total_value),
                "cash": to_float(p.cash),
                "frozen_cash": to_float(p.frozen_cash),
                "positions_value": to_float(p.positions_value),
                "cash_interest": to_float(getattr(p, "cash_interest", 0)),
                "parking_value": to_float(getattr(p, "parking_value", 0)),
            }
        )
    return pd.DataFrame(rows)


def _parking_to_df(parking_log: list) -> pd.DataFrame:
    """停泊成交记录（不计入策略换手）。"""
    rows = []
    for t in parking_log:
        rows.append(
            {
                "symbol": t.symbol,
                "timestamp": t.timestamp,
                "quantity": t.quantity,
                "price": _safe_float(t.price),
                "is_buy": t.is_buy,
                "notional": _safe_float(t.notional),
                "fee": _safe_float(t.fee),
                "reason": t.reason,
            }
        )
    return pd.DataFrame(rows)


def _trades_to_df(trade_log: list) -> pd.DataFrame:
    rows = []
    for t in trade_log:
        cost = getattr(t, "cost", None)
        rows.append(
            {
                "symbol": t.symbol,
                "timestamp": t.timestamp,
                "quantity": t.quantity,
                "price": _safe_float(t.price),
                "is_buy": t.is_buy,
                "amount": _safe_float(getattr(t, "amount", 0)),
                "commission": _safe_float(getattr(cost, "commission", 0)),
                "stamp_duty": _safe_float(getattr(cost, "stamp_duty", 0)),
                "transfer_fee": _safe_float(getattr(cost, "transfer_fee", 0)),
                "slippage_cost": _safe_float(getattr(t, "slippage_cost", 0)),
            }
        )
    return pd.DataFrame(rows)


def _serialize_dict(d: dict) -> dict:
    """将 Decimal/Enum 转为 JSON 可序列化格式。"""
    out = {}
    for k, v in d.items():
        if isinstance(v, Decimal):
            out[k] = _safe_float(v)
        elif hasattr(v, "value"):
            out[k] = v.value
        else:
            out[k] = str(v)
    return out
