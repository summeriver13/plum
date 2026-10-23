"""交易级 alpha 归因：被买入/卖出标的的前瞻收益。

用途：解释「调仓频率差异」为何有 11.9pp/年 的结构性部分。
如果某个频率下「买入组」的前瞻收益系统性低于「卖出组」，说明该频率的换仓
本身是负 alpha 交易（越涨越卖、越跌越买），换手不只是成本问题。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config.paths import MARKET_PATH

HORIZONS: tuple[int, ...] = (5, 10, 20)  # 前瞻交易日
PERIOD = "1d"
BUY_ACTIONS = ("建仓", "加仓")
SELL_ACTIONS = ("减仓", "清仓")
ACTIONS = ("建仓", "加仓", "减仓", "清仓")
NAN = math.nan


def safe_float(value: Any, default: float = NAN) -> float:
    """容错转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def market_file(symbol: str, market_root: Path | None = None) -> Path:
    """把 SH.600000 / SZ.000001 映射到市场副本文件路径。"""
    root = Path(market_root) if market_root is not None else Path(MARKET_PATH)
    market, code = symbol.split(".", 1)
    return root / PERIOD / market / f"{code}.parquet"


def load_adjusted_close(symbol: str, market_root: Path | None = None) -> pd.Series:
    """读取复权收盘价序列（close / adj_factor，索引为交易日）。"""
    path = market_file(symbol, market_root)
    try:
        frame = pd.read_parquet(path, columns=["close", "adj_factor"])
    except (OSError, ValueError, KeyError) as exc:
        raise FileNotFoundError(f"读取行情失败: {path}（{exc}）") from exc
    close = frame["close"].astype(float)
    adj = frame["adj_factor"].astype(float).replace(0.0, NAN)
    series = (close / adj).dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


def load_trades(run_dir: Path | str) -> pd.DataFrame:
    """读取回测 trade_log.csv 并规整类型。"""
    path = Path(run_dir) / "data" / "trade_log.csv"
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"读取交易日志失败: {path}（{exc}）") from exc
    frame["timestamp"] = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    frame["symbol"] = frame["symbol"].astype(str)
    frame["quantity"] = frame["quantity"].astype(int)
    frame["is_buy"] = frame["is_buy"].astype(bool)
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def classify_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """按持仓变化把每笔交易标记为 建仓/加仓/减仓/清仓。"""
    frame = trades.copy()
    qty = frame["quantity"].to_numpy(dtype=int)
    is_buy = frame["is_buy"].to_numpy(dtype=bool)
    signed = np.where(is_buy, qty, -qty)
    frame["signed_qty"] = signed
    frame["holding_after"] = frame.groupby("symbol")["signed_qty"].cumsum()
    frame["holding_before"] = frame["holding_after"] - frame["signed_qty"]
    fresh = frame["holding_before"].to_numpy(dtype=int) == 0
    closed = frame["holding_after"].to_numpy(dtype=int) == 0
    frame["action"] = np.select(
        [is_buy & fresh, is_buy & ~fresh, ~is_buy & closed],
        ["建仓", "加仓", "清仓"],
        default="减仓",
    )
    return frame


def date_positions(series: pd.Series) -> dict[Any, int]:
    """交易日 → 位置索引，便于按交易日前瞻取值。"""
    return {pd.Timestamp(day): pos for pos, day in enumerate(series.index)}


def attach_forward_returns(
    trades: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """为每笔交易附加各期限的前瞻收益（复权口径，百分比）。"""
    frame = trades.copy()
    for horizon in horizons:
        frame[f"fwd_{horizon}d"] = np.nan
    for key, group in frame.groupby("symbol", sort=False):
        symbol = str(key)
        series = load_adjusted_close(symbol, market_root)
        positions = date_positions(series)
        values = series.to_numpy(dtype=float)
        days = [pd.Timestamp(day) for day in group["timestamp"]]
        for horizon in horizons:
            column = f"fwd_{horizon}d"
            out = np.full(len(group), NAN)
            for offset, day in enumerate(days):
                pos = positions.get(day)
                if pos is None or pos + horizon >= len(values):
                    continue
                base = safe_float(values[pos])
                if not np.isfinite(base) or base <= 0:
                    continue
                out[offset] = (safe_float(values[pos + horizon]) / base - 1.0) * 100.0
            frame.loc[group.index, column] = out
    return frame


def summarize(
    trades: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> dict[str, Any]:
    """按交易动作 × 前瞻期限汇总均值/中位数/样本数。"""
    table: dict[str, Any] = {}
    for action in ACTIONS:
        subset = trades[trades["action"] == action]
        entry: dict[str, Any] = {"count": len(subset)}
        for horizon in horizons:
            values = subset[f"fwd_{horizon}d"].dropna()
            entry[f"mean_{horizon}d"] = safe_float(values.mean())
            entry[f"median_{horizon}d"] = safe_float(values.median())
        table[action] = entry
    return table


def buy_sell_spread(trades: pd.DataFrame, horizon: int = 10) -> dict[str, float]:
    """买入组与卖出组的前瞻收益差（百分点），即该频率的「换仓 alpha」。"""
    column = f"fwd_{horizon}d"
    buys = trades[trades["action"].isin(BUY_ACTIONS)][column].dropna()
    sells = trades[trades["action"].isin(SELL_ACTIONS)][column].dropna()
    return {
        "buy_mean": safe_float(buys.mean()),
        "sell_mean": safe_float(sells.mean()),
        "spread": safe_float(buys.mean()) - safe_float(sells.mean()),
        "buy_n": len(buys),
        "sell_n": len(sells),
    }
