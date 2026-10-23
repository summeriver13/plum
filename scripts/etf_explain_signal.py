"""ETF 轮动可解释性：排序信号质量（候选池、IC、top-N 选择效应）。

复现策略的选股口径：250 日动量排序 → 流动性/波动率过滤 → （可选）MA 趋势过滤 →
同质去重填名额。前向收益取「调仓日收盘 → 下一调仓日收盘」（与引擎的收盘成交一致）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.strategy.native.etf_common import AMOUNT_WINDOW, CORR_WINDOW, VOL_WINDOW

TRADING_DAYS_PER_YEAR = 242
"""与引擎报告同口径的年化交易日数。"""


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（None/NaN/异常值回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _int(value: Any, default: int = 0) -> int:
    """安全转 int（None/NaN/越界回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass
class SignalConfig:
    """与批次口径一致的排序设置（默认 = v4 R36：250 日动量 × 无门槛 × top5）。"""

    momentum_window: int = 250
    ma_window: int = 200
    require_trend: bool = False
    min_amount: float = 1.0e7
    min_vol: float = 0.005
    max_corr: float = 0.95
    select_num: int = 5
    hold_days: int = 10
    trend_buffer: float = 0.0


def rebalance_dates(index: pd.DatetimeIndex, hold_days: int) -> list[pd.Timestamp]:
    """策略口径调仓日：第 hold_days、2×hold_days… 个交易日（从回测首日计数）。"""
    stamps: list[pd.Timestamp] = list(index)
    return [stamps[i] for i in range(hold_days - 1, len(stamps), hold_days)]


def candidates_at(
    adj_close: pd.DataFrame,
    amount: pd.DataFrame,
    date: pd.Timestamp,
    config: SignalConfig,
) -> pd.DataFrame:
    """给定日期的合格候选（动量降序）；列：symbol, momentum。"""
    history = adj_close.loc[:date]
    if len(history) < config.momentum_window + 1:
        return pd.DataFrame(columns=["symbol", "momentum"])

    last = history.iloc[-1]
    base = history.iloc[-config.momentum_window - 1]
    momentum = (last / base - 1.0).replace([math.inf, -math.inf], pd.NA).dropna()
    amount20 = amount.loc[:date].tail(AMOUNT_WINDOW).mean()
    # 注：不能先 dropna()——面板含未上市标的的 NaN，dropna 会清空整行；
    # std 默认按列 skipna，正是需要的「各标的自己的 20 日波动」
    vol = history.tail(VOL_WINDOW + 1).pct_change().std()

    keep = [
        symbol
        for symbol in momentum.index
        if _num(amount20.get(symbol)) >= config.min_amount
        and _num(vol.get(symbol)) >= config.min_vol
    ]
    if config.require_trend:
        ma = history.tail(config.ma_window).mean()
        keep = [
            symbol
            for symbol in keep
            if _num(last.get(symbol))
            >= _num(ma.get(symbol)) * (1.0 + config.trend_buffer)
        ]

    ranked = sorted(keep, key=lambda symbol: _num(momentum.get(symbol)), reverse=True)
    return pd.DataFrame(
        {"symbol": ranked, "momentum": [_num(momentum.get(s)) for s in ranked]}
    )


def pick_slots(
    adj_close: pd.DataFrame,
    date: pd.Timestamp,
    ranked: pd.DataFrame,
    config: SignalConfig,
) -> list[str]:
    """按排名填名额，跳过高相关同质标的（与策略 ``_fill_slots`` 同口径）。"""
    returns = adj_close.loc[:date].tail(CORR_WINDOW + 1).pct_change().dropna()
    slots: list[str] = []
    for symbol in ranked["symbol"]:
        if len(slots) >= config.select_num:
            break
        if _is_homogeneous(returns, symbol, slots, config.max_corr):
            continue
        slots.append(symbol)
    return slots


def _is_homogeneous(
    returns: pd.DataFrame,
    symbol: str,
    chosen: list[str],
    max_corr: float,
) -> bool:
    """与已选标的相关性超阈值（同指数不同 ETF）则视为同质。"""
    if max_corr >= 1.0 or not chosen or symbol not in returns.columns:
        return False
    for other in chosen:
        if other not in returns.columns:
            continue
        pair = pd.concat([returns[symbol], returns[other]], axis=1).dropna()
        if len(pair) < 20:
            continue
        left, right = pair.iloc[:, 0], pair.iloc[:, 1]
        if left.std() == 0 or right.std() == 0:
            continue
        if _num(pair.corr().iloc[0, 1]) > max_corr:
            return True
    return False


def evaluate_signal(
    adj_close: pd.DataFrame,
    amount: pd.DataFrame,
    dates: list[pd.Timestamp],
    config: SignalConfig,
) -> pd.DataFrame:
    """逐调仓日评估：候选数 / Spearman IC / top-N 与池子等权的前向收益 / 选择效应。"""
    rows: list[dict[str, Any]] = []
    for date in dates:
        position = _int(adj_close.index.get_indexer([date])[0])
        if position < 0 or position + config.hold_days >= len(adj_close):
            break
        ranked = candidates_at(adj_close, amount, date, config)
        if ranked.empty:
            continue

        momentum = pd.Series(ranked["momentum"].to_numpy(), index=ranked.index)
        forward = (
            adj_close.iloc[position + config.hold_days] / adj_close.iloc[position] - 1.0
        )
        forward_series = pd.Series(
            [_num(forward.get(symbol)) for symbol in ranked["symbol"]],
            index=ranked.index,
        )
        ic = _spearman(momentum, forward_series)
        slots = pick_slots(adj_close, date, ranked, config)
        in_slots = ranked["symbol"].isin(slots)
        top_fwd = _num(forward_series[in_slots].mean())
        pool_fwd = _num(forward_series.mean())
        rows.append(
            {
                "date": date,
                "candidates": len(ranked),
                "ic": _num(ic),
                "top_n": len(slots),
                "top_fwd": top_fwd,
                "pool_fwd": pool_fwd,
                "spread": top_fwd - pool_fwd,
                "slots": ",".join(slots),
            }
        )
    return pd.DataFrame(rows)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman 秩相关（先取秩再做 Pearson，避免 pandas corr 的重载歧义）。"""
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return 0.0
    ranks = pair.rank()
    return _num(ranks.iloc[:, 0].corr(ranks.iloc[:, 1]))


def summarize_signal(frame: pd.DataFrame, config: SignalConfig) -> dict[str, float]:
    """汇总 IC 与选择效应（含 t 值与纸面年化）。"""
    if frame.empty:
        return {}
    ic = frame["ic"]
    rounds = len(frame)
    ic_mean = _num(ic.mean())
    ic_std = _num(ic.std())
    ic_t = ic_mean / (ic_std / max(rounds, 1) ** 0.5) if ic_std > 0 else 0.0
    top_growth = _num((1.0 + frame["top_fwd"]).prod())
    pool_growth = _num((1.0 + frame["pool_fwd"]).prod())
    years = rounds * config.hold_days / TRADING_DAYS_PER_YEAR
    return {
        "rounds": _num(rounds),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_t": ic_t,
        "ic_positive_share": _num((ic > 0).mean()),
        "spread_mean": _num(frame["spread"].mean()),
        "spread_positive_share": _num((frame["spread"] > 0).mean()),
        "top_annual": top_growth ** (1.0 / years) - 1.0 if years > 0 else 0.0,
        "pool_annual": pool_growth ** (1.0 / years) - 1.0 if years > 0 else 0.0,
        "candidates_mean": _num(frame["candidates"].mean()),
    }
