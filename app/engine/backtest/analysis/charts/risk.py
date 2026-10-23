"""风险 + 风险调整图表 — 日收益分布、滚动夏普/回撤。"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from app.engine.backtest.analysis.charts.core import (
    PALETTE,
    new_figure,
    percent_formatter,
    save_and_close,
)

TRADING_DAYS_PER_YEAR = 242
DEFAULT_RISK_FREE_RATE = 0.02


def daily_return_distribution(
    daily_returns: list[float],
    out: str = "",
) -> None:
    """日收益直方图 + 正态曲线叠加。"""
    rets = np.array(daily_returns) * 100
    rets = rets[~np.isnan(rets)]

    fig, ax = new_figure(figsize=(10, 5))
    ax.hist(
        rets,
        bins=60,
        density=True,
        alpha=0.6,
        color=PALETTE["equity"],
        edgecolor="white",
    )

    # 正态曲线（纯 numpy 实现，不依赖 scipy）
    mu, sigma = np.mean(rets), np.std(rets)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(
        x,
        pdf,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"N({mu:.2f}, {sigma:.2f})",
    )

    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_xlabel("日收益 (%)")
    ax.set_ylabel("密度")
    ax.set_title("日收益分布")
    ax.legend()

    skew = float(np.mean(((rets - mu) / sigma) ** 3)) if sigma > 0 else 0
    kurt = float(np.mean(((rets - mu) / sigma) ** 4)) - 3 if sigma > 0 else 0
    ax.text(
        0.02,
        0.95,
        f"偏度={skew:.2f}  峰度={kurt:.2f}  VaR95={np.percentile(rets, 5):.2f}%",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.annotate(
        "导读：蓝柱=实际日收益分布，虚线=正态分布。\n"
        "如果左边尾巴比正态肥（峰度>1），说明极端亏损比预期多。偏度<0=亏的日子比赚的日子更极端。",
        xy=(0.5, -0.28),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.25)

    if out:
        save_and_close(fig, out)


def rolling_sharpe_drawdown(
    timestamps: list[datetime],
    daily_returns: list[float],
    equity_values: list[float],
    window: int = TRADING_DAYS_PER_YEAR,
    out: str = "",
) -> None:
    """滚动夏普 + 滚动最大回撤（双图）。"""
    rets = np.array(daily_returns)
    values = np.array(equity_values)
    pad = min(window, len(rets))

    # 滚动夏普
    rolling_sharpe = []
    for i in range(pad, len(rets) + 1):
        w = rets[:i][-window:]
        excess = w - DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        s = (
            np.mean(excess) / np.std(w) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if np.std(w) > 0
            else 0
        )
        rolling_sharpe.append(s)

    # 滚动最大回撤
    rolling_mdd = []
    for i in range(pad, len(values)):
        w = values[: i + 1][-window:]
        peak = np.maximum.accumulate(w)
        rolling_mdd.append(np.min((w - peak) / peak))

    ts_sharpe = timestamps[pad:]
    ts_mdd = timestamps[pad : len(rolling_mdd) + pad][: len(rolling_mdd)]
    min_len = min(len(ts_sharpe), len(ts_mdd), len(rolling_sharpe), len(rolling_mdd))
    ts_sharpe = ts_sharpe[:min_len]
    ts_mdd = ts_mdd[:min_len]
    rolling_sharpe = rolling_sharpe[:min_len]
    rolling_mdd = rolling_mdd[:min_len]

    fig, (ax1, ax2) = new_figure(2, 1, figsize=(14, 7))
    fig.subplots_adjust(hspace=0.2)

    ax1.plot(ts_sharpe, rolling_sharpe, color=PALETTE["sharpe"], linewidth=0.8)
    ax1.axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("滚动夏普")
    ax1.set_title("滚动 1 年夏普 + 最大回撤")

    ax2.fill_between(ts_mdd, 0, rolling_mdd, color=PALETTE["drawdown"], alpha=0.4)
    ax2.plot(ts_mdd, rolling_mdd, color=PALETTE["drawdown"], linewidth=0.8)
    ax2.set_ylabel("滚动最大回撤")
    percent_formatter(ax2)

    ax2.annotate(
        "导读：上图夏普>0 说明过去一年赚钱，下图是过去一年的最大回撤。\n"
        "如果夏普长期低于 0 或回撤越来越大，策略正在恶化。",
        xy=(0.5, -0.35),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.22)

    if out:
        save_and_close(fig, out)
