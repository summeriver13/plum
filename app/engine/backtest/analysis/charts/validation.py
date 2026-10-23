"""策略真实性图表 — QQ 图、水下热力图。"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd

from app.engine.backtest.analysis.charts.core import (
    PALETTE,
    new_figure,
    save_and_close,
)


def _approx_erfinv(x: float) -> float:
    """近似逆误差函数（用于 Q-Q 图，替代 scipy.stats.norm.ppf）。"""
    if abs(x) >= 1:
        return np.inf if x > 0 else -np.inf
    a = 0.147
    t = np.log(1 - x * x) / 2 + 2 / (np.pi * a)
    return np.sign(x) * np.sqrt(np.sqrt(t * t - np.log(1 - x * x) / a) - t)


def qq_plot(daily_returns: list[float], out: str = "") -> None:
    """QQ 图：日收益 vs 理论正态分位数。"""
    rets = np.array(daily_returns)
    rets = rets[~np.isnan(rets)]
    n = len(rets)

    if n < 10:
        return

    fig, ax = new_figure(figsize=(7, 7))

    # QQ 图：用 numpy 实现，不依赖 scipy
    sorted_rets = np.sort(rets)
    n = len(sorted_rets)
    theoretical = np.array(
        [
            np.sqrt(2) * _approx_erfinv(2 * (i + 0.5) / n - 1) * np.std(rets)
            + np.mean(rets)
            for i in range(n)
        ]
    )
    ax.scatter(theoretical, sorted_rets, s=15, alpha=0.5, color=PALETTE["equity"])
    ax.plot(theoretical, theoretical, color="red", linewidth=1, linestyle="--")

    ax.set_xlabel("理论分位数")
    ax.set_ylabel("样本分位数")
    ax.set_title("Q-Q 图 — 日收益正态性检验")
    ax.annotate(
        "导读：蓝点应落在红色虚线上才符合正态分布。\n"
        "两端偏离红线=厚尾（大亏大赚比预期多）。整体弯曲=偏态。",
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.18)

    if out:
        save_and_close(fig, out)


def underwater_heatmap(
    timestamps: list[datetime],
    equity_values: list[float],
    out: str = "",
) -> None:
    """水下热力图 — 横轴月/纵轴年，颜色=当月最大回撤深度。"""
    values = np.array(equity_values)
    df = pd.DataFrame(
        {
            "date": timestamps,
            "value": values,
        }
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 计算每个时间点的回撤
    peak = np.maximum.accumulate(values)
    df["dd"] = (values - peak) / peak

    # 按月聚合：取当月最深的回撤
    grouped = cast("pd.Series", df.groupby(["year", "month"])["dd"].min())
    monthly = cast("pd.DataFrame", grouped.unstack())
    grid = np.asarray(monthly.values, dtype=float)

    fig, ax = new_figure(figsize=(12, max(3, len(monthly) * 0.55)))
    im = ax.imshow(
        grid * 100,
        aspect="equal",
        cmap="YlOrRd",
        vmin=-30,
        vmax=0,
        interpolation="nearest",
    )

    # 白色格线分隔
    rows, cols = grid.shape
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [
            "1月",
            "2月",
            "3月",
            "4月",
            "5月",
            "6月",
            "7月",
            "8月",
            "9月",
            "10月",
            "11月",
            "12月",
        ],
        fontsize=9,
    )
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index.astype(str), fontsize=9)
    ax.set_title("水下热力图 — 月度最大回撤 (%)", fontsize=12, pad=12)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("回撤 (%)", fontsize=9)

    # 数值标注
    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            if not np.isnan(val):
                depth = abs(val) / 30  # 归一化深度
                color = "white" if depth > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    weight="bold" if depth > 0.85 else "normal",
                )

    # 底部导读
    ax.annotate(
        "导读：颜色越深（红）回撤越大，白色=无回撤。\n"
        "如果每年同一月份出现深色块，策略有季节性弱点。",
        xy=(0.5, -0.25),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.22)

    if out:
        save_and_close(fig, out)
