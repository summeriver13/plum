"""扩展图表 — 滚动收益曲线、回撤特写、周历效应、仓位暴露。"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd

from app.engine.backtest.analysis.charts.core import (
    PALETTE,
    new_figure,
    percent_formatter,
    save_and_close,
)

TRADING_DAYS = 242


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值返回 default）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def rolling_annual_return(
    timestamps: list[datetime],
    daily_returns: list[float],
    window: int = TRADING_DAYS,
    out: str = "",
) -> None:
    """滚动 1 年收益率曲线 — 看策略收益是否在衰减。"""
    rets = np.array(daily_returns)
    n = len(rets)
    if n < window:
        return

    rolling = []
    for i in range(window, n + 1):
        w = rets[:i][-window:]
        r = (1 + w).prod() ** (TRADING_DAYS / len(w)) - 1
        rolling.append(r)

    ts = timestamps[window:]
    rolling = rolling[: len(ts)]

    fig, ax = new_figure(figsize=(12, 5))
    ax.plot(ts, [r * 100 for r in rolling], color=PALETTE["equity"], linewidth=1.2)
    ax.axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    ax.axhline(
        y=np.mean(rolling) * 100,
        color="gray",
        linewidth=0.5,
        linestyle=":",
        label=f"Avg={np.mean(rolling) * 100:.1f}%",
    )
    ax.fill_between(
        ts, 0, [r * 100 for r in rolling], alpha=0.15, color=PALETTE["equity"]
    )
    ax.set_ylabel("年化收益 (%)")
    ax.set_title("滚动 1 年年化收益")
    ax.legend(fontsize=9)
    ax.annotate(
        "导读：这条线越平稳越好。如果后半段明显下滑，说明策略在过去一年失效了。\n"
        "虚线是均值，低于均值的时间越长越危险。",
        xy=(0.5, -0.22),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.2)

    if out:
        save_and_close(fig, out)


def drawdown_showcase(
    timestamps: list[datetime],
    equity_values: list[float],
    top_n: int = 5,
    out: str = "",
) -> None:
    """最差 N 次回撤特写 — 看恢复速度。"""
    values = np.array(equity_values)
    ts = timestamps

    # 找出所有回撤区间
    peak = values[0]
    in_dd = False
    dd_start = 0
    dd_peak = values[0]
    min_val = values[0]
    drawdowns: list[dict] = []

    for i, v in enumerate(values):
        if v > peak:
            peak = v
            if in_dd:
                dd = (dd_peak - min_val) / dd_peak
                drawdowns.append(
                    {
                        "start": dd_start,
                        "end": i - 1,
                        "depth": dd,
                        "peak_idx": values.tolist().index(
                            dd_peak, max(0, dd_start - 50)
                        ),
                    }
                )
                in_dd = False
        dd = (peak - v) / peak
        if dd > 0 and not in_dd:
            in_dd = True
            dd_start = i
            dd_peak = peak
            min_val = v
        elif in_dd:
            min_val = min(min_val, v)

    if in_dd:
        dd = (dd_peak - min_val) / dd_peak
        drawdowns.append(
            {
                "start": dd_start,
                "end": len(values) - 1,
                "depth": dd,
                "peak_idx": values.tolist().index(dd_peak, max(0, dd_start - 50)),
            }
        )

    drawdowns.sort(key=lambda x: x["depth"], reverse=True)
    top = drawdowns[:top_n]

    fig, ax = new_figure(figsize=(12, 6))
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    for idx, dd in enumerate(top):
        span = max(20, dd["end"] - dd["start"] + 1)
        seg_start = dd["start"] - span // 3
        seg_end = dd["end"] + span // 3
        seg_start = max(0, seg_start)
        seg_end = min(len(values), seg_end)

        peak_v = values[dd["peak_idx"]]
        seg_vals = np.array(values[seg_start:seg_end]) / peak_v
        x_days = np.arange(len(seg_vals))

        ax.plot(
            x_days,
            seg_vals,
            color=colors[idx],
            linewidth=1.2,
            label=f"#{idx + 1} {ts[dd['start']].strftime('%Y-%m')} DD={dd['depth']:.1%}",
        )

    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("净值 (相对于回撤前高点)")
    ax.set_xlabel("交易日")
    ax.set_title("最差 5 次回撤特写 — 看恢复速度")
    ax.legend(fontsize=8, loc="lower left")
    ax.annotate(
        "导读：每条线是独立的一次回撤，从回撤前高点=1.00 开始。\n"
        "线越早回到 1.0 以上说明恢复越快。如果多条线长期低于 1.0，策略有生存危机。",
        xy=(0.5, -0.22),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.2)

    if out:
        save_and_close(fig, out)


def weekly_distribution(
    timestamps: list[datetime],
    daily_returns: list[float],
    out: str = "",
) -> None:
    """周内收益分布 — 箱线图看周历效应。"""
    df = pd.DataFrame({"date": timestamps[1:], "ret": daily_returns})
    df["day"] = df["date"].apply(lambda d: d.strftime("%a"))
    df["ret_pct"] = df["ret"] * 100
    order = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    day_col = cast("pd.Series", df["day"])
    ret_pct = cast("pd.Series", df["ret_pct"])

    fig, ax = new_figure(figsize=(10, 5))
    data = [ret_pct[day_col == d].dropna().values for d in order]
    bp = ax.boxplot(
        data,
        labels=["周一", "周二", "周三", "周四", "周五"],
        patch_artist=True,
        showfliers=False,
        widths=0.4,
    )

    for patch, color in zip(
        bp["boxes"],
        [
            PALETTE["profit"],
            PALETTE["profit"],
            PALETTE["equity"],
            PALETTE["equity"],
            PALETTE["drawdown"],
        ],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("日收益 (%)")
    ax.set_title("周内收益分布")

    # 均值标注
    for i, d in enumerate(order):
        vals = ret_pct[day_col == d].dropna()
        if len(vals) > 0:
            ax.text(
                i + 1,
                vals.mean(),
                f"{vals.mean():.1f}%",
                ha="center",
                fontsize=8,
                color="black",
                va="bottom" if vals.mean() >= 0 else "top",
            )

    ax.annotate(
        "导读：箱体中间线是中位数，箱子上下是 25%/75% 分位。\n"
        "如果某几天中位数明显偏离零，说明存在周历效应——策略在这些日子系统性跑赢或跑输。",
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


def position_exposure(
    timestamps: list[datetime],
    equity_values: list[float],
    cash_values: list[float],
    out: str = "",
) -> None:
    """仓位暴露曲线 — 总仓位比例 + 现金占比。"""
    values = np.array(equity_values)
    cash = np.array(cash_values)
    exposure = (values - cash) / values
    exposure = np.clip(exposure, 0, 1)

    fig, (ax1, ax2) = new_figure(2, 1, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.25)

    # 上图：仓位暴露
    ax1.fill_between(timestamps, 0, exposure, color=PALETTE["equity"], alpha=0.4)
    ax1.plot(timestamps, exposure, color=PALETTE["equity"], linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("仓位占比")
    ax1.set_title("仓位暴露时序")
    percent_formatter(ax1)

    avg_exp = _num(np.mean(exposure))
    ax1.axhline(
        y=avg_exp,
        color="red",
        linewidth=0.5,
        linestyle="--",
        label=f"平均仓位 {avg_exp:.1%}",
    )
    ax1.legend(fontsize=9)

    # 下图：仓位分布直方图
    ax2.hist(exposure, bins=20, color=PALETTE["profit"], alpha=0.5, edgecolor="white")
    ax2.axvline(x=avg_exp, color="red", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("仓位占比")
    ax2.set_ylabel("天数")
    ax2.set_title("仓位分布")

    # 标注
    idle = (exposure < 0.05).sum() / len(exposure) * 100
    full = (exposure > 0.95).sum() / len(exposure) * 100
    ax2.text(
        0.02,
        0.95,
        f"空仓: {idle:.0f}%  满仓: {full:.0f}%  平均: {avg_exp:.0%}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.subplots_adjust(bottom=0.12)
    ax2.annotate(
        "导读：上图是每天的总仓位占比，下图是仓位分布直方图。\n"
        "高仓位+高收益=策略择时能力强。高仓位+低收益=可能是在赌博。空仓天数多说明策略有选择性。",
        xy=(0.5, -0.55),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )

    if out:
        save_and_close(fig, out)
