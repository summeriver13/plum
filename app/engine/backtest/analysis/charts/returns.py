"""收益图表 — 净值曲线、回撤、月度热力图、年度柱状图。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.engine.backtest.analysis.charts.core import (
    PALETTE,
    new_figure,
    percent_formatter,
    save_and_close,
)


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值返回 default，NaN/inf 原样传递）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/非数值返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_year_month(key: object) -> tuple[int, int]:
    """把 ``stack().idxmax()`` 返回的索引键还原为 (年, 月)。

    Raises:
        TypeError: 索引键不是 (年, 月) 二元组——无法拼接标签，
            与其静默输出错误标签，不如显式报错。
    """
    if isinstance(key, tuple) and len(key) == 2:
        return _int(key[0]), _int(key[1])
    raise TypeError(f"预期 (年, 月) 索引键，实际: {key!r}")


def equity_curve(
    equity_values: list[float],
    timestamps: list[datetime],
    benchmark_values: list[float] | None = None,
    out: str = "",
) -> None:
    """净值曲线 + 回撤曲线（上下双图）。

    Args:
        equity_values: 策略每日净值
        timestamps: 日期序列
        benchmark_values: 基准每日净值（可选）
        out: 保存路径，空字符串表示不保存只展示
    """
    values = np.array(equity_values)
    init = values[0] if values[0] != 0 else 1
    nav = values / init

    # 计算回撤
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak

    fig, (ax1, ax2) = new_figure(2, 1, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.15)

    # 上图: 净值
    ax1.plot(timestamps, nav, color=PALETTE["equity"], linewidth=1.2, label="Strategy")
    if benchmark_values is not None:
        bm = np.array(benchmark_values)
        bm_nav = bm / bm[0] if bm[0] != 0 else bm
        ax1.plot(
            timestamps,
            bm_nav,
            color=PALETTE["benchmark"],
            linewidth=0.8,
            alpha=0.7,
            label="Benchmark",
        )
    ax1.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax1.set_ylabel("净值 (NAV)")
    ax1.legend(loc="upper left")
    ax1.set_title("净值曲线与回撤")

    # 下图: 回撤
    ax2.fill_between(timestamps, 0, drawdown, color=PALETTE["drawdown"], alpha=0.4)
    ax2.plot(timestamps, drawdown, color=PALETTE["drawdown"], linewidth=0.8)
    ax2.set_ylabel("回撤")
    percent_formatter(ax2)
    ax2.set_ylim(bottom=min(drawdown.min() * 1.2, -0.05))

    # 底部导读
    ax2.annotate(
        "导读：上图是策略净值曲线（起始=1.00），下图是回撤深度（负数）。\n"
        "净值长期在 1.00 以上=正收益。回撤面积越小越好，回到零轴越快越好。",
        xy=(0.5, -0.35),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.15)

    if out:
        save_and_close(fig, out)


def monthly_heatmap(
    timestamps: list[datetime],
    daily_returns: list[float],
    out: str = "",
) -> None:
    """月度收益热力图 — 横轴月/纵轴年，颜色=收益率。

    学术级：对称色阶居零、白色格线、汇总统计行。
    """
    df = pd.DataFrame(
        {
            "date": timestamps[1:],
            "ret": daily_returns,
        }
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = (
        df.groupby(["year", "month"])["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
        .unstack()
        * 100
    )

    v = max(abs(monthly.values.min()), abs(monthly.values.max()), 3.0)

    fig, ax = new_figure(figsize=(12, max(3, len(monthly) * 0.55)))
    im = ax.imshow(
        monthly.values,
        aspect="equal",
        cmap="RdYlGn",
        vmin=-v,
        vmax=v,
        interpolation="nearest",
    )

    # 手动绘制白色网格线
    rows, cols = monthly.shape
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        fontsize=9,
    )
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index.astype(str), fontsize=9)

    start = timestamps[0].strftime("%Y-%m") if timestamps else ""
    end = timestamps[-1].strftime("%Y-%m") if timestamps else ""
    ax.set_title(f"Monthly Returns (%)  |  {start} ~ {end}", fontsize=12, pad=12)

    # 标注数值 — 自适应颜色
    for i in range(rows):
        for j in range(cols):
            val = monthly.values[i, j]
            if not np.isnan(val):
                luminance = abs(val) / max(v, 1)
                color = "white" if luminance > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    weight="bold" if abs(val) > v * 0.8 else "normal",
                )

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Return (%)", fontsize=9)

    # 底部汇总行
    annotations = _heatmap_summary(monthly)
    ax.text(
        0.5,
        -0.25,
        annotations,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )

    # 调整底部边距给汇总行留空间
    fig.subplots_adjust(bottom=0.18)

    if out:
        save_and_close(fig, out)


def _heatmap_summary(monthly: pd.DataFrame) -> str:
    """生成热力图底部的汇总统计字符串。"""
    n_months = _int(monthly.notna().sum().sum())
    avg = _num(monthly.mean().mean())
    pos_ratio = _int((monthly > 0).sum().sum()) / max(n_months, 1) * 100
    best = _num(monthly.max().max())
    worst = _num(monthly.min().min())
    best_year, best_month = _as_year_month(monthly.stack().idxmax())
    worst_year, worst_month = _as_year_month(monthly.stack().idxmin())

    return (
        f"Avg: {avg:.1f}%  |  Positive: {pos_ratio:.0f}%  |  "
        f"Best: {best:.1f}% ({best_year}-{best_month:02d})  |  "
        f"Worst: {worst:.1f}% ({worst_year}-{worst_month:02d})  |  "
        f"Months: {n_months}"
    )


def yearly_bars(
    timestamps: list[datetime],
    daily_returns: list[float],
    out: str = "",
) -> None:
    """年度收益柱状图。"""
    df = pd.DataFrame(
        {
            "date": timestamps[1:],
            "ret": daily_returns,
        }
    )
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1) * 100

    fig, ax = new_figure(figsize=(12, 6))
    colors = [PALETTE["profit"] if v >= 0 else PALETTE["loss"] for v in yearly.values]
    ax.bar(yearly.index.astype(str), yearly.values, color=colors, width=0.35)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 扩大 Y 轴上下留白
    ymin = min(yearly.values.min() * 1.5, -5)
    ymax = max(yearly.values.max() * 1.5, 5)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("年收益 (%)")
    ax.set_title("年度收益")

    for i, v in enumerate(yearly.values):
        offset = ymax * 0.04
        ax.text(
            i,
            v + offset if v >= 0 else v - offset,
            f"{v:.1f}%",
            ha="center",
            fontsize=10,
        )

    ax.annotate(
        "导读：每年一根柱子，绿色=赚钱/红色=亏钱。\n"
        "如果连续两年以上是红色，策略可能有根本性问题。",
        xy=(0.5, -0.35),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="gray",
        fontstyle="italic",
    )
    fig.subplots_adjust(bottom=0.25)

    if out:
        save_and_close(fig, out)
