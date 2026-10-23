"""回测图表包 — 12 张标准化图 + 一键出图。

每张图一个子文件夹，含 chart.png + data.csv。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config.paths import CHARTS_PATH
from app.engine.backtest.analysis._numeric import equity_to_numpy
from app.engine.backtest.analysis.charts.core import set_style
from app.engine.backtest.analysis.charts.exposure import (
    drawdown_showcase,
    position_exposure,
    rolling_annual_return,
    weekly_distribution,
)
from app.engine.backtest.analysis.charts.returns import (
    equity_curve,
    monthly_heatmap,
    yearly_bars,
)
from app.engine.backtest.analysis.charts.risk import (
    daily_return_distribution,
    rolling_sharpe_drawdown,
)
from app.engine.backtest.analysis.charts.trades import trade_scatter
from app.engine.backtest.analysis.charts.validation import (
    qq_plot,
    underwater_heatmap,
)

CHART_SPECS = [
    ("01_净值曲线", "策略净值与回撤"),
    ("02_月度收益热力图", "月度收益矩阵"),
    ("03_年度收益", "年度收益柱状图"),
    ("04_日收益分布", "日收益直方图 + 正态拟合"),
    ("05_滚动夏普回撤", "滚动 1 年夏普 + 最大回撤"),
    ("06_交易散点图", "持仓天数 vs 收益"),
    ("07_QQ图", "日收益正态性检验"),
    ("08_水下热力图", "月度最大回撤矩阵"),
    ("09_滚动年化收益", "滚动 1 年收益曲线"),
    ("10_回撤特写", "最差 5 次回撤特写"),
    ("11_周历效应", "周一到周五收益分布"),
    ("12_仓位暴露", "仓位占比时序 + 分布"),
]


def generate_all(
    result,
    report,
    output_dir: str = str(CHARTS_PATH),
    prefix: str = "",
    benchmark_values: list[float] | None = None,
) -> dict[str, str]:
    set_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    equity = result.equity_curve
    values = equity_to_numpy(equity)
    cash_vals = [float(p.cash) for p in equity]
    pos_vals = [float(p.positions_value) for p in equity]
    timestamps = [p.timestamp for p in equity]
    daily_rets = report.daily_returns
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak

    # 基准：如果没传就用等权组合近似（无基准时用虚线表示）
    bm_values = benchmark_values
    if bm_values is None and len(values) > 1:
        # 等权基准：归一化总资产作为简单对照
        bm_values = [v / values[0] for v in values]  # 归一化到 1.0

    saved: dict[str, str] = {}

    # --- 01 ---
    d = _dir(out, "01_净值曲线")
    equity_curve(
        values, timestamps, benchmark_values=bm_values, out=str(d / "chart.png")
    )
    _csv(
        d,
        "data.csv",
        {
            "日期": timestamps,
            "总资产": values,
            "现金": cash_vals,
            "持仓市值": pos_vals,
            "回撤": drawdowns.tolist(),
        },
    )
    saved["01"] = str(d / "chart.png")

    # --- 02 ---
    d = _dir(out, "02_月度收益热力图")
    monthly_heatmap(timestamps, daily_rets, out=str(d / "chart.png"))
    _df(d, "data.csv", _monthly_returns_df(timestamps, daily_rets))
    saved["02"] = str(d / "chart.png")

    # --- 03 ---
    d = _dir(out, "03_年度收益")
    yearly_bars(timestamps, daily_rets, out=str(d / "chart.png"))
    _df(d, "data.csv", _yearly_returns_df(timestamps, daily_rets))
    saved["03"] = str(d / "chart.png")

    # --- 04 ---
    d = _dir(out, "04_日收益分布")
    daily_return_distribution(daily_rets, out=str(d / "chart.png"))
    _csv(d, "data.csv", {"日收益": daily_rets})
    saved["04"] = str(d / "chart.png")

    # --- 05 ---
    if len(daily_rets) > 242:
        d = _dir(out, "05_滚动夏普回撤")
        rolling_sharpe_drawdown(
            timestamps, daily_rets, values, out=str(d / "chart.png")
        )
        s, m = _rolling_metrics(daily_rets, values, timestamps)
        _csv(d, "data.csv", {"日期": s[0], "滚动夏普": s[1], "滚动回撤": m[1]})
        saved["05"] = str(d / "chart.png")

    # --- 06 ---
    if result.trade_log:
        d = _dir(out, "06_交易散点图")
        trade_scatter(result.trade_log, out=str(d / "chart.png"))
        _df(d, "data.csv", _trade_pairs_df(result.trade_log))
        saved["06"] = str(d / "chart.png")

    # --- 07 ---
    if len(daily_rets) > 30:
        d = _dir(out, "07_QQ图")
        qq_plot(daily_rets, out=str(d / "chart.png"))
        sr = np.sort(daily_rets)
        n = len(sr)
        th = np.array(
            [
                np.sqrt(2) * _erfinv(2 * (i + 0.5) / n - 1) * np.std(sr) + np.mean(sr)
                for i in range(n)
            ]
        )
        _csv(d, "data.csv", {"理论分位": th.tolist(), "样本分位": sr.tolist()})
        saved["07"] = str(d / "chart.png")

    # --- 08 ---
    d = _dir(out, "08_水下热力图")
    underwater_heatmap(timestamps, values, out=str(d / "chart.png"))
    _df(d, "data.csv", _monthly_drawdown_df(timestamps, drawdowns.tolist()))
    saved["08"] = str(d / "chart.png")

    # --- 09 ---
    if len(daily_rets) > 242:
        d = _dir(out, "09_滚动年化收益")
        rolling_annual_return(timestamps, daily_rets, out=str(d / "chart.png"))
        rl = _rolling_returns(daily_rets, timestamps)
        _csv(d, "data.csv", {"日期": rl[0], "滚动年化收益": rl[1]})
        saved["09"] = str(d / "chart.png")

    # --- 10 ---
    d = _dir(out, "10_回撤特写")
    drawdown_showcase(timestamps, values, out=str(d / "chart.png"))
    saved["10"] = str(d / "chart.png")

    # --- 11 ---
    d = _dir(out, "11_周历效应")
    weekly_distribution(timestamps, daily_rets, out=str(d / "chart.png"))
    saved["11"] = str(d / "chart.png")

    # --- 12 ---
    d = _dir(out, "12_仓位暴露")
    position_exposure(timestamps, values, cash_vals, out=str(d / "chart.png"))
    _csv(
        d,
        "data.csv",
        {"日期": timestamps, "仓位占比": _exposure_series(values, cash_vals)},
    )
    saved["12"] = str(d / "chart.png")

    return saved


# --- helpers ---


def _dir(root, name) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _csv(folder, filename, data) -> None:
    lengths = {k: len(v) for k, v in data.items() if hasattr(v, "__len__")}
    mx = max(lengths.values()) if lengths else 1
    rows = []
    for i in range(mx):
        row = {}
        for k, v in data.items():
            row[k] = (
                v[i]
                if hasattr(v, "__len__") and i < len(v)
                else (v if not hasattr(v, "__len__") else None)
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / filename, index=False)


def _df(folder, filename, df) -> None:
    if not df.empty:
        df.to_csv(folder / filename, index=True)


def _monthly_returns_df(ts, rets) -> pd.DataFrame:
    df = pd.DataFrame({"date": ts[1:], "ret": rets})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return (
        df.groupby(["year", "month"])["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
        .unstack()
    )


def _yearly_returns_df(ts, rets) -> pd.DataFrame:
    df = pd.DataFrame({"date": ts[1:], "ret": rets})
    df["year"] = df["date"].dt.year
    return (
        df.groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
        .rename(columns={"ret": "年收益"})
    )


def _monthly_drawdown_df(ts, dds) -> pd.DataFrame:
    df = pd.DataFrame({"date": ts, "dd": dds})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.groupby(["year", "month"])["dd"].min().unstack()


def _rolling_metrics(rets, vals, ts) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    W = 242
    r, v = np.array(rets), np.array(vals)
    rs, rm = [], []
    for i in range(W, len(r) + 1):
        w = r[:i][-W:]
        rs.append(np.mean(w) / np.std(w) * np.sqrt(W) if np.std(w) > 0 else 0)
    for i in range(W, len(v)):
        w = v[: i + 1][-W:]
        pk = np.maximum.accumulate(w)
        rm.append(float(np.min((w - pk) / pk)))
    mn = min(len(ts[W:]), len(rs), len(rm))
    return ((ts[W : W + mn], rs[:mn]), (ts[W : W + mn], rm[:mn]))


def _rolling_returns(rets, ts) -> tuple[Any, ...]:
    W = 242
    r = np.array(rets)
    rl = []
    for i in range(W, len(r) + 1):
        w = r[:i][-W:]
        rl.append((1 + w).prod() ** (W / len(w)) - 1)
    mn = min(len(ts[W:]), len(rl))
    return (ts[W : mn + W], rl[:mn])


def _exposure_series(vals, cash) -> np.ndarray:
    v, c = np.array(vals), np.array(cash)
    return (v - c) / np.where(v > 0, v, 1)


def _trade_pairs_df(trade_log) -> pd.DataFrame:
    from app.engine.backtest.analysis._fifo import fifo_match, pairs_to_df

    return pairs_to_df(fifo_match(trade_log))


def _erfinv(x) -> float:
    if abs(x) >= 1:
        return float("inf") if x > 0 else float("-inf")
    a = 0.147
    t = np.log(1 - x * x) / 2 + 2 / (np.pi * a)
    return np.sign(x) * np.sqrt(np.sqrt(t * t - np.log(1 - x * x) / a) - t)
