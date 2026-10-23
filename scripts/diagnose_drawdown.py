"""回撤诊断：定位最大回撤区间、拆解来源（beta/alpha/波动），并预览风控叠加效果。

用法::

    uv run python scripts/diagnose_drawdown.py --run-dir data/results/.../2026xxxx_xxxxxx
    uv run python scripts/diagnose_drawdown.py --run-dir ... --benchmark IDX.000852

说明：风控叠加（趋势/波动率目标）是「用日收益乘以暴露系数」的近似预览，
未考虑离散成交与交易成本，只用于排序候选方案；正式验证须走引擎回测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = PROJECT_ROOT / "data" / "market" / "1d"
TRADING_DAYS = 244  # 年化交易日
INDEX_NAMES = {
    "IDX.000852": "中证1000",
    "IDX.399303": "国证2000",
    "IDX.000905": "中证500",
}


def load_equity(run_dir: Path) -> pd.DataFrame:
    """读取权益曲线（timestamp 索引，含 total_value/positions_value）。"""
    eq = pd.read_csv(run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"])
    return eq.set_index("timestamp").sort_index()


def load_index_bars(symbol: str) -> pd.Series:
    """读取市场副本里的指数收盘价序列。"""
    code = symbol.replace("IDX.", "")
    path = MARKET_DIR / "IDX" / f"{code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"指数数据不存在: {path}")
    df = pd.read_parquet(path, columns=["close"])
    return df["close"].astype(float).sort_index()


def drawdown_episodes(nav: pd.Series, top_n: int = 5) -> list[dict[str, Any]]:
    """找出最深的 N 段回撤（含起止日、谷底、恢复日）。"""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    episodes: list[dict[str, Any]] = []
    in_dd = False
    start_idx = 0
    trough_idx = 0
    for i, (ts, value) in enumerate(dd.items()):
        if value < 0 and not in_dd:
            in_dd, start_idx, trough_idx = True, max(i - 1, 0), i
        elif in_dd and value < dd.iloc[trough_idx]:
            trough_idx = i
        elif in_dd and value >= 0:
            episodes.append(_episode(nav, dd, start_idx, trough_idx, i))
            in_dd = False
    if in_dd:
        episodes.append(_episode(nav, dd, start_idx, trough_idx, None))
    return sorted(episodes, key=lambda e: e["depth"])[:top_n]


def _episode(
    nav: pd.Series, dd: pd.Series, start: int, trough: int, recover: int | None
) -> dict[str, Any]:
    """构造单段回撤记录。"""
    peak_value = float(nav.iloc[start])
    trough_value = float(nav.iloc[trough])
    recover_date = "未恢复" if recover is None else str(dd.index[recover].date())
    return {
        "peak": str(dd.index[start].date()),
        "trough": str(dd.index[trough].date()),
        "recover": recover_date,
        "depth": float(dd.iloc[trough]),
        "days_to_trough": int(trough - start),
        "days_in_dd": int((recover if recover is not None else len(dd) - 1) - start),
        "recovery_ratio": float(nav.iloc[-1] / peak_value) if recover is None else 1.0,
        "peak_value": peak_value,
        "trough_value": trough_value,
    }


def beta_alpha(
    strategy_ret: pd.Series, bench_ret: pd.Series
) -> tuple[float, float, float]:
    """回归 r_s = alpha + beta × r_b，返回 (beta, 年化 alpha, R²)。"""
    df = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    if len(df) < 30:
        return float("nan"), float("nan"), float("nan")
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    resid = y - (beta * x + intercept)
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
    return float(beta), float(intercept * TRADING_DAYS), float(r2)


def apply_overlay(
    nav: pd.Series, exposure: pd.Series, cost_bps: float = 0.0
) -> pd.Series:
    """把暴露系数乘到日收益上（近似预览；cost_bps 为每次变动的成本，基点）。

    暴露信号滞后一日生效（收盘后才能知道 MA/波动），避免前视偏差。
    注意：本函数仅供排序候选方案，正式验证须走引擎回测。
    """
    ret = nav.pct_change().fillna(0.0)
    expo = exposure.reindex(nav.index).ffill().shift(1).fillna(1.0)
    turnover = expo.diff().abs().fillna(0.0)
    adj = expo * ret - turnover * cost_bps / 10000.0
    return (1.0 + adj).cumprod()


def trend_exposure(index_close: pd.Series, window: int, risk_off: float) -> pd.Series:
    """指数收盘价在 MA(window) 之上满仓，否则 risk_off。"""
    ma = index_close.rolling(window).mean()
    return (index_close > ma).astype(float).where(ma.notna(), 1.0) * (
        1.0 - risk_off
    ) + risk_off


def vol_target_exposure(
    index_close: pd.Series, target_vol: float, window: int = 20
) -> pd.Series:
    """按指数实现波动率缩放暴露：min(1, 目标波动 / 实现波动)。"""
    realized = index_close.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)
    ratio = (target_vol / realized).clip(upper=1.0)
    return ratio.where(realized.notna(), 1.0)


def resample_exposure(exposure: pd.Series, every: int = 5) -> pd.Series:
    """把日频暴露信号降采样到调仓频率：仅在每 every 个交易日更新一次。"""
    sampled = exposure.copy()
    keep = pd.Series(False, index=exposure.index)
    keep.iloc[::every] = True
    return sampled.where(keep).ffill()


def summarize(nav: pd.Series) -> dict[str, float]:
    """年化收益/波动/Sharpe/最大回撤/Calmar（nav 可为净值或累计因子）。"""
    ret = nav.pct_change().fillna(0.0)
    years = max(len(ret) / TRADING_DAYS, 1e-9)
    growth = float(nav.iloc[-1] / nav.iloc[0])
    annual = growth ** (1 / years) - 1 if growth > 0 else -1.0
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    peak = nav.cummax()
    max_dd = float((nav / peak - 1.0).min())
    return {
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": annual / vol if vol > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": annual / abs(max_dd) if max_dd < 0 else 0.0,
    }


def print_episodes(episodes: list[dict[str, Any]]) -> None:
    """打印回撤区间表。"""
    print("\n=== 最深回撤区间 ===")
    print(
        f"{'峰值日':>12} {'谷底日':>12} {'恢复日':>12} {'深度':>8} {'下跌天数':>8} {'区间天数':>8}"
    )
    for ep in episodes:
        print(
            f"{ep['peak']:>12} {ep['trough']:>12} {ep['recover']:>12}"
            f" {ep['depth']:>7.1%} {ep['days_to_trough']:>8} {ep['days_in_dd']:>8}"
        )


def print_previews(
    nav: pd.Series, index_close: pd.Series
) -> list[tuple[str, dict[str, float]]]:
    """打印各风控叠加的近似效果对比。"""
    base = summarize(nav)
    rows: list[tuple[str, dict[str, float]]] = [("基线（满仓）", base)]
    for level in (0.8, 0.7, 0.6, 0.5, 0.4):
        expo = pd.Series(level, index=nav.index)
        rows.append((f"固定仓位 {level:.0%}", summarize(apply_overlay(nav, expo, 0))))
    for window in (60, 120, 200):
        expo = resample_exposure(trend_exposure(index_close, window, 0.0))
        rows.append(
            (f"趋势 MA{window}（清仓）", summarize(apply_overlay(nav, expo, 40)))
        )
        expo_half = resample_exposure(trend_exposure(index_close, window, 0.5))
        rows.append(
            (f"趋势 MA{window}（半仓）", summarize(apply_overlay(nav, expo_half, 40)))
        )
    nav_ma = nav.rolling(60).mean()
    expo_nav = resample_exposure(
        (nav > nav_ma).astype(float).where(nav_ma.notna(), 1.0)
    )
    rows.append(("策略自身 MA60", summarize(apply_overlay(nav, expo_nav, 40))))
    vol20 = vol_target_exposure(index_close, 0.15, 20)
    rows.append(
        (
            "指数波动率目标 15%",
            summarize(apply_overlay(nav, resample_exposure(vol20), 40)),
        )
    )
    combined = resample_exposure(
        trend_exposure(index_close, 120, 0.5)
        * vol_target_exposure(index_close, 0.15, 20)
    )
    rows.append(("MA120半仓 × 波动15%", summarize(apply_overlay(nav, combined, 40))))
    print("\n=== 风控叠加近似预览（信号滞后一日、调仓频率采样、仅排序用）===")
    print(
        f"{'方案':<24} {'年化':>8} {'波动':>8} {'Sharpe':>7} {'最大回撤':>9} {'Calmar':>7}"
    )
    for name, s in rows:
        print(
            f"{name:<24} {s['annual_return']:>7.1%} {s['annual_vol']:>7.1%}"
            f" {s['sharpe']:>7.2f} {s['max_drawdown']:>8.1%} {s['calmar']:>7.2f}"
        )
    return rows


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="回撤诊断与风控预览")
    parser.add_argument("--run-dir", required=True, help="回测 run 目录")
    parser.add_argument(
        "--benchmark", default="IDX.000852", help="基准指数（默认中证1000）"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eq = load_equity(run_dir)
    nav = eq["total_value"]
    print(
        f"区间: {nav.index[0].date()} ~ {nav.index[-1].date()}（{len(nav)} 个交易日）"
    )
    print(
        f"总收益: {nav.iloc[-1] / nav.iloc[0] - 1:.1%} | 年化仓位: "
        f"{(eq['positions_value'] / eq['total_value']).mean():.1%}"
    )

    episodes = drawdown_episodes(nav)
    print_episodes(episodes)

    index_close = load_index_bars(args.benchmark)
    bench_ret = index_close.pct_change()
    beta, alpha, r2 = beta_alpha(nav.pct_change(), bench_ret)
    print(f"\n=== 基准 {INDEX_NAMES.get(args.benchmark, args.benchmark)} 回归 ===")
    print(f"Beta={beta:.2f} | 年化Alpha={alpha:.1%} | R²={r2:.2f}")
    for ep in episodes[:3]:
        seg = bench_ret.loc[ep["peak"] : ep["trough"]]
        seg_bench = float((1 + seg).prod() - 1) if len(seg) else 0.0
        print(
            f"  {ep['peak']} ~ {ep['trough']}: 策略 {ep['depth']:.1%} | 基准 {seg_bench:.1%}"
        )

    rows = print_previews(nav, index_close)

    report = {
        "run_dir": str(run_dir),
        "benchmark": args.benchmark,
        "beta": beta,
        "alpha_annual": alpha,
        "r2": r2,
        "episodes": episodes,
        "previews": {name: stats for name, stats in rows},
    }
    out_path = run_dir / "drawdown_diagnosis.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n诊断结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
