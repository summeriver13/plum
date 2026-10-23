"""ETF 轮动可解释性报告：持仓与贡献、分类暴露、排序信号质量。

用法：
    uv run python scripts/explain_etf_rotation.py --run-dir <run_dir> \
        --momentum-window 250 --select-num 5 --hold-days 10

一次回答三个问题：① 赚的钱来自哪些标的与哪类资产；② 250 日动量排序有无预测力
（Spearman IC、top-N vs 池子等权 = 选择效应）；③ 摩擦与停泊各占多少（对账残差）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.etf_explain_attribution import Attribution, quantify_attribution  # noqa: E402
from scripts.etf_explain_panels import (  # noqa: E402
    adjusted_close,
    classify_symbols,
    load_panels,
)
from scripts.etf_explain_signal import (  # noqa: E402
    SignalConfig,
    evaluate_signal,
    rebalance_dates,
    summarize_signal,
)

MARKET_ROOT = PROJECT_ROOT / "data" / "market" / "etf"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "etf_explain"
RESIDUAL_NOISE_YUAN = 1000.0
"""对账残差超过该值（元）才提示疑似缺陷：份额取整量级实测 <100 元。"""

NAME_MAP = {
    "SH.510300": "沪深300",
    "SH.510500": "中证500",
    "SH.512100": "中证1000",
    "SH.518880": "黄金",
    "SH.511010": "国债",
    "SH.513100": "纳指",
    "SH.513500": "标普500",
    "SH.512480": "半导体",
    "SH.512690": "酒",
    "SH.512880": "证券",
    "SH.512010": "医药",
    "SH.515030": "新能车",
    "SH.512660": "军工",
    "SH.512800": "银行",
    "SH.512760": "芯片",
    "SZ.159985": "豆粕",
    "SZ.159920": "恒生",
}


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（None/NaN/异常值回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _label(symbol: str) -> str:
    """标的展示名（未收录的用代码本身）。"""
    return f"{symbol}({NAME_MAP[symbol]})" if symbol in NAME_MAP else symbol


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="ETF 轮动可解释性报告")
    parser.add_argument("--run-dir", required=True, help="回测运行目录（含 data/）")
    parser.add_argument(
        "--market-root", default=str(MARKET_ROOT), help="ETF 行情副本根目录"
    )
    parser.add_argument("--momentum-window", type=int, default=250)
    parser.add_argument("--ma-window", type=int, default=200)
    parser.add_argument(
        "--require-trend", action="store_true", help="排序前加 MA 趋势过滤"
    )
    parser.add_argument("--select-num", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--min-amount", type=float, default=1.0e7)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def _column(frame: pd.DataFrame, name: str) -> Any:
    """取数值列（缺失时返回全 0 数组，兼容停泊升级前的旧批次）。"""
    if name in frame.columns:
        return frame[name].to_numpy(dtype=float)
    return np.zeros(len(frame), dtype=float)


def print_overview(equity: pd.DataFrame, trades: pd.DataFrame) -> None:
    """概况：区间、参与度、停泊占比、交易笔数。"""
    stamps = equity.index.to_series()
    total = _column(equity, "total_value")
    positions = _column(equity, "positions_value")
    cash = _column(equity, "cash")
    parking = _column(equity, "parking_value")
    years = len(equity) / 242
    print("\n== 1. 概况 ==")
    print(
        f"  区间 {stamps.iloc[0]:%Y-%m-%d} ~ {stamps.iloc[-1]:%Y-%m-%d}（{len(equity)} 个交易日）"
    )
    print(
        f"  期末净值 {total[-1] / total[0] - 1:+.2%}｜年均 {(total[-1] / total[0]) ** (1 / years) - 1:+.2%}"
    )
    print(
        f"  平均持仓占比 {positions.mean() / total.mean():.1%}｜平均现金 {cash.mean() / total.mean():.1%}"
        f"｜平均停泊 {parking.mean() / total.mean():.1%}"
    )
    print(f"  成交 {len(trades)} 笔，涉及 {trades['symbol'].nunique()} 只标的")


def print_holdings(attribution: Attribution, top: int = 12) -> None:
    """逐标的贡献（毛贡献 − 摩擦 = 净贡献）与分类暴露。"""
    frame = attribution.by_symbol
    if frame.empty:
        print("\n== 2. 持仓与贡献 ==（无成交）")
        return
    print(
        f"\n== 2. 逐标的贡献（净合计 {attribution.net_contribution:,.0f} 元"
        f" / 权益变化 {attribution.realized_equity_change:,.0f} 元）=="
    )
    header = (
        f"  {'标的':<22}{'分类':<12}{'毛贡献(元)':>12}{'摩擦(元)':>10}"
        f"{'净贡献(元)':>12}{'占比':>8}{'持有天数':>8}{'建仓次数':>8}"
    )
    print(header)
    for row in frame.head(top).itertuples():
        print(_holding_line(row))
    if len(frame) > top:
        print("  —— 亏损侧 ——")
        for row in frame.tail(5).itertuples():
            print(_holding_line(row))

    print("\n== 3. 分类暴露（按持仓市值的年度均值占比）==")
    print_exposure(attribution.exposure)


def _holding_line(row: Any) -> str:
    """单只标的的贡献行。"""
    return (
        f"  {_label(str(row.symbol)):<22}{row.category:<12}"
        f"{row.pnl:>12,.0f}{row.fees:>10,.0f}{row.net:>12,.0f}"
        f"{row.net_share:>8.1%}{row.days_held:>8}{row.entries:>8}"
    )


def print_exposure(exposure: pd.DataFrame) -> None:
    """分类暴露的年度均值占比（按持仓市值）。"""
    if exposure.empty or len(exposure.columns) == 0:
        print("  （无持仓市值序列）")
        return
    # 年份键必须与 exposure 同索引对齐，否则 groupby 按索引对齐后全为 NaN、分组清空
    years = exposure.index.to_series().dt.year
    yearly = cast("pd.DataFrame", exposure.groupby(years).mean())
    weights = yearly.div(yearly.sum(axis=1), axis=0)
    for year, row in weights.iterrows():
        cells = " | ".join(
            f"{name} {value:.0%}" for name, value in row.items() if value > 0.001
        )
        print(f"  {year}: {cells}")


def print_signal(
    frame: pd.DataFrame, summary: dict[str, float], config: SignalConfig
) -> None:
    """排序信号质量与年度稳定性。"""
    print("\n== 4. 排序信号质量（250 日动量 → 下期收益）==")
    if frame.empty or not summary:
        print("  （样本不足）")
        return
    print(
        f"  调仓次数 {_num(summary['rounds']):.0f}｜平均候选池 {summary['candidates_mean']:.0f} 只"
        f"｜平均 IC {summary['ic_mean']:+.3f}（t={summary['ic_t']:.2f}，正比例 {summary['ic_positive_share']:.0%}）"
    )
    print(
        f"  选择效应（top{config.select_num} - 池子等权）均值 {summary['spread_mean']:+.2%}"
        f"｜为正比例 {summary['spread_positive_share']:.0%}"
    )
    print(
        f"  纸面年化：top{config.select_num} {summary['top_annual']:+.2%}"
        f"｜池子等权 {summary['pool_annual']:+.2%}"
        f"｜差 {summary['top_annual'] - summary['pool_annual']:+.2%}（无摩擦、不再平衡成本）"
    )

    frame = frame.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    yearly = frame.groupby("year").agg(
        rounds=("ic", "size"),
        ic=("ic", "mean"),
        spread=("spread", "mean"),
        top=("top_fwd", "mean"),
        pool=("pool_fwd", "mean"),
    )
    print("  分年：")
    for year, row in yearly.iterrows():
        print(
            f"    {year}: {_num(row['rounds']):>3.0f} 次｜IC {_num(row['ic']):+.3f}｜选择效应 {_num(row['spread']):+.2%}"
            f"｜top {_num(row['top']):+.2%}｜池子 {_num(row['pool']):+.2%}"
        )


def main() -> int:
    """入口：加载行情与结果，打印三维归因并落盘 JSON。"""
    args = parse_args()
    run_dir = Path(args.run_dir)
    equity = pd.read_csv(
        run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    index = pd.DatetimeIndex(equity.index)
    trades = pd.read_csv(run_dir / "data" / "trade_log.csv", parse_dates=["timestamp"])
    print(f"运行目录: {run_dir}")
    print_overview(equity, trades)

    print("\n读取 ETF 行情副本…", flush=True)
    panels = load_panels(Path(args.market_root))
    adj_close = adjusted_close(panels).reindex(equity.index).ffill()
    amount = panels["amount"].reindex(equity.index).ffill()
    classes = classify_symbols(adj_close)

    attribution = quantify_attribution(run_dir, panels, classes)
    print_holdings(attribution)
    print(
        f"\n  摩擦合计 {attribution.costs:,.0f} 元｜对账残差 {attribution.residual:,.0f} 元"
        f"（权益变化 {attribution.realized_equity_change:,.0f} 元）"
    )
    _print_residual_hint(attribution)

    config = SignalConfig(
        momentum_window=args.momentum_window,
        ma_window=args.ma_window,
        require_trend=args.require_trend,
        min_amount=args.min_amount,
        select_num=args.select_num,
        hold_days=args.hold_days,
    )
    dates = rebalance_dates(index, config.hold_days)
    frame = evaluate_signal(adj_close, amount, dates, config)
    summary = summarize_signal(frame, config)
    print_signal(frame, summary, config)

    yearly_rows: list[dict[str, Any]] = []
    if not frame.empty:
        with_year = frame.copy()
        with_year["year"] = pd.to_datetime(with_year["date"]).dt.year
        grouped = cast(
            "pd.DataFrame",
            with_year.groupby("year")[["ic", "spread", "top_fwd", "pool_fwd"]].mean(),
        )
        yearly_rows = list(grouped.reset_index().to_dict(orient="records"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "config": config.__dict__,
        "signal": summary,
        "costs": attribution.costs,
        "residual": attribution.residual,
        "equity_change": attribution.realized_equity_change,
        "by_symbol": attribution.by_symbol.to_dict(orient="records")
        if not attribution.by_symbol.empty
        else [],
        "yearly_signal": yearly_rows,
    }
    out = out_dir / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n汇总已写入: {out}")
    return 0


def _print_residual_hint(attribution: Attribution) -> None:
    """残差异常时提示疑似的引擎缺陷（纯取整量级实测 <100 元）。"""
    if abs(attribution.residual) < RESIDUAL_NOISE_YUAN:
        return
    print(
        f"    残差量级与「买入预估费 − 实际佣金」{attribution.buy_fee_gap:,.0f} 元一致时，"
        "该批次疑带 D1 缺陷（引擎旧版买入凭空入账）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
