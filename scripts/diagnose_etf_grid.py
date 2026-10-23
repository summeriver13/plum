"""ETF 网格交易面板诊断 —— 先验证「区间收割」有没有 edge，再决定是否实现策略。

为什么先做面板诊断
    网格的收益公式很朴素（``往返次数 × 步长 × 每档资金 − 费用``），但「A 股 ETF
    到底有没有足够的往返」是经验问题。本脚本对全市场 ETF 逐个回放纸面网格，
    并把结果拆成「已实现收割 / 库存浮盈 / 停泊利息」后会回答两个问题：
    1. 网格在**同平均仓位**下能不能跑赢买入持有（即网格的择时价值）；
    2. 「品种的趋势强度」是不是决定网格成败的那把尺子（按趋势分位分档看 edge）。

口径：模型细节见 ``scripts/etf_grid_model.py``；样本 = 成交额达标且有足够历史的 ETF。

用法：
    uv run python scripts/diagnose_etf_grid.py                       # 1%/2%/3%
    uv run python scripts/diagnose_etf_grid.py --steps 0.005 0.01 --min-bars 1200
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLUM_DATA_MARKET_PATH", str(ROOT / "data" / "market" / "etf"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etf_explain_panels import (  # noqa: E402
    _int,
    _num,
    adjusted_close,
    classify_symbols,
    load_panels,
)
from scripts.etf_grid_model import (  # noqa: E402
    CASH_YIELD,
    FEE_RATE,
    GATE_BLOCK,
    GATE_EXIT,
    GATE_OFF,
    INIT_LOTS,
    MAX_LOTS,
    GridResult,
    analyse_symbol,
)

MARKET_ROOT = Path(os.environ["PLUM_DATA_MARKET_PATH"])  # load_panels 内部再拼 1d/交易所
OUT_DIR = ROOT / "data" / "results" / "etf_grid_diag"

DEFAULT_STEPS = (0.01, 0.02, 0.03)
START = datetime(2020, 1, 2)
END = datetime(2026, 9, 16)
MIN_AMOUNT = 1.0e7  # 20 日均成交额门槛：1000 万元
MIN_BARS = 1500  # 默认只留全窗口标的（约 6.2 年），避免短历史把中位数拉偏
TREND_BUCKETS = 5  # 趋势强度分档数

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Aggregate:
    """一组标的的汇总（中位数为主）。"""

    group: str
    n: int
    grid: float
    fair: float
    hold: float
    edge_fair: float
    realized: float
    win_rate: float
    grid_dd: float
    fair_dd: float
    position: float
    round_trips: int
    fees: float
    trend: float


def build_sample(panels: dict[str, pd.DataFrame], min_bars: int) -> pd.DataFrame:
    """过滤样本：区间内历史足够且 20 日均成交额达标的 ETF（后复权价宽表）。"""
    close = adjusted_close(panels)
    amount = panels["amount"]
    close = close.loc[(close.index >= START) & (close.index <= END)]
    keep = []
    for symbol in close.columns:
        series = close[symbol].dropna()
        if len(series) < min_bars:
            continue
        turnover = amount[symbol].dropna().tail(len(series)).mean()
        if np.isfinite(turnover) and _num(turnover) >= MIN_AMOUNT:
            keep.append(symbol)
    return close[keep]


def collect(sample: pd.DataFrame, steps: tuple[float, ...], gate: str = GATE_OFF) -> list[GridResult]:
    """逐标的 × 逐步长回放（可选趋势闸门）。"""
    results: list[GridResult] = []
    for index, symbol in enumerate(sample.columns, 1):
        prices = sample[symbol].dropna().to_numpy(dtype="float64")
        results.extend(
            analyse_symbol(symbol, prices, step, gate=gate) for step in steps
        )
        if index % 100 == 0:
            print(f"  已回放 {index}/{len(sample.columns)} 只", flush=True)
    return results


def _aggregate(group: str, frame: pd.DataFrame) -> Aggregate:
    """单组汇总。"""
    return Aggregate(
        group=group,
        n=len(frame),
        grid=frame["grid_return"].median(),
        fair=frame["fair_return"].median(),
        hold=frame["hold_return"].median(),
        edge_fair=(frame["grid_return"] - frame["fair_return"]).median(),
        realized=frame["realized"].median(),
        win_rate=_num((frame["grid_return"] > frame["fair_return"]).mean()),
        grid_dd=frame["grid_max_dd"].median(),
        fair_dd=frame["fair_max_dd"].median(),
        position=frame["avg_position"].median(),
        round_trips=_int(frame["round_trips"].median()),
        fees=frame["fees"].median(),
        trend=frame["trend_ratio"].median(),
    )


def summarize(
    frame: pd.DataFrame, categories: dict[str, str], *, by_trend: bool
) -> pd.DataFrame:
    """按步长 ×（类别 或 趋势强度分档）汇总。"""
    frame = frame.copy()
    if by_trend:
        frame["group"] = frame.groupby("step", observed=True)["trend_ratio"].transform(
            lambda values: pd.qcut(
                values.rank(method="first"), TREND_BUCKETS, labels=False
            )
        )
        frame["group"] = "趋势分位 " + (frame["group"] + 1).astype(int).astype(str)
    else:
        frame["group"] = frame["symbol"].map(categories).fillna("未知")
    rows = [
        {"step": step, **_aggregate(group, part).__dict__}
        for (step, group), part in frame.groupby(["step", "group"], observed=True)
    ]
    rows.extend(
        {"step": step, **_aggregate("全样本", part).__dict__}
        for step, part in frame.groupby("step", observed=True)
    )
    return pd.DataFrame(rows).sort_values(["step", "group"])


def format_table(summary: pd.DataFrame, title: str, *, by_trend: bool) -> str:
    """渲染 markdown 表。``网格−同仓位`` = 网格 − 公平基准（再平衡 + 停泊 + 计费）。"""
    unit = "趋势强度" if by_trend else "类别"
    lines = [
        f"### {title}",
        "",
        f"| 步长 | {unit} | 标的数 | 网格收益 | 买入持有 | 网格−同仓位 | 已实现收割 | "
        "胜率 | 网格回撤 | 同仓位回撤 | 平均仓位 | 往返 | 费用 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['step']:.1%} | {row['group']} | {_int(row['n'])} | "
            f"{row['grid']:+.1%} | {row['hold']:+.1%} | {row['edge_fair']:+.2%} | "
            f"{row['realized']:+.1%} | {row['win_rate']:.0%} | {row['grid_dd']:.1%} | "
            f"{row['fair_dd']:.1%} | {row['position']:.0%} | "
            f"{row['round_trips']:.0f} | {row['fees']:.2%} |"
        )
    return "\n".join(lines)


def main() -> None:
    """入口：加载面板 → 回放 → 汇总落盘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=float, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--min-bars", type=int, default=MIN_BARS)
    parser.add_argument("--cash-yield", type=float, default=CASH_YIELD)
    parser.add_argument(
        "--trend-gate",
        choices=(GATE_OFF, GATE_BLOCK, GATE_EXIT),
        default=GATE_OFF,
        help="趋势闸门：off 无 / block_buys 破位只卖不买 / exit 破位清仓",
    )
    args = parser.parse_args()

    print("读取 ETF 行情副本…", flush=True)
    panels = load_panels(MARKET_ROOT)
    sample = build_sample(panels, args.min_bars)
    categories = classify_symbols(adjusted_close(panels))
    print(
        f"样本 {len(sample.columns)} 只（成交额 ≥ {MIN_AMOUNT / 1e7:.0f}000 万、"
        f"≥ {args.min_bars} 个交易日、{sample.index[0].date()} ~ "
        f"{sample.index[-1].date()}｜趋势闸门 {args.trend_gate}）",
        flush=True,
    )

    results = collect(sample, tuple(args.steps), args.trend_gate)
    frame = pd.DataFrame([asdict(item) for item in results])
    by_category = summarize(frame, categories, by_trend=False)
    by_trend = summarize(frame, categories, by_trend=True)
    category_table = format_table(by_category, "按类别", by_trend=False)
    trend_table = format_table(
        by_trend, "按趋势强度分档（1 = 最震荡，5 = 最单边）", by_trend=True
    )
    print("\n" + category_table + "\n\n" + trend_table, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"summary_{stamp}.json").write_text(
        json.dumps(
            {
                "steps": list(args.steps),
                "cash_yield": args.cash_yield,
                "trend_gate": args.trend_gate,
                "min_bars": args.min_bars,
                "max_lots": MAX_LOTS,
                "init_lots": INIT_LOTS,
                "fee_rate": FEE_RATE,
                "sample": len(sample.columns),
                "range": [str(sample.index[0].date()), str(sample.index[-1].date())],
                "by_category": by_category.to_dict(orient="records"),
                "by_trend": by_trend.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / f"per_symbol_{stamp}.json").write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n已写出：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
