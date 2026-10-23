"""降档交付审计 CLI：调仓日「目标暴露 vs 实际暴露」缺口 + 涨跌停/停牌拒单代价。

回答的问题：趋势降档（如 2024-01 型崩盘）在回测里到底有没有被执行？
- 若调仓日实际暴露高于目标，说明减仓没做出来（现金约束 / 拒单 / 选股不足）；
- 被「跌停拒单」「停牌撤单」卡住的持仓，其真实退出延迟到下一个调仓日，
  延迟期间的价格差就是回测无法体现的额外执行损失。

用法::

    uv run python scripts/audit_derisk_delivery.py --round-id 8
    uv run python scripts/audit_derisk_delivery.py --round-id 8 --window 2024-01-01 2024-02-08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from derisk_audit import (  # noqa: E402 — 依赖上面的 sys.path 注入
    delayed_exit_cost,
    episodes,
    exposure_for,
    find_run,
    load_specs,
    parse_date,
    parse_events,
    rebalance_gaps,
    to_float,
    window_stats,
)

DEFAULT_BATCH = PROJECT_ROOT / "data" / "results" / "drawdown_rounds_v10"
DEFAULT_LOGS = [
    PROJECT_ROOT / "data" / "results" / "_logs" / "drawdown_rounds_v10.log",
    PROJECT_ROOT / "data" / "results" / "_logs" / "drawdown_rounds_v10_refine.log",
    PROJECT_ROOT / "data" / "results" / "_logs" / "drawdown_rounds_v10_ma.log",
]
PCT = lambda value: f"{value:6.1%}"  # noqa: E731 — 表格格式化
MONEY = lambda value: f"{value:,.0f}"  # noqa: E731 — 表格格式化


def print_window_detail(
    gaps: pd.DataFrame, events: pd.DataFrame, costs: pd.DataFrame, start: str, end: str
) -> float:
    """打印指定窗口的逐调仓明细与拒单明细，返回窗口内延迟成本合计。"""
    lo, hi = parse_date(start), parse_date(end)
    sub = gaps.loc[(gaps.index >= lo) & (gaps.index <= hi)]
    print(f"\n=== 窗口明细 {lo.date()} → {hi.date()}（调仓 {len(sub)} 次）===")
    if len(sub):
        print(sub.to_string(float_format=PCT))
    ev = events.loc[(events["date"] >= lo) & (events["date"] <= hi)]
    if len(ev):
        print("\n拒单 / 撤单事件:")
        print(ev.sort_values("date").to_string(index=False))
    cost = costs.loc[(costs["date"] >= lo) & (costs["date"] <= hi)]
    if len(cost):
        print("\n延迟退出明细（成本 = 数量 × (实际成交价 − 意图退出日收盘价)）:")
        print(cost.to_string(index=False, float_format="{:,.3f}".format))
    total = to_float(cost["cost"].sum(), 0.0) if len(cost) else 0.0
    print(f"\n窗口延迟成本合计: {total:,.0f} 元")
    return total


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="降档交付审计")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--log", type=Path, nargs="*", default=None)
    parser.add_argument("--round-id", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--window", nargs=2, default=None, metavar=("START", "END"))
    args = parser.parse_args()

    run_dir = args.run_dir or find_run(args.batch, args.round_id)
    equity = pd.read_csv(
        run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"]
    )
    equity["expo"] = equity["positions_value"] / equity["total_value"]
    equity = equity.set_index("timestamp")
    trades = pd.read_csv(run_dir / "data" / "trade_log.csv", parse_dates=["timestamp"])

    exposure = exposure_for(load_specs(args.batch).get(args.round_id, {}), run_dir)
    gaps = rebalance_gaps(equity, trades, exposure)
    events = parse_events(args.log or DEFAULT_LOGS, args.round_id)
    costs = delayed_exit_cost(events, trades)

    print(f"运行目录: {run_dir}")
    print(f"调仓日 {len(gaps)} 个｜拒单事件 {len(events)} 起｜延迟退出 {len(costs)} 笔")
    print(
        "全期缺口（实际 − 目标）: 平均 %+.2f%%｜中位 %+.2f%%｜95 分位 %+.2f%%｜最大 %+.2f%%"
        % (
            gaps["gap"].mean() * 100,
            gaps["gap"].median() * 100,
            gaps["gap"].quantile(0.95) * 100,
            gaps["gap"].max() * 100,
        )
    )

    rows = [
        window_stats(w, gaps, costs, events) for w in episodes(equity["total_value"], 5)
    ]
    print("\n=== 最大回撤窗口的降档交付 ===")
    print(pd.DataFrame(rows).to_string(index=False))

    if args.window:
        print_window_detail(gaps, events, costs, args.window[0], args.window[1])

    out = args.batch / f"derisk_audit_round{args.round_id}.json"
    out.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "gap_summary": {
                    k: to_float(v) for k, v in gaps["gap"].describe().items()
                },
                "windows": rows,
                "delayed_cost_total": to_float(costs["cost"].sum(), 0.0)
                if len(costs)
                else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
