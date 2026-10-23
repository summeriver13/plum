"""交易级 alpha 归因 CLI：对比不同频率版本的「换仓 alpha」。

用法::

    uv run python scripts/analyze_trade_alpha.py --rounds 1 15 --horizon 10

输出：各动作的前瞻收益均值/中位数，以及「买入组 − 卖出组」的差额（百分点）。
差额为负说明该频率的换仓是负 alpha 交易。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.trade_alpha import (  # noqa: E402
    HORIZONS,
    attach_forward_returns,
    buy_sell_spread,
    classify_trades,
    load_trades,
    summarize,
)

DEFAULT_BATCH = PROJECT_ROOT / "data" / "results" / "drawdown_rounds_v10"
ROUND_DIR_TEMPLATE = "drawdown_round{rid}strategy"


@dataclass(frozen=True)
class RoundInfo:
    """一轮回测的定位信息。"""

    round_id: int
    run_dir: Path


def find_run(batch_dir: Path, round_id: int) -> RoundInfo:
    """定位某轮最新一次运行的目录。"""
    parent = batch_dir / "plum" / ROUND_DIR_TEMPLATE.format(rid=round_id)
    runs = sorted(path for path in parent.glob("*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"第 {round_id} 轮没有运行目录: {parent}")
    return RoundInfo(round_id=round_id, run_dir=runs[-1])


def analyze(info: RoundInfo, horizons: tuple[int, ...]) -> dict[str, Any]:
    """单轮交易归因：分类 → 前瞻收益 → 汇总。"""
    trades = classify_trades(load_trades(info.run_dir))
    enriched = attach_forward_returns(trades, horizons=horizons)
    actions = sorted(set(enriched["action"].astype(str)))
    counts = {action: len(enriched[enriched["action"] == action]) for action in actions}
    return {
        "round_id": info.round_id,
        "run_dir": str(info.run_dir),
        "trades": len(enriched),
        "action_counts": counts,
        "summary": summarize(enriched, horizons=horizons),
        "spread": {
            f"{horizon}d": buy_sell_spread(enriched, horizon) for horizon in horizons
        },
    }


def render(results: list[dict[str, Any]], horizons: tuple[int, ...]) -> str:
    """渲染对比表。"""
    lines = ["# 交易级归因（前瞻收益，单位：百分点）", ""]
    for item in results:
        lines.append(f"## R{item['round_id']}（{item['trades']} 笔）")
        header = (
            "| 动作 | 笔数 | "
            + " | ".join(f"{h}日均值 | {h}日中位" for h in horizons)
            + " |"
        )
        lines.append(header)
        lines.append("| --- | --- |" + " --- | --- |" * len(horizons))
        for action, stats in item["summary"].items():
            cells = " | ".join(
                f"{stats[f'mean_{h}d']:+.2f} | {stats[f'median_{h}d']:+.2f}"
                for h in horizons
            )
            lines.append(f"| {action} | {stats['count']} | {cells} |")
        spread = item["spread"][f"{horizons[-1]}d"]
        lines.append(
            f"\n**换仓 alpha（{horizons[-1]} 日）**：买入 {spread['buy_mean']:+.2f}"
            f"｜卖出 {spread['sell_mean']:+.2f}｜差额 **{spread['spread']:+.2f}**"
            f"（买入 {spread['buy_n']} 笔 / 卖出 {spread['sell_n']} 笔）"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="交易级 alpha 归因")
    parser.add_argument("--batch", default=str(DEFAULT_BATCH))
    parser.add_argument("--rounds", nargs="*", type=int, default=[1, 15])
    parser.add_argument(
        "--horizon", type=int, default=10, help="汇总使用的期限（写入 JSON）"
    )
    parser.add_argument("--out", default=None, help="JSON 输出路径")
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    results = [analyze(find_run(batch_dir, rid), HORIZONS) for rid in args.rounds]
    print(render(results, HORIZONS))
    out = (
        Path(args.out) if args.out else batch_dir / f"trade_alpha_r{args.horizon}d.json"
    )
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
