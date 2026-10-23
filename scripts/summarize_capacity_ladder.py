"""汇总容量阶梯（同一策略在不同资金规模下的真回测）结果。

每个阶梯批次是 `backtest_drawdown_rounds.py --initial-cash <规模> --out-dir <目录>` 的产物，
本脚本按初始资金排序，输出年化/Sharpe/回撤/换手/平均暴露对照表，并写入 JSON。

用法::

    uv run python scripts/summarize_capacity_ladder.py data/results/capacity_v10_20m \
        data/results/capacity_v10_100m data/results/capacity_v10_300m data/results/capacity_v10_1b
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_FILE = PROJECT_ROOT / "data" / "results" / "capacity_ladder.json"


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _percent(value: object) -> float:
    """把 '23.24%' / 23.24 归一化为百分数。"""
    text = str(value).strip().rstrip("%")
    return _num(text)


def find_run(target: Path) -> Path | None:
    """返回运行目录：传入运行目录原样返回，传入批次目录取最新一次运行。"""
    if (target / "metadata.json").exists():
        return target
    runs = sorted(p.parent for p in target.rglob("metadata.json"))
    return runs[-1] if runs else None


def average_exposure(run_dir: Path) -> float:
    """平均暴露 = mean(持仓市值 / 总资产)。"""
    frame = pd.read_csv(
        run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"]
    )
    ratios = []
    for row in frame.itertuples(index=False):
        total = _num(getattr(row, "total_value", 0))
        positions = _num(getattr(row, "positions_value", 0))
        if total > 0:
            ratios.append(positions / total)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _read_json(path: Path) -> dict:
    """读取 JSON 对象（不可读/非法时返回空 dict）。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  警告：{path} 读取失败（{exc}）")
        return {}
    return payload if isinstance(payload, dict) else {}


def describe(run_dir: Path) -> dict:
    """读取一次运行的容量相关指标。"""
    metadata = _read_json(run_dir / "metadata.json")
    report = _read_json(run_dir / "report.json")
    income, risk = report.get("收益") or {}, report.get("风险") or {}
    adj, trading = report.get("风险调整") or {}, report.get("交易行为") or {}
    bench = report.get("基准对比") or {}
    return {
        "aum": _num(metadata.get("initial_cash")),
        "annual_return_pct": _percent(income.get("annual_return")),
        "sharpe": _num(adj.get("sharpe")),
        "max_drawdown_pct": _percent(risk.get("max_drawdown")),
        "calmar": _num(adj.get("calmar")),
        "turnover": str(trading.get("annual_turnover", "")),
        "avg_exposure_pct": average_exposure(run_dir) * 100,
        "alpha_pct": _percent(bench.get("alpha")),
        "run_dir": str(run_dir),
    }


def render(rows: list[dict]) -> str:
    """渲染阶梯表。"""
    lines = [
        "| 初始资金 | 年化 | Sharpe | 最大回撤 | Calmar | 平均暴露 | 年换手 | Alpha(风格) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['aum'] / 1e8:.2f} 亿 | {row['annual_return_pct']:.2f}% "
            f"| {row['sharpe']:.2f} | {row['max_drawdown_pct']:.2f}% "
            f"| {row['calmar']:.2f} | {row['avg_exposure_pct']:.1f}% "
            f"| {row['turnover']} | {row['alpha_pct']:.2f}% |"
        )
    return "\n".join(lines)


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="容量阶梯汇总")
    parser.add_argument("batches", nargs="+", help="各资金档位的结果批次目录")
    parser.add_argument("--out", default=str(OUT_FILE), help="JSON 输出路径")
    args = parser.parse_args()

    rows = []
    for raw in args.batches:
        batch_dir = Path(raw)
        run_dir = find_run(batch_dir)
        if run_dir is None:
            print(f"  跳过（无运行）: {batch_dir}")
            continue
        rows.append(describe(run_dir))
    rows.sort(key=lambda item: item["aum"])
    print(render(rows))
    Path(args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
