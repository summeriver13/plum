"""容量分析：把一次基准回测放大到不同资金规模，估计流动性与冲击成本上限。

做法：以已有回测的成交记录为基准，按资金放大倍数缩放出单规模，再对照标的当日成交额：

1. **参与率** p = 委托金额 / 当日成交额，随资金线性放大；
2. **成交上限**：单笔 ≤ 参与率上限（默认 10%）→ 超出部分需要拆单到后续交易日（或干脆买不满）；
3. **冲击成本**：平方根模型 r = k × sqrt(p)，单笔成本 ≈ 委托金额 × r；
4. **年化拖累** = Σ单笔成本 / 平均组合市值 / 年数。

输出各资金档位的容量指标表（JSON + 屏幕）。这是**上界估计**：忽略冲击自我放大、
忽略卖出端在下跌中成交额萎缩（实盘更差），也应与真回测（同资金重跑）对照。

用法::

    uv run python scripts/analyze_capacity.py --run-dir data/results/.../<run_id>
    uv run python scripts/analyze_capacity.py --run-dir <...> --aum 1e6 2e7 1e8 3e8 1e9
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

MARKET_DIR = PROJECT_ROOT / "data" / "market" / "1d"
OUT_FILE = PROJECT_ROOT / "data" / "results" / "capacity_analysis.json"
TRADING_DAYS = 242
PARTICIPATION_CAP = 0.10
IMPACT_COEFFICIENT = 0.025
DEFAULT_AUM = (1e6, 2e7, 1e8, 3e8, 1e9, 3e9)


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    """容错转 int。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _quantile(values: list[float], q: float) -> float:
    """线性插值分位数（values 需已升序排序）。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lower = max(_safe_int(math.floor(pos)), 0)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _day(value: object) -> pd.Timestamp | None:
    """把时间戳归一化为自然日（失败返回 None）。"""
    try:
        stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp.normalize()


@dataclass(frozen=True)
class TradeRow:
    """一笔成交及其成交日流动性。"""

    day: pd.Timestamp
    notional: float
    amount: float


def load_trades(run_dir: Path, cache: dict[str, pd.DataFrame]) -> list[TradeRow]:
    """读取成交记录并补齐当日成交额。"""
    path = run_dir / "data" / "trade_log.csv"
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    rows: list[TradeRow] = []
    missing = 0
    for row in frame.itertuples(index=False):
        day = _day(getattr(row, "timestamp", None))
        symbol = str(getattr(row, "symbol", ""))
        notional = _num(getattr(row, "amount", 0))
        if day is None or notional <= 0:
            continue
        amount = _daily_amount(symbol, day, cache)
        if amount <= 0:
            missing += 1
            continue
        rows.append(TradeRow(day=day, notional=notional, amount=amount))
    if missing:
        print(f"  提示：{missing} 笔成交缺少当日成交额，已跳过（不施加约束）")
    return rows


def _daily_amount(
    symbol: str, day: pd.Timestamp, cache: dict[str, pd.DataFrame]
) -> float:
    """symbol 在 day 的成交额（元）；无数据返回 0。"""
    if symbol not in cache:
        try:
            market, code = symbol.split(".", 1)
        except ValueError:
            return 0.0
        path = MARKET_DIR / market / f"{code}.parquet"
        if not path.exists():
            cache[symbol] = pd.DataFrame()
        else:
            frame = pd.read_parquet(path, columns=["amount", "volume", "close"])
            stamps = [_day(item) for item in pd.to_datetime(frame.index)]
            frame.index = pd.DatetimeIndex(
                [item for item in stamps if item is not None]
            )
            cache[symbol] = frame.sort_index()
    frame = cache[symbol]
    if frame.empty or day not in frame.index:
        return 0.0
    row = frame.loc[day]
    amount = _num(row.get("amount", 0))
    if amount > 0:
        return amount
    return _num(row.get("volume", 0)) * _num(row.get("close", 0))


@dataclass(frozen=True)
class CapacityPoint:
    """一个资金档位的容量指标。"""

    aum: float
    median_participation: float
    p90_participation: float
    capped_share: float
    median_split_days: float
    annual_cost_pct: float
    annual_return_pct: float | None


def evaluate(
    rows: list[TradeRow], base_aum: float, aum: float, avg_capital: float, years: float
) -> CapacityPoint:
    """计算给定资金规模下的参与率、拆单需求与成本拖累。"""
    scale = aum / base_aum if base_aum > 0 else 1.0
    parts: list[float] = []
    splits: list[float] = []
    cost_ratio = 0.0
    capped = 0
    for row in rows:
        participation = row.notional * scale / row.amount
        parts.append(participation)
        capped_p = min(participation, PARTICIPATION_CAP)
        # 冲击成本按「基准资金口径的成交额 × 冲击率」累加：
        # 金额与资金同步放大，相除后只剩冲击率的变化（r = k·sqrt(p)）。
        impact = row.notional * IMPACT_COEFFICIENT * math.sqrt(max(capped_p, 0.0))
        cost_ratio += impact / avg_capital if avg_capital > 0 else 0.0
        if participation > PARTICIPATION_CAP:
            capped += 1
            splits.append(participation / PARTICIPATION_CAP)
    series = sorted(parts)
    drag = cost_ratio / years if years > 0 else 0.0
    return CapacityPoint(
        aum=aum,
        median_participation=_quantile(series, 0.5),
        p90_participation=_quantile(series, 0.9),
        capped_share=capped / len(parts) if parts else 0.0,
        median_split_days=_quantile(sorted(splits), 0.5) if splits else 1.0,
        annual_cost_pct=drag * 100,
        annual_return_pct=None,
    )


def average_capital(run_dir: Path) -> tuple[float, float]:
    """返回（平均组合市值，年数）。"""
    frame = pd.read_csv(
        run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"]
    )
    values = [_num(item) for item in frame["total_value"].tolist()]
    positive = [item for item in values if item > 0]
    if not positive:
        return 0.0, 0.0
    return sum(positive) / len(positive), len(positive) / TRADING_DAYS


def render(
    points: Iterable[CapacityPoint], base_return: float | None, base_drag: float
) -> str:
    """渲染容量表。

    估算年化 = 基准年化 − **相对基准资金的额外冲击成本**（基准资金自身的冲击已在基准收益里）。
    """
    lines = [
        "| 资金规模 | 中位参与率 | p90 参与率 | 超上限单子占比 | 中位拆单天数 | 冲击成本年化 | 估算年化收益 |",
        "|---|---|---|---|---|---|---|",
    ]
    for point in points:
        incremental = point.annual_cost_pct - base_drag
        estimate = (
            f"{base_return - incremental:.2f}%" if base_return is not None else "—"
        )
        lines.append(
            f"| {point.aum / 1e8:.2f} 亿 | {point.median_participation:.3%} "
            f"| {point.p90_participation:.3%} | {point.capped_share:.1%} "
            f"| {point.median_split_days:.2f} | {point.annual_cost_pct:.2f}% | {estimate} |"
        )
    return "\n".join(lines)


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="流动性容量分析")
    parser.add_argument("--run-dir", required=True, help="基准回测目录（100 万口径）")
    parser.add_argument(
        "--aum",
        nargs="*",
        type=float,
        default=list(DEFAULT_AUM),
        help="待评估资金档位（元）",
    )
    parser.add_argument(
        "--base-aum", type=float, default=1e6, help="基准回测的初始资金"
    )
    parser.add_argument("--out", default=str(OUT_FILE), help="JSON 输出路径")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "data" / "trade_log.csv").exists():
        raise SystemExit(f"缺少 trade_log.csv: {run_dir}")

    cache: dict[str, pd.DataFrame] = {}
    rows = load_trades(run_dir, cache)
    avg_cap, years = average_capital(run_dir)
    base_return = _base_return(run_dir)
    print(f"基准运行: {run_dir}")
    print(
        f"成交笔数: {len(rows)}｜平均组合市值: {avg_cap / 1e4:,.0f} 万元｜年数: {years:.2f}"
    )
    print(
        f"标的当日成交额中位: {_quantile(sorted(r.amount for r in rows), 0.5) / 1e8:.2f} 亿元"
    )

    points = [evaluate(rows, args.base_aum, aum, avg_cap, years) for aum in args.aum]
    base_drag = evaluate(
        rows, args.base_aum, args.base_aum, avg_cap, years
    ).annual_cost_pct
    print("\n" + render(points, base_return, base_drag))
    print(
        "\n注：参与率上限（10%）使模型在大资金档位饱和，“拆单天数”才是真正的约束——"
        "超出上限的委托需拆到多个交易日（或买不满），上表年化未反映这部分未成交损失，"
        "因此 >1 亿档位应以下方的真回测结果为准。"
    )

    payload = {
        "run_dir": str(run_dir),
        "trade_count": len(rows),
        "average_capital": avg_cap,
        "years": years,
        "base_return_pct": base_return,
        "base_drag_pct": base_drag,
        "participation_cap": PARTICIPATION_CAP,
        "impact_coefficient": IMPACT_COEFFICIENT,
        "points": [point.__dict__ for point in points],
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {args.out}")
    return 0


def _base_return(run_dir: Path) -> float | None:
    """读取基准运行的年化收益（百分数）。"""
    path = run_dir / "report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    income = report.get("收益") or {}
    text = str(income.get("annual_return", "")).rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
