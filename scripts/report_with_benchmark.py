"""用指定基准重算既有回测结果的报告（无需重跑回测）。

背景：报告里的 alpha / beta / 信息比率完全依赖基准指数。策略真实暴露是
「等权微盘」风格，用市值加权宽基（沪深300 / 中证1000）当基准会把风格收益
整体算成 alpha。本脚本原地重算既有结果目录的 report.json / report.md / charts，
并把批次汇总 JSON 里的基准字段同步刷新。

基准代码：
    STYLE.MICROCAP   plum 自建微盘等权风格指数（推荐，见 data/benchmarks/）
    IDX.399303       国证2000（市值加权，历史批次用过的对照基准）
    IDX.000852       中证1000

用法::

    uv run python scripts/report_with_benchmark.py --batch drawdown_rounds_v10
    uv run python scripts/report_with_benchmark.py --batch v9_1 --benchmark IDX.399303
    uv run python scripts/report_with_benchmark.py --run-dir data/results/<...>/<run_id>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest_multifactor_rounds import extract  # noqa: E402 — 复用既有指标抽取

from app.data.schema import CostBreakdown, TradeRecord  # noqa: E402
from app.engine.backtest.analysis.result_store import ResultStore  # noqa: E402
from app.engine.backtest.models import BacktestResult, EquityPoint  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
DEFAULT_BENCHMARK = "STYLE.MICROCAP"
BENCH_KEYS = (
    "benchmark_symbol",
    "benchmark_return",
    "benchmark_annual",
    "alpha",
    "beta",
    "tracking_error",
    "information_ratio",
)
METADATA_OWNED = (
    "run_id",
    "strategy",
    "start_date",
    "end_date",
    "initial_cash",
    "saved_at",
)


def _as_int(value: object, default: int = 0) -> int:
    """容错转 int（CSV 里的 float/NaN/None 均可）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _decimal(value: object, default: str = "0") -> Decimal:
    """容错转 Decimal（CSV 里的 float/NaN 均可）。"""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _ensure_datetime(value: object, fallback: datetime) -> datetime:
    """容错转 datetime。"""
    try:
        stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return fallback if pd.isna(stamp) else stamp.to_pydatetime()


@dataclass(frozen=True)
class RunSnapshot:
    """一次既有运行的落盘内容（CSV + metadata）。"""

    run_dir: Path
    run_id: str
    family: str
    strategy_name: str
    metadata: dict[str, Any]


def load_result(snapshot: RunSnapshot) -> BacktestResult:
    """把 equity_curve.csv / trade_log.csv 还原成 BacktestResult。"""
    data_dir = snapshot.run_dir / "data"
    equity = pd.read_csv(data_dir / "equity_curve.csv", parse_dates=["timestamp"])
    trades = pd.read_csv(data_dir / "trade_log.csv", parse_dates=["timestamp"])
    md = snapshot.metadata
    stamp = _ensure_datetime(md.get("start_date"), datetime(2020, 1, 2))
    result = BacktestResult(
        symbol=str(md.get("strategy", "")),
        start_date=stamp,
        end_date=_ensure_datetime(md.get("end_date"), stamp),
        initial_cash=_decimal(md.get("initial_cash", 1_000_000), "1000000"),
        trade_log=_trade_records(trades),
        equity_curve=_equity_points(equity),
        metadata=dict(md.get("engine") or {}),
    )
    return result


def _equity_points(frame: pd.DataFrame) -> list[EquityPoint]:
    """还原每日权益点。

    注意：equity_curve.csv 未落盘 positions_detail，这里用 positions_value > 0
    近似还原（只影响报告的「空仓比」）。
    """
    points: list[EquityPoint] = []
    for row in frame.itertuples(index=False):
        positions_value = _decimal(getattr(row, "positions_value", 0))
        detail = {"reconstructed": 1} if positions_value > 0 else {}
        points.append(
            EquityPoint(
                timestamp=_ensure_datetime(
                    getattr(row, "timestamp", None), datetime(2020, 1, 2)
                ),
                total_value=_decimal(getattr(row, "total_value", 0)),
                cash=_decimal(getattr(row, "cash", 0)),
                frozen_cash=_decimal(getattr(row, "frozen_cash", 0)),
                positions_value=positions_value,
                positions_detail=detail,
                cash_interest=_decimal(getattr(row, "cash_interest", 0)),
            )
        )
    return points


def _trade_records(frame: pd.DataFrame) -> list[TradeRecord]:
    """还原成交记录（成本明细逐项回填）。"""
    records: list[TradeRecord] = []
    for row in frame.itertuples(index=False):
        cost = CostBreakdown(
            commission=_decimal(getattr(row, "commission", 0)),
            transfer_fee=_decimal(getattr(row, "transfer_fee", 0)),
            stamp_duty=_decimal(getattr(row, "stamp_duty", 0)),
        )
        records.append(
            TradeRecord(
                parent_order_id="",
                slippage_cost=_decimal(getattr(row, "slippage_cost", 0)),
                timestamp=_ensure_datetime(
                    getattr(row, "timestamp", None), datetime(2020, 1, 2)
                ),
                symbol=str(getattr(row, "symbol", "")),
                quantity=_as_int(getattr(row, "quantity", 0)),
                price=_decimal(getattr(row, "price", 0)),
                cost=cost,
                is_buy=bool(getattr(row, "is_buy", False)),
            )
        )
    return records


def find_runs(batch_dir: Path) -> list[RunSnapshot]:
    """扫描批次目录下的所有运行。"""
    snapshots: list[RunSnapshot] = []
    for meta_path in sorted(batch_dir.rglob("metadata.json")):
        snapshot = _snapshot(batch_dir, meta_path)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _snapshot(batch_dir: Path, meta_path: Path) -> RunSnapshot | None:
    """把单个 metadata.json 读成运行快照（层级不符或不可读时返回 None）。"""
    run_dir = meta_path.parent
    rel = run_dir.relative_to(batch_dir)
    if len(rel.parts) < 3:
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  跳过（metadata 不可读）: {run_dir}（{exc}）")
        return None
    return RunSnapshot(
        run_dir=run_dir,
        run_id=rel.parts[-1],
        family=rel.parts[0],
        strategy_name=rel.parts[1],
        metadata=metadata,
    )


def refresh_explicit(run_dirs: Iterable[str], benchmark: str) -> None:
    """只刷新显式指定的运行目录（不推广到整个批次，不动汇总文件）。"""
    for raw in run_dirs:
        run_dir = Path(raw)
        if not run_dir.is_dir():
            raise SystemExit(f"运行目录不存在: {run_dir}")
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"缺少 metadata.json: {run_dir}")
        snapshot = _snapshot(run_dir.parents[2], meta_path)
        if snapshot is None:
            raise SystemExit(f"目录层级不符合 family/strategy/run_id: {run_dir}")
        try:
            refresh_run(snapshot, run_dir.parents[2], benchmark)
        except Exception as exc:  # noqa: BLE001 — 单个失败不中断
            print(f"  失败 {run_dir}：{type(exc).__name__}: {exc}", flush=True)


def refresh_run(snapshot: RunSnapshot, batch_dir: Path, benchmark: str) -> dict | None:
    """重算单次运行的报告，返回刷新后的基准指标。"""
    old = _read_bench(snapshot.run_dir)
    result = load_result(snapshot)
    meta = {
        key: value
        for key, value in snapshot.metadata.items()
        if key not in METADATA_OWNED and key != "engine"
    }
    from app.engine.backtest.analysis.report import comprehensive_report

    report = comprehensive_report(
        result, run_id=snapshot.run_id, benchmark_symbol=benchmark
    )
    ResultStore(root=batch_dir).save(
        result,
        run_id=snapshot.run_id,
        meta=meta,
        report=report,
        benchmark_symbol=benchmark,
        family=snapshot.family,
        strategy_name=snapshot.strategy_name,
    )
    new = extract(snapshot.run_dir / "report.json")
    _print_delta(snapshot, old, new)
    return new


def _read_bench(run_dir: Path) -> dict[str, Any]:
    """读取运行目录里已有的基准指标（用于打印前后对比）。"""
    path = run_dir / "report.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(report.get("基准对比") or {})


def _print_delta(
    snapshot: RunSnapshot, old: dict[str, Any], new: dict[str, Any]
) -> None:
    """打印单次运行的基准字段变化。"""
    head = f"  完成 {snapshot.strategy_name}/{snapshot.run_id}"
    fields = ("benchmark_return", "alpha", "beta", "information_ratio")
    parts = [f"{key}: {old.get(key, '—')} → {new.get(key, '—')}" for key in fields]
    print(f"{head}\n      " + " | ".join(parts), flush=True)


def update_summaries(batch_dir: Path, benchmark: str) -> int:
    """把批次内所有 rounds_summary_*.json 的基准字段刷新为当前基准。"""
    updated = 0
    for path in sorted(batch_dir.glob("rounds_summary_*.json")):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"  跳过汇总（不可读）: {path.name}（{exc}）")
            continue
        changed = False
        for item in items:
            report_path = Path(str(item.get("run_dir", ""))) / "report.json"
            if not report_path.exists():
                continue
            metrics = extract(report_path)
            for key in BENCH_KEYS:
                if metrics.get(key) is not None:
                    item[key] = metrics[key]
            changed = True
        if changed:
            path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            updated += 1
    return updated


def _iter_batches(names: Iterable[str]) -> list[Path]:
    """把批次名解析成目录。"""
    targets = []
    for name in names:
        candidate = Path(name)
        batch_dir = candidate if candidate.is_dir() else RESULTS_ROOT / name
        if not batch_dir.is_dir():
            raise SystemExit(f"批次目录不存在: {batch_dir}")
        targets.append(batch_dir)
    return targets


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="用指定基准重算既有回测报告")
    parser.add_argument(
        "--batch",
        nargs="*",
        default=[],
        help="结果批次目录（data/results 下的一级子目录）",
    )
    parser.add_argument("--run-dir", nargs="*", default=[], help="直接指定单次运行目录")
    parser.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK, help="基准代码（默认 STYLE.MICROCAP）"
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="不刷新 rounds_summary_*.json"
    )
    args = parser.parse_args()

    if not args.batch and not args.run_dir:
        parser.error("至少指定 --batch 或 --run-dir")

    print(f"基准: {args.benchmark}")
    refresh_explicit(args.run_dir, args.benchmark)
    for batch_dir in _iter_batches(args.batch):
        snapshots = find_runs(batch_dir)
        print(
            f"\n{'=' * 70}\n批次 {batch_dir.name}：{len(snapshots)} 次运行\n{'=' * 70}"
        )
        for snapshot in snapshots:
            try:
                refresh_run(snapshot, batch_dir, args.benchmark)
            except Exception as exc:  # noqa: BLE001 — 单次失败不中断整批
                print(
                    f"  失败 {snapshot.run_dir}：{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if not args.no_summary:
            count = update_summaries(batch_dir, args.benchmark)
            print(f"  汇总文件已刷新: {count} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
