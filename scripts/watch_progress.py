"""批次回测进度观察器。

用途：回测批次（v10 降回撤 34 轮、ETF 轮动、容量阶梯等）在后台顺序执行时，
实时看「已完成几轮、当前跑到哪、每轮指标、预计剩余时间」。

数据来源全部是磁盘落盘的事实，不依赖日志缓冲：

- 已完成轮次 = 批次目录下 `plum/<策略名>/<开始时刻>/` 的个数；
- 当前轮次与单轮耗时 = 由上一条的开始时刻推算（轮内不可见，粒度 ~1 轮）；
- 每轮指标 = 该轮 `report.json`（优先读「收益/风险/风险调整」三节）。

用法：
    uv run python scripts/watch_progress.py data/results/_verify_clean_v10_all --total 34
    uv run python scripts/watch_progress.py <批次目录> --once      # 只看一眼
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

STRATEGY_GLOB = "*/*"
TIME_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_EVERY_SECONDS = 20
BAR_WIDTH = 24
RECENT_ROWS = 5
SECTIONS = ("收益", "风险", "风险调整")
WANTED = {"annual_return": "年化", "max_drawdown": "回撤", "sharpe": "Sharpe"}


def _load_json(path: Path) -> dict[str, Any]:
    """安全读取 JSON（失败返回空字典）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _collect_metrics(node: Any, out: dict[str, str]) -> None:
    """在给定节点内寻找目标指标（缺失的键不覆盖已找到的值）。"""
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                key in WANTED
                and key not in out
                and isinstance(value, str | int | float)
            ):
                out[key] = str(value)
            else:
                _collect_metrics(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_metrics(item, out)


def read_metrics(run_dir: Path) -> dict[str, str]:
    """读取单轮指标：优先三节，结构变动时全量兜底搜索。"""
    report = _load_json(run_dir / "report.json")
    out: dict[str, str] = {}
    for section in SECTIONS:
        sub = report.get(section)
        if isinstance(sub, dict):
            _collect_metrics(sub, out)
    if not out:
        _collect_metrics(report, out)
    return out


def _parse_start(run_dir: Path) -> datetime | None:
    """从结果目录名解析该轮开始时刻（目录名即时间戳）。"""
    try:
        return datetime.strptime(run_dir.name, TIME_FORMAT)
    except ValueError:
        return None


def scan_runs(batch_dir: Path) -> list[tuple[datetime, Path]]:
    """扫描批次内已出现的轮次，按开始时刻升序。"""
    root = batch_dir / "plum"
    runs: list[tuple[datetime, Path]] = []
    for path in root.glob(STRATEGY_GLOB):
        if not path.is_dir():
            continue
        started = _parse_start(path)
        if started is not None:
            runs.append((started, path))
    return sorted(runs)


def round_label(run_dir: Path) -> str:
    """轮次标签：策略目录名去掉 drawdown_/strategy 前后缀。"""
    name = run_dir.parent.name
    return name.removeprefix("drawdown_").removesuffix("strategy")


def _minutes_per_round(runs: list[tuple[datetime, Path]]) -> float | None:
    """单轮耗时中位数（分钟）：由相邻轮次的开始时刻差推算。"""
    if len(runs) < 2:
        return None
    gaps = [
        (runs[i + 1][0] - runs[i][0]).total_seconds() / 60.0
        for i in range(len(runs) - 1)
    ]
    return statistics.median(gaps)


def _bar(done: int, total: int | None) -> str:
    """进度条（# 已完成，. 未完成）。"""
    if not total or total <= 0:
        return ""
    filled = min(BAR_WIDTH, round(BAR_WIDTH * done / total))
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def _eta(runs: list[tuple[datetime, Path]], total: int | None) -> str:
    """预计剩余时间。"""
    pace = _minutes_per_round(runs)
    if pace is None or not total:
        return "-"
    remaining = max(0, total - len(runs))
    finish = datetime.now() + timedelta(minutes=pace * remaining)
    return f"{pace:.1f} 分钟/轮 → 剩 {remaining} 轮，约 {finish:%H:%M} 完成"


def snapshot(batch_dir: Path, total: int | None) -> str:
    """生成一次快照文本。"""
    runs = scan_runs(batch_dir)
    lines = [
        f"{datetime.now():%H:%M:%S}  已完成 {len(runs)}"
        + (f"/{total}" if total else "")
        + f" 轮 {_bar(len(runs), total)}"
    ]
    if runs:
        current = runs[-1]
        lines.append(
            f"当前/最新: {round_label(current[1])}  起于 {current[0]:%H:%M:%S}"
        )
    lines.append(f"节奏: {_eta(runs, total)}")
    for _, run_dir in runs[-RECENT_ROWS:]:
        metrics = read_metrics(run_dir)
        row = "  ".join(
            f"{label} {metrics.get(key, '-')}" for key, label in WANTED.items()
        )
        lines.append(f"  {round_label(run_dir):<12} {row}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="批次回测进度观察器")
    parser.add_argument("batch_dir", type=Path, help="批次结果目录（含 plum/ 子目录）")
    parser.add_argument(
        "--total", type=int, default=None, help="该批次总轮数（用于进度条与 ETA）"
    )
    parser.add_argument(
        "--every", type=float, default=DEFAULT_EVERY_SECONDS, help="刷新秒数"
    )
    parser.add_argument("--once", action="store_true", help="只输出一次快照")
    return parser.parse_args()


def main() -> int:
    """入口：循环刷新或单次快照。"""
    args = parse_args()
    if not args.batch_dir.is_dir():
        print(f"批次目录不存在: {args.batch_dir}")
        return 1
    while True:
        print(snapshot(args.batch_dir, args.total), flush=True)
        if args.once:
            return 0
        time.sleep(max(1.0, args.every))
        print("-" * 60, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
