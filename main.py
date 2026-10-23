"""Plum 量化交易系统 — 回测入口。

用法:
    uv run python main.py run --strategy <预设名或完整类路径>
    uv run python main.py --strategy <预设名或完整类路径>   # 兼容旧用法

预设名写在 app/config/strategies.toml；没有预设文件时（如公开仓），
--strategy 直接给完整类路径，例如 app.strategy.native.tsmom.TimeSeriesTrendStrategy。

数据准备已自带数据湖：见 scripts/build_plum_market_from_qdata.py。
"""

import argparse
import logging
import tomllib
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.config.logging import setup_logging
from app.config.settings import get_config

settings = get_config()
setup_logging(settings)
logger = logging.getLogger(__name__)

PRESETS_PATH = Path(__file__).resolve().parent / "app" / "config" / "strategies.toml"

CLASS_PATH_HINT = "app.strategy.native.tsmom.TimeSeriesTrendStrategy"


def _load_presets() -> dict[str, dict[str, Any]]:
    """读取策略预设；文件不存在时返回空表（公开仓不带预设）。"""
    if not PRESETS_PATH.is_file():
        return {}
    with PRESETS_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _normalize(entry: dict[str, Any]) -> dict[str, Any]:
    """把 TOML 里的预设补成 run() 需要的形状。"""
    start = entry.get("start", datetime(2020, 1, 2))
    return {
        "class": entry["class"],
        "benchmark": entry.get("benchmark"),
        "start": datetime(start.year, start.month, start.day),
        "kwargs": entry.get("kwargs", {}),
    }


def _import_class(path: str):
    """动态导入策略类。"""
    module_path, class_name = path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run(
    strategy_id: str = "fundamental_small_cap",
    initial_cash: Decimal = Decimal("1000000"),
    exclude_st: bool = True,
) -> str:
    """运行一次完整回测。

    Args:
        strategy_id: 预设名（app/config/strategies.toml）或完整类路径 module.Class
        initial_cash: 初始资金
        exclude_st: 是否过滤 ST

    Returns:
        结果目录路径
    """
    from app.engine.backtest.runner import BacktestRunner

    presets = _load_presets()
    if strategy_id in presets:
        entry = _normalize(presets[strategy_id])
        strategy_class = _import_class(entry["class"])
        kwargs = entry["kwargs"]
        benchmark = entry["benchmark"]
        start = entry["start"]
    elif "." in strategy_id:
        strategy_class = _import_class(strategy_id)
        kwargs = {}
        benchmark = None
        start = datetime(2020, 1, 2)
    else:
        available = "、".join(sorted(presets)) or (
            "（无预设，需给完整类路径，如 " + CLASS_PATH_HINT + "）"
        )
        raise ValueError(f"未知策略 {strategy_id!r}；可用预设：{available}")

    runner = BacktestRunner(
        strategy_class=strategy_class,
        strategy_kwargs=kwargs,
        benchmark_symbol=benchmark,
        start=start,
        end=datetime.now(),
        initial_cash=initial_cash,
        exclude_st=exclude_st,
    )
    return str(runner.run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plum 量化回测")
    sub = parser.add_subparsers(dest="command")

    # ---- run ----
    p_run = sub.add_parser("run", help="运行回测")
    p_run.add_argument(
        "--strategy", "-s", default="fundamental_small_cap", help="策略 ID"
    )
    p_run.add_argument("--cash", type=float, default=1_000_000, help="初始资金")
    p_run.add_argument("--no-exclude-st", action="store_true", help="不过滤 ST")

    # ---- 兼容旧用法 ----
    parser.add_argument(
        "--strategy", "-s", dest="strategy_legacy", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--cash", type=float, dest="cash_legacy", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no-exclude-st",
        dest="no_exclude_st_legacy",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    else:
        # 兼容旧用法: python main.py --strategy xxx
        _cmd_run(
            argparse.Namespace(
                strategy=args.strategy_legacy or "fundamental_small_cap",
                cash=args.cash_legacy or 1_000_000,
                no_exclude_st=args.no_exclude_st_legacy or False,
            )
        )


def _cmd_run(args) -> None:
    run_dir = run(
        strategy_id=args.strategy,
        initial_cash=Decimal(str(args.cash)),
        exclude_st=not args.no_exclude_st,
    )
    logger.info("完成。结果目录: %s", run_dir)
    logger.info("报告: %s/report.md", run_dir)


if __name__ == "__main__":
    main()
