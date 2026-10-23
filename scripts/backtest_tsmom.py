"""ETF 时序趋势跟随（TSMOM）回测 —— 首批 5 轮。

口径与 ETF 轮动批次一致（2020-01-02 起、基准国证 2000、ETF T+0 + ETF 费率、
未在场名额留现金按 cash_yield 停泊）。轮次：T1 双确认·12 只裁剪池 /
T2 双确认·20 只全池对照 / T3 仅动量 / T4 仅均线 / T5 无过滤常持（对照）。

用法：uv run python scripts/backtest_tsmom.py [--rounds 1 2]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在导入 app.* 之前设定：settings 在导入期解析市场目录
os.environ.setdefault(
    "PLUM_DATA_MARKET_PATH", str(PROJECT_ROOT / "data" / "market" / "etf")
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest_etf_rotation import read_exposure  # noqa: E402 — 复用仓位读取
from backtest_multifactor_rounds import (  # noqa: E402 — 复用指标抽取与成本口径
    IMPACT_COEFFICIENT,
    INITIAL_CASH,
    PARTICIPATION_LIMIT,
    RunSettings,
    extract,
)

from app.engine.backtest.runner import BacktestRunner  # noqa: E402
from app.engine.backtest.scenarios.commission import EtfCommissionModel  # noqa: E402
from app.engine.backtest.scenarios.impact import SqrtImpactModel  # noqa: E402
from app.strategy.native.tsmom import TimeSeriesTrendStrategy  # noqa: E402

START = datetime(2020, 1, 2)
END = datetime(2026, 9, 16)
BENCHMARK = "IDX.399303"  # 国证 2000
CASH_YIELD = 0.015  # 空仓停泊年化（与多因子小市值/ETF 轮动同口径）

CORE_BASKET: tuple[str, ...] = (
    "SH.510300",
    "SH.510500",
    "SH.512100",  # 宽基
    "SH.512480",
    "SH.512690",
    "SH.512880",
    "SH.512010",  # 行业
    "SH.515030",
    "SH.512660",
    "SH.512800",
    "SH.512760",
    "SH.518880",  # 黄金
)
EXTRA_BASKET: tuple[str, ...] = (
    "SH.511010",
    "SH.511220",  # 债
    "SH.513100",
    "SH.513500",
    "SZ.159920",
    "SH.513030",  # 跨境
    "SZ.159985",
    "SZ.159980",  # 商品
)
FULL_BASKET: tuple[str, ...] = CORE_BASKET + EXTRA_BASKET


@dataclass(frozen=True)
class TrendRound:
    """一轮 TSMOM 实验的配置。"""

    round_id: int
    title: str
    hypothesis: str
    basket: tuple[str, ...] = CORE_BASKET
    require_momentum: bool = True
    require_ma: bool = True
    hold_days: int = 21
    momentum_window: int = 252
    ma_window: int = 200


ROUNDS: list[TrendRound] = [
    TrendRound(
        round_id=1,
        title="双确认 · 12 只裁剪池",
        hypothesis=(
            "面板诊断显示宽基/行业/黄金有正区分度，双确认 + 等权，"
            "仓位随在场宽度自动升降"
        ),
    ),
    TrendRound(
        round_id=2,
        title="双确认 · 20 只全池对照",
        hypothesis="加回跨境/商品/债（诊断负区分度或无效），量化池裁剪贡献多少",
        basket=FULL_BASKET,
    ),
    TrendRound(
        round_id=3,
        title="仅 12 个月动量 · 12 只",
        hypothesis="去掉均线确认，检验 12 个月动量单条件是否已足够",
        require_ma=False,
    ),
    TrendRound(
        round_id=4,
        title="仅 200 日均线 · 12 只",
        hypothesis="去掉动量确认，检验 Faber 式 200 日均线单条件的表现",
        require_momentum=False,
    ),
    TrendRound(
        round_id=5,
        title="无过滤常持 · 12 只（对照）",
        hypothesis="等权买入持有的月度再平衡版本，分离选品收益与趋势择时收益",
        require_momentum=False,
        require_ma=False,
    ),
]

_WARMUP_UNTIL: datetime | None = None  # 由 main 按 --start 设置（策略预热截止日）


def build_strategy_class(spec: TrendRound) -> type:
    """把轮次配置固化成策略类。"""

    class _Round(TimeSeriesTrendStrategy):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                strategy_id=f"tsmom_round{spec.round_id}",
                basket=spec.basket,
                hold_days=spec.hold_days,
                momentum_window=spec.momentum_window,
                ma_window=spec.ma_window,
                require_momentum=spec.require_momentum,
                require_ma=spec.require_ma,
                warmup_until=_WARMUP_UNTIL,
                **kwargs,
            )

    _Round.__name__ = f"TsmomRound{spec.round_id}Strategy"
    return _Round


def run_round(
    spec: TrendRound, settings: RunSettings, start: datetime, end: datetime
) -> dict:
    """执行单轮回测并返回结果。"""
    print(f"\n{'=' * 78}\n第 {spec.round_id} 轮：{spec.title}\n{'=' * 78}", flush=True)
    print(f"假设: {spec.hypothesis}", flush=True)
    print(
        f"配置: basket {len(spec.basket)} 只 | 调仓 {spec.hold_days} 日 | "
        f"动量窗 {spec.momentum_window} 日{'（关）' if not spec.require_momentum else ''} | "
        f"MA{spec.ma_window}{'（关）' if not spec.require_ma else ''}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = BacktestRunner(
        strategy_class=build_strategy_class(spec),
        benchmark_symbol=BENCHMARK,
        start=start,
        end=end,
        initial_cash=settings.initial_cash,
        results_root=str(settings.out_dir),
        participation_limit=settings.participation_limit,
        impact=SqrtImpactModel(coefficient=settings.impact_coefficient),
        cash_yield=settings.cash_yield,
        park_ratio=settings.park_ratio,
        parking_symbol=settings.parking_symbol,
        settlement="t0",
        commission=EtfCommissionModel(),
        # 份额拆分/合并按参考价复权持仓数量（ETF 必须有）
        share_adjustment=True,
    )
    run_dir = Path(runner.run())
    elapsed = time.perf_counter() - t0
    metrics = extract(run_dir / "report.json")
    metrics.update(
        round_id=spec.round_id,
        title=spec.title,
        hypothesis=spec.hypothesis,
        basket_size=len(spec.basket),
        basket=list(spec.basket),
        require_momentum=spec.require_momentum,
        require_ma=spec.require_ma,
        hold_days=spec.hold_days,
        avg_exposure=read_exposure(run_dir),
        run_dir=str(run_dir),
        elapsed_sec=round(elapsed, 1),
    )
    print(
        f"完成 {elapsed / 60:.1f} 分钟 | 年化 {metrics['annual_return']} | "
        f"最大回撤 {metrics['max_drawdown']} | Sharpe {metrics['sharpe']} | "
        f"Calmar {metrics['calmar']} | 平均仓位 {metrics['avg_exposure']}",
        flush=True,
    )
    return metrics


def print_summary(results: list[dict]) -> None:
    """打印五轮对比表。"""
    print(f"\n{'=' * 78}\n汇总（基准 {BENCHMARK}）\n{'=' * 78}", flush=True)
    header = (
        f"{'轮':>3} {'年化':>8} {'回撤':>8} {'Sharpe':>7} {'Calmar':>7}"
        f" {'超额':>8} {'IR':>6} {'仓位':>7} {'换手':>7}"
    )
    print(header, flush=True)
    for item in results:
        exposure = item.get("avg_exposure")
        print(
            f"{item['round_id']:>3} {str(item.get('annual_return')):>8} "
            f"{str(item.get('max_drawdown')):>8} {str(item.get('sharpe')):>7} "
            f"{str(item.get('calmar')):>7} {str(item.get('excess_return')):>8} "
            f"{str(item.get('information_ratio')):>6} "
            f"{(f'{exposure:.1%}' if exposure is not None else '—'):>7} "
            f"{str(item.get('turnover')):>7}",
            flush=True,
        )
    print(
        f"\n基准年化 {results[0].get('benchmark_annual')} | 区间 {results[0].get('period')}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="ETF 时序趋势跟随（TSMOM）回测")
    parser.add_argument("--rounds", nargs="*", type=int, help="只跑指定轮次")
    parser.add_argument("--start", default=START.date().isoformat(), help="回测起点")
    parser.add_argument("--end", default=END.date().isoformat(), help="回测终点")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="结果目录（默认 data/results/tsmom_rounds）",
    )
    parser.add_argument(
        "--cash-yield", type=float, default=CASH_YIELD, help="小数口径，如 0.015"
    )
    parser.add_argument("--park-ratio", type=float, default=1.0)
    parser.add_argument(
        "--parking-symbol",
        default=None,
        help="停泊标的（真实持仓口径，如 SH.511010 国债 ETF；默认关闭）",
    )
    return parser.parse_args()


def main() -> int:
    """入口：按顺序跑轮次并落盘汇总。"""
    args = parse_args()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "data" / "results" / "tsmom_rounds"
    )
    settings = RunSettings(
        out_dir=out_dir,
        participation_limit=PARTICIPATION_LIMIT,
        impact_coefficient=IMPACT_COEFFICIENT,
        initial_cash=INITIAL_CASH,
        cash_yield=args.cash_yield,
        park_ratio=args.park_ratio,
        parking_symbol=args.parking_symbol,
    )
    global _WARMUP_UNTIL
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    _WARMUP_UNTIL = start
    wanted = {spec.round_id for spec in ROUNDS}
    if args.rounds:
        wanted &= set(args.rounds)
    specs = [spec for spec in ROUNDS if spec.round_id in wanted]
    print(
        f"TSMOM 回测：{len(specs)} 轮 | 区间 {start.date()} ~ {end.date()}",
        flush=True,
    )
    print(f"市场副本: {os.environ['PLUM_DATA_MARKET_PATH']}", flush=True)
    results = [run_round(spec, settings, start, end) for spec in specs]
    print_summary(results)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    out = settings.out_dir / f"rounds_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n汇总已写出: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
