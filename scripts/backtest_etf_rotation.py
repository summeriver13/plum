"""ETF 双动量轮动 —— 批次扫描（v1 策略层 / v2 方向与窗口 / v3 反转短窗与长窗分散 / v4 标的池与门槛）。

标的池：全部 1,659 只沪深 ETF（含 2005 年以来退市/合并品种，避免幸存者偏差）；
v4 的 R32~R34 用「与 A 股宽基参照 ETF 的 60 日收益相关性 ≥ 0.6」分类受限池。
区间：2020-01-02 起（湖侧指数只覆盖 2020 年起，早于此无基准可算 alpha/IR；
与多因子小市值批次区间一致，便于横向比较）。
基准：国证 2000（IDX.399303，ETF 副本自带 IDX 桶）。
规则：ETF T+0 交收 + ETF 费率（免印花税/过户费）+ 按数据判定的 10%/20% 涨跌停。
避险：未通过绝对趋势过滤的名额留空仓，由引擎按 cash_yield 计提停泊收益。

用法:
    uv run python scripts/backtest_etf_rotation.py                 # v1 全部五轮
    uv run python scripts/backtest_etf_rotation.py --batch v2      # v2 方向与窗口三轮
    uv run python scripts/backtest_etf_rotation.py --batch v3      # v3 反转短窗与长窗分散三轮
    uv run python scripts/backtest_etf_rotation.py --batch v4      # v4 标的池与门槛五轮
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在导入 app.* 之前设定：settings 在导入期解析市场目录
os.environ.setdefault(
    "PLUM_DATA_MARKET_PATH", str(PROJECT_ROOT / "data" / "market" / "etf")
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest_multifactor_rounds import (  # noqa: E402 — 复用指标抽取与既有成本口径
    IMPACT_COEFFICIENT,
    INITIAL_CASH,
    PARTICIPATION_LIMIT,
    RunSettings,
    extract,
)

from app.engine.backtest.runner import BacktestRunner  # noqa: E402
from app.engine.backtest.scenarios.commission import EtfCommissionModel  # noqa: E402
from app.engine.backtest.scenarios.impact import SqrtImpactModel  # noqa: E402
from app.strategy.native.etf_rotation import EtfRotationStrategy  # noqa: E402
from etf_rotation_rounds import BATCH_ROUNDS, EtfRound  # noqa: E402

START = datetime(2020, 1, 2)
END = datetime(2026, 9, 16)
BENCHMARK = "IDX.399303"  # 国证 2000
# 空仓停泊年化（小数口径，引擎按 yield/252 逐日计提）：1.5% —— 与多因子小市值策略同口径。
# 起点取 2020-01-02 的原因：湖侧指数只覆盖 2020 年起，早于此无基准可算 alpha/IR，
# 且与微盘策略批次区间一致，便于横向比较。
CASH_YIELD = 0.015


_WARMUP_UNTIL: datetime | None = None  # 由 main 按 --start 设置（策略预热截止日）


def build_strategy_class(spec: EtfRound) -> type:
    """把轮次配置固化成策略类（目标数量模式）。"""

    class _Round(EtfRotationStrategy):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                strategy_id=f"etf_round{spec.round_id}",
                select_num=spec.select_num,
                hold_days=spec.hold_days,
                momentum_window=spec.momentum_window,
                ma_window=spec.ma_window,
                require_trend=spec.require_trend,
                reversal=spec.reversal,
                min_amount=spec.min_amount,
                max_corr=spec.max_corr,
                universe=spec.universe,
                ref_min_corr=spec.ref_min_corr,
                warmup_until=_WARMUP_UNTIL,
                **kwargs,
            )

    _Round.__name__ = f"EtfRound{spec.round_id}Strategy"
    return _Round


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（NaN/None/异常值回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def read_exposure(run_dir: Path) -> float | None:
    """从净值曲线读出平均仓位（市值/总资产），用于判断是否真的满仓。"""
    path = run_dir / "data" / "equity_curve.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    total = cast("pd.Series", pd.to_numeric(frame["total_value"], errors="coerce"))
    positions = cast(
        "pd.Series", pd.to_numeric(frame["positions_value"], errors="coerce")
    )
    valid = (total > 0) & positions.notna()
    if not bool(valid.any()):
        return None
    ratio = cast("pd.Series", positions[valid] / total[valid])
    return _num(ratio.mean())


def _pool_label(spec: EtfRound) -> str:
    """标的池口径的简短描述（全池 / 白名单 / 相关性分类阈值）。"""
    if spec.universe:
        return f"白名单 {len(spec.universe)} 只（{spec.universe[0]} 等）"
    if spec.ref_min_corr > 0:
        return f"A 股股票型（与宽基参照 60 日相关性 ≥ {spec.ref_min_corr:.2f}）"
    return "全池"


def run_round(
    spec: EtfRound, settings: RunSettings, start: datetime, end: datetime
) -> dict:
    """执行单轮回测并返回结果。"""
    print(f"\n{'=' * 78}\n第 {spec.round_id} 轮：{spec.title}\n{'=' * 78}", flush=True)
    print(f"假设: {spec.hypothesis}", flush=True)
    print(
        f"配置: 持仓 {spec.select_num} 只 | 调仓 {spec.hold_days} 日 | "
        f"动量窗 {spec.momentum_window} 日 | 趋势窗 MA{spec.ma_window} | "
        f"绝对过滤 {'开' if spec.require_trend else '关'} | "
        f"流动性门槛 {spec.min_amount / 1e7:.0f}000 万 | "
        f"标的池 {_pool_label(spec)}",
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
        # 份额拆分/合并按参考价复权持仓数量（ETF 必须有：实测 5:1 拆分会造成 5 倍假亏损）
        share_adjustment=True,
    )
    run_dir = Path(runner.run())
    elapsed = time.perf_counter() - t0
    metrics = extract(run_dir / "report.json")
    metrics.update(
        round_id=spec.round_id,
        title=spec.title,
        hypothesis=spec.hypothesis,
        select_num=spec.select_num,
        hold_days=spec.hold_days,
        momentum_window=spec.momentum_window,
        ma_window=spec.ma_window,
        require_trend=spec.require_trend,
        min_amount=spec.min_amount,
        max_corr=spec.max_corr,
        universe=spec.universe,
        ref_min_corr=spec.ref_min_corr,
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
    header = f"{'轮':>3} {'年化':>8} {'回撤':>8} {'Sharpe':>7} {'Calmar':>7} {'超额':>8} {'IR':>6} {'仓位':>7} {'换手':>7}"
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
    parser = argparse.ArgumentParser(
        description="ETF 双动量轮动回测（v1 五轮 / v2 方向与窗口 / v3 反转短窗与长窗分散）"
    )
    parser.add_argument(
        "--rounds", nargs="*", type=int, help="只跑指定轮次（默认全部）"
    )
    parser.add_argument(
        "--batch",
        choices=tuple(BATCH_ROUNDS),
        default="v1",
        help="批次：v1 五轮 / v2 方向与窗口三轮 / v3 反转短窗与长窗分散三轮",
    )
    parser.add_argument("--start", default=START.date().isoformat(), help="回测起点")
    parser.add_argument("--end", default=END.date().isoformat(), help="回测终点")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="结果目录（默认 data/results/etf_rounds_<batch>）",
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
        else PROJECT_ROOT / "data" / "results" / f"etf_rounds_{args.batch}"
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
    specs_all = BATCH_ROUNDS[args.batch]
    wanted = {spec.round_id for spec in specs_all}
    if args.rounds:
        wanted &= set(args.rounds)
    specs = [spec for spec in specs_all if spec.round_id in wanted]
    print(
        f"ETF 轮动回测：{len(specs)} 轮 | 区间 {start.date()} ~ {end.date()}",
        flush=True,
    )
    print(f"市场副本: {os.environ['PLUM_DATA_MARKET_PATH']}", flush=True)
    print(
        f"停泊: 年化 {settings.cash_yield:.2%} × {settings.park_ratio:.0%} 闲置资金"
        + (f" / 标的 {settings.parking_symbol}" if settings.parking_symbol else ""),
        flush=True,
    )
    results = [run_round(spec, settings, start, end) for spec in specs]
    print_summary(results)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    out = settings.out_dir / f"rounds_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n汇总已写出: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
