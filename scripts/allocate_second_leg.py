"""第二收益源「闲置资金出资」情景检验：用真实策略净值曲线做加法检验。

R32 平均暴露仅 ~51%，即平均约 49% 的 NAV 常年停在零收益现金上。因此追加第二条腿
有两种截然不同的做法：

- **替代式（稀释）**：``(1-w)×策略 + w×腿``，把策略的仓位切一部分给腿；
- **出资式（加法）**：``策略 + w×腿``，用原本闲置的现金买腿，不减少策略仓位（w ≤ 闲置比例）。

判定标准不是"腿自己赚多少"，而是组合的年化/回撤/Calmar 是否同时改善。

**后处理近似的边界**：本脚本假定策略仓位规模不随 NAV 变化。引擎内真实实现会让
仓位随 NAV 等比例放大（二阶效应），结论落地前需用真实回测复核。

用法::

    uv run python scripts/allocate_second_leg.py --leg IDX.399006 --leg STYLE.QUALITY
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.engine.backtest.analysis.benchmarks import (  # noqa: E402
    benchmark_label,
    load_benchmark_prices,
)

TRADING_DAYS = 252
ANNUALIZATION = TRADING_DAYS**0.5
NOT_A_NUMBER: float = np.nan
WEIGHTS = (0.1, 0.2, 0.3, 0.4)
PARK_RATES = (0.010, 0.015, 0.020, 0.025)  # 逆回购/货基情景年化（湖里无 repo 利率序列）
DEFAULT_RUN_DIR = Path(
    "data/results/drawdown_rounds_v10/plum/drawdown_round32strategy/20260916_153935"
)
EQUITY_FILE = "data/equity_curve.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "cache" / "second_leg_allocation.json"
IDLE_COLUMNS = ("cash", "frozen_cash")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """容错转 float（numpy 标量/None 皆可）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric(values: Any) -> pd.Series:
    """容错转数值 Series（非数值 → NaN）。"""
    return cast(pd.Series, pd.to_numeric(values, errors="coerce"))


def _daily_index(values: Any) -> pd.DatetimeIndex:
    """把时间戳序列归一到日期索引（丢弃 16:00 UTC 时点部分）。"""
    stamps = cast(pd.Series, pd.to_datetime(pd.Series(list(values)), errors="coerce"))
    return pd.DatetimeIndex(stamps.dt.floor("D"))


def _metrics(returns: pd.Series) -> dict[str, float]:
    """年化收益/波动/Sharpe/最大回撤/Calmar。"""
    if returns.empty:
        return {}
    nav = (1.0 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    growth = _safe_float(nav.iloc[-1])
    annual = growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else 0.0
    vol = _safe_float(returns.std()) * ANNUALIZATION
    drawdown = _safe_float((nav / nav.cummax() - 1.0).min())
    return {
        "annual": annual,
        "vol": vol,
        "sharpe": annual / vol if vol > 0 else 0.0,
        "max_dd": drawdown,
        "calmar": annual / abs(drawdown) if drawdown < 0 else 0.0,
    }


def _load_equity(run_dir: Path) -> pd.DataFrame:
    """读取策略净值曲线（日期索引）。"""
    path = run_dir / EQUITY_FILE
    if not path.exists():
        raise FileNotFoundError(f"缺少净值曲线：{path}")
    frame = pd.read_csv(path)
    frame.index = _daily_index(frame["timestamp"])
    return frame.sort_index()


def _load_leg_returns(symbol: str) -> pd.Series:
    """加载候选腿的日收益序列。"""
    prices = load_benchmark_prices(symbol)
    if prices is None or prices.empty:
        raise ValueError(f"基准 {symbol} 无数据")
    series = prices.sort_index()
    closes = (
        pd.Series(series.to_numpy(dtype="float64"), index=_daily_index(series.index))
        .groupby(level=0)
        .last()
    )
    return closes.pct_change().dropna().rename(symbol)


def _idle_fraction(equity: pd.DataFrame) -> float:
    """平均闲置现金比例（cash + frozen_cash 占 NAV）。"""
    total = _numeric(cast(pd.Series, equity["total_value"]))
    idle = _numeric(cast(pd.Series, equity[IDLE_COLUMNS[0]]))
    for name in IDLE_COLUMNS[1:]:
        idle = idle + _numeric(cast(pd.Series, equity[name]))
    ratio = (idle / total).replace([np.inf, -np.inf], np.nan).dropna()
    return _safe_float(ratio.mean())


def _parking_leg(index: pd.Index, rate: float) -> pd.Series:
    """合成「逆回购停泊」腿：零波动、日利率 = 年化 ÷ 交易日数。"""
    return pd.Series(rate / TRADING_DAYS, index=index, name=f"CASH@{rate:.1%}")


def _scenario(
    strategy: pd.Series, leg: pd.Series, weight: float, funded: bool
) -> dict[str, Any]:
    """构造组合收益序列：funded=True 为出资式（加法），否则为替代式（稀释）。"""
    common = strategy.index.intersection(leg.index)
    base = strategy.loc[common]
    extra = leg.loc[common]
    mixed = base + weight * extra if funded else (1.0 - weight) * base + weight * extra
    return {
        "weight": weight,
        "mode": "出资式" if funded else "替代式",
        **_metrics(mixed),
    }


def _render(baseline: dict[str, float], idle: float, results: dict[str, Any]) -> None:
    """打印组合情景对比表。"""
    print(
        f"\n策略基线：年化 {baseline['annual']:.2%}｜Sharpe {baseline['sharpe']:.2f}｜"
        f"回撤 {baseline['max_dd']:.2%}｜Calmar {baseline['calmar']:.2f}"
    )
    print(f"平均闲置现金比例：{idle:.1%}（出资式上限）")
    for symbol, item in results.items():
        print(f"\n=== 追加 {symbol}（{item['label']}） ===")
        for mode in ("替代式", "出资式"):
            rows = [row for row in item["scenarios"] if row["mode"] == mode]
            if not rows:
                continue
            print(f"  {mode}：")
            for row in rows:
                flag = "✓" if row["calmar"] > baseline["calmar"] else " "
                print(
                    f"    权重 {row['weight']:>4.0%}：年化 {row['annual']:.2%}｜"
                    f"Sharpe {row['sharpe']:.2f}｜回撤 {row['max_dd']:.2%}｜"
                    f"Calmar {row['calmar']:.2f} {flag}"
                )


def main() -> int:
    """命令行入口：检验候选腿在两种出资方式下的组合效果。"""
    parser = argparse.ArgumentParser(description="第二收益源出资情景检验")
    parser.add_argument(
        "--run-dir", default=str(DEFAULT_RUN_DIR), help="策略回测结果目录"
    )
    parser.add_argument(
        "--leg", action="append", default=None, help="候选腿代码，可重复"
    )
    args = parser.parse_args()
    legs = tuple(args.leg) if args.leg else ("STYLE.QUALITY",)

    equity = _load_equity(Path(args.run_dir))
    total = _numeric(cast(pd.Series, equity["total_value"]))
    strategy = total.pct_change().dropna()
    baseline = _metrics(strategy)
    idle = _idle_fraction(equity)
    results: dict[str, Any] = {}
    for symbol in legs:
        leg = _load_leg_returns(symbol)
        scenarios: list[dict[str, Any]] = []
        for weight in WEIGHTS:
            if weight <= idle:
                scenarios.append(_scenario(strategy, leg, weight, funded=True))
            scenarios.append(_scenario(strategy, leg, weight, funded=False))
        results[symbol] = {
            "label": benchmark_label(symbol),
            "scenarios": scenarios,
        }
    for rate in PARK_RATES:
        leg = _parking_leg(strategy.index, rate)
        key = str(leg.name)
        scenarios = [
            _scenario(strategy, leg, weight, funded=True)
            for weight in WEIGHTS
            if weight <= idle
        ]
        results[key] = {
            "label": f"国债逆回购停泊 {rate:.1%}",
            "scenarios": scenarios,
        }
    _render(baseline, idle, results)
    payload = {
        "run_dir": str(args.run_dir),
        "baseline": baseline,
        "idle_fraction": idle,
        "legs": results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n结果已写入 {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
