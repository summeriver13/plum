"""第二收益源判定：候选风格腿与微盘风格的相关性结构检验。

平均相关系数会骗人（可转债平时与微盘相关 0.3、2024-01 一起崩），因此判定以
**危机窗口联合分布**为主、平均相关为辅。判据：

1. 危机窗口相关系数 ≤ 0.2（最好是负的）
2. 独立 Sharpe ≥ 0.5
3. 可随时抽回（T+1 变现）

输出：全期/月度/下行相关、对微盘回归的 beta/alpha、微盘各回撤窗口两腿表现、
组合叠加效果（固定权重日再平衡）。JSON 落盘到 data/cache/second_leg_judgment.json。

用法::

    uv run python scripts/judge_second_leg.py
    uv run python scripts/judge_second_leg.py --leg STYLE.QUALITY --leg IDX.000300
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
NOT_A_NUMBER: float = np.nan  # 常量：格式化时的非数值哨兵
RISK_FREE = 0.0
CRISIS_DD_THRESHOLD = -0.10  # 微盘回撤超过该阈值视为一次危机
CRISIS_TOP_N = 5
BLEND_WEIGHTS = (0.2, 0.3, 0.4)
DOWNSIDE_QUANTILE = 0.1  # 用于下行相关的微盘最差交易日比例
BASELINE_SYMBOL = "STYLE.MICROCAP"
DEFAULT_LEGS = ("STYLE.QUALITY",)
OUTPUT_JSON = PROJECT_ROOT / "data" / "cache" / "second_leg_judgment.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """容错转 float（numpy 标量/None 皆可）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """容错转 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _daily_index(values: Any) -> pd.DatetimeIndex:
    """把时间戳序列归一到日期索引（丢弃 16:00 UTC 之类的时点部分）。"""
    stamps = cast(pd.Series, pd.to_datetime(pd.Series(list(values)), errors="coerce"))
    return pd.DatetimeIndex(stamps.dt.floor("D"))


def _load_returns(symbol: str) -> pd.Series:
    """加载基准净值并转成日收益序列（日期索引，升序）。"""
    prices = load_benchmark_prices(symbol)
    if prices is None or prices.empty:
        raise ValueError(f"基准 {symbol} 无数据（先运行 build_*_index.py --publish）")
    series = prices.sort_index()
    index = _daily_index(series.index)
    values = (
        pd.Series(series.to_numpy(dtype="float64"), index=index).groupby(level=0).last()
    )
    return values.pct_change().dropna().rename(symbol)


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
        "sharpe": (annual - RISK_FREE) / vol if vol > 0 else 0.0,
        "max_dd": drawdown,
        "calmar": annual / abs(drawdown) if drawdown < 0 else 0.0,
        "total": growth - 1.0,
    }


def _episode_record(
    nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    """构造一次回撤区间记录（峰→谷，谷在区间内取最小，不越界到后续行情）。"""
    history = nav.loc[:start]
    peak_day = cast(pd.Timestamp, history.idxmax())
    peak_value = _safe_float(history.max())
    window = nav.loc[peak_day:end]
    trough_day = cast(pd.Timestamp, window.idxmin())
    return {
        "peak": peak_day,
        "trough": trough_day,
        "days": _safe_int((trough_day - peak_day).days),
        "depth": _safe_float(window.min() / peak_value - 1.0),
    }


def _drawdown_episodes(nav: pd.Series) -> list[dict[str, Any]]:
    """找出微盘净值回撤超过阈值的独立区间，按深度排序取前 N（区间去重）。"""
    drawdown = nav / nav.cummax() - 1.0
    episodes: dict[tuple[str, str], dict[str, Any]] = {}
    start: pd.Timestamp | None = None
    for day, value in drawdown.items():
        moment = cast(pd.Timestamp, day)
        deep = _safe_float(value, 0.0) <= CRISIS_DD_THRESHOLD
        if deep and start is None:
            start = moment
        elif not deep and start is not None:
            record = _episode_record(nav, start, moment)
            episodes[(str(record["peak"]), str(record["trough"]))] = record
            start = None
    if start is not None:
        last = cast(pd.Timestamp, drawdown.index[-1])
        record = _episode_record(nav, start, last)
        episodes[(str(record["peak"]), str(record["trough"]))] = record
    ranked = sorted(episodes.values(), key=lambda item: _safe_float(item["depth"]))
    return ranked[:CRISIS_TOP_N]


def _window_stats(
    benchmark: pd.Series, leg: pd.Series, peak: pd.Timestamp, trough: pd.Timestamp
) -> dict[str, Any]:
    """危机窗口内两腿的累计收益与相关系数。"""
    window = benchmark.loc[peak:trough].index
    bench_ret = benchmark.reindex(window).dropna()
    leg_ret = leg.reindex(bench_ret.index).dropna()
    common = bench_ret.index.intersection(leg_ret.index)
    return {
        "peak": str(peak.date()),
        "trough": str(trough.date()),
        "corr": _safe_float(bench_ret.loc[common].corr(leg_ret.loc[common])),
        "benchmark": _safe_float((1.0 + bench_ret.loc[common]).prod() - 1.0),
        "leg": _safe_float((1.0 + leg_ret.loc[common]).prod() - 1.0),
    }


def _blend(benchmark: pd.Series, leg: pd.Series, weight: float) -> dict[str, float]:
    """按固定权重日再平衡的组合指标（不含再平衡成本）。"""
    common = benchmark.index.intersection(leg.index)
    mixed = weight * leg.loc[common] + (1.0 - weight) * benchmark.loc[common]
    return {"weight": weight, **_metrics(mixed)}


def _correlation_block(benchmark: pd.Series, leg: pd.Series) -> dict[str, Any]:
    """全期日/月相关与下行相关。"""
    common = benchmark.index.intersection(leg.index)
    bench_ret = benchmark.loc[common]
    leg_ret = leg.loc[common]
    cutoff = _safe_float(bench_ret.quantile(DOWNSIDE_QUANTILE))
    downside = bench_ret <= cutoff
    monthly_bench = (1.0 + bench_ret).resample("ME").prod() - 1.0
    monthly_leg = (1.0 + leg_ret).resample("ME").prod() - 1.0
    return {
        "daily": _safe_float(bench_ret.corr(leg_ret)),
        "monthly": _safe_float(monthly_bench.corr(monthly_leg)),
        "downside": _safe_float(bench_ret[downside].corr(leg_ret[downside])),
        "days": len(common),
    }


def _regression(benchmark: pd.Series, leg: pd.Series) -> dict[str, float]:
    """腿对微盘的 beta/alpha（年化）与 R²。"""
    common = benchmark.index.intersection(leg.index)
    x = benchmark.loc[common].to_numpy(dtype="float64")
    y = leg.loc[common].to_numpy(dtype="float64")
    beta = _safe_float(np.polyfit(x, y, 1)[0]) if x.size > 1 else 0.0
    alpha = _safe_float(y.mean() - beta * x.mean()) * TRADING_DAYS
    correlation = _safe_float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else 0.0
    return {"beta": beta, "alpha": alpha, "r2": correlation**2}


def _judge_leg(benchmark: pd.Series, leg: pd.Series, symbol: str) -> dict[str, Any]:
    """对单条候选腿给出完整判定结果。"""
    nav = (1.0 + benchmark).cumprod()
    episodes = _drawdown_episodes(nav)
    return {
        "symbol": symbol,
        "label": benchmark_label(symbol),
        "standalone": _metrics(leg),
        "correlation": _correlation_block(benchmark, leg),
        "regression": _regression(benchmark, leg),
        "crisis": [
            _window_stats(benchmark, leg, item["peak"], item["trough"])
            for item in episodes
        ],
        "blend": [_blend(benchmark, leg, weight) for weight in BLEND_WEIGHTS],
    }


def _fmt(value: Any, percent: bool = False) -> str:
    """格式化输出（百分比或两位小数）。"""
    number = _safe_float(value, NOT_A_NUMBER)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.2%}" if percent else f"{number:.2f}"


def _render(results: dict[str, Any], baseline: pd.Series) -> None:
    """打印判定报告。"""
    print(f"\n基准腿（微盘风格）：{_metrics(baseline)}")
    base = _metrics(baseline)
    print(
        f"  年化 {_fmt(base.get('annual'), True)}｜Sharpe {_fmt(base.get('sharpe'))}｜"
        f"回撤 {_fmt(base.get('max_dd'), True)}｜Calmar {_fmt(base.get('calmar'))}"
    )
    for symbol, item in results.items():
        standalone = item["standalone"]
        correlation = item["correlation"]
        regression = item["regression"]
        print(f"\n=== {symbol}（{item['label']}） ===")
        print(
            f"独立表现：年化 {_fmt(standalone.get('annual'), True)}｜"
            f"Sharpe {_fmt(standalone.get('sharpe'))}｜"
            f"回撤 {_fmt(standalone.get('max_dd'), True)}｜"
            f"Calmar {_fmt(standalone.get('calmar'))}"
        )
        print(
            f"相关性：日 {_fmt(correlation['daily'])}｜月 {_fmt(correlation['monthly'])}｜"
            f"下行日 {_fmt(correlation['downside'])}"
        )
        print(
            f"对微盘：beta {_fmt(regression['beta'])}｜"
            f"alpha {_fmt(regression['alpha'], True)}｜R² {_fmt(regression['r2'])}"
        )
        print("危机窗口（微盘前 5 次回撤）：")
        for row in item["crisis"]:
            print(
                f"  {row['peak']}→{row['trough']}　微盘 {_fmt(row['benchmark'], True)}"
                f"　候选腿 {_fmt(row['leg'], True)}　窗口相关 {_fmt(row['corr'])}"
            )
        print("叠加效果（固定权重日再平衡）：")
        for row in item["blend"]:
            print(
                f"  权重 {row['weight']:.0%}：年化 {_fmt(row['annual'], True)}｜"
                f"Sharpe {_fmt(row['sharpe'])}｜回撤 {_fmt(row['max_dd'], True)}｜"
                f"Calmar {_fmt(row['calmar'])}"
            )


def main() -> int:
    """命令行入口：判定候选风格腿能否作为第二收益源。"""
    parser = argparse.ArgumentParser(description="第二收益源判定")
    parser.add_argument(
        "--leg", action="append", default=None, help="候选腿代码，可重复"
    )
    args = parser.parse_args()
    legs = tuple(args.leg) if args.leg else DEFAULT_LEGS

    benchmark = _load_returns(BASELINE_SYMBOL)
    results: dict[str, Any] = {}
    for symbol in legs:
        results[symbol] = _judge_leg(benchmark, _load_returns(symbol), symbol)
    _render(results, benchmark)
    payload = {
        "baseline": BASELINE_SYMBOL,
        "baseline_metrics": _metrics(benchmark),
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
