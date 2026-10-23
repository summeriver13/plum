"""卫星仓（行业 ETF 深跌买入）与微盘 / TSMOM / ETF 轮动的互补性检验。

四块检验，全部基于既有回测产物（卫星仓为纸面净值导出，其余为引擎回测）：
① 全相关（日/月）与对微盘的 beta；② 条件表现（微盘最差 10% 日、微盘月跌幅 < -5%）；
③ 三腿权重扫描（基线 = 微盘 × TSMOM，叠加卫星仓 0~20%）；
④ 卫星仓闲置现金的停泊替换（国债 ETF）。口径与引擎报告一致（242 交易日、rf 2%），
工具函数复用 ``scripts/analyze_etf_overlay.py``。

用法::
    uv run python scripts/analyze_band_overlay.py
    uv run python scripts/analyze_band_overlay.py --sat-dir <导出目录> --sat-label deep40x60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_etf_overlay import (  # noqa: E402
    _as_float,
    daily_return,
    equity_metrics,
    latest_run_dir,
    load_equity,
    market_root,
    park_overlay,
)
from app.engine.backtest.analysis.report import TRADING_DAYS_PER_YEAR  # noqa: E402

RESULTS = ROOT / "data" / "results"
OUT_DIR = RESULTS / "etf_band_diag" / "overlay"

LEGS = (
    ("微盘", "_verify_clean_v10_all", "drawdown_round33strategy"),
    ("TSMOM", "tsmom_rounds", "tsmom_round2strategy"),
    ("ETF轮动", "etf_rounds_v4", "etf_round36strategy"),
)
BAND_REL = "etf_band_diag"
DEFAULT_PARKS = ("SH.511010", "SH.518880")  # 国债ETF / 黄金ETF
MONTH_TAIL = -0.05  # 「微盘月跌幅」阈值
WORST_QUANTILE = 0.10  # 「微盘最差日」分位
MICRO_WEIGHTS = (0.5, 0.6, 0.7, 0.8)
SAT_SHARES = (0.0, 0.05, 0.10, 0.15, 0.20)


def band_label(run_dir: Path) -> str:
    """从 run dir 推断卫星仓标签（``.../etf_band<label>strategy/<stamp>``）。"""
    name = run_dir.parent.name
    return name.removeprefix("etf_band").removesuffix("strategy") or "band"


def align(series: dict[str, pd.Series]) -> pd.DataFrame:
    """按时间戳对齐多腿日收益（内连接，丢弃任一条缺失的日期）。"""
    frame = pd.concat(series, axis=1)
    return frame.dropna()


def month_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """日收益 → 月收益。"""
    return (1.0 + returns).resample("ME").prod() - 1.0


def correlation_table(returns: pd.DataFrame) -> dict[str, Any]:
    """日/月相关矩阵与各腿对微盘的 beta。"""
    monthly = month_returns(returns)
    micro = cast("pd.Series", returns["微盘"])
    betas = {
        column: _as_float(cast("pd.Series", returns[column]).cov(micro) / micro.var())
        for column in returns.columns
        if column != "微盘"
    }
    month_annual = {
        column: _as_float((1.0 + cast("pd.Series", monthly[column])).prod())
        ** (12.0 / len(monthly))
        - 1.0
        for column in monthly.columns
    }
    return {
        "daily_corr": returns.corr().round(3).to_dict(),
        "monthly_corr": monthly.corr().round(3).to_dict(),
        "beta_on_micro": betas,
        "monthly_annual": month_annual,
    }


def yearly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """分年收益（各腿），索引为年份字符串。"""
    years = cast("pd.DatetimeIndex", returns.index).year
    rows = [
        {
            "year": str(year),
            **{
                column: _as_float((1.0 + cast("pd.Series", group[column])).prod()) - 1.0
                for column in group.columns
            },
        }
        for year, group in returns.groupby(years)
    ]
    return pd.DataFrame(rows).set_index("year")


def conditional_stats(returns: pd.DataFrame) -> dict[str, Any]:
    """微盘最差日 / 微盘大跌月里，各腿的条件表现与条件相关。"""
    micro = cast("pd.Series", returns["微盘"])
    worst_days = returns[micro <= micro.quantile(WORST_QUANTILE)]
    tail_micro = cast("pd.Series", worst_days["微盘"])
    monthly = month_returns(returns)
    tail_months = monthly[cast("pd.Series", monthly["微盘"]) <= MONTH_TAIL]
    return {
        "最差10%日": {
            "days": len(worst_days),
            "avg_daily": {
                column: _as_float(cast("pd.Series", worst_days[column]).mean())
                for column in returns.columns
            },
            "corr_with_micro": {
                column: _as_float(cast("pd.Series", worst_days[column]).corr(tail_micro))
                for column in returns.columns
                if column != "微盘"
            },
        },
        "微盘大跌月": {
            "months": len(tail_months),
            "avg_monthly": {
                column: _as_float(cast("pd.Series", tail_months[column]).mean())
                for column in monthly.columns
            },
            "win_rate": {
                column: _as_float((cast("pd.Series", tail_months[column]) > 0).mean())
                for column in monthly.columns
            },
        },
    }


def three_leg(
    returns: pd.DataFrame, sat: str, micro_weight: float, sat_share: float
) -> pd.Series:
    """日度再平衡的三腿净值：基线（微盘 × TSMOM）与卫星仓按权重混合。"""
    base = micro_weight * returns["微盘"] + (1.0 - micro_weight) * returns["TSMOM"]
    combined = (1.0 - sat_share) * base + sat_share * returns[sat]
    return (1.0 + combined).cumprod()


def sweep(returns: pd.DataFrame, sat: str) -> list[dict[str, Any]]:
    """权重扫描：基线权重 × 卫星仓份额的组合指标。"""
    rows: list[dict[str, Any]] = []
    for weight in MICRO_WEIGHTS:
        for share in SAT_SHARES:
            metrics = equity_metrics(three_leg(returns, sat, weight, share))
            rows.append(
                {"micro_weight": weight, "sat_share": share, **metrics}
            )
    return rows


def format_sweep(rows: list[dict[str, Any]], sat: str) -> str:
    """权重扫描表（按 Sharpe 排序输出前若干行）。"""
    lines = [
        f"### 三腿权重扫描（基线 = 微盘×TSMOM，叠加 {sat}）",
        "",
        "| 微盘权重 | 卫星仓份额 | 年化 | 波动 | Sharpe | 最大回撤 | Calmar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['micro_weight']:.0%} | {row['sat_share']:.0%} | "
            f"{row['annual_return']:+.2%} | {row['annual_vol']:.2%} | "
            f"{row['sharpe']:.2f} | {row['max_drawdown']:.2%} | {row['calmar']:.2f} |"
        )
    return "\n".join(lines)


def resolve_run_dir(value: str) -> Path:
    """把用户给的运行目录约束在 ``data/results`` 之下（防路径穿越）。"""
    path = Path(value).expanduser().resolve()
    if RESULTS.resolve() not in path.parents:
        raise SystemExit(f"运行目录必须位于 {RESULTS} 之下：{path}")
    return path


def resolve_legs(args: argparse.Namespace) -> dict[str, Path]:
    """定位各腿运行目录，并附上卫星仓（可多个）。"""
    dirs = {label: latest_run_dir(RESULTS / rel / "plum" / strategy) for label, rel, strategy in LEGS}
    for raw in args.sat_dir:
        run_dir = resolve_run_dir(raw)
        dirs[band_label(run_dir)] = run_dir
    if not args.sat_dir:
        raise SystemExit("需要 --sat-dir 指定卫星仓净值目录（由 diagnose_etf_band.py 导出）")
    return dirs


def main() -> None:
    """入口：相关性 → 条件表现 → 三腿扫描 → 停泊替换 → 落盘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat-dir", nargs="*", default=[], help="卫星仓净值目录（可多个）")
    parser.add_argument("--parks", nargs="*", default=list(DEFAULT_PARKS))
    args = parser.parse_args()

    dirs = resolve_legs(args)
    for label, path in dirs.items():
        print(f"{label:<12} {path}")
    prices = {label: load_equity(path) for label, path in dirs.items()}
    series_map = {
        label: daily_return(cast("pd.Series", frame["total_value"]))
        for label, frame in prices.items()
    }
    returns_all = pd.concat(series_map, axis=1)
    satellite_keys = [band_label(resolve_run_dir(raw)) for raw in args.sat_dir]
    main_sat = satellite_keys[0]
    order = list(dict.fromkeys([*[name for name, _, _ in LEGS], *satellite_keys]))
    returns = align({label: cast("pd.Series", returns_all[label]) for label in order})

    report: dict[str, Any] = {"generated": datetime.now().isoformat(timespec="seconds")}
    table = correlation_table(returns)
    corr_matrix = cast("pd.DataFrame", returns.corr())
    print("\n=== 相关性（日收益，下三角）===")
    print(corr_matrix.round(3).to_string())
    print("\n=== 相关性与 beta 摘要 ===")
    for column, value in table["beta_on_micro"].items():
        corr = _as_float(corr_matrix.loc[column, "微盘"])
        print(f"  {column:<12} 日相关 {corr:+.3f}  beta(对微盘) {value:+.3f}")
    print("\n=== 分年收益 ===")
    print(yearly_returns(returns).to_string(float_format=lambda v: f"{v:+.1%}"))

    conditions = conditional_stats(returns)
    print("\n=== 条件表现 ===")
    print(json.dumps(conditions, ensure_ascii=False, indent=2))
    rows = sweep(returns, main_sat)
    print("\n" + format_sweep(rows, main_sat))

    root = market_root()
    park_rows: list[dict[str, Any]] = []
    for symbol in args.parks:
        exchange, code = symbol.split(".")
        frame = pd.read_parquet(root / "1d" / exchange / f"{code}.parquet")
        hfq = frame["close"].astype(float) * frame["adj_factor"].astype(float)
        curve = park_overlay(prices[main_sat], hfq.pct_change(fill_method=None).rename(symbol))
        park_rows.append({"symbol": symbol, **equity_metrics(curve)})
    print("\n=== 卫星仓闲置现金的停泊替换 ===")
    print(pd.DataFrame(park_rows).to_string(float_format=lambda v: f"{v:+.3f}"))

    report.update(
        {
            "dirs": {label: str(path) for label, path in dirs.items()},
            "satellite": satellite_keys,
            "correlation": table,
            "yearly": yearly_returns(returns).to_dict(),
            "conditional": conditions,
            "sweep": rows,
            "park_rows": park_rows,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写出：{out}")


if __name__ == "__main__":
    main()
