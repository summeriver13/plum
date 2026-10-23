"""叠加 sleeve 与微盘暴露的互补性检验（#19 / TSMOM 复用）。

三块检验，全部基于既有回测产物（不新跑回测）：① 相关性（微盘 vs sleeve 日收益）；
② 组合模拟（日度再平衡权重扫描）；③ 停泊替换（降档期现金 1.5% → 国债/黄金 ETF 的
反事实净值，两条腿各做一遍）。口径与引擎报告一致（242 交易日/年、rf 2%），
净值与市场副本共用同一套存储时间戳直接对齐（已验证 0 未匹配）。

用法::
    uv run python scripts/analyze_etf_overlay.py [--micro-round 31]
    uv run python scripts/analyze_etf_overlay.py --etf-dir <TSMOM run 目录> --sleeve-label TSMOM-T2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 与引擎报告同口径（242 交易日/年、无风险利率 2%）
from app.engine.backtest.analysis.report import (  # noqa: E402
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)
from etf_overlay_report import (  # noqa: E402
    report_combos,
    report_correlation,
    report_overlay,
)

RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
DEFAULT_MARKET_ROOT = PROJECT_ROOT / "data" / "market" / "etf"
DEFAULT_OUT_DIR = RESULTS_ROOT / "etf_overlay_analysis"
WEIGHT_STEPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
DEFAULT_PARKS = ("SH.511010", "SH.518880")  # 国债ETF / 黄金ETF


def market_root() -> Path:
    """ETF 市场副本路径（与 backtest_etf_rotation.py 同一环境变量约定）。"""
    return Path(os.environ.get("PLUM_DATA_MARKET_PATH") or DEFAULT_MARKET_ROOT)


def latest_run_dir(strategy_dir: Path) -> Path:
    """取策略目录下时间戳最新的运行目录。"""
    runs = sorted(path for path in strategy_dir.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"{strategy_dir} 下没有运行目录")
    return runs[-1]


def load_equity(run_dir: Path) -> pd.DataFrame:
    """读取单轮回测的净值曲线（timestamp 为索引）。"""
    path = run_dir / "data" / "equity_curve.csv"
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    for column in ("total_value", "cash", "positions_value", "cash_interest"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_market_return(symbol: str, root: Path) -> pd.Series:
    """市场副本里某标的的复权日收益（close × adj_factor）。"""
    exchange, code = symbol.split(".")
    frame = pd.read_parquet(root / "1d" / exchange / f"{code}.parquet")
    hfq = frame["close"].astype(float) * frame["adj_factor"].astype(float)
    return hfq.pct_change(fill_method=None).rename(symbol)


def daily_return(equity: pd.Series) -> pd.Series:
    """净值 → 日收益。"""
    return equity.pct_change(fill_method=None)


def _as_float(value: Any, default: float = math.nan) -> float:
    """安全转 float：None/非数值/NaN 回退默认值（与回测脚本同惯例）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def equity_metrics(equity: pd.Series) -> dict[str, Any]:
    """净值 → 年化/波动/Sharpe/最大回撤/Calmar（与引擎报告同口径）。"""
    daily = daily_return(equity).dropna()
    years = (len(daily) + 1) / TRADING_DAYS_PER_YEAR
    growth = _as_float(equity.iloc[-1]) / _as_float(equity.iloc[0])
    cagr = growth ** (1.0 / years) - 1.0 if years > 0 else math.nan
    std = _as_float(daily.std(ddof=0))
    vol = std * math.sqrt(TRADING_DAYS_PER_YEAR)
    excess = daily - DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = (
        _as_float(excess.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0
        else math.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    mdd = _as_float(drawdown.min())
    return {
        "annual_return": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else math.nan,
        "days": len(daily),
    }


def align_returns(micro: pd.Series, etf: pd.Series) -> pd.DataFrame:
    """按时间戳对齐两条日收益序列。"""
    joined = pd.concat({"micro": micro, "etf": etf}, axis=1)
    return joined.dropna()


def correlation_stats(micro: pd.Series, etf: pd.Series) -> dict[str, Any]:
    """全样本与分年的相关系数、ETF 对微盘的 beta。"""
    joined = align_returns(micro, etf)
    yearly: dict[str, float] = {}
    for year, group in joined.groupby(joined.index.year):
        yearly[str(year)] = _as_float(group["micro"].corr(group["etf"]))
    beta = _as_float(joined["etf"].cov(joined["micro"]) / joined["micro"].var())
    return {
        "overall": _as_float(joined["micro"].corr(joined["etf"])),
        "yearly": yearly,
        "beta_etf_on_micro": beta,
        "days": len(joined),
    }


def combo_equity(micro: pd.Series, etf: pd.Series, micro_weight: float) -> pd.Series:
    """日度再平衡的组合净值（micro_weight 为微盘权重）。"""
    joined = align_returns(micro, etf)
    combined = micro_weight * joined["micro"] + (1.0 - micro_weight) * joined["etf"]
    return (1.0 + combined).cumprod()


def sweep_rows(micro_r: pd.Series, sleeve_r: pd.Series) -> list[dict[str, Any]]:
    """权重扫描：每档权重的组合指标。"""
    rows: list[dict[str, Any]] = []
    for weight in WEIGHT_STEPS:
        metrics = equity_metrics(combo_equity(micro_r, sleeve_r, weight))
        rows.append({"micro_weight": weight, **metrics})
    return rows


def parking_block(
    equity: pd.DataFrame, symbols: tuple[str, ...], root: Path
) -> tuple[dict[str, Any], float, list[dict[str, Any]], dict[str, pd.Series]]:
    """单腿停泊替换：基准指标、平均现金占比、各避险 ETF 行与替代净值。"""
    baseline = equity_metrics(equity["total_value"])
    cash_share = _as_float((equity["cash"] / equity["total_value"]).mean(), 0.0)
    rows: list[dict[str, Any]] = []
    curves: dict[str, pd.Series] = {}
    for symbol in symbols:
        park_r = load_market_return(symbol, root)
        curve = park_overlay(equity, park_r)
        curves[symbol] = curve
        rows.append(
            {
                "symbol": symbol,
                "park_annual_return": _as_float(park_r.dropna().mean())
                * TRADING_DAYS_PER_YEAR,
                **equity_metrics(curve),
            }
        )
    return baseline, cash_share, rows, curves


def park_overlay(equity: pd.DataFrame, park_return: pd.Series) -> pd.Series:
    """现金停泊替换为避险 ETF 的反事实净值。"""
    joined = equity.join(park_return.rename("park_return"), how="inner")
    previous = joined["total_value"].shift(1)
    r = joined["total_value"].pct_change(fill_method=None)
    cash_share = (joined["cash"].shift(1) / previous).fillna(0.0)
    cash_contrib = (joined["cash_interest"] / previous).fillna(0.0)
    r_new = r - cash_contrib + cash_share * joined["park_return"].fillna(0.0)
    return (1.0 + r_new.fillna(0.0)).cumprod()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="ETF 风控叠加与微盘暴露互补性检验（#19）"
    )
    parser.add_argument(
        "--micro-round", type=int, default=33, help="微盘轮次（v10_clean）"
    )
    parser.add_argument(
        "--etf-round", type=int, default=23, help="sleeve 轮次（etf_rounds_v3）"
    )
    parser.add_argument(
        "--micro-dir", default=None, help="微盘运行目录（默认按轮次自动定位）"
    )
    parser.add_argument(
        "--etf-dir", default=None, help="sleeve 运行目录（默认按轮次自动定位）"
    )
    parser.add_argument("--micro-label", default=None, help="微盘显示标签，如 R33")
    parser.add_argument(
        "--sleeve-label", default=None, help="sleeve 显示标签，如 TSMOM-T2"
    )
    parser.add_argument(
        "--parks", nargs="*", default=list(DEFAULT_PARKS), help="替换现金停泊的避险 ETF"
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def default_run_dir(relative: str, strategy: str) -> Path:
    """按仓库约定定位结果目录下的最新运行目录。"""
    return latest_run_dir(RESULTS_ROOT / relative / "plum" / strategy)


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """定位微盘与 sleeve 的运行目录。"""
    micro = args.micro_dir or default_run_dir(
        "_verify_clean_v10_all", f"drawdown_round{args.micro_round}strategy"
    )
    sleeve = args.etf_dir or default_run_dir(
        "etf_rounds_v3", f"etf_round{args.etf_round}strategy"
    )
    return Path(micro), Path(sleeve)


def main() -> int:
    """入口：相关性 → 权重扫描 → 双腿停泊替换 → 汇总落盘。"""
    args = parse_args()
    micro_dir, sleeve_dir = resolve_dirs(args)
    micro_label = args.micro_label or f"R{args.micro_round}"
    sleeve_label = args.sleeve_label or f"R{args.etf_round}"
    print(f"微盘运行目录（{micro_label}）: {micro_dir}")
    print(f"叠加 sleeve 运行目录（{sleeve_label}）: {sleeve_dir}")

    micro_df, sleeve_df = load_equity(micro_dir), load_equity(sleeve_dir)
    micro_r = daily_return(micro_df["total_value"])
    sleeve_r = daily_return(sleeve_df["total_value"])

    corr = correlation_stats(micro_r, sleeve_r)
    report_correlation(corr, micro_label, sleeve_label)
    combo_rows = sweep_rows(micro_r, sleeve_r)
    report_combos(combo_rows)

    parks = tuple(args.parks)
    root = market_root()
    park_blocks: dict[str, Any] = {}
    curves_by_leg: dict[str, dict[str, pd.Series]] = {}
    for leg_label, frame in (("微盘", micro_df), (f"sleeve {sleeve_label}", sleeve_df)):
        baseline, cash_share, rows, curves = parking_block(frame, parks, root)
        report_overlay(baseline, rows, cash_share, title=f"现金停泊替换（{leg_label}）")
        park_blocks[leg_label] = {
            "baseline": baseline,
            "cash_share_mean": cash_share,
            "rows": rows,
        }
        curves_by_leg[leg_label] = curves

    enhanced_rows: list[dict[str, Any]] = []
    if parks:
        parked_r = daily_return(curves_by_leg[f"sleeve {sleeve_label}"][parks[0]])
        enhanced_rows = sweep_rows(micro_r, parked_r)
        report_combos(
            enhanced_rows, title=f"组合模拟（微盘 × {parks[0]} 停泊后的 sleeve）"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "micro_dir": str(micro_dir),
        "sleeve_dir": str(sleeve_dir),
        "micro_label": micro_label,
        "sleeve_label": sleeve_label,
        "correlation": corr,
        "combos": combo_rows,
        "park_blocks": park_blocks,
        "enhanced_combos": enhanced_rows,
    }
    out = out_dir / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
