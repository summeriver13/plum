"""TSMOM（时序趋势跟随）面板诊断 —— 先验证信号，再决定是否实现策略。

逐资产统计「信号状态 → 后 1/3/6 月收益」，信号分解为 12 个月动量（mom）、
200 日均线（ma）与双确认（dual），给出状态开/关两组的均值、胜率与 Welch t 值。

口径：后复权价（close × adj_factor）；每 21 个交易日评估一次，前向收益按收盘到收盘。
注意 h=63/126 在月频采样下有 3/6 倍窗口重叠，t 值需按重叠系数折减看待。

用法：uv run python scripts/diagnose_tsmom_panel.py [--since 2020-01-01]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLUM_DATA_MARKET_PATH", str(ROOT / "data" / "market" / "etf"))
MARKET_ROOT = Path(os.environ["PLUM_DATA_MARKET_PATH"]) / "1d"
OUT_DIR = ROOT / "data" / "results" / "tsmom_diag"

STEP = 21  # 评估步长（约 1 个月）
LOOKBACK = 252  # 12 个月动量窗
MA_WINDOW = 200  # 200 日均线
HORIZONS = {21: "1月", 63: "3月", 126: "6月"}

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Asset:
    """诊断标的。"""

    symbol: str
    name: str
    category: str


ASSETS: list[Asset] = [
    Asset("SH.510300", "沪深300ETF", "宽基"),
    Asset("SH.510500", "中证500ETF", "宽基"),
    Asset("SH.512100", "中证1000ETF", "宽基"),
    Asset("SH.511010", "国债ETF", "债"),
    Asset("SH.511220", "城投债ETF", "债"),
    Asset("SH.518880", "黄金ETF", "黄金"),
    Asset("SH.513100", "纳指ETF", "跨境"),
    Asset("SH.513500", "标普500ETF", "跨境"),
    Asset("SZ.159920", "恒生ETF", "跨境"),
    Asset("SH.513030", "德国30ETF", "跨境"),
    Asset("SZ.159985", "豆粕ETF", "商品"),
    Asset("SZ.159980", "有色ETF", "商品"),
    Asset("SH.512480", "半导体ETF", "行业"),
    Asset("SH.512690", "酒ETF", "行业"),
    Asset("SH.512880", "证券ETF", "行业"),
    Asset("SH.512010", "医药ETF", "行业"),
    Asset("SH.515030", "新能源车ETF", "行业"),
    Asset("SH.512660", "军工ETF", "行业"),
    Asset("SH.512800", "银行ETF", "行业"),
    Asset("SH.512760", "芯片ETF", "行业"),
]


def _safe(value: Any, default: float = math.nan) -> float:
    """安全转 float（NaN/None/异常回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _as_int(value: Any, default: int = 0) -> int:
    """安全转 int（异常回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(value: float) -> str:
    """百分比格式化（NaN 显示 —）。"""
    return "—" if value != value else f"{value:+.2%}"


def load_adj_close(symbol: str) -> pd.Series | None:
    """读取单个标的的后复权收盘价（剔除非正价脏行；历史不足则跳过）。"""
    exchange, code = symbol.split(".")
    if exchange not in ("SH", "SZ") or not code.isdigit():
        return None
    path = MARKET_ROOT / exchange / f"{code}.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=["close", "adj_factor"])
    close = frame["close"].astype("float64")
    factor = frame["adj_factor"].astype("float64")
    price = (close * factor).where((close > 0) & (factor > 0)).dropna()
    if len(price) < LOOKBACK + max(HORIZONS) + STEP:
        return None
    return price


def build_signals(values: np.ndarray) -> dict[str, np.ndarray]:
    """构造三个布尔信号：mom（12 个月动量为正）/ ma（站上 200 日线）/ dual。"""
    count = len(values)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    mom = np.zeros(count, dtype=bool)
    ma = np.zeros(count, dtype=bool)
    mom[LOOKBACK:] = values[LOOKBACK:] > values[:-LOOKBACK]
    rolling_mean = (cumulative[MA_WINDOW:] - cumulative[:-MA_WINDOW]) / MA_WINDOW
    ma[MA_WINDOW - 1 :] = values[MA_WINDOW - 1 :] >= rolling_mean
    return {"mom": mom, "ma": ma, "dual": mom & ma}


def forward_return(values: np.ndarray, horizon: int) -> np.ndarray:
    """前向收益（评估日 → h 个交易日后），数据不足处为 NaN。"""
    fwd = np.full(values.shape, math.nan, dtype="float64")
    if horizon < len(values):
        fwd[:-horizon] = values[horizon:] / values[:-horizon] - 1.0
    return fwd


def _welch_t(on_values: np.ndarray, off_values: np.ndarray, diff: float) -> float:
    """状态开/关两组均值的 Welch t 值（样本不足或零方差返回 NaN）。"""
    if on_values.size < 2 or off_values.size < 2:
        return math.nan
    var_on = _safe(on_values.var(ddof=1)) / on_values.size
    var_off = _safe(off_values.var(ddof=1)) / off_values.size
    denom = math.sqrt(var_on + var_off) if var_on + var_off > 0 else 0.0
    return math.nan if denom <= 0 else diff / denom


def stat_block(fwd: np.ndarray, on: np.ndarray, off: np.ndarray) -> dict[str, Any]:
    """状态开/关两组的前向收益统计。"""
    on_values = fwd[on]
    off_values = fwd[off]
    mean_on = _safe(on_values.mean()) if on_values.size else math.nan
    mean_off = _safe(off_values.mean()) if off_values.size else math.nan
    win_on = _safe((on_values > 0).mean()) if on_values.size else math.nan
    diff = mean_on - mean_off
    return {
        "n_on": _as_int(on_values.size),
        "n_off": _as_int(off_values.size),
        "mean_on": mean_on,
        "mean_off": mean_off,
        "win_on": win_on,
        "diff": diff,
        "t": _welch_t(on_values, off_values, diff),
    }


def diagnose(asset: Asset, since: np.datetime64 | None) -> dict[str, Any] | None:
    """单资产全套诊断（3 信号 × 3 期限）。"""
    price = load_adj_close(asset.symbol)
    if price is None:
        return None
    values = price.to_numpy(dtype="float64")
    dates = price.index.to_numpy()
    entry: dict[str, Any] = {
        "symbol": asset.symbol,
        "name": asset.name,
        "category": asset.category,
        "first": str(dates[0])[:10],
        "last": str(dates[-1])[:10],
        "signals": {},
    }
    for name, mask in build_signals(values).items():
        per_horizon: dict[str, Any] = {}
        for horizon in HORIZONS:
            fwd = forward_return(values, horizon)
            valid = np.isfinite(fwd)
            valid[: LOOKBACK + 1] = False
            grid = np.zeros(values.shape, dtype=bool)
            grid[LOOKBACK::STEP] = True
            valid &= grid
            if since is not None:
                valid &= dates >= since
            per_horizon[str(horizon)] = stat_block(fwd, valid & mask, valid & ~mask)
        entry["signals"][name] = per_horizon
    return entry


def _fmt_t(value: float) -> str:
    """t 值格式化（NaN 显示 —）。"""
    return "—" if value != value else f"{value:.2f}"


def _display_date(value: Any) -> str:
    """K 线时间戳（交易日 D 存为 D-1 16:00）折算为交易日日期。"""
    try:
        stamp = pd.Timestamp(value) + pd.Timedelta(hours=8)
    except (TypeError, ValueError):
        return "—"
    return stamp.strftime("%Y-%m-%d")


def print_asset(entry: dict[str, Any]) -> None:
    """打印单资产的双确认信号表（3 个期限）。"""
    print(
        f"\n{entry['symbol']} {entry['name']} [{entry['category']}]"
        f"  {_display_date(entry['first'])} ~ {_display_date(entry['last'])}"
    )
    dual = entry["signals"]["dual"]
    for horizon in HORIZONS:
        block = dual[str(horizon)]
        print(
            f"    {HORIZONS[horizon]}: on {_fmt_pct(block['mean_on'])}"
            f" / off {_fmt_pct(block['mean_off'])}"
            f" / diff {_fmt_pct(block['diff'])}"
            f" (t={_fmt_t(block['t'])}, n_on={block['n_on']})"
        )


def print_decomposition(entries: list[dict[str, Any]], horizon: int = 63) -> None:
    """打印 3 月期限下 mom / ma / dual 的均值差与 t 值分解。"""
    print(f"\n== 信号分解（{HORIZONS[horizon]}前向收益：均值差(t)） ==")
    print(f"{'标的':<12} {'mom':>20} {'ma':>20} {'dual':>20}")
    for entry in entries:
        cells = []
        for name in ("mom", "ma", "dual"):
            block = entry["signals"][name][str(horizon)]
            cells.append(f"{_fmt_pct(block['diff'])}(t={_fmt_t(block['t'])})")
        print(f"{entry['symbol']:<12} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")


def print_pooled(entries: list[dict[str, Any]]) -> None:
    """按类别与全体汇总（双确认信号，各期限）。"""
    print("\n== 汇总（双确认信号，均值差的等权平均） ==")
    categories = sorted({entry["category"] for entry in entries})
    for horizon in HORIZONS:
        print(f"\n{HORIZONS[horizon]}（{horizon}日）:")
        for category in categories:
            group = [e for e in entries if e["category"] == category]
            diffs = [e["signals"]["dual"][str(horizon)]["diff"] for e in group]
            valid = [d for d in diffs if d == d]
            positive = sum(1 for d in valid if d > 0)
            mean_diff = sum(valid) / len(valid) if valid else math.nan
            print(
                f"    {category:<4} 资产 {len(group)}"
                f" | 平均差 {_fmt_pct(mean_diff)}"
                f" | diff>0 {positive}/{len(group)}"
            )
        all_diffs = [e["signals"]["dual"][str(horizon)]["diff"] for e in entries]
        valid_all = [d for d in all_diffs if d == d]
        mean_all = sum(valid_all) / len(valid_all) if valid_all else math.nan
        print(f"    全体 资产 {len(entries)} | 平均差 {_fmt_pct(mean_all)}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="TSMOM 面板诊断")
    parser.add_argument("--since", default=None, help="只统计该日期之后的评估点")
    return parser.parse_args()


def main() -> int:
    """入口：逐资产诊断 + 汇总落盘。"""
    args = parse_args()
    since = np.datetime64(args.since) if args.since else None
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for asset in ASSETS:
        entry = diagnose(asset, since)
        if entry is None:
            missing.append(f"{asset.symbol}({asset.name})")
            continue
        entries.append(entry)
        print_asset(entry)
    print_decomposition(entries)
    print_pooled(entries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "step": STEP,
        "lookback": LOOKBACK,
        "ma_window": MA_WINDOW,
        "since": args.since,
        "market_root": str(MARKET_ROOT),
        "assets": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写出: {path}")
    if missing:
        print(f"缺失/历史不足: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
