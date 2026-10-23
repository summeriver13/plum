"""行业 ETF 位置策略面板诊断 —— 时序/横截面「低位买入」到底有没有 edge？

两种组合构建（信号与模型细节见 ``scripts/etf_band_model.py``）：
- ``--mode band``：每只行业按回撤阶梯分配预算，反弹到高位减仓（回撤-反弹对称）；
- ``--mode xsec``：每 ``--hold-days`` 选出跌得最深的 ``--top-n`` 个行业等权持有。

池子：``scripts/etf_industry_pool.py``（名称映射行业 + 同质去重 + 限定 A 股股票型）。
基准：同仓位投池子等权（公平基准）、恒定权重、池子等权满仓（均含 1.5% 停泊）。
``--export-label`` 会把纸面净值写成引擎 run dir 结构，供叠加分析脚本复用。

用法：
    uv run python scripts/diagnose_etf_band.py
    uv run python scripts/diagnose_etf_band.py --mode xsec --dd-max -0.4 --top-n 5 \
        --hold-days 60 --export-label deep40x60
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLUM_DATA_MARKET_PATH", str(ROOT / "data" / "market" / "etf"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etf_band_model import (  # noqa: E402
    LEVELS,
    build_targets,
    build_xsec_targets,
    bucket_report,
    equity_frame,
    format_results,
    forward_matrix,
    simulate,
    simulate_sleeve,
    summarize,
)
from scripts.etf_explain_panels import (  # noqa: E402
    _num,
    adjusted_close,
    classify_symbols,
    load_panels,
)
from scripts.etf_industry_pool import (  # noqa: E402
    build_pool,
    load_names,
    symbol_name,
)

MARKET_ROOT = Path(os.environ["PLUM_DATA_MARKET_PATH"])
OUT_DIR = ROOT / "data" / "results" / "etf_band_diag"
RUN_ROOT = OUT_DIR / "plum"
START = datetime(2020, 1, 2)
END = datetime(2026, 9, 16)
FWD_HORIZONS = (20, 60, 120)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def print_signal_diagnostics(
    adj: pd.DataFrame, lookback: int, levels: tuple[float, ...]
) -> dict[str, object]:
    """位置分档 → 未来收益（信号信息含量，与组合构建独立）。"""
    window = max(lookback // 2, 120)
    windowed = cast("pd.DataFrame", adj.rolling(lookback, min_periods=window))
    dd = adj / cast("pd.DataFrame", windowed.max()) - 1.0
    rb = adj / cast("pd.DataFrame", windowed.min()) - 1.0
    report: dict[str, object] = {}
    print("\n=== 信号诊断：位置分档 → 未来收益 ===", flush=True)
    for horizon in FWD_HORIZONS:
        report[f"buckets_{horizon}"] = {}
        for demean, prefix in ((False, "绝对"), (True, "超额")):
            forward = forward_matrix(adj, horizon, demean)
            for level, label in ((dd, "dd"), (rb, "rb")):
                rows = bucket_report(level, forward, label, levels=levels)
                text = "  ".join(
                    f"{row.bucket}: {row.mean:+.2%}(n={row.n})" for row in rows
                )
                print(f"  未来 {horizon} 日 [{label}·{prefix}] {text}", flush=True)
                cast("dict[str, object]", report[f"buckets_{horizon}"])[
                    f"{label}_{'excess' if demean else 'abs'}"
                ] = [asdict(row) for row in rows]
    return report


def export_curve(
    label: str, index: pd.Index, equity: np.ndarray, exposure: np.ndarray
) -> Path:
    """把纸面净值写成引擎 run dir 结构（供叠加分析复用）。"""
    run_dir = RUN_ROOT / f"etf_band{label}strategy" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    frame = equity_frame(index, equity, exposure)
    frame.to_csv(run_dir / "data" / "equity_curve.csv", index=False)
    return run_dir


def main() -> None:
    """入口：建池 → 信号诊断 → 组合回放 → 基准对比 →（可选）导出净值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS))
    parser.add_argument("--confirm", action="store_true", help="动量破位则不持仓")
    parser.add_argument("--no-cap", action="store_true", help="去掉高位减仓上限")
    parser.add_argument("--base", type=float, default=0.0, help="基础仓位份额（余下按信号加仓）")
    parser.add_argument("--mode", choices=("band", "xsec"), default="band")
    parser.add_argument("--dd-max", type=float, default=-0.40, help="横截面版的门槛")
    parser.add_argument("--top-n", type=int, default=5, help="横截面版持仓只数")
    parser.add_argument("--hold-days", type=int, default=60, help="横截面版调仓间隔")
    parser.add_argument("--export-label", default=None, help="导出净值时使用的标签")
    args = parser.parse_args()
    levels = tuple(args.levels)

    print("读取 ETF 行情副本…", flush=True)
    panels = load_panels(MARKET_ROOT)
    full = adjusted_close(panels)
    adj = full.loc[START:END]
    pool = build_pool(adj, panels["amount"].loc[START:END], load_names(), classify_symbols(full))
    if not pool:
        raise SystemExit("行业 ETF 池为空：检查名称匹配或样本门槛")
    symbols = sorted(pool)
    adj = adj[symbols]
    ret = adj.pct_change()
    names = load_names()
    print(f"行业 ETF 池：{len(symbols)} 个行业", flush=True)
    for symbol in symbols:
        print(f"  {pool[symbol]:<6} {symbol}  {symbol_name(names, symbol)}", flush=True)

    report = print_signal_diagnostics(adj, args.lookback, levels)
    sleeve = cast("pd.Series", ret.mean(axis=1)).fillna(0.0).to_numpy(dtype="float64")
    if args.mode == "xsec":
        targets = build_xsec_targets(
            adj,
            args.lookback,
            dd_max=args.dd_max,
            top_n=args.top_n,
            hold_days=args.hold_days,
        )
        budget, label = 1.0, f"xsec{abs(args.dd_max):.0%}x{args.hold_days}"
    else:
        targets = build_targets(
            adj,
            args.lookback,
            confirm=args.confirm,
            cap_enabled=not args.no_cap,
            levels=levels,
            base=args.base,
        )
        budget, label = 1.0 / len(symbols), "band"
    equity, exposure = simulate(targets, ret, budget)
    average = _num(np.mean(exposure))
    flat = np.full(len(exposure), average)
    results = [
        summarize("低位买入/高位卖出", equity, exposure),
        summarize("基准：同仓位投池子等权", simulate_sleeve(exposure, sleeve), exposure),
        summarize(f"基准：恒定权重 {average:.0%}", simulate_sleeve(flat, sleeve), flat),
        summarize(
            "基准：池子等权满仓",
            simulate_sleeve(np.ones(len(exposure)), sleeve),
            np.ones(len(exposure)),
        ),
    ]
    print("\n=== 组合回放 ===", flush=True)
    print(format_results(results), flush=True)

    report.update(
        {
            "lookback": args.lookback,
            "mode": args.mode,
            "levels": list(levels),
            "confirm": args.confirm,
            "cap_enabled": not args.no_cap,
            "base": args.base,
            "dd_max": args.dd_max,
            "top_n": args.top_n,
            "hold_days": args.hold_days,
            "pool": {symbol: pool[symbol] for symbol in symbols},
            "results": [asdict(item) for item in results],
        }
    )
    if args.export_label:
        run_dir = export_curve(args.export_label, adj.index, equity, exposure)
        print(f"\n净值已导出（可作 --etf-dir 传参）：{run_dir}", flush=True)
        report["export_dir"] = str(run_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"汇总已写出：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
