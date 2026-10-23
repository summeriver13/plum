"""从数据湖重建 plum 的 **ETF** 回测数据集（与个股副本分离）。

个股副本见 `build_plum_market_from_qdata.py`，产物在 `data/market/1d`；
本脚本产出 `data/market/etf_1d`，两者互不污染（个股策略的标的枚举
不会意外看到 ETF，反之亦然）。

数据来源：`clean/etf_daily`（QMT 采集 + 自算后复权因子）
- `adj_factor` 与个股同约定：`hfq = close × adj_factor`，上市日锚定 1.0
- ETF 无股本表，故无 `turn` / `turn_calc` / 市值 / 估值字段
- `limit_rate`（涨跌停制度）数据湖未采集 ETF 版本，留空由规则层决定
- `is_st` 恒为 False；`is_delisted` 仅标在已退市标的的最后一根 K 线

时间戳约定与个股副本一致（plum 全仓统一）：
    北京时间 D → `D - 1 天 + 16:00`（即 UTC 前一日 16:00）

用法::

    uv run python scripts/build_plum_etf_market_from_qdata.py \
        --out-dir data/market/etf_1d
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_plum_market_from_qdata import (  # noqa: E402
    PROJECT_ROOT,
    TS_OFFSET,
    load_index_frame,
    read_year_partitions,
    resolve_data_root,
    write_indexes,
)

logger = logging.getLogger("build-plum-etf-market")

ETF_DATASET = "etf_daily"
INSTRUMENTS_DATASET = "instruments"
# 必须非空：plum 的 Bar 构造对这些列直接做 Decimal() 转换
REQUIRED_COLUMNS = ("open", "high", "low", "close")
# 缺失时的默认值（停牌/首日场景）
DEFAULT_FILL = {"volume": 0, "amount": 0.0, "pre_close": 0.0, "adj_factor": 1.0}
OUTPUT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "adj_factor",
    "is_st",
    "is_delisted",
    "limit_rate",
)

# 涨跌停制度：沪深 ETF 常规为 10%，跟踪科创板/创业板指数的 ETF 为 20%。
# 湖侧未采集 ETF 的涨跌停标注，此处自行判定：
#   1) 代码 588xxx 必为科创板 ETF → 20%
#   2) 否则看数据证据：10% 制度下，因最小报价单位 0.001 舍入，价格最高可到
#      pre_close ×(1 + 10% + 0.0005/pre_close)；超过这个上界的，只可能是 20% 制度
# 采用名称关键字会误判（「双创龙头ETF」「AAA科创债ETF」都会被盖到），故不用名称。
LIMIT_RATE_WIDE = Decimal("0.20")
LIMIT_RATE_NARROW = Decimal("0.10")
WIDE_LIMIT_CODE_PREFIX = "588"
TICK_HALF = 0.0005  # 最小报价单位 0.001 的一半（相对误差上界）
EVIDENCE_TOLERANCE = 1e-4


def read_snapshot(dataset: str) -> pd.DataFrame:
    """读取非年份分区的快照数据集（如 instruments，可能带二级子目录）。"""
    root = resolve_data_root() / "clean" / dataset
    paths = sorted(root.glob("*.parquet")) or sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 {dataset} 快照：{root}")
    frames = [pd.read_parquet(path) for path in paths]
    logger.info("%s：读取 %d 个文件，%d 行", dataset, len(paths), sum(len(f) for f in frames))
    return pd.concat(frames, ignore_index=True)


def load_delisted_symbols(etf_symbols: set[str]) -> set[str]:
    """从 instruments 快照取已退市且属于 ETF 标的池的集合。

    退市不可逆，故快照即 PIT 安全。注意 instruments 是全市场表
    （含个股/指数/可转债），必须限定在 ETF 符号内，否则统计与语义都会串味
    （实测全市场 1,185 只退市标的均为个股，ETF 目前一只都没标记）。
    """
    frame = read_snapshot(INSTRUMENTS_DATASET)
    if "is_delisted" not in frame.columns:
        logger.warning("instruments 无 is_delisted 列，跳过退市标记")
        return set()
    flag = frame["is_delisted"].fillna(False).astype(bool)
    symbols = frame["symbol"].astype(str)
    return set(symbols[flag & symbols.isin(list(etf_symbols))])


def to_plum_frame(group: pd.DataFrame) -> pd.DataFrame:
    """→ plum 格式（timestamp 索引 + object 列 + None 表示缺失）。"""
    frame = group.drop(columns=["symbol"]).copy()
    frame["timestamp"] = (
        pd.to_datetime(frame["datetime"]) - pd.Timedelta(days=1) + TS_OFFSET
    )
    frame = frame.drop(columns=["datetime"]).set_index("timestamp").sort_index()
    frame = frame.reindex(columns=[c for c in OUTPUT_COLUMNS if c in frame.columns])
    for column in frame.columns:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def resolve_limit_rate(symbol: str, group: pd.DataFrame) -> Decimal:
    """判定标的的涨跌停比例（10% / 20%），判据见模块顶部注释。"""
    code = symbol.split(".")[0]
    if code.startswith(WIDE_LIMIT_CODE_PREFIX):
        return LIMIT_RATE_WIDE
    valid = group[group["pre_close"] > 0]
    if valid.empty:
        return LIMIT_RATE_NARROW
    ratio = cast("pd.Series", valid["close"] / valid["pre_close"])
    deviation = (ratio - 1.0).abs()
    ceiling = _num(LIMIT_RATE_NARROW) + TICK_HALF / valid["pre_close"].astype(float)
    excess = cast("pd.Series", deviation - ceiling)
    return (
        LIMIT_RATE_WIDE
        if _num(excess.max()) > EVIDENCE_TOLERANCE
        else LIMIT_RATE_NARROW
    )


def build_symbol_frame(
    symbol: str, group: pd.DataFrame, delisted: bool
) -> pd.DataFrame:
    """单标的：清洗 → 补默认值 → 标注制度/退市 → 转 plum 格式。"""
    group = cast("pd.DataFrame", group.sort_values("datetime").reset_index(drop=True))
    group = cast("pd.DataFrame", group.dropna(subset=list(REQUIRED_COLUMNS)))
    group = cast("pd.DataFrame", group[group["close"] > 0])
    if group.empty:
        return pd.DataFrame()
    for column, value in DEFAULT_FILL.items():
        if column in group.columns:
            group[column] = group[column].fillna(value)
    group["is_st"] = False
    group["is_delisted"] = False
    group["limit_rate"] = resolve_limit_rate(symbol, group)
    if delisted:
        group.loc[group.index[-1], "is_delisted"] = True
    return to_plum_frame(group)


def write_symbol(frame: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    """写出单标的 parquet（SH/SZ 分目录，代码不带交易所后缀）。"""
    code, exchange = symbol.split(".")
    directory = out_dir / exchange
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{code}.parquet", compression="snappy")


def _num(value: object, default: float = 0.0) -> float:
    """安全转 float（失败或 NaN 返回默认值）。"""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def verify_sample(out_dir: Path, expected: dict[str, int], sample: int) -> int:
    """抽样回读校验（行数/时间戳单调/因子起点/价格为正），返回问题数。"""
    names = sorted(expected)[:: max(1, len(expected) // sample)][:sample]
    problems = 0
    for symbol in names:
        code, exchange = symbol.split(".")
        path = out_dir / exchange / f"{code}.parquet"
        if not path.exists():
            logger.error("[校验] 缺文件 %s", path)
            problems += 1
            continue
        frame = pd.read_parquet(path)
        factor_base = _num(frame["adj_factor"].iloc[0], default=-1.0)
        checks = {
            "行数": len(frame) == expected[symbol],
            "时间戳单调": frame.index.is_monotonic_increasing,
            "因子起点": abs(factor_base - 1.0) < 1e-6,
            "价格为正": bool((frame["close"].astype(float) > 0).all()),
        }
        for name, ok in checks.items():
            if not ok:
                logger.error("[校验] %s %s 不通过", symbol, name)
                problems += 1
    logger.info("抽样校验 %d 只，问题 %d 项", len(names), problems)
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从数据湖重建 plum 的 ETF 回测数据集")
    parser.add_argument(
        "--out-dir", default=str(PROJECT_ROOT / "data" / "market" / "etf" / "1d")
    )
    parser.add_argument("--start", default="2005-01-01", help="ETF 最早 IPO 于 2005")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--symbols", nargs="*", help="仅重建指定标的（湖侧格式 510300.SH）")
    parser.add_argument("--verify-sample", type=int, default=20)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="不重建指数桶（ETF 副本自带 IDX，基准与趋势叠加依赖它）",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
    args = parse_args()
    start = cast("pd.Timestamp", pd.Timestamp(args.start))
    end = cast("pd.Timestamp", pd.Timestamp(args.end))
    out_dir = Path(args.out_dir)

    bars = cast("pd.DataFrame", read_year_partitions(ETF_DATASET))
    bars["datetime"] = pd.to_datetime(bars["datetime"]).astype("datetime64[ns]")
    window = (bars["datetime"] >= start) & (bars["datetime"] <= end)
    bars = cast("pd.DataFrame", bars[window])
    if args.symbols:
        bars = cast("pd.DataFrame", bars[bars["symbol"].isin(args.symbols)])
    bars = cast("pd.DataFrame", bars.sort_values(["symbol", "datetime"]))
    if bars.empty:
        logger.error("区间 %s ~ %s 无数据", args.start, args.end)
        return 1
    delisted = load_delisted_symbols(set(bars["symbol"].astype(str)))
    logger.info(
        "区间 %s ~ %s：%d 行，%d 只标的（其中已退市 %d 只）",
        args.start,
        args.end,
        len(bars),
        bars["symbol"].nunique(),
        len(delisted),
    )
    if args.dry_run:
        logger.info("dry-run：将输出到 %s", out_dir)
        return 0

    written = 0
    skipped: list[str] = []
    expected: dict[str, int] = {}
    for symbol, group in bars.groupby("symbol", sort=False, observed=True):
        key = str(symbol)
        frame = build_symbol_frame(key, group, key in delisted)
        if frame.empty:
            skipped.append(key)
            continue
        write_symbol(frame, key, out_dir)
        expected[key] = len(frame)
        written += 1
    logger.info("写出 %d 只到 %s（跳过 %d 只）", written, out_dir, len(skipped))
    if skipped:
        logger.warning("跳过标的示例：%s", ", ".join(sorted(skipped)[:5]))

    problems = verify_sample(out_dir, expected, args.verify_sample) if expected else 0
    if not args.skip_index and not args.symbols:
        # write_indexes 内部按 symbol 分组并转 plum 格式，传入湖侧原始帧即可
        indexes = write_indexes(load_index_frame(start, end), out_dir)
        logger.info("指数桶写出 %d 个到 %s", indexes, out_dir / "IDX")
    root = resolve_data_root()
    logger.info("数据湖根目录：%s（仅读取 clean/%s）", root, ETF_DATASET)
    return 1 if problems or not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
