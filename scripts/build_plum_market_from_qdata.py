"""从数据湖（qdata）重建 plum 回测数据集。

动机
----
plum 原有的 `data/market/1d` 来自 QMT 迁移，存在两个致命缺陷：

1. **池子缺 499 只标的**：含整个深主板 000xxx 系列（平安银行、万科A 等），
   属于系统性缺失，无法用补字段的方式修复。
2. **市值字段只覆盖 56%**：缺失组在 2020-2025 的中位收益为 -39.9%，
   覆盖组仅 -1.1%（差 38.9 个百分点），使小市值策略回测产生严重幸存者偏差。

本脚本改用数据湖 clean 层（5,444 只，默认 2020-01-01 ~ 当天，可用 --start/--end 覆盖）重建：

- 行情：`clean/daily`（含退市股、停牌剔除后的规范行）
- 复权：`clean/daily` 内联的 `adj_factor`
- 估值：`clean/daily_valuation`（pe_ttm/pb_mrq/ps_ttm/pcf_ncf_ttm）
- 市值：**直接取 `clean/daily_valuation` 的 total_mv/circ_mv**（湖侧优先用 clean/financials
  的 QMT PIT 股本计算，缺失时回退东财股本/turn 反推）；东财股本 × 收盘仅作副本侧兜底。
  旧版副本自行用东财股本算 total_mv，对缺增发记录的标的（如 600355）会低估一半市值。

输出格式严格复刻 plum 原格式
--------------------------
- index 名为 `timestamp`，**naive**，值 = 交易日 - 1 天 + 16:00（QMT 的 UTC 表示）
- 列 dtype 为 `object`，缺失值为 `None`（避免产生 Decimal('NaN')）

指数（基准）
-----------
`clean/index_daily` → `1d/IDX/<代码>.parquet`，plum 符号形如 `IDX.000852`。
桶名 `IDX` 不在股票池扫描的 SH/SZ/BJ 白名单内，因此不会污染股票宇宙；
timestamp 与个股采用**完全相同**的约定，否则基准按日期对齐会静默回退为空。

按标的循环处理：`merge_asof` 要求左键单调，单标的组内天然满足；同时与
「按标的写文件」合并为一次遍历。

用法
----
    uv run python scripts/build_plum_market_from_qdata.py --dry-run
    uv run python scripts/build_plum_market_from_qdata.py --out-dir data/market/1d
    uv run python scripts/build_plum_market_from_qdata.py --skip-index   # 只重建个股
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("build-plum-market")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

ENV_KEY = "QDATA_ROOT"
ENV_FILE = PROJECT_ROOT.parent / "market-data" / ".env"
EQUITY_DATASET_PATH = (
    "raw",
    "eastmoney",
    "equity_shares",
    "symbol=ALL",
    "equity_shares.parquet",
)
DAILY_DATASET = "daily"
INDEX_DATASET = "index_daily"
VALUATION_DATASET = "daily_valuation"
INDEX_BUCKET = "IDX"  # 指数目录桶；plum 符号 = IDX.<6位代码>
VERIFY_SAMPLE_EVERY = 200  # 一致性抽检间隔（每 N 只标的抽 1 只）
TS_OFFSET = pd.Timedelta(hours=16)  # 交易日 - 1 天 + 16:00
VALUATION_COLUMNS = ("pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm")
# 从数据湖估值层取回的市场数据列。total_mv/circ_mv 已由湖侧优先用 QMT PIT 股本
# 计算（见 market-data/app/clean/shares.py），plum 侧直接消费，不再自行用东财股本推算
# （东财对部分标的缺增发记录，会使市值低估，如 600355 仅剩股改行）。
VALUATION_FETCH_COLUMNS = (*VALUATION_COLUMNS, "total_mv", "circ_mv")
# 换手率两列并存，与湖契约一致：turn = 券商口径（覆盖不全，2020 年约 38%），
# turn_calc = 自算口径（成交量 ÷ 公告日生效流通股本，覆盖 99.9%+）。
# 消费方按需选择；需要单一序列时优先 turn_calc（或自行首次非空合并）。
TURN_CALC_COLUMN = "turn_calc"
# 这些列必须非空：plum 的 Bar 构造会对它们直接做 Decimal()/int() 转换，
# None 会让 Decimal(str(None)) 抛 ConversionSyntax 而丢失整只标的。
REQUIRED_COLUMNS = ("open", "high", "low", "close")
# 这些列缺失时用默认值填充（停牌/首日场景）
DEFAULT_FILL = {"volume": 0, "amount": 0.0, "pre_close": 0.0, "adj_factor": 1.0}
OUTPUT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "turn_calc",
    "pre_close",
    "adj_factor",
    "is_st",
    "is_delisted",
    "limit_rate",
    "limit_reason",
    *VALUATION_COLUMNS,
    "total_mv",
    "circ_mv",
)


def resolve_data_root() -> Path:
    """解析数据湖根目录：优先环境变量，其次 market-data/.env（不硬编码路径）。"""
    value = os.environ.get(ENV_KEY, "").strip()
    if not value and ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
        value = os.environ.get(ENV_KEY, "").strip()
    if not value:
        raise RuntimeError(f"未找到 {ENV_KEY}（环境变量或 {ENV_FILE}）")
    return Path(value).expanduser()


def read_year_partitions(
    dataset: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """读取某数据集的全部年份分区。"""
    root = resolve_data_root() / "clean" / dataset
    paths = sorted(root.glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 {dataset} 分区：{root}")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    logger.info(
        "%s：读取 %d 个分区，%d 行", dataset, len(paths), sum(len(f) for f in frames)
    )
    return pd.concat(frames, ignore_index=True)


def load_equity_shares() -> pd.DataFrame:
    """加载东财总股本变动记录，返回按标的分组的字典。"""
    path = resolve_data_root().joinpath(*EQUITY_DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少股本数据：{path}（先运行 fetch_equity_shares.py）"
        )
    frame = pd.read_parquet(path, columns=["symbol", "effective_date", "total_shares"])
    frame["effective_date"] = pd.to_datetime(frame["effective_date"]).astype(
        "datetime64[ns]"
    )
    frame["total_shares"] = pd.to_numeric(frame["total_shares"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "effective_date", "total_shares"])
    logger.info("股本：%d 条，覆盖 %d 只", len(frame), frame["symbol"].nunique())
    grouped = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        grouped[str(symbol)] = group.sort_values("effective_date").reset_index(
            drop=True
        )
    return grouped


def load_valuation_map() -> dict[str, pd.DataFrame]:
    """加载估值字段（含市值）并按标的分组。"""
    frame = read_year_partitions(
        VALUATION_DATASET, ["symbol", "datetime", *VALUATION_FETCH_COLUMNS]
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"]).astype("datetime64[ns]")
    return {
        str(symbol): group.drop(columns=["symbol"]).set_index("datetime")
        for symbol, group in frame.groupby("symbol", sort=False)
    }


def build_symbol_frame(
    symbol: str,
    group: pd.DataFrame,
    equity_group: pd.DataFrame | None,
    valuation: pd.DataFrame | None,
) -> pd.DataFrame:
    """单标的：附加估值（含市值）→ 东财股本仅作市值兜底 → 转 plum 格式。"""
    group = group.sort_values("datetime").reset_index(drop=True)
    group = group.dropna(subset=list(REQUIRED_COLUMNS))
    for column, value in DEFAULT_FILL.items():
        if column in group.columns:
            group[column] = group[column].fillna(value)
    if group.empty:
        return pd.DataFrame()
    if equity_group is not None:
        group = pd.merge_asof(
            group,
            equity_group[["effective_date", "total_shares"]],
            left_on="datetime",
            right_on="effective_date",
            direction="backward",
        )
        fallback_total_mv = group["total_shares"] * group["close"]
    else:
        fallback_total_mv = pd.Series(np.nan, index=group.index)
    group = group.drop(columns=["total_shares", "effective_date"], errors="ignore")
    if valuation is not None:
        group = group.join(valuation, on="datetime")
    else:
        for column in VALUATION_COLUMNS:
            group[column] = np.nan
    for column in ("total_mv", "circ_mv"):
        if column not in group.columns:
            group[column] = np.nan
    # 湖侧市值缺失时才用东财股本 × 收盘价兜底
    group["total_mv"] = group["total_mv"].fillna(fallback_total_mv)
    return to_plum_frame(group)


def to_plum_frame(group: pd.DataFrame) -> pd.DataFrame:
    """→ plum 格式（timestamp index + object 列 + None 缺失）。"""
    frame = group.drop(columns=["symbol"]).copy()
    frame["timestamp"] = (
        pd.to_datetime(frame["datetime"]) - pd.Timedelta(days=1) + TS_OFFSET
    )
    frame = frame.drop(columns=["datetime"]).set_index("timestamp").sort_index()
    frame = frame.reindex(columns=[c for c in OUTPUT_COLUMNS if c in frame.columns])
    for column in frame.columns:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def _turn_expected(source: pd.DataFrame, column: str) -> int:
    """湖侧该换手率列的非空行数（turn 与 turn_calc 各自对拍）。"""
    if column not in source.columns:
        return 0
    return int(source[column].notna().sum())


def _written_columns(path: Path) -> set[str]:
    """读取 parquet 列名（只读 footer，不加载数据）。"""
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _check_symbol(
    path: Path, source: pd.DataFrame, valuation: pd.DataFrame | None
) -> list[str]:
    """单标的抽检，返回异常说明列表（空 = 通过）。

    副本缺少待检列时跳过该项（兼容最小化测试/旧副本），
    但不静默通过金额与覆盖度校验。
    """
    problems: list[str] = []
    written_columns = _written_columns(path)
    # amount：湖里非空且为正的行，副本必须复现同样的正值
    keeps = source["amount"].notna() & (source["amount"] > 0)
    if keeps.sum() > 0 and "amount" in written_columns:
        lake_sum = float(source.loc[keeps, "amount"].sum())
        written = pd.read_parquet(path, columns=["amount"])
        store_sum = float(
            pd.to_numeric(written["amount"], errors="coerce").fillna(0.0).sum()
        )
        if abs(store_sum - lake_sum) > max(1.0, lake_sum * 1e-6):
            problems.append(f"成交额合计 湖 {lake_sum:.0f} / 副本 {store_sum:.0f}")
    # turn / turn_calc：湖侧该列有值的行，副本对应列必须有值
    turn_columns = [
        column for column in ("turn", TURN_CALC_COLUMN) if column in written_columns
    ]
    if turn_columns:
        written_turn = pd.read_parquet(path, columns=turn_columns)
        for column in turn_columns:
            expected_turn = _turn_expected(source, column)
            actual_turn = int(written_turn[column].notna().sum())
            if actual_turn < expected_turn:
                problems.append(
                    f"{column} 覆盖 湖 {expected_turn} 行 / 副本 {actual_turn} 行"
                )
    # total_mv / circ_mv：湖侧有值的行，副本对应列必须有值
    # （只比较该标的实际有 K 线的日期：估值层有而行情层无的日期不算丢值）
    market_columns = [
        column for column in ("total_mv", "circ_mv") if column in written_columns
    ]
    if market_columns:
        written_mv = pd.read_parquet(path, columns=market_columns)
        for column in market_columns:
            expected_mv = 0
            if valuation is not None and column in valuation.columns:
                in_window = valuation.index.isin(pd.DatetimeIndex(source["datetime"]))
                expected_mv = int(valuation.loc[in_window, column].notna().sum())
            actual_mv = int(written_mv[column].notna().sum())
            if actual_mv < expected_mv:
                problems.append(
                    f"{column} 覆盖 湖 {expected_mv} 行 / 副本 {actual_mv} 行"
                )
    return problems


def verify_sample(
    out_dir: Path,
    bars: pd.DataFrame,
    valuation_map: dict[str, pd.DataFrame],
    sample_every: int,
) -> int:
    """抽检写出的 parquet 与内存帧是否一致，防止静默丢值。

    历史缺陷：副本的 amount 有 14 万行（2020 年 15.4%）在湖里有值却被写成 0，
    circ_mv 则从未从湖里取（全空），turn 曾被忽略。都不报错，只污染回测。
    """
    symbols = sorted(bars["symbol"].astype(str).unique())
    checked = bad = 0
    for symbol in symbols[::sample_every]:
        code, exchange = symbol.split(".")
        path = out_dir / exchange / f"{code}.parquet"
        if not path.exists():
            bad += 1
            logger.warning("副本缺少标的文件：%s", symbol)
            continue
        source = bars[bars["symbol"].astype(str) == symbol]
        checked += 1
        problems = _check_symbol(path, source, valuation_map.get(symbol))
        if problems:
            bad += 1
            logger.warning("副本与湖不一致：%s %s", symbol, "；".join(problems))
    logger.info("抽检：%d 只标的，异常 %d 只", checked, bad)
    return bad


def write_symbol(frame: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    """写出单标的 parquet（SH/SZ 分目录，代码不带交易所后缀）。"""
    code, exchange = symbol.split(".")
    directory = out_dir / exchange
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{code}.parquet", compression="snappy")


def load_index_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """读取数据湖指数日线；没有数据时返回空表（不阻断个股重建）。"""
    root = resolve_data_root() / "clean" / INDEX_DATASET
    paths = sorted(root.glob("year=*/*.parquet"))
    if not paths:
        logger.warning("未找到指数数据：%s（先运行 market-data 的 sync index）", root)
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame[(frame["datetime"] >= start) & (frame["datetime"] <= end)]
    logger.info("指数：%d 行，%d 个", len(frame), frame["symbol"].nunique())
    return frame.sort_values(["symbol", "datetime"])


def build_index_frame(group: pd.DataFrame) -> pd.DataFrame:
    """指数 → plum 格式：无复权/停牌/估值概念，对应列填默认值。"""
    frame = group.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["adj_factor"] = 1.0
    frame["is_st"] = False
    frame["is_delisted"] = False
    for column, value in DEFAULT_FILL.items():
        if column in frame.columns:
            frame[column] = frame[column].fillna(value)
    return to_plum_frame(frame)


def _verify_enabled() -> bool:
    """抽检开关：设置 PLUM_BUILD_VERIFY=0 可跳过（大批量重建时可关）。"""
    return os.environ.get("PLUM_BUILD_VERIFY", "1").strip() not in {
        "0",
        "false",
        "False",
    }


def write_indexes(frame: pd.DataFrame, out_dir: Path) -> int:
    """按指数写出 `IDX/<代码>.parquet`，返回写出个数。"""
    if frame.empty:
        return 0
    directory = out_dir / INDEX_BUCKET
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for symbol, group in frame.groupby("symbol", sort=False, observed=True):
        code = str(symbol).split(".")[0]
        index_frame = build_index_frame(group)
        if index_frame.empty:
            continue
        index_frame.to_parquet(directory / f"{code}.parquet", compression="snappy")
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从数据湖重建 plum 回测数据集")
    parser.add_argument(
        "--out-dir", default=str(PROJECT_ROOT / "data" / "market" / "1d")
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=date.today().isoformat(), help="默认截至当天")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-index", action="store_true", help="不重建指数（基准）目录"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    out_dir = Path(args.out_dir)

    bars = read_year_partitions(DAILY_DATASET)
    bars["datetime"] = pd.to_datetime(bars["datetime"]).astype("datetime64[ns]")
    bars = bars[(bars["datetime"] >= start) & (bars["datetime"] <= end)]
    bars = bars.sort_values(["symbol", "datetime"])
    symbols = sorted(bars["symbol"].astype(str).unique())
    logger.info(
        "区间 %s ~ %s：%d 行，%d 只标的", args.start, args.end, len(bars), len(symbols)
    )

    equity_map = load_equity_shares()
    valuation_map = load_valuation_map()
    if args.dry_run:
        logger.info("dry-run：将输出 %d 只标的到 %s", len(symbols), out_dir)
        return 0

    written = no_equity = skipped = 0
    for symbol, group in bars.groupby("symbol", sort=False, observed=True):
        key = str(symbol)
        equity_group = equity_map.get(key)
        if equity_group is None:
            no_equity += 1
        frame = build_symbol_frame(key, group, equity_group, valuation_map.get(key))
        if frame.empty:
            skipped += 1
            continue
        write_symbol(frame, key, out_dir)
        written += 1
        if written % 500 == 0:
            logger.info("进度 %d/%d", written, len(symbols))
    logger.info(
        "完成：写出 %d 只标的到 %s（无股本 %d 只，价格全缺 %d 只）",
        written,
        out_dir,
        no_equity,
        skipped,
    )
    if not args.skip_index:
        indexes = write_indexes(load_index_frame(start, end), out_dir)
        logger.info("指数写出 %d 个到 %s", indexes, out_dir / INDEX_BUCKET)
    if _verify_enabled():
        bad = verify_sample(out_dir, bars, valuation_map, VERIFY_SAMPLE_EVERY)
        if bad:
            logger.error(
                "副本一致性抽检未通过：%d 只标的存在丢值，请检查后再用于回测", bad
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
