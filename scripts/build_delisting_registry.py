"""从数据湖构建退市注册表（DelistingRegistry）——回测的「生死簿」。

背景
----
回测引擎的退市保护依赖 `data/registry/delisting_registry.json`，但该文件从未
生成，`is_delisted()` 恒返回 False，导致：

1. **僵尸持仓**：退市股永久滞留持仓，价格冻结在最后收盘价，净值曲线走平。
   v2 回测的 R1 轮因此「卡死」——2022-06 买入的退市股把资金锁到 2026-09，
   4 年无法交易，净值从 100 万跌到 37.7 万后再无变化。
2. **幽灵买入**：已退市标的仍可能被选入并买入。

数据来源
--------
- `clean/instruments`：`ipo_date`（首个交易日）、`out_date`（退市日）、`is_delisted`
- `clean/daily`：各标的最后一个有行情的交易日（用于交叉校验）

输出
----
`data/registry/delisting_registry.json`，键为 plum 代码格式（如 `SH.603157`），
字段与 `DelistingRegistry._load()` 契约一致：
`symbol / first_trading_day / last_trading_day / delisting_date / reason / is_active`

语义约定
--------
- `last_trading_day` 取 `out_date`（官方退市日，通常晚于最后交易日数日）。
  退市后引擎在首个不早于该日的交易日按最后收盘价清算，释放资金。
- 非退市标的 `is_active=True`，不写 `last_trading_day`——「尚未上市」由
  `first_trading_day` 表达，引擎在买入时按日期拦截。

用法
----
    uv run python scripts/build_delisting_registry.py
    uv run python scripts/build_delisting_registry.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("build-delisting-registry")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

ENV_KEY = "QDATA_ROOT"
ENV_FILE = PROJECT_ROOT.parent / "market-data" / ".env"
INSTRUMENTS_DATASET = "instruments"
DAILY_DATASET = "daily"
STOCK_SECURITY_TYPE = 1
OUTPUT_PATH = PROJECT_ROOT / "data" / "registry" / "delisting_registry.json"
DATE_FORMAT = "%Y-%m-%d"


def resolve_data_root() -> Path:
    """解析数据湖根目录：优先环境变量，其次 market-data/.env（不硬编码路径）。"""
    value = os.environ.get(ENV_KEY, "").strip()
    if not value and ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
        value = os.environ.get(ENV_KEY, "").strip()
    if not value:
        raise RuntimeError(f"未找到 {ENV_KEY}（环境变量或 {ENV_FILE}）")
    return Path(value).expanduser()


def to_plum_symbol(symbol: str) -> str:
    """数据湖代码 `603157.SH` → plum 代码 `SH.603157`。"""
    code, exchange = symbol.split(".")
    return f"{exchange}.{code}"


def load_instruments() -> pd.DataFrame:
    """读取标的元数据（含 ipo_date / out_date / is_delisted）。"""
    root = resolve_data_root() / "clean" / INSTRUMENTS_DATASET
    paths = sorted(root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 {INSTRUMENTS_DATASET}：{root}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame[frame["security_type"] == STOCK_SECURITY_TYPE].copy()
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce")
    logger.info("instruments：%d 只股票标的（含退市股）", len(frame))
    return frame


def load_last_trading_days() -> dict[str, pd.Timestamp]:
    """读取各标的最后一个有行情的交易日（用于交叉校验与缺失兜底）。"""
    root = resolve_data_root() / "clean" / DAILY_DATASET
    paths = sorted(root.glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 {DAILY_DATASET}：{root}")
    last: dict[str, pd.Timestamp] = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=["symbol", "datetime"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        group = frame.groupby("symbol", sort=False)["datetime"].max()
        for symbol, value in group.items():
            key = str(symbol)
            if key not in last or value > last[key]:
                last[key] = value
    logger.info("daily：%d 只有行情记录", len(last))
    return last


def format_date(value: pd.Timestamp | None) -> str | None:
    """Timestamp → `YYYY-MM-DD`；空值返回 None。"""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime(DATE_FORMAT)


def build_record(row: pd.Series, last_bar: pd.Timestamp | None) -> tuple[str, dict]:
    """单标的 → 注册表记录（键为 plum 代码）。"""
    symbol = to_plum_symbol(str(row["symbol"]))
    is_delisted = bool(row["is_delisted"])
    out_date = row["out_date"] if not pd.isna(row["out_date"]) else None
    # 退市日缺失时（脏数据）用最后一个交易日兜底，宁可早触发不可漏触发
    end_date = out_date if out_date is not None else last_bar
    record: dict = {
        "symbol": symbol,
        "first_trading_day": format_date(row["ipo_date"]),
        "is_active": not is_delisted,
    }
    if is_delisted and end_date is not None:
        record["last_trading_day"] = format_date(end_date)
        record["delisting_date"] = format_date(end_date)
        record["reason"] = "退市"
    return symbol, record


def write_registry(records: dict[str, dict], output_path: Path) -> None:
    """原子写入 JSON（引擎按该格式加载）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 plum 退市注册表")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instruments = load_instruments()
    last_days = load_last_trading_days()

    records: dict[str, dict] = {}
    missing_end = 0
    gap_days: list[float] = []
    for _, row in instruments.iterrows():
        last_bar = last_days.get(str(row["symbol"]))
        symbol, record = build_record(row, last_bar)
        records[symbol] = record
        if not record["is_active"] and "last_trading_day" not in record:
            missing_end += 1
        elif not record["is_active"] and last_bar is not None:
            end = datetime.strptime(record["last_trading_day"], DATE_FORMAT)
            gap_days.append((end - last_bar).days)

    delisted = sum(1 for record in records.values() if not record["is_active"])
    logger.info(
        "注册表：%d 条（退市 %d，缺失退市日 %d，退市日距最后交易日中位 %s 天）",
        len(records),
        delisted,
        missing_end,
        pd.Series(gap_days).median() if gap_days else "NA",
    )
    if args.dry_run:
        logger.info("dry-run：未写盘")
        return 0
    write_registry(records, Path(args.output))
    logger.info("已写入 %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
