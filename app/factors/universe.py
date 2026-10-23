"""股票池管理 — 带幸存者偏差保护的股票池。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config.paths import FACTOR_UNIVERSE_CACHE


@dataclass
class UniverseRecord:
    """股票在某一天的状态记录。"""

    code: str
    code_name: str
    status: str
    date: datetime

    @property
    def is_active(self) -> bool:
        return self.status == "1"


@dataclass
class Universe:
    """带历史追踪的股票池，用于幸存者偏差保护。"""

    date: datetime
    symbols: set[str]
    records: list[UniverseRecord] = field(default_factory=list)

    def contains(self, symbol: str) -> bool:
        return symbol in self.symbols


class UniverseManager:
    """管理股票池快照和查询。

    核心功能：通过追踪每个历史日期存在的股票，防止幸存者偏差。
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        storage_root: Path | str | None = None,
    ) -> None:
        if cache_dir is None:
            if storage_root is None:
                storage_root = FACTOR_UNIVERSE_CACHE
            else:
                storage_root = Path(storage_root)
            self.cache_dir = Path(storage_root) / "factors" / "universe"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: dict[datetime, Universe] = {}

    def add_snapshot(self, universe: Universe) -> None:
        """添加股票池快照。"""
        self._snapshots[universe.date] = universe

    def get_universe(self, date: datetime) -> Universe | None:
        """获取给定日期或之前最近日期的股票池快照。"""
        if not self._snapshots:
            return None

        dates = sorted(self._snapshots.keys())
        for d in reversed(dates):
            if d <= date:
                return self._snapshots[d]
        return None

    def get_active_symbols(self, date: datetime) -> set[str]:
        """获取指定日期的活跃股票集合。"""
        universe = self.get_universe(date)
        if universe is None:
            return set()
        return universe.symbols

    def save(self) -> None:
        """保存所有快照到缓存。"""
        for date, universe in self._snapshots.items():
            date_str = date.strftime("%Y%m%d")
            path = self.cache_dir / f"universe_{date_str}.parquet"

            records = [
                {
                    "code": r.code,
                    "code_name": r.code_name,
                    "status": r.status,
                    "date": r.date,
                }
                for r in universe.records
            ]

            df = pd.DataFrame(records)
            df.to_parquet(path, compression="snappy")

    def load(self, start_date: datetime, end_date: datetime) -> None:
        """从缓存加载股票池快照。"""
        for path in self.cache_dir.glob("universe_*.parquet"):
            date_str = path.stem.replace("universe_", "")
            date = datetime.strptime(date_str, "%Y%m%d")

            if start_date <= date <= end_date:
                df = pd.read_parquet(path)
                records = [
                    UniverseRecord(
                        code=row["code"],
                        code_name=row["code_name"],
                        status=row["status"],
                        date=row["date"],
                    )
                    for _, row in df.iterrows()
                ]
                symbols = {r.code for r in records if r.is_active}
                self._snapshots[date] = Universe(
                    date=date, symbols=symbols, records=records
                )

    def filter_tradable(
        self,
        symbols: list[str],
        capital: int,
        experience_years: int = 0,
        exclude_st: bool = True,
    ) -> list[str]:
        """按 A 股交易资格过滤股票。

        A 股准入规则：
        - 科创板 (688): 需 50 万资金
        - 北交所 (BJ): 需 50 万资金
        - 创业板 (300): 需 10 万资金 + 2 年经验
        - ST 股票：默认排除（防御性策略）
        """
        result: list[str] = []
        for symbol in symbols:
            code = symbol.split(".")[0] if "." in symbol else symbol
            if code.startswith("688") and capital < 500_000:
                continue
            if symbol.endswith(".BJ") and capital < 500_000:
                continue
            if code.startswith("300") and (experience_years == 0 or capital < 100_000):
                continue
            if exclude_st and ("ST" in symbol or "*ST" in symbol):
                continue
            result.append(symbol)
        return result
