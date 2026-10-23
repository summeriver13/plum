"""财务数据持久化存储。

按股票分片存储 Parquet 文件，配套 meta.json 记录同步状态。
复用现有 app/data/store/ 的存储模式。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.paths import FINANCIAL_PATH

logger = logging.getLogger(__name__)


class FinancialStore:
    """财务数据 Parquet 存储。

    目录结构:
        {root}/
        ├── {symbol}/
        │   ├── financial.parquet
        │   └── meta.json
        └── ...

    Usage:
        store = FinancialStore(root="data/financial")
        store.write("600035.SH", df)
        df = store.read("600035.SH", start="2024", end="2025")
    """

    def __init__(self, root: str | Path = str(FINANCIAL_PATH)) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # -- 写入 --

    def write(
        self,
        symbol: str,
        df: pd.DataFrame,
        meta_update: dict[str, Any] | None = None,
    ) -> None:
        """写入单只股票的财务数据。"""
        if df.empty:
            logger.warning("%s 财务数据为空，跳过写入", symbol)
            return

        symbol_dir = self._root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = symbol_dir / "financial.parquet"
        df.to_parquet(parquet_path, index=True)
        self._update_meta(symbol, meta_update or {}, df)

        logger.debug(
            "%s 已写入: %d 行, %d 列",
            symbol,
            len(df),
            len(df.columns),
        )

    def _update_meta(
        self,
        symbol: str,
        extra: dict[str, Any],
        df: pd.DataFrame,
    ) -> None:
        """更新 meta.json。"""
        meta_path = self._root / symbol / "meta.json"
        current = self.get_meta(symbol)

        periods = (
            df.index.tolist()
            if isinstance(df.index, pd.DatetimeIndex)
            else [str(i) for i in df.index]
        )

        current.update(
            {
                "symbol": symbol,
                "last_sync": datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns),
                "report_periods": periods,
            }
        )
        current.update(extra)

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2, default=str)

    # -- 读取 --

    def read(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读取单只股票的财务数据。"""
        parquet_path = self._root / symbol / "financial.parquet"
        if not parquet_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(parquet_path)
        if df.empty:
            return df

        if start and end:
            df = df.loc[start:end]
        elif start:
            df = df.loc[start:]
        elif end:
            df = df.loc[:end]

        return df

    # -- 批量读取 --

    def read_many(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """批量读取多只股票的财务数据。"""
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.read(symbol, start, end)
            if not df.empty:
                result[symbol] = df
        return result

    # -- Meta 管理 --

    def get_meta(self, symbol: str) -> dict[str, Any]:
        """获取股票的同步元数据。"""
        meta_path = self._root / symbol / "meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def list_symbols(self) -> list[str]:
        """列出所有已存储的股票代码。"""
        if not self._root.exists():
            return []
        return [
            d.name
            for d in self._root.iterdir()
            if d.is_dir() and (d / "financial.parquet").exists()
        ]

    def get_last_sync(self, symbol: str) -> str | None:
        """获取最后同步时间。"""
        meta = self.get_meta(symbol)
        return meta.get("last_sync")
