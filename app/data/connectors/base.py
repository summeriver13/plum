"""数据连接器基类 — 定义数据源统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from app.data.schema import Bar, Instrument, Quote

RawDataQMT = dict[str, pd.DataFrame]
RawDataList = list[dict[str, Any]]
RawData = RawDataQMT | RawDataList


@runtime_checkable
class DataSource(Protocol):
    """数据源协议 — 定义必需的数据源行为。"""

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str = "1d",
    ) -> list[Bar]: ...

    def get_quote(self, symbol: str) -> Quote: ...

    def get_instrument(self, symbol: str) -> Instrument: ...


class BaseConnector(ABC):
    """数据连接器基类 — 提供通用的 bar 解析逻辑。"""

    @abstractmethod
    def _fetch_raw_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str,
    ) -> RawData:
        """从数据源获取原始数据。子类必须实现。"""

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str = "1d",
    ) -> list[Bar]:
        raw = self._fetch_raw_bars(symbol, start, end, period)
        return self._parse_bars(raw, symbol)

    @staticmethod
    def _parse_bars(raw: RawData, symbol: str) -> list[Bar]:
        """从原始数据解析 bar（QMT dict 格式或 list 格式）。"""
        if isinstance(raw, dict) and "close" in raw:
            return BaseConnector._parse_qmt_format(raw, symbol)
        if isinstance(raw, list):
            return BaseConnector._parse_list_format(raw, symbol)
        return []

    @staticmethod
    def _parse_qmt_format(raw: dict[str, pd.DataFrame], symbol: str) -> list[Bar]:
        """解析 MiniQMT xtdata 格式：{field: DataFrame}。"""
        close_df = raw.get("close")
        if close_df is None or close_df.empty:
            return []

        s_close = close_df.get(symbol)
        if s_close is None:
            return []

        s_open = raw.get("open", pd.DataFrame()).get(symbol)
        s_high = raw.get("high", pd.DataFrame()).get(symbol)
        s_low = raw.get("low", pd.DataFrame()).get(symbol)
        s_vol = raw.get("volume", pd.DataFrame()).get(symbol)
        s_amount = raw.get("amount", pd.DataFrame()).get(symbol)

        bars = []
        for dt, val_close in s_close.items():
            if pd.isna(val_close):
                continue

            ts = pd.to_datetime(dt)
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts,
                    open=Decimal(str(s_open[dt] if s_open is not None else 0)),
                    high=Decimal(str(s_high[dt] if s_high is not None else 0)),
                    low=Decimal(str(s_low[dt] if s_low is not None else 0)),
                    close=Decimal(str(val_close)),
                    volume=int(s_vol[dt] if s_vol is not None else 0),
                    amount=Decimal(str(s_amount[dt] if s_amount is not None else 0)),
                ),
            )
        return bars

    @staticmethod
    def _parse_list_format(raw: list[dict[str, Any]], symbol: str) -> list[Bar]:
        """解析 list[dict] 格式（如 BaoStock）。"""
        return [Bar.from_dict(row, symbol) for row in raw]
