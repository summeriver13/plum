"""BaoStock 数据连接器。

封装 BaoStock API，提供 K 线、股票列表、复权因子等数据获取能力。
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime
from typing import Any

import baostock as bs
import pandas as pd

from app.data.connectors.base import BaseConnector
from app.data.schema import Instrument, Quote

logger = logging.getLogger(__name__)

SOCKET_TIMEOUT_S: int | float = 30

_FREQ_MAP = {
    "1d": "d",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "1m": "1",
}


def normalize_symbol(symbol: str) -> str:
    """转换为 BaoStock 格式（sh.600000 或 sz.000001）。"""
    if "." in symbol:
        parts = symbol.split(".")
        if len(parts) == 2:
            left, right = parts
            if left.lower() in ("sh", "sz"):
                return f"{left.lower()}.{right}"
            if right.lower() in ("sh", "sz"):
                return f"{right.lower()}.{left}"
        raise ValueError(f"无效的股票代码格式: {symbol}")

    if symbol.startswith("000"):
        return f"sh.{symbol}"
    if symbol.startswith("399"):
        return f"sz.{symbol}"
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"
    if symbol.startswith(("510", "511", "512", "513", "515", "588")):
        return f"sh.{symbol}"
    if symbol.startswith(("159", "161")):
        return f"sz.{symbol}"

    raise ValueError(f"未知的股票代码格式: {symbol}")


def is_index_code(symbol: str) -> bool:
    """检查是否为指数代码。"""
    code = symbol.rsplit(".", 1)[-1] if "." in symbol else symbol
    return code.startswith("000") or code.startswith("399")


def is_etf_code(symbol: str) -> bool:
    """检查是否为 ETF 代码。"""
    code = symbol.rsplit(".", 1)[-1] if "." in symbol else symbol
    return code.startswith(
        ("510", "511", "512", "513", "515", "588")
    ) or code.startswith("159")


class BaoStockClient:
    """BaoStock 登录/登出生命周期管理（通过上下文管理器）。"""

    def __init__(self) -> None:
        self._logged_in = False

    def __enter__(self) -> BaoStockClient:
        if not self._logged_in:
            bs.login()
            self._logged_in = True
            socket.setdefaulttimeout(SOCKET_TIMEOUT_S)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._logged_in:
            try:
                bs.logout()
            except Exception:
                logger.debug("BaoStock 登出失败", exc_info=True)
        self._logged_in = False

    @property
    def is_connected(self) -> bool:
        return self._logged_in


class BaoStockConnector(BaseConnector):
    """BaoStock 数据连接器。

    提供 K 线、股票列表、复权因子等数据获取能力。
    """

    def __init__(self) -> None:
        self._client = BaoStockClient()

    def __enter__(self) -> BaoStockConnector:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.__exit__(exc_type, exc_val, exc_tb)

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError("BaoStock 不支持实时行情")

    def get_instrument(self, symbol: str) -> Instrument:
        raise NotImplementedError("BaoStock 不支持证券详情查询")

    def _fetch_raw_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str,
    ) -> list[dict[str, Any]]:
        return self.fetch_kline(symbol, start, end, period)

    def fetch_kline(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str = "1d",
    ) -> list[dict[str, Any]]:
        """获取 K 线数据。"""
        bs_code = normalize_symbol(symbol)
        bs_freq = _FREQ_MAP.get(period, "d")

        if period == "1d":
            fields = (
                "date,open,high,low,close,volume,amount,"
                "isST,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
            )
        else:
            fields = "date,time,open,high,low,close,volume,amount"

        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency=bs_freq,
            adjustflag="3",
        )

        data: list[dict[str, Any]] = []
        while rs is not None and rs.error_code == "0" and rs.next():
            row: dict[str, Any] = dict(zip(rs.fields, rs.get_row_data(), strict=True))
            row["adj_factor"] = 1.0

            for _field in ("open", "high", "low", "close", "volume", "amount"):
                val = row.get(_field)
                if isinstance(val, str):
                    try:
                        row[_field] = float(val) if val.strip() else 0.0
                    except ValueError:
                        logger.debug(
                            "数据字段解析失败，使用默认值 0.0: %s",
                            _field,
                            exc_info=True,
                        )
                        row[_field] = 0.0

            time_val = row.get("time")
            if time_val and str(time_val).strip():
                row["timestamp"] = datetime.strptime(str(time_val)[:14], "%Y%m%d%H%M%S")
            else:
                row["timestamp"] = datetime.strptime(str(row["date"]), "%Y-%m-%d")

            data.append(row)
        return data

    def query_all_stock(self, date: datetime | None = None) -> list[StockInfo]:
        """查询全部股票。"""
        if date is None:
            date = datetime.now()
        day_str = date.strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=day_str)

        stocks: list[StockInfo] = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            stocks.append(
                StockInfo(
                    code=row[0],
                    trade_status=row[1],
                    code_name=row[2],
                )
            )
        return stocks

    def query_adjust_factor(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """查询复权因子。"""
        bs_code = normalize_symbol(symbol)
        try:
            rs = bs.query_adjust_factor(
                bs_code,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
        except Exception:
            logger.warning("BaoStock 复权因子查询失败", exc_info=True)
            return pd.DataFrame()

        data: list[dict[str, object]] = []
        if rs is None or rs.error_code != "0":
            return pd.DataFrame()
        while rs is not None and rs.next():
            row: dict[str, object] = dict(
                zip(rs.fields, rs.get_row_data(), strict=True)
            )
            data.append(row)

        if not data:
            return pd.DataFrame(columns=["date", "code", "foreAdjustFactor"])
        return pd.DataFrame(data)


class StockInfo:
    """股票基本信息。"""

    def __init__(self, code: str, trade_status: str, code_name: str) -> None:
        self.code = code
        self.trade_status = trade_status
        self.code_name = code_name

    @property
    def is_active(self) -> bool:
        return self.trade_status == "1"

    @property
    def is_suspended(self) -> bool:
        return self.trade_status == "2"

    @property
    def is_delisted(self) -> bool:
        return self.trade_status == "3"
