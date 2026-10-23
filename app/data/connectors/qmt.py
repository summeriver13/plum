"""QMT (MiniQMT) 数据连接器。

通过 xtquant 库连接本地 MiniQMT 进程，提供 K 线、股票列表、复权因子等数据。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from app.data.connectors.base import BaseConnector
from app.data.schema import Instrument, Quote

logger = logging.getLogger(__name__)

_xtdata: Any = None

_PERIOD_MAP: dict[str, str] = {
    "1d": "1d",
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
}


def _get_xtdata() -> Any:
    global _xtdata
    if _xtdata is None:
        try:
            from xtquant import xtdata
        except ImportError as e:
            raise ImportError(
                "MiniQMT (xtquant) 未安装。请运行: pip install xtquant"
            ) from e
        _xtdata = xtdata
    return _xtdata


def to_qmt(symbol: str) -> str:
    """标准化股票代码为 QMT 格式 (600519.SH / 000001.SZ)。

    支持输入格式：
    - QMT 格式: 600519.SH → 600519.SH（不变）
    - Plum 格式: sh.600519 → 600519.SH
    - 纯代码: 600519 → 600519.SH（自动推断交易所）
    """
    if "." in symbol:
        parts = symbol.split(".", 1)
        # 已经是 QMT 格式 (600519.SH)
        if parts[1].upper() in ("SH", "SZ", "BJ"):
            return f"{parts[0]}.{parts[1].upper()}"
        # Plum 格式 (sh.600519)
        exchange, code = parts
        exchange = exchange.upper()
        if exchange in ("SH", "SZ", "BJ"):
            return f"{code}.{exchange}"
    # 纯代码，自动推断
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    if symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    return f"{symbol}.SH"


def from_qmt(qmt_symbol: str) -> str:
    """QMT 格式转内部格式（内部已统一使用 QMT 格式，直接返回）。"""
    return qmt_symbol


class QMTClient:
    """MiniQMT 连接管理器。"""

    def __init__(self) -> None:
        self._connected = False

    def __enter__(self) -> QMTClient:
        xtdata = _get_xtdata()
        xtdata.get_trading_dates("SH")
        self._connected = True
        return self

    def __exit__(self, *args: object) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


class QMTConnector(BaseConnector):
    """QMT 数据连接器。"""

    _KLINE_FIELDS = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "preClose",
        "suspendFlag",
    ]

    def __init__(self) -> None:
        self._client = QMTClient()
        self._xtdata = _get_xtdata()
        import os

        qmt_path = os.environ.get("PLUM_QMT_PATH", "")
        if qmt_path:
            self._xtdata.data_dir = os.path.join(qmt_path, "datadir")

    def __enter__(self) -> QMTConnector:
        self._client.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._client.__exit__(*args)

    def get_quote(self, symbol: str) -> Quote:
        qmt_sym = to_qmt(symbol)
        ticks = self._xtdata.get_full_tick([qmt_sym])
        tick = ticks.get(qmt_sym)
        if not tick:
            raise RuntimeError(f"获取 {symbol} 行情失败，MiniQMT 是否运行？")
        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            last=Decimal(str(tick.get("lastPrice", 0))),
            open=Decimal(str(tick.get("openPrice", 0))),
            high=Decimal(str(tick.get("highPrice", 0))),
            low=Decimal(str(tick.get("lowPrice", 0))),
            volume=int(float(tick.get("volume", 0))),
            amount=Decimal(str(tick.get("turnover", 0))),
            bid1=Decimal(str(tick.get("bid1", 0))),
            ask1=Decimal(str(tick.get("ask1", 0))),
        )

    def get_instrument(self, symbol: str) -> Instrument:
        qmt_sym = to_qmt(symbol)
        detail = self._xtdata.get_instrument_detail(qmt_sym)
        if not detail:
            raise RuntimeError(f"获取 {symbol} 证券信息失败")
        return Instrument.from_detail(detail, symbol)

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
        qmt_symbol = to_qmt(symbol)
        qmt_period = _PERIOD_MAP.get(period, period)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        data = self._xtdata.get_market_data_ex(
            field_list=self._KLINE_FIELDS,
            stock_list=[qmt_symbol],
            period=qmt_period,
            start_time=start_str,
            end_time=end_str,
            dividend_type="none",
        )

        if not data or qmt_symbol not in data:
            return []

        df = data[qmt_symbol]
        if df is None or df.empty:
            return []

        bars: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                ts_raw = row["time"]
                if isinstance(ts_raw, (int, float)):
                    ts_val = int(ts_raw)
                    if ts_val > 10**10:
                        ts_val = ts_val // 1000
                    ts = datetime.fromtimestamp(ts_val, tz=UTC).replace(tzinfo=None)
                elif isinstance(ts_raw, pd.Timestamp):
                    ts = ts_raw.to_pydatetime()
                else:
                    ts = pd.Timestamp(ts_raw).to_pydatetime()

                suspend_flag = int(row.get("suspendFlag", 0))
                is_st = suspend_flag == 1

                bars.append(
                    {
                        "timestamp": ts,
                        "open": float(row["open"]) if pd.notna(row["open"]) else 0.0,
                        "high": float(row["high"]) if pd.notna(row["high"]) else 0.0,
                        "low": float(row["low"]) if pd.notna(row["low"]) else 0.0,
                        "close": float(row["close"]) if pd.notna(row["close"]) else 0.0,
                        "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                        "amount": float(row["amount"])
                        if pd.notna(row["amount"])
                        else 0.0,
                        "preClose": float(row.get("preClose", 0))
                        if pd.notna(row.get("preClose", 0))
                        else 0.0,
                        "peTTM": None,
                        "pbMRQ": None,
                        "psTTM": None,
                        "pcfNcfTTM": None,
                        "isST": is_st,
                    }
                )
            except Exception:
                logger.warning("Bar 解析失败，跳过该条记录", exc_info=True)
                continue
        return bars

    def query_all_stock(self, date: datetime | None = None) -> list[Any]:
        """查询全部 A 股股票。"""
        raw = self._xtdata.get_stock_list_in_sector("沪深A股")
        if not raw:
            return []
        stocks = []
        for qmt_code in raw:
            detail = self._xtdata.get_instrument_detail(qmt_code)
            if not detail:
                continue
            instr_status = int(detail.get("InstrumentStatus", 0))
            status = "1" if instr_status == 0 else "2"
            name = str(detail.get("InstrumentName", ""))
            code = from_qmt(qmt_code)
            stocks.append(StockInfo(code=code, trade_status=status, code_name=name))
        return stocks

    def query_adjust_factor(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """查询复权因子。"""
        qmt_symbol = to_qmt(symbol)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        raw = self._xtdata.get_divid_factors(qmt_symbol, start_str, end_str)
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            return pd.DataFrame(columns=["date", "code", "foreAdjustFactor"])

        rows = []
        for idx, row in raw.iterrows():
            dr = row.get("dr")
            if dr is None:
                continue
            ts_raw = row.get("time")
            if isinstance(ts_raw, (int, float)):
                ts_val = int(ts_raw)
                if ts_val > 10**10:
                    ts_val = ts_val // 1000
                date_str = datetime.utcfromtimestamp(ts_val).strftime("%Y-%m-%d")
            else:
                date_str = str(idx)
            rows.append(
                {"date": date_str, "code": symbol, "foreAdjustFactor": float(dr)}
            )

        return pd.DataFrame(rows)


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
