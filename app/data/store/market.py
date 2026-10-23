"""K 线数据存储和检索的 Catalog 管理器。

特性：
- 自动从网络获取（如果数据未缓存）
- 增量更新（仅获取缺失的日期范围）
- 因子存储与 TTL 跟踪
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from app.config.paths import MARKET_PATH
from app.data.schema import BAR_SCHEMA_VERSION, Bar
from app.data.store.base_provider import _BaseCatalog, symbol_to_exchange_code

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.data.connectors.base import BaseConnector


def _opt_decimal(val: Any) -> Decimal | None:
    """可选 Decimal：空/非有限（NaN、±inf）返回 None。"""
    if val is None:
        return None
    try:
        value = Decimal(str(val))
    except Exception:
        logger.debug("Decimal 转换失败，返回 None", exc_info=True)
        return None
    return value if value.is_finite() else None


def _opt_text(val: Any) -> str | None:
    """可选字符串：空/NaN 返回 None（parquet 缺失列读出为 float NaN）。"""
    if val is None:
        return None
    text = str(val).strip()
    return None if not text or text.lower() in {"nan", "none", "null"} else text


def _int(val: Any, default: int = 0) -> int:
    """容错转 int（空/非数值/NaN 返回 default）。"""
    try:
        return int(val)
    except (TypeError, ValueError):
        logger.debug("int 转换失败，返回 %s", default, exc_info=True)
        return default


class MarketCatalog(_BaseCatalog):
    """管理 bar 数据的 Parquet 存储，支持自动获取。

    数据组织方式：
        data/market/
            1d/SH/600519.parquet
            1d/SZ/000001.parquet
            ...
        data/market/meta.json
    """

    ROOT_NAME = ""

    def __init__(
        self,
        catalog_root: Path | str | None = None,
        adapter: BaseConnector | None = None,
    ) -> None:
        if catalog_root is None:
            try:
                from app.config.settings import get_config

                catalog_root = get_config().data.market_path
            except Exception:
                logger.debug("配置加载失败，使用默认 market 路径", exc_info=True)
                catalog_root = str(MARKET_PATH)  # fallback
        super().__init__(catalog_root)
        self._adapter = adapter
        self._bars_cache: dict[tuple[str, str, int | None, int | None], list[Bar]] = {}

    def clear_bars_cache(self) -> None:
        """清除 bar 缓存。"""
        self._bars_cache.clear()

    def set_adapter(self, adapter: BaseConnector) -> None:
        """设置数据适配器用于自动获取。"""
        self._adapter = adapter

    def _get_adapter(self) -> BaseConnector:
        if self._adapter is None:
            from app.data.connectors.baostock import BaoStockConnector

            self._adapter = BaoStockConnector()
            self._adapter.__enter__()
        return self._adapter

    def _symbol_path(self, symbol: str, period: str, *parts: str) -> Path:
        """获取股票和周期的 parquet 文件路径。

        分桶：period/exchange/code.parquet
        示例：1d/SH/600519.parquet

        Raises:
            TypeError: 传入额外路径段时——本实现只支持 ``period`` 单层分桶，
                与基类“多部件拼文件名”的约定不兼容，宁可显式报错也不静默忽略。
        """
        if parts:
            raise TypeError(
                "MarketCatalog._symbol_path 只接受 (symbol, period)，"
                f"收到额外部件: {parts}"
            )
        exchange, code = symbol_to_exchange_code(symbol)
        return Path(period) / exchange / f"{code}.parquet"

    def _read(self, path: Path) -> pd.DataFrame:
        """读取 DataFrame，支持旧路径格式回退。"""
        resolved = path if path.is_absolute() else self._storage.root.joinpath(path)
        if resolved.exists():
            return super()._read(resolved)

        parent = resolved.parent
        stem = resolved.stem
        exchange = parent.name

        old_path = parent.parent / f"{exchange}.{stem}.parquet"
        if old_path.exists():
            return super()._read(old_path)

        buggy_path = parent.parent / stem / f"{exchange}.parquet"
        if buggy_path.exists():
            return super()._read(buggy_path)

        return pd.DataFrame()

    def _bars_to_df(self, bars: list[Bar]) -> pd.DataFrame:
        """将 Bar 列表转换为 DataFrame。"""
        if not bars:
            return pd.DataFrame()
        records = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "amount": bar.amount,
                "pre_close": bar.pre_close,
                "adj_factor": bar.adj_factor,
                "is_st": bar.is_st,
                "pe_ttm": bar.pe_ttm,
                "pb_mrq": bar.pb_mrq,
                "ps_ttm": bar.ps_ttm,
                "pcf_ncf_ttm": bar.pcf_ncf_ttm,
                "total_mv": bar.total_mv,
                "circ_mv": bar.circ_mv,
                "is_delisted": bar.is_delisted,
                "total_equity": bar.total_equity,
                "np_atoopc_ttm": bar.np_atoopc_ttm,
                "np_atoopc_q": bar.np_atoopc_q,
                "np_yoy": bar.np_yoy,
            }
            for bar in bars
        ]
        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        df.attrs["schema_version"] = BAR_SCHEMA_VERSION
        return df

    def _df_to_bars(self, df: pd.DataFrame, symbol: str) -> list[Bar]:
        """将 DataFrame 转换为 Bar 列表，隐式兼容旧 schema。

        v0 兼容: 旧 Parquet 缺少 is_delisted 列 → 默认 False。
        未来版本兼容: 新增字段通过 .get() 提供默认值。
        """
        if df.empty:
            return []

        # 隐式版本检测: is_delisted 列是否存在
        _has_is_delisted = "is_delisted" in df.columns

        records = cast(list[dict[str, Any]], df.reset_index().to_dict("records"))
        bars = []
        for r in records:
            raw_ts = pd.Timestamp(r["timestamp"])
            if raw_ts is pd.NaT:
                raise ValueError(f"{symbol} 存在 NaT 时间戳，行情数据可能损坏")
            ts = raw_ts.to_pydatetime()
            if not isinstance(ts, datetime):
                raise ValueError(f"{symbol} 时间戳无法转换为 datetime: {raw_ts!r}")
            is_delisted = bool(r["is_delisted"]) if _has_is_delisted else False
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts,
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    volume=_int(r["volume"]),
                    amount=Decimal(str(r.get("amount", 0))),
                    pre_close=Decimal(str(r.get("pre_close", 0))),
                    adj_factor=Decimal(str(r.get("adj_factor", 1.0))),
                    is_st=bool(r.get("is_st", False)),
                    pe_ttm=_opt_decimal(r.get("pe_ttm")),
                    pb_mrq=_opt_decimal(r.get("pb_mrq")),
                    ps_ttm=_opt_decimal(r.get("ps_ttm")),
                    pcf_ncf_ttm=_opt_decimal(r.get("pcf_ncf_ttm")),
                    total_mv=_opt_decimal(r.get("total_mv")),
                    circ_mv=_opt_decimal(r.get("circ_mv")),
                    is_delisted=is_delisted,
                    total_equity=_opt_decimal(r.get("total_equity")),
                    np_atoopc_ttm=_opt_decimal(r.get("np_atoopc_ttm")),
                    np_atoopc_q=_opt_decimal(r.get("np_atoopc_q")),
                    np_yoy=_opt_decimal(r.get("np_yoy")),
                    turn=_opt_decimal(r.get("turn")),
                    turn_calc=_opt_decimal(r.get("turn_calc")),
                    limit_rate=_opt_decimal(r.get("limit_rate")),
                    limit_reason=_opt_text(r.get("limit_reason")),
                )
            )
        return bars

    # ─── 公开接口 ──────────────────────────────────────────────

    def load_bars(
        self,
        symbol: str,
        period: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """从本地加载 K 线数据。

        Args:
            symbol: 股票代码
            period: 周期
            start: 起始日期（可选过滤）
            end: 结束日期（可选过滤）
        """
        path = self._symbol_path(symbol, period)
        df = self._read(path)
        if df.empty:
            return []
        bars = self._df_to_bars(df, symbol)
        if start is not None:
            bars = [b for b in bars if b.timestamp >= start]
        if end is not None:
            bars = [b for b in bars if b.timestamp <= end]
        return bars

    def save_bars(self, bars: list[Bar], symbol: str, period: str = "1d") -> None:
        """保存 K 线数据到本地。"""
        if not bars:
            return
        df = self._bars_to_df(bars)
        path = self._symbol_path(symbol, period)
        self._write(df, path, mode="append", dedup_column="timestamp")
        key = self._make_key(symbol, period)
        self._register(key, path, df)

    def list_symbols(self, period: str = "1d") -> list[str]:
        """列出本地所有股票代码。"""
        keys = self.list_cached_keys()
        symbols = set()
        for key in keys:
            if f":{period}" in key:
                sym = key.split(":")[0]
                symbols.add(sym)
        # fallback: 扫描文件系统（注册表可能不完整）
        if not symbols:
            for parquet_file in self._storage.root.rglob("*.parquet"):
                stem = parquet_file.stem
                parent = parquet_file.parent.name  # SH/SZ/BJ
                if stem.isdigit() and parent in ("SH", "SZ", "BJ"):
                    symbols.add(f"{parent}.{stem}")
        return list(symbols)

    def load_df(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        """直接加载 DataFrame。"""
        path = self._symbol_path(symbol, period)
        return self._read(path)

    def update_universe(self, stocks: list, date: datetime) -> None:
        """更新股票池元数据。"""
        stock_list = [s.code if hasattr(s, "code") else s for s in stocks]
        entry = {
            "stocks": stock_list,
            "updated_at": date.isoformat(),
            "count": len(stock_list),
        }
        self._storage.set_entry("universe", entry)

    def update_stock_names(self, stocks: list) -> None:
        """更新股票名称映射。"""
        names = {}
        for s in stocks:
            code = s.code if hasattr(s, "code") else s
            name = s.code_name if hasattr(s, "code_name") else ""
            names[code] = name
        self._storage.set_entry("stock_names", names)

    def get_stock_names(self) -> dict[str, str]:
        """获取股票名称映射（code → name）。"""
        return self._storage.get_entry("stock_names") or {}

    def get_universe(self) -> dict[str, Any]:
        """获取股票池。"""
        return self._storage.get_entry("universe") or {"stocks": []}

    def get_last_hot_sync(self) -> dict[str, Any]:
        """获取最近热同步元数据。"""
        return self._storage.get_entry("_hot_sync") or {}

    def update_last_hot_sync(self, last_hot_sync: datetime) -> None:
        """更新最近热同步时间。"""
        self._storage.set_entry(
            "_hot_sync",
            {
                "last_hot_sync": last_hot_sync.isoformat(),
            },
        )
