"""模拟交易数据源 — 从 CatalogManager 获取实时行情。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.data.schema import Bar


class PaperDataFeed:
    """模拟交易实时行情数据源。

    封装 CatalogManager 提供逐笔行情数据。
    """

    def __init__(
        self,
        catalog: object | None = None,
    ) -> None:
        self._catalog = catalog
        self._subscribed: set[str] = set()
        self._current_bars: dict[str, Bar] = {}
        self._tick_callbacks: list[Callable[[dict[str, Bar]], None]] = []

    def get_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[Bar]]:
        """获取历史 K 线数据。"""
        if self._catalog is None:
            return {}
        result: dict[str, list[Bar]] = {}
        for symbol in symbols:
            bars = self._catalog.load_bars(symbol, start=start, end=end)
            if bars:
                result[symbol] = bars
        return result

    def subscribe(self, symbols: list[str]) -> None:
        """订阅实时行情。"""
        self._subscribed.update(symbols)

    def on_tick(self, callback: Callable[[dict[str, Bar]], None]) -> None:
        """注册实时行情回调。"""
        self._tick_callbacks.append(callback)

    def update_bars(self, bars: dict[str, Bar]) -> None:
        """更新当前 K 线并通知回调。"""
        self._current_bars.update(bars)
        for callback in self._tick_callbacks:
            callback(bars)

    def get_current_bars(self) -> dict[str, Bar]:
        """获取当前 K 线快照。"""
        return self._current_bars.copy()

    def get_subscribed(self) -> set[str]:
        """获取已订阅的股票列表。"""
        return self._subscribed.copy()
