"""模拟交易环境 — 实时数据模拟循环。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from app.data.schema import Bar, Order
from app.engine.backtest.portfolio import Portfolio
from app.engine.paper.broker import PaperBroker, PaperBrokerConfig
from app.engine.paper.clock import PaperClock, PaperClockConfig
from app.engine.paper.data_feed import PaperDataFeed

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PaperEnvironmentConfig:
    """模拟交易环境配置。"""

    clock: PaperClockConfig = field(default_factory=PaperClockConfig)
    broker: PaperBrokerConfig = field(default_factory=PaperBrokerConfig)
    tick_interval: float = 3.0


class PaperEnvironment:
    """模拟交易环境，使用实时数据。

    集成 PaperClock、PaperDataFeed 和 PaperBroker
    构成模拟交易循环。

    用法:
        env = PaperEnvironment(catalog)
        env.initialize(portfolio)

        env.on_bar(lambda ts, bars, portfolio:
            signals = strategy.on_bar(ts, bars, portfolio)
            orders = gateway.process_signals(signals, bars, ts)
            for order in orders:
                env.broker.submit_order(order)
        )

        env.start()
    """

    def __init__(
        self,
        catalog: object | None = None,
        config: PaperEnvironmentConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self.config = config or PaperEnvironmentConfig()

        self._clock = PaperClock(self.config.clock)
        self._data_feed = PaperDataFeed(catalog)
        self._broker = PaperBroker(self.config.broker)

        self._portfolio: Portfolio | None = None
        self._bar_callback: (
            Callable[[datetime, dict[str, Bar], Portfolio], None] | None
        ) = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def env_type(self) -> str:
        return "paper"

    @property
    def clock(self) -> PaperClock:
        return self._clock

    @property
    def data_feed(self) -> PaperDataFeed:
        return self._data_feed

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    def initialize(self, portfolio: Portfolio) -> None:
        """初始化环境。

        参数:
            portfolio: 初始持仓状态
        """
        self._portfolio = portfolio
        self._broker.set_portfolio(portfolio)

    def on_bar(
        self,
        callback: Callable[[datetime, dict[str, Bar], Portfolio], None],
    ) -> None:
        """设置每根 K 线的回调函数。

        参数:
            callback: 函数(ts, bars, portfolio) -> None
        """
        self._bar_callback = callback

    def subscribe(self, symbols: list[str]) -> None:
        """订阅股票实时行情。

        参数:
            symbols: 要订阅的股票代码列表
        """
        self._data_feed.subscribe(symbols)

    def start(self) -> None:
        """在后台线程启动模拟交易循环。"""
        if self._running:
            logger.warning("PaperEnvironment already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("PaperEnvironment started")

    def _run_loop(self) -> None:
        """模拟交易主循环。"""
        while self._running:
            if not self._clock.is_market_open(self._clock.now()):
                self._clock.sleep(self.config.tick_interval)
                continue

            ts = self._clock.now()
            bars = self._data_feed.get_current_bars()

            if bars and self._bar_callback and self._portfolio:
                self._data_feed.update_bars(bars)
                self._broker.set_current_bars(bars)
                try:
                    self._bar_callback(ts, bars, self._portfolio)
                except Exception as e:
                    logger.error(f"Error in bar callback: {e}")

            self._clock.sleep(self.config.tick_interval)

    def stop(self) -> None:
        """停止模拟交易循环。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PaperEnvironment stopped")

    def sync_portfolio(self) -> Portfolio:
        """获取当前持仓状态。

        返回:
            当前持仓
        """
        if self._portfolio is None:
            raise RuntimeError("Portfolio not initialized")
        return self._portfolio

    def get_pending_orders(self) -> list[Order]:
        """获取所有待成交订单。

        返回:
            待成交订单列表
        """
        return self._broker.get_pending_orders()
