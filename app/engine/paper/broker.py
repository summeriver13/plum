"""模拟交易券商 — 带延迟的模拟撮合。"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from threading import Lock

from app.data.schema import Bar, Order, OrderStatus
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel


@dataclass
class PaperBrokerConfig:
    """模拟交易券商配置。"""

    fill_delay_ms: tuple[int, int] = (500, 2000)
    fill_rate: float = 0.95
    commission: AShareCommissionModel | None = None


class PaperBroker:
    """模拟交易券商。

    模拟真实撮合行为：
    - 随机延迟（500ms-2000ms）
    - 部分成交率（可配置）
    - 价格滑点模拟
    """

    def __init__(
        self,
        config: PaperBrokerConfig | None = None,
    ) -> None:
        self.config = config or PaperBrokerConfig()
        self.commission = self.config.commission or AShareCommissionModel()
        self._portfolio: Portfolio | None = None
        self._current_bars: dict[str, Bar] = {}
        self._orders: dict[str, Order] = {}
        self._pending_orders: dict[str, Order] = {}
        self._fill_callbacks: list[Callable[[Order], None]] = []
        self._lock = Lock()

    def set_portfolio(self, portfolio: Portfolio) -> None:
        """设置持仓引用。"""
        self._portfolio = portfolio

    def set_current_bars(self, bars: dict[str, Bar]) -> None:
        """设置当前 K 线数据用于成交价格计算。"""
        self._current_bars = bars

    def submit_order(self, order: Order) -> str:
        """提交订单进行模拟撮合。

        返回:
            生成的订单 ID
        """
        order_id = f"PP_{uuid.uuid4().hex[:8]}"
        ts = datetime.now()

        bar = self._current_bars.get(order.symbol)
        if bar is None:
            raise ValueError(f"No bar data for {order.symbol}")

        pending_order = Order(
            order_id=order_id,
            symbol=order.symbol,
            qty=order.qty,
            status=OrderStatus.PENDING,
            created_ts=ts,
        )

        with self._lock:
            self._pending_orders[order_id] = pending_order

        self._simulate_fill_async(order_id, pending_order)

        return order_id

    def _simulate_fill_async(self, order_id: str, order: Order) -> None:
        """在后台线程模拟成交。"""
        import threading

        thread = threading.Thread(
            target=self._do_fill,
            args=(order_id, order),
            daemon=True,
        )
        thread.start()

    def _do_fill(self, order_id: str, order: Order) -> None:
        """执行模拟成交。"""
        delay = random.uniform(
            self.config.fill_delay_ms[0] / 1000,
            self.config.fill_delay_ms[1] / 1000,
        )
        time.sleep(delay)

        with self._lock:
            if order_id not in self._pending_orders:
                return

            bar = self._current_bars.get(order.symbol)
            if bar is None:
                order.status = OrderStatus.REJECTED
                return

            fill_rate = random.random()
            if fill_rate > self.config.fill_rate:
                order.status = OrderStatus.REJECTED
                self._orders[order_id] = order
                del self._pending_orders[order_id]
                return

            is_buy = order.is_buy
            commission_cost = self.commission.calculate(
                bar.close, abs(order.qty), is_buy=is_buy
            )

            order.price = bar.close
            order.status = OrderStatus.FILLED
            order.commission = commission_cost.total

            self._orders[order_id] = order
            del self._pending_orders[order_id]

            if self._portfolio is not None:
                self._update_portfolio(order, bar)

        for callback in self._fill_callbacks:
            callback(order)

    def _update_portfolio(self, trade: Order, bar: Bar) -> None:
        """更新持仓。"""
        if self._portfolio is None:
            return

        self._portfolio.apply_trade(
            trade.symbol,
            trade.qty,
            bar.close,
            trade.commission or Decimal("0"),
            trade.created_ts,
        )

    def cancel_order(self, order_id: str) -> bool:
        """尝试撤单。

        返回:
            成功返回 True，否则 False
        """
        with self._lock:
            if order_id in self._pending_orders:
                order = self._pending_orders[order_id]
                order.status = OrderStatus.CANCELLED
                self._orders[order_id] = order
                del self._pending_orders[order_id]
                return True
        return False

    def get_order_status(self, order_id: str) -> Order | None:
        """查询订单状态。"""
        with self._lock:
            if order_id in self._pending_orders:
                return self._pending_orders[order_id]
            return self._orders.get(order_id)

    def get_pending_orders(self) -> list[Order]:
        """获取所有待成交订单。"""
        with self._lock:
            return list(self._pending_orders.values())

    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """注册成交回调。"""
        self._fill_callbacks.append(callback)
