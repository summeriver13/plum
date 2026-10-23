from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.data.schema import Bar
from app.engine.live.gateway.base import (
    AccountInfo,
    BaseGateway,
    Direction,
    OrderRequest,
    OrderResponse,
    OrderStatus,
)
from app.rule.trading.calculator import ASHARE_RULES

logger = logging.getLogger(__name__)


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/NaN/非数值返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, InvalidOperation):
        return default


@dataclass
class MockGateway(BaseGateway):
    """模拟交易网关。

    模拟真实撮合行为，支持滑点和佣金计算。
    """

    initial_cash: int = 40_000_0000
    slippage_bps: float = 10.0
    min_commission: float = 5.0

    _connected: bool = field(default=False, repr=False)
    _subscribed: set[str] = field(default_factory=set, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, repr=False)
    _available: dict[str, int] = field(default_factory=dict, repr=False)
    _today_bought: dict[str, int] = field(default_factory=dict, repr=False)
    _cash: int = field(default=0, repr=False)
    _bars: dict[str, Bar] = field(default_factory=dict, repr=False)
    _orders: dict[str, OrderRequest] = field(default_factory=dict, repr=False)
    _order_id: int = field(default=0, repr=False)
    _on_trade_callback: Callable | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._cash = self.initial_cash

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def on_trade(self, callback: Callable[[str, int, int, bool], None]) -> None:
        self._on_trade_callback = callback

    def subscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed.add(s)

    def unsubscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed.discard(s)

    def update_bars(self, bars: dict[str, Bar]) -> None:
        self._bars.update(bars)

    def get_current_bars(self) -> dict[str, Bar]:
        return dict(self._bars)

    def send_order(self, order: OrderRequest) -> OrderResponse:
        if not self._connected:
            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason="not_connected",
            )

        self._order_id += 1
        order_id = f"MOCK_{self._order_id}"

        if order.symbol not in self._subscribed:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason="not_subscribed",
            )

        bar = self._bars.get(order.symbol)
        if bar is None:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason="no_data",
            )

        exec_price = self._calc_exec_price(bar, order.direction)
        exec_price_int = _int(exec_price)
        self._orders[order_id] = order

        filled_qty = order.quantity
        if order.direction == Direction.BUY:
            cost = filled_qty * exec_price_int
            commission = max(_int(cost * 0.00002), _int(self.min_commission * 10000))
            total_cost = cost + commission
            if total_cost > self._cash:
                raw_qty = _int(self._cash / (exec_price_int * 1.00002))
                filled_qty = ASHARE_RULES.floor_to_lot(order.symbol, raw_qty)
                if not ASHARE_RULES.is_valid_lot(order.symbol, filled_qty):
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        symbol=order.symbol,
                        direction=order.direction,
                        quantity=order.quantity,
                        reject_reason="insufficient_cash",
                    )
                commission = max(
                    _int(filled_qty * exec_price_int * 0.00002),
                    _int(self.min_commission * 10000),
                )

            self._cash -= _int(filled_qty * exec_price_int + commission)
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + filled_qty
            )
            self._available[order.symbol] = (
                self._available.get(order.symbol, 0) + filled_qty
            )
            self._today_bought[order.symbol] = (
                self._today_bought.get(order.symbol, 0) + filled_qty
            )
        else:
            available_qty = self._available.get(order.symbol, 0)
            if filled_qty > available_qty:
                if available_qty <= 0:
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        symbol=order.symbol,
                        direction=order.direction,
                        quantity=order.quantity,
                        reject_reason="t1_restriction",
                    )
                filled_qty = available_qty
            if filled_qty <= 0:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    symbol=order.symbol,
                    direction=order.direction,
                    quantity=order.quantity,
                    reject_reason="no_position",
                )
            revenue = filled_qty * exec_price_int
            commission = max(_int(revenue * 0.00002), _int(self.min_commission * 10000))
            stamp_duty = _int(revenue * 0.00005)
            self._cash += _int(revenue - commission - stamp_duty)
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) - filled_qty
            )
            self._available[order.symbol] = (
                self._available.get(order.symbol, 0) - filled_qty
            )
            if self._positions[order.symbol] == 0:
                del self._positions[order.symbol]
                if order.symbol in self._available:
                    del self._available[order.symbol]

        if filled_qty > 0 and self._on_trade_callback is not None:
            cb = self._on_trade_callback
            if callable(cb):
                try:
                    cb(
                        order.symbol,
                        filled_qty,
                        exec_price_int,
                        order.direction == Direction.BUY,
                    )
                except Exception:
                    logger.warning("模拟网关成交回调执行失败", exc_info=True)

        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.FILLED,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            filled_quantity=filled_qty,
            avg_price=exec_price_int,
        )

    def cancel_order(self, order_id: str) -> OrderResponse:
        order = self._orders.get(order_id)
        if order is None:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol="",
                direction=Direction.BUY,
                quantity=0,
                reject_reason="order_not_found",
            )
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
        )

    def get_positions(self) -> dict[str, int]:
        return dict(self._positions)

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            cash=self._cash,
            frozen_cash=0,
            positions=dict(self._positions),
            available=dict(self._available),
        )

    def _calc_exec_price(self, bar: Bar, direction: Direction) -> Decimal:
        slippage = Decimal(str(self.slippage_bps)) / Decimal("10000")
        if direction == Direction.BUY:
            return bar.close * (Decimal("1") + slippage)
        else:
            return bar.close * (Decimal("1") - slippage)

    def reset(self) -> None:
        self._cash = self.initial_cash
        self._positions.clear()
        self._available.clear()
        self._today_bought.clear()
        self._orders.clear()
        self._bars.clear()

    def end_of_day(self) -> None:
        for symbol, qty in self._today_bought.items():
            self._available[symbol] = self._available.get(symbol, 0) - qty
            if self._available[symbol] <= 0 and symbol in self._available:
                del self._available[symbol]
        self._today_bought.clear()

    @property
    def account_id(self) -> str:
        return "mock_account"
