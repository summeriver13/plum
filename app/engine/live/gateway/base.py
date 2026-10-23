"""交易网关基类 — 统一订单执行接口。"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from app.data.schema import OrderStatus

if TYPE_CHECKING:
    from app.data.schema import Bar


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class Direction(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    symbol: str
    direction: Direction
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: int | None = None
    order_id: str = ""
    created_ts: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.order_id:
            self.order_id = str(uuid.uuid4())


@dataclass
class OrderResponse:
    order_id: str
    status: OrderStatus
    symbol: str
    direction: Direction
    quantity: int
    broker_id: str = ""
    """券商委托编号。"""
    filled_quantity: int = 0
    avg_price: int = 0
    reject_reason: str = ""
    updated_ts: datetime = field(default_factory=datetime.now)


@dataclass
class AccountInfo:
    cash: int
    frozen_cash: int = 0
    positions: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    updated_ts: datetime = field(default_factory=datetime.now)


class BaseGateway(ABC):
    """交易网关抽象接口。

    每个环境提供自己的 Gateway 实现：
    - MockGateway: 模拟撮合（用于测试）
    - XtQuantGateway: 实盘通过 XtQuant API
    """

    @abstractmethod
    def subscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_current_bars(self) -> dict[str, Bar]:
        raise NotImplementedError

    @abstractmethod
    def send_order(self, order: OrderRequest) -> OrderResponse:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    def check_status(self) -> bool:
        """心跳检查。"""
        return self.is_connected

    @abstractmethod
    def on_trade(self, callback: Callable[[str, int, int, bool], None]) -> None:
        """注册成交回调。

        回调签名:
        callback(symbol: str, quantity: int, price: int, is_buy: bool) -> None
        """
        raise NotImplementedError

    def set_notify_callback(self, callback: object) -> None:
        """注册通知回调（可选）。"""
        pass
