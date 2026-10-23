"""回测核心数据模型。

定义信号、目标持仓、策略状态和回测结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class SignalDirection(Enum):
    """信号方向。"""

    LONG = 1
    SHORT = -1
    FLAT = 0


class SignalType(Enum):
    """信号类型。"""

    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    REGIME_SHIFT = "regime_shift"


@dataclass(frozen=True)
class Signal:
    """交易信号。"""

    symbol: str
    direction: SignalDirection
    signal_type: SignalType = SignalType.ENTRY
    strength: float = 1.0
    reason: str = ""
    source: str = ""
    timestamp: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 多策略场景下 strength 可 > 1.0（通过 cap_weight × max_positions / select_num 推导）
        # 引擎的 Cash Guard 会防止实际超额分配
        pass


@dataclass(frozen=True)
class TargetPosition:
    """目标持仓。"""

    symbol: str
    quantity: int
    signal: Signal | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(f"目标持仓数量必须 >= 0，当前值: {self.quantity}")


@dataclass
class StrategyState:
    """可序列化的策略状态。"""

    strategy_id: str
    warmup_period: int = 0
    last_updated: datetime | None = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "warmup_period": self.warmup_period,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyState:
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])
        return cls(
            strategy_id=data["strategy_id"],
            warmup_period=data.get("warmup_period", 0),
            last_updated=last_updated,
            data=data.get("data", {}),
        )


@dataclass(frozen=True)
class Order:
    """交易订单。

    价格和金额使用 Decimal 保证精度。
    """

    order_id: str
    symbol: str
    quantity: int
    price: Decimal
    is_buy: bool
    timestamp: datetime
    status: str = "pending"
    filled_quantity: int = 0
    filled_price: Decimal = Decimal("0")
    reject_reason: str | None = None

    @property
    def amount(self) -> Decimal:
        return Decimal(self.filled_quantity) * self.filled_price


@dataclass
class EquityPoint:
    """每日权益记录。"""

    timestamp: datetime
    total_value: Decimal
    cash: Decimal
    frozen_cash: Decimal
    positions_value: Decimal
    positions_detail: dict[str, int] = field(default_factory=dict)
    cash_interest: Decimal = Decimal("0")
    """当日计提的闲置资金收益（元）；未开启利率口径停泊时为 0。"""

    parking_value: Decimal = Decimal("0")
    """停泊标的当日市值（元）；未启用标的停泊时为 0。"""


@dataclass
class StrategyAttribution:
    """单策略归因统计。

    追踪每个子策略在整个回测期间的独立表现。
    """

    strategy_id: str
    pnl: Decimal = Decimal("0")
    """累计盈亏。"""
    nav_series: list[float] = field(default_factory=list)
    """净值序列（与 equity_curve 时间对齐）。"""
    trade_count: int = 0
    """该策略产生的交易数。"""
    long_signal_count: int = 0
    """买入信号数。"""
    flat_signal_count: int = 0
    """卖出信号数。"""


@dataclass
class BacktestResult:
    """回测结果容器。"""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_cash: Decimal
    trade_log: list = field(default_factory=list)
    order_log: list = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    parking_log: list = field(default_factory=list)
    """停泊标的成交记录（不计入策略换手）。"""
    attribution: dict[str, StrategyAttribution] = field(default_factory=dict)
    """子策略级归因: {strategy_id: StrategyAttribution}。"""
    metadata: dict = field(default_factory=dict)
