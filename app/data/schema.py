"""数据模型 — 所有价格/金额使用 Decimal 避免浮点精度问题。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TypedDict

logger = logging.getLogger(__name__)

# ── Schema 版本控制 ────────────────────────────────────
# 每次修改 Bar/Bars 的字段集合时递增此版本号。
# Parquet 文件存储 schema_version，读取时自动触发兼容性迁移。
# 版本历史:
#   0 — 初始版本，不含 is_delisted（该字段在序列化中丢失）
#   1 — 修复 is_delisted 序列化，加入 df.attrs["schema_version"]
#   2 — 新增 total_equity/np_atoopc_ttm/np_atoopc_q 财务字段
#   3 — 新增 turn 换手率与数据湖涨跌停制度标注 limit_rate/limit_reason
#   4 — 新增 turn_calc 自算换手率（成交量 ÷ 公告日生效的流通股本）
BAR_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class Bar:
    """K线 (OHLCV) 数据。

    结算字段（价格/资金）使用 Decimal 保证精确计算。
    成交量为 int（股数，天然整数）。
    成交额为 Decimal（元）。

    估值因子（来自 BaoStock K线 API）：
    - pe_ttm: 动态市盈率（TTM）
    - pb_mrq: 市净率
    - ps_ttm: 市销率（TTM）
    - pcf_ncf_ttm: 市现率（TTM）
    """

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    amount: Decimal  # 成交额（元）
    pre_close: Decimal = Decimal("0")  # 昨收价，用于涨跌停计算
    adj_factor: Decimal = Decimal("1.0")  # 前复权因子
    is_st: bool = False  # ST 状态（PIT 合规）
    is_delisted: bool = False  # 最后交易日标志（PIT 合规）

    # 估值因子（可选，某些日期可能为 None）
    pe_ttm: Decimal | None = None
    pb_mrq: Decimal | None = None
    ps_ttm: Decimal | None = None
    pcf_ncf_ttm: Decimal | None = None

    # 市值（从 QMT 获取，Bar 裸数据可能为 None）
    total_mv: Decimal | None = None  # 总市值（元）
    circ_mv: Decimal | None = None  # 流通市值（元）

    # 财务基本面（从 QMT Income/Balance 表 PIT 对齐后存入）
    total_equity: Decimal | None = None  # 净资产（归属母公司所有者权益）
    np_atoopc_ttm: Decimal | None = None  # 归母净利润 TTM（过去 4 季度累计）
    np_atoopc_q: Decimal | None = None  # 归母净利润单季
    np_yoy: Decimal | None = None  # 归母净利润同比增长率

    # 日线契约 v3/v4（数据湖标注）
    turn: Decimal | None = None  # 券商口径换手率（%，数据源提供）
    turn_calc: Decimal | None = None  # 自算换手率（%，成交量 ÷ 公告日生效的流通股本）
    limit_rate: Decimal | None = None  # 当日涨跌停比例；0=无涨跌幅限制，None=未标注
    limit_reason: str | None = None  # limit_rate 判定依据（normal/st/ipo_first_5d/…）

    @property
    def close_price(self) -> Decimal:
        return self.close

    @property
    def pre_close_price(self) -> Decimal:
        return self.pre_close

    def adjusted_close(self, target_factor: Decimal) -> Decimal:
        return self.close * target_factor / self.adj_factor

    def adjusted_price(self, price: Decimal, target_factor: Decimal) -> Decimal:
        return price * target_factor / self.adj_factor

    def is_limit_up(self, limit_rate: Decimal = Decimal("0.10")) -> bool:
        """检查当前 K 线是否涨停。"""
        if self.pre_close == Decimal("0"):
            return False
        return self.close >= self.pre_close * (Decimal("1") + limit_rate)

    def is_limit_down(self, limit_rate: Decimal = Decimal("0.10")) -> bool:
        """检查当前 K 线是否跌停。"""
        if self.pre_close == Decimal("0"):
            return False
        return self.close <= self.pre_close * (Decimal("1") - limit_rate)

    @classmethod
    def from_dict(cls, data: dict[str, object], symbol: str) -> Bar:
        def to_decimal(val: object) -> Decimal:
            if val is None:
                return Decimal("0")
            return Decimal(str(val))

        def opt_decimal(val: object) -> Decimal | None:
            """转换为 Decimal，空/缺失/非有限值返回 None。"""
            if val is None:
                return None
            s = str(val).strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return None
            try:
                value = Decimal(s)
            except Exception:
                logger.debug("Decimal 转换失败，返回 None", exc_info=True)
                return None
            return value if value.is_finite() else None

        def opt_text(val: object) -> str | None:
            """转换为字符串，空/缺失值返回 None。"""
            if val is None:
                return None
            text = str(val).strip()
            return None if not text or text.lower() in {"nan", "none", "null"} else text

        def to_int(val: object) -> int:
            """转换为整数股数；非数值/非有限值（NaN/Inf）回退 0，不抛异常。"""
            try:
                value = to_decimal(val)
                return int(value) if value.is_finite() else 0
            except Exception:  # noqa: BLE001 - 脏数据不应中断整批解析
                logger.debug("整数转换失败，返回 0", exc_info=True)
                return 0

        ts_raw = data["timestamp"]
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromisoformat(str(ts_raw))

        is_st_raw = data.get("is_st", data.get("isST", "0"))
        if isinstance(is_st_raw, str):
            is_st = is_st_raw.strip() == "1"
        else:
            is_st = bool(is_st_raw)

        return cls(
            symbol=symbol,
            timestamp=ts,
            open=to_decimal(data["open"]),
            high=to_decimal(data["high"]),
            low=to_decimal(data["low"]),
            close=to_decimal(data["close"]),
            volume=to_int(data["volume"]),
            amount=to_decimal(data.get("amount", 0)),
            pre_close=to_decimal(data.get("pre_close", data.get("close", 0))),
            adj_factor=to_decimal(data.get("adj_factor", 1.0)),
            is_st=is_st,
            pe_ttm=opt_decimal(data.get("pe_ttm", data.get("peTTM"))),
            pb_mrq=opt_decimal(data.get("pb_mrq", data.get("pbMRQ"))),
            ps_ttm=opt_decimal(data.get("ps_ttm", data.get("psTTM"))),
            pcf_ncf_ttm=opt_decimal(data.get("pcf_ncf_ttm", data.get("pcfNcfTTM"))),
            turn=opt_decimal(data.get("turn")),
            turn_calc=opt_decimal(data.get("turn_calc")),
            limit_rate=opt_decimal(data.get("limit_rate")),
            limit_reason=opt_text(data.get("limit_reason", data.get("limitReason"))),
        )


@dataclass(frozen=True, slots=True)
class Quote:
    """实时行情。"""

    symbol: str
    timestamp: datetime
    last: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    volume: int
    amount: Decimal  # 成交额（元）
    bid1: Decimal
    ask1: Decimal


@dataclass(frozen=True, slots=True)
class Instrument:
    """证券/合约信息。"""

    code: str
    name: str
    exchange: str  # SH / SZ
    full_code: str

    @classmethod
    def from_detail(cls, detail: dict[str, object], full_code: str) -> Instrument:
        """从 QMT 品种详情创建 Instrument。"""
        return cls(
            code=str(detail["InstrumentID"]),
            name=str(detail["InstrumentName"]),
            exchange=str(detail["ExchangeID"]),
            full_code=full_code,
        )


class OrderStatus(Enum):
    """统一订单生命周期状态。

    覆盖回测（EXECUTED）和实盘（SUBMITTED/PARTIAL_FILLED/FILLED）。
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class Order:
    """带生命周期跟踪的交易订单。"""

    order_id: str
    symbol: str
    qty: int
    created_ts: datetime
    filled_qty: int = 0
    status: OrderStatus = OrderStatus.PENDING
    reject_reason: str | None = None
    frozen_amount: Decimal = Decimal("0")
    """买入挂单冻结的资金（元）；成交/撤单后按此金额精确释放，不靠公式重算。"""

    @property
    def is_buy(self) -> bool:
        return self.qty > 0

    @property
    def is_sell(self) -> bool:
        return self.qty < 0

    def cancel(self, reason: str) -> None:
        self.status = OrderStatus.CANCELLED
        self.reject_reason = reason

    def reject(self, reason: str) -> None:
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason

    def execute(self, filled_qty: int) -> None:
        self.filled_qty = filled_qty
        self.status = OrderStatus.EXECUTED


@dataclass(frozen=True)
class CostBreakdown:
    """交易成本明细。

    所有字段使用 Decimal（元）。
    """

    commission: Decimal  # 佣金
    transfer_fee: Decimal  # 过户费
    stamp_duty: Decimal  # 印花税（仅卖出收取）

    @property
    def total(self) -> Decimal:
        return self.commission + self.transfer_fee + self.stamp_duty


@dataclass(frozen=True)
class TradeRecord:
    """不可变的交易执行记录，带审计追踪。"""

    parent_order_id: str
    slippage_cost: Decimal
    timestamp: datetime
    symbol: str
    quantity: int
    price: Decimal
    cost: CostBreakdown
    is_buy: bool

    @property
    def amount(self) -> Decimal:
        return Decimal(abs(self.quantity)) * self.price

    @property
    def fee(self) -> Decimal:
        """总费用（佣金 + 过户费 + 印花税）。"""
        return self.cost.total

    @property
    def net_amount(self) -> Decimal:
        """净现金流：买入为负（流出），卖出为正（流入）。"""
        if self.is_buy:
            return -(self.amount + self.cost.total + self.slippage_cost)
        return self.amount - self.cost.total - self.slippage_cost


class EquityPoint(TypedDict):
    """每日权益记录。"""

    timestamp: datetime
    total_value: Decimal
    cash: Decimal
