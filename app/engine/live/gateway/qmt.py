"""QMT Gateway — 迅投 XtQuant 实盘交易网关。

对接迅投 XtQuant API，提供实盘与模拟盘统一接口。
通过 BaseGateway 抽象层与回测引擎解耦，实现同一策略代码
在回测（MockGateway）与实盘（QmtGateway）之间无缝切换。

价格体系：
  - QMT 内部: int * 10000（如 10.5 元 → 105000）
  - Plum 内部: Decimal（如 Decimal("10.5")）
  - 所有进出 QMT API 的价格均在此模块内完成转换

配置来源：
  - config.toml [broker] 节 + 环境变量 PLUM_BROKER_* 覆盖
  - 关键字段: qmt_path, account_id, session_id, trade_env
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from app.config.settings import get_config
from app.engine.live.gateway.base import (
    AccountInfo,
    BaseGateway,
    Direction,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)

# QMT 价格精度因子：QMT 内部价格 = 实际价格 × 10000
QMT_PRICE_FACTOR = Decimal("10000")


# ─── 懒加载 xtquant ──────────────────────────────────────────────

_xt = None


def _get_xt() -> tuple[Any, Any, Any]:
    """惰性加载 xtquant 模块。

    QMT 客户端可能未安装在当前环境，模块级静态导入
    会导致整个 gateway 包无法加载。使用此函数按需导入。

    Returns:
        ``(xtconstant, XtQuantTrader, StockAccount)`` 三元组。元素类型是可选
        依赖，无法静态确定；显式标注为 Any，避免调用方被推断成错误类型。
    """
    global _xt
    if _xt is None:
        from xtquant import xtconstant  # noqa: F401
        from xtquant.xttrader import XtQuantTrader  # noqa: F401
        from xtquant.xttype import StockAccount  # noqa: F401

        _xt = (xtconstant, XtQuantTrader, StockAccount)
    return _xt


# ─── 价格转换工具 ────────────────────────────────────────────────


def _price_to_qmt(price: Decimal | int) -> int:
    """Plum Decimal 价格 → QMT int 价格（×10000）。

    Raises:
        ValueError: 价格非法（NaN/Inf）或无法量化时抛出——实盘下单宁可
            直接失败，也不静默按非法价格发单。
    """
    if isinstance(price, int):
        return price
    try:
        scaled = (price * QMT_PRICE_FACTOR).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        return int(scaled)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"价格无法转换为 QMT 整数价格: {price!r}") from exc


def _price_from_qmt(qmt_price: int | float) -> Decimal:
    """QMT int 价格 → Plum Decimal 价格（÷10000）。"""
    return Decimal(str(qmt_price)) / QMT_PRICE_FACTOR


# ─── QMT 订单状态映射 ────────────────────────────────────────────

# xtconstant 订单状态 → Plum OrderStatus
_QMT_STATUS_MAP: dict[int, OrderStatus] = {
    48: OrderStatus.SUBMITTED,  # 已发送
    49: OrderStatus.SUBMITTED,  # 未成交
    50: OrderStatus.PARTIAL_FILLED,  # 部分成交
    51: OrderStatus.FILLED,  # 完全成交
    52: OrderStatus.FILLED,  # 部分撤单
    53: OrderStatus.PARTIAL_FILLED,  # 已撤余单（视作部分成交）
    54: OrderStatus.CANCELLED,  # 已撤
    55: OrderStatus.REJECTED,  # 废单
    56: OrderStatus.REJECTED,  # 待确认
    57: OrderStatus.REJECTED,  # 已确认（未发送）
}


def _map_qmt_direction(direction: Direction) -> int:
    """Plum Direction → xtconstant 买卖方向。"""
    _, xtconstant, _ = _get_xt()
    if direction == Direction.BUY:
        return xtconstant.STOCK_BUY
    return xtconstant.STOCK_SELL


def _map_qmt_price_type(order_type: OrderType) -> int:
    """Plum OrderType → xtconstant 价格类型。"""
    _, xtconstant, _ = _get_xt()
    if order_type == OrderType.LIMIT:
        return xtconstant.FIX_PRICE  # 限价
    return xtconstant.LATEST_PRICE  # 最新价


# ─── QmtGateway ──────────────────────────────────────────────────


@dataclass
class QmtGateway(BaseGateway):
    """迅投 XtQuant 交易网关。

    通过 xtquant.xttrader.XtQuantTrader 连接 QMT 客户端，
    将 Plum 统一订单接口映射到 QMT 底层 API。

    预设路径与账户可通过环境变量覆盖 config.toml 默认值：
      PLUM_BROKER_QMT_PATH     — QMT 安装路径
      PLUM_BROKER_ACCOUNT_ID   — 资金账号
      PLUM_BROKER_SESSION_ID   — 会话 ID
      PLUM_BROKER_TRADE_ENV    — sim（模拟）｜real（实盘）

    Usage:
        gw = QmtGateway()
        gw.connect()
        resp = gw.send_order(OrderRequest(symbol="000001.SZ", direction=Direction.BUY, quantity=100))
        logger.info("当前持仓: %s", gw.get_positions())
    """

    # ── 配置字段 ──

    qmt_path: str = ""
    """QMT/miniQMT 安装目录。"""

    account_id: str = ""
    """资金账号。"""

    session_id: int = 9527
    """会话 ID，默认 9527。"""

    trade_env: str = "sim"
    """交易环境: sim（模拟盘）或 real（实盘）。"""

    # ── 内部状态（repr=False 避免日志泄漏敏感字段） ──

    _trader: Any = field(default=None, repr=False, init=False)
    _account: Any = field(default=None, repr=False, init=False)
    _connected: bool = field(default=False, repr=False, init=False)
    _subscribed: set[str] = field(default_factory=set, repr=False, init=False)
    _on_trade_callback: Callable[[str, int, int, bool], None] | None = field(
        default=None, repr=False, init=False
    )
    _config_loaded: bool = field(default=False, repr=False, init=False)

    # ── 配置加载 ──────────────────────────────────────────────

    def _ensure_config(self) -> None:
        """从全局配置加载 QMT 参数（仅首次调用）。"""
        if self._config_loaded:
            return

        cfg = get_config()
        broker = cfg.broker

        # 仅在用户未显式设置时使用配置文件值
        if not self.qmt_path:
            self.qmt_path = broker.qmt_path
        if not self.account_id:
            self.account_id = broker.account_id
        if self.session_id == 9527:
            self.session_id = broker.session_id
        if self.trade_env == "sim":
            self.trade_env = broker.trade_env

        self._config_loaded = True

    # ── 连接管理 ──────────────────────────────────────────────

    def connect(self) -> bool:
        """连接 QMT 客户端并订阅账户。

        流程:
          1. 创建 XtQuantTrader 实例
          2. 启动交易服务 (.start())
          3. 建立连接 (.connect())
          4. 订阅 StockAccount (.subscribe())

        Returns:
            True 表示连接成功，False 表示失败。
        """
        self._ensure_config()

        if not self.qmt_path:
            logger.error(
                "QMT 路径未配置，请设置 PLUM_BROKER_QMT_PATH 或 config.toml [broker].qmt_path"
            )
            return False
        if not self.account_id:
            logger.error(
                "资金账号未配置，请设置 PLUM_BROKER_ACCOUNT_ID 或 config.toml [broker].account_id"
            )
            return False

        try:
            _, XtQuantTrader, StockAccount = _get_xt()

            self._trader = XtQuantTrader(self.qmt_path, self.session_id)
            self._account = StockAccount(self.account_id, "STOCK")
            logger.info(
                "QMT 交易组件已创建: path=%s session=%d", self.qmt_path, self.session_id
            )

            self._trader.start()

            connect_result = self._trader.connect()
            if connect_result != 0:
                logger.error("QMT 连接失败，返回码: %d", connect_result)
                self._connected = False
                return False

            # 订阅账户，QMT 开始推送该账户的委托/成交
            subscribe_result = self._trader.subscribe(self._account)
            if subscribe_result != 0:
                logger.error("QMT 账户订阅失败，返回码: %d", subscribe_result)
                self._connected = False
                return False

            # 注册 xtquant 底层成交回调 → 桥接到 on_trade
            self._trader.register_callback(self._xt_trade_callback)

            self._connected = True
            logger.info(
                "QMT 已连接: account=%s env=%s session=%d",
                self.account_id,
                self.trade_env,
                self.session_id,
            )
            return True

        except ImportError as e:
            logger.error("xtquant 模块未安装，无法连接 QMT: %s", e)
            self._connected = False
            return False
        except (ConnectionError, TimeoutError):
            logger.exception("QMT 连接时发生异常")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """断开与 QMT 的连接。"""
        if self._trader is not None:
            self._trader.unsubscribe(self._account)
        self._connected = False
        self._subscribed.clear()
        self._on_trade_callback = None
        logger.info("QMT 已断开连接")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── 行情订阅 ──────────────────────────────────────────────

    def subscribe(self, symbols: list[str]) -> None:
        """订阅指定证券的行情推送。

        QMT 的行情订阅通过行情驱动模块（xtdata）完成，
        此处维护内部订阅列表供 send_order 时校验。
        """
        for s in symbols:
            self._subscribed.add(s)
        logger.debug("已订阅 %d 个品种", len(symbols))

    def unsubscribe(self, symbols: list[str]) -> None:
        """取消订阅指定证券。"""
        for s in symbols:
            self._subscribed.discard(s)

    # ── 行情查询 ──────────────────────────────────────────────

    def get_current_bars(self) -> dict[str, Any]:
        """获取当前订阅品种的实时 Bar 数据。

        通过 xtdata 获取最新行情快照，转换为 Plum Bar 格式。
        """
        from app.data.schema import Bar

        if not self._subscribed:
            return {}

        result: dict[str, Bar] = {}
        try:
            from xtquant import xtdata

            for symbol in self._subscribed:
                try:
                    snapshot = xtdata.get_full_tick([symbol])
                    if not snapshot or symbol not in snapshot:
                        continue
                    tick = snapshot[symbol]
                    now = datetime.now()
                    result[symbol] = Bar(
                        symbol=symbol,
                        timestamp=now,
                        open=_price_from_qmt(tick.get("open", 0)),
                        high=_price_from_qmt(tick.get("high", 0)),
                        low=_price_from_qmt(tick.get("low", 0)),
                        close=_price_from_qmt(tick.get("lastPrice", 0)),
                        volume=int(tick.get("volume", 0)),
                        amount=Decimal(str(tick.get("amount", 0))),
                        pre_close=_price_from_qmt(tick.get("lastClose", 0)),
                    )
                except (ValueError, KeyError):
                    logger.debug("获取 %s 实时行情失败", symbol, exc_info=True)
        except ImportError:
            logger.warning("xtdata 模块不可用，无法获取实时 Bars")

        return result

    # ── 订单执行 ──────────────────────────────────────────────

    def send_order(self, order: OrderRequest) -> OrderResponse:
        """向 QMT 发送下单指令。

        将 Plum OrderRequest 转换为 xttrader.order_stock() 参数，
        并映射返回的 xtconstant 委托状态为 Plum OrderStatus。

        Args:
            order: Plum 统一订单请求

        Returns:
            包含 broker_id 和映射后状态的 OrderResponse
        """
        if not self._connected:
            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason="not_connected",
            )

        if order.symbol not in self._subscribed:
            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason="not_subscribed",
            )

        try:
            _, xtconstant, _ = _get_xt()

            # 委托类型必须是 xtconstant.STOCK_BUY(23)/STOCK_SELL(24)——
            # order_stock 的 order_type 入参就是买卖方向，不是报价类型。
            direction = _map_qmt_direction(order.direction)
            price_type = _map_qmt_price_type(order.order_type)

            # QMT 价格: limit_price 若未指定则填 0（市价单）
            qmt_price = 0
            if order.limit_price is not None:
                qmt_price = _price_to_qmt(Decimal(str(order.limit_price)))

            strategy_name = "plum"
            order_remark = f"plum:{order.order_id}"

            # 调用 QMT 下单接口
            qmt_order_id = self._trader.order_stock(
                self._account,
                order.symbol,
                direction,
                order.quantity,
                price_type,
                qmt_price,
                strategy_name,
                order_remark,
            )

            if qmt_order_id < 0:
                # QMT 返回负数为下单失败
                error_msgs = {
                    -1: "参数错误",
                    -2: "连接断开",
                    -3: "资金不足",
                    -4: "持仓不足",
                    -5: "交易受限",
                }
                reason = error_msgs.get(qmt_order_id, f"下单失败(code={qmt_order_id})")
                return OrderResponse(
                    order_id=order.order_id,
                    status=OrderStatus.REJECTED,
                    symbol=order.symbol,
                    direction=order.direction,
                    quantity=order.quantity,
                    broker_id=str(qmt_order_id),
                    reject_reason=reason,
                )

            logger.info(
                "QMT 下单成功: symbol=%s dir=%s qty=%d broker_id=%d plum_id=%s",
                order.symbol,
                order.direction.value,
                order.quantity,
                qmt_order_id,
                order.order_id,
            )

            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.SUBMITTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                broker_id=str(qmt_order_id),
            )

        except ImportError:
            logger.warning("xtquant 未安装，订单被拒绝", exc_info=True)
            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                broker_id="",
                reject_reason="xtquant_not_installed",
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.exception("QMT 下单异常: %s", order.symbol)
            return OrderResponse(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                reject_reason=str(e)[:200],
            )

    def cancel_order(self, order_id: str) -> OrderResponse:
        """向 QMT 发送撤单指令。

        通过 broker_id（QMT 委托编号）撤销订单。

        Args:
            order_id: broker_id（QMT 返回的委托编号字符串）

        Returns:
            撤单结果 OrderResponse
        """
        if not self._connected:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol="",
                direction=Direction.BUY,
                quantity=0,
                reject_reason="not_connected",
            )

        try:
            qmt_order_id = int(order_id)
            result = self._trader.cancel_order_stock(self._account, qmt_order_id)

            if result == 0:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.CANCELLED,
                    symbol="",
                    direction=Direction.BUY,
                    quantity=0,
                    broker_id=order_id,
                )
            else:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    symbol="",
                    direction=Direction.BUY,
                    quantity=0,
                    broker_id=order_id,
                    reject_reason=f"取消失败(code={result})",
                )

        except ValueError:
            logger.warning("订单参数无效", exc_info=True)
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol="",
                direction=Direction.BUY,
                quantity=0,
                reject_reason="invalid_order_id",
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.exception("QMT 撤单异常: %s", order_id)
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                symbol="",
                direction=Direction.BUY,
                quantity=0,
                reject_reason=str(e)[:200],
            )

    # ── 账户查询 ──────────────────────────────────────────────

    def get_positions(self) -> dict[str, int]:
        """查询当前持仓。

        通过 query_stock_positions() 获取 QMT 实际持仓，
        返回 {symbol: volume} 字典（volume 为总持仓股数）。

        Returns:
            {股票代码: 持仓股数}
        """
        if not self._connected or self._trader is None:
            return {}

        try:
            positions = self._trader.query_stock_positions(self._account)
            if positions is None:
                return {}

            result: dict[str, int] = {}
            for pos in positions:
                code = str(getattr(pos, "stock_code", ""))
                if not code:
                    continue
                volume = int(getattr(pos, "volume", 0))
                if volume > 0:
                    result[code] = volume

            return result

        except (ConnectionError, RuntimeError):
            logger.exception("查询 QMT 持仓异常")
            return {}

    def get_account_info(self) -> AccountInfo:
        """查询账户资产信息。

        通过 query_stock_asset() 获取总资产、现金、市值、冻结资金。

        Returns:
            AccountInfo 包含 cash/frozen_cash/positions/available
        """
        if not self._connected or self._trader is None:
            return AccountInfo(cash=0)

        try:
            asset = self._trader.query_stock_asset(self._account)
            if asset is None:
                return AccountInfo(cash=0, positions={})

            # QMT 资产字段为 float 元，需转为 int（分, ×10000）
            cash = int(float(getattr(asset, "cash", 0)) * 10000)
            frozen_cash = int(float(getattr(asset, "frozen_cash", 0)) * 10000)

            positions = self.get_positions()

            # available = positions 中的可用部分
            available: dict[str, int] = {}
            try:
                raw_positions = self._trader.query_stock_positions(self._account)
                if raw_positions:
                    for pos in raw_positions:
                        code = str(getattr(pos, "stock_code", ""))
                        can_use = int(getattr(pos, "can_use_volume", 0))
                        if code and can_use >= 0:
                            available[code] = can_use
            except (ConnectionError, RuntimeError):
                logger.debug("查询可用持仓失败，回退到全部", exc_info=True)
                available = dict(positions)

            return AccountInfo(
                cash=cash,
                frozen_cash=frozen_cash,
                positions=positions,
                available=available,
            )

        except (ConnectionError, ValueError):
            logger.exception("查询 QMT 账户异常")
            return AccountInfo(cash=0)

    # ── 成交回调 ──────────────────────────────────────────────

    def on_trade(self, callback: Callable[[str, int, int, bool], None]) -> None:
        """注册成交回调。

        回调签名（与 BaseGateway 一致）:
          callback(symbol: str, quantity: int, price: int, is_buy: bool) -> None

        当 QMT 推送成交（委托状态变更为 51=完全成交 或 50=部分成交）时，
        _xt_trade_callback 会桥接并触发此回调。
        """
        self._on_trade_callback = callback

    def _xt_trade_callback(self, data: dict) -> None:
        """xtquant 底层回调桥接器。

        xtquant callback 参数是一个 dict，结构如下:
          {
            'order_type': 0,           # xtconstant 类型
            'order_id': '12345678',     # QMT 委托编号
            'stock_code': '000001.SZ',  # 股票代码
            'order_status': 51,         # 委托状态
            'order_volume': 100,        # 委托数量
            'traded_volume': 100,       # 已成交数量
            'traded_price': 105000,     # 成交均价 (×10000)
            'strategy_name': 'plum',    # 策略名称
            'order_remark': '...',      # 订单备注（含 plum order_id）
          }

        仅当有实际成交且已注册回调时触发。
        """
        if self._on_trade_callback is None:
            return

        try:
            order_status = int(data.get("order_status", -1))
            traded_volume = int(data.get("traded_volume", 0))

            # 仅在有实际成交时通知
            if traded_volume <= 0:
                return

            # 仅在成交/部分成交状态通知
            mapped_status = _QMT_STATUS_MAP.get(order_status)
            if mapped_status not in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILLED):
                return

            symbol = str(data.get("stock_code", ""))
            if not symbol:
                return

            traded_price = int(data.get("traded_price", 0))
            is_buy = int(data.get("order_type", 0)) in (
                23,
                24,
            )  # STOCK_BUY / 担保品买入

            callback = self._on_trade_callback
            if callable(callback):
                try:
                    callback(symbol, traded_volume, traded_price, is_buy)
                except Exception:
                    # ⚠️ 宽松捕获，原因: 用户回调可能抛出任意异常，不应影响网关正常运行
                    logger.warning("成交回调执行失败", exc_info=True)

        except Exception:
            # ⚠️ 宽松捕获，原因: QMT 回调数据结构来自 xtquant，格式可能随版本变化
            logger.debug("处理 XT 成交回调异常", exc_info=True)

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def account(self) -> str:
        """返回当前资金账号（掩码处理）。"""
        if not self.account_id:
            return "unknown"
        aid = self.account_id
        if len(aid) > 6:
            return aid[:3] + "***" + aid[-3:]
        return "***"
