"""实盘交易引擎 — 基于事件驱动的实盘交易循环。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from threading import Event, Thread
from typing import TYPE_CHECKING

from app.data.schema import Bar
from app.engine.backtest.models import Signal, SignalDirection
from app.engine.backtest.portfolio import Portfolio, Position
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.live.gateway.base import BaseGateway, Direction, OrderRequest, OrderType
from app.engine.live.risk import RiskGuard
from app.engine.live.syncer import AccountSyncer
from app.rule.trading.calculator import ASHARE_RULES

if TYPE_CHECKING:
    from app.engine.backtest.executor import ExecutorConfig

logger = logging.getLogger(__name__)


@dataclass
class LiveEngineConfig:
    """实盘交易引擎配置。"""

    heartbeat_interval: float = 60.0
    sync_interval: float = 300.0
    initial_cash: int = 40_000_0000
    max_positions: int = 8
    min_commission: float = 5.0
    watchdog_max_retries: int = 5
    watchdog_base_delay: float = 1.0


class LiveEnvironment:
    """实盘交易环境。

    集成交易网关、风控、对账器构成实盘交易循环。

    用法:
        gateway = MockGateway(initial_cash=40_000_0000)
        portfolio = Portfolio(initial_cash=Decimal("4000000"))
        env = LiveEnvironment(gateway, portfolio, commission_model)

        env.set_strategy(strategy)
        env.start(["600519.SH"])
    """

    def __init__(
        self,
        gateway: BaseGateway,
        portfolio: Portfolio,
        commission: AShareCommissionModel,
        executor_config: ExecutorConfig | None = None,
        config: LiveEngineConfig | None = None,
    ) -> None:
        self.gateway = gateway
        self.portfolio = portfolio
        self.commission = commission
        self.config = config or LiveEngineConfig()
        self.executor_config = executor_config

        self._running = False
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_sync: datetime | None = None
        self._strategy: object = None
        self._on_trade_callbacks: list[Callable] = []
        self._notify_callback: Callable | None = None
        self._risk_guard: RiskGuard | None = None
        self._syncer: AccountSyncer | None = None
        self._watchdog_failures: int = 0
        self._watchdog_paused: bool = False

    @property
    def env_type(self) -> str:
        return "live"

    def set_strategy(self, strategy: object) -> None:
        self._strategy = strategy

    def set_risk_guard(self, risk_guard: RiskGuard) -> None:
        self._risk_guard = risk_guard

    def set_syncer(self, syncer: AccountSyncer) -> None:
        self._syncer = syncer

    def set_notify_callback(self, callback: Callable) -> None:
        """设置通知回调。"""
        self._notify_callback = callback

    def on_trade(self, callback: Callable) -> None:
        self._on_trade_callbacks.append(callback)

    def _notify(self, event_type: str, **kwargs: object) -> None:
        """触发通知回调。"""
        if self._notify_callback is not None:
            try:
                self._notify_callback(event_type, **kwargs)
            except Exception as e:
                logger.warning("notify callback error: %s", e)

    def start(self, symbols: list[str]) -> None:
        """启动实盘交易循环。"""
        if self._running:
            return

        self.gateway.subscribe(symbols)
        self.gateway.on_trade(self._on_filled)
        if self._notify_callback is not None:
            self.gateway.set_notify_callback(self._notify_callback)
        connected = self.gateway.connect()
        if not connected:
            raise RuntimeError("Gateway connection failed")

        self._sync_portfolio()
        self._running = True
        self._stop_event.clear()

        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止实盘交易循环。"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.gateway.disconnect()

    def _run_loop(self) -> None:
        """实盘交易主循环。"""
        while self._running and not self._stop_event.is_set():
            try:
                self._heartbeat()
                self._sync_if_needed()
            except Exception as e:
                logger.error("LiveEngine heartbeat error: %s", e)

            self._stop_event.wait(timeout=self.config.heartbeat_interval)

    def _heartbeat(self) -> None:
        """心跳检查 + 策略信号处理。"""
        if self._risk_guard is not None and self._risk_guard.is_circuit_broken:
            self._notify(
                "circuit_break",
                reason="日损失超过阈值，交易已暂停",
            )
            self.stop()
            return

        bars = self.gateway.get_current_bars()
        if not bars:
            return

        ts = datetime.now()
        current_positions = self.gateway.get_positions()

        self._update_portfolio_positions(current_positions)

        if self._strategy is None:
            return

        if not hasattr(self._strategy, "on_bar"):
            return

        signals = self._strategy.on_bar(ts, bars, self.portfolio)

        if hasattr(self._strategy, "flush_state"):
            self._strategy.flush_state()

        target_positions = self._convert_signals(signals, bars)

        for symbol, target_qty in target_positions.items():
            current_qty = current_positions.get(symbol, 0)
            if target_qty > current_qty:
                self._buy(symbol, target_qty - current_qty, bars.get(symbol))
            elif target_qty < current_qty:
                self._sell(symbol, current_qty - target_qty, bars.get(symbol))

    def _update_portfolio_positions(self, broker_positions: dict[str, int]) -> None:
        for symbol, broker_qty in broker_positions.items():
            if symbol in self.portfolio.positions:
                local_pos = self.portfolio.positions[symbol]
                local_pos.quantity = broker_qty
                local_pos.available = broker_qty
            else:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol, quantity=broker_qty
                )

        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in broker_positions:
                del self.portfolio.positions[symbol]

    def _convert_signals(
        self,
        signals: list,
        bars: dict[str, Bar],
    ) -> dict[str, int]:
        if not signals:
            return {}

        result: dict[str, int] = {}
        portfolio_value = self.portfolio.total_value
        max_pos = (
            self.executor_config.max_positions
            if self.executor_config is not None
            else self.config.max_positions
        )

        for signal in signals:
            if not isinstance(signal, Signal):
                continue

            symbol = signal.symbol
            bar = bars.get(symbol)
            if bar is None:
                continue

            price = bar.close
            if price <= Decimal("0"):
                continue

            if signal.direction == SignalDirection.FLAT:
                result[symbol] = 0
            elif signal.direction == SignalDirection.LONG:
                per_position = portfolio_value / Decimal(max_pos)
                target_value = per_position * Decimal(str(signal.strength))
                qty_float = target_value / price
                qty = int(qty_float)
                qty = ASHARE_RULES.floor_to_lot(symbol, qty)
                if ASHARE_RULES.is_valid_lot(symbol, qty):
                    result[symbol] = qty

        return result

    def _buy(self, symbol: str, quantity: int, bar: Bar | None) -> None:
        if bar is None or quantity <= 0:
            return

        if self._risk_guard is not None:
            allowed, reason = self._risk_guard.pre_check(
                symbol=symbol,
                direction="BUY",
                quantity=quantity,
                price=int(bar.close),
                current_positions=self.gateway.get_positions(),
                portfolio_value=int(self.portfolio.total_value),
            )
            if not allowed:
                logger.warning("Buy rejected by RiskGuard: %s", reason)
                self._notify(
                    "risk_reject",
                    symbol=symbol,
                    reason=reason,
                    order_desc=f"买入 {quantity}股 MARKET",
                )
                return

        order = OrderRequest(
            symbol=symbol,
            direction=Direction.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
        self.gateway.send_order(order)

    def _sell(self, symbol: str, quantity: int, bar: Bar | None) -> None:
        if bar is None or quantity <= 0:
            return

        account = self.gateway.get_account_info()
        available_qty = account.available.get(symbol, 0)
        sell_qty = min(quantity, available_qty)

        if not ASHARE_RULES.is_valid_lot(symbol, sell_qty):
            return

        sell_qty = ASHARE_RULES.floor_to_lot(symbol, sell_qty)

        if self._risk_guard is not None:
            allowed, reason = self._risk_guard.pre_check(
                symbol=symbol,
                direction="SELL",
                quantity=sell_qty,
                price=int(bar.close),
                current_positions=account.positions,
                portfolio_value=int(self.portfolio.total_value),
            )
            if not allowed:
                logger.warning("Sell rejected by RiskGuard: %s", reason)
                self._notify(
                    "risk_reject",
                    symbol=symbol,
                    reason=reason,
                    order_desc=f"卖出 {sell_qty}股 MARKET",
                )
                return

        order = OrderRequest(
            symbol=symbol,
            direction=Direction.SELL,
            quantity=sell_qty,
            order_type=OrderType.MARKET,
        )
        self.gateway.send_order(order)

    def _on_filled(
        self,
        symbol: str,
        quantity: int,
        price: int,
        is_buy: bool,
    ) -> None:
        cost = self.commission.calculate(Decimal(str(price / 10_000)), quantity, is_buy)

        self.portfolio.apply_trade(
            symbol=symbol,
            quantity=quantity if is_buy else -quantity,
            price=Decimal(str(price / 10_000)),
            fee=cost.total,
            timestamp=datetime.now(),
        )

        for callback in self._on_trade_callbacks:
            callback(symbol, quantity, price, is_buy)

        self._notify(
            "trade", symbol=symbol, quantity=quantity, price_int=price, is_buy=is_buy
        )

    def _sync_portfolio(self) -> None:
        account = self.gateway.get_account_info()

        self.portfolio.cash = int(account.cash)
        self.portfolio.frozen_cash = int(account.frozen_cash)

        broker_positions = account.positions.copy()

        for symbol, local_pos in list(self.portfolio.positions.items()):
            broker_qty = broker_positions.get(symbol, 0)
            if broker_qty == 0:
                del self.portfolio.positions[symbol]
            else:
                local_pos.quantity = broker_qty
                local_pos.available = broker_qty

        for symbol, broker_qty in broker_positions.items():
            if symbol not in self.portfolio.positions:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol, quantity=broker_qty
                )

        self._last_sync = datetime.now()

    def _attempt_reconnect(self) -> bool:
        max_retries = self.config.watchdog_max_retries
        base_delay = self.config.watchdog_base_delay

        for attempt in range(max_retries):
            delay = base_delay * (2**attempt)
            logger.warning(
                "Watchdog reconnect attempt %d/%d in %.1fs...",
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)

            if self.gateway.connect():
                self._watchdog_failures = 0
                logger.info("Watchdog reconnect succeeded")
                return True

        logger.error("Watchdog: all %d reconnect attempts failed", max_retries)
        self._notify(
            "reconnect_fail", account_id=getattr(self.gateway, "account_id", "")
        )
        return False

    def _sync_if_needed(self) -> None:
        healthy = self.gateway.check_status()

        if not healthy and not self._watchdog_paused:
            logger.warning("Watchdog: connection lost, pausing LiveEngine")
            self._notify("conn_lost")
            self._watchdog_paused = True
            self._running = False

        if self._watchdog_paused:
            if self._attempt_reconnect():
                self._watchdog_paused = False
                self._running = True
                self._sync_portfolio()
                logger.info("Watchdog: resumed after reconnect")
            return

        self._do_sync()

    def _do_sync(self) -> None:
        if self._syncer is not None:
            result = self._syncer.sync()
            if result.needs_recovery:
                logger.warning(
                    "AccountSyncer: recovery needed, cash_diff=%d, position_diffs=%s",
                    result.cash_diff,
                    result.position_diffs,
                )

        if self._last_sync is None:
            self._sync_portfolio()
            return

        elapsed = (datetime.now() - self._last_sync).total_seconds()
        if elapsed >= self.config.sync_interval:
            self._sync_portfolio()

    @property
    def is_running(self) -> bool:
        return self._running
