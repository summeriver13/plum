"""钉钉机器人通知 — 交易层事件告警。

事件类型与触发条件:

    on_trade        — 成交回报 (买入/卖出完成)
    on_order_error  — 委托被拒 (柜台拒绝/提交失败)
    on_cancel_fail  — 撤单失败
    on_conn_lost    — 连接断开
    on_reconnect_fail — 重连失败
    on_circuit_break — 熔断触发 (风控阈值)
    on_risk_reject  — 风控拒绝 (下单前被 RiskGuard 拦截)
    on_sync_alert   — 持仓对账偏差告警
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EventSeverity(Enum):
    """通知严重等级。"""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DingTalkNotifier:
    """钉钉机器人通知器。

    线程安全。所有通知通过独立线程发送，不会阻塞交易主循环。
    网络失败时静默记录日志，不影响交易。
    """

    webhook_url: str
    """钉钉机器人 Webhook URL。"""

    enabled: bool = True
    """禁用时静默。"""

    _log_buys: bool = field(default=True)
    _log_sells: bool = field(default=True)
    _keyword: str = "Plum"

    def _send(self, title: str, content: str, severity: EventSeverity) -> bool:
        """发送 Markdown 消息到钉钉。"""
        if not self.enabled or not self.webhook_url:
            return True

        import threading

        signed_content = f"{content}\n\n> **{self._keyword}**"

        def _http_post() -> None:
            try:
                payload = json.dumps(
                    {
                        "msgtype": "markdown",
                        "markdown": {
                            "title": title,
                            "text": signed_content,
                        },
                    }
                ).encode("utf-8")

                req = urllib.request.Request(
                    self.webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read().decode("utf-8")
                    if "errmsg" in body and "ok" not in body.lower():
                        logger.warning("DingTalk send failed: %s", body)
                    else:
                        logger.debug("DingTalk sent: %s", title)
            except Exception as e:
                logger.warning("DingTalk notify error: %s", e)

        t = threading.Thread(target=_http_post, daemon=True)
        t.start()
        return True

    def _fmt_price(self, price_int: int) -> str:
        return f"{price_int / 10_000:.4f}"

    def _fmt_value(self, amount_int: int) -> str:
        return f"{amount_int / 10_000:.2f}"

    def _fmt_time(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def on_trade(
        self, symbol: str, quantity: int, price_int: int, is_buy: bool
    ) -> None:
        """成交回报。"""
        if is_buy and not self._log_buys:
            return
        if not is_buy and not self._log_sells:
            return

        direction = "买入" if is_buy else "卖出"
        title = f"{direction} 成交"
        amount = quantity * price_int
        content = (
            f"## {title}\n\n"
            f"- **标的**: `{symbol}`\n"
            f"- **方向**: {direction}\n"
            f"- **数量**: {quantity} 股\n"
            f"- **成交价**: {self._fmt_price(price_int)} 元\n"
            f"- **成交额**: {self._fmt_value(amount)} 元\n"
            f"- **时间**: {self._fmt_time()}\n"
        )
        self._send(title, content, EventSeverity.INFO)

    def on_order_error(self, symbol: str, reason: str) -> None:
        """委托被拒。"""
        title = "委托被拒"
        content = (
            f"## {title}\n\n"
            f"- **标的**: `{symbol}`\n"
            f"- **原因**: `{reason}`\n"
            f"- **时间**: {self._fmt_time()}\n"
        )
        self._send(title, content, EventSeverity.ERROR)

    def on_circuit_break(self, reason: str) -> None:
        """熔断触发。"""
        title = "熔断触发"
        content = (
            f"## {title}\n\n"
            f"- **原因**: `{reason}`\n"
            f"- **时间**: {self._fmt_time()}\n"
            f"- **状态**: 交易已暂停，请人工介入\n"
        )
        self._send(title, content, EventSeverity.CRITICAL)

    def on_risk_reject(self, symbol: str, reason: str, order_desc: str) -> None:
        """风控拒绝。"""
        title = "风控拒绝"
        content = (
            f"## {title}\n\n"
            f"- **标的**: `{symbol}`\n"
            f"- **委托**: {order_desc}\n"
            f"- **拒绝原因**: `{reason}`\n"
            f"- **时间**: {self._fmt_time()}\n"
        )
        self._send(title, content, EventSeverity.WARN)

    def on_sync_alert(self, alert_type: str, message: str) -> None:
        """对账偏差告警。"""
        title = f"对账告警: {alert_type}"
        content = (
            f"## {title}\n\n- **详情**: {message}\n- **时间**: {self._fmt_time()}\n"
        )
        self._send(title, content, EventSeverity.WARN)

    def on_event(self, event_type: str, **kwargs: object) -> None:
        """统一事件入口，按 event_type 分发。"""
        dispatch = {
            "trade": self.on_trade,
            "order_error": self.on_order_error,
            "circuit_break": self.on_circuit_break,
            "risk_reject": self.on_risk_reject,
            "sync_alert": self.on_sync_alert,
        }
        handler = dispatch.get(event_type)
        if handler:
            handler(**kwargs)
