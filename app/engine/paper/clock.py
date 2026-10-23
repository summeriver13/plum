"""模拟交易时钟 — 使用系统时间并识别交易时段。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from app.rule.trading.calculator import ASHARE_RULES


@dataclass
class PaperClockConfig:
    """模拟交易时钟配置。"""

    trading_hours_only: bool = True
    tick_interval: float = 3.0
    market_open_hour: int = 9
    market_close_hour: int = 15


class PaperClock:
    """模拟交易实时时钟。

    使用系统时间但遵守 A 股交易时段。
    """

    def __init__(
        self,
        config: PaperClockConfig | None = None,
    ) -> None:
        self.config = config or PaperClockConfig()

    def now(self) -> datetime:
        """获取当前系统时间。"""
        return datetime.now()

    def next_bar_time(self) -> datetime | None:
        """下一次心跳时间。"""
        return self.now()

    def is_market_open(self, dt: datetime) -> bool:
        """检查当前是否在交易时段。"""
        if not ASHARE_RULES.is_trading_day(dt.date()):
            return False

        if self.config.trading_hours_only:
            hour = dt.hour
            minute = dt.minute

            morning_open = self.config.market_open_hour * 60
            morning_close = 11 * 60 + 30
            afternoon_open = 13 * 60
            afternoon_close = self.config.market_close_hour * 60

            current_minutes = hour * 60 + minute

            is_morning = morning_open <= current_minutes <= morning_close
            is_afternoon = afternoon_open <= current_minutes < afternoon_close

            return is_morning or is_afternoon

        return True

    def wait_until(self, dt: datetime) -> None:
        """等待到指定时间。"""
        now = self.now()
        if dt > now:
            sleep_seconds = (dt - now).total_seconds()
            if sleep_seconds > 0:
                time.sleep(min(sleep_seconds, 1.0))

    def sleep(self, seconds: float) -> None:
        """休眠指定秒数。"""
        time.sleep(seconds)
