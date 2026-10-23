"""退市注册表 - 跟踪股票生命周期（诞生、死亡、复活）。

解决回测中的两个关键问题：

1. 僵尸持仓问题：
   不跟踪退市日期，退市股票的持仓会"冻结"在最后交易价格，
   产生永不变化的幽灵权益，污染权益曲线。

2. 幸存者偏差问题：
   如果只同步"当前存活"的股票，回测将在平行宇宙中运行，
   所有过去的赢家都在，所有输家都不可见，夸大绩效指标。

用法：
    registry = DelistingRegistry()
    registry.record_last_trading_day("000001.SZ", last_date)
    if registry.is_delisted("乐视退", date):
        block_signal()  # 禁止买入！
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config.paths import DELISTING_REGISTRY_FILE


@dataclass
class StockLifecycle:
    """股票的完整生命周期记录。"""

    symbol: str
    first_trading_day: datetime | None = None
    last_trading_day: datetime | None = None
    delisting_date: datetime | None = None  # 正式退市日期
    reason: str | None = None  # "退市整理"、"暂停上市"、"吸收合并" 等
    is_active: bool = True


@dataclass
class DelistingEvent:
    """退市事件记录。"""

    symbol: str
    last_trading_day: datetime
    delisting_date: datetime | None  # 可能与最后交易日不同
    reason: str
    residual_value: float = 0.0  # A 股退市通常为 0


class DelistingRegistry:
    """跟踪股票退市事件的注册表。

    这是回测系统的"生死簿"，记录股票的诞生、死亡和死因。

    关键特性：
    - PIT 合规：能回答"股票 X 在日期 Y 是否存活？"
    - 跟踪退市原因：区分终止类型
    - 支持复活：股票可以重新上市（IPO、借壳）
    """

    def __init__(self, registry_path: Path | str = str(DELISTING_REGISTRY_FILE)):
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        """从磁盘加载注册表。"""
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
                result = {}
                for symbol, info in raw.items():
                    result[symbol] = info.copy()
                    for key in [
                        "first_trading_day",
                        "last_trading_day",
                        "delisting_date",
                    ]:
                        if info.get(key):
                            result[symbol][key] = datetime.fromisoformat(info[key])
                return result
        return {}

    def _save(self) -> None:
        """保存注册表到磁盘。"""
        data = {}
        for symbol, info in self._data.items():
            data[symbol] = info.copy()
            for key in ["first_trading_day", "last_trading_day", "delisting_date"]:
                if data[symbol].get(key):
                    data[symbol][key] = data[symbol][key].isoformat()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_first_trading_day(self, symbol: str, date: datetime) -> None:
        """记录股票首次上市（或重新上市）日期。

        如果股票之前已退市，这处理"复活"：
        - 更新 first_trading_day 为新日期
        - 清除 last_trading_day（股票重新存活）
        """
        if symbol not in self._data:
            self._data[symbol] = {}
        self._data[symbol]["symbol"] = symbol
        self._data[symbol]["first_trading_day"] = date
        self._data[symbol]["is_active"] = True
        # 复活：清除退市信息
        if self._data[symbol].get("last_trading_day"):
            self._data[symbol]["last_trading_day"] = None
            self._data[symbol]["delisting_date"] = None
            self._data[symbol]["reason"] = None
        self._save()

    def record_last_trading_day(
        self,
        symbol: str,
        date: datetime,
        reason: str | None = None,
    ) -> None:
        """记录股票的最后交易日。

        这是最重要的方法 — 在数据同步期间调用，捕获股票停止交易的时间。

        参数:
            symbol: 股票代码（如 "000001.SZ"）
            date: 最后交易日期
            reason: 可选的退市原因
        """
        if symbol not in self._data:
            self._data[symbol] = {"symbol": symbol}

        self._data[symbol]["last_trading_day"] = date
        self._data[symbol]["delisting_date"] = date
        self._data[symbol]["reason"] = reason or "未知"
        self._data[symbol]["is_active"] = False
        self._save()

    def is_delisted(self, symbol: str, on_date: datetime) -> bool:
        """检查股票在给定日期之前或当天是否已退市。

        PIT 合规 — 回答"股票 X 在日期 Y 是否已死？"

        参数:
            symbol: 股票代码
            on_date: 要检查的日期

        返回:
            True 表示股票在 on_date 之前或当天已退市
        """
        if symbol not in self._data:
            return False  # 未知股票默认存活

        info = self._data[symbol]
        last_day = info.get("last_trading_day")
        if last_day is None:
            return False  # 仍在交易或未知

        return on_date >= last_day

    def can_trade(self, symbol: str, on_date: datetime) -> bool:
        """检查股票在给定日期是否可以交易。

        参数:
            symbol: 股票代码
            on_date: 要检查的日期

        返回:
            True 表示股票在 on_date 可以交易
        """
        if symbol not in self._data:
            return True  # 未知股票默认可交易

        info = self._data[symbol]
        first_day = info.get("first_trading_day")
        last_day = info.get("last_trading_day")

        # 股票尚未上市
        if first_day and on_date < first_day:
            return False

        # 股票已停止交易
        return not (last_day and on_date > last_day)

    def get_lifecycle(self, symbol: str) -> StockLifecycle | None:
        """获取股票的完整生命周期记录。"""
        if symbol not in self._data:
            return None

        info = self._data[symbol]
        return StockLifecycle(
            symbol=symbol,
            first_trading_day=info.get("first_trading_day"),
            last_trading_day=info.get("last_trading_day"),
            delisting_date=info.get("delisting_date"),
            reason=info.get("reason"),
            is_active=info.get("is_active", True),
        )

    def get_all_delisted(self) -> list[str]:
        """获取所有退市股票代码。"""
        return [s for s, info in self._data.items() if not info.get("is_active", True)]

    def get_all_active(self) -> list[str]:
        """获取所有当前活跃的股票代码。"""
        return [s for s, info in self._data.items() if info.get("is_active", True)]

    def merge(self, other: DelistingRegistry) -> None:
        """合并另一个注册表（用于分布式同步）。"""
        for symbol, info in other._data.items():
            if symbol not in self._data:
                self._data[symbol] = info.copy()
            else:
                # 保留信息更多的
                for key, value in info.items():
                    if value and key not in self._data[symbol]:
                        self._data[symbol][key] = value
        self._save()

    def get_delisting_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[DelistingEvent]:
        """获取日期范围内的所有退市事件。

        用于审计和报告。
        """
        events = []
        for symbol, info in self._data.items():
            if not info.get("is_active", True) and info.get("last_trading_day"):
                last_day = info["last_trading_day"]
                if start and last_day < start:
                    continue
                if end and last_day > end:
                    continue
                events.append(
                    DelistingEvent(
                        symbol=symbol,
                        last_trading_day=last_day,
                        delisting_date=info.get("delisting_date"),
                        reason=info.get("reason", "未知"),
                    )
                )
        return sorted(events, key=lambda e: e.last_trading_day)


# 便捷单例实例
_registry: DelistingRegistry | None = None


def get_registry() -> DelistingRegistry:
    """获取全局 DelistingRegistry 实例。"""
    global _registry
    if _registry is None:
        _registry = DelistingRegistry()
    return _registry
