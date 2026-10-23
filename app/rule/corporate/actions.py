"""上市公司行为数据类和处理器基类。

定义行为类型枚举、行为数据结构、结果数据类和处理器基类。

金额单位：所有金额使用 int ×10⁴ 表示（如 105500 = 10.55元）
比例和价格使用 Decimal 保证精度。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ActionType(Enum):
    """上市公司行为类型。

    - DIVIDEND: 现金分红（除息）
    - BONUS: 送股/红股（除权）
    - RIGHTS_ISSUE: 配股
    - SPLIT: 拆股
    - MERGE: 并股
    """

    DIVIDEND = "dividend"
    BONUS = "bonus"
    RIGHTS_ISSUE = "rights_issue"
    SPLIT = "split"
    MERGE = "merge"


@dataclass(frozen=True)
class CorporateAction(ABC):
    """上市公司行为基类。

    属性:
        symbol: 股票代码
        ex_date: 除权除息日
        pub_date: 公告日期
        event_type: 行为类型
    """

    symbol: str
    ex_date: datetime
    pub_date: datetime
    event_type: ActionType

    @property
    @abstractmethod
    def description(self) -> str:
        """人类可读的描述。"""
        pass


@dataclass(frozen=True)
class DividendAction(CorporateAction):
    """现金分红事件（除息）。

    示例: 10派5元 → dividend_per_share = Decimal("0.5")
    处理: 现金增加 = 持仓股数 × 每股分红金额
    """

    dividend_per_share: Decimal

    def __post_init__(self) -> None:
        if self.event_type != ActionType.DIVIDEND:
            raise ValueError("Event type must be DIVIDEND")

    @property
    def description(self) -> str:
        return f"现金分红 {self.dividend_per_share:.2f}元/股"


@dataclass(frozen=True)
class BonusAction(CorporateAction):
    """送股/红股事件（除权）。

    示例: 10送5 → bonus_ratio = Decimal("0.5")
    处理: 股数增加，成本均摊
    """

    bonus_ratio: Decimal

    def __post_init__(self) -> None:
        if self.event_type != ActionType.BONUS:
            raise ValueError("Event type must be BONUS")

    @property
    def description(self) -> str:
        return f"送股 10送{self.bonus_ratio * 10:.0f}"


@dataclass(frozen=True)
class RightsIssueAction(CorporateAction):
    """配股事件。

    示例: 10配3，配股价8元 → ratio=Decimal("0.3"), subscription_price=Decimal("8.0")
    处理: 认购则增加股数和成本，放弃则成本被稀释
    """

    subscription_price: Decimal
    ratio: Decimal
    is_renounceable: bool = True

    def __post_init__(self) -> None:
        if self.event_type != ActionType.RIGHTS_ISSUE:
            raise ValueError("Event type must be RIGHTS_ISSUE")

    @property
    def description(self) -> str:
        return f"配股 10配{self.ratio * 10:.0f}, 配股价 {self.subscription_price:.2f}元"


@dataclass(frozen=True)
class SplitAction(CorporateAction):
    """拆股事件。

    示例: 1拆2 → split_ratio = Decimal("2.0")
    处理: 股数增加，成本均摊
    """

    split_ratio: Decimal

    def __post_init__(self) -> None:
        if self.event_type != ActionType.SPLIT:
            raise ValueError("Event type must be SPLIT")

    @property
    def description(self) -> str:
        return f"拆股 1拆{self.split_ratio:.0f}"


@dataclass
class ActionResult:
    """行为处理结果。

    属性:
        action: 关联的行为
        success: 是否成功处理
        quantity_change: 股数变化（正为增加，负为减少）
        cash_change: 现金变化（int ×10⁴）
        cost_basis_change: 成本变化（int ×10⁴）
        message: 处理说明
    """

    action: CorporateAction
    success: bool
    quantity_change: int = 0
    cash_change: Decimal = Decimal("0")
    """现金变化（元）。"""
    cost_basis_change: Decimal = Decimal("0")
    """成本变化（元）。"""
    message: str = ""

    @property
    def description(self) -> str:
        """人类可读的处理结果描述。"""
        parts = []
        if self.quantity_change != 0:
            sign = "+" if self.quantity_change > 0 else ""
            parts.append(f"股数 {sign}{self.quantity_change}")
        if self.cash_change != 0:
            sign = "+" if self.cash_change > 0 else ""
            cash_yuan = self.cash_change / 10000
            parts.append(f"现金 {sign}{cash_yuan:.2f}元")
        if self.cost_basis_change != 0:
            sign = "+" if self.cost_basis_change > 0 else ""
            parts.append(f"成本 {sign}{self.cost_basis_change}")
        return ", ".join(parts) if parts else "无变化"


class ActionProcessor(ABC):
    """行为处理器抽象基类。

    实现此基类以创建特定行为的处理逻辑。
    """

    @abstractmethod
    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        """处理一个行为事件。

        参数:
            action: 行为事件
            quantity: 当前持仓股数
            cost_basis: 当前成本（元）
            available_cash: 可用资金（元）

        返回:
            ActionResult 处理结果
        """
        pass


class CorporateActionProcessor:
    """上市公司行为主处理器。

    管理多种行为处理器的注册和调用。

    使用示例：
        processor = CorporateActionProcessor()
        processor.register(ActionType.DIVIDEND, DividendProcessor())
        result = processor.process(action, quantity=1000, cost_basis=105500, available_cash=50000)
    """

    def __init__(self) -> None:
        self._processors: dict[ActionType, ActionProcessor] = {}
        self._action_log: list[ActionResult] = []

    def register(self, action_type: ActionType, processor: ActionProcessor) -> None:
        """注册一个行为处理器。

        参数:
            action_type: 行为类型
            processor: 对应的处理器
        """
        self._processors[action_type] = processor

    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        """处理一个行为事件。

        参数:
            action: 行为事件
            quantity: 当前持仓股数
            cost_basis: 当前成本（元）
            available_cash: 可用资金（元）

        返回:
            ActionResult 处理结果
        """
        processor = self._processors.get(action.event_type)
        if processor is None:
            result = ActionResult(
                action=action,
                success=False,
                message=f"No processor for {action.event_type.value}",
            )
        else:
            result = processor.process(action, quantity, cost_basis, available_cash)

        self._action_log.append(result)
        return result

    def get_action_log(self) -> list[ActionResult]:
        """获取所有已处理的行为记录。

        返回:
            ActionResult 列表
        """
        return self._action_log.copy()

    def clear_log(self) -> None:
        """清空行为记录。"""
        self._action_log.clear()
