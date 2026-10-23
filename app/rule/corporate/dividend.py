"""上市公司行为具体处理器实现。

包括：分红处理器、送股处理器、配股处理器、拆股处理器。
使用 Decimal 保证金额和比例计算精度。
"""

from __future__ import annotations

from decimal import Decimal

from app.rule.corporate.actions import (
    ActionProcessor,
    ActionResult,
    BonusAction,
    CorporateAction,
    DividendAction,
    RightsIssueAction,
    SplitAction,
)


def _as_int(value: Decimal, default: int = 0) -> int:
    """安全转 int（金额/比例异常时回退默认值）。"""
    try:
        return int(value)
    except (ArithmeticError, TypeError, ValueError):
        return default


class DividendProcessor(ActionProcessor):
    """现金分红处理器。

    处理规则：
    - 现金增加 = 持仓股数 × 每股分红金额
    - 持仓数量不变
    - 成本均价不变
    """

    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        if not isinstance(action, DividendAction):
            return ActionResult(
                action=action, success=False, message="Invalid action type"
            )

        if quantity <= 0:
            return ActionResult(
                action=action,
                success=True,
                message="No position, dividend skipped",
            )

        dividend_amount = _as_int(
            Decimal(quantity) * action.dividend_per_share * Decimal("10000")
        )
        cash_change = Decimal(dividend_amount)
        div_yuan = action.dividend_per_share
        div_total = Decimal(dividend_amount) / Decimal("10000")

        return ActionResult(
            action=action,
            success=True,
            quantity_change=0,
            cash_change=cash_change,
            cost_basis_change=Decimal("0"),
            message=f"获得分红 {div_yuan:.2f}元/股 × {quantity}股 = {div_total:.2f}元",
        )


class BonusProcessor(ActionProcessor):
    """送股/红股处理器。

    处理规则：
    - 新增股数 = 持仓股数 × 送股比例
    - 现金不变
    - 成本均摊：总成本不变，股数增加，单位成本下降
    """

    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        if not isinstance(action, BonusAction):
            return ActionResult(
                action=action, success=False, message="Invalid action type"
            )

        if quantity <= 0:
            return ActionResult(
                action=action,
                success=True,
                message="No position, bonus skipped",
            )

        new_shares = _as_int(Decimal(quantity) * action.bonus_ratio)
        quantity_change = new_shares
        cost_basis_change = Decimal("0")

        return ActionResult(
            action=action,
            success=True,
            quantity_change=quantity_change,
            cash_change=Decimal("0"),
            cost_basis_change=cost_basis_change,
            message=f"获得送股 10送{action.bonus_ratio * 10:.0f}, 增加 {new_shares}股",
        )


class RightsIssueProcessor(ActionProcessor):
    """配股处理器。

    处理规则：
    - 可认购股数 = 持仓股数 × 配股比例
    - 认购资金 = 可认购股数 × 配股价
    - 若资金不足或 auto_subscribe=False，自动放弃
    - 配股后成本增加（以认购价买入新股）
    """

    def __init__(self, auto_subscribe: bool = True) -> None:
        """初始化配股处理器。

        参数:
            auto_subscribe: 是否自动认购。True则资金足够时自动认购，False则放弃。
        """
        self.auto_subscribe = auto_subscribe

    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        if not isinstance(action, RightsIssueAction):
            return ActionResult(
                action=action, success=False, message="Invalid action type"
            )

        if quantity <= 0:
            return ActionResult(
                action=action,
                success=True,
                message="No position, rights issue skipped",
            )

        new_shares = _as_int(Decimal(quantity) * action.ratio)
        subscription_cost = _as_int(
            Decimal(new_shares) * action.subscription_price * Decimal("10000")
        )
        cost_yuan = Decimal(subscription_cost) / Decimal("10000")

        if not self.auto_subscribe:
            return ActionResult(
                action=action,
                success=True,
                message=f"放弃配股 {new_shares}股 (需 {cost_yuan:.2f}元)",
            )

        if available_cash < subscription_cost:
            avail_yuan = Decimal(available_cash) / Decimal("10000")
            return ActionResult(
                action=action,
                success=False,
                message=(
                    f"资金不足，放弃配股 (需 {cost_yuan:.2f}元, "
                    f"可用 {avail_yuan:.2f}元)"
                ),
            )

        cash_change = -Decimal(subscription_cost)
        quantity_change = new_shares
        cost_basis_change = Decimal(subscription_cost)
        price_yuan = action.subscription_price

        return ActionResult(
            action=action,
            success=True,
            quantity_change=quantity_change,
            cash_change=cash_change,
            cost_basis_change=cost_basis_change,
            message=(
                f"认购配股 {new_shares}股 × {price_yuan:.2f}元 = {cost_yuan:.2f}元"
            ),
        )


class SplitProcessor(ActionProcessor):
    """拆股处理器。

    处理规则：
    - 新股数 = 原股数 × 拆股比例
    - 成本均摊
    """

    def process(
        self,
        action: CorporateAction,
        quantity: int,
        cost_basis: Decimal,
        available_cash: Decimal,
    ) -> ActionResult:
        if not isinstance(action, SplitAction):
            return ActionResult(
                action=action, success=False, message="Invalid action type"
            )

        if quantity <= 0:
            return ActionResult(
                action=action,
                success=True,
                message="No position, split skipped",
            )

        new_quantity = _as_int(Decimal(quantity) * action.split_ratio)
        quantity_change = new_quantity - quantity
        split_ratio = action.split_ratio
        cost_basis_change = Decimal("0")

        return ActionResult(
            action=action,
            success=True,
            quantity_change=quantity_change,
            cash_change=Decimal("0"),
            cost_basis_change=cost_basis_change,
            message=(f"拆股 1拆{split_ratio:.0f}, {quantity}股 → {new_quantity}股"),
        )
