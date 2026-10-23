"""资金不足保护 — 检查买入资金是否充足。"""

from __future__ import annotations

from decimal import Decimal


class CashGuard:
    """资金不足检查器。

    当买入资金不足时，计算最大可买数量。
    """

    def is_affordable(
        self,
        price: Decimal,
        quantity: int,
        available_cash: Decimal,
        fee_rate: Decimal = Decimal("0.00022"),
        min_commission: Decimal = Decimal("5.0"),
    ) -> tuple[bool, int]:
        """检查买入是否资金充足。

        参数:
            price: 成交价格
            quantity: 请求买入数量
            available_cash: 可用资金
            fee_rate: 综合费率
            min_commission: 最低佣金

        返回:
            (是否充足, 最大可买数量)
        """
        if quantity <= 0:
            return True, 0

        amount = Decimal(quantity) * price
        fee = max(amount * fee_rate, min_commission)
        total = amount + fee

        if total <= available_cash:
            return True, quantity

        # 计算最大可买数量（按 100 股整数倍）
        if price > Decimal("0"):
            max_qty = int(available_cash / (price * (Decimal("1") + fee_rate)))
            max_qty = (max_qty // 100) * 100
            return False, max_qty

        return False, 0
