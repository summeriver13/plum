"""A 股交易成本模型。

个股（现值，保持既有回测口径不变）：

- 佣金：万分之二（最低 5 元）
- 过户费：万分之零点二
- 印花税：万分之五（仅卖出收取）

ETF（``EtfCommissionModel``）：

- 佣金：万分之零点五，**无最低收费**（多数券商的 ETF 口径；若按最低 5 元，
  1M 规模下单笔约 5 万元的委托会变成万分之一，成本直接翻倍）
- 过户费：0（ETF 不收过户费）
- 印花税：0（ETF 免印花税）

费率用 ``CommissionRules`` 参数化：既能按券商口径调整，也能在
成本归因实验里传零费率模型做对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.data.schema import CostBreakdown


@dataclass(frozen=True)
class CommissionRules:
    """费用规则参数（费率均为 Decimal 比例）。"""

    commission_rate: Decimal
    min_commission: Decimal
    transfer_fee_rate: Decimal
    stamp_duty_rate: Decimal


ASHARE_RULES = CommissionRules(
    commission_rate=Decimal("0.0002"),
    min_commission=Decimal("5.0"),
    transfer_fee_rate=Decimal("0.00002"),
    stamp_duty_rate=Decimal("0.0005"),
)

ETF_RULES = CommissionRules(
    commission_rate=Decimal("0.00005"),
    min_commission=Decimal("0"),
    transfer_fee_rate=Decimal("0"),
    stamp_duty_rate=Decimal("0"),
)

ZERO_RULES = CommissionRules(
    commission_rate=Decimal("0"),
    min_commission=Decimal("0"),
    transfer_fee_rate=Decimal("0"),
    stamp_duty_rate=Decimal("0"),
)


class AShareCommissionModel:
    """A 股交易成本计算（默认个股费率，可传入其他规则）。"""

    # 兼容旧调用对类常量的引用（口径即 ASHARE_RULES）
    COMMISSION_RATE = ASHARE_RULES.commission_rate
    MIN_COMMISSION = ASHARE_RULES.min_commission
    TRANSFER_FEE_RATE = ASHARE_RULES.transfer_fee_rate
    STAMP_DUTY_RATE = ASHARE_RULES.stamp_duty_rate

    def __init__(self, rules: CommissionRules = ASHARE_RULES) -> None:
        self.rules = rules

    def calculate(
        self,
        price: Decimal,
        quantity: int,
        is_buy: bool,
    ) -> CostBreakdown:
        """计算交易成本。

        参数:
            price: 成交价格（元）
            quantity: 交易数量（正数）
            is_buy: 是否买入

        返回:
            CostBreakdown 成本明细
        """
        amount = Decimal(quantity) * price
        rules = self.rules

        commission = max(amount * rules.commission_rate, rules.min_commission)
        transfer_fee = amount * rules.transfer_fee_rate
        stamp_duty = Decimal("0") if is_buy else amount * rules.stamp_duty_rate

        return CostBreakdown(
            commission=commission,
            transfer_fee=transfer_fee,
            stamp_duty=stamp_duty,
        )


class EtfCommissionModel(AShareCommissionModel):
    """ETF 交易成本：免印花税、免过户费、佣金万 0.5 无最低。"""

    def __init__(self, rules: CommissionRules = ETF_RULES) -> None:
        super().__init__(rules)
