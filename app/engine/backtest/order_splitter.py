"""订单拆单器 — TWAP 固定金额拆单。

生产环境用 TWAP 算法将大单拆成小额子单，分散到时间段执行。
回测环境下拆单主要为保持与实盘逻辑一致。

阈值逻辑:
    - 单笔金额 < min_split_value → 不拆单
    - 单笔金额 ≥ min_split_value → 拆成每笔 split_amount 的子单
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal

from app.data.schema import Order


def _as_int(value: object, default: int = 0) -> int:
    """安全转 int（非数值/非有限值回退默认值）。"""
    try:
        number = Decimal(str(value))
        return int(number) if number.is_finite() else default
    except Exception:  # noqa: BLE001 - 配置脏值不应中断回测
        return default


@dataclass
class SplitConfig:
    """拆单配置。"""

    min_split_value: Decimal = Decimal("50000")  # 低于此金额不拆单
    split_amount: Decimal = Decimal("8000")  # 每笔子单金额
    split_jitter: Decimal = Decimal("2000")  # 金额随机抖动范围


@dataclass
class OrderSplitter:
    """订单拆单器。

    用法:
        splitter = OrderSplitter()
        sub_orders = splitter.split(order, price)
    """

    config: SplitConfig = field(default_factory=SplitConfig)

    def should_split(self, order: Order, price: Decimal) -> bool:
        """判断是否需要拆单。"""
        order_value = Decimal(abs(order.qty)) * price
        return order_value >= self.config.min_split_value

    def split(self, order: Order, price: Decimal) -> list[Order]:
        """将大单拆解为小单列表。

        Args:
            order: 原始订单
            price: 当前价格

        Returns:
            子订单列表（含原始订单本身，当不需要拆单时返回单元素列表）
        """
        if not self.should_split(order, price):
            return [order]

        total_qty = abs(order.qty)
        remaining = total_qty
        sub_orders = []
        seq = 0

        while remaining > 0:
            # 随机每笔金额，在 split_amount ± split_jitter 范围内
            jitter = random.randint(
                -_as_int(self.config.split_jitter),
                _as_int(self.config.split_jitter),
            )
            target_amount = max(100, _as_int(self.config.split_amount) + jitter)
            target_qty = _as_int(Decimal(target_amount) / price)
            target_qty = (target_qty // 100) * 100  # 整手
            target_qty = max(100, min(target_qty, remaining))

            sub_orders.append(
                Order(
                    order_id=f"{order.order_id}_SPLIT_{seq}",
                    symbol=order.symbol,
                    qty=target_qty * (1 if order.qty > 0 else -1),
                    created_ts=order.created_ts,
                )
            )
            seq += 1
            remaining -= target_qty

        return sub_orders


def split_orders(orders: list[Order], price: Decimal) -> list[Order]:
    """便捷函数：对订单列表执行拆单。

    Args:
        orders: 原始订单列表
        price: 当前价格

    Returns:
        拆单后的订单列表
    """
    splitter = OrderSplitter()
    result = []
    for order in orders:
        result.extend(splitter.split(order, price))
    return result
