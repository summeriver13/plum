"""成交价异常自检：捕捉「幽灵滑点」类静默故障。

背景（2026-09 排查）：回测成交价由 `Engine._get_exec_price` 按当日价格叠加滑点与
冲击后，再裁剪到以 `pre_close` 为基准的涨跌停带内。当数据层 `pre_close` 与当日
OHLC 不同基准（除权/复牌漏调，如 pre_close=1.7 而 close=0.52）时，裁剪会把成交价
强行推到涨/跌停价——最坏情况买单按跌停价成交，单笔虚构成本可达成交额的 66%。

这类故障**不报错、不抛异常**，只污染数字，因此需要事后口径自检：
正常冲击成本在 bp 级（当日成交额的 0.0x%），任何单笔滑点超过成交额 2% 都可疑。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

# 单笔滑点 / 成交额 的异常阈值：正常平方根冲击成本远低于此
SLIPPAGE_ANOMALY_RATIO = 0.02


@dataclass(frozen=True)
class SlippageAnomaly:
    """滑点异常自检结果。"""

    count: int
    cost: Decimal
    max_ratio: float

    @property
    def is_clean(self) -> bool:
        return self.count == 0

    def warning(self) -> str | None:
        """异常时返回告警文案，正常时返回 None。"""
        if self.is_clean:
            return None
        return (
            f"⚠️ 检出 {self.count} 笔异常滑点（单笔 > 成交额 "
            f"{SLIPPAGE_ANOMALY_RATIO:.0%}），合计 {self.cost:,.0f} 元，"
            f"最大单笔占比 {self.max_ratio:.2%}；"
            "疑似涨跌停裁剪遇上 pre_close 与当日价格不自洽的数据，请核查数据层"
        )


def scan_slippage_anomalies(trade_log: Sequence[Any]) -> SlippageAnomaly:
    """统计滑点占比超过阈值的成交笔数与金额。"""
    count = 0
    cost = Decimal("0")
    max_ratio = 0.0

    for trade in trade_log:
        amount = Decimal(str(getattr(trade, "amount", 0) or 0))
        if amount <= 0:
            continue
        ratio = float(Decimal(str(getattr(trade, "slippage_cost", 0) or 0)) / amount)
        max_ratio = max(max_ratio, ratio)
        if ratio > SLIPPAGE_ANOMALY_RATIO:
            count += 1
            cost += Decimal(str(getattr(trade, "slippage_cost", 0) or 0))

    return SlippageAnomaly(count=count, cost=cost, max_ratio=max_ratio)
