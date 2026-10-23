"""Buy and Hold Strategy - Entry-Only High Conviction Trading.

实现"死拿"(Buy and Hold)策略：

1. **仅入场模式**: 仅在满足条件时生成做多信号，一旦入场永不因价格波动退出。

2. **入场逻辑追踪**: 记录入场原因，仅在逻辑被破坏时退出。

3. **退出触发条件(逻辑破坏)**:
   - 基本面：连续3年分红中断, ROE转负
   - 退市：股票进入退市流程
   - 估值：极端高估（可选）

这不是被动策略，而是高确信度策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.data.schema import Bar
from app.strategy.base import BaseStrategy, StatefulMixin

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ExitCondition:
    """触发退出的条件。"""

    condition_type: str  # "fundamental", "delisting", "valuation", "stop_loss"
    trigger_date: datetime
    reason: str
    severity: str  # "warning", "critical"


@dataclass
class PositionThesis:
    """持仓入场逻辑。"""

    symbol: str
    entry_date: datetime
    entry_price: Decimal
    entry_reason: str  # e.g., "ICBC_DEEP_VALUE", "PDD_EFFICIENCY"
    entry_score: float  # 0-100

    # 退出条件
    exit_conditions: list[ExitCondition] = field(default_factory=list)

    # 追踪逻辑是否被破坏
    thesis_broken: bool = False
    break_date: datetime | None = None
    break_reason: str | None = None

    # 财务健康追踪
    consecutive_dividend_years: int = 0  # 连续分红年数
    last_dividend_date: datetime | None = None
    min_acceptable_roe: float = 0.0  # ROE低于此值退出

    def check_fundamental_exit(
        self,
        current_date: datetime,
        dividends_paid: bool,
        current_roe: float,
    ) -> ExitCondition | None:
        """检查是否满足基本面退出条件。"""
        if not dividends_paid:
            # 分红中断
            return ExitCondition(
                condition_type="fundamental",
                trigger_date=current_date,
                reason="分红中断",
                severity="warning",
            )

        object.__setattr__(
            self, "consecutive_dividend_years", self.consecutive_dividend_years + 1
        )

        if current_roe < self.min_acceptable_roe:
            return ExitCondition(
                condition_type="fundamental",
                trigger_date=current_date,
                reason=f"ROE {current_roe:.1f}% < {self.min_acceptable_roe:.1f}%",
                severity="critical",
            )

        return None


@dataclass
class BuyAndHoldStrategy(BaseStrategy, StatefulMixin):
    """仅入场型Buy and Hold策略。

    理念：
    - 入场：极度挑剔（高阈值）
    - 持有：极度耐心（宽容忍度）
    - 退出：仅在逻辑破坏时（基本面失败）

    与传统策略的关键区别：
    - 无基于价格的退出信号
    - 无止损单
    - 无基于波动的再平衡
    - 仅在逻辑破坏时退出
    """

    # 入场阈值
    min_entry_score: int = 80  # 最低确信度分数
    min_dividend_yield: float = 0.02  # 最低2%股息率
    max_entry_pb: float = 1.5  # 最大市净率

    # 退出阈值
    consecutive_dividend_cut_years: int = 3  # 3年无分红退出
    allow_price_exit: bool = False  # 不因价格退出
    max_loss_tolerance: float = -0.50  # -50%止损（逻辑必定有误）

    # 仓位管理
    max_positions: int = 5  # 高确信度 = 更少仓位
    position_size: float = 0.20  # 每仓位20% = 最多5个仓位

    # 内部状态
    _theses: dict[str, PositionThesis] = field(default_factory=dict, repr=False)

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        """生成信号 - 仅入场，无价格退出。"""
        signals: list[Signal] = []

        # 检查退出条件（退市、基本面）
        exit_signals = self._check_exit_conditions(ts, bars, portfolio)
        signals.extend(exit_signals)

        # 检查新入场机会
        entry_signals = self._check_entry_conditions(ts, bars, portfolio)
        signals.extend(entry_signals)

        return signals

    def _check_entry_conditions(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        """检查新入场机会。"""
        signals: list[Signal] = []

        # 不要过度配置
        current_positions = len(portfolio.positions)
        if current_positions >= self.max_positions:
            return signals

        available_slots = self.max_positions - current_positions

        # 寻找候选（简化版 - 实际实现会使用因子）
        candidates = []
        for symbol, bar in bars.items():
            if symbol in portfolio.positions:
                continue  # 已持有

            # 计算入场分数（简化版）
            score = self._calculate_entry_score(symbol, bar)

            if score >= self.min_entry_score:
                candidates.append((symbol, bar, score))

        # 按分数排序，取前N个
        candidates.sort(key=lambda x: x[2], reverse=True)
        for symbol, bar, score in candidates[:available_slots]:
            signals.append(
                Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_type=SignalType.ENTRY,
                    strength=score / 100.0,
                    reason=f"HIGH_CONVICTION:{score:.0f}",
                    source="BuyAndHoldStrategy",
                    timestamp=ts,
                )
            )

            # 记录入场逻辑
            self._theses[symbol] = PositionThesis(
                symbol=symbol,
                entry_date=ts,
                entry_price=bar.close,
                entry_reason=f"SCORE_{score:.0f}",
                entry_score=score,
                consecutive_dividend_years=0,
                min_acceptable_roe=5.0,  # ROE < 5% 退出
            )

        return signals

    def _check_exit_conditions(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        """检查退出条件（仅逻辑破坏）。"""
        signals: list[Signal] = []

        for symbol in list(portfolio.positions.keys()):
            # 跳过无记录的持仓
            thesis = self._theses.get(symbol)
            if thesis is None:
                continue

            # 检查1：退市
            bar = bars.get(symbol)
            if bar and bar.close == Decimal("0"):
                thesis.thesis_broken = True
                object.__setattr__(thesis, "break_date", ts)
                object.__setattr__(thesis, "break_reason", "DELISTING")
                signals.append(
                    Signal(
                        symbol=symbol,
                        direction=SignalDirection.FLAT,
                        signal_type=SignalType.EXIT,
                        strength=1.0,
                        reason="THESIS_BREAK:DELISTING",
                        source="BuyAndHoldStrategy",
                        timestamp=ts,
                    )
                )
                continue

            # 检查2：极端亏损容忍（逻辑必定有误）
            if bar and thesis.entry_price > Decimal("0"):
                loss = (bar.close - thesis.entry_price) / thesis.entry_price
                if loss <= Decimal(str(self.max_loss_tolerance)):
                    thesis.thesis_broken = True
                    object.__setattr__(thesis, "break_date", ts)
                    object.__setattr__(
                        thesis,
                        "break_reason",
                        f"LOSS_EXCEEDED:{float(loss) * 100:.0f}%",
                    )
                    signals.append(
                        Signal(
                            symbol=symbol,
                            direction=SignalDirection.FLAT,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            reason="THESIS_BREAK:LOSS_EXCEEDED",
                            source="BuyAndHoldStrategy",
                            timestamp=ts,
                        )
                    )

        return signals

    def _calculate_entry_score(self, symbol: str, bar: Bar) -> int:
        """计算入场确信度分数。

        这是简化版本。实际实现应：
        - 使用效率因子（PDD风格股票）
        - 使用安全网因子（ICBC风格股票）
        - 使用周期因子（牧原风格股票）
        """
        score = 50  # 基础分数

        # 简化：低波动 = 更高分数
        if bar.close > Decimal("0") and bar.pre_close > Decimal("0"):
            volatility = abs(bar.close - bar.pre_close) / bar.pre_close
            if volatility < Decimal("0.02"):  # 低波动
                score += 20

        # 简化：稳定价格历史 = 更安全
        score += 20

        return min(100, score)

    def get_thesis(self, symbol: str) -> PositionThesis | None:
        """获取持仓的入场逻辑。"""
        return self._theses.get(symbol)

    def get_all_theses(self) -> dict[str, PositionThesis]:
        """获取所有记录的逻辑。"""
        return self._theses.copy()

    def get_broken_theses(self) -> list[PositionThesis]:
        """获取所有被破坏的逻辑（用于分析）。"""
        return [t for t in self._theses.values() if t.thesis_broken]

    def get_active_theses(self) -> list[PositionThesis]:
        """获取所有活跃（未破坏）的逻辑。"""
        return [t for t in self._theses.values() if not t.thesis_broken]

    def flush_state(self) -> None:
        """清除每bar缓存的状态。"""
        self._state_buffer.clear()

    @property
    def description(self) -> str:
        return (
            f"BuyAndHold(entry_score>={self.min_entry_score}, "
            f"max_loss={self.max_loss_tolerance * 100:.0f}%, "
            f"no_dividend_years={self.consecutive_dividend_cut_years})"
        )
