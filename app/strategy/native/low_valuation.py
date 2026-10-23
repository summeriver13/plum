"""Plum 原生策略 — 只依赖数据层与因子框架。

直接从 Bar 数据模型读取估值字段，执行多因子选股，
不经过 DataFrame 中间层与外部因子列计算。

实现路径:
    Bar.pe_ttm / Bar.pb_mrq → 直接计算 → 排名 → Signal

优势:
    - 零外部依赖，纯 Plum 生态
    - 可被 FactorEngine 调度，FactorStore 持久化
    - 类型安全（Decimal 计算，无 DataFrame 中间层）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.data.schema import Bar
from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.strategy.base import BaseStrategy, RebalanceMixin


@dataclass
class LowValuationNative(BaseStrategy, RebalanceMixin):
    """低估值策略 — Plum 原生实现。

    EP + BP 双因子等权排名选股。
    直接读 Bar 的 pe_ttm / pb_mrq，无需 DataFrame 中间层。

    因子:
        EP = 1.0 / PE_TTM  — 盈利收益率（高=便宜）
        BP = 1.0 / PB_MRQ  — 市净率倒数（高=便宜）
        composite = EP + BP（等权加总，降序选股）

    用法:
        strategy = LowValuationNative(
            strategy_id="low_val_native",
            select_num=10,
            hold_period="W",
        )
        engine.run(symbols, start, end, strategy)
    """

    strategy_id: str = "low_valuation_native"
    select_num: int = 10
    hold_period: str = "W"
    offset: int = 0
    cap_weight: float = 1.0

    # BaseStrategy / StatefulMixin
    warmup_period: int = 0
    watched_symbols: set[str] = field(default_factory=set)
    _state: dict = field(default_factory=dict, init=False, repr=False)
    _state_buffer: dict = field(default_factory=dict, init=False, repr=False)
    _last_updated: datetime | None = field(default=None, init=False, repr=False)

    # RebalanceMixin 内部状态
    _business_day_count: int = field(default=0, init=False, repr=False)
    _period_start_count: int = field(default=-1, init=False, repr=False)
    _last_rebalance_ts: datetime | None = field(default=None, init=False, repr=False)
    _last_result: bool = field(default=False, init=False, repr=False)

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Any,
    ) -> list[Signal]:
        """每个 bar 的回调 — 纯 Plum 原生因子计算。"""
        if not self.is_rebalance_date(ts):
            return []

        # ── Step 1: 计算 EP + BP 复合得分 ──
        scores: list[tuple[str, float, float, float]] = []
        for symbol, bar in bars.items():
            ep = 0.0
            bp = 0.0

            if bar.pe_ttm is not None and bar.pe_ttm > 0:
                ep = float(Decimal("1.0") / bar.pe_ttm)
            if bar.pb_mrq is not None and bar.pb_mrq > 0:
                bp = float(Decimal("1.0") / bar.pb_mrq)

            if ep > 0 or bp > 0:
                composite = ep + bp
                scores.append((symbol, composite, ep, bp))

        if not scores:
            return []

        # ── Step 2: 按复合得分降序（分越高越便宜） ──
        scores.sort(key=lambda x: x[1], reverse=True)

        # ── Step 3: 取前 select_num ──
        selected = scores[: self.select_num]
        selected_set = {s[0] for s in selected}

        # ── Step 4: 生成信号 ──
        current_set = set(portfolio.positions.keys())
        signals: list[Signal] = []

        # 卖出：当前持仓但不在新选股列表中
        for symbol in current_set - selected_set:
            signals.append(
                Signal(
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    signal_type=SignalType.EXIT,
                    reason="调出选股池",
                    source=self.strategy_id,
                    timestamp=ts,
                )
            )

        # 买入：新选入但不在当前持仓中
        for symbol, composite, ep, bp in selected:
            if symbol in current_set:
                continue
            signals.append(
                Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_type=SignalType.ENTRY,
                    strength=min(1.0, max(0.1, composite / max(1.0, scores[0][1]))),
                    reason=f"EP={ep:.4f} BP={bp:.4f}",
                    source=self.strategy_id,
                    timestamp=ts,
                )
            )

        return signals

    def flush_state(self) -> None:
        pass

    def get_state(self) -> dict:
        return {"strategy_id": self.strategy_id, "data": self._state}

    def set_state(self, state: dict) -> None:
        self._state = state.get("data", {})
