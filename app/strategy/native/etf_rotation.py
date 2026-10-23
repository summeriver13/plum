"""ETF 双动量轮动策略（目标数量模式）。

设计口径与理由（改之前先看完，每条都对应一次实测结论）：

价格口径
    一律用后复权价 ``close × adj_factor`` 计算动量与均线。ETF 有分红，也有
    **份额合并**（实测 21,398 行因子 < 1，如 159919 在 2012-12-03 由 0.813 跳到 2.108），
    用原始价会把合并当天算成 +160% 的假暴涨并污染动量排名。

避险资产
    避险名额**留空仓**，由引擎按 ``cash_yield`` 计提停泊收益（与多因子小市值策略同口径）。
    不把货币 ETF 当避险资产：511990/159001 的分红未被 QMT 口径捕捉（后复权年化 0.00%），
    511880 虽可用（2.53%）但显式现金假设更透明，也便于与微盘策略横向比较。

风险资产池
    剔除低波动品种（货币/债券 ETF）：20 日波动率 < ``min_vol`` 者不进候选。
    实测 2026 年流动性达标的 1,322 只 ETF 中有 368 只属此类。

标的池口径（可选）
    默认全池（与 v1~v3 批次可比）。开启 ``ref_min_corr`` 后按「与 A 股宽基参照 ETF
    的 60 日收益相关性」分类，只留 A 股股票型（宽基/行业/主题），排除跨境/商品/债——
    依据是 TSMOM 面板诊断：时序 edge 只存在于 A 股股票型与黄金，跨境/商品为负。

同质标的去重
    同一指数的多只 ETF 动量排名会挤在一起（如 3 只科创 50），若不去重，
    top-K 就退化成单一风格的杠杆下注；故用 60 日收益相关性 > ``max_corr`` 的判为同质，
    只保留动量更高的那一只（实测同一指数不同 ETF 的相关性普遍 > 0.98）。

空仓名额即风险预算
    未通过绝对趋势过滤（后复权价 < MA×``trend_buffer``）的名额不填补，直接留现金，
    于是"通过过滤的标的数"天然形成连续降档，无需另设暴露序列。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.strategy.native.etf_common import (
    LOT_SIZE,
    MIN_TRADE_RATIO,
    _int,
    _num,
    _split_ratio,
)
from app.strategy.native.etf_selection import (
    DEFAULT_REF_CORR_WINDOW,
    DEFAULT_REF_SYMBOLS,
    EtfSelectionMixin,
)

DEFAULT_SELECT_NUM = 3
DEFAULT_HOLD_DAYS = 10
DEFAULT_MOMENTUM_WINDOW = 60
DEFAULT_MA_WINDOW = 120
DEFAULT_MIN_AMOUNT = 1.0e7  # 20 日均成交额门槛：1000 万元
DEFAULT_MIN_VOL = 0.005  # 20 日波动率门槛：0.5%（排除货币/债券类）
DEFAULT_MAX_CORR = 0.95


class EtfRotationStrategy(EtfSelectionMixin):
    """ETF 双动量轮动：相对动量排名选强 + 绝对趋势过滤 + 未入选名额留现金。"""

    def __init__(
        self,
        strategy_id: str = "etf_rotation",
        select_num: int = DEFAULT_SELECT_NUM,
        hold_days: int = DEFAULT_HOLD_DAYS,
        momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
        ma_window: int = DEFAULT_MA_WINDOW,
        min_amount: float = DEFAULT_MIN_AMOUNT,
        min_vol: float = DEFAULT_MIN_VOL,
        trend_buffer: float = 0.0,
        require_trend: bool = True,
        reversal: bool = False,
        max_corr: float = DEFAULT_MAX_CORR,
        universe: Sequence[str] = (),
        ref_symbols: Sequence[str] = DEFAULT_REF_SYMBOLS,
        ref_corr_window: int = DEFAULT_REF_CORR_WINDOW,
        ref_min_corr: float = 0.0,
        target_exposure: float = 1.0,
        min_trade_ratio: float = MIN_TRADE_RATIO,
        warmup_until: datetime | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.select_num = select_num
        self.hold_days = max(1, hold_days)
        self.momentum_window = momentum_window
        self.ma_window = ma_window
        self.min_amount = min_amount
        self.min_vol = min_vol
        self.trend_buffer = trend_buffer
        self.require_trend = require_trend
        self.reversal = reversal
        self.max_corr = max_corr
        self.universe = tuple(universe)
        self.ref_symbols = tuple(ref_symbols)
        self.ref_corr_window = ref_corr_window
        self.ref_min_corr = ref_min_corr
        self.target_exposure = target_exposure
        self.min_trade_ratio = min_trade_ratio
        self._adj_close: dict[str, list[float]] = {}
        self._amount: dict[str, list[float]] = {}
        self._raw_close: dict[str, list[float]] = {}
        self._targets: dict[str, int] = {}
        self._day = 0
        if warmup_until is not None:
            self._load_warmup(warmup_until)
        self._load_refs()

    # ------------------------------------------------------------------ 主循环

    def on_bar(
        self, ts: datetime, bars: dict[str, Any], portfolio: Any
    ) -> dict[str, int]:
        """每交易日：先同步拆分日目标，再登记历史，最后按调仓周期给出目标持仓。"""
        self._sync_targets_on_split(bars)
        self._record(bars)
        self._record_refs(ts)
        self._day += 1
        if self._day % self.hold_days == 0:
            self._targets = self._rebalance(bars, portfolio)
        return self._targets

    def _sync_targets_on_split(self, bars: dict[str, Any]) -> None:
        """拆分/合并日同步缓存的目标股数。

        引擎已在当日按同一比例复权持仓数量；若目标股数不跟着变，
        下次比较会把「复权后的持仓」当成超配，反向卖出多余的份额
        （实测 159582 的 5:1 拆分会多卖 80% 仓位）。
        """
        for symbol, target in list(self._targets.items()):
            closes = self._raw_close.get(symbol)
            bar = bars.get(symbol)
            if not closes or bar is None or target <= 0:
                continue
            ratio = _split_ratio(closes[-1], _num(getattr(bar, "pre_close", 0.0)))
            if ratio is None:
                continue
            self._targets[symbol] = _int(target * ratio)

    def flush_state(self) -> None:
        """无内部待刷状态（满足引擎协议）。"""
        return None

    # ------------------------------------------------------------------ 历史登记

    def _record(self, bars: dict[str, Any]) -> None:
        """登记当日后复权价与成交额；停牌日（无 K 线）不登记，均线按实际交易日滚动。"""
        for symbol, bar in bars.items():
            price = _num(bar.close) * _num(getattr(bar, "adj_factor", 1.0), 1.0)
            if price <= 0 or not math.isfinite(price):
                continue
            self._adj_close.setdefault(symbol, []).append(price)
            self._amount.setdefault(symbol, []).append(_num(bar.amount))
            self._raw_close.setdefault(symbol, []).append(_num(bar.close))

    def _warmup(self) -> int:
        """建仓所需最少历史长度（动量窗与均线窗取大者，再加 1）。"""
        return max(self.momentum_window + 1, self.ma_window + 1)

    # ------------------------------------------------------------------ 选股

    def _ranked_candidates(self) -> list[str]:
        """按动量排序的候选池（已过标的池/流动性/波动率/趋势过滤）。

        ``reversal=True`` 时排序反向（买动量最弱的），用于验证「短周期反转
        主导」的假设：实测 60 日动量选前 3 名的前向 10 日收益中位 -0.24%，
        低于全池中位 +0.16%。
        """
        ranked: list[tuple[str, float]] = []
        for symbol, closes in self._adj_close.items():
            if len(closes) < self._warmup():
                continue
            if not self._passes_universe(symbol):
                continue
            if self.require_trend and not self._passes_trend(symbol):
                continue
            if not self._is_liquid(symbol):
                continue
            base = closes[-self.momentum_window - 1]
            if base <= 0:
                continue
            ranked.append((symbol, closes[-1] / base - 1.0))
        ranked.sort(key=lambda item: item[1], reverse=not self.reversal)
        return [symbol for symbol, _ in ranked]

    def _fill_slots(self, ranked: list[str]) -> list[str]:
        """按排名填名额，跳过高相关同质标的。"""
        slots: list[str] = []
        for symbol in ranked:
            if len(slots) >= self.select_num:
                break
            if self._is_homogeneous(symbol, slots):
                continue
            slots.append(symbol)
        return slots

    # ------------------------------------------------------------------ 目标持仓

    def _rebalance(self, bars: dict[str, Any], portfolio: Any) -> dict[str, int]:
        """调仓日：算出目标持仓 dict（先退出、再减仓、最后加仓）。"""
        portfolio_value = _num(getattr(portfolio, "total_value", 0.0))
        if portfolio_value <= 0:
            return self._current_targets(portfolio)
        slots = self._fill_slots(self._ranked_candidates())
        slot_value = portfolio_value * self.target_exposure / max(1, self.select_num)
        desired: dict[str, int] = {}
        for symbol in slots:
            bar = bars.get(symbol)
            price = _num(getattr(bar, "close", 0.0)) if bar else 0.0
            if price <= 0:
                continue
            quantity = _int(slot_value / price) // LOT_SIZE * LOT_SIZE
            if quantity > 0:
                desired[symbol] = quantity
        return self._ordered_targets(desired, bars, portfolio, portfolio_value)

    def _current_targets(self, portfolio: Any) -> dict[str, int]:
        """组合估值异常时的兜底：维持现有持仓不动。"""
        return {
            symbol: _int(position.quantity)
            for symbol, position in portfolio.positions.items()
            if position.quantity
        }

    def _ordered_targets(
        self,
        desired: dict[str, int],
        bars: dict[str, Any],
        portfolio: Any,
        portfolio_value: float,
    ) -> dict[str, int]:
        """与现有持仓对比，产出带顺序的目标 dict：退出 → 减仓 → 加仓。"""
        held = {
            symbol: _int(position.quantity)
            for symbol, position in portfolio.positions.items()
            if position.quantity
        }
        exits: dict[str, int] = {}
        reduces: dict[str, int] = {}
        increases: dict[str, int] = {}
        for symbol in sorted(set(held) | set(desired)):
            current = held.get(symbol, 0)
            target = desired.get(symbol, 0)
            if target == current:
                continue
            bar = bars.get(symbol)
            price = _num(getattr(bar, "close", 0.0)) if bar else 0.0
            if target != 0 and self._is_small_trade(
                target, current, price, portfolio_value
            ):
                continue
            if target == 0:
                exits[symbol] = 0
            elif target < current:
                reduces[symbol] = target
            else:
                increases[symbol] = target
        return {**exits, **reduces, **increases}

    def _is_small_trade(
        self, target: int, current: int, price: float, portfolio_value: float
    ) -> bool:
        """变动市值小于阈值则不动（退出除外，由调用方保证）。"""
        if price <= 0:
            return True
        return abs(target - current) * price < self.min_trade_ratio * portfolio_value
