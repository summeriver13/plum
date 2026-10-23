"""信号管线 — 可组合的过滤器链。

在 strategy.on_bar() 产生原始信号之后、仓位分配之前，
按顺序执行一组 SignalFilter，对信号做增强/过滤/缩放。

内置实现：
    ExposureFilter   — 全局仓位暴露度控制（timing override）
    ScaleInFilter    — 分步建仓
    RiskFilter       — 风控约束（集中度/ST/流动性）

用法:
    from app.engine.backtest.pipeline import SignalPipeline, ExposureFilter

    pipeline = SignalPipeline(filters=[
        ExposureFilter(exposure_fn=lambda ctx: 0.5 if is_bear(ctx) else 1.0),
        ScaleInFilter(targets=[0.31, 0.68, 1.0]),
    ])

    engine = Engine(pipeline=pipeline, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Protocol

from app.engine.backtest.models import Signal, SignalDirection

logger = logging.getLogger(__name__)


# ── 上下文 ──────────────────────────────────────────────


@dataclass
class FilterContext:
    """过滤器上下文 — 信号管线可访问的所有运行时信息。"""

    ts: datetime
    bars: dict[str, Any]
    portfolio: Any  # Portfolio


# ── 协议 ────────────────────────────────────────────────


class SignalFilter(Protocol):
    """信号过滤器协议。

    每个过滤器接收信号列表和上下文，返回处理后的信号列表。
    可以：过滤掉某些信号、修改 strength、添加新信号。
    """

    def apply(
        self,
        signals: list[Signal],
        context: FilterContext,
    ) -> list[Signal]: ...


# ── 管线 ────────────────────────────────────────────────


@dataclass
class SignalPipeline:
    """可组合的信号过滤器链。

    按 filters 顺序依次执行，每个过滤器的输出是下一个的输入。
    若 filters 为空，信号原样透传。
    """

    filters: list[SignalFilter] = field(default_factory=list)

    def apply(
        self,
        signals: list[Signal],
        context: FilterContext,
    ) -> list[Signal]:
        if not signals:
            return signals
        for f in self.filters:
            signals = f.apply(signals, context)
        return signals


# ── 内置过滤器 ──────────────────────────────────────────


@dataclass
class ExposureFilter:
    """全局仓位暴露度控制。

    根据 exposure_fn 返回值缩放所有 LONG 信号的 strength。
    exposure=1.0 → 满仓，exposure=0.5 → 半仓，exposure=0.0 → 全部 FLAT。

    典型用法：接入大盘择时 signal_override。
    """

    exposure_fn: Callable[[FilterContext], float]
    """返回 [0.0, 1.0] 的暴露度。1.0=满仓，0.0=全部清仓。"""

    def apply(
        self,
        signals: list[Signal],
        context: FilterContext,
    ) -> list[Signal]:
        try:
            exposure = float(self.exposure_fn(context))
        except Exception:
            logger.debug("ExposureFilter: exposure_fn 异常，按满仓处理", exc_info=True)
            return signals

        exposure = max(0.0, min(1.0, exposure))

        if exposure >= 1.0:
            return signals

        if exposure <= 0.0:
            # 全部转为 FLAT
            return [
                replace(
                    sig,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    reason=f"{sig.reason} [exposure=0]",
                )
                for sig in signals
            ]

        # 等比缩放 LONG 信号的 strength
        return [
            replace(
                sig,
                strength=sig.strength * exposure,
                reason=f"{sig.reason} [exp={exposure:.2f}]",
            )
            if sig.direction == SignalDirection.LONG
            else sig
            for sig in signals
        ]


@dataclass
class ScaleInFilter:
    """分步建仓过滤器。

    同一只股票首次入选时仅以 target[0] 比例建仓，
    后续每期依次推进到下一个 target，直至 target[-1]（满仓）。

    Args:
        targets: 建仓进度，如 [0.31, 0.68, 1.0]
        track_by: 按什么追踪建仓进度，"symbol"（股票维度）或 "source"（策略维度）
    """

    targets: list[float] = field(default_factory=lambda: [0.31, 0.68, 1.0])
    track_by: str = "symbol"

    _progress: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    """{symbol_or_source: 当前进度索引}"""

    def __post_init__(self) -> None:
        if not self.targets or self.targets[-1] != 1.0:
            raise ValueError("targets 最后一项必须为 1.0（满仓）")

    def apply(
        self,
        signals: list[Signal],
        context: FilterContext,
    ) -> list[Signal]:
        result: list[Signal] = []

        for sig in signals:
            if sig.direction != SignalDirection.LONG:
                result.append(sig)
                continue

            key = sig.symbol if self.track_by == "symbol" else sig.source
            step = self._progress.get(key, 0)
            scale = self.targets[min(step, len(self.targets) - 1)]

            # 推进到下一步（最多到最后一档）
            self._progress[key] = min(step + 1, len(self.targets) - 1)

            if scale <= 0.0:
                continue  # 跳过，不建仓

            result.append(
                replace(
                    sig,
                    strength=sig.strength * scale,
                    reason=f"{sig.reason} [scale-in {step + 1}/{len(self.targets)}]",
                )
            )

        return result

    def reset(self, key: str | None = None) -> None:
        """重置建仓进度。

        Args:
            key: 重置特定 symbol/source。None 表示全部重置。
        """
        if key is None:
            self._progress.clear()
        else:
            self._progress.pop(key, None)


@dataclass
class RiskFilter:
    """风控约束过滤器。

    - 排除 ST 股票
    - 单票集中度上限（按 signal.strength 过滤占比过高的）
    - 最小成交额过滤
    """

    exclude_st: bool = True
    max_per_stock_pct: float | None = None
    """单票最大仓位占比。超过此比例的信号 strength 被缩放。None 为不限制。"""
    min_amount: float | None = None
    """最小成交额（元），低于此值的股票不买入。"""

    def apply(
        self,
        signals: list[Signal],
        context: FilterContext,
    ) -> list[Signal]:
        result: list[Signal] = []
        total_strength = sum(
            s.strength for s in signals if s.direction == SignalDirection.LONG
        )

        for sig in signals:
            # ST 过滤
            if self.exclude_st:
                bar = context.bars.get(sig.symbol)
                if bar is not None and getattr(bar, "is_st", False):
                    continue

            # 最小成交额过滤
            if self.min_amount is not None:
                bar = context.bars.get(sig.symbol)
                if bar is not None:
                    amount = float(getattr(bar, "amount", 0) or 0)
                    if 0 < amount < self.min_amount:
                        continue

            # 集中度约束
            if (
                self.max_per_stock_pct is not None
                and sig.direction == SignalDirection.LONG
                and total_strength > 0
            ):
                share = sig.strength / total_strength
                if share > self.max_per_stock_pct:
                    capped = sig.strength * (self.max_per_stock_pct / share)
                    sig = replace(
                        sig,
                        strength=capped,
                        reason=f"{sig.reason} [capped@{self.max_per_stock_pct:.0%}]",
                    )

            result.append(sig)

        return result


__all__ = [
    "SignalFilter",
    "FilterContext",
    "SignalPipeline",
    "ExposureFilter",
    "ScaleInFilter",
    "RiskFilter",
]
