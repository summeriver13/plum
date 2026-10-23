"""参数稳定性分析 — Walk-Forward、Sharpe Gradient、Parameter Plateau。

从 mvp/backtest/analysis/stability.py 移植，适配 Decimal 权益值体系。
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd

from app.engine.backtest.analysis._numeric import equity_to_numpy, to_float


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值字符串返回 default，NaN/inf 原样传递）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/非数值字符串返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


@dataclass
class WalkForwardResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics
    sharpe_degradation: float
    return_degradation: float


@dataclass
class GridPoint:
    params: dict[str, Any]
    metrics: PerformanceMetrics
    rank: int = 0


@dataclass
class SharpeGradientResult:
    parameter_name: str
    parameter_values: list[float]
    sharpe_values: list[float]
    gradient_values: list[float]
    unstable_indices: list[int]
    plateau_ratio: float
    best_sharpe: float
    best_param: float


@dataclass
class PlateauRegion:
    param_start: float
    param_end: float
    region_type: str
    min_sharpe: float
    max_sharpe: float
    stability_score: float


@dataclass
class ParameterPlateauResult:
    parameter_name: str
    parameter_values: list[float]
    sharpe_values: list[float]
    regions: list[PlateauRegion]
    stable_params: list[float]
    danger_zones: list[tuple[float, float]]
    best_sharpe: float
    best_param: float


@dataclass
class GridSweepResult:
    grid_size: int
    best_params: dict[str, Any]
    best_metrics: PerformanceMetrics
    all_points: list[GridPoint]
    stability_score: float
    sharpe_gradient: SharpeGradientResult | None = None
    plateau: ParameterPlateauResult | None = None


class StabilityAnalyzer:
    def __init__(
        self,
        analyze_fn: Callable[..., tuple[list, list]],
        rf: float = 0.02,
    ) -> None:
        self._analyze_fn = analyze_fn
        self._rf = rf

    def walk_forward(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        train_days: int = 252,
        test_days: int = 60,
        overlap_days: int = 20,
        strategy_factory: Callable[..., Any] | None = None,
        **strategy_kwargs: Any,
    ) -> list[WalkForwardResult]:
        # 通过 partial 绑定工厂 + 基础参数，使每个 OOS 运行
        # 都能用新鲜参数重建策略。
        if strategy_factory is not None:
            bound_fn = partial(
                self._analyze_fn,
                strategy_factory=partial(strategy_factory, **strategy_kwargs),
            )
        else:
            bound_fn = partial(self._analyze_fn, strategy_factory=None)

        total_days = (end - start).days
        if total_days < train_days + test_days:
            return []

        results: list[WalkForwardResult] = []
        cursor = cast(pd.Timestamp, pd.Timestamp(start))

        while True:
            cursor_ts = cast(pd.Timestamp, pd.Timestamp(cursor))
            train_end = cursor_ts + pd.Timedelta(days=train_days)
            test_end = train_end + pd.Timedelta(days=test_days)

            if test_end > end:
                break

            is_bars, is_equity = bound_fn(symbols, cursor_ts, train_end)

            oos_start = train_end + pd.Timedelta(days=1)
            oos_bars, oos_equity = bound_fn(symbols, oos_start, test_end)

            is_metrics = self._compute_metrics(is_equity, is_bars)
            oos_metrics = self._compute_metrics(oos_equity, oos_bars)

            sharpe_deg = (
                (oos_metrics.sharpe_ratio - is_metrics.sharpe_ratio)
                / abs(is_metrics.sharpe_ratio)
                if is_metrics.sharpe_ratio != 0
                else 0.0
            )
            ret_deg = (
                (oos_metrics.annualized_return - is_metrics.annualized_return)
                / abs(is_metrics.annualized_return)
                if is_metrics.annualized_return != 0
                else 0.0
            )

            _train_end = cast(datetime, train_end.to_pydatetime())
            _test_start = cast(datetime, oos_start.to_pydatetime())
            _test_end = cast(datetime, test_end.to_pydatetime())

            results.append(
                WalkForwardResult(
                    train_start=cast(datetime, cursor_ts.to_pydatetime()),
                    train_end=_train_end,
                    test_start=_test_start,
                    test_end=_test_end,
                    in_sample=is_metrics,
                    out_of_sample=oos_metrics,
                    sharpe_degradation=sharpe_deg,
                    return_degradation=ret_deg,
                )
            )

            cursor = cast(
                datetime,
                pd.Timestamp(cursor) + pd.Timedelta(days=test_days - overlap_days),
            ).to_pydatetime()  # type: ignore[union-attr] — cursor 经过 pd.Timestamp 包装后保证有 to_pydatetime

        return results

    def grid_sweep(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        param_grid: dict[str, list[Any]],
        strategy_factory: Callable[..., Any] | None = None,
        top_k: int = 5,
        grad_threshold: float = 0.5,
        **strategy_kwargs: Any,
    ) -> GridSweepResult:
        if strategy_factory is not None:
            base_factory = partial(strategy_factory, **strategy_kwargs)
        else:
            base_factory = None  # type: ignore[assignment] — None 赋值给 Callable 类型变量，运行时由 partial 覆盖

        param_names = list(param_grid.keys())
        param_combinations = list(
            itertools.product(*[param_grid[k] for k in param_names])
        )

        points: list[GridPoint] = []

        for combo in param_combinations:
            params = dict(zip(param_names, combo, strict=True))
            bound = partial(base_factory, **params) if base_factory else None  # type: ignore[arg-type] — **params 动态匹配策略工厂构造函数签名
            bars, equity = self._analyze_fn(
                symbols,
                start,
                end,
                strategy_factory=bound,
            )
            metrics = self._compute_metrics(equity, bars)
            points.append(GridPoint(params=params, metrics=metrics))

        points.sort(key=lambda p: p.metrics.sharpe_ratio, reverse=True)

        for i, pt in enumerate(points):
            pt.rank = i + 1

        top_points = points[:top_k]
        best_params = top_points[0].params if top_points else {}
        best_metrics = top_points[0].metrics if top_points else points[0].metrics

        avg_sharpe = np.mean([p.metrics.sharpe_ratio for p in points])
        top_avg_sharpe = np.mean([p.metrics.sharpe_ratio for p in top_points])
        stability_score = (
            _num(top_avg_sharpe / abs(avg_sharpe)) if avg_sharpe != 0 else 0.0
        )

        grad_result: SharpeGradientResult | None = None
        plateau_result: ParameterPlateauResult | None = None

        if len(param_names) == 1 and len(points) >= 3:
            pname = param_names[0]
            pvals = [pt.params[pname] for pt in points]
            sharpes = [pt.metrics.sharpe_ratio for pt in points]

            grad_result = self.sharpe_gradient(pname, pvals, sharpes, grad_threshold)
            plateau_result = self.parameter_plateau(pname, pvals, sharpes)

        return GridSweepResult(
            grid_size=len(points),
            best_params=best_params,
            best_metrics=best_metrics,
            all_points=points,
            stability_score=stability_score,
            sharpe_gradient=grad_result,
            plateau=plateau_result,
        )

    def sharpe_gradient(
        self,
        param_name: str,
        param_values: list[float],
        sharpe_values: list[float],
        threshold: float = 0.5,
    ) -> SharpeGradientResult:
        gradients: list[float] = []
        for i in range(len(param_values) - 1):
            d_param = param_values[i + 1] - param_values[i]
            d_sharpe = sharpe_values[i + 1] - sharpe_values[i]
            grad = d_sharpe / d_param if d_param != 0 else 0.0
            gradients.append(grad)

        unstable: list[int] = [i for i, g in enumerate(gradients) if abs(g) > threshold]

        stable_count = len(gradients) - len(unstable)
        plateau_ratio = stable_count / len(gradients) if gradients else 0.0

        best_idx = _int(np.argmax(sharpe_values))
        best_sharpe = sharpe_values[best_idx]
        best_param = param_values[best_idx]

        return SharpeGradientResult(
            parameter_name=param_name,
            parameter_values=param_values,
            sharpe_values=sharpe_values,
            gradient_values=gradients,
            unstable_indices=unstable,
            plateau_ratio=plateau_ratio,
            best_sharpe=best_sharpe,
            best_param=best_param,
        )

    def parameter_plateau(
        self,
        param_name: str,
        param_values: list[float],
        sharpe_values: list[float],
        stability_threshold: float = 0.1,
        sharpe_floor: float = 0.0,
    ) -> ParameterPlateauResult:
        regions: list[PlateauRegion] = []
        danger_zones: list[tuple[float, float]] = []

        if len(param_values) < 3:
            return ParameterPlateauResult(
                parameter_name=param_name,
                parameter_values=param_values,
                sharpe_values=sharpe_values,
                regions=[],
                stable_params=[],
                danger_zones=[],
                best_sharpe=max(sharpe_values) if sharpe_values else 0.0,
                best_param=param_values[0] if param_values else 0.0,
            )

        local_stable_start = 0
        for i in range(1, len(param_values)):
            delta = abs(sharpe_values[i] - sharpe_values[i - 1])
            if delta > stability_threshold:
                region_sharpes = sharpe_values[local_stable_start:i]
                max_s = max(region_sharpes)
                min_s = min(region_sharpes)

                if max(region_sharpes) >= sharpe_floor:
                    region_type = (
                        "STABLE"
                        if (max_s - min_s) < stability_threshold * 2
                        else "MODERATE"
                    )
                else:
                    region_type = "DANGER"
                    danger_zones.append(
                        (param_values[local_stable_start], param_values[i - 1])
                    )

                regions.append(
                    PlateauRegion(
                        param_start=param_values[local_stable_start],
                        param_end=param_values[i - 1],
                        region_type=region_type,
                        min_sharpe=min_s,
                        max_sharpe=max_s,
                        stability_score=_num(1.0 - (max_s - min_s) / max_s)
                        if max_s > 0
                        else 0.0,
                    )
                )
                local_stable_start = i

        last_sharpes = sharpe_values[local_stable_start:]
        if last_sharpes:
            regions.append(
                PlateauRegion(
                    param_start=param_values[local_stable_start],
                    param_end=param_values[-1],
                    region_type="STABLE"
                    if min(last_sharpes) >= sharpe_floor
                    else "DANGER",
                    min_sharpe=min(last_sharpes),
                    max_sharpe=max(last_sharpes),
                    stability_score=0.5,
                )
            )

        stable_params = [
            param_values[i] for i, s in enumerate(sharpe_values) if s >= sharpe_floor
        ]

        best_idx = _int(np.argmax(sharpe_values))

        return ParameterPlateauResult(
            parameter_name=param_name,
            parameter_values=param_values,
            sharpe_values=sharpe_values,
            regions=regions,
            stable_params=stable_params,
            danger_zones=danger_zones,
            best_sharpe=sharpe_values[best_idx],
            best_param=param_values[best_idx],
        )

    def _compute_metrics(
        self,
        equity_curve: list[Any],
        trade_log: list[Any],
    ) -> PerformanceMetrics:
        if not equity_curve or len(equity_curve) < 2:
            return PerformanceMetrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                num_trades=len(trade_log) if trade_log else 0,
            )

        # EquityPoint 的 total_value 是 Decimal，转为 float 用于 numpy 计算
        first = to_float(equity_curve[0].total_value)
        last = to_float(equity_curve[-1].total_value)

        total_return = (last - first) / first
        trading_days = len(equity_curve)
        ann_return = (1 + total_return) ** (242 / trading_days) - 1

        values = equity_to_numpy(equity_curve)
        returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([0.0])
        returns = returns[~np.isnan(returns)]

        ann_vol = _num(np.std(returns)) * np.sqrt(242) if len(returns) > 1 else 0.0
        sharpe = (ann_return - self._rf) / ann_vol if ann_vol > 1e-10 else 0.0

        downside = returns[returns < 0]
        downside_std = (
            _num(np.std(downside)) * np.sqrt(242) if len(downside) > 1 else 0.0
        )
        sortino = (
            (ann_return - self._rf) / downside_std if downside_std > 1e-10 else 0.0
        )

        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        num_trades = len(trade_log) if trade_log else 0
        win_rate = 0.0
        if trade_log:
            buys = [t for t in trade_log if t.is_buy]
            if buys:
                win_rate = 0.5

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades,
        )


class PrintReport:
    """终端打印（已迁移至 _printer.py）。"""

    @staticmethod
    def walk_forward(results) -> None:
        from ._printer import StabilityPrinter

        StabilityPrinter.walk_forward(results)

    @staticmethod
    def grid_sweep(result) -> None:
        from ._printer import StabilityPrinter

        StabilityPrinter.grid_sweep(result)

    @staticmethod
    def sharpe_gradient(result) -> None:
        from ._printer import StabilityPrinter

        StabilityPrinter.sharpe_gradient(result)

    @staticmethod
    def plateau(result) -> None:
        from ._printer import StabilityPrinter

        StabilityPrinter.plateau(result)
