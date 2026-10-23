"""绩效分析器 — 基于 Decimal 的回测结果分析。

从 mvp/backtest/analysis/analyzer.py 移植，适配 Decimal 金额体系。
所有货币值使用 Decimal（元），移除 /10000 和 *10000 转换。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import cast

import numpy as np
import pandas as pd

from app.data.schema import TradeRecord
from app.engine.backtest.analysis._numeric import to_float
from app.engine.backtest.analysis.slippage_check import scan_slippage_anomalies
from app.engine.backtest.models import EquityPoint

TRADING_DAYS_PER_YEAR = 242
DEFAULT_RISK_FREE_RATE = 0.02


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值返回 default，NaN/inf 原样传递）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/非数值返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ─── 向后兼容转发 ─────────────────────────────────────

# 打印方法已迁移至 analysis/_printer.py，保留此转发以兼容旧调用。
# 新代码请直接使用 from app.engine.backtest.analysis._printer import AnalyzerPrinter

from ._printer import _ljust, _rjust, _visual_width  # noqa: E402, F401

# ─── 数据模型 ─────────────────────────────────────────


@dataclass
class TradeSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_loss_ratio: float
    total_profit: Decimal
    total_loss: Decimal
    avg_holding_days: float


@dataclass
class FIFOMatch:
    """FIFO 匹配对 — 兼容别名，委托 _fifo.MatchedPair。"""

    realized_pnl: Decimal
    holding_days: int


@dataclass
class FIFOReport:
    """基于 FIFO 的盈亏归因报告。"""

    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_trades: int
    matched_pairs: list[FIFOMatch]
    avg_holding_days: float
    win_rate: float


@dataclass
class FrictionReport:
    """交易摩擦成本审计报告。"""

    total_commission: Decimal
    total_stamp_duty: Decimal
    total_slippage_cost: Decimal
    total_friction: Decimal
    friction_ratio: float
    avg_friction_per_trade: Decimal
    # 自检：滑点占比异常的成交（幽灵滑点报警）
    slippage_anomaly_count: int = 0
    slippage_anomaly_cost: Decimal = Decimal("0")
    max_slippage_ratio: float = 0.0


@dataclass
class SortinoReport:
    """增强鲁棒性指标，包括 Sortino。"""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_date: datetime | None
    max_drawdown_duration: int
    calmar_ratio: float
    tail_ratio: float
    skewness: float
    kurtosis: float


@dataclass
class BenchmarkReport:
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    active_return: float
    active_return_mean: float
    active_return_std: float
    benchmark_total_return: float
    benchmark_annualized_return: float


@dataclass
class DrawdownPeriod:
    """回撤周期：从高点（start）到低点（end），然后恢复。"""

    start: datetime | None  # 回撤前高点的日期
    end: datetime | None  # 低点的日期
    duration_days: int  # 从高点到低点的日历天数
    max_drawdown: float  # 最大回撤比率（负值，如 -0.1668）


@dataclass
class PerformanceReport:
    initial_value: Decimal
    final_value: Decimal
    total_return: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_start: datetime | None  # 最大回撤高点日期
    max_drawdown_date: datetime | None  # 最大回撤低点日期
    longest_drawdown: DrawdownPeriod | None  # 最长连续回撤周期
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    drawdown_time_ratio: float  # 水下时间占比：收益处于水下（未创新高）的时间比例
    position_time_ratio: float  # 持仓时间占比：有仓位的交易日占总交易日的比例
    trading_days: int
    risk_free_rate: float
    trade_summary: TradeSummary | None
    fifo_report: FIFOReport | None = None
    friction_report: FrictionReport | None = None
    sortino_report: SortinoReport | None = None
    benchmark_report: BenchmarkReport | None = None

    def to_dict(self) -> dict:
        result = {
            "initial_value": to_float(self.initial_value),
            "final_value": to_float(self.final_value),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_start": self.max_drawdown_start,
            "max_drawdown_date": self.max_drawdown_date,
            "longest_drawdown_days": (
                self.longest_drawdown.duration_days if self.longest_drawdown else None
            ),
            "longest_drawdown_max": (
                self.longest_drawdown.max_drawdown if self.longest_drawdown else None
            ),
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "drawdown_time_ratio": self.drawdown_time_ratio,
            "position_time_ratio": self.position_time_ratio,
            "trading_days": self.trading_days,
            "risk_free_rate": self.risk_free_rate,
        }
        if self.trade_summary:
            result["trade_summary"] = {
                "total_trades": self.trade_summary.total_trades,
                "winning_trades": self.trade_summary.winning_trades,
                "losing_trades": self.trade_summary.losing_trades,
                "win_rate": self.trade_summary.win_rate,
                "avg_profit": self.trade_summary.avg_profit,
                "avg_loss": self.trade_summary.avg_loss,
                "profit_loss_ratio": self.trade_summary.profit_loss_ratio,
                "total_profit": to_float(self.trade_summary.total_profit),
                "total_loss": to_float(self.trade_summary.total_loss),
                "avg_holding_days": self.trade_summary.avg_holding_days,
            }
        if self.fifo_report:
            result["fifo_report"] = {
                "total_realized_pnl": to_float(self.fifo_report.total_realized_pnl),
                "total_unrealized_pnl": to_float(self.fifo_report.total_unrealized_pnl),
                "total_trades": self.fifo_report.total_trades,
                "avg_holding_days": self.fifo_report.avg_holding_days,
                "win_rate": self.fifo_report.win_rate,
            }
        if self.friction_report:
            result["friction_report"] = {
                "total_commission": to_float(self.friction_report.total_commission),
                "total_stamp_duty": to_float(self.friction_report.total_stamp_duty),
                "total_slippage_cost": to_float(
                    self.friction_report.total_slippage_cost
                ),
                "total_friction": to_float(self.friction_report.total_friction),
                "friction_ratio": self.friction_report.friction_ratio,
                "avg_friction_per_trade": to_float(
                    self.friction_report.avg_friction_per_trade
                ),
                "slippage_anomaly_count": self.friction_report.slippage_anomaly_count,
                "slippage_anomaly_cost": to_float(
                    self.friction_report.slippage_anomaly_cost
                ),
                "max_slippage_ratio": self.friction_report.max_slippage_ratio,
            }
        if self.sortino_report:
            result["sortino_report"] = {
                "sortino_ratio": self.sortino_report.sortino_ratio,
                "max_drawdown_duration": self.sortino_report.max_drawdown_duration,
                "tail_ratio": self.sortino_report.tail_ratio,
                "skewness": self.sortino_report.skewness,
                "kurtosis": self.sortino_report.kurtosis,
            }
        return result


class Analyzer:
    @staticmethod
    def calculate(
        equity_curve: list[EquityPoint],
        rf: float = DEFAULT_RISK_FREE_RATE,
        trade_log: list[TradeRecord] | None = None,
        enable_fifo: bool = True,
        enable_friction: bool = True,
        benchmark_equity: list[EquityPoint] | list[float] | pd.Series | None = None,
    ) -> PerformanceReport:
        if not equity_curve:
            raise ValueError("Equity curve cannot be empty")

        df = pd.DataFrame(equity_curve)
        # EquityPoint 字段已是 Decimal，转为 float 用于计算
        df["total_value"] = df["total_value"].apply(to_float).astype(float)
        df["positions_value"] = df["positions_value"].apply(to_float).astype(float)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        initial_value = df["total_value"].iloc[0]
        final_value = df["total_value"].iloc[-1]
        trading_days = len(df)

        total_return = final_value / initial_value - 1
        ann_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1

        returns = pd.Series(df["total_value"].pct_change().fillna(0))
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        rolling_max = cast("pd.Series", df["total_value"].cummax())
        drawdown = (df["total_value"] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        max_dd_date = None if pd.isna(max_dd_idx) else max_dd_idx

        # 找到最大回撤周期的高点日期
        max_dd_start: datetime | None = None
        if max_dd_date is not None:
            peak_value_at_trough = rolling_max.loc[max_dd_date]
            # 向前回溯找到该峰值首次达到的位置
            peak_reached = rolling_max[rolling_max >= peak_value_at_trough]
            if not peak_reached.empty:
                # 获取峰值首次达到的时间（不晚于低点）
                valid_peaks = peak_reached[peak_reached.index <= max_dd_date]
                if not valid_peaks.empty:
                    max_dd_start = valid_peaks.index[0]  # 首次出现 = 高点

        # 找到最长回撤周期及其最大回撤
        longest_dd: DrawdownPeriod | None = None
        longest_duration = 0
        current_dd_start: datetime | None = None
        current_dd_max_dd: float = 0.0

        for ts, value in df["total_value"].items():
            peak = rolling_max.loc[ts]
            if value < peak:
                # 处于回撤中
                dd = (value - peak) / peak
                if current_dd_start is None:
                    # 开始新一轮回撤
                    current_dd_start = ts
                    current_dd_max_dd = dd
                else:
                    # 持续回撤
                    current_dd_max_dd = min(current_dd_max_dd, dd)
                duration = (ts - current_dd_start).days if current_dd_start else 0
                if duration > longest_duration:
                    longest_duration = duration
                    # 找到该最长周期的高点日期
                    period_peak_date = (
                        rolling_max[
                            (rolling_max.index >= current_dd_start)
                            & (rolling_max.index <= ts)
                        ].idxmax()
                        if current_dd_start
                        else None
                    )
                    longest_dd = DrawdownPeriod(
                        start=period_peak_date,
                        end=ts,
                        duration_days=duration,
                        max_drawdown=current_dd_max_dd,
                    )
            else:
                # 恢复 - 重置
                current_dd_start = None
                current_dd_max_dd = 0.0

        sharpe = (ann_return - rf) / ann_vol if ann_vol > 1e-10 else 0.0
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
        drawdown_time_ratio = (
            (drawdown < 0).sum() / trading_days if trading_days > 0 else 0.0
        )
        position_time_ratio = (
            (df["positions_value"] > 0).sum() / trading_days
            if trading_days > 0
            else 0.0
        )

        trade_summary = None
        fifo_report = None
        friction_report = None
        sortino_report = None

        if trade_log:
            trade_summary = Analyzer._analyze_trades(trade_log)

            if enable_fifo:
                fifo_report = Analyzer.fifo_attribution(trade_log)

            if enable_friction:
                friction_report = Analyzer.friction_audit(trade_log)

        if trading_days > 1:
            sortino_report = Analyzer._calculate_sortino(
                returns, rf, ann_return, max_dd, max_dd_date
            )

        benchmark_report: BenchmarkReport | None = None
        if benchmark_equity is not None:
            if isinstance(benchmark_equity, pd.Series):
                bm_series = benchmark_equity
            else:
                bm_df = pd.DataFrame(benchmark_equity)
                bm_series = pd.Series(
                    bm_df["total_value"].apply(to_float).values
                    if "total_value" in bm_df.columns
                    else bm_df.iloc[:, 0].apply(to_float).values,
                    index=pd.to_datetime(bm_df.index)
                    if hasattr(bm_df.index, "to_pydatetime")
                    else pd.to_datetime(bm_df.index),
                )

            combined = pd.concat(
                [df["total_value"], bm_series],
                axis=1,
            )
            combined.columns = ["strategy", "benchmark"]
            combined = combined.ffill().dropna()
            strategy_ret = combined.iloc[:, 0].pct_change().fillna(0)
            benchmark_ret = combined.iloc[:, 1].pct_change().fillna(0)

            n = len(combined)
            bm_total = combined.iloc[-1, 1] / combined.iloc[0, 1] - 1
            bm_ann = (1 + bm_total) ** (TRADING_DAYS_PER_YEAR / n) - 1

            cov = strategy_ret.cov(benchmark_ret)
            var = benchmark_ret.var()
            beta = cov / var if var > 1e-10 else 1.0

            alpha = ann_return - rf - beta * (bm_ann - rf)

            active_ret = strategy_ret - benchmark_ret
            tracking_error = active_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            information_ratio = (
                alpha / tracking_error if tracking_error > 1e-10 else 0.0
            )

            benchmark_report = BenchmarkReport(
                alpha=alpha,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                active_return=_num(combined.iloc[-1, 0] / combined.iloc[0, 0] - 1)
                - bm_total,
                active_return_mean=_num(active_ret.mean()),
                active_return_std=_num(active_ret.std()),
                benchmark_total_return=bm_total,
                benchmark_annualized_return=bm_ann,
            )

        return PerformanceReport(
            initial_value=Decimal(str(initial_value)),
            final_value=Decimal(str(final_value)),
            total_return=total_return,
            annualized_return=ann_return,
            max_drawdown=max_dd,
            max_drawdown_start=max_dd_start,
            max_drawdown_date=max_dd_date,
            longest_drawdown=longest_dd,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            drawdown_time_ratio=drawdown_time_ratio,
            position_time_ratio=position_time_ratio,
            trading_days=trading_days,
            risk_free_rate=rf,
            trade_summary=trade_summary,
            fifo_report=fifo_report,
            friction_report=friction_report,
            sortino_report=sortino_report,
            benchmark_report=benchmark_report,
        )

    @staticmethod
    def _fifo_match_trades(
        trade_log: list[TradeRecord],
    ) -> list[FIFOMatch]:
        """FIFO 买卖匹配 — 委托统一 _fifo 引擎。"""
        from ._fifo import fifo_match

        return [
            FIFOMatch(realized_pnl=p.realized_pnl, holding_days=p.holding_days)
            for p in fifo_match(trade_log, with_fees=True)
        ]

    @staticmethod
    def fifo_attribution(trade_log: list[TradeRecord]) -> FIFOReport:
        matched_pairs = Analyzer._fifo_match_trades(trade_log)

        total_realized = sum((m.realized_pnl for m in matched_pairs), Decimal("0"))
        winning = [m for m in matched_pairs if m.realized_pnl > 0]

        avg_holding = (
            sum(m.holding_days for m in matched_pairs) / len(matched_pairs)
            if matched_pairs
            else 0.0
        )
        win_rate = len(winning) / len(matched_pairs) if matched_pairs else 0.0

        return FIFOReport(
            total_realized_pnl=total_realized,
            total_unrealized_pnl=Decimal("0"),  # 未实现：需当前市场价格
            total_trades=len(matched_pairs),
            matched_pairs=matched_pairs,
            avg_holding_days=avg_holding,
            win_rate=win_rate,
        )

    @staticmethod
    def friction_audit(trade_log: list[TradeRecord]) -> FrictionReport:
        """审计交易摩擦成本。

        直接使用 TradeRecord 的 cost 明细（commission, transfer_fee, stamp_duty），
        确保与回测执行期间通过 AShareCommissionModel 计算的费用一致。
        """
        total_commission = Decimal("0")
        total_stamp_duty = Decimal("0")
        total_slippage = Decimal("0")

        for trade in trade_log:
            # trade.cost 包含佣金 + 过户费 + 印花税
            total_commission += trade.cost.commission + trade.cost.transfer_fee
            total_stamp_duty += trade.cost.stamp_duty
            total_slippage += trade.slippage_cost

        total_friction = total_commission + total_stamp_duty + total_slippage
        total_amount = sum((t.amount for t in trade_log), Decimal("0"))
        friction_ratio = (
            to_float(total_friction / total_amount) if total_amount > 0 else 0.0
        )
        avg_friction = total_friction / len(trade_log) if trade_log else Decimal("0")
        anomaly = scan_slippage_anomalies(trade_log)

        return FrictionReport(
            total_commission=total_commission,
            total_stamp_duty=total_stamp_duty,
            total_slippage_cost=total_slippage,
            total_friction=total_friction,
            friction_ratio=friction_ratio,
            avg_friction_per_trade=avg_friction,
            slippage_anomaly_count=anomaly.count,
            slippage_anomaly_cost=anomaly.cost,
            max_slippage_ratio=anomaly.max_ratio,
        )

    @staticmethod
    def _calculate_sortino(
        returns: pd.Series,
        rf: float,
        ann_return: float,
        max_dd: float,
        max_dd_date: datetime | None,
    ) -> SortinoReport:
        daily_rf = rf / TRADING_DAYS_PER_YEAR
        excess_returns = returns - daily_rf

        ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        sharpe = (ann_return - rf) / ann_vol if ann_vol > 1e-10 else 0.0
        sortino = (ann_return - rf) / downside_std if downside_std > 1e-10 else 0.0
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        tail_ratio = (
            abs(returns[returns > 0].mean() / returns[returns < 0].mean())
            if len(returns[returns < 0]) > 0
            else 0.0
        )

        dd_duration = 0
        in_drawdown = False
        max_duration = 0
        rolling_max = (1 + returns).cumprod().cummax()
        drawdown_series = (1 + returns).cumprod() / rolling_max - 1
        for val in drawdown_series:
            if val < 0:
                if in_drawdown:
                    dd_duration += 1
                else:
                    in_drawdown = True
                    dd_duration = 1
                max_duration = max(max_duration, dd_duration)
            else:
                in_drawdown = False

        try:
            returns_series = pd.Series(np.nan_to_num(returns, nan=0.0))
            skewness = float(returns_series.skew())  # type: ignore[arg-type] — np.nan_to_num 返回值类型与 pandas.Series.skew 不兼容
        except (TypeError, ValueError):
            skewness = 0.0
        try:
            returns_series = pd.Series(np.nan_to_num(returns, nan=0.0))
            kurtosis = float(returns_series.kurtosis())  # type: ignore[arg-type] — np.nan_to_num 返回值类型与 pandas.Series.kurtosis 不兼容
        except (TypeError, ValueError):
            kurtosis = 0.0

        return SortinoReport(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            max_drawdown_duration=max_duration,
            calmar_ratio=calmar,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    @staticmethod
    def _analyze_trades(trade_log: list[TradeRecord]) -> TradeSummary:
        """使用 FIFO 匹配进行回合交易分析。"""
        if not trade_log:
            return TradeSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                profit_loss_ratio=0.0,
                total_profit=Decimal("0"),
                total_loss=Decimal("0"),
                avg_holding_days=0.0,
            )

        # 使用通用 FIFO 匹配引擎 - 单次遍历
        matched_pairs = Analyzer._fifo_match_trades(trade_log)

        if not matched_pairs:
            return TradeSummary(
                total_trades=len(trade_log),
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                profit_loss_ratio=0.0,
                total_profit=Decimal("0"),
                total_loss=Decimal("0"),
                avg_holding_days=0.0,
            )

        # 从匹配对中提取盈利和持仓天数
        profits = [m.realized_pnl for m in matched_pairs]
        holding_days = [m.holding_days for m in matched_pairs]

        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]

        total_profit = sum(winning, Decimal("0")) if winning else Decimal("0")
        total_loss = abs(sum(losing, Decimal("0"))) if losing else Decimal("0")

        avg_profit = to_float(total_profit / len(winning)) if winning else 0.0
        avg_loss = to_float(total_loss / len(losing)) if losing else 0.0

        pl_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0

        return TradeSummary(
            total_trades=len(trade_log),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(profits) if profits else 0.0,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_loss_ratio=pl_ratio,
            total_profit=total_profit,
            total_loss=total_loss,
            avg_holding_days=sum(holding_days) / len(holding_days)
            if holding_days
            else 0.0,
        )

    # ── 终端打印（已迁移至 _printer.py）───
    # 保留此静态方法作为兼容转发，
    # 新代码请使用 from ._printer import AnalyzerPrinter

    @staticmethod
    def print_report(report: PerformanceReport) -> None:
        from ._printer import AnalyzerPrinter

        AnalyzerPrinter.print_report(report)


@dataclass
class SweepResult:
    parameter_name: str
    parameter_values: list[float]
    returns: list[float]
    sharpe_ratios: list[float]
    max_drawdowns: list[float]
    friction_ratios: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "parameter": self.parameter_values,
                "return": self.returns,
                "sharpe": self.sharpe_ratios,
                "max_dd": self.max_drawdowns,
                "friction_ratio": self.friction_ratios,
            }
        )


@dataclass
class MonteCarloResult:
    num_simulations: int
    initial_value: Decimal
    final_values: list[Decimal]
    max_drawdowns: list[float]
    returns: list[float]
    var_95: float
    cvar_95: float
    probability_of_ruin: float

    @property
    def avg_final_value(self) -> Decimal:
        return sum(self.final_values) / len(self.final_values)

    @property
    def worst_case_dd(self) -> float:
        return min(self.max_drawdowns) if self.max_drawdowns else 0.0

    @property
    def percentile_95_dd(self) -> float:
        if not self.max_drawdowns:
            return 0.0
        return _num(sorted(self.max_drawdowns)[_int(len(self.max_drawdowns) * 0.95)])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "simulation": range(len(self.final_values)),
                "final_value": [to_float(v) for v in self.final_values],
                "return": self.returns,
                "max_drawdown": self.max_drawdowns,
            }
        )


class ParameterSweep:
    @staticmethod
    def sweep_slippage(
        trade_log: list[TradeRecord],
        equity_curve: list[EquityPoint],
        slippage_range: tuple[float, float] = (0.0005, 0.002),
        num_points: int = 10,
    ) -> SweepResult:
        """滑点假设的敏感性分析。

        注意：这是一阶近似。通过从最终权益中减去摩擦成本来
        回算滑点影响，不重新运行完整回测。这忽略了：
        - 复合效应（滑点影响进出场价格，而不仅是最终盈亏）
        - 对仓位大小和信号生成的影响
        - 对投资组合再平衡的非线性影响

        对于准确的敏感性分析，应使用每个参数组合重新运行完整回测引擎。
        """
        base_friction = sum(
            (t.cost.total + t.slippage_cost for t in trade_log), Decimal("0")
        )
        base_amount = sum((t.amount for t in trade_log), Decimal("0"))
        initial_value = (
            equity_curve[0].total_value if equity_curve else Decimal("1000000")
        )
        final_value = (
            equity_curve[-1].total_value if equity_curve else Decimal("1000000")
        )

        # 从权益曲线计算实际波动率
        if len(equity_curve) > 1:
            values = pd.Series([to_float(p.total_value) for p in equity_curve])
            rets = values.pct_change().fillna(0)
            ann_vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            ann_vol = 0.15  # 回退到典型市场波动率

        # 从权益曲线计算最大回撤
        values_list = [to_float(p.total_value) for p in equity_curve]
        rolling_max = pd.Series(values_list).cummax()
        drawdown = (pd.Series(values_list) - rolling_max) / rolling_max
        max_dd = drawdown.min() if not drawdown.empty else 0.0

        slippage_values = [
            slippage_range[0]
            + i * (slippage_range[1] - slippage_range[0]) / (num_points - 1)
            for i in range(num_points)
        ]

        returns = []
        sharpe_ratios = []
        max_dds = []
        friction_ratios = []

        for slippage in slippage_values:
            slippage_cost = to_float(base_amount) * slippage
            total_friction = to_float(base_friction) + slippage_cost
            friction_ratio = (
                total_friction / to_float(base_amount) if base_amount > 0 else 0.0
            )

            adjusted_final = to_float(final_value) - total_friction
            ret = (adjusted_final - to_float(initial_value)) / to_float(initial_value)

            ann_return = (
                (1 + ret) ** (TRADING_DAYS_PER_YEAR / len(equity_curve)) - 1
                if len(equity_curve) > 1
                else 0
            )
            sharpe = (
                (ann_return - DEFAULT_RISK_FREE_RATE) / ann_vol
                if ann_vol > 1e-10
                else 0
            )

            returns.append(ret)
            sharpe_ratios.append(sharpe)
            max_dds.append(max_dd)
            friction_ratios.append(friction_ratio)

        return SweepResult(
            parameter_name="slippage_bps",
            parameter_values=[s * 10000 for s in slippage_values],
            returns=returns,
            sharpe_ratios=sharpe_ratios,
            max_drawdowns=max_dds,
            friction_ratios=friction_ratios,
        )

    @staticmethod
    def print_sweep_report(result: SweepResult) -> None:
        from ._printer import SweepPrinter

        SweepPrinter.print_sweep_report(result)


class MonteCarloSimulator:
    @staticmethod
    def simulate_trade_sequence(
        trade_log: list[TradeRecord],
        equity_curve: list[EquityPoint],
        num_simulations: int = 10000,
        random_seed: int | None = None,
    ) -> MonteCarloResult:
        if random_seed is not None:
            random.seed(random_seed)

        initial_value = (
            equity_curve[0].total_value if equity_curve else Decimal("1000000")
        )

        returns_list = []
        for i in range(1, len(equity_curve)):
            prev = to_float(equity_curve[i - 1].total_value)
            curr = to_float(equity_curve[i].total_value)
            if prev > 0:
                returns_list.append((curr - prev) / prev)

        if not returns_list:
            returns_list = [0.0]

        final_values: list[Decimal] = []
        max_drawdowns = []
        sim_returns = []

        for _ in range(num_simulations):
            shuffled = returns_list.copy()
            random.shuffle(shuffled)

            equity = to_float(initial_value)
            peak = equity
            max_dd = 0.0
            for ret in shuffled:
                equity *= 1 + ret
                peak = max(peak, equity)
                dd = (equity - peak) / peak
                max_dd = min(max_dd, dd)

            final_values.append(Decimal(str(equity)))
            max_drawdowns.append(max_dd)
            sim_returns.append(
                (equity - to_float(initial_value)) / to_float(initial_value)
            )

        sorted_returns = sorted(sim_returns)
        var_95 = sorted_returns[_int(len(sorted_returns) * 0.05)]
        tail_count = max(1, _int(len(sorted_returns) * 0.05))
        cvar_95 = sum(sorted_returns[:tail_count]) / tail_count

        ruin_count = sum(1 for v in final_values if v < initial_value * Decimal("0.5"))
        prob_ruin = ruin_count / num_simulations

        return MonteCarloResult(
            num_simulations=num_simulations,
            initial_value=initial_value,
            final_values=final_values,
            max_drawdowns=max_drawdowns,
            returns=sim_returns,
            var_95=var_95,
            cvar_95=cvar_95,
            probability_of_ruin=prob_ruin,
        )

    @staticmethod
    def print_monte_carlo_report(result: MonteCarloResult) -> None:
        from ._printer import MonteCarloPrinter

        MonteCarloPrinter.print_monte_carlo_report(result)
