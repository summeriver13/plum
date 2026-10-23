"""扩展分析器 — 基于 Decimal 的逐标的、市场状态归因分析。

从 mvp/backtest/analysis/analyzer_ext.py 移植，适配 Decimal 金额体系。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from app.data.schema import TradeRecord
from app.engine.backtest.models import BacktestResult, EquityPoint

from ._fifo import fifo_match
from .analyzer import Analyzer, PerformanceReport


@dataclass(frozen=True, slots=True)
class RegimeAwareReport:
    bull_stats: dict[str, float]
    bear_stats: dict[str, float]
    sideways_stats: dict[str, float]
    regime_switches: list[tuple[str, str, str]]


@dataclass(frozen=True, slots=True)
class PerSymbolReport:
    symbol: str
    total_trades: int
    win_rate: float
    avg_holding_days: float
    realized_pnl: Decimal
    total_fees: Decimal
    avg_entry_price: Decimal
    avg_exit_price: Decimal


@dataclass(frozen=True, slots=True)
class PositionAttributionReport:
    entry_price_distribution: list[tuple[str, int]]
    holding_period_distribution: list[tuple[str, int]]
    avg_entry_to_exit: float


@dataclass(frozen=True, slots=True)
class DrawdownRecoveryReport:
    max_drawdown: float
    max_drawdown_date: datetime | None
    drawdown_duration_days: int
    recovery_time_days: int | None
    avg_recovery_days: float


@dataclass(frozen=True, slots=True)
class ExtendedReport:
    base_performance: PerformanceReport
    regime_report: RegimeAwareReport
    per_symbol_reports: list[PerSymbolReport]
    position_attribution: PositionAttributionReport
    drawdown_recovery: DrawdownRecoveryReport

    def to_dict(self) -> dict:
        return {
            "base_performance": self.base_performance.to_dict(),
            "regime_report": {
                "bull_stats": self.regime_report.bull_stats,
                "bear_stats": self.regime_report.bear_stats,
                "sideways_stats": self.regime_report.sideways_stats,
                "regime_switches": [
                    list(t) for t in self.regime_report.regime_switches
                ],
            },
            "per_symbol_reports": [
                {
                    "symbol": r.symbol,
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "avg_holding_days": r.avg_holding_days,
                    "realized_pnl": float(r.realized_pnl),
                    "total_fees": float(r.total_fees),
                    "avg_entry_price": float(r.avg_entry_price),
                    "avg_exit_price": float(r.avg_exit_price),
                }
                for r in self.per_symbol_reports
            ],
            "position_attribution": {
                "entry_price_distribution": [
                    list(t) for t in self.position_attribution.entry_price_distribution
                ],
                "holding_period_distribution": [
                    list(t)
                    for t in self.position_attribution.holding_period_distribution
                ],
                "avg_entry_to_exit": self.position_attribution.avg_entry_to_exit,
            },
            "drawdown_recovery": {
                "max_drawdown": self.drawdown_recovery.max_drawdown,
                "max_drawdown_date": (
                    self.drawdown_recovery.max_drawdown_date.isoformat()
                    if self.drawdown_recovery.max_drawdown_date
                    else None
                ),
                "drawdown_duration_days": self.drawdown_recovery.drawdown_duration_days,
                "recovery_time_days": self.drawdown_recovery.recovery_time_days,
                "avg_recovery_days": self.drawdown_recovery.avg_recovery_days,
            },
        }


class AnalyzerExt:
    @staticmethod
    def _bucket_entry_profit(pct: float) -> str:
        if abs(pct) < 0.05:
            return "0-5%"
        if abs(pct) < 0.10:
            return "5-10%"
        if abs(pct) < 0.20:
            return "10-20%"
        return "20%+"

    @staticmethod
    def _bucket_holding(days: int) -> str:
        if days <= 5:
            return "1-5d"
        if days <= 10:
            return "5-10d"
        if days <= 20:
            return "10-20d"
        return "20+d"

    @staticmethod
    def _infer_regimes_from_equity(
        equity_points: list[EquityPoint],
    ) -> dict[str, str]:
        if not equity_points:
            return {}
        records: list[dict] = [
            {"timestamp": p.timestamp, "total_value": float(p.total_value)}
            for p in equity_points
        ]
        df = pd.DataFrame(records)
        df["value"] = df["total_value"].astype(float)
        df["date"] = df["timestamp"].dt.date
        df.sort_values("timestamp", inplace=True)
        ma_window = 20
        df["ma"] = df["value"].rolling(window=ma_window, min_periods=1).mean()
        df["slope"] = df["value"].diff()
        regimes: dict[str, str] = {}
        date_series = df["date"]
        value_series = df["value"]
        ma_series = df["ma"]
        slope_series = df["slope"]
        for i in range(len(df)):
            date_obj = date_series.iloc[i]
            ma_val = ma_series.iloc[i]
            value_val = float(value_series.iloc[i])
            slope_raw = slope_series.iloc[i]
            slope_val = 0.0
            if isinstance(slope_raw, float):
                if not pd.isna(slope_raw):
                    slope_val = slope_raw
            elif isinstance(slope_raw, (int, float)):
                slope_val = float(slope_raw)
            date_str = str(date_obj)
            if isinstance(ma_val, float) and pd.isna(ma_val):
                regimes[date_str] = "SIDEWAYS"
                continue
            ma_float = float(ma_val) if isinstance(ma_val, float) else 0.0
            above_ma = value_val > ma_float
            slope_pos = slope_val > 0
            if above_ma and slope_pos:
                regimes[date_str] = "BULL"
            elif not above_ma and not slope_pos:
                regimes[date_str] = "BEAR"
            else:
                regimes[date_str] = "SIDEWAYS"
        return regimes

    @staticmethod
    def calculate(
        result: BacktestResult, regime_labels: dict[str, str] | None = None
    ) -> ExtendedReport:
        equity_list: list[EquityPoint] = result.equity_curve
        base = Analyzer.calculate(equity_list, trade_log=result.trade_log)
        regime_by_date: dict[str, str] = (
            regime_labels
            if regime_labels is not None
            else AnalyzerExt._infer_regimes_from_equity(equity_list)
        )

        if equity_list:
            dates = [p.timestamp.date().isoformat() for p in equity_list]
            regimes_ordered = [regime_by_date.get(d, "SIDEWAYS") for d in dates]
            segments: list[tuple[str, str, int]] = []
            if regimes_ordered:
                cur = regimes_ordered[0]
                start = dates[0]
                for i in range(1, len(regimes_ordered)):
                    if regimes_ordered[i] != cur:
                        segments.append((cur, start, i - 1))
                        cur = regimes_ordered[i]
                        start = dates[i]
                segments.append((cur, start, len(regimes_ordered) - 1))
        else:
            segments = []

        regime_switches: list[tuple[str, str, str]] = []
        if segments and len(segments) > 1:
            prev = segments[0]
            for seg in segments[1:]:
                regime_switches.append((seg[1], prev[0], seg[0]))
                prev = seg

        all_symbols: dict[str, list[TradeRecord]] = {}
        for t in result.trade_log:
            all_symbols.setdefault(t.symbol, []).append(t)

        symbol_reports: list[PerSymbolReport] = []
        for symbol, trades in all_symbols.items():
            pairs = fifo_match(trades, with_fees=True)
            if pairs:
                total_trades = len(pairs)
                realized_total = sum((p.realized_pnl for p in pairs), Decimal("0"))
                total_fees = sum((p.buy_fee + p.sell_fee for p in pairs), Decimal("0"))
                total_qty = sum(p.qty for p in pairs)
                avg_entry_price = sum(
                    (p.entry_price * Decimal(p.qty) for p in pairs), Decimal("0")
                ) / Decimal(max(total_qty, 1))
                avg_exit_price = sum(
                    (p.exit_price * Decimal(p.qty) for p in pairs), Decimal("0")
                ) / Decimal(max(total_qty, 1))
                win_trades = sum(1 for p in pairs if p.is_win)
                win_rate = win_trades / max(total_trades, 1)
                avg_holding = sum(p.holding_days for p in pairs) / max(len(pairs), 1)
                symbol_reports.append(
                    PerSymbolReport(
                        symbol=symbol,
                        total_trades=total_trades,
                        win_rate=win_rate,
                        avg_holding_days=avg_holding,
                        realized_pnl=realized_total,
                        total_fees=total_fees,
                        avg_entry_price=avg_entry_price,
                        avg_exit_price=avg_exit_price,
                    )
                )

        entry_bucket_counts: dict[str, int] = {
            "0-5%": 0,
            "5-10%": 0,
            "10-20%": 0,
            "20%+": 0,
        }
        holding_bucket_counts: dict[str, int] = {
            "1-5d": 0,
            "5-10d": 0,
            "10-20d": 0,
            "20+d": 0,
        }
        total_days = 0
        match_count = 0
        for _symbol, trades in all_symbols.items():
            for p in fifo_match(trades, with_fees=True):
                profit_pct = float(p.exit_price / p.entry_price) - 1
                entry_bucket_counts[AnalyzerExt._bucket_entry_profit(profit_pct)] += 1
                holding_bucket_counts[AnalyzerExt._bucket_holding(p.holding_days)] += 1
                total_days += p.holding_days
                match_count += 1

        avg_entry_to_exit = total_days / max(match_count, 1) if match_count else 0.0
        position_attribution = PositionAttributionReport(
            entry_price_distribution=list(entry_bucket_counts.items()),
            holding_period_distribution=list(holding_bucket_counts.items()),
            avg_entry_to_exit=float(avg_entry_to_exit),
        )

        max_drawdown = 0.0
        max_dd_date: datetime | None = None
        dd_duration = 0
        recovery_days: int | None = None
        avg_recovery_days: float = 0.0
        if equity_list and len(equity_list) > 1:
            values = [float(p.total_value) for p in equity_list]
            timestamps = [p.timestamp for p in equity_list]
            rolling_max = pd.Series(values).cummax()
            drawdown = (pd.Series(values) - rolling_max) / rolling_max
            if not drawdown.empty:
                max_dd = float(drawdown.min())
                max_dd_idx = int(drawdown.idxmin())
                max_dd_date = timestamps[max_dd_idx]
                max_drawdown = max_dd
                in_drawdown = False
                cur_len = 0
                max_len = 0
                for v in drawdown:
                    if v < 0:
                        if not in_drawdown:
                            in_drawdown = True
                            cur_len = 1
                        else:
                            cur_len += 1
                        max_len = max(max_len, cur_len)
                    else:
                        in_drawdown = False
                        cur_len = 0
                dd_duration = int(max_len)
                target = rolling_max.iloc[max_dd_idx]
                recovered_idx = None
                for i in range(max_dd_idx + 1, len(values)):
                    if values[i] >= target:
                        recovered_idx = i
                        break
                if recovered_idx is not None and max_dd_date is not None:
                    recovery_days = (timestamps[recovered_idx] - max_dd_date).days
                    avg_recovery_days = float(recovery_days)

        drawdown_report = DrawdownRecoveryReport(
            max_drawdown=float(max_drawdown),
            max_drawdown_date=max_dd_date,
            drawdown_duration_days=dd_duration,
            recovery_time_days=recovery_days,
            avg_recovery_days=avg_recovery_days,
        )

        # TODO: 从各 regime 区间的实际日收益计算 return/sharpe/max_dd
        # 当前占位，等分区间收益聚合逻辑完成后再填充
        import warnings

        warnings.warn(
            "ExtendedAnalyzer: 区间分析使用 0.0 占位值。需实现分区间日收益聚合逻辑。",
            UserWarning,
            stacklevel=2,
        )
        regime_report = RegimeAwareReport(
            bull_stats={"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            bear_stats={"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            sideways_stats={"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            regime_switches=regime_switches,
        )

        return ExtendedReport(
            base_performance=base,
            regime_report=regime_report,
            per_symbol_reports=symbol_reports,
            position_attribution=position_attribution,
            drawdown_recovery=drawdown_report,
        )
