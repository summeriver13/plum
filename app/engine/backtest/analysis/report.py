"""五层统一分析报告 — 收益/风险/风险调整/交易行为/策略真实性。

输入 BacktestResult，输出 ComprehensiveReport，覆盖机构级回测分析。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.engine.backtest.analysis._numeric import equity_to_numpy
from app.engine.backtest.analysis.slippage_check import scan_slippage_anomalies

logger = logging.getLogger(__name__)


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值字符串返回 default），保持 NaN/inf 原样传递。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_mapping(value: object) -> dict[str, Any]:
    """把「字典 / JSON 字符串」统一成 dict。

    结果目录的 metadata.json 会把嵌套字典序列化成字符串（如 exec_sanity），
    从盘上重建报告时需要还原，否则自检字段会静默丢失。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/非数值字符串返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


TRADING_DAYS_PER_YEAR = 242
DEFAULT_RISK_FREE_RATE = 0.02


# ── 报告容器 ──────────────────────────────────────────


@dataclass
class ComprehensiveReport:
    run_id: str = ""
    strategy: str = ""
    start: str = ""
    end: str = ""
    trading_days: int = 0

    # Layer 1: 收益
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: dict[str, float] = field(default_factory=dict)
    yearly_returns: dict[str, float] = field(default_factory=dict)
    win_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Layer 2: 风险
    max_drawdown: float = 0.0
    max_drawdown_start: str = ""
    max_drawdown_end: str = ""
    max_drawdown_days: int = 0
    drawdown_intervals: list[dict] = field(default_factory=list)  # 前 5 次最大回撤详表
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_single_day_loss: float = 0.0

    # Layer 3: 风险调整
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Layer 4: 交易行为
    total_trades: int = 0
    win_rate: float = 0.0
    avg_holding_days: float = 0.0
    annual_turnover: float = 0.0
    idle_ratio: float = 0.0
    profit_factor: float = 0.0

    # Layer 5: 真实性
    half_sharpe: float = 0.0
    daily_skew: float = 0.0
    daily_kurtosis: float = 0.0
    recovery_factor: float = 0.0

    # Layer 6: 基准对比（需传 benchmark_values）
    benchmark_symbol: str = ""
    benchmark_total_return: float = 0.0
    benchmark_annual_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0

    # Layer 7: 摩擦成本
    total_friction: float = 0.0
    friction_ratio: float = 0.0
    # 自检：滑点占比异常的成交（幽灵滑点报警）
    slippage_anomaly_count: int = 0
    slippage_anomaly_cost: float = 0.0
    max_slippage_ratio: float = 0.0

    # 自检：涨跌停裁剪发生率（裁剪 = 成交价被涨跌停带收紧，静默故障的伴随指标）
    exec_price_calls: int = 0
    clip_count: int = 0
    clip_ratio: float = 0.0
    clip_notional: float = 0.0
    no_limit_calls: int = 0  # 数据湖标注为「无涨跌幅限制」而跳过裁剪的次数

    # Layer 8: 闲置资金收益（国债逆回购 / 货基停泊）
    cash_interest_total: float = 0.0  # 累计利息（元）
    cash_interest_annual: float = 0.0  # 对年化的算术贡献（日频可加口径）
    cash_interest_share: float = 0.0  # 累计利息占累计收益金额比
    cash_ratio: float = 0.0  # 平均闲置资金（现金 + 冻结）占总权益比
    cash_yield_rate: float = 0.0  # 实配停泊年化收益率
    park_ratio: float = 1.0  # 实配可停泊资金比例
    # 剥离停泊后的对照指标（近似：仓位规模未重算）
    annual_return_ex: float = 0.0
    sharpe_ex: float = 0.0
    max_drawdown_ex: float = 0.0
    calmar_ex: float = 0.0

    # 原始数据
    equity_values: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "period": f"{self.start} ~ {self.end}",
            "trading_days": self.trading_days,
            "收益": {
                "total_return": f"{self.total_return:.2%}",
                "annual_return": f"{self.annual_return:.2%}",
                "win_ratio": f"{self.win_ratio:.2%}",
                "avg_win": f"{self.avg_win:.2%}",
                "avg_loss": f"{self.avg_loss:.2%}",
            },
            "风险": {
                "max_drawdown": f"{self.max_drawdown:.2%}",
                "max_drawdown_start": self.max_drawdown_start,
                "max_drawdown_end": self.max_drawdown_end,
                "max_drawdown_days": self.max_drawdown_days,
                "drawdown_intervals": self.drawdown_intervals,
                "annual_volatility": f"{self.annual_volatility:.2%}",
                "downside_volatility": f"{self.downside_volatility:.2%}",
                "VaR_95": f"{self.var_95:.2%}",
                "CVaR_95": f"{self.cvar_95:.2%}",
                "max_single_day_loss": f"{self.max_single_day_loss:.2%}",
            },
            "风险调整": {
                "sharpe": round(self.sharpe_ratio, 2),
                "sortino": round(self.sortino_ratio, 2),
                "calmar": round(self.calmar_ratio, 2),
                "omega": round(self.omega_ratio, 2),
            },
            "交易行为": {
                "total_trades": self.total_trades,
                "win_rate": f"{self.win_rate:.2%}",
                "avg_holding_days": round(self.avg_holding_days, 1),
                "annual_turnover": f"{self.annual_turnover:.1f}x",
                "idle_ratio": f"{self.idle_ratio:.1%}",
                "profit_factor": round(self.profit_factor, 2),
            },
            "真实性": {
                "half_sharpe": round(self.half_sharpe, 2),
                "daily_skew": round(self.daily_skew, 2),
                "daily_kurtosis": round(self.daily_kurtosis, 2),
                "recovery_factor": round(self.recovery_factor, 2),
            },
            "基准对比": {
                "benchmark_symbol": self.benchmark_symbol,
                "benchmark_return": f"{self.benchmark_total_return:.2%}",
                "benchmark_annual": f"{self.benchmark_annual_return:.2%}",
                "alpha": f"{self.alpha:.2%}",
                "beta": round(self.beta, 3),
                "tracking_error": f"{self.tracking_error:.2%}",
                "information_ratio": round(self.information_ratio, 3),
            },
            "摩擦成本": {
                "total_friction": round(self.total_friction, 2),
                "friction_ratio": f"{self.friction_ratio:.4%}",
                "slippage_anomaly_count": self.slippage_anomaly_count,
                "slippage_anomaly_cost": round(self.slippage_anomaly_cost, 2),
                "max_slippage_ratio": f"{self.max_slippage_ratio:.4%}",
            },
            "成交价自检": {
                "exec_price_calls": self.exec_price_calls,
                "clip_count": self.clip_count,
                "clip_ratio": f"{self.clip_ratio:.2%}",
                "clip_notional": round(self.clip_notional, 2),
                "no_limit_calls": self.no_limit_calls,
            },
            # 明细数据（供 MD 报告展开）
            "monthly_returns": self.monthly_returns,
            "yearly_returns": self.yearly_returns,
            "daily_returns": self.daily_returns,
        }
        # 只在确实配置了停泊收益时输出分项，避免旧报告出现空section
        if self.cash_interest_total > 0:
            result["闲置资金收益"] = {
                "cash_interest_total": round(self.cash_interest_total, 2),
                "cash_interest_annual": f"{self.cash_interest_annual:.2%}",
                "cash_interest_share": f"{self.cash_interest_share:.1%}",
                "cash_ratio": f"{self.cash_ratio:.1%}",
                "cash_yield_rate": f"{self.cash_yield_rate:.2%}",
                "park_ratio": f"{self.park_ratio:.0%}",
                "annual_return_ex": f"{self.annual_return_ex:.2%}",
                "sharpe_ex": round(self.sharpe_ex, 2),
                "max_drawdown_ex": f"{self.max_drawdown_ex:.2%}",
                "calmar_ex": round(self.calmar_ex, 2),
            }
        return result

    def summary(self) -> str:
        """单行摘要。"""
        return (
            f"{self.strategy} | {self.start[:4]}~{self.end[:4]} | "
            f"Return={self.total_return:.1%} | "
            f"Sharpe={self.sharpe_ratio:.2f} | "
            f"MDD={self.max_drawdown:.1%} | "
            f"Trades={self.total_trades} | "
            f"WinRate={self.win_rate:.1%}"
        )


# ── 计算入口 ──────────────────────────────────────────


def comprehensive_report(
    result,
    run_id: str = "",
    benchmark_values: list[float] | None = None,
    benchmark_symbol: str | None = None,
) -> ComprehensiveReport:
    """从 BacktestResult 生成七层分析报告。

    Args:
        result: engine.run() 返回的 BacktestResult
        run_id: 运行标识
        benchmark_values: 基准净值序列（与 equity 同长度），用于 alpha/beta
        benchmark_symbol: 基准指数代码（如 000852.SH），自动从 catalog 加载并对齐日期
    """
    # 自动加载基准指数
    if benchmark_symbol and benchmark_values is None:
        benchmark_values = _load_benchmark(benchmark_symbol, result)
    equity = result.equity_curve
    if not equity or len(equity) < 2:
        return ComprehensiveReport(run_id=run_id)

    values = equity_to_numpy(equity)
    timestamps = [p.timestamp for p in equity]
    init = values[0]
    if init == 0:
        return ComprehensiveReport(run_id=run_id)

    # 基础
    total_ret = (values[-1] / init) - 1
    days = len(values)
    years = days / TRADING_DAYS_PER_YEAR
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    daily_rets = np.diff(values) / values[:-1]
    daily_rets = daily_rets[~np.isnan(daily_rets)]

    # Layer 1 — 收益
    report = ComprehensiveReport(
        run_id=run_id,
        strategy=result.symbol,
        start=timestamps[0].strftime("%Y-%m-%d"),
        end=timestamps[-1].strftime("%Y-%m-%d"),
        trading_days=days,
        total_return=total_ret,
        annual_return=annual_ret,
        benchmark_symbol=benchmark_symbol or "",
        equity_values=values,
        daily_returns=daily_rets.tolist(),
    )

    # Monthly/yearly returns
    report.monthly_returns = _monthly_returns(timestamps, values)
    report.yearly_returns = _yearly_returns(timestamps, values)

    # Win/loss
    wins = daily_rets[daily_rets > 0]
    losses = daily_rets[daily_rets < 0]
    report.win_ratio = len(wins) / len(daily_rets) if len(daily_rets) > 0 else 0
    report.avg_win = _num(np.mean(wins)) if len(wins) > 0 else 0
    report.avg_loss = _num(np.mean(losses)) if len(losses) > 0 else 0

    # Layer 2 — 风险（追踪所有回撤区间，不止最大一次）
    peak = values[0]
    mdd = 0.0
    mdd_start_i = mdd_end_i = mdd_pk_i = 0
    in_drawdown = False
    dd_start = 0
    dd_peak = values[0]
    dd_trough = values[0]
    dd_trough_idx = 0
    all_drawdowns: list[dict] = []

    for i, v in enumerate(values):
        if v > peak:
            peak = v
            # 退出回撤 -> 记录
            if in_drawdown:
                depth = (dd_peak - dd_trough) / dd_peak
                recovery_days = i - dd_trough_idx if dd_trough_idx > dd_start else 0
                all_drawdowns.append(
                    {
                        "start": timestamps[dd_start].strftime("%Y-%m-%d"),
                        "end": timestamps[i - 1].strftime("%Y-%m-%d"),
                        "depth": depth,
                        "days": i - dd_start,
                        "recovery": recovery_days,
                    }
                )
                in_drawdown = False
        current_dd = (peak - v) / peak
        if current_dd > 0 and not in_drawdown:
            in_drawdown = True
            dd_start = i
            dd_peak = peak
            dd_trough = v
            dd_trough_idx = i
        elif in_drawdown:
            if v < dd_trough:
                dd_trough = v
                dd_trough_idx = i
        # track max
        if current_dd > mdd:
            mdd = current_dd
            mdd_end_i = i
            mdd_pk_i = values.tolist().index(peak)
            mdd_start_i = mdd_pk_i
            for j in range(mdd_pk_i, 0, -1):
                if values[j - 1] < values[j]:
                    mdd_start_i = j
                    break

    # 结束时仍在回撤中
    if in_drawdown:
        depth = (dd_peak - dd_trough) / dd_peak
        all_drawdowns.append(
            {
                "start": timestamps[dd_start].strftime("%Y-%m-%d"),
                "end": timestamps[-1].strftime("%Y-%m-%d"),
                "depth": depth,
                "days": len(values) - dd_start,
                "recovery": -1,  # 未恢复
            }
        )

    # 取深度前 5
    all_drawdowns.sort(key=lambda x: x["depth"], reverse=True)
    report.drawdown_intervals = all_drawdowns[:5]

    report.max_drawdown = mdd
    report.max_drawdown_days = mdd_end_i - mdd_start_i if mdd_end_i > mdd_start_i else 0
    report.max_drawdown_start = (
        timestamps[mdd_start_i].strftime("%Y-%m-%d")
        if mdd_start_i < len(timestamps)
        else ""
    )
    report.max_drawdown_end = (
        timestamps[mdd_end_i].strftime("%Y-%m-%d")
        if mdd_end_i < len(timestamps)
        else ""
    )

    # Volatility
    daily_std = _num(np.std(daily_rets)) if len(daily_rets) > 1 else 0
    report.annual_volatility = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    neg_rets = daily_rets[daily_rets < 0]
    if len(neg_rets) > 1:
        neg_std = _num(np.std(neg_rets))
        report.downside_volatility = neg_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    # VaR / CVaR
    if len(daily_rets) > 5:
        report.var_95 = _num(np.percentile(daily_rets, 5))
        report.cvar_95 = _num(np.mean(daily_rets[daily_rets <= report.var_95]))
        report.max_single_day_loss = _num(np.min(daily_rets))

    # Layer 3 — 风险调整收益
    excess = daily_rets - DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    if daily_std > 0:
        report.sharpe_ratio = _num(
            np.mean(excess) / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    if report.downside_volatility > 0:
        report.sortino_ratio = (
            annual_ret - DEFAULT_RISK_FREE_RATE
        ) / report.downside_volatility
    if mdd > 0:
        report.calmar_ratio = annual_ret / mdd
    if len(losses) > 0 and len(wins) > 0:
        total_gain = _num(np.sum(wins))
        total_loss = _num(np.abs(np.sum(losses)))
        report.omega_ratio = total_gain / total_loss if total_loss > 0 else math.inf

    # Layer 4 — 交易行为
    trade_log = result.trade_log
    report.total_trades = len(trade_log)
    if trade_log:
        from ._fifo import fifo_match

        pairs = fifo_match(trade_log)
        if pairs:
            report.win_rate = sum(1 for p in pairs if p.is_win) / len(pairs)
            report.avg_holding_days = _num(np.mean([p.holding_days for p in pairs]))
            total_gain = _num(sum(p.gross_pnl for p in pairs if p.gross_pnl > 0))
            total_loss = abs(_num(sum(p.gross_pnl for p in pairs if p.gross_pnl < 0)))
            report.profit_factor = (
                total_gain / total_loss if total_loss > 0 else math.inf
            )

        # Turnover: 年化换手率 = 总交易额 / (平均组合市值 × 年数)
        total_traded = 0.0
        for t in trade_log:
            qty = abs(t.quantity) if hasattr(t, "quantity") else 0
            price = _num(t.price) if hasattr(t, "price") else 0
            total_traded += qty * price
        avg_value = _num(np.mean(values)) if len(values) > 0 else 1e6
        report.annual_turnover = (
            total_traded / (avg_value * years) if years > 0 and avg_value > 0 else 0
        )

    # Idle ratio — how many days with zero positions
    positions_flat = 0
    for p in equity:
        detail = getattr(p, "positions_detail", None) or {}
        if not detail or sum(detail.values()) == 0:
            positions_flat += 1
    report.idle_ratio = positions_flat / days if days > 0 else 0

    # Layer 5 — 真实性
    # Half-period Sharpe (first half vs full)
    half = len(daily_rets) // 2
    if half > 10:
        half_daily = daily_rets[:half]
        half_std = _num(np.std(half_daily))
        if half_std > 0:
            report.half_sharpe = _num(
                np.mean(half_daily - DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
                / half_std
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

    # Distribution — 纯 numpy 实现，无需 scipy
    if len(daily_rets) > 10:
        mu = np.mean(daily_rets)
        sigma = np.std(daily_rets)
        if sigma > 1e-10:
            z = (daily_rets - mu) / sigma
            report.daily_skew = _num(np.mean(z**3))
            report.daily_kurtosis = _num(np.mean(z**4)) - 3  # excess kurtosis

    # Recovery factor
    if mdd > 0 and abs(total_ret) > 1e-8:
        report.recovery_factor = abs(total_ret) / mdd

    # Layer 6 — 基准对比（需传入 benchmark_values）
    if benchmark_values and len(benchmark_values) == len(values):
        bm = np.array(benchmark_values, dtype=float)
        bm_init = bm[0]
        if bm_init > 0:
            report.benchmark_total_return = _num(bm[-1] / bm_init - 1)
            bm_years = len(bm) / TRADING_DAYS_PER_YEAR
            report.benchmark_annual_return = (
                (1 + report.benchmark_total_return) ** (1 / bm_years) - 1
                if bm_years > 0
                else 0
            )
            bm_rets = np.diff(bm) / bm[:-1]
            bm_rets = bm_rets[~np.isnan(bm_rets)]

            # Align strategy and benchmark returns
            min_len = min(len(daily_rets), len(bm_rets))
            s_ret = daily_rets[:min_len]
            b_ret = bm_rets[:min_len]

            cov = np.cov(s_ret, b_ret)[0, 1] if len(s_ret) > 1 else 0
            bm_var = np.var(b_ret) if len(b_ret) > 1 else 0
            report.beta = _num(cov / bm_var) if bm_var > 1e-10 else 1.0
            report.alpha = _num(
                report.annual_return
                - DEFAULT_RISK_FREE_RATE
                - report.beta
                * (report.benchmark_annual_return - DEFAULT_RISK_FREE_RATE)
            )

            active_rets = s_ret - b_ret
            te = _num(np.std(active_rets)) * np.sqrt(_num(TRADING_DAYS_PER_YEAR))
            report.tracking_error = _num(te)
            report.information_ratio = _num(report.alpha / te) if te > 1e-10 else 0.0

    # Layer 7 — 摩擦成本（从 trade_log 汇总）
    trade_log = result.trade_log
    if trade_log:
        total_friction = 0.0
        for t in trade_log:
            cost = getattr(t, "cost", None)
            if cost:
                total_friction += _num(getattr(cost, "total", 0) or 0)
            total_friction += _num(getattr(t, "slippage_cost", 0) or 0)
        report.total_friction = total_friction
        if total_friction > 0 and init > 0:
            report.friction_ratio = total_friction / init

        anomaly = scan_slippage_anomalies(trade_log)
        report.slippage_anomaly_count = anomaly.count
        report.slippage_anomaly_cost = _num(anomaly.cost)
        report.max_slippage_ratio = anomaly.max_ratio
        if anomaly.warning():
            logger.warning(anomaly.warning())

    sanity = _as_mapping((getattr(result, "metadata", None) or {}).get("exec_sanity"))
    report.exec_price_calls = _int(sanity.get("exec_price_calls", 0) or 0)
    report.clip_count = _int(sanity.get("clip_count", 0) or 0)
    report.clip_ratio = _num(sanity.get("clip_ratio", 0.0) or 0.0)
    report.clip_notional = _num(sanity.get("clip_notional", 0.0) or 0.0)
    report.no_limit_calls = _int(sanity.get("no_limit_calls", 0) or 0)

    # Layer 8 — 闲置资金收益分项
    _apply_cash_interest(report, result, equity, values, years)

    return report


# ── 闲置资金收益 ──────────────────────────────────────


def _max_drawdown_depth(values: np.ndarray) -> float:
    """仅算最大回撤深度（不做区间追踪）。"""
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(peak > 0, (peak - values) / peak, 0.0)
    return _num(np.max(depth))


def _daily_cash_ratios(equity: list) -> list[float]:
    """每日闲置资金（现金 + 冻结）占总权益的比例。"""
    ratios: list[float] = []
    for point in equity:
        total = _num(getattr(point, "total_value", 0) or 0)
        if total <= 0:
            continue
        idle = _num(getattr(point, "cash", 0) or 0) + _num(
            getattr(point, "frozen_cash", 0) or 0
        )
        ratios.append(idle / total)
    return ratios


def _apply_cash_interest(
    report: ComprehensiveReport,
    result,
    equity: list,
    values: np.ndarray,
    years: float,
) -> None:
    """填充分项：累计利息、平均闲置比例、以及剔除停泊后的对照指标。

    对照口径是**近似**：只从净值序列里减掉每日利息本金，
    不重算仓位规模（要精确对照需重跑一次零利率回测）。
    """
    ratios = _daily_cash_ratios(equity)
    report.cash_ratio = _num(np.mean(ratios)) if ratios else 0.0

    metadata = getattr(result, "metadata", None) or {}
    report.cash_yield_rate = _num(metadata.get("cash_yield", 0.0) or 0.0)
    report.park_ratio = _num(metadata.get("park_ratio", 1.0) or 1.0)

    interest = np.array(
        [_num(getattr(point, "cash_interest", 0) or 0) for point in equity],
        dtype=float,
    )
    total = _num(interest.sum())
    report.cash_interest_total = total
    if total <= 0 or years <= 0:
        return

    stripped = values - np.cumsum(interest)
    start = stripped[0] if stripped[0] > 0 else values[0]
    annual_ex = (stripped[-1] / start) ** (1 / years) - 1 if start > 0 else 0.0
    report.annual_return_ex = _num(annual_ex)

    # 算术贡献：Σ(利息_t / 前一日净值) / 年数，与日收益口径可加。
    # 除比 CAGR 差额小：后者还包含利息再投资带来的仓位放大。
    base = np.concatenate([[values[0] - interest[0]], values[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        flows = np.where(base > 0, interest / base, 0.0)
    report.cash_interest_annual = _num(np.sum(flows) / years)

    ex_rets = np.diff(stripped) / stripped[:-1]
    ex_rets = ex_rets[~np.isnan(ex_rets)]
    if len(ex_rets) > 1:
        ex_std = _num(np.std(ex_rets))
        if ex_std > 0:
            report.sharpe_ex = _num(
                np.mean(ex_rets - DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
                / ex_std
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
    report.max_drawdown_ex = _max_drawdown_depth(stripped)
    if report.max_drawdown_ex > 0:
        report.calmar_ex = report.annual_return_ex / report.max_drawdown_ex

    gain = values[-1] - values[0]
    if gain > 0:
        report.cash_interest_share = total / gain


# ── 辅助 ──────────────────────────────────────────────


def _load_benchmark(symbol: str, result) -> list[float] | None:
    """加载基准价格并对其到策略交易日期。

    交易所指数（``IDX.*``）走 MarketCatalog；plum 自建风格指数（``STYLE.*``）
    走 data/benchmarks。详见 app.engine.backtest.analysis.benchmarks。

    Returns:
        与 result.equity_curve 等长的收盘价序列，或 None
    """
    from app.engine.backtest.analysis.benchmarks import load_aligned_benchmark

    try:
        return load_aligned_benchmark(
            symbol, [p.timestamp for p in result.equity_curve]
        )
    except Exception:  # noqa: BLE001 — 基准失败降级为无基准，不阻断报告
        logger.debug("基准对齐失败: %s", symbol, exc_info=True)
        return None


def _monthly_returns(timestamps, values):
    """计算每月收益：(月末值 / 月初值) - 1。"""
    by_month: dict[str, list[float]] = {}
    for t, v in zip(timestamps, values):
        key = f"{t.year}-{t.month:02d}"
        by_month.setdefault(key, []).append(v)
    result = {}
    for key, vals in by_month.items():
        if vals and vals[0] > 0:
            result[key] = vals[-1] / vals[0] - 1
    return dict(sorted(result.items()))


def _yearly_returns(timestamps, values):
    """计算每年收益：(年末值 / 年初值) - 1。"""
    by_year: dict[str, list[float]] = {}
    for t, v in zip(timestamps, values):
        key = str(t.year)
        by_year.setdefault(key, []).append(v)
    result = {}
    for key, vals in by_year.items():
        if vals and vals[0] > 0:
            result[key] = vals[-1] / vals[0] - 1
    return dict(sorted(result.items()))
