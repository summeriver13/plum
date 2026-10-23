"""ETF 轮动可解释性：逐日重建持仓，把权益变化拆到标的 / 摩擦 / 停泊。

权益恒等式（实测对账到 200 元内，残差来自份额取整）：
    权益变化 = Σ标的（毛贡献 − 摩擦）+ 停泊/利息 + 残差

三个口径约定（与引擎保持一致）：
- 毛贡献用**原始收盘价**差值：后复权价 = 原始价 × 累计复权因子，用它算差值会把金额
  放大到当前因子倍（实测残差 -3,500 万）；
- 公司行为（拆分/合并/分红）与引擎 `_apply_share_adjustments` 同口径：adj_factor 比值
  偏离 1 超过 ``ADJ_FACTOR_EPSILON`` 即按比例调整份额，并按份额增量计提增益；
- 摩擦 = 佣金/印花税/过户费 + 滑点（成交价偏离当日收盘价的部分，当日就发生、
  不计入收盘价口径的毛贡献，必须单独扣）；
- 交易以收盘价成交，故当日新建仓不产生当日盈亏（与引擎一致）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.etf_explain_panels import _int, _num  # 共享的数值安全转换

ADJ_FACTOR_EPSILON = 1e-6
"""与引擎同阈：复权因子比值偏差低于该值时视为无公司行为。"""

BUY_FEE_ESTIMATE_RATE = 0.00022
"""引擎买入预估费费率（``_execute_single_order`` 旧版用它算「退还」）。

历史缺陷 D1（2026-09-20 修正，见 ``docs/策略与回测/引擎缺陷与修正记录.md``）：
旧版在买入成交后做 ``cash += 预估成本 − 实际成本``，而 ``freeze_cash`` 只累加
frozen_cash、不动 cash，于是「预估费 − 实际佣金」凭空入账。本常量保留用于
**探测旧批次**：当对账残差与该差额同量级时提示该批次带 D1 缺陷。"""


@dataclass
class Attribution:
    """逐标的贡献分解结果。"""

    by_symbol: pd.DataFrame
    """列：symbol, category, pnl, fees, net, days_held, entries, net_share。"""

    exposure: pd.DataFrame
    """逐日分类暴露（元）：index=日期，columns=分类（按当日持仓市值）。"""

    costs: float
    """全部交易摩擦（佣金/印花税/过户费 + 滑点），元。"""

    net_contribution: float
    """Σ标的净贡献（毛贡献 − 摩擦），元。"""

    buy_fee_gap: float
    """Σ买入(成交额 × ``BUY_FEE_ESTIMATE_RATE`` − 佣金)，元。

    仅作缺陷探测用：残差与该值同量级时，该批次疑带 D1 缺陷（修正后应为 0）。"""

    realized_equity_change: float
    """回测期内权益变化（期末 − 期初），元。"""

    residual: float
    """权益变化 −（净贡献 + 停泊贡献）：未解释的残差，元（预期仅剩取整量级）。"""


def quantify_attribution(
    run_dir: Path,
    panels: dict[str, pd.DataFrame],
    classes: dict[str, str],
) -> Attribution:
    """逐日重建持仓，把权益变化拆到「标的贡献 / 摩擦成本 / 停泊 / 残差」。"""
    equity = pd.read_csv(
        run_dir / "data" / "equity_curve.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    trades = pd.read_csv(run_dir / "data" / "trade_log.csv", parse_dates=["timestamp"])

    close = panels["close"].reindex(equity.index).ffill()
    factor = panels["adj_factor"].reindex(equity.index).ffill()

    ledger = _Ledger()
    exposure_rows: list[dict[str, Any]] = []
    trades_by_day = {ts: group for ts, group in trades.groupby("timestamp")}
    prev_close: pd.Series | None = None

    for ts in equity.index:
        if prev_close is not None:
            _accrue_daily_pnl(ledger, factor.loc[ts], prev_close, close.loc[ts])
        for trade in _day_trades(trades_by_day.get(ts)):
            _apply_trade(ledger, trade, factor.loc[ts])
        prev_close = close.loc[ts]
        exposure_rows.append(
            _class_exposure(ts, ledger.holdings, close.loc[ts], classes)
        )

    total_costs = sum(ledger.fees.values())
    parking_pnl = _parking_pnl(run_dir, equity)
    totals = equity["total_value"].to_numpy(dtype=float)
    realized = _num(totals[-1] - totals[0])

    rows = [
        {
            "symbol": symbol,
            "category": classes.get(symbol, "其他/低相关"),
            "pnl": value,
            "fees": ledger.fees.get(symbol, 0.0),
            "net": value - ledger.fees.get(symbol, 0.0),
            "days_held": ledger.days_held.get(symbol, 0),
            "entries": ledger.entries.get(symbol, 0),
        }
        for symbol, value in ledger.pnl.items()
    ]
    net_total = _num(sum(row["net"] for row in rows))
    for row in rows:
        row["net_share"] = row["net"] / realized if realized else 0.0
    frame = (
        pd.DataFrame(rows).sort_values("net", ascending=False)
        if rows
        else pd.DataFrame()
    )
    return Attribution(
        by_symbol=frame,
        exposure=pd.DataFrame(exposure_rows).fillna(0.0).set_index("date")
        if exposure_rows
        else pd.DataFrame(),
        costs=total_costs,
        net_contribution=net_total,
        buy_fee_gap=_buy_fee_refund(trades),
        realized_equity_change=realized,
        residual=realized - (net_total + parking_pnl),
    )


@dataclass
class _Ledger:
    """逐日重建持仓用的可变账本。"""

    holdings: dict[str, int] = field(default_factory=dict)
    last_factor: dict[str, float] = field(default_factory=dict)
    pnl: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    fees: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    days_held: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    entries: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _accrue_daily_pnl(
    ledger: _Ledger,
    factor_row: pd.Series,
    prev_close: pd.Series,
    close_row: pd.Series,
) -> None:
    """逐日计提持仓损益（就地修改）：份额复权增益 + 价格变动。

    权益恒等式 ``qty新×收新 − qty旧×收旧 = qty新×(收新−收旧) + (qty新−qty旧)×收旧``：
    分红与拆分/合并都表现为复权因子跳变，只记价格变动会漏掉「分红再投资」那部分
    增益（实测残差 692 万）。
    """
    for symbol in list(ledger.holdings):
        before = _num(prev_close.get(symbol, 0.0))
        current = _num(close_row.get(symbol, 0.0))
        factor = _num(factor_row.get(symbol, 0.0))
        ratio = _share_adjustment_ratio(factor, ledger.last_factor.get(symbol, 0.0))
        if ratio is not None:
            adjusted = _int(ledger.holdings[symbol] * ratio)
            ledger.pnl[symbol] += (adjusted - ledger.holdings[symbol]) * before
            ledger.holdings[symbol] = adjusted
        if factor > 0:
            ledger.last_factor[symbol] = factor
        if before > 0 and current > 0:
            ledger.pnl[symbol] += ledger.holdings[symbol] * (current - before)
            ledger.days_held[symbol] += 1


def _share_adjustment_ratio(factor: float, base: float) -> float | None:
    """持仓相对上次同步的复权比例；无需调整时返回 None（引擎同口径）。"""
    if base <= 0 or factor <= 0:
        return None
    ratio = factor / base
    return None if abs(ratio - 1.0) < ADJ_FACTOR_EPSILON else ratio


def _apply_trade(ledger: _Ledger, trade: Any, factor_row: pd.Series) -> None:
    """把一笔成交计入账本（trade_log 的 quantity 恒为非负，方向由 is_buy 决定）。"""
    symbol = str(trade.symbol)
    signed = _int(trade.quantity)
    if not _as_bool(getattr(trade, "is_buy", True)):
        signed = -signed
    if signed > 0 and ledger.holdings.get(symbol, 0) == 0:
        ledger.entries[symbol] += 1
    ledger.holdings[symbol] = ledger.holdings.get(symbol, 0) + signed
    ledger.fees[symbol] += _trade_fees(trade)
    factor = _num(factor_row.get(symbol, 0.0))
    if factor > 0:
        ledger.last_factor[symbol] = factor


def _buy_fee_refund(trades: pd.DataFrame) -> float:
    """买入手续费预估退还：Σ(成交额 × ``BUY_FEE_ESTIMATE_RATE`` − 佣金)。"""
    is_buy = trades["is_buy"].map(_as_bool).to_numpy(dtype=bool)
    notional = trades["quantity"].to_numpy(dtype=float) * trades["price"].to_numpy(dtype=float)
    refund = notional * BUY_FEE_ESTIMATE_RATE - trades["commission"].to_numpy(dtype=float)
    return _num(refund[is_buy].sum())


def _class_exposure(
    ts: pd.Timestamp,
    holdings: dict[str, int],
    close_row: pd.Series,
    classes: dict[str, str],
) -> dict[str, Any]:
    """当日各分类的持仓市值（元）。"""
    totals: dict[str, float] = defaultdict(float)
    for symbol, quantity in holdings.items():
        if quantity == 0:
            continue
        totals[classes.get(symbol, "其他/低相关")] += quantity * _num(
            close_row.get(symbol, 0.0)
        )
    row: dict[str, Any] = {"date": ts}
    row.update(totals)
    return row


def _parking_pnl(run_dir: Path, equity: pd.DataFrame) -> float:
    """停泊贡献：停泊成交现金流 + 期末停泊市值；无标的停泊时退回利率利息合计。"""
    log_path = run_dir / "data" / "parking_log.csv"
    if not log_path.exists():
        interest = (
            equity["cash_interest"].to_numpy(dtype=float)
            if "cash_interest" in equity.columns
            else np.zeros(len(equity), dtype=float)
        )
        return _num(interest.sum())
    log = pd.read_csv(log_path)
    notional = log["notional"].to_numpy(dtype=float)
    fee = log["fee"].to_numpy(dtype=float)
    quantity = log["quantity"].to_numpy(dtype=int)
    side = (quantity > 0).astype(float) - (quantity < 0).astype(float)
    flow = _num((-notional * side - fee).sum())
    final_value = (
        _num(equity["parking_value"].to_numpy(dtype=float)[-1])
        if "parking_value" in equity.columns
        else 0.0
    )
    return flow + final_value


def _as_bool(value: Any) -> bool:
    """安全转 bool（兼容 CSV 中的 True/False 字符串）。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _day_trades(group: pd.DataFrame | None) -> list:
    """当日成交记录（无成交返回空列表）。"""
    return [] if group is None else list(group.itertuples())


def _trade_fees(trade: Any) -> float:
    """单笔成交的摩擦（佣金 + 印花税 + 过户费 + 滑点成本）。"""
    return (
        _num(getattr(trade, "commission", 0.0))
        + _num(getattr(trade, "stamp_duty", 0.0))
        + _num(getattr(trade, "transfer_fee", 0.0))
        + _num(getattr(trade, "slippage_cost", 0.0))
    )
