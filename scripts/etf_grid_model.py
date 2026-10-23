"""ETF 网格模型（诊断用，非策略）：步进网格回放与单标的指标。

模型口径（保守，与实盘网格一致的假设）
    - 步进网格：相对「上一次成交价」上下各 ``step`` 挂一格，跌破买 1 档、涨破卖 1 档；
    - 成交价 = 档位价（限价单语义；跳空穿越按档位价成交 = 保守价）；
    - 触发只看**收盘价**，不用日内高低点 —— 避免「同一天既买又卖」的乐观假设；
    - 库存上限 ``MAX_LOTS`` 档即满仓，起点 ``INIT_LOTS`` 档，每档等资金；
    - 单边费用 ``FEE_RATE``（ETF 佣金 5bp，无印花税、无最低五元）；
    - 空闲现金按 ``CASH_YIELD`` 年化计息（网格常态闲置资金，不计息会低估网格）。

收益拆解
    期末权益 − 1 ≈ 已实现收割（移动平均成本法，含费用）+ 库存浮盈/亏 + 停泊利息。
    「网格 − 同平均仓位持有」才是网格的择时价值（把仓位差异剔除）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.etf_explain_panels import _num  # noqa: E402  （与仓库一致的安全转换）

MAX_LOTS = 10  # 库存上限（档）= 满仓
INIT_LOTS = 5  # 起点档数 = 半仓（实盘网格常态）
FEE_RATE = 0.00005  # 单边费用（ETF 佣金）
CASH_YIELD = 0.015  # 闲置现金年化（国债逆回购口径）
TRADING_DAYS = 252
MA_WINDOW = 200  # 趋势闸门均线窗口
GATE_OFF = "off"
GATE_BLOCK = "block_buys"
GATE_EXIT = "exit"
EPS = 1e-12


@dataclass(frozen=True)
class GridResult:
    """单标的单步长的网格回放结果（收益率口径，起始资金 = 1.0）。

    ``mix_*`` 为静态同配比（不再平衡、不计停泊），``fair_*`` 为公平基准
    （日度再平衡 + 闲置现金停泊 + 再平衡换手计佣）。"""

    symbol: str
    step: float
    grid_return: float
    grid_max_dd: float
    hold_return: float
    hold_max_dd: float
    mix_return: float
    mix_max_dd: float
    fair_return: float
    fair_max_dd: float
    avg_position: float
    realized: float
    trend_ratio: float
    round_trips: int
    fees: float
    bars: int


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """滚动均线（不回看未来：``ma[i]`` 只用到 ``values[:i+1]``）。"""
    cumulative = np.cumsum(values, dtype="float64")
    window_avg = cumulative / np.arange(1, len(values) + 1)
    if len(values) > window:
        window_avg[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return window_avg


def fair_benchmark(prices: np.ndarray, weight: float) -> np.ndarray:
    """同配比「公平基准」权益曲线：恒定权重日度再平衡 + 停泊 + 换手佣金。

    与网格比较时，基准必须同等对待闲置资金（计停泊）并同等计费，
    否则网格 edge 会虚高约 1pp/年（旧版基准既不再平衡也不计停泊）。
    """
    equity = np.empty(len(prices))
    equity[0] = 1.0
    if weight <= 0.0:
        equity[1:] = 1.0
        return equity
    cash_rate = CASH_YIELD / TRADING_DAYS
    value = 1.0
    for index in range(1, len(prices)):
        growth = _num(prices[index] / prices[index - 1])
        cash_part = (1.0 - weight) * (1.0 + cash_rate)
        total = weight * growth + cash_part
        value *= total
        drift = weight * growth / total  # 当日自然漂移后的仓位
        value *= 1.0 - FEE_RATE * abs(weight - drift)  # 再平衡回目标权重
        equity[index] = value
    return equity


def _max_drawdown(equity: np.ndarray) -> float:
    """最大回撤（负值）。"""
    peak = np.maximum.accumulate(equity)
    return _num((equity / peak - 1.0).min())


def trend_ratio(prices: np.ndarray) -> float:
    """趋势强度 = |年化收益| / 年化波动（价格路径的「效率比」，越高越像单边趋势）。

    网格的对手就是趋势：该值越大，网格越容易踏空或套牢。
    """
    returns = np.diff(prices) / prices[:-1]
    std = _num(np.std(returns, ddof=1)) if len(returns) > 2 else 0.0
    if std <= EPS:
        return 0.0
    years = len(prices) / TRADING_DAYS
    annual = (prices[-1] / prices[0]) ** (1.0 / max(years, EPS)) - 1.0
    return abs(_num(annual)) / (std * np.sqrt(TRADING_DAYS))


def simulate_grid(
    values: list[float],
    step: float,
    *,
    cash_yield: float = CASH_YIELD,
    gate: str = GATE_OFF,
) -> tuple[np.ndarray, float, float, float, int]:
    """回放步进网格，返回（逐日权益、已实现盈亏、费用、平均仓位、卖出次数）。

    ``values`` 为后复权收盘价（Python float 列表，避免循环内重复转换）。
    ``gate`` 为趋势闸门：``off`` 无闸门；``block_buys`` 价格低于 MA200 只卖不买；
    ``exit`` 低于 MA200 时清仓并在回到均线上方前不再买入。
    """
    lot_cash = 1.0 / MAX_LOTS
    p0 = values[0]
    shares = INIT_LOTS * lot_cash / p0
    avg_cost = p0 * (1.0 + FEE_RATE)
    cash = 1.0 - INIT_LOTS * lot_cash * (1.0 + FEE_RATE)
    last_fill = p0
    next_buy, next_sell = p0 * (1.0 - step), p0 * (1.0 + step)
    lots, realized, fees, sells = INIT_LOTS, 0.0, 0.0, 0
    daily_rate = cash_yield / TRADING_DAYS if cash_yield > 0 else 0.0
    ma = rolling_mean(np.asarray(values, dtype="float64"), MA_WINDOW) if gate != GATE_OFF else None

    equity = np.empty(len(values))
    position_value = np.empty(len(values))
    for index, value in enumerate(values):
        blocked = ma is not None and index >= MA_WINDOW - 1 and value < ma[index]
        if blocked and gate == GATE_EXIT and lots > 0:
            # 破位清仓：按当日收盘市价出货，网格线从清仓价重锚
            proceeds = shares * value
            realized += proceeds - shares * avg_cost - proceeds * FEE_RATE
            fees += proceeds * FEE_RATE
            cash += proceeds * (1.0 - FEE_RATE)
            shares, lots = 0.0, 0
            last_fill = value
            next_buy, next_sell = value * (1.0 - step), value * (1.0 + step)
        # 买入：跌到下一买点以下（跳空可连续成交多档），成交价 = 档位价
        while not blocked and value <= next_buy and lots < MAX_LOTS:
            bought = lot_cash / next_buy
            spend = lot_cash * (1.0 + FEE_RATE)
            shares += bought
            avg_cost = (avg_cost * (shares - bought) + spend) / shares
            cash -= spend
            fees += lot_cash * FEE_RATE
            lots += 1
            last_fill = next_buy
            next_buy, next_sell = last_fill * (1.0 - step), last_fill * (1.0 + step)
        # 卖出：涨到下一卖点以上，成交价 = 档位价
        while value >= next_sell and lots > 0:
            sold = lot_cash / next_sell
            realized += sold * (next_sell - avg_cost) - lot_cash * FEE_RATE
            shares -= sold
            cash += lot_cash * (1.0 - FEE_RATE)
            fees += lot_cash * FEE_RATE
            lots -= 1
            sells += 1
            last_fill = next_sell
            next_buy, next_sell = last_fill * (1.0 - step), last_fill * (1.0 + step)
        if daily_rate:
            cash *= 1.0 + daily_rate
        position_value[index] = shares * value
        equity[index] = cash + position_value[index]

    ratio = position_value / np.maximum(equity, EPS)
    return equity, realized, fees, _num(ratio.mean()), sells


def analyse_symbol(
    symbol: str, prices: np.ndarray, step: float, *, gate: str = GATE_OFF
) -> GridResult:
    """单标的：网格 vs 静态同配比 vs 公平基准 vs 买入持有（各含最大回撤）。"""
    equity, realized, fees, avg_pos, sells = simulate_grid(prices.tolist(), step, gate=gate)
    hold = prices / prices[0]
    mix = avg_pos * hold + (1.0 - avg_pos)
    fair = fair_benchmark(prices, avg_pos)
    return GridResult(
        symbol=symbol,
        step=step,
        grid_return=_num(equity[-1] - 1.0),
        grid_max_dd=_max_drawdown(equity),
        hold_return=_num(hold[-1] - 1.0),
        hold_max_dd=_max_drawdown(hold),
        mix_return=_num(mix[-1] - 1.0),
        mix_max_dd=_max_drawdown(mix),
        fair_return=_num(fair[-1] - 1.0),
        fair_max_dd=_max_drawdown(fair),
        avg_position=avg_pos,
        realized=realized,
        trend_ratio=trend_ratio(prices),
        round_trips=sells,
        fees=fees,
        bars=len(prices),
    )
