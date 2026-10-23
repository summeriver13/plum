"""行业 ETF 位置带（回撤-反弹）信号与组合数学模型。

信号口径（回撤-反弹对称）
    低点（建仓下限）：``dd = close / rolling_max(L) - 1``
        dd < -15% → 下限 1/3　dd < -25% → 2/3　dd < -35% → 1
    高点（减仓上限）：``rb = close / rolling_min(L) - 1``
        rb > +15% → 上限 2/3　rb > +25% → 1/3　rb > +35% → 0
    合成：``目标 = clamp(dd 下限, 上限 = rb 上限)`` —— 卖出优先；
    可选 ``confirm``：``mom120 <= 0`` 时目标清 0（不接下落刀）；
    可选 ``cap_enabled=False``：去掉 rb 上限，只做「越跌越买」（用于分解卖出侧贡献）。

组合口径
    每只预算 ``1/N``，闲置资金计 1.5% 停泊；日频调仓到目标，换手按 ETF 佣金 5bp 计成本；
    当日收盘信号 → 次日生效（无前视）。基准必须给闲置现金同样计停泊，否则 edge 会虚高。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

FEE_RATE = 0.00005  # ETF 佣金（无印花税）
CASH_YIELD = 0.015  # 闲置资金停泊年化
TRADING_DAYS = 252
LEVELS = (0.15, 0.25, 0.35)  # 建仓/减仓三档阈值
LADDER = (1.0 / 3.0, 2.0 / 3.0, 1.0)  # 三档对应仓位
MOM_WINDOW = 120


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（None/NaN/异常值回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _int(value: Any, default: int = 0) -> int:
    """安全转 int（None/NaN/越界回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass(frozen=True)
class Result:
    """一条净值曲线的汇总。"""

    name: str
    total: float
    annual: float
    max_dd: float
    sharpe: float
    calmar: float
    exposure: float
    turnover: float


@dataclass(frozen=True)
class Bucket:
    """位置分档的样本量与未来收益均值。"""

    bucket: str
    n: int
    mean: float


def build_targets(
    adj: pd.DataFrame,
    lookback: int,
    *,
    confirm: bool = False,
    cap_enabled: bool = True,
    levels: tuple[float, ...] = LEVELS,
    base: float = 0.0,
) -> pd.DataFrame:
    """目标权重（0~1，未乘预算）：dd 给下限，rb（可选）给上限。

    ``base`` > 0 时改为「基础仓位 + 信号加仓」：每只先恒持 ``base`` 份额，
    余下 ``1 - base`` 按位置阶梯分配（用于把信号叠加在恒定仓位之上）。
    """
    window = max(lookback // 2, MOM_WINDOW)
    rolling_max = cast("pd.DataFrame", adj.rolling(lookback, min_periods=window).max())
    rolling_min = cast("pd.DataFrame", adj.rolling(lookback, min_periods=window).min())
    dd = adj / rolling_max - 1.0
    rb = adj / rolling_min - 1.0
    floor = pd.DataFrame(0.0, index=dd.index, columns=dd.columns)
    for level, size in zip(levels, LADDER):
        floor = floor.mask(dd <= -level, size)
    floor = cast("pd.DataFrame", floor.where(dd.notna() & rb.notna(), 0.0))
    if confirm:
        floor = floor.where(adj > adj.shift(MOM_WINDOW), 0.0)
    if not cap_enabled:
        signal = floor.fillna(0.0)
    else:
        cap = pd.DataFrame(1.0, index=rb.index, columns=rb.columns)
        for level, size in zip(levels, LADDER):
            cap = cap.mask(rb >= level, 1.0 - size)
        signal = cast("pd.DataFrame", floor.clip(upper=cap)).fillna(0.0)
    if base <= 0.0:
        return signal
    scaled = cast("pd.DataFrame", base + (1.0 - base) * signal)
    return scaled


def build_xsec_targets(
    adj: pd.DataFrame,
    lookback: int,
    *,
    dd_max: float = -0.40,
    top_n: int = 5,
    hold_days: int = 60,
) -> pd.DataFrame:
    """横截面版：每 ``hold_days`` 选出跌得最深的 ``top_n`` 个行业等权持有。

    与 ``build_targets`` 的区别：不做单品种仓位阶梯，而是把资金集中到
    「当日位置最深、且跌破 ``dd_max``」的少数行业上（不足 ``top_n`` 只时留现金）。
    """
    window = max(lookback // 2, MOM_WINDOW)
    highest = cast("pd.DataFrame", adj.rolling(lookback, min_periods=window).max())
    depth = (adj / highest - 1.0).to_numpy(dtype="float64")
    targets = np.zeros(depth.shape)
    picked: list[int] = []
    for step in range(depth.shape[0]):
        if step % hold_days == 0:
            row = depth[step]
            eligible = np.flatnonzero(np.isfinite(row) & (row <= dd_max))
            deepest = eligible[np.argsort(row[eligible])[:top_n]]
            picked = [_int(value) for value in deepest]
        targets[step, picked] = 1.0 / top_n
    return pd.DataFrame(targets, index=adj.index, columns=list(adj.columns))


def equity_frame(
    index: pd.Index,
    equity: np.ndarray,
    exposure: np.ndarray,
    *,
    scale: float = 1_000_000.0,
) -> pd.DataFrame:
    """纸面净值 → 引擎 run dir 的 equity_curve 结构（供叠加分析脚本复用）。"""
    invested = np.clip(exposure, 0.0, 1.0)
    cash_share = 1.0 - invested
    return pd.DataFrame(
        {
            "timestamp": index.to_numpy(),
            "total_value": equity * scale,
            "cash": equity * cash_share * scale,
            "frozen_cash": 0.0,
            "positions_value": equity * invested * scale,
            "cash_interest": equity * cash_share * (CASH_YIELD / TRADING_DAYS) * scale,
        }
    )


def simulate(
    targets: pd.DataFrame, ret: pd.DataFrame, budget: float
) -> tuple[np.ndarray, np.ndarray]:
    """日频调仓纸面回测，返回（净值、逐日股票仓位）。"""
    weight = targets.to_numpy(dtype="float64") * budget
    daily = np.nan_to_num(ret.to_numpy(dtype="float64"), nan=0.0)
    cash_rate = CASH_YIELD / TRADING_DAYS
    equity = np.ones(len(targets))
    exposure = np.zeros(len(targets))
    for step in range(1, len(targets)):
        held = weight[step - 1]
        port_ret = _num(held @ daily[step])
        drift = held * (1.0 + daily[step]) / (1.0 + port_ret)
        cost = FEE_RATE * _num(np.abs(weight[step] - drift).sum())
        equity[step] = (
            equity[step - 1]
            * (1.0 + port_ret)
            * (1.0 - cost)
            * (1.0 + cash_rate * (1.0 - _num(held.sum())))
        )
        exposure[step] = _num(weight[step].sum())
    return equity, exposure


def simulate_sleeve(exposure: np.ndarray, sleeve: np.ndarray) -> np.ndarray:
    """把给定仓位序列投到「池子等权」袖套上（同仓位基准用）。"""
    cash_rate = CASH_YIELD / TRADING_DAYS
    equity = np.ones(len(exposure))
    for step in range(1, len(exposure)):
        held = exposure[step - 1]
        port_ret = _num(held * sleeve[step])
        drift = held * (1.0 + sleeve[step]) / (1.0 + port_ret)
        cost = FEE_RATE * abs(exposure[step] - drift)
        equity[step] = (
            equity[step - 1]
            * (1.0 + port_ret)
            * (1.0 - cost)
            * (1.0 + cash_rate * (1.0 - held))
        )
    return equity


def summarize(name: str, equity: np.ndarray, exposure: np.ndarray) -> Result:
    """净值 → 指标。"""
    years = len(equity) / TRADING_DAYS
    total = _num(equity[-1]) - 1.0
    annual = (1.0 + total) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = np.maximum.accumulate(equity)
    max_dd = _num((equity / peak - 1.0).min())
    steps = np.diff(equity) / equity[:-1]
    vol = _num(np.std(steps, ddof=1)) * np.sqrt(TRADING_DAYS)
    sharpe = _num(np.mean(steps)) * TRADING_DAYS / vol if vol > 1e-9 else 0.0
    return Result(
        name=name,
        total=total,
        annual=annual,
        max_dd=max_dd,
        sharpe=sharpe,
        calmar=annual / abs(max_dd) if max_dd < 0 else 0.0,
        exposure=_num(np.mean(exposure)),
        turnover=_num(np.abs(np.diff(exposure)).sum()) / max(years, 1e-9),
    )


def forward_matrix(adj: pd.DataFrame, horizon: int, demean: bool = False) -> pd.DataFrame:
    """未来 horizon 日收益矩阵；``demean=True`` 时减去同期池子均值（横截面超额）。"""
    forward = cast("pd.DataFrame", adj.shift(-horizon) / adj - 1.0)
    return forward.sub(forward.mean(axis=1), axis=0) if demean else forward


def bucket_report(
    level: pd.DataFrame,
    forward: pd.DataFrame,
    label: str,
    *,
    levels: tuple[float, ...] = LEVELS,
) -> list[Bucket]:
    """把位置指标分档，统计各档未来收益（``forward`` 见 ``forward_matrix``）。"""
    values = forward.to_numpy(dtype="float64").ravel()
    flat = level.to_numpy(dtype="float64").ravel()
    valid = np.isfinite(flat) & np.isfinite(values)
    depth = np.digitize(-flat[valid] if label == "dd" else flat[valid], levels)
    horizon_return = values[valid]
    edges = (
        f"{label} 未达一档(<{levels[0]:.0%})",
        f"{label} 一档({levels[0]:.0%}~{levels[1]:.0%})",
        f"{label} 二档({levels[1]:.0%}~{levels[2]:.0%})",
        f"{label} 三档(>={levels[2]:.0%})",
    )
    return [
        Bucket(
            bucket=edges[index],
            n=_int((depth == index).sum()),
            mean=_num(horizon_return[depth == index].mean())
            if (depth == index).any()
            else 0.0,
        )
        for index in range(len(levels) + 1)
    ]


def format_results(results: list[Result]) -> str:
    """渲染 markdown 表。"""
    lines = [
        "| 变体 | 总收益 | 年化 | 回撤 | Sharpe | Calmar | 平均仓位 | 换手/年 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.name} | {item.total:+.1%} | {item.annual:+.1%} | "
            f"{item.max_dd:.1%} | {item.sharpe:.2f} | {item.calmar:.2f} | "
            f"{item.exposure:.0%} | {item.turnover:.1f}x |"
        )
    return "\n".join(lines)
