"""可解释性评估的统计原语：回归、分解、逐年/分段、集中度、超额显著性。

纯计算模块（不含 I/O），供 ``scripts/explain_results.py`` 调用。
约定：alpha 一律年化（日频 alpha × 252）；t 值用 Newey-West 修正自相关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NW_LAGS = 5  # Newey-West 滞后阶数（覆盖一周自相关）


@dataclass(frozen=True)
class Regression:
    """单因子回归结果。"""

    beta: float
    alpha_annual: float
    r_squared: float
    t_alpha: float
    t_beta: float


def safe_float(value: Any, default: float = 0.0) -> float:
    """容错转 float（numpy 标量、None、字符串皆可）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """容错转 int（年份、计数等）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def joined_pair(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """按日期对齐（取交集），列名固定为 r（策略）/ b（基准）。"""
    return pd.concat(
        [returns.rename("r"), benchmark.rename("b")], axis=1, sort=True
    ).dropna()


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘，返回 (系数, 残差)。"""
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"回归求解失败: {exc}") from exc
    return coef, y - x @ coef


def newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int = NW_LAGS) -> np.ndarray:
    """Newey-West 异方差 + 自相关稳健标准误（Bartlett 核）。"""
    try:
        _, k = x.shape
        xtx_inv = np.linalg.pinv(x.T @ x)
        scores = x * resid[:, None]
        meat = scores.T @ scores
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1.0)
            gamma = (x[lag:] * resid[lag:, None]).T @ (x[:-lag] * resid[:-lag, None])
            meat += weight * (gamma + gamma.T)
        return np.diag(xtx_inv @ meat @ xtx_inv)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"Newey-West 标准误计算失败: {exc}") from exc


def regress(returns: pd.Series, benchmark: pd.Series) -> Regression:
    """日频 OLS：r = alpha + beta · b。"""
    joined = joined_pair(returns, benchmark)
    y = joined["r"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(joined)), b])
    coef, resid = ols(y, x)
    variance = newey_west_se(x, resid)
    ss_tot = safe_float(((y - y.mean()) ** 2).sum())
    ss_res = safe_float((resid**2).sum())
    return Regression(
        beta=safe_float(coef[1]),
        alpha_annual=safe_float(coef[0]) * TRADING_DAYS,
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        t_alpha=safe_float(coef[0] / np.sqrt(variance[0])),
        t_beta=safe_float(coef[1] / np.sqrt(variance[1])),
    )


def summarize_regression(name: str, reg: Regression) -> dict[str, float | str]:
    """回归结果 → 可序列化字典。"""
    return {
        "名称": name,
        "beta": reg.beta,
        "alpha": reg.alpha_annual,
        "t_alpha": reg.t_alpha,
        "t_beta": reg.t_beta,
        "r_squared": reg.r_squared,
    }


def decompose(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """累计对数收益 → alpha / beta / 残差三部分。"""
    joined = joined_pair(returns, benchmark)
    y = joined["r"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    coef, resid = ols(y, np.column_stack([np.ones(len(joined)), b]))
    total = safe_float(np.log1p(y).sum())
    return {
        "总对数收益": total,
        "总收益": safe_float(np.expm1(total)),
        "alpha 部分": safe_float(coef[0]) * len(joined),
        "beta 部分": safe_float(np.log1p(coef[1] * b).sum()),
        "残差部分": safe_float(np.log1p(resid).sum()),
    }


def _period_row(group: pd.DataFrame) -> dict[str, float]:
    """单个区间的读数行。"""
    reg = regress(cast(pd.Series, group["r"]), cast(pd.Series, group["b"]))
    nav = (1 + group["r"]).cumprod()
    std = safe_float(group["r"].std())
    return {
        "策略": safe_float(np.expm1(np.log1p(group["r"]).sum())),
        "基准": safe_float(np.expm1(np.log1p(group["b"]).sum())),
        "Sharpe": safe_float(group["r"].mean()) / std * np.sqrt(TRADING_DAYS)
        if std > 0
        else 0.0,
        "beta": reg.beta,
        "alpha": reg.alpha_annual,
        "t_alpha": reg.t_alpha,
        "R²": reg.r_squared,
        "回撤": safe_float((nav / nav.cummax() - 1).min()),
    }


def yearly_table(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """逐年读数（策略/基准收益、Sharpe、beta、alpha、R²、年内回撤）。"""
    joined = joined_pair(returns, benchmark)
    index = cast(pd.DatetimeIndex, joined.index)
    rows = {
        safe_int(year): _period_row(group) for year, group in joined.groupby(index.year)
    }
    return pd.DataFrame(rows).T


def sub_period_table(
    returns: pd.Series, benchmark: pd.Series, segments: int = 2
) -> pd.DataFrame:
    """等交易日分段读数（样本内稳健性：alpha 是否只靠某一段）。"""
    joined = joined_pair(returns, benchmark)
    bounds = np.linspace(0, len(joined), segments + 1).astype(int)
    rows: dict[str, dict[str, float]] = {}
    for start, end in zip(bounds[:-1], bounds[1:]):
        chunk = joined.iloc[start:end]
        label = f"{chunk.index[0]:%Y-%m}~{chunk.index[-1]:%Y-%m}"
        rows[label] = _period_row(chunk)
    return pd.DataFrame(rows).T


def concentration(returns: pd.Series) -> dict[str, float]:
    """月度贡献集中度：前/后 10% 月份各贡献多少累计对数收益。"""
    monthly = (1 + returns).resample("ME").prod() - 1
    contrib = np.log1p(monthly).sort_values(ascending=False)
    total = safe_float(contrib.sum())
    top_n = max(1, safe_int(round(len(contrib) * 0.10)))
    return {
        "月数": safe_float(len(contrib)),
        "总对数收益": total,
        "前10%月份贡献占比": safe_float(contrib.head(top_n).sum() / total)
        if total
        else 0.0,
        "最差10%月份拖累占比": safe_float(contrib.tail(top_n).sum() / total)
        if total
        else 0.0,
        "正月占比": safe_float((monthly > 0).mean()),
    }


def excess_stats(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """超额收益显著性（日频/月频）与跟踪误差、信息比率、月胜率。"""
    joined = joined_pair(returns, benchmark)
    excess = joined["r"] - joined["b"]
    te = safe_float(excess.std()) * np.sqrt(TRADING_DAYS)
    monthly = (1 + joined).resample("ME").prod() - 1
    monthly_excess = monthly["r"] - monthly["b"]
    spread = safe_float(monthly_excess.std())
    return {
        "年化跟踪误差": te,
        "信息比率": safe_float(excess.mean()) * TRADING_DAYS / te if te > 0 else 0.0,
        "月频超额 t 值": safe_float(monthly_excess.mean())
        / spread
        * np.sqrt(len(monthly_excess))
        if spread > 0
        else 0.0,
        "月胜率": safe_float((monthly["r"] > monthly["b"]).mean()),
    }
