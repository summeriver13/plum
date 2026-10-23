"""因子中性化 — 行业哑变量 + 市值对数回归取残差。

使用线性回归（numpy 闭式解 OLS）对目标因子进行市值/行业等风格剥离，
提取残差作为纯净 alpha。无外部 ML 库依赖。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_residual(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """闭式解 OLS 回归，返回残差 y - Xβ。

    β = (XᵀX)⁻¹Xᵀy
    residual = y - Xβ
    """
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta
        return residual.ravel()
    except np.linalg.LinAlgError:
        return np.zeros_like(y).ravel()


def _factors_linear_regression(
    df: pd.DataFrame,
    factor: str,
    neutralize_list: list[str],
    industry: str | None = None,
) -> pd.DataFrame:
    """单日截面线性回归 — 内部函数。

    Args:
        df: 单个交易日的数据切片
        factor: 被中性化的目标因子列名
        neutralize_list: 作为自变量 X 的列名列表
        industry: 行业列名（可选）

    Returns:
        附加了 '{factor}_中性' 列（残差）的 DataFrame
    """
    train_col = list(neutralize_list)

    if industry:
        ind_list = df[industry].unique().tolist()
        industry_cols = [c for c in df.columns if c.startswith("所属行业_")]
        for col in industry_cols:
            if col not in train_col:
                if col.replace("所属行业_", "") in ind_list:
                    train_col.append(col)

    train = df[train_col].values.astype(np.float64)
    label = df[[factor]].values.astype(np.float64)

    residual = _ols_residual(train, label)
    df[factor + "_中性"] = residual

    return df


def factor_neutralization(
    data: pd.DataFrame,
    factor: str,
    neutralize_list: list[str],
    industry: str | None = None,
) -> pd.DataFrame:
    """因子中性化 — 行业哑变量 + 市值对数回归取残差。

    对指定因子做横截面线性回归，剥离 neutralize_list 中列的影响，
    将残差作为纯净因子值存入 '{factor}_中性' 列。

    算法:
        对每个交易日 t:
            X = neutralize_list 各列（+ 行业哑变量如果指定）
            y = factor 列
            β = least_squares(X, y)
            '{factor}_中性' = y - Xβ

    Args:
        data: 包含 交易日期、股票代码、factor、neutralize_list 各列的 DataFrame
        factor: 被中性化的目标因子列名
        neutralize_list: 剥离变量列名，如 ['总市值']
        industry: 行业列名（可选），会生成哑变量加入回归

    Returns:
        原始 data 附加 '{factor}_中性' 列
    """
    copy_cols = ["交易日期", "股票代码", factor] + neutralize_list
    if industry:
        copy_cols.append(industry)

    df = data[copy_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=copy_cols, how="any")

    if industry:
        ind_dummies = pd.get_dummies(
            df[industry], prefix="所属行业", prefix_sep="_", dummy_na=False
        )
        if "所属行业_综合" in ind_dummies.columns:
            ind_dummies.drop(columns=["所属行业_综合"], inplace=True)
    else:
        ind_dummies = pd.DataFrame()

    df = pd.concat([df, ind_dummies], axis=1)

    # 保存原始索引以便后续合并
    original_index = df.index.copy()

    df = df.groupby(["交易日期"], group_keys=False, observed=True).apply(
        _factors_linear_regression,
        factor=factor,
        neutralize_list=neutralize_list,
        industry=industry,
    )

    # groupby.apply 可能丢失分组键列，从原始索引恢复
    df.index = original_index
    if "交易日期" not in df.columns:
        df["交易日期"] = data.loc[df.index, "交易日期"]
    if "股票代码" not in df.columns:
        df["股票代码"] = data.loc[df.index, "股票代码"]

    merge_cols = ["交易日期", "股票代码", factor + "_中性"]
    data = pd.merge(data, df[merge_cols], on=["交易日期", "股票代码"], how="left")

    return data
