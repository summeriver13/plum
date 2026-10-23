"""ETF 叠加检验的呈现层 —— 表格格式化与打印（供 analyze_etf_overlay.py 使用）。"""

from __future__ import annotations

from typing import Any


def fmt_pct(value: float) -> str:
    """百分比格式化（NaN 显示 —）。"""
    return "—" if value is None or value != value else f"{value:+.2%}"


def report_correlation(
    corr: dict[str, Any], micro_label: str, sleeve_label: str
) -> None:
    """打印相关性与 beta。"""
    print(f"\n== 日收益相关性（微盘 {micro_label} vs {sleeve_label}） ==")
    print(
        f"全样本 corr = {corr['overall']:.3f}"
        f" | ETF 对微盘 beta = {corr['beta_etf_on_micro']:.3f}"
        f" | 样本 {corr['days']} 天"
    )
    yearly = " | ".join(
        f"{year}: {value:.3f}" for year, value in corr["yearly"].items()
    )
    print(f"分年 corr: {yearly}")


def report_combos(
    rows: list[dict[str, Any]], title: str = "组合模拟（日度再平衡）"
) -> None:
    """打印组合权重扫描表。"""
    print(f"\n== {title} ==")
    print(
        f"{'微盘权重':>8} {'年化':>9} {'波动':>8} {'Sharpe':>7} {'回撤':>9} {'Calmar':>7}"
    )
    for row in rows:
        print(
            f"{row['micro_weight']:>8.0%} {fmt_pct(row['annual_return']):>9}"
            f" {fmt_pct(row['annual_vol']):>8} {row['sharpe']:>7.2f}"
            f" {fmt_pct(row['max_drawdown']):>9} {row['calmar']:>7.2f}"
        )


def report_overlay(
    baseline: dict[str, float],
    rows: list[dict[str, Any]],
    cash_share: float,
    title: str = "现金停泊替换（反事实）",
) -> None:
    """打印停泊替换表。"""
    print(f"\n== {title} ==")
    print(
        f"基准（现金 1.5%）: 年化 {fmt_pct(baseline['annual_return'])}"
        f" | Sharpe {baseline['sharpe']:.2f}"
        f" | 回撤 {fmt_pct(baseline['max_drawdown'])}"
        f" | Calmar {baseline['calmar']:.2f}"
        f" | 平均现金占比 {cash_share:.1%}"
    )
    for row in rows:
        print(
            f"{row['symbol']}（ETF 年化 {fmt_pct(row['park_annual_return'])}）:"
            f" 年化 {fmt_pct(row['annual_return'])}"
            f" | Sharpe {row['sharpe']:.2f}"
            f" | 回撤 {fmt_pct(row['max_drawdown'])}"
            f" | Calmar {row['calmar']:.2f}"
        )
