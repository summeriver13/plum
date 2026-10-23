"""买入持有分析器 — 长期投资绩效指标。

从 mvp/backtest/analysis/buy_and_hold_analyzer.py 移植，适配 Decimal 金额体系。
所有货币值使用 Decimal（元），移除 /10000 和 *10000 转换。

为"死拿"投资者提供专用指标：
1. 总收益 (DRIP): 价格收益 + 股息再投资
2. 回本周期: 通过股息收回初始投资的年数
3. 净资产增长: 内在价值复合增长率
4. 生存概率: 黑天鹅压力测试分析

.. warning::
   DRIP 计算使用硬编码平均股息率 3%（见 calculate_drip 方法）。
   需要从 FinancialStore 读取真实股息数据。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.engine.backtest.models import EquityPoint

from app.engine.backtest.analysis._numeric import equity_to_numpy, to_float


@dataclass
class DRIPReport:
    """股息再投资报告。

    模拟股息现金流按收盘价再投资。
    """

    initial_value: Decimal  # 初始投资金额 (Decimal 元)
    price_return: float  # 纯粹的价格涨幅
    dividend_total: Decimal  # 已收股息总额 (Decimal 元)
    drip_final_value: Decimal  # 再投资后的最终价值 (Decimal 元)
    drip_return: float  # 含再投资的总收益率
    annualized_drip: float  # 年化再投资收益率
    dividend_yield_avg: float  # 平均股息率
    shares_accumulated: float  # 再投资增加股数
    reinvestment_ratio: float  # 再投资股息 / 总股息


@dataclass
class PaybackReport:
    """回本周期报告。

    计算通过股息和资本回收需要多久收回初始投资。
    """

    initial_investment: Decimal  # 初始投资金额 (Decimal 元)
    payback_years: float | None  # 完全回本年数（None = 无法回本）
    dividends_received: Decimal  # 已收股息总额 (Decimal 元)
    capital_recovered: Decimal  # 卖出回收资本 (Decimal 元)
    total_recovered: Decimal  # 现金回收总额 (Decimal 元)
    remaining_investment: Decimal  # 期末剩余投资 (Decimal 元)
    recovery_ratio: float  # 回收总额 / 初始投资

    # 年度明细
    annual_dividends: dict[int, Decimal]  # 年份 -> 股息金额


@dataclass
class BookValueReport:
    """净资产增长报告。

    追踪独立于市场价格的内在价值增长。
    """

    initial_book_value: Decimal  # 初始每股净资产 (Decimal 元)
    final_book_value: Decimal  # 最终每股净资产
    bv_cagr: float  # 净资产年化增长率
    bv_total_growth: float  # 净资产总涨幅
    price_to_book_start: float  # 期初市净率
    price_to_book_end: float  # 期末市净率
    pb_contraction: float  # 市净率压缩幅度


@dataclass
class SurvivalScenario:
    """黑天鹅生存场景。"""

    name: str  # 例如："2008 金融危机"
    description: str
    price_shock: float  # 例如：-50% = 下跌 50%
    dividend_cut: float  # 例如：-30% = 股息削减 30%
    recovery_years: int  # 恢复年数
    survival_probability: float  # 0-1
    new_payback_years: float | None  # 场景后回本年数


@dataclass
class SurvivalReport:
    """生存概率报告。

    对历史黑天鹅事件进行压力测试。
    """

    scenarios: list[SurvivalScenario]
    worst_case: SurvivalScenario | None
    avg_survival_probability: float
    portfolio_survival_score: float  # 0-100

    # 黑天鹅历史事件
    historical_events: list[str]  # 例如：["2008", "2015", "2020"]
    has_survived_all: bool


@dataclass
class EntryThesis:
    """记录入场逻辑。"""

    symbol: str
    entry_date: datetime
    entry_price: Decimal  # Decimal 元
    entry_reason: str  # 例如："ICBC_DEEP_VALUE"、"PDD_EFFICIENCY"
    thesis_broken: bool = False
    break_date: datetime | None = None
    break_reason: str | None = None


@dataclass
class BuyAndHoldReport:
    """完整的买入持有分析报告。

    整合所有长期持有者关注的指标。
    """

    # 基本信息
    symbol: str
    period_start: datetime
    period_end: datetime
    holding_years: float

    # 核心指标
    drip_report: DRIPReport
    payback_report: PaybackReport
    book_value_report: BookValueReport
    survival_report: SurvivalReport

    # 入场逻辑追踪
    entry_thesis: EntryThesis | None = None

    # 传统指标（对比用）
    price_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 导出。"""
        return {
            "symbol": self.symbol,
            "period": f"{self.period_start.date()} ~ {self.period_end.date()}",
            "holding_years": self.holding_years,
            "drip_return": self.drip_report.drip_return,
            "annualized_drip": self.drip_report.annualized_drip,
            "payback_years": self.payback_report.payback_years,
            "bv_cagr": self.book_value_report.bv_cagr,
            "survival_score": self.survival_report.portfolio_survival_score,
            "price_return": self.price_return,
            "max_drawdown": self.max_drawdown,
        }


class BuyAndHoldAnalyzer:
    """长期持有策略分析器。"""

    # 历史黑天鹅事件
    BLACK_SWAN_EVENTS = {
        "2008": {
            "name": "2008 金融危机",
            "price_drop": 0.73,  # -73%
            "dividend_cut": 0.20,  # -20%
            "recovery_years": 5,
        },
        "2015": {
            "name": "2015 A股股灾",
            "price_drop": 0.52,  # -52%
            "dividend_cut": 0.15,
            "recovery_years": 2,
        },
        "2018": {
            "name": "2018 贸易战",
            "price_drop": 0.32,  # -32%
            "dividend_cut": 0.10,
            "recovery_years": 1,
        },
        "2020": {
            "name": "2020 COVID",
            "price_drop": 0.30,  # -30%
            "dividend_cut": 0.15,
            "recovery_years": 1,
        },
        "2022": {
            "name": "2022 熊市",
            "price_drop": 0.25,  # -25%
            "dividend_cut": 0.10,
            "recovery_years": 1,
        },
    }

    @staticmethod
    def calculate_drip(
        equity_curve: list[EquityPoint],
        positions_history: list[dict],  # [{date, qty, dividend_per_share}]
        initial_investment: Decimal,
    ) -> DRIPReport:
        """计算股息再投资（DRIP）收益。

        Args:
            equity_curve: 价格历史
            positions_history: 股息事件 [{date, qty, dividend_per_share}]
            initial_investment: 初始投资金额 (Decimal 元)

        Returns:
            含再投资分析的 DRIPReport
        """
        if not equity_curve:
            return DRIPReport(
                initial_value=initial_investment,
                price_return=0.0,
                dividend_total=Decimal("0"),
                drip_final_value=initial_investment,
                drip_return=0.0,
                annualized_drip=0.0,
                dividend_yield_avg=0.0,
                shares_accumulated=0.0,
                reinvestment_ratio=0.0,
            )

        first_price = to_float(equity_curve[0].total_value)
        last_price = to_float(equity_curve[-1].total_value)

        # 价格涨跌
        price_return = (last_price - first_price) / first_price if first_price else 0.0

        # 模拟股息再投资
        drip_value = to_float(initial_investment)
        total_dividends = 0.0
        reinvested = 0.0
        extra_shares = 0.0

        # 假设平均股息率（简化版）
        # 实际实现中应使用真实股息数据
        # TODO: 从 FinancialStore 读取真实股息记录替换此硬编码
        avg_dividend_yield = 0.03  # 3% 平均值
        warnings.warn(
            "BuyAndHoldAnalyzer: 使用硬编码平均股息率 3%。"
            "需从 FinancialStore 读取真实股息数据。",
            UserWarning,
            stacklevel=2,
        )

        trading_days = len(equity_curve)
        years = trading_days / 242

        # 简化版：假设持续股息再投资
        for _i, point in enumerate(equity_curve):
            daily_value = to_float(point.total_value)
            # 每日股息累计（简化版）
            daily_dividend = daily_value * avg_dividend_yield / 242

            if daily_dividend > 0:
                # 按当前价格再投资
                reinvested_shares = (
                    daily_dividend / to_float(point.total_value)
                    if to_float(point.total_value)
                    else 0
                )
                extra_shares += reinvested_shares
                drip_value += daily_dividend
                total_dividends += daily_dividend
                reinvested += daily_dividend

        # 计算最终值
        drip_return = (
            (drip_value - to_float(initial_investment)) / to_float(initial_investment)
            if initial_investment
            else 0.0
        )
        annualized_drip = (1 + drip_return) ** (1 / years) - 1 if years > 0 else 0.0

        return DRIPReport(
            initial_value=initial_investment,
            price_return=price_return,
            dividend_total=Decimal(str(total_dividends)),
            drip_final_value=Decimal(str(drip_value)),
            drip_return=drip_return,
            annualized_drip=annualized_drip,
            dividend_yield_avg=avg_dividend_yield,
            shares_accumulated=extra_shares,
            reinvestment_ratio=reinvested / total_dividends if total_dividends else 0.0,
        )

    @staticmethod
    def calculate_payback(
        initial_investment: Decimal,
        annual_dividends: list[Decimal],  # Year 1, 2, 3...
        exit_value: Decimal,
    ) -> PaybackReport:
        """计算股息回本周期。

        Args:
            initial_investment: 初始投资金额 (Decimal 元)
            annual_dividends: 年度股息收入 (Decimal 元)
            exit_value: 退出时回收价值 (Decimal 元)

        Returns:
            含周期分析的 PaybackReport
        """
        cumulative = Decimal("0")
        payback_years = None
        annual_breakdown: dict[int, Decimal] = {}

        for year, div in enumerate(annual_dividends, 1):
            cumulative += div
            annual_breakdown[year] = div
            if payback_years is None and cumulative >= initial_investment:
                # 插值计算不足一年
                prev_cumulative = cumulative - div
                remaining = initial_investment - prev_cumulative
                partial_year = to_float(remaining / div) if div > 0 else 0
                payback_years = year - 1 + partial_year

        total_divs = sum(annual_dividends) if annual_dividends else Decimal("0")
        total_recovered = total_divs + exit_value
        capital_recovered = min(total_divs, initial_investment)
        remaining = max(Decimal("0"), initial_investment - total_divs)

        return PaybackReport(
            initial_investment=initial_investment,
            payback_years=payback_years,
            dividends_received=total_divs,
            capital_recovered=capital_recovered,
            total_recovered=total_recovered,
            remaining_investment=max(Decimal("0"), remaining - exit_value),
            recovery_ratio=to_float(total_recovered / initial_investment)
            if initial_investment
            else 0.0,
            annual_dividends=annual_breakdown,
        )

    @staticmethod
    def calculate_survival(
        initial_value: Decimal,
        current_dividend_yield: float = 0.03,
        volatility: float = 0.20,
    ) -> SurvivalReport:
        """计算压力场景下的生存概率。

        Args:
            initial_value: 初始组合价值
            current_dividend_yield: 当前股息率
            volatility: 价格波动率

        Returns:
            含场景分析的 SurvivalReport
        """
        scenarios: list[SurvivalScenario] = []

        for _year_key, event in BuyAndHoldAnalyzer.BLACK_SWAN_EVENTS.items():
            price_shock = event["price_drop"]
            div_cut = event["dividend_cut"]
            recovery = event["recovery_years"]

            # 计算冲击后价值
            post_shock_value = to_float(initial_value) * (1 - price_shock)

            # 削减后的新股息收入
            new_yield = current_dividend_yield * (1 - div_cut)
            new_annual_div = post_shock_value * new_yield

            # 恢复时间（股息减少后）
            original_annual_div = to_float(initial_value) * current_dividend_yield
            div_shortfall_ratio = (
                new_annual_div / original_annual_div if original_annual_div else 1.0
            )

            # 生存概率基于：
            # 1. 损失幅度
            # 2. 股息恢复速度
            survival_prob = 1.0
            if price_shock > 0.5:
                survival_prob *= 0.7  # 严重下跌
            if div_shortfall_ratio < 0.5:
                survival_prob *= 0.8  # 股息大幅削减

            scenario = SurvivalScenario(
                name=event["name"],
                description=(
                    f"Price -{price_shock * 100:.0f}%, Dividend -{div_cut * 100:.0f}%"
                ),
                price_shock=price_shock,
                dividend_cut=div_cut,
                recovery_years=recovery,
                survival_probability=survival_prob,
                new_payback_years=recovery / div_shortfall_ratio
                if div_shortfall_ratio > 0
                else None,
            )
            scenarios.append(scenario)

        # 计算评分
        worst_case = min(scenarios, key=lambda s: s.survival_probability)
        avg_survival = sum(s.survival_probability for s in scenarios) / len(scenarios)
        survival_score = avg_survival * 100

        # 检查是否在所有事件中生存（假设长期持有者为是）
        has_survived = all(s.survival_probability > 0.5 for s in scenarios)

        return SurvivalReport(
            scenarios=scenarios,
            worst_case=worst_case,
            avg_survival_probability=avg_survival,
            portfolio_survival_score=survival_score,
            historical_events=list(BuyAndHoldAnalyzer.BLACK_SWAN_EVENTS.keys()),
            has_survived_all=has_survived,
        )

    @staticmethod
    def calculate_full_report(
        symbol: str,
        period_start: datetime,
        period_end: datetime,
        initial_investment: Decimal,
        initial_price: Decimal,
        final_price: Decimal,
        equity_curve: list[EquityPoint],
        positions_history: list[dict] | None = None,
    ) -> BuyAndHoldReport:
        """计算完整的买入持有分析报告。

        Args:
            symbol: 股票代码
            period_start: 开始日期
            period_end: 结束日期
            initial_investment: 初始投资 (Decimal 元)
            initial_price: 起始价格 (Decimal 元)
            final_price: 结束价格 (Decimal 元)
            equity_curve: 价格/价值历史
            positions_history: 股息事件（可选）

        Returns:
            完整的 BuyAndHoldReport
        """
        holding_years = (period_end - period_start).days / 365.25

        # 计算 DRIP
        drip_report = BuyAndHoldAnalyzer.calculate_drip(
            equity_curve,
            positions_history or [],
            initial_investment,
        )

        # 计算回本周期（从 DRIP 估算）
        years_count = int(holding_years) + 1
        estimated_annual_divs = (
            [drip_report.dividend_total // years_count for _ in range(years_count)]
            if years_count > 0
            else []
        )
        payback_report = BuyAndHoldAnalyzer.calculate_payback(
            initial_investment,
            estimated_annual_divs,
            final_price,
        )

        # 净资产（简化版 - 用价格代理）
        bv_cagr = (
            (
                (to_float(final_price) / to_float(initial_price)) ** (1 / holding_years)
                - 1
            )
            if holding_years > 0 and initial_price
            else 0.0
        )
        book_value_report = BookValueReport(
            initial_book_value=initial_price,
            final_book_value=final_price,
            bv_cagr=bv_cagr,
            bv_total_growth=(to_float(final_price) - to_float(initial_price))
            / to_float(initial_price)
            if initial_price
            else 0.0,
            price_to_book_start=1.0,
            price_to_book_end=1.0,
            pb_contraction=0.0,
        )

        # 生存分析
        current_yield = drip_report.dividend_yield_avg
        values_arr = equity_to_numpy(equity_curve)
        vol = (
            float(np.std(values_arr)) / float(np.mean(values_arr))
            if len(equity_curve) > 1
            else 0.2
        )
        survival_report = BuyAndHoldAnalyzer.calculate_survival(
            initial_investment,
            current_yield,
            vol,
        )

        # 传统指标
        price_return = (
            (to_float(final_price) - to_float(initial_price)) / to_float(initial_price)
            if initial_price
            else 0.0
        )

        # 最大回撤
        max_dd = 0.0
        peak = to_float(initial_investment)
        for point in equity_curve:
            pv = to_float(point.total_value)
            peak = max(peak, pv)
            dd = (pv - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)

        return BuyAndHoldReport(
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            holding_years=holding_years,
            drip_report=drip_report,
            payback_report=payback_report,
            book_value_report=book_value_report,
            survival_report=survival_report,
            price_return=price_return,
            max_drawdown=abs(max_dd),
            volatility=vol,
        )

    @staticmethod
    def print_report(report: BuyAndHoldReport) -> None:
        """终端打印（已迁移至 _printer.py）。"""
        from ._printer import BuyAndHoldPrinter

        BuyAndHoldPrinter.print_report(report)
