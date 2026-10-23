"""终端打印报告 — 所有 print_report 方法集中于此。

将渲染逻辑与分析逻辑分离。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .analyzer import PerformanceReport
    from .buy_and_hold_analyzer import BuyAndHoldReport
    from .stability import (
        GridSweepResult,
        ParameterPlateauResult,
        SharpeGradientResult,
        WalkForwardResult,
    )

from app.engine.backtest.analysis._numeric import to_float

# ── 视觉宽度工具 ──────────────────────────────────────

_WIDTH_PATTERN = re.compile(
    r"[\u1100-\u115f\u2329-\u232a\u2e80-\u303e\u3040-\ua4cf"
    r"\uac00-\ud7a3\uf900-\ufaff\ufe10-\ufe1f\ufe30-\ufe6f"
    r"\uff00-\uff60\uffe0-\uffe6\u3000-\u303f\uff01-\uffee]"
)


def _visual_width(text: str) -> int:
    """Return the terminal display width of text (Chinese = 2, ASCII = 1)."""
    return sum(2 if _WIDTH_PATTERN.match(ch) else 1 for ch in text)


def _ljust(text: str, width: int) -> str:
    """Left-justified within visual width (Chinese chars count as 2)."""
    vw = _visual_width(text)
    return text + " " * max(0, width - vw)


def _rjust(text: str, width: int) -> str:
    """Right-justified within visual width (Chinese chars count as 2)."""
    vw = _visual_width(text)
    return " " * max(0, width - vw) + text


# ── Analyzer 打印 ─────────────────────────────────────


class AnalyzerPrinter:
    """PerformanceReport 的终端格式化输出。"""

    @staticmethod
    def print_report(report: PerformanceReport) -> None:
        AnalyzerPrinter._print_header()
        AnalyzerPrinter._print_basic_info(report)
        AnalyzerPrinter._print_drawdown(report)
        AnalyzerPrinter._print_risk_metrics(report)
        AnalyzerPrinter._print_sortino(report)
        AnalyzerPrinter._print_benchmark(report)
        AnalyzerPrinter._print_general(report)
        AnalyzerPrinter._print_fifo(report)
        AnalyzerPrinter._print_friction(report)
        AnalyzerPrinter._print_trade_summary(report)
        print("\n" + "=" * 70)

    @staticmethod
    def _print_header() -> None:
        print("=" * 70)
        print("                    绩 效 报 告")
        print("=" * 70)

    @staticmethod
    def _print_basic_info(report: PerformanceReport) -> None:
        _v0 = _rjust(f"{to_float(report.initial_value):,.2f}", 18)
        _v1 = _rjust(f"{to_float(report.final_value):,.2f}", 18)
        print(f"\n{_ljust('初始资金', 25)} {_v0}")
        print(f"{_ljust('最终资金', 25)} {_v1}")
        ret_pct = report.total_return * 100
        ann_pct = report.annualized_return * 100
        print(f"\n{_ljust('总收益率', 25)} {_rjust(f'{ret_pct:.2f}%', 18)}")
        print(f"{_ljust('年化收益率', 25)} {_rjust(f'{ann_pct:.2f}%', 18)}")

    @staticmethod
    def _print_drawdown(report: PerformanceReport) -> None:
        mdd_pct = report.max_drawdown * 100
        print(f"\n{_ljust('最大回撤', 25)} {_rjust(f'{mdd_pct:.2f}%', 18)}")
        if report.max_drawdown_start:
            _v = _rjust(str(report.max_drawdown_start.date()), 18)
            print(f"{_ljust('最大回撤开始', 25)} {_v}")
        if report.max_drawdown_date:
            _v = _rjust(str(report.max_drawdown_date.date()), 18)
            print(f"{_ljust('最大回撤结束', 25)} {_v}")
        if report.longest_drawdown:
            ld = report.longest_drawdown
            ld_pct = ld.max_drawdown * 100
            _v1 = _rjust(str(ld.duration_days), 17)
            _v2 = _rjust(f"{ld_pct:.2f}%", 18)
            print(f"{_ljust('最长回撤', 25)} {_v1} 天  (最大: {_v2})")
            if ld.start and ld.end:
                _v = _rjust(str(ld.start.date()), 18)
                print(f"{_ljust('最长回撤区间', 25)} {_v} ~ {ld.end.date()}")
        dd_ratio_pct = report.drawdown_time_ratio * 100
        print(f"{_ljust('水下时间占比', 25)} {_rjust(f'{dd_ratio_pct:.1f}%', 18)}")
        pos_ratio_pct = report.position_time_ratio * 100
        print(f"{_ljust('持仓时间占比', 25)} {_rjust(f'{pos_ratio_pct:.1f}%', 18)}")

    @staticmethod
    def _print_risk_metrics(report: PerformanceReport) -> None:
        vol_pct = report.volatility * 100
        print(f"\n{_ljust('年化波动率', 25)} {_rjust(f'{vol_pct:.2f}%', 18)}")
        print(f"{_ljust('夏普比率', 25)} {_rjust(f'{report.sharpe_ratio:.2f}', 18)}")
        print(f"{_ljust('卡尔马比率', 25)} {_rjust(f'{report.calmar_ratio:.2f}', 18)}")

    @staticmethod
    def _print_sortino(report: PerformanceReport) -> None:
        if not report.sortino_report:
            return
        sr = report.sortino_report
        print(f"{_ljust('索提诺比率', 25)} {_rjust(f'{sr.sortino_ratio:.2f}', 18)}")
        print(f"{_ljust('尾部比率', 25)} {_rjust(f'{sr.tail_ratio:.2f}', 18)}")
        print(f"{_ljust('收益偏度', 25)} {_rjust(f'{sr.skewness:.3f}', 18)}")
        print(f"{_ljust('收益峰度', 25)} {_rjust(f'{sr.kurtosis:.3f}', 18)}")
        _v = _rjust(str(sr.max_drawdown_duration), 18)
        print(f"{_ljust('最大回撤持续期', 25)} {_v} 天")

    @staticmethod
    def _print_benchmark(report: PerformanceReport) -> None:
        if not report.benchmark_report:
            return
        br = report.benchmark_report
        print("\n" + "-" * 70)
        print("                    基 准 对 比")
        print("-" * 70)
        _v = _rjust(f"{br.benchmark_total_return * 100:.2f}%", 18)
        print(f"{_ljust('基准总收益率', 25)} {_v}")
        bm_ann_pct = br.benchmark_annualized_return * 100
        print(f"{_ljust('基准年化收益率', 25)} {_rjust(f'{bm_ann_pct:.2f}%', 18)}")
        _v = _rjust(f"{br.active_return * 100:.2f}%", 18)
        print(f"{_ljust('主动收益', 25)} {_v}")
        print(f"{_ljust('阿尔法', 25)} {_rjust(f'{br.alpha * 100:.2f}%', 18)}")
        print(f"{_ljust('贝塔', 25)} {_rjust(f'{br.beta:.3f}', 18)}")
        _v = _rjust(f"{br.tracking_error * 100:.2f}%", 18)
        print(f"{_ljust('跟踪误差', 25)} {_v}")
        print(f"{_ljust('信息比率', 25)} {_rjust(f'{br.information_ratio:.3f}', 18)}")

    @staticmethod
    def _print_general(report: PerformanceReport) -> None:
        print(f"\n{_ljust('交易日', 25)} {_rjust(str(report.trading_days), 18)}")
        _v = _rjust(f"{report.risk_free_rate * 100:.1f}%", 18)
        print(f"{_ljust('无风险利率', 25)} {_v}")

    @staticmethod
    def _print_fifo(report: PerformanceReport) -> None:
        if not report.fifo_report:
            return
        fr = report.fifo_report
        print("\n" + "-" * 70)
        print("                    FIFO 盈 亏 归 因")
        print("-" * 70)
        _pnl = to_float(fr.total_realized_pnl)
        print(f"{_ljust('已实现盈亏', 25)} {_rjust(f'{_pnl:,.2f}', 18)}")
        print(f"{_ljust('FIFO 胜率', 25)} {_rjust(f'{fr.win_rate * 100:.2f}%', 18)}")
        _v = _rjust(f"{fr.avg_holding_days:.1f}", 18)
        print(f"{_ljust('平均持仓天数', 25)} {_v}")

    @staticmethod
    def _print_friction(report: PerformanceReport) -> None:
        if not report.friction_report:
            return
        fr = report.friction_report
        print("\n" + "-" * 70)
        print("                    交 易 摩 擦 成 本")
        print("-" * 70)
        _comm = to_float(fr.total_commission)
        _stamp = to_float(fr.total_stamp_duty)
        _slip = to_float(fr.total_slippage_cost)
        _fric = to_float(fr.total_friction)
        _avg_fric = to_float(fr.avg_friction_per_trade)
        print(f"{_ljust('总佣金', 25)} {_rjust(f'{_comm:,.2f}', 18)}")
        print(f"{_ljust('总印花税', 25)} {_rjust(f'{_stamp:,.2f}', 18)}")
        print(f"{_ljust('总滑点成本', 25)} {_rjust(f'{_slip:,.2f}', 18)}")
        print(f"{_ljust('总摩擦成本', 25)} {_rjust(f'{_fric:,.2f}', 18)}")
        if fr.slippage_anomaly_count:
            print(
                f"  ⚠️ 异常滑点 {fr.slippage_anomaly_count} 笔，"
                f"合计 {to_float(fr.slippage_anomaly_cost):,.0f} 元"
                f"（单笔 > 成交额 2%，疑似数据/裁剪问题）"
            )
        _v = _rjust(f"{fr.friction_ratio * 100:.4f}%", 18)
        print(f"{_ljust('摩擦比率', 25)} {_v}")
        print(f"{_ljust('平均每笔摩擦', 25)} {_rjust(f'{_avg_fric:,.2f}', 18)}")

    @staticmethod
    def _print_trade_summary(report: PerformanceReport) -> None:
        if not report.trade_summary:
            return
        ts = report.trade_summary
        print("\n" + "-" * 70)
        print("                    交 易 分 析")
        print("-" * 70)
        print(f"{_ljust('总交易次数', 25)} {_rjust(str(ts.total_trades), 18)}")
        print(f"{_ljust('盈利交易', 25)} {_rjust(str(ts.winning_trades), 18)}")
        print(f"{_ljust('亏损交易', 25)} {_rjust(str(ts.losing_trades), 18)}")
        print(f"{_ljust('胜率', 25)} {_rjust(f'{ts.win_rate * 100:.2f}%', 18)}")
        _avg_profit = ts.avg_profit
        _avg_loss = ts.avg_loss
        print(f"{_ljust('平均盈利', 25)} {_rjust(f'{_avg_profit:,.2f}', 18)}")
        print(f"{_ljust('平均亏损', 25)} {_rjust(f'{_avg_loss:,.2f}', 18)}")
        print(f"{_ljust('盈亏比', 25)} {_rjust(f'{ts.profit_loss_ratio:.2f}', 18)}")
        _v = _rjust(f"{ts.avg_holding_days:.1f}", 18)
        print(f"{_ljust('平均持仓天数', 25)} {_v}")


# ── 参数扫描打印 ──────────────────────────────────────


class SweepPrinter:
    @staticmethod
    def print_sweep_report(result) -> None:
        print("=" * 80)
        print("                  参数敏感性分析")
        print(f"                  ({result.parameter_name})")
        print("=" * 80)
        print(
            f"\n{'参数':<15} {'收益率':<12} {'夏普':<10} "
            f"{'最大回撤':<12} {'摩擦率':<12}"
        )
        print("-" * 80)
        for i, param in enumerate(result.parameter_values):
            print(
                f"{param:<15.2f} "
                f"{result.returns[i] * 100:>10.2f}% "
                f"{result.sharpe_ratios[i]:>9.2f} "
                f"{result.max_drawdowns[i] * 100:>10.2f}% "
                f"{result.friction_ratios[i] * 100:>10.4f}%"
            )
        print("-" * 80)


# ── 蒙特卡洛打印 ──────────────────────────────────────


class MonteCarloPrinter:
    @staticmethod
    def print_monte_carlo_report(result) -> None:
        print("=" * 80)
        print("                  蒙 特 卡 洛 模 拟")
        print("=" * 80)
        print(f"\n{'模拟次数':<25} {result.num_simulations:>15,}")
        print(f"{'初始值':<25} {to_float(result.initial_value):>15,.2f}")
        print(f"{'平均最终值':<25} {to_float(result.avg_final_value):>15,.2f}")
        print(f"{'最小最终值':<25} {to_float(min(result.final_values)):>15,.2f}")
        print(f"{'最大最终值':<25} {to_float(max(result.final_values)):>15,.2f}")
        avg_ret = sum(result.returns) / len(result.returns) * 100
        print(f"\n{'平均收益率':<25} {avg_ret:>14.2f}%")
        print(f"{'最差收益率':<25} {min(result.returns) * 100:>14.2f}%")
        print(f"{'最佳收益率':<25} {max(result.returns) * 100:>14.2f}%")
        print(f"\n{'最差回撤':<25} {result.worst_case_dd * 100:>14.2f}%")
        print(f"{'95分位回撤':<25} {result.percentile_95_dd * 100:>14.2f}%")
        avg_mdd = sum(result.max_drawdowns) / len(result.max_drawdowns) * 100
        print(f"{'平均回撤':<25} {avg_mdd:>14.2f}%")
        print(f"\n{'VaR (95%)':<25} {result.var_95 * 100:>14.2f}%")
        print(f"{'CVaR (95%)':<25} {result.cvar_95 * 100:>14.2f}%")
        print(f"{'破产概率':<25} {result.probability_of_ruin * 100:>14.2f}%")
        print("=" * 80)


# ── 买入持有打印 ──────────────────────────────────────


class BuyAndHoldPrinter:
    @staticmethod
    def print_report(report: BuyAndHoldReport) -> None:
        print("=" * 70)
        print("              买 入 持 有 分 析")
        print("=" * 70)
        print(f"\n{'股票代码':<20} {report.symbol}")
        print(f"{'周期':<20} {report.period_start.date()} ~ {report.period_end.date()}")
        print(f"{'持有年数':<20} {report.holding_years:.1f}")
        drip = report.drip_report
        print("\n" + "-" * 70)
        print("              股息再投资 (DRIP)")
        print("-" * 70)
        print(f"{'初始投资':<25} {to_float(drip.initial_value):>15,.2f}")
        print(f"{'价格收益':<25} {drip.price_return * 100:>15.2f}%")
        print(f"{'总股息':<25} {to_float(drip.dividend_total):>15,.2f}")
        print(f"{'DRIP 最终价值':<25} {to_float(drip.drip_final_value):>15,.2f}")
        print(f"{'DRIP 收益率':<25} {drip.drip_return * 100:>15.2f}%")
        print(f"{'年化 DRIP':<25} {drip.annualized_drip * 100:>15.2f}%")
        print(f"{'平均股息率':<25} {drip.dividend_yield_avg * 100:>15.2f}%")
        pb = report.payback_report
        print("\n" + "-" * 70)
        print("              回 本 分 析")
        print("-" * 70)
        payback_str = f"{pb.payback_years:.1f} 年" if pb.payback_years else "未回本"
        print(f"{'回本周期':<25} {payback_str:>15}")
        print(f"{'已收股息':<25} {to_float(pb.dividends_received):>15,.2f}")
        print(f"{'资本回收':<25} {to_float(pb.capital_recovered):>15,.2f}")
        print(f"{'总回收':<25} {to_float(pb.total_recovered):>15,.2f}")
        print(f"{'回收比率':<25} {pb.recovery_ratio * 100:>15.1f}%")
        bv = report.book_value_report
        print("\n" + "-" * 70)
        print("              净资产 (内在价值)")
        print("-" * 70)
        print(f"{'净资产 CAGR':<25} {bv.bv_cagr * 100:>15.2f}%")
        print(f"{'净资产总增长':<25} {bv.bv_total_growth * 100:>15.2f}%")
        sr = report.survival_report
        print("\n" + "-" * 70)
        print("              生存概率 (压力测试)")
        print("-" * 70)
        print(f"{'生存评分':<25} {sr.portfolio_survival_score:>15.1f}/100")
        print(f"{'平均生存概率':<25} {sr.avg_survival_probability * 100:>15.1f}%")
        print("\n  黑天鹅事件:")
        for s in sr.scenarios:
            prob = s.survival_probability * 100
            symbol = "✅" if prob >= 80 else "⚠️" if prob >= 50 else "❌"
            print(f"    {symbol} {s.name}: {prob:.0f}% 生存")
        if sr.has_survived_all:
            print("\n  ✅ 组合在所有历史黑天鹅事件中存活")
        print("\n" + "=" * 70)


# ── 稳定性分析打印 ────────────────────────────────────


class StabilityPrinter:
    @staticmethod
    def walk_forward(results: list[WalkForwardResult]) -> None:
        print("=" * 80)
        print("              WALK-FORWARD ANALYSIS")
        print("=" * 80)
        print(
            f"\n{'训练期':<22} {'测试期':<22} "
            f"{'IS Sharpe':>9} {'OOS Sharpe':>10} "
            f"{'Sharpe 退化':>11} {'IS Ret%':>8} {'OOS Ret%':>9}"
        )
        print("-" * 80)
        for r in results:
            train_period = f"{r.train_start.date()} ~ {r.train_end.date()}"
            test_period = f"{r.test_start.date()} ~ {r.test_end.date()}"
            print(
                f"{train_period:<22} {test_period:<22} "
                f"{r.in_sample.sharpe_ratio:>9.2f} "
                f"{r.out_of_sample.sharpe_ratio:>10.2f} "
                f"{r.sharpe_degradation:>10.1%} "
                f"{r.in_sample.annualized_return:>7.1%} "
                f"{r.out_of_sample.annualized_return:>8.1%}"
            )
        if len(results) >= 2:
            avg_deg = np.mean([r.sharpe_degradation for r in results])
            oos_sharpes = [r.out_of_sample.sharpe_ratio for r in results]
            print()
            print(f"平均夏普退化: {avg_deg:+.1%}")
            print(f"平均 OOS Sharpe: {np.mean(oos_sharpes):.2f}")
            print(f"最小 OOS Sharpe:   {min(oos_sharpes):.2f}")
            print(f"最大 OOS Sharpe:   {max(oos_sharpes):.2f}")
        print("-" * 80)

    @staticmethod
    def grid_sweep(result: GridSweepResult) -> None:
        print("=" * 80)
        print("              参 数 网 格 扫 描")
        print("=" * 80)
        print(f"\n网格大小: {result.grid_size}")
        print(f"最佳参数: {result.best_params}")
        print(f"最佳 Sharpe: {result.best_metrics.sharpe_ratio:.2f}")
        print(f"最佳年化收益: {result.best_metrics.annualized_return:.2%}")
        print(f"稳定性评分: {result.stability_score:.2f}")
        print()
        print("-" * 80)
        print(
            f"{'排名':<5} {'Sharpe':>7} {'年化收益':>8} {'最大回撤':>8} {'交易数':>7}"
        )
        print("-" * 80)
        for pt in result.all_points[:15]:
            print(
                f"{pt.rank:<5} "
                f"{pt.metrics.sharpe_ratio:>7.2f} "
                f"{pt.metrics.annualized_return:>7.1%} "
                f"{pt.metrics.max_drawdown:>7.1%} "
                f"{pt.metrics.num_trades:>7}"
            )
        if len(result.all_points) > 15:
            print(f"... 还有 {len(result.all_points) - 15} 个")
        print("-" * 80)

    @staticmethod
    def sharpe_gradient(result: SharpeGradientResult) -> None:
        print()
        print("=" * 80)
        print("              SHARPE 梯 度 分 析")
        print("=" * 80)
        print(
            f"\n参数: {result.parameter_name}"
            f"  |  最佳: {result.best_param} (Sharpe={result.best_sharpe:.2f})"
            f"  |  平台比率: {result.plateau_ratio:.1%}"
        )
        print()
        header = f"{result.parameter_name:<12} {'Sharpe':>8} {'梯度':>10} {'状态':>8}"
        print(header)
        print("-" * len(header))
        for i, (pv, sh) in enumerate(
            zip(result.parameter_values, result.sharpe_values)
        ):
            grad = result.gradient_values[i] if i < len(result.gradient_values) else 0.0
            status = "⚠️ 不稳定" if i in result.unstable_indices else "✅ 稳定"
            print(f"{pv:<12.4f} {sh:>8.2f} {grad:>10.3f} {status:>8}")
        print("-" * len(header))

    @staticmethod
    def plateau(result: ParameterPlateauResult) -> None:
        print()
        print("=" * 80)
        print("              参 数 平 台 图")
        print("=" * 80)
        print(
            f"\n参数: {result.parameter_name}"
            f"  |  最佳: {result.best_param} (Sharpe={result.best_sharpe:.2f})"
        )
        print()
        for region in result.regions:
            symbol = (
                "🟢"
                if region.region_type == "STABLE"
                else "🟡"
                if region.region_type == "MODERATE"
                else "🔴"
            )
            print(
                f"  {symbol} [{region.param_start:.2f} ~ {region.param_end:.2f}]"
                f"  {region.region_type}"
                f"  Sharpe: {region.min_sharpe:.2f} ~ {region.max_sharpe:.2f}"
            )
        if result.danger_zones:
            print()
            print("🔴 危险区域 (Sharpe 低于底线):")
            for dz in result.danger_zones:
                print(f"  [{dz[0]:.2f} ~ {dz[1]:.2f}]")
        if result.stable_params:
            print()
            print(f"🟢 稳定参数 ({len(result.stable_params)} 个值):")
            shown = sorted(result.stable_params)[:10]
            suffix = "..." if len(result.stable_params) > 10 else ""
            print(f"  {shown}{suffix}")
        print("-" * 80)
