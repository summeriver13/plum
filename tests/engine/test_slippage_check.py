"""成交价自检（幽灵滑点告警）测试。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.engine.backtest.analysis.slippage_check import (
    SLIPPAGE_ANOMALY_RATIO,
    scan_slippage_anomalies,
)


@dataclass
class _Trade:
    """测试替身：只暴露自检需要的两个字段。"""

    amount: Decimal
    slippage_cost: Decimal


def test_clean_log_reports_zero() -> None:
    """正常 bp 级滑点不报警。"""
    log = [
        _Trade(amount=Decimal("100000"), slippage_cost=Decimal("120")),  # 12bp
        _Trade(amount=Decimal("50000"), slippage_cost=Decimal("10")),  # 2bp
    ]
    result = scan_slippage_anomalies(log)
    assert result.count == 0
    assert result.cost == Decimal("0")
    assert result.is_clean is True
    assert result.warning() is None
    assert 0 < result.max_ratio < SLIPPAGE_ANOMALY_RATIO


def test_anomalous_trade_is_flagged() -> None:
    """滑点占成交额 66% 的成交必须被标记（实测幽灵滑点形态）。"""
    log = [
        _Trade(amount=Decimal("100000"), slippage_cost=Decimal("100")),
        _Trade(amount=Decimal("50643"), slippage_cost=Decimal("33431")),  # 66%
    ]
    result = scan_slippage_anomalies(log)
    assert result.count == 1
    assert result.cost == Decimal("33431")
    assert result.max_ratio > 0.6
    assert result.warning() is not None
    assert "异常滑点" in result.warning()


def test_zero_amount_trade_is_skipped() -> None:
    """成交额为 0 的行跳过，不产生除零或误报。"""
    result = scan_slippage_anomalies(
        [_Trade(amount=Decimal("0"), slippage_cost=Decimal("999"))]
    )
    assert result.count == 0
    assert result.max_ratio == 0.0


def test_threshold_boundary() -> None:
    """恰好等于阈值不算异常（严格大于才算）。"""
    exact = _Trade(amount=Decimal("1000"), slippage_cost=Decimal("20"))
    assert scan_slippage_anomalies([exact]).count == 0
    over = _Trade(amount=Decimal("1000"), slippage_cost=Decimal("20.01"))
    assert scan_slippage_anomalies([over]).count == 1


def test_analyzer_friction_audit_includes_anomaly() -> None:
    """摩擦成本审计报告携带自检字段。"""
    from app.engine.backtest.analysis.analyzer import Analyzer

    class _Cost:
        commission = Decimal("5")
        transfer_fee = Decimal("0")
        stamp_duty = Decimal("0")
        total = Decimal("5")

    @dataclass
    class _FullTrade:
        amount: Decimal
        slippage_cost: Decimal
        cost: _Cost

    report = Analyzer.friction_audit(
        [
            _FullTrade(
                amount=Decimal("10000"), slippage_cost=Decimal("300"), cost=_Cost()
            )
        ]
    )
    assert report.slippage_anomaly_count == 1
    assert report.slippage_anomaly_cost == Decimal("300")
    assert report.max_slippage_ratio > 0.02
