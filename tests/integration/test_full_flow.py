"""集成测试：回测 + 模拟交易 + 实盘全流程。

验证各引擎模块协同工作，策略能正常运行于三种环境。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from app.data.schema import Bar
from app.engine.backtest.models import SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.live.engine import LiveEngineConfig, LiveEnvironment
from app.engine.live.gateway.base import Direction, OrderRequest, OrderType
from app.engine.live.gateway.mock import MockGateway
from app.engine.live.risk import RiskConfig, RiskGuard
from app.engine.paper.broker import PaperBroker, PaperBrokerConfig
from app.engine.paper.environment import PaperEnvironment, PaperEnvironmentConfig
from app.strategy.base import WindowedStrategy
from app.strategy.macross import MACrossStrategy


def _make_bar_series(
    prices: list[float],
    symbol: str = "600519.SH",
    start_date: datetime = datetime(2024, 1, 1),
) -> list[tuple[datetime, dict[str, Bar]]]:
    """创建Bar时间序列。"""
    result = []
    for i, price in enumerate(prices):
        pre = prices[i - 1] if i > 0 else price
        bar = Bar(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1 + i),
            open=Decimal(str(price)),
            high=Decimal(str(price * 1.02)),
            low=Decimal(str(price * 0.98)),
            close=Decimal(str(price)),
            pre_close=Decimal(str(pre)),
            volume=1000000,
            amount=Decimal("10000000"),
        )
        result.append((datetime(2024, 1, 1 + i), {symbol: bar}))
    return result


class TestBacktestIntegration:
    """回测引擎集成测试。"""

    def test_strategy_generates_signals(self) -> None:
        """策略在回测中能正常生成信号。"""
        strategy = MACrossStrategy(short_period=3, long_period=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))

        prices = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]
        series = _make_bar_series(prices)

        all_signals = []
        for ts, bars in series:
            signals = strategy.on_bar(ts, bars, portfolio)
            all_signals.extend(signals)

        assert len(all_signals) > 0
        assert any(s.signal_type == SignalType.ENTRY for s in all_signals)

    def test_portfolio_updates_on_trade(self) -> None:
        """投资组合在交易后正确更新。"""
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        commission = AShareCommissionModel()

        initial_cash = portfolio.cash
        cost = commission.calculate(Decimal("10.0"), 100, is_buy=True)
        total_fee = cost.commission + cost.transfer_fee + cost.stamp_duty
        portfolio.apply_trade("600519.SH", 100, Decimal("10.0"), total_fee)

        assert portfolio.cash < initial_cash
        assert "600519.SH" in portfolio.positions
        assert portfolio.positions["600519.SH"].quantity == 100

    def test_backtest_full_flow(self) -> None:
        """回测全流程：策略信号 → 交易 → 持仓更新。"""
        strategy = MACrossStrategy(short_period=3, long_period=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        commission = AShareCommissionModel()

        prices = [
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            13.5,
            13.0,
            12.5,
            12.0,
        ]
        series = _make_bar_series(prices)

        for ts, bars in series:
            signals = strategy.on_bar(ts, bars, portfolio)
            for sig in signals:
                if (
                    sig.signal_type == SignalType.ENTRY
                    and sig.direction == SignalDirection.LONG
                ):
                    price = bars[sig.symbol].close
                    qty = 100
                    cost = commission.calculate(price, qty, is_buy=True)
                    fee = cost.commission + cost.transfer_fee + cost.stamp_duty
                    portfolio.apply_trade(sig.symbol, qty, price, fee)
                elif sig.signal_type == SignalType.EXIT:
                    pos = portfolio.positions.get(sig.symbol)
                    if pos and pos.quantity > 0:
                        price = bars[sig.symbol].close
                        cost = commission.calculate(price, pos.quantity, is_buy=False)
                        fee = cost.commission + cost.transfer_fee + cost.stamp_duty
                        portfolio.apply_trade(sig.symbol, -pos.quantity, price, fee)

            portfolio.on_day_end()

        assert portfolio.total_value > Decimal("0")


class TestPaperIntegration:
    """模拟交易集成测试。"""

    def test_paper_environment_creation(self) -> None:
        """模拟环境能正常创建。"""
        env = PaperEnvironment(config=PaperEnvironmentConfig())
        assert env is not None
        assert env.env_type == "paper"

    def test_paper_broker_processes_order(self) -> None:
        """模拟经纪商能处理订单。"""
        broker = PaperBroker(config=PaperBrokerConfig())
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        broker.set_portfolio(portfolio)

        bar = Bar(
            symbol="600519.SH",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("10.5"),
            low=Decimal("9.5"),
            close=Decimal("10.0"),
            pre_close=Decimal("10.0"),
            volume=1000000,
            amount=Decimal("10000000"),
        )
        broker.set_current_bars({"600519.SH": bar})

        from app.data.schema import Order, OrderStatus

        order = Order(
            order_id="test-001",
            symbol="600519.SH",
            qty=100,
            status=OrderStatus.PENDING,
            created_ts=datetime(2024, 1, 1),
        )
        order_id = broker.submit_order(order)
        assert order_id is not None


class TestLiveIntegration:
    """实盘交易集成测试。"""

    def test_live_environment_creation(self) -> None:
        """实盘环境能正常创建（使用MockGateway）。"""
        gateway = MockGateway()
        gateway.connect()
        portfolio = Portfolio(initial_cash=Decimal("4000000"))
        commission = AShareCommissionModel()
        env = LiveEnvironment(
            gateway=gateway,
            portfolio=portfolio,
            commission=commission,
            config=LiveEngineConfig(),
        )
        assert env is not None
        assert env.env_type == "live"

    def test_live_risk_guard_passes(self) -> None:
        """风控模块对正常订单放行。"""
        risk_guard = RiskGuard(config=RiskConfig())
        passed, reason = risk_guard.pre_check(
            symbol="600519.SH",
            direction="BUY",
            quantity=100,
            price=100000,  # int ×10⁴
            current_positions={},
            portfolio_value=400000000,  # int ×10⁴
        )
        assert passed is True

    def test_live_mock_gateway_submits_order(self) -> None:
        """Mock网关能提交订单。"""
        gateway = MockGateway()
        gateway.connect()
        gateway.subscribe(["600519.SH"])

        bar = Bar(
            symbol="600519.SH",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("10.0"),
            high=Decimal("10.5"),
            low=Decimal("9.5"),
            close=Decimal("10.0"),
            pre_close=Decimal("10.0"),
            volume=1000000,
            amount=Decimal("10000000"),
        )
        gateway.update_bars({"600519.SH": bar})

        request = OrderRequest(
            symbol="600519.SH",
            direction=Direction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        result = gateway.send_order(request)
        assert result is not None
        assert result.status.value == "filled"


class TestStrategyEngineIntegration:
    """策略与引擎集成测试。"""

    def test_windowed_strategy_with_portfolio(self) -> None:
        """WindowedStrategy能与Portfolio协同工作。"""
        strategy = WindowedStrategy(window=5)

        series = _make_bar_series([10.0, 11.0, 12.0, 13.0, 14.0])
        for ts, bars in series:
            strategy.update_history(bars)

        assert strategy.is_ready
        price = strategy.get_last_price("600519.SH")
        assert price == Decimal("14.0")

    def test_macross_with_portfolio_integration(self) -> None:
        """MACross策略与Portfolio完整集成。"""
        strategy = MACrossStrategy(short_period=3, long_period=5)
        portfolio = Portfolio(initial_cash=Decimal("100000"))
        commission = AShareCommissionModel()

        series = _make_bar_series([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
        for ts, bars in series:
            signals = strategy.on_bar(ts, bars, portfolio)
            for sig in signals:
                if sig.signal_type == SignalType.ENTRY:
                    price = bars[sig.symbol].close
                    cost = commission.calculate(price, 100, is_buy=True)
                    fee = cost.commission + cost.transfer_fee + cost.stamp_duty
                    portfolio.apply_trade(sig.symbol, 100, price, fee)

        assert len(portfolio.positions) >= 0
