"""回测运行器 — 策略 → 选股 → 回测 → 报告 全自动流水线。

用法:
    from app.engine.backtest.runner import BacktestRunner

    runner = BacktestRunner(
        strategy_class=MyStrategy,        # 策略由调用方注入，引擎不内置任何策略
        strategy_kwargs={"select_num": 20},
        benchmark_symbol="000852.SH",     # 中证1000 小盘基准
        start=datetime(2020, 1, 2),
        end=datetime(2025, 5, 30),
    )
    runner.run()

= 三步完成:
    1. 从 QMT 拉全市场数据 → 策略选股 → stock universe
    2. Engine.run(symbols, start, end, strategy) → BacktestResult
    3. comprehensive_report + ResultStore.save → data/results/{run_id}/
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from collections.abc import Callable
from typing import Any

from app.config.logging import start_run
from app.config.paths import MARKET_PATH, RESULTS_PATH

logger = logging.getLogger(__name__)


class BacktestRunner:
    """回测流水线运行器。

    封装了选股 → 回测 → 报告的全流程，一次调用完成。
    """

    def __init__(
        self,
        strategy_class: type,
        strategy_kwargs: dict[str, Any] | None = None,
        benchmark_symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: Decimal = Decimal("1000000"),
        results_root: str | Path = str(RESULTS_PATH),
        prewarm: Callable[[list[str]], None] | None = None,
        **engine_kwargs: Any,
    ):
        """
        Args:
            strategy_class: 策略类（如 FundamentalSmallCapStrategy）
            strategy_kwargs: 策略参数（如 select_num=20）
            benchmark_symbol: 基准指数（如 "000852.SH"）
            start: 回测起始日期
            end: 回测截止日期
            initial_cash: 初始资金
            results_root: 结果存储根目录
            prewarm: 选股前对股票池做数据预热的回调（如批量下载财务数据），
                默认不预热；预热逻辑属于具体策略所在层，引擎不感知
            **engine_kwargs: 传给 Engine 的额外参数（如 exclude_st, max_positions）
        """
        self.strategy_class = strategy_class
        self.strategy_kwargs = strategy_kwargs or {}
        self.benchmark_symbol = benchmark_symbol
        self.start = start or datetime(2020, 1, 2)
        self.end = end or datetime.now()
        self.initial_cash = initial_cash
        self.results_root = Path(results_root)
        self.prewarm = prewarm
        self.engine_kwargs = engine_kwargs

    def run(self) -> Path:
        """执行完整回测流水线。

        Returns:
            结果目录路径
        """
        # ── 0. 启动运行上下文 ────────────────────────────
        rid = start_run()
        logger.info("运行 ID: %s", rid)

        # ── 1. 组装策略 ────────────────────────────────────
        strategy = self.strategy_class(**self.strategy_kwargs)
        strategy_id = getattr(strategy, "strategy_id", self.strategy_class.__name__)

        logger.info("策略: %s", strategy_id)

        # ── 2. 获取股票池 ──────────────────────────────────
        symbols = self._get_universe(strategy)
        logger.info("股票池: %d 只", len(symbols))

        # ── 2.5 数据预热（可选）────────────────────────────
        #         避免回测首日 on_bar 时串行阻塞；预热实现由调用方注入
        if self.prewarm is not None:
            self.prewarm(symbols)

        # ── 3. 创建引擎 ────────────────────────────────────
        from app.engine.backtest.engine import Engine
        from app.engine.backtest.portfolio import Portfolio
        from app.engine.backtest.scenarios.commission import AShareCommissionModel

        # 佣金模型可由调用方覆盖（如成本归因对照实验传零成本模型），默认 A 股费率
        commission_model = (
            self.engine_kwargs.pop("commission", None) or AShareCommissionModel()
        )
        # 交收制度由调用方指定：股票 t1；ETF/可转债 t0（当日买入即可卖）
        settlement = self.engine_kwargs.pop("settlement", "t1")
        engine = Engine(
            portfolio=Portfolio(
                initial_cash=self.initial_cash, settlement=settlement
            ),
            catalog=self._get_catalog(),
            commission=commission_model,
            enable_order_splitting=self.engine_kwargs.pop(
                "enable_order_splitting", False
            ),
            **self.engine_kwargs,
        )

        # ── 4. 回测 ────────────────────────────────────────
        logger.info(
            "回测: %s ~ %s (%d 只)", self.start.date(), self.end.date(), len(symbols)
        )
        result = engine.run(symbols, self.start, self.end, strategy)

        # ── 5. 生成报告 ────────────────────────────────────
        from app.engine.backtest.analysis.report import comprehensive_report

        report = comprehensive_report(
            result,
            run_id=strategy_id,
            benchmark_symbol=self.benchmark_symbol,
        )

        # ── 6. 落盘 ────────────────────────────────────────
        from app.engine.backtest.analysis.result_store import (
            ResultStore,
            detect_strategy_family,
            detect_strategy_name,
        )

        family = detect_strategy_family(self.strategy_class)
        strategy_name = detect_strategy_name(self.strategy_class)

        store = ResultStore(root=self.results_root)
        run_dir = store.save(
            result,
            meta={"strategy_id": strategy_id},
            report=report,
            benchmark_symbol=self.benchmark_symbol,
            family=family,
            strategy_name=strategy_name,
        )

        logger.info("结果: %s", run_dir)
        logger.info("报告: %s/report.md", run_dir)
        return run_dir

    # ── 内部 ──────────────────────────────────────────

    def _get_catalog(self):
        """获取数据目录管理器。"""
        from app.data.store.market import MarketCatalog

        return MarketCatalog()

    def _get_universe(self, strategy) -> list[str]:
        """从 catalog 获取缓存股票列表。"""
        return self._universe_fallback(strategy)

    def _universe_fallback(self, strategy) -> list[str]:
        """回退：从 catalog 获取所有有缓存的股票列表。"""

        market_root = MARKET_PATH / "1d"
        cached: set[str] = set()
        for d in ["SH", "SZ"]:
            for f in (market_root / d).glob("*.parquet"):
                cached.add(f"{d}.{f.stem}")

        stocks = sorted(cached)
        if not stocks:
            raise RuntimeError("catalog 中无可用股票缓存")

        # 全市场截面排名策略需要完整候选池，不做截断
        return stocks
