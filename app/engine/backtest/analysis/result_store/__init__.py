"""回测结果持久化。

自动落盘 BacktestResult 到 CSV，按家族/策略/时间 分目录。
目录结构:
    data/results/{family}/{strategy_name}/{timestamp}/
    ├── data/
    │   ├── equity_curve.csv
    │   └── trade_log.csv
    ├── charts/
    │   └── *.png              (如果传了 report 参数)
    ├── metadata.json
    ├── report.json
    └── report.md              (Markdown 报告，嵌入图表 + 指标汇总)

family 映射:
    app/strategy/       → plum
    app/graft/kapok/    → kapok
    app/graft/hologram_rose/ → rose
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.paths import RESULTS_PATH
from app.engine.backtest.analysis.result_store._io import (
    _equity_to_df,
    _parking_to_df,
    _serialize_dict,
    _trades_to_df,
)
from app.engine.backtest.analysis.result_store._md import _write_md_report
from app.engine.backtest.models import BacktestResult
from app.engine.backtest.provenance import collect_provenance

logger = logging.getLogger(__name__)


def _safe_num(value: object, default: float = 0.0) -> float:
    """容错转 float（Decimal/None 均可）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: object) -> None:
    """写入 JSON（失败抛 RuntimeError，保留原始原因）。"""
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"结果文件写入失败: {path}（{exc}）") from exc


def _benchmark_nav(symbol: str | None, result: BacktestResult) -> list[float] | None:
    """加载基准价格并归一化为净值（净值曲线绘图用）；不可用时返回 None。"""
    if not symbol:
        return None
    from app.engine.backtest.analysis.benchmarks import load_aligned_benchmark

    try:
        values = load_aligned_benchmark(
            symbol, [point.timestamp for point in result.equity_curve]
        )
    except Exception:  # noqa: BLE001 — 绘制基准失败不阻断落盘
        logger.debug("基准净值加载失败: %s", symbol, exc_info=True)
        return None
    if not values or values[0] <= 0:
        return None
    return [value / values[0] for value in values]


def detect_strategy_family(strategy_class: type) -> str:
    """根据策略类的模块路径推断所属家族。

    Returns:
        "plum" | "kapok" | "rose"
    """
    module = strategy_class.__module__
    if "graft.kapok" in module:
        return "kapok"
    if "graft.hologram_rose" in module:
        return "rose"
    return "plum"


def detect_strategy_name(strategy_class: type) -> str:
    """根据策略类名推断英文策略名。

    PascalCase → snake_case, 去掉 Strategy 后缀。
    """
    import re

    name = strategy_class.__name__
    # PascalCase → snake_case
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    # 去掉末尾的 _strategy
    name = re.sub(r"_strategy$", "", name)
    return name


class ResultStore:
    """回测结果持久化存储。

    Usage:
        store = ResultStore(root="data/results")
        store.save(result, meta={"strategy_id": "fund_small"}, report=report)
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            try:
                from app.config.settings import get_config

                root = get_config().data.results_path
            except Exception:
                logger.debug("配置加载失败，使用默认 results 路径", exc_info=True)
                root = str(RESULTS_PATH)
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: BacktestResult,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
        report: Any = None,
        benchmark_symbol: str | None = None,
        family: str | None = None,
        strategy_name: str | None = None,
    ) -> Path:
        """保存回测结果。

        Args:
            result: engine.run() 返回的 BacktestResult
            run_id: 运行标识，默认用时间戳
            meta: 额外元数据（策略参数等）
            report: ComprehensiveReport 对象，传了就在同目录下出图
            benchmark_symbol: 基准指数代码（如 000852.SH），传了自动重算报告
            family: 策略家族（plum/kapok/rose）
            strategy_name: 策略英文名（如 fundamental_small_cap）
        """
        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = meta.get("strategy_id", "") if meta else ""
            if not name:
                name = strategy_name or result.metadata.get("strategy_id", "")
            if not name:
                name = "unnamed"
            run_id = ts

        # 构建嵌套路径: {family}/{strategy_name}/{run_id}
        parts: list[str] = [str(self._root)]
        if family:
            parts.append(family)
        if strategy_name:
            parts.append(strategy_name)
        parts.append(run_id)
        run_dir = Path(*parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        data_dir = run_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Equity curve
        if result.equity_curve:
            df_equity = _equity_to_df(result.equity_curve)
            df_equity.to_csv(data_dir / "equity_curve.csv", index=False)

        # Trade log
        if result.trade_log:
            df_trades = _trades_to_df(result.trade_log)
            df_trades.to_csv(data_dir / "trade_log.csv", index=False)

        # 停泊标的成交（不计入策略换手）
        if result.parking_log:
            df_parking = _parking_to_df(result.parking_log)
            df_parking.to_csv(data_dir / "parking_log.csv", index=False)

        # Metadata
        md = {
            "run_id": run_id,
            "strategy": result.symbol,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_cash": _safe_num(result.initial_cash),
            "saved_at": datetime.now().isoformat(),
            "provenance": collect_provenance(),  # 数据副本指纹 + 代码状态（复核数字前先比指纹）
        }
        if meta:
            md.update(meta)
        if result.metadata:
            md["engine"] = _serialize_dict(result.metadata)

        _write_json(run_dir / "metadata.json", md)

        # Report（指定基准则重算，图表与 JSON 均使用重算后的报告）
        if report is not None and hasattr(report, "to_dict"):
            if benchmark_symbol:
                from app.engine.backtest.analysis.report import comprehensive_report

                report = comprehensive_report(
                    result, run_id, benchmark_symbol=benchmark_symbol
                )
            _write_json(run_dir / "report.json", report.to_dict())
        # Charts（净值曲线绘制基准线；基准缺失时回退为归一化自身）
        if report is not None:
            from app.engine.backtest.analysis.charts import generate_all

            generate_all(
                result,
                report,
                str(run_dir / "charts"),
                benchmark_values=_benchmark_nav(benchmark_symbol, result),
            )

        # Markdown report (embed charts + metrics)
        _write_md_report(run_dir, report, meta)

        logger.info("Result saved: %s", run_dir)
        return run_dir

    def load(self, run_id: str) -> dict[str, Any]:
        """加载保存的回测结果。"""
        run_dir = self._root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {run_dir}")

        result: dict[str, Any] = {}
        data_dir = run_dir / "data"
        eq_path = data_dir / "equity_curve.csv"
        if eq_path.exists():
            result["equity_curve"] = pd.read_csv(eq_path, parse_dates=["timestamp"])

        tr_path = data_dir / "trade_log.csv"
        if tr_path.exists():
            result["trade_log"] = pd.read_csv(tr_path, parse_dates=["timestamp"])

        md_path = run_dir / "metadata.json"
        if md_path.exists():
            try:
                result["metadata"] = json.loads(md_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("metadata 读取失败: %s（%s）", md_path, exc)

        return result

    def list_runs(self, family: str | None = None) -> list[str]:
        """列出所有已保存的回测运行。

        Args:
            family: 可选，只列出指定家族的运行
        """
        if not self._root.exists():
            return []
        runs: list[str] = []
        root = self._root / family if family else self._root
        if not root.exists():
            return []
        for family_dir in root.iterdir() if family is None else [root]:
            if not family_dir.is_dir():
                continue
            for strategy_dir in family_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue
                for run_dir in strategy_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "metadata.json").exists():
                        runs.append(str(run_dir.relative_to(self._root)))
        return sorted(runs, reverse=True)
