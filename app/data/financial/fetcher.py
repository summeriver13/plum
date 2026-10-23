"""QMT 财务数据获取器。

封装 xtdata 的 download_financial_data() 和 get_financial_data()，
提供批量下载和按需获取两种模式。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.data.financial.schema import QMT_TABLE_LIST

logger = logging.getLogger(__name__)


class FinancialFetcher:
    """QMT xtdata 财务数据获取器。

    使用前需确保 MiniQMT 客户端已启动且财务数据已通过
    界面端"数据管理 → 财务数据"下载到本地。

    Usage:
        fetcher = FinancialFetcher()
        fetcher.download(["600035.SH", "000100.SZ"])
        data = fetcher.fetch(["600035.SH"], start="20240101", end="20251231")
    """

    def __init__(self) -> None:
        self._xtdata = None

    @property
    def xtdata(self) -> Any:
        """懒加载 xtdata 模块。"""
        if self._xtdata is None:
            try:
                from xtquant import xtdata

                self._xtdata = xtdata
            except ImportError as e:
                raise ImportError(
                    "MiniQMT (xtquant) 未安装。请运行: pip install xtquant"
                ) from e
        return self._xtdata

    # ── 下载 ──────────────────────────────────────────────

    def download(
        self,
        stock_list: list[str],
        table_list: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """下载财务数据到本地 QMT 数据目录。

        优先使用 download_financial_data2（支持起止日期和回调），
        降级到 download_financial_data。

        Args:
            stock_list: QMT 格式股票列表，如 ["600035.SH", "000100.SZ"]
            table_list: 表名列表，默认四张表全部下载
            start: 起始日期 YYYYMMDD，仅 download_financial_data2 支持
            end: 结束日期 YYYYMMDD，仅 download_financial_data2 支持
            progress_callback: 进度回调，签名 callback(progress_dict)

        Returns:
            {'downloaded': int, 'failed': int, 'errors': list[str]}
        """
        tables = table_list or QMT_TABLE_LIST
        xt = self.xtdata

        logger.info(
            "开始下载财务数据: %d 只股票, 表: %s, 日期: %s~%s",
            len(stock_list),
            tables,
            start or "全部",
            end or "全部",
        )

        errors: list[str] = []

        # 尝试使用增强版 API（支持起止日期）
        download_fn = getattr(xt, "download_financial_data2", None)
        if download_fn is not None and start is not None:
            try:
                if progress_callback:
                    download_fn(stock_list, tables, start, end, progress_callback)
                else:
                    download_fn(stock_list, tables, start, end)
                return {"downloaded": len(stock_list), "failed": 0, "errors": []}
            except Exception as e:
                errors.append(f"download_financial_data2 失败: {e}")
                logger.warning("download_financial_data2 失败，降级到基础版: %s", e)

        # 基础版 API
        try:
            xt.download_financial_data(stock_list, tables)
        except Exception as e:
            errors.append(f"download_financial_data 失败: {e}")
            logger.error("财务数据下载失败: %s", e)
            return {"downloaded": 0, "failed": len(stock_list), "errors": errors}

        logger.info("财务数据下载完成: %d 只股票", len(stock_list))
        return {"downloaded": len(stock_list), "failed": 0, "errors": errors}

    # ── 获取 ──────────────────────────────────────────────

    def fetch(
        self,
        stock_list: list[str],
        table_list: list[str] | None = None,
        start: str = "",
        end: str = "",
        report_type: str = "report_time",
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """获取已下载的财务数据。

        Args:
            stock_list: 股票列表
            table_list: 表名列表，默认全部
            start: 起始时间 YYYYMMDD，空字符串表示全部
            end: 结束时间 YYYYMMDD
            report_type: 'report_time' 按报告期 | 'announce_time' 按公告日

        Returns:
            {stock_code: {table_name: DataFrame}}
            DataFrame 包含 m_anntime（披露日）、m_timetag（截止日）
            和对应的财务字段列
        """
        tables = table_list or QMT_TABLE_LIST
        xt = self.xtdata

        try:
            data = xt.get_financial_data(
                stock_list=stock_list,
                table_list=tables,
                start_time=start,
                end_time=end,
                report_type=report_type,
            )
        except Exception as e:
            logger.error("获取财务数据失败: %s", e)
            raise RuntimeError(f"获取财务数据失败: {e}") from e

        return self._validate_and_log(data, stock_list, tables)

    def _validate_and_log(
        self,
        data: dict,
        stock_list: list[str],
        tables: list[str],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """验证返回数据完整性。"""
        missing_stocks: list[str] = []
        empty_count = 0

        for stock in stock_list:
            if stock not in data:
                missing_stocks.append(stock)
                continue
            for table in tables:
                df = data[stock].get(table)
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    empty_count += 1
                    break  # 一只股票只计一次

        available = [s for s in stock_list if s in data]
        no_data = len(stock_list) - len(available) + empty_count

        # 摘要一行，不逐只刷屏
        if no_data > 0:
            logger.info(
                "获取财务数据: %d/%d 只有数据, %d 只无数据 (新股/长期停牌), 表: %s",
                len(available) - empty_count,
                len(stock_list),
                no_data,
                tables,
            )
        else:
            logger.info(
                "获取财务数据: %d/%d 只股票有数据, 表: %s",
                len(available),
                len(stock_list),
                tables,
            )

        # 仅 debug 级别保留逐只明细
        if missing_stocks:
            logger.debug("无返回数据: %s", missing_stocks)

        return data

    # ── 便捷方法：一步获取 + 验证 ────────────────────────

    def fetch_table(
        self,
        stock_list: list[str],
        table: str,
        start: str = "",
        end: str = "",
        report_type: str = "report_time",
    ) -> dict[str, pd.DataFrame]:
        """获取单张表的财务数据。

        Returns:
            {stock_code: DataFrame}
        """
        raw = self.fetch(stock_list, [table], start, end, report_type)
        return {
            stock: tables[table]
            for stock, tables in raw.items()
            if table in tables and not tables[table].empty
        }

    def fetch_all_stocks(
        self,
        start: str = "",
        end: str = "",
        report_type: str = "report_time",
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """获取全部 A 股的财务数据。

        Returns:
            同 fetch()
        """
        xt = self.xtdata
        all_stocks = xt.get_stock_list_in_sector("沪深A股")
        return self.fetch(all_stocks, start=start, end=end, report_type=report_type)

    # ── 自检 ──────────────────────────────────────────────

    def ping(self) -> bool:
        """快速连接检测。

        Returns:
            True 如果 QMT 服务可达
        """
        try:
            xt = self.xtdata
            xt.get_trading_dates("SH")
            return True
        except Exception:
            logger.warning("财务数据源健康检查失败", exc_info=True)
            return False

    def validate_schema(
        self,
        test_stocks: list[str] | None = None,
    ) -> dict[str, Any]:
        """启动时校验 QMT 财务数据 Schema。

        验证四张表的关键字段是否存在，确保与 preprocessor 的映射一致。
        建议在首次获取财务数据前调用一次。

        Args:
            test_stocks: 测试股票列表，默认取沪深 300 前 3 只

        Returns:
            {
                "ok": bool,
                "tables": {table: {"ok": bool, "missing_fields": [...]}},
                "dividend": {"ok": bool, "has_interest": bool},
            }
        """
        xt = self.xtdata

        if test_stocks is None:
            all_stocks = xt.get_stock_list_in_sector("沪深300")
            stocks: list[str] = list(all_stocks[:3]) if all_stocks else ["600519.SH"]
        else:
            stocks = test_stocks

        # 关键字段定义
        required_fields: dict[str, list[str]] = {
            "Income": [
                "net_profit_excl_min_int_inc",
                "revenue",
                "total_expense",
                "oper_profit",
            ],
            "Balance": ["total_equity"],
            "CashFlow": [
                "net_cash_flows_oper_act",
                "cash_pay_acq_const_fiolta",
            ],
            "PershareIndex": [
                "du_return_on_equity",
                "inc_net_profit_rate",
                "net_profit",
                "s_fa_bps",
                "s_fa_eps_basic",
                "equity_roe",
            ],
        }

        tables_list = list(required_fields.keys())

        # 下载并获取测试数据
        try:
            xt.download_financial_data2(stocks, tables_list, "20200101", "20251231")
        except Exception:
            logger.warning("Schema 验证下载失败，尝试使用缓存数据")

        data = self.fetch(stocks, tables_list, "", "", "report_time")

        # 逐表校验
        result: dict[str, Any] = {"ok": True, "tables": {}}
        for table, fields in required_fields.items():
            all_present = True
            missing: list[str] = []
            # 在所有测试股票中检查字段
            for stock in stocks:
                if stock not in data:
                    continue
                df = data[stock].get(table)
                if df is None or df.empty:
                    continue
                for field in fields:
                    if field not in df.columns:
                        all_present = False
                        if field not in missing:
                            missing.append(field)

            result["tables"][table] = {
                "ok": all_present,
                "missing_fields": missing,
            }
            if not all_present:
                result["ok"] = False
                logger.error("财务 Schema 校验失败: %s 缺少字段 %s", table, missing)

        # 分红接口
        dividend_ok = False
        has_interest = False
        try:
            div = xt.get_divid_factors(stocks[0], "20200101", "20251231")
            if div is not None and not div.empty:
                has_interest = "interest" in div.columns
                dividend_ok = has_interest
        except Exception as e:
            logger.warning("分红接口校验失败: %s", e)

        result["dividend"] = {"ok": dividend_ok, "has_interest": has_interest}
        if not dividend_ok:
            logger.warning("分红接口缺少 interest 字段")

        if result["ok"]:
            logger.info("财务 Schema 校验全部通过: %d 张表 OK", len(tables_list))
        return result
