"""财务字段映射 — QMT 表/字段 → 输出列。

映射表由数据层持有：QMT 是数据源，输出列名是下游因子与策略共同依赖的契约。
策略层只消费这里定义的列名，不反向定义数据层。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FinancialTable(str, Enum):
    """QMT 财务数据表名。"""

    INCOME = "Income"
    BALANCE = "Balance"
    CASH_FLOW = "CashFlow"
    PER_SHARE_INDEX = "PershareIndex"


@dataclass
class FieldMapping:
    """单个字段映射：QMT 源字段 → 输出列名。"""

    qmt_table: FinancialTable
    qmt_field: str
    column: str
    description: str
    is_precomputed: bool = False
    needs_ttm: bool = False


# ═══════════════════════════════════════════════════════
# 输出列 ← QMT 源字段 的完整映射表
# ═══════════════════════════════════════════════════════

FIELD_MAP: list[FieldMapping] = [
    # 利润表 (Income)
    FieldMapping(
        qmt_table=FinancialTable.INCOME,
        qmt_field="net_profit_excl_min_int_inc",
        column="归母净利润",
        description="归属于母公司所有者的净利润（单季）",
        needs_ttm=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.INCOME,
        qmt_field="net_profit_incl_min_int_inc",
        column="净利润",
        description="净利润（含少数股东，单季）",
        needs_ttm=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.INCOME,
        qmt_field="revenue",
        column="营业总收入",
        description="营业总收入（单季）",
        needs_ttm=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.INCOME,
        qmt_field="total_expense",
        column="营业成本",
        description="营业成本（单季）",
        needs_ttm=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.INCOME,
        qmt_field="oper_profit",
        column="营业利润",
        description="营业利润（单季）",
        needs_ttm=True,
    ),
    # 每股指标 (PershareIndex) — 大部分为预计算值
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="du_return_on_equity",
        column="ROE",
        description="净资产收益率（TTM）",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="inc_net_profit_rate",
        column="归母净利润同比",
        description="归属于母公司所有者的净利润同比增长率",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="inc_net_profit",
        column="归母净利润_预计算",
        description="归属于母公司所有者的净利润（预计算，用于交叉验证）",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="net_profit",
        column="净利润率",
        description="净利率",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="gross_profit",
        column="毛利率",
        description="毛利率",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="du_profit_rate",
        column="净利润同比",
        description="净利润同比增长率",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="inc_revenue_rate",
        column="营业收入同比",
        description="主营收入同比增长率",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="s_fa_bps",
        column="每股净资产",
        description="每股净资产",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="s_fa_eps_basic",
        column="基本每股收益",
        description="基本每股收益",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="s_fa_ocfps",
        column="每股经营现金流",
        description="每股经营活动现金流量",
        is_precomputed=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.PER_SHARE_INDEX,
        qmt_field="equity_roe",
        column="加权ROE",
        description="加权净资产收益率",
        is_precomputed=True,
    ),
    # 现金流量表 (CashFlow)
    FieldMapping(
        qmt_table=FinancialTable.CASH_FLOW,
        qmt_field="net_cash_flows_oper_act",
        column="经营现金流",
        description="经营活动产生的现金流量净额（单季）",
        needs_ttm=True,
    ),
    FieldMapping(
        qmt_table=FinancialTable.CASH_FLOW,
        qmt_field="cash_pay_acq_const_fiolta",
        column="资本支出",
        description="购建固定资产/无形资产支付的现金（单季）",
        needs_ttm=True,
    ),
]

# 派生计算字段（由 preprocessor 计算，不从 QMT 直接取）
DERIVED_FIELDS: dict[str, str] = {
    "自由现金流": "经营现金流(TTM) - 资本支出(TTM)",
    "ROE_np_单季": "归母净利润(单季) / 净资产",
    "归母净利润同比增速": "归母净利润同比",
}

# 全部财务列名（方便查询）
FIN_COLUMNS: list[str] = [fm.column for fm in FIELD_MAP]

# 每张财务表的通用元数据列
TABLE_META_COLUMNS: set[str] = {"m_anntime", "m_timetag"}
