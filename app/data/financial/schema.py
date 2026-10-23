"""财务数据模型 — QMT 表结构、字段映射、输出列定义。

覆盖四张表：Income（利润表）、Balance（资产负债表）、
CashFlow（现金流量表）、PershareIndex（每股指标）。

分红数据来自 xtdata.get_divid_factors() 的 interest 字段，
复用 QMTConnector 现有接口。
"""

from __future__ import annotations

from app.data.financial.field_map import (
    DERIVED_FIELDS,
    FIELD_MAP,
    TABLE_META_COLUMNS,
    FieldMapping,
    FinancialTable,
)

# xtdata.get_financial_data() 需要的表名列表
QMT_TABLE_LIST: list[str] = [
    FinancialTable.INCOME.value,
    FinancialTable.BALANCE.value,
    FinancialTable.CASH_FLOW.value,
    FinancialTable.PER_SHARE_INDEX.value,
]

# 字段映射、输出列与元数据列的定义在 field_map.py，这里只做转发
__all__ = [
    "DERIVED_FIELDS",
    "FIELD_MAP",
    "FieldMapping",
    "FinancialTable",
    "QMT_FIELD_DEFINITIONS",
    "QMT_TABLE_LIST",
    "TABLE_META_COLUMNS",
]

# ═══════════════════════════════════════════════════════════
# QMT 四张表的完整字段定义（用于文档和调试）
# ═══════════════════════════════════════════════════════════

QMT_FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "Income": {
        "m_anntime": "披露日期",
        "m_timetag": "截止日期",
        "net_profit_excl_min_int_inc": "归属母公司所有者的净利润",
        "net_profit_incl_min_int_inc": "净利润（含少数股东）",
        "net_profit_incl_min_int_inc_after": "净利润（扣除非经常性损益后）",
        "revenue": "营业总收入",
        "revenue_inc": "营业收入",
        "total_operating_cost": "营业总成本",
        "total_expense": "营业成本",
        "less_taxes_surcharges_ops": "营业税金及附加",
        "sale_expense": "销售费用",
        "less_gerl_admin_exp": "管理费用",
        "financial_expense": "财务费用",
        "research_expenses": "研发费用",
        "less_impair_loss_assets": "资产减值损失",
        "plus_net_invest_inc": "投资收益",
        "oper_profit": "营业利润",
        "plus_non_oper_rev": "营业外收入",
        "less_non_oper_exp": "营业外支出",
        "tot_profit": "利润总额",
        "inc_tax": "所得税",
        "change_income_fair_value": "公允价值变动收益",
        "minority_int_inc": "少数股东损益",
        "total_income": "综合收益总额",
        "s_fa_eps_basic": "基本每股收益",
        "s_fa_eps_diluted": "稀释每股收益",
    },
    "Balance": {
        "m_anntime": "披露日期",
        "m_timetag": "截止日期",
    },
    "CashFlow": {
        "m_anntime": "披露日期",
        "m_timetag": "截止日期",
        "goods_sale_and_service_render_cash": "销售商品提供劳务收到的现金",
        "stot_cash_in_oper_act": "经营活动现金流入小计",
        "stot_cash_out_oper_act": "经营活动现金流出小计",
        "net_cash_flows_oper_act": "经营活动产生的现金流量净额",
        "cass_paid_for_assets": "购建固定资产/无形资产支付的现金",
    },
    "PershareIndex": {
        "m_anntime": "披露日期",
        "m_timetag": "截止日期",
        "du_return_on_equity": "净资产收益率",
        "inc_net_profit_rate": "归属净利润同比增长",
        "inc_net_profit": "归属净利润",
        "net_profit": "净利率",
        "gross_profit": "毛利率",
        "du_profit_rate": "净利润同比增长",
        "inc_revenue_rate": "主营收入同比增长",
        "s_fa_bps": "每股净资产",
        "s_fa_eps_basic": "基本每股收益",
        "s_fa_ocfps": "每股经营活动现金流量",
        "equity_roe": "加权净资产收益率",
        "net_roe": "摊薄净资产收益率",
        "total_roe": "摊薄总资产收益率",
        "s_fa_undistributedps": "每股未分配利润",
        "s_fa_surpluscapitalps": "每股资本公积金",
        "inc_revenue": "主营收入",
        "inc_gross_profit": "毛利润",
        "inc_profit_before_tax": "利润总额",
        "du_profit": "净利润",
    },
}
