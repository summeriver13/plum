"""Tests for corporate action processors."""

from datetime import datetime
from decimal import Decimal

from app.rule.corporate.actions import (
    ActionType,
    BonusAction,
    DividendAction,
    RightsIssueAction,
    SplitAction,
)
from app.rule.corporate.dividend import (
    BonusProcessor,
    DividendProcessor,
    RightsIssueProcessor,
    SplitProcessor,
)


class TestDividendProcessor:
    def test_process_dividend(self):
        processor = DividendProcessor()
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is True
        assert result.cash_change == 5000000
        assert result.quantity_change == 0
        assert result.cost_basis_change == 0
        assert "分红" in result.message

    def test_process_dividend_no_position(self):
        processor = DividendProcessor()
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=0, cost_basis=0, available_cash=50000
        )
        assert result.success is True
        assert "skipped" in result.message

    def test_process_dividend_wrong_action_type(self):
        processor = DividendProcessor()
        action = BonusAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.BONUS,
            bonus_ratio=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is False
        assert "Invalid action type" in result.message


class TestBonusProcessor:
    def test_process_bonus(self):
        processor = BonusProcessor()
        action = BonusAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.BONUS,
            bonus_ratio=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is True
        assert result.quantity_change == 500
        assert result.cash_change == 0
        assert result.cost_basis_change == 0
        assert "送股" in result.message

    def test_process_bonus_no_position(self):
        processor = BonusProcessor()
        action = BonusAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.BONUS,
            bonus_ratio=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=0, cost_basis=0, available_cash=50000
        )
        assert result.success is True
        assert "skipped" in result.message

    def test_process_bonus_wrong_action_type(self):
        processor = BonusProcessor()
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is False
        assert "Invalid action type" in result.message


class TestRightsIssueProcessor:
    def test_process_rights_issue_subscribe(self):
        processor = RightsIssueProcessor(auto_subscribe=True)
        action = RightsIssueAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.RIGHTS_ISSUE,
            subscription_price=Decimal("8.0"),
            ratio=Decimal("0.3"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000000
        )
        assert result.success is True
        assert result.quantity_change == 300
        assert result.cash_change == -24000000
        assert result.cost_basis_change == 24000000
        assert "认购配股" in result.message

    def test_process_rights_issue_insufficient_funds(self):
        processor = RightsIssueProcessor(auto_subscribe=True)
        action = RightsIssueAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.RIGHTS_ISSUE,
            subscription_price=Decimal("8.0"),
            ratio=Decimal("0.3"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=1000
        )
        assert result.success is False
        assert "资金不足" in result.message

    def test_process_rights_issue_renounce(self):
        processor = RightsIssueProcessor(auto_subscribe=False)
        action = RightsIssueAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.RIGHTS_ISSUE,
            subscription_price=Decimal("8.0"),
            ratio=Decimal("0.3"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=500000
        )
        assert result.success is True
        assert "放弃配股" in result.message

    def test_process_rights_issue_no_position(self):
        processor = RightsIssueProcessor(auto_subscribe=True)
        action = RightsIssueAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.RIGHTS_ISSUE,
            subscription_price=Decimal("8.0"),
            ratio=Decimal("0.3"),
        )
        result = processor.process(
            action, quantity=0, cost_basis=0, available_cash=500000
        )
        assert result.success is True
        assert "skipped" in result.message

    def test_process_rights_issue_wrong_action_type(self):
        processor = RightsIssueProcessor(auto_subscribe=True)
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=500000
        )
        assert result.success is False
        assert "Invalid action type" in result.message


class TestSplitProcessor:
    def test_process_split(self):
        processor = SplitProcessor()
        action = SplitAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.SPLIT,
            split_ratio=Decimal("2.0"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is True
        assert result.quantity_change == 1000
        assert result.cash_change == 0
        assert result.cost_basis_change == 0
        assert "拆股" in result.message
        assert "1000股 → 2000股" in result.message

    def test_process_split_no_position(self):
        processor = SplitProcessor()
        action = SplitAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.SPLIT,
            split_ratio=Decimal("2.0"),
        )
        result = processor.process(
            action, quantity=0, cost_basis=0, available_cash=50000
        )
        assert result.success is True
        assert "skipped" in result.message

    def test_process_split_wrong_action_type(self):
        processor = SplitProcessor()
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=Decimal("0.5"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is False
        assert "Invalid action type" in result.message

    def test_process_split_ratio_3_for_1(self):
        processor = SplitProcessor()
        action = SplitAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.SPLIT,
            split_ratio=Decimal("3.0"),
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is True
        assert result.quantity_change == 2000
