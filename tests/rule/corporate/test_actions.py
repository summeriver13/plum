"""Tests for corporate actions module."""

import pytest
from datetime import datetime

from app.rule.corporate.actions import (
    ActionProcessor,
    ActionResult,
    ActionType,
    BonusAction,
    CorporateActionProcessor,
    DividendAction,
    RightsIssueAction,
    SplitAction,
)


class TestDividendAction:
    def test_create_dividend_action(self):
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        assert action.symbol == "600000.SH"
        assert action.dividend_per_share == 0.5
        assert "现金分红" in action.description

    def test_dividend_action_wrong_event_type(self):
        with pytest.raises(ValueError):
            DividendAction(
                symbol="600000.SH",
                ex_date=datetime(2024, 1, 1),
                pub_date=datetime(2023, 12, 1),
                event_type=ActionType.BONUS,
                dividend_per_share=0.5,
            )


class TestBonusAction:
    def test_create_bonus_action(self):
        action = BonusAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.BONUS,
            bonus_ratio=0.5,
        )
        assert action.bonus_ratio == 0.5
        assert "送股" in action.description

    def test_bonus_action_wrong_event_type(self):
        with pytest.raises(ValueError):
            BonusAction(
                symbol="600000.SH",
                ex_date=datetime(2024, 1, 1),
                pub_date=datetime(2023, 12, 1),
                event_type=ActionType.DIVIDEND,
                bonus_ratio=0.5,
            )


class TestRightsIssueAction:
    def test_create_rights_issue_action(self):
        action = RightsIssueAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.RIGHTS_ISSUE,
            subscription_price=8.0,
            ratio=0.3,
        )
        assert action.subscription_price == 8.0
        assert action.ratio == 0.3
        assert action.is_renounceable is True
        assert "配股" in action.description

    def test_rights_issue_action_wrong_event_type(self):
        with pytest.raises(ValueError):
            RightsIssueAction(
                symbol="600000.SH",
                ex_date=datetime(2024, 1, 1),
                pub_date=datetime(2023, 12, 1),
                event_type=ActionType.DIVIDEND,
                subscription_price=8.0,
                ratio=0.3,
            )


class TestSplitAction:
    def test_create_split_action(self):
        action = SplitAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.SPLIT,
            split_ratio=2.0,
        )
        assert action.split_ratio == 2.0
        assert "拆股" in action.description

    def test_split_action_wrong_event_type(self):
        with pytest.raises(ValueError):
            SplitAction(
                symbol="600000.SH",
                ex_date=datetime(2024, 1, 1),
                pub_date=datetime(2023, 12, 1),
                event_type=ActionType.DIVIDEND,
                split_ratio=2.0,
            )


class TestActionResult:
    def test_empty_result(self):
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        result = ActionResult(action=action, success=True)
        assert result.success is True
        assert result.quantity_change == 0
        assert result.cash_change == 0
        assert result.cost_basis_change == 0
        assert result.message == ""
        assert result.description == "无变化"

    def test_result_with_changes(self):
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        result = ActionResult(
            action=action,
            success=True,
            quantity_change=100,
            cash_change=50000,
            cost_basis_change=-10000,
        )
        assert "股数" in result.description
        assert "现金" in result.description
        assert "成本" in result.description

    def test_result_negative_changes(self):
        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        result = ActionResult(
            action=action,
            success=True,
            quantity_change=-50,
            cash_change=-10000,
        )
        desc = result.description
        assert "+" not in desc or "无变化" not in desc


class TestCorporateActionProcessor:
    def test_register_and_process(self):
        class MockProcessor(ActionProcessor):
            def process(self, action, quantity, cost_basis, available_cash):
                return ActionResult(
                    action=action, success=True, message="Mock processed"
                )

        processor = CorporateActionProcessor()
        processor.register(ActionType.DIVIDEND, MockProcessor())

        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is True
        assert result.message == "Mock processed"

    def test_process_unregistered_action(self):
        processor = CorporateActionProcessor()

        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        result = processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )
        assert result.success is False
        assert "No processor" in result.message

    def test_action_log(self):
        class MockProcessor(ActionProcessor):
            def process(self, action, quantity, cost_basis, available_cash):
                return ActionResult(action=action, success=True, message="Processed")

        processor = CorporateActionProcessor()
        processor.register(ActionType.DIVIDEND, MockProcessor())

        action1 = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        action2 = DividendAction(
            symbol="000001.SZ",
            ex_date=datetime(2024, 2, 1),
            pub_date=datetime(2024, 1, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.3,
        )

        processor.process(
            action1, quantity=1000, cost_basis=100000, available_cash=50000
        )
        processor.process(action2, quantity=500, cost_basis=50000, available_cash=25000)

        log = processor.get_action_log()
        assert len(log) == 2

        processor.clear_log()
        assert len(processor.get_action_log()) == 0

    def test_get_action_log_returns_copy(self):
        class MockProcessor(ActionProcessor):
            def process(self, action, quantity, cost_basis, available_cash):
                return ActionResult(action=action, success=True)

        processor = CorporateActionProcessor()
        processor.register(ActionType.DIVIDEND, MockProcessor())

        action = DividendAction(
            symbol="600000.SH",
            ex_date=datetime(2024, 1, 1),
            pub_date=datetime(2023, 12, 1),
            event_type=ActionType.DIVIDEND,
            dividend_per_share=0.5,
        )
        processor.process(
            action, quantity=1000, cost_basis=100000, available_cash=50000
        )

        log1 = processor.get_action_log()
        log2 = processor.get_action_log()
        assert log1 is not log2
