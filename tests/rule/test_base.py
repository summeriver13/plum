"""Tests for rule base module."""

import pytest

from app.rule.base import Rule, RuleChain, RuleResult


class ConcreteRule(Rule):
    """Concrete rule implementation for testing."""

    def __init__(self, should_pass: bool = True, description: str = "Test Rule"):
        self._should_pass = should_pass
        self._description = description

    def validate(self, *args, **kwargs) -> RuleResult:
        if self._should_pass:
            return RuleResult(is_valid=True)
        return RuleResult(is_valid=False, message="Validation failed")

    def description(self) -> str:
        return self._description


class TestRuleResult:
    def test_valid_result(self):
        result = RuleResult(is_valid=True)
        assert result.is_valid is True
        assert result.message == ""

    def test_invalid_result_with_message(self):
        result = RuleResult(is_valid=False, message="Error occurred")
        assert result.is_valid is False
        assert result.message == "Error occurred"

    def test_frozen_dataclass(self):
        result = RuleResult(is_valid=True)
        with pytest.raises(Exception):
            result.is_valid = False


class TestRuleChain:
    def test_empty_chain_passes(self):
        chain = RuleChain()
        all_valid, results = chain.validate()
        assert all_valid is True
        assert results == []

    def test_single_rule_pass(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=True))
        all_valid, results = chain.validate()
        assert all_valid is True
        assert len(results) == 1
        assert results[0].is_valid is True

    def test_single_rule_fail(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=False))
        all_valid, results = chain.validate()
        assert all_valid is False
        assert len(results) == 1
        assert results[0].is_valid is False

    def test_multiple_rules_all_pass(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=True, description="Rule 1"))
        chain.add(ConcreteRule(should_pass=True, description="Rule 2"))
        chain.add(ConcreteRule(should_pass=True, description="Rule 3"))
        all_valid, results = chain.validate()
        assert all_valid is True
        assert len(results) == 3

    def test_short_circuit_on_first_failure(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=False, description="Rule 1"))
        chain.add(ConcreteRule(should_pass=True, description="Rule 2"))
        chain.add(ConcreteRule(should_pass=True, description="Rule 3"))
        all_valid, results = chain.validate()
        assert all_valid is False
        assert len(results) == 1
        assert results[0].is_valid is False

    def test_short_circuit_on_middle_failure(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=True, description="Rule 1"))
        chain.add(ConcreteRule(should_pass=False, description="Rule 2"))
        chain.add(ConcreteRule(should_pass=True, description="Rule 3"))
        all_valid, results = chain.validate()
        assert all_valid is False
        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False

    def test_add_returns_self_for_chaining(self):
        chain = RuleChain()
        result = chain.add(ConcreteRule(should_pass=True))
        assert result is chain

    def test_chained_add(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=True)).add(ConcreteRule(should_pass=True))
        all_valid, results = chain.validate()
        assert all_valid is True
        assert len(results) == 2

    def test_describe(self):
        chain = RuleChain()
        chain.add(ConcreteRule(should_pass=True, description="Rule A"))
        chain.add(ConcreteRule(should_pass=True, description="Rule B"))
        descriptions = chain.describe()
        assert descriptions == ["Rule A", "Rule B"]

    def test_validate_with_args(self):
        class ArgCapturingRule(Rule):
            def __init__(self):
                self.captured_args = None
                self.captured_kwargs = None

            def validate(self, *args, **kwargs) -> RuleResult:
                self.captured_args = args
                self.captured_kwargs = kwargs
                return RuleResult(is_valid=True)

            def description(self) -> str:
                return "Arg Rule"

        rule = ArgCapturingRule()
        chain = RuleChain()
        chain.add(rule)
        chain.validate("arg1", "arg2", key1="value1", key2="value2")
        assert rule.captured_args == ("arg1", "arg2")
        assert rule.captured_kwargs == {"key1": "value1", "key2": "value2"}

    def test_init_with_rules(self):
        rules = [
            ConcreteRule(should_pass=True),
            ConcreteRule(should_pass=True),
        ]
        chain = RuleChain(rules=rules)
        all_valid, results = chain.validate()
        assert all_valid is True
        assert len(results) == 2
