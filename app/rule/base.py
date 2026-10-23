"""规则基类和规则链实现。

本模块提供规则验证系统的基础组件：
- RuleResult: 规则验证的不可变结果
- Rule: 所有规则的抽象基类（策略模式）
- RuleChain: 职责链模式，用于顺序执行规则

使用示例：
    # 单个规则
    rule = LotRule(rules)
    result = rule.validate(symbol="600000.SH", quantity=100)

    # 规则链
    chain = RuleChain([LotRule(rules), PriceLimitRule(rules, ...)])
    all_valid, results = chain.validate(symbol="600000.SH", quantity=100)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RuleResult:
    """规则验证的不可变结果。

    属性:
        is_valid: 验证是否通过
        message: 验证失败时的错误消息，通过时为空
    """

    is_valid: bool
    message: str = ""


class Rule(ABC):
    """所有规则的抽象基类。

    实现此类以创建具体的验证规则。
    为获得最佳性能，每个规则应保持无状态。
    """

    @abstractmethod
    def validate(self, *args, **kwargs) -> RuleResult:
        """执行规则验证。

        返回:
            验证通过的 RuleResult(is_valid=True)，否则返回 is_valid=False
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """人类可读的规则描述，用于日志和调试。"""
        pass


class RuleChain:
    """规则验证的职责链模式。

    规则按顺序执行，遇到失败则停止。
    短路求值：首次失败即停止。

    属性:
        _rules: 按执行顺序排列的规则列表
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = rules or []

    def add(self, rule: Rule) -> "RuleChain":
        """向链中添加规则。

        返回 self 以支持方法链式调用。

        参数:
            rule: 要添加的规则

        返回:
            self，用于链式调用
        """
        self._rules.append(rule)
        return self

    def validate(self, *args, **kwargs) -> tuple[bool, list[RuleResult]]:
        """对链中的所有规则进行验证。

        顺序执行规则，遇到失败则短路。

        参数:
            *args: 传递给每个规则 validate() 的位置参数
            **kwargs: 传递给每个规则 validate() 的关键字参数

        返回:
            (all_valid, results) 元组：
            - all_valid: 所有规则是否都通过
            - results: 按执行顺序排列的 RuleResult 列表
        """
        results: list[RuleResult] = []
        for rule in self._rules:
            result = rule.validate(*args, **kwargs)
            results.append(result)
            if not result.is_valid:
                return False, results
        return True, results

    def describe(self) -> list[str]:
        """获取链中所有规则的描述。

        返回:
            按链顺序排列的人类可读描述列表
        """
        return [rule.description() for rule in self._rules]
