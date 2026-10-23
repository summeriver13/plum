"""Tests for chain factory module."""

from app.rule.chain import create_order_validation_chain, create_price_validation_chain
from app.rule.trading.calculator import ASHARE_RULES


class TestCreateOrderValidationChain:
    def test_basic_order_chain(self):
        chain = create_order_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            quantity=100,
        )
        all_valid, results = chain.validate(symbol="600000.SH", quantity=100)
        assert all_valid is True
        assert len(results) == 2

    def test_market_order_chain(self):
        chain = create_order_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            quantity=1000,
            is_market_order=True,
        )
        all_valid, results = chain.validate(symbol="600000.SH", quantity=1000)
        assert all_valid is True
        assert len(results) == 3

    def test_invalid_quantity_chain(self):
        chain = create_order_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            quantity=50,
        )
        all_valid, results = chain.validate(symbol="600000.SH", quantity=50)
        assert all_valid is False

    def test_selling_chain(self):
        chain = create_order_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            quantity=100,
            is_selling=True,
            current_holding=100,
        )
        all_valid, results = chain.validate(symbol="600000.SH", quantity=100)
        assert all_valid is True


class TestCreatePriceValidationChain:
    def test_basic_price_chain(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=100000,
            pre_close=100000,
        )
        all_valid, results = chain.validate(price=100000)
        assert all_valid is True
        assert len(results) == 2

    def test_price_chain_with_cage(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="688000.SH",
            price=100000,
            pre_close=100000,
            ref_price=100000,
        )
        all_valid, results = chain.validate(price=100000)
        assert all_valid is True
        assert len(results) == 3

    def test_price_chain_above_limit(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=200000,
            pre_close=100000,
        )
        all_valid, results = chain.validate(price=200000)
        assert all_valid is False

    def test_price_chain_st_stock(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=105000,
            pre_close=100000,
            is_st=True,
        )
        all_valid, results = chain.validate(price=105000)
        assert all_valid is True

    def test_price_chain_st_stock_exceed(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=106000,
            pre_close=100000,
            is_st=True,
        )
        all_valid, results = chain.validate(price=106000)
        assert all_valid is False

    def test_price_chain_new_listing(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=140000,
            pre_close=100000,
            is_new_listing=True,
        )
        all_valid, results = chain.validate(price=140000)
        assert all_valid is True

    def test_price_chain_invalid_tick(self):
        chain = create_price_validation_chain(
            rules=ASHARE_RULES,
            symbol="600000.SH",
            price=100050,
            pre_close=100000,
        )
        all_valid, results = chain.validate(price=100050)
        assert all_valid is False
