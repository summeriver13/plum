"""Tests for A-share trading rules."""

from app.rule.trading.calculator import ASHARE_RULES
from app.rule.trading.board import (
    Board,
    _infer_board_from_code,
    get_board,
    get_board_config,
)
from app.rule.trading.validators import (
    LotRule,
    MarketOrderRule,
    OrderQuantityRule,
    PriceCageRule,
    PriceLimitRule,
    PriceTickRule,
)


class TestBoardDetection:
    def test_star_board(self):
        assert get_board("688000.SH") == Board.STAR
        assert get_board("689999.SH") == Board.STAR
        assert get_board("688001") == Board.STAR

    def test_bse_board(self):
        assert get_board("430000.BJ") == Board.BSE
        assert get_board("830000.BJ") == Board.BSE
        assert get_board("899999.BJ") == Board.BSE

    def test_gem_board(self):
        assert get_board("300000.SZ") == Board.GEM
        assert get_board("301999.SZ") == Board.GEM

    def test_main_board(self):
        assert get_board("600000.SH") == Board.MAIN
        assert get_board("000001.SZ") == Board.MAIN
        assert get_board("601000.SH") == Board.MAIN

    def test_infer_board_from_code_non_digit(self):
        assert _infer_board_from_code("SH") == Board.MAIN

    def test_get_board_with_missing_exchange(self):
        assert get_board("688000") == Board.STAR
        assert get_board("300000") == Board.GEM


class TestBoardConfig:
    def test_main_board_config(self):
        config = get_board_config("600000.SH")
        assert config.lot_size == 100
        assert config.min_order == 100
        assert config.max_limit_order == 1_000_000
        assert config.max_market_order == 50_000
        assert config.supports_odd_lot is False
        assert config.price_limit_pct == 0.1

    def test_gem_board_config(self):
        config = get_board_config("300000.SZ")
        assert config.lot_size == 100
        assert config.price_limit_pct == 0.2

    def test_star_board_config(self):
        config = get_board_config("688000.SH")
        assert config.lot_size == 200
        assert config.supports_odd_lot is True
        assert config.price_limit_pct == 0.2

    def test_bse_board_config(self):
        config = get_board_config("430000.BJ")
        assert config.lot_size == 100
        assert config.supports_odd_lot is True
        assert config.price_limit_pct == 0.3


class TestTradingRulesLot:
    def test_floor_to_lot_main_board(self):
        assert ASHARE_RULES.floor_to_lot("600000.SH", 150) == 100
        assert ASHARE_RULES.floor_to_lot("600000.SH", 250) == 200
        assert ASHARE_RULES.floor_to_lot("600000.SH", 99) == 0

    def test_floor_to_lot_star_board(self):
        assert ASHARE_RULES.floor_to_lot("688000.SH", 250) == 250
        assert ASHARE_RULES.floor_to_lot("688000.SH", 199) == 199

    def test_is_valid_lot_main_board(self):
        assert ASHARE_RULES.is_valid_lot("600000.SH", 100) is True
        assert ASHARE_RULES.is_valid_lot("600000.SH", 200) is True
        assert ASHARE_RULES.is_valid_lot("600000.SH", 50) is False

    def test_is_valid_lot_star_board(self):
        assert ASHARE_RULES.is_valid_lot("688000.SH", 200) is True
        assert ASHARE_RULES.is_valid_lot("688000.SH", 100) is False


class TestTradingRulesOrderValidation:
    def test_valid_order_main_board(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", 100)
        assert is_valid is True
        assert msg == ""

    def test_zero_quantity(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", 0)
        assert is_valid is False
        assert "positive" in msg

    def test_negative_quantity(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", -100)
        assert is_valid is False

    def test_exceeds_max_limit_order(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", 1_000_001)
        assert is_valid is False
        assert "exceeds maximum" in msg

    def test_exceeds_max_market_order(self):
        is_valid, msg = ASHARE_RULES.validate_order(
            "600000.SH", 50_001, is_market_order=True
        )
        assert is_valid is False

    def test_below_min_order(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", 50)
        assert is_valid is False
        assert "below minimum" in msg

    def test_odd_lot_main_board(self):
        is_valid, msg = ASHARE_RULES.validate_order("600000.SH", 150)
        assert is_valid is False
        assert "multiple of lot size" in msg

    def test_odd_lot_star_board(self):
        is_valid, msg = ASHARE_RULES.validate_order("688000.SH", 250)
        assert is_valid is True

    def test_sell_all_holding(self):
        is_valid, msg = ASHARE_RULES.validate_order(
            "600000.SH", 150, is_selling=True, current_holding=150
        )
        assert is_valid is True


class TestTradingRulesRoundOrder:
    def test_round_down_main_board(self):
        result = ASHARE_RULES.round_order("600000.SH", 150)
        assert result == 100

    def test_round_down_star_board(self):
        result = ASHARE_RULES.round_order("688000.SH", 250)
        assert result == 250

    def test_round_below_min(self):
        result = ASHARE_RULES.round_order("600000.SH", 50)
        assert result == 0

    def test_round_clip_to_max(self):
        result = ASHARE_RULES.round_order("600000.SH", 2_000_000)
        assert result == 1_000_000

    def test_round_sell_all(self):
        result = ASHARE_RULES.round_order(
            "600000.SH", 150, is_selling=True, current_holding=150
        )
        assert result == 150


class TestTradingRulesPrice:
    def test_get_price_tick(self):
        assert ASHARE_RULES.get_price_tick() == 100
        assert ASHARE_RULES.get_price_tick("600000.SH") == 100

    def test_round_price(self):
        assert ASHARE_RULES.round_price(105500) == 105500
        assert ASHARE_RULES.round_price(105550) == 105500
        assert ASHARE_RULES.round_price(105599) == 105500

    def test_get_min_order_value(self):
        value = ASHARE_RULES.get_min_order_value("600000.SH")
        assert value == 100 * 100


class TestPriceLimit:
    def test_main_board_limit(self):
        limit = ASHARE_RULES.get_price_limit("600000.SH", 100000)
        assert limit.low == 90000
        assert limit.high == 110000

    def test_main_board_st_limit(self):
        limit = ASHARE_RULES.get_price_limit("600000.SH", 100000, is_st=True)
        assert limit.low == 95000
        assert limit.high == 105000

    def test_gem_board_limit(self):
        limit = ASHARE_RULES.get_price_limit("300000.SZ", 100000)
        assert limit.low == 80000
        assert limit.high == 120000

    def test_star_board_limit(self):
        limit = ASHARE_RULES.get_price_limit("688000.SH", 100000)
        assert limit.low == 80000
        assert limit.high == 120000

    def test_new_listing_main_board(self):
        limit = ASHARE_RULES.get_price_limit("600000.SH", 100000, is_new_listing=True)
        assert limit.low == 56000
        assert limit.high == 144000


class TestPriceCage:
    def test_main_board_price_cage(self):
        cage = ASHARE_RULES.get_price_cage("600000.SH", 100000)
        assert cage.low == 98000
        assert cage.high == 102000

    def test_star_board_price_cage(self):
        cage = ASHARE_RULES.get_price_cage("688000.SH", 100000)
        assert cage.low == 97000
        assert cage.high == 102000

    def test_bse_board_price_cage(self):
        cage = ASHARE_RULES.get_price_cage("430000.BJ", 100000)
        assert cage.low == 95000
        assert cage.high == 105000

    def test_price_cage_auction_time(self):
        cage = ASHARE_RULES.get_price_cage(
            "600000.SH", 100000, trading_time="opening_auction"
        )
        assert cage.low == 0
        assert cage.high == int(1e9)


class TestValidatePrice:
    def test_valid_price(self):
        is_valid, msg = ASHARE_RULES.validate_price("600000.SH", 100000, 100000)
        assert is_valid is True

    def test_price_above_limit(self):
        is_valid, msg = ASHARE_RULES.validate_price("600000.SH", 120000, 100000)
        assert is_valid is False
        assert "高于涨停价" in msg

    def test_price_below_limit(self):
        is_valid, msg = ASHARE_RULES.validate_price("600000.SH", 80000, 100000)
        assert is_valid is False
        assert "低于跌停价" in msg

    def test_price_cage_violation_star(self):
        is_valid, msg = ASHARE_RULES.validate_price(
            "688000.SH", 103000, 100000, ref_price=100000
        )
        assert is_valid is False
        assert "价格笼子" in msg


class TestMarketOrder:
    def test_valid_market_order(self):
        is_valid, msg = ASHARE_RULES.validate_market_order(
            "600000.SH", 1000, "BUY", 100000
        )
        assert is_valid is True

    def test_market_order_exceeds_limit(self):
        is_valid, msg = ASHARE_RULES.validate_market_order(
            "600000.SH", 60000, "BUY", 100000
        )
        assert is_valid is False
        assert "exceeds market order limit" in msg

    def test_market_order_negative_quantity(self):
        is_valid, msg = ASHARE_RULES.validate_market_order(
            "600000.SH", 0, "BUY", 100000
        )
        assert is_valid is False
        assert "quantity must be positive" in msg


class TestRuleClasses:
    def test_lot_rule_pass(self):
        rule = LotRule(ASHARE_RULES)
        result = rule.validate("600000.SH", 100)
        assert result.is_valid is True

    def test_lot_rule_fail(self):
        rule = LotRule(ASHARE_RULES)
        result = rule.validate("600000.SH", 150)
        assert result.is_valid is False

    def test_lot_rule_star_board_pass(self):
        rule = LotRule(ASHARE_RULES)
        result = rule.validate("688000.SH", 250)
        assert result.is_valid is True

    def test_order_quantity_rule_pass(self):
        rule = OrderQuantityRule(ASHARE_RULES, symbol="600000.SH")
        result = rule.validate("600000.SH", 100)
        assert result.is_valid is True

    def test_order_quantity_rule_fail(self):
        rule = OrderQuantityRule(ASHARE_RULES, symbol="600000.SH")
        result = rule.validate("600000.SH", 0)
        assert result.is_valid is False

    def test_price_tick_rule_pass(self):
        rule = PriceTickRule(ASHARE_RULES)
        result = rule.validate(100000)
        assert result.is_valid is True

    def test_price_tick_rule_fail(self):
        rule = PriceTickRule(ASHARE_RULES)
        result = rule.validate(100050)
        assert result.is_valid is False

    def test_price_limit_rule_pass(self):
        rule = PriceLimitRule(ASHARE_RULES, symbol="600000.SH", pre_close=100000)
        result = rule.validate(100000)
        assert result.is_valid is True

    def test_price_limit_rule_fail(self):
        rule = PriceLimitRule(ASHARE_RULES, symbol="600000.SH", pre_close=100000)
        result = rule.validate(200000)
        assert result.is_valid is False

    def test_price_cage_rule_pass(self):
        rule = PriceCageRule(ASHARE_RULES, symbol="688000.SH", ref_price=100000)
        result = rule.validate(100000)
        assert result.is_valid is True

    def test_price_cage_rule_fail(self):
        rule = PriceCageRule(ASHARE_RULES, symbol="688000.SH", ref_price=100000)
        result = rule.validate(103000)
        assert result.is_valid is False

    def test_market_order_rule_pass(self):
        rule = MarketOrderRule(ASHARE_RULES)
        result = rule.validate("600000.SH", 1000)
        assert result.is_valid is True

    def test_market_order_rule_fail(self):
        rule = MarketOrderRule(ASHARE_RULES)
        result = rule.validate("600000.SH", 60000)
        assert result.is_valid is False

    def test_rule_descriptions(self):
        assert "整手" in LotRule(ASHARE_RULES).description()
        assert (
            "范围" in OrderQuantityRule(ASHARE_RULES, symbol="600000.SH").description()
        )
        assert "跳动" in PriceTickRule(ASHARE_RULES).description()
        assert (
            "涨跌停"
            in PriceLimitRule(
                ASHARE_RULES, symbol="600000.SH", pre_close=100000
            ).description()
        )
        assert (
            "笼子"
            in PriceCageRule(
                ASHARE_RULES, symbol="688000.SH", ref_price=100000
            ).description()
        )
        assert "市价单" in MarketOrderRule(ASHARE_RULES).description()


class TestGetValidPriceRange:
    def test_without_ref_price(self):
        result = ASHARE_RULES.get_valid_price_range("600000.SH", 100000)
        assert result.low == 90000
        assert result.high == 110000

    def test_with_ref_price_main_board(self):
        result = ASHARE_RULES.get_valid_price_range(
            "600000.SH", 100000, ref_price=100000
        )
        assert result.low == 98000
        assert result.high == 102000

    def test_with_ref_price_star_board(self):
        result = ASHARE_RULES.get_valid_price_range(
            "688000.SH", 100000, ref_price=100000
        )
        assert result.low == 97000
        assert result.high == 102000
