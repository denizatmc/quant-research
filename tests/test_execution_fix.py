"""FIX codec tests: the checksum and body-length must be exactly right, or it's not FIX."""

import pytest

from quantlab.execution.fix import new_order_single, parse_fix


def test_roundtrip_preserves_fields():
    msg = new_order_single("ME", "YOU", 7, "abc", "MSFT", "SELL", 500, "LIMIT", price=410.25)
    parsed = parse_fix(msg.to_wire())
    assert parsed.get(55) == "MSFT"
    assert parsed.get(54) == "2"          # SELL
    assert parsed.get(38) == "500"
    assert parsed.get(44) == "410.25"


def test_checksum_validates():
    msg = new_order_single("ME", "YOU", 1, "o1", "AAPL", "BUY", 100)
    # parse_fix recomputes and compares the checksum; a clean message must pass.
    parse_fix(msg.to_wire())


def test_tampered_message_rejected():
    msg = new_order_single("ME", "YOU", 1, "o1", "AAPL", "BUY", 100)
    wire = msg.to_wire().replace("55=AAPL", "55=TSLA")  # alter the symbol, leave checksum stale
    with pytest.raises(ValueError, match="checksum mismatch"):
        parse_fix(wire)


def test_limit_requires_price():
    with pytest.raises(ValueError):
        new_order_single("ME", "YOU", 1, "o1", "AAPL", "BUY", 100, "LIMIT", price=None)
