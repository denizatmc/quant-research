"""A minimal FIX 4.2 encoder/decoder.

FIX (Financial Information eXchange) is the lingua franca of electronic order routing — it's
how an order gets from a buy-side system to a broker or exchange (BIST included). A FIX
message is just a flat list of `tag=value` fields joined by the SOH (0x01) control byte,
wrapped by a header (tags 8/9/35) and a trailer (tag 10, a checksum).

I implemented the parts that show the protocol is understood rather than pulling in a full
engine (QuickFIX et al.): correct field ordering, the BodyLength (tag 9) and CheckSum (tag
10) computations — which are the two things people get subtly wrong — and a NewOrderSingle
(MsgType=D) builder plus a parser/validator. No session layer or TCP; this is the message
codec, which is the interesting part.
"""

from __future__ import annotations

from dataclasses import dataclass

# The real delimiter is the non-printable SOH (0x01). We keep it as the true byte for
# correctness but expose a pipe-rendering for human-readable logs/tests.
SOH = "\x01"

# A small slice of the FIX 4.2 data dictionary — enough for an order round-trip.
TAG_NAMES = {
    8: "BeginString",
    9: "BodyLength",
    35: "MsgType",
    34: "MsgSeqNum",
    49: "SenderCompID",
    56: "TargetCompID",
    52: "SendingTime",
    11: "ClOrdID",
    55: "Symbol",
    54: "Side",          # 1=Buy, 2=Sell
    38: "OrderQty",
    40: "OrdType",       # 1=Market, 2=Limit
    44: "Price",
    59: "TimeInForce",   # 0=Day, 1=GTC, 3=IOC
    10: "CheckSum",
}

SIDE = {"BUY": "1", "SELL": "2"}
ORD_TYPE = {"MARKET": "1", "LIMIT": "2"}


def _checksum(body: str) -> str:
    """FIX checksum: sum of every byte mod 256, zero-padded to three digits.

    It's computed over the whole message *up to and including* the delimiter before tag 10,
    which is why the builder appends it last."""
    total = sum(ord(c) for c in body) % 256
    return f"{total:03d}"


def _field(tag: int, value) -> str:
    return f"{tag}={value}{SOH}"


@dataclass
class FixMessage:
    fields: list[tuple[int, str]]

    def get(self, tag: int) -> str | None:
        for t, v in self.fields:
            if t == tag:
                return v
        return None

    def to_wire(self) -> str:
        return "".join(_field(t, v) for t, v in self.fields)

    def pretty(self) -> str:
        """Human-readable rendering: SOH shown as '|' and tags annotated with their names."""
        parts = []
        for t, v in self.fields:
            name = TAG_NAMES.get(t, "?")
            parts.append(f"{t}({name})={v}")
        return " | ".join(parts)


def new_order_single(
    sender: str,
    target: str,
    seq_num: int,
    cl_ord_id: str,
    symbol: str,
    side: str,
    qty: float,
    ord_type: str = "MARKET",
    price: float | None = None,
    sending_time: str = "20240101-00:00:00.000",
    tif: str = "0",
) -> FixMessage:
    """Build a NewOrderSingle (35=D), with BodyLength and CheckSum computed correctly.

    The two derived fields are the fiddly part: BodyLength (9) counts the bytes *after* tag 9
    up to and including the delimiter before the checksum, and CheckSum (10) is taken over
    everything before it. Get either wrong and the counterparty rejects the message, so the
    order of operations here matters.
    """
    if ord_type.upper() == "LIMIT" and price is None:
        raise ValueError("A limit order needs a price.")

    # Body = everything between the header's BeginString/BodyLength and the checksum.
    body_fields: list[tuple[int, str]] = [
        (35, "D"),
        (34, str(seq_num)),
        (49, sender),
        (56, target),
        (52, sending_time),
        (11, cl_ord_id),
        (55, symbol),
        (54, SIDE[side.upper()]),
        (38, f"{qty:g}"),
        (40, ORD_TYPE[ord_type.upper()]),
        (59, tif),
    ]
    if price is not None:
        body_fields.insert(-1, (44, f"{price:g}"))

    body_str = "".join(_field(t, v) for t, v in body_fields)
    begin_string = "FIX.4.2"
    body_length = len(body_str)  # in bytes; SOH counts as one byte each, which it is

    header = [(8, begin_string), (9, str(body_length))]
    # Checksum is over header(8,9) + body, then appended as tag 10.
    pre_checksum = "".join(_field(t, v) for t, v in header) + body_str
    checksum = _checksum(pre_checksum)

    fields = header + body_fields + [(10, checksum)]
    return FixMessage(fields=fields)


def parse_fix(raw: str) -> FixMessage:
    """Parse a raw FIX string (SOH- or pipe-delimited) and validate its checksum.

    Accepting '|' as well as the real SOH is a small convenience so logged messages can be
    fed straight back in. Raises if the checksum doesn't reconcile — that's the whole point
    of having one.
    """
    delim = SOH if SOH in raw else "|"
    parts = [p for p in raw.split(delim) if p]
    fields: list[tuple[int, str]] = []
    for p in parts:
        tag_str, _, value = p.partition("=")
        fields.append((int(tag_str), value))

    msg = FixMessage(fields=fields)
    stated = msg.get(10)
    if stated is not None:
        # Recompute over everything before tag 10 and compare.
        body_upto_checksum = "".join(_field(t, v) for t, v in fields if t != 10)
        if _checksum(body_upto_checksum) != stated:
            raise ValueError(
                f"FIX checksum mismatch: stated {stated}, computed {_checksum(body_upto_checksum)}"
            )
    return msg
