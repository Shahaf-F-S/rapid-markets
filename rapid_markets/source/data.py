# data.py

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from rapid_markets.base import labels


__all__ = [
    'Book',
    'Trade'
]


@dataclass(slots=True, frozen=True, repr=False)
class Book:

    timestamp: dt.datetime
    exchange: str
    symbol: str

    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}[{self.exchange}: {self.symbol}]("
            f"timestamp: {self.timestamp}, "
            f"depth: {self.levels}, "
            f"bid={(self.bid_price(), self.bid_quantity())}, "
            f"ask={(self.ask_price(), self.ask_quantity())}"
            f")"
        )

    def bid_price(self, index: int = 0) -> float:
        return self.bids[index][0]

    def bid_quantity(self, index: int = 0) -> float:
        return self.bids[index][1]

    def ask_price(self, index: int = 0) -> float:
        return self.asks[index][0]

    def ask_quantity(self, index: int = 0) -> float:
        return self.asks[index][1]

    @property
    def levels(self) -> int:
        return min((len(self.asks), len(self.bids)))

    def data(self, limit: int | None = 10, index: int = 0) -> dict[str, ...]:
        limit = limit or self.levels
        return {
            labels.TIMESTAMP: self.timestamp,
            labels.EXCHANGE: self.exchange,
            labels.SYMBOL: self.symbol,
            labels.BID_PRICE: self.bid_price(index),
            labels.BID_QUANTITY: self.bid_quantity(index),
            labels.ASK_PRICE: self.ask_price(index),
            labels.ASK_QUANTITY: self.ask_quantity(index),
            labels.BIDS: self.bids[:limit],
            labels.ASKS: self.asks[:limit]
        }

    @classmethod
    def load(cls, data: dict[str, ...]) -> Book:
        data = data.copy()

        if isinstance(data[labels.TIMESTAMP], str):
            data[labels.TIMESTAMP] = dt.datetime.fromisoformat(data[labels.TIMESTAMP])

        elif isinstance(data[labels.TIMESTAMP], (int, float)):
            data[labels.TIMESTAMP] = dt.datetime.fromtimestamp(data[labels.TIMESTAMP], dt.UTC)

        data.pop(labels.BID_PRICE, None)
        data.pop(labels.ASK_PRICE, None)
        data.pop(labels.BID_QUANTITY, None)
        data.pop(labels.ASK_QUANTITY, None)
        return Book(**data)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else float("nan")

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("nan")

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        return self.spread / self.mid * 1e4

    def imbalance(self, levels: int = 1) -> float:
        """
        Order-book imbalance over the top `levels` price levels.

        OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        Range: [-1, 1].  Positive → buy pressure; negative → sell pressure.
        """
        bid_qty = sum(q for _, q in self.bids[:levels])
        ask_qty = sum(q for _, q in self.asks[:levels])
        denom = bid_qty + ask_qty
        return (bid_qty - ask_qty) / denom if denom else 0.0

    def weighted_mid(self, level: int = 0) -> float:
        """
        Stoikov's imbalance-adjusted mid price.
        Sits between best bid and best ask, weighted by queue sizes.
        Better short-term predictor than the raw mid.
        """
        bq = self.bids[level][1] if self.bids else 0.0
        aq = self.asks[level][1] if self.asks else 0.0
        denom = bq + aq

        if not denom:
            return self.mid

        return self.best_ask * (bq / denom) + self.best_bid * (aq / denom)

    def depth(self, levels: int = 5) -> tuple[float, float]:
        """Total bid and ask notional over top `levels` levels."""
        bid_depth = sum(p * q for p, q in self.bids[:levels])
        ask_depth = sum(p * q for p, q in self.asks[:levels])
        return bid_depth, ask_depth


@dataclass(slots=True, frozen=True)
class Trade:

    timestamp: dt.datetime
    exchange: str
    symbol: str

    price: float
    quantity: float
    side: Literal[labels.BUY, labels.SELL]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}[{self.exchange}: {self.symbol}]("
            f"timestamp: {self.timestamp}, "
            f"{self.side.lower()}: ({self.price}, {self.quantity})"
            f")"
        )

    def data(self, *args, **kwargs) -> dict[str, float | str | dt.datetime]:
        id(args)
        id(kwargs)
        return {
            labels.TIMESTAMP: self.timestamp,
            labels.EXCHANGE: self.exchange,
            labels.SYMBOL: self.symbol,
            labels.PRICE: self.price,
            labels.QUANTITY: self.quantity,
            labels.SIDE: self.side
        }

    @classmethod
    def load(cls, data: dict[str, ...]) -> Trade:
        return Trade(**data)

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == labels.BUY else -self.quantity
