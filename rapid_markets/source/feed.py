# feed.py

from abc import ABC, abstractmethod
from typing import Iterable

import ccxt.pro as ccxt


__all__ = [
    'ExchangeFeed',
    'CCXTFeed'
]


class ExchangeFeed(ABC):

    def __init__(self, name: str, symbols: Iterable[str] | None = None):
        self.name = name
        self.subscribed: set[str] = set(symbols or ())

    @abstractmethod
    async def watch_order_book(self, symbol: str) -> dict:
        ...

    @abstractmethod
    async def watch_trades(self, symbol: str) -> list[dict]:
        ...


class CCXTFeed(ExchangeFeed):

    def __init__(
        self,
        exchange: str | ccxt.Exchange | None = None,
        symbols: Iterable[str] | None = None
    ):
        if isinstance(exchange, str):
            name = exchange
            exchange = getattr(ccxt, name)({'enableRateLimit': True})

        else:
            exchange: ccxt.Exchange
            name = exchange.name

        exchange: ccxt.Exchange
        super().__init__(name=name, symbols=symbols)
        self.exchange = exchange

    async def watch_order_book(self, symbol: str) -> dict:
        self.subscribed.add(symbol)
        return await self.exchange.watch_order_book(symbol)

    async def watch_trades(self, symbol: str) -> list[dict]:
        self.subscribed.add(symbol)
        return await self.exchange.watch_trades(symbol)
