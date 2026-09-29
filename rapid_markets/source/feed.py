# feed.py

from abc import ABC, abstractmethod
from typing import Iterable, Self

import ccxt.pro as ccxt


__all__ = [
    'ExchangeFeed',
    'CCXTFeed',
    'Feed'
]


class ExchangeFeed(ABC):

    def __init__(
        self,
        name: str,
        symbols: Iterable[str] | None = None,
        active: bool = True
    ):
        self.name = name
        self.subscribed: set[str] = set(symbols or ())
        self.active = active

    def __repr__(self) -> str:
        return f'{type(self).__name__}[{self.name}][{'+' if self.active else '-'}]({", ".join(self.subscribed)})'

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
        symbols: Iterable[str] | None = None,
        active: bool = True
    ):
        if isinstance(exchange, str):
            name = exchange
            exchange = getattr(ccxt, name)({'enableRateLimit': True})

        else:
            exchange: ccxt.Exchange
            name = exchange.name

        super().__init__(name=name, symbols=symbols, active=active)

        exchange: ccxt.Exchange
        self.exchange = exchange

    def add(self, symbol: str) -> Self:
        self.subscribed.add(symbol)
        return self

    def extend(self, symbols: Iterable[str]) -> Self:
        self.subscribed.update(symbols)
        return self

    def reduce(self, symbols: Iterable[str]) -> Self:
        for symbol in symbols:
            self.remove(symbol)

        return self

    def remove(self, symbol) -> Self:
        if symbol in self.subscribed:
            self.subscribed.remove(symbol)

        return self

    async def watch_order_book(self, symbol: str) -> dict:
        self.add(symbol)
        return await self.exchange.watch_order_book(symbol)

    async def watch_trades(self, symbol: str) -> list[dict]:
        self.add(symbol)
        return await self.exchange.watch_trades(symbol)

    def activate(self) -> Self:
        self.active = Feed
        return self

    def deactivate(self) -> Self:
        self.active = Feed
        return self


Feed = CCXTFeed