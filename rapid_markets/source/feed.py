# feed.py

import datetime as dt
from abc import ABC, abstractmethod
from typing import Iterable, Self, Callable

import ccxt.pro as ccxt

from rapid_markets.source import Book, Trade


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
        active: bool = True,
        sleep: float = 0.001
    ):
        self.name = name
        self.sleep = sleep

        self._active = active
        self._subscribed: set[str] = set(symbols or ())
        self._listeners: list[Callable[[Self], None]] = []

    def __repr__(self) -> str:
        return f'{type(self).__name__}[{self.name}][{'+' if self.active else '-'}]({", ".join(self.subscribed)})'

    @abstractmethod
    async def books(self, symbol: str) -> list[Book]:
        ...

    @abstractmethod
    async def trades(self, symbol: str) -> list[Trade]:
        ...

    def listen(self, listener: Callable[[Self], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unlisten(self, listener: Callable[[Self], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            listener(self)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self._notify()

    @property
    def subscribed(self) -> frozenset[str]:
        return frozenset(self._subscribed)

    def add(self, symbol: str) -> Self:
        return self.extend((symbol,))

    def extend(self, symbols: Iterable[str]) -> Self:
        if new := set(symbols) - self._subscribed:
            self._subscribed |= new
            self._notify()

        return self

    def remove(self, symbol: str) -> Self:
        return self.reduce((symbol,))

    def reduce(self, symbols: Iterable[str]) -> Self:
        if gone := self._subscribed & set(symbols):
            self._subscribed -= gone
            self._notify()

        return self

    def activate(self) -> Self:
        self.active = True
        return self

    def deactivate(self) -> Self:
        self.active = False
        return self


class CCXTFeed(ExchangeFeed):

    def __init__(
        self,
        exchange: str | ccxt.Exchange | None = None,
        symbols: Iterable[str] | None = None,
        active: bool = True,
        sleep: float = 0.001
    ):
        if isinstance(exchange, str):
            name = exchange
            exchange = getattr(ccxt, name)({'enableRateLimit': True})

        else:
            exchange: ccxt.Exchange
            name = exchange.name

        super().__init__(
            name=name, symbols=symbols, active=active, sleep=sleep
        )

        exchange: ccxt.Exchange
        self.exchange = exchange

    async def books(self, symbol: str) -> list[Book]:
        self.add(symbol)

        orderbook = await self.exchange.watch_order_book(symbol)

        bids, asks = orderbook['bids'], orderbook['asks']

        books = []

        if bids and asks:
            book = Book(
                timestamp=dt.datetime.now(dt.UTC),
                exchange=self.name, symbol=symbol,
                bids=bids, asks=asks
            )
            books.append(book)

        return books

    async def trades(self, symbol: str) -> list[Trade]:
        self.add(symbol)

        raw_trades = await self.exchange.watch_trades(symbol)

        trades = []

        for trade_data in raw_trades:
            trade = Trade(
                timestamp=dt.datetime.now(dt.UTC),
                exchange=self.name, symbol=symbol,
                price=trade_data['price'], quantity=trade_data['amount'],
                side=trade_data['side'].capitalize()
            )
            trades.append(trade)

        return trades


Feed = CCXTFeed