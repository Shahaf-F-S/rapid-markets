# record.py

import datetime as dt
from typing import Iterable, AsyncGenerator
import warnings
from aiostream import stream

from ccxt.base.errors import NetworkError

from rapid_markets.base.control import Control
from rapid_markets.source.data import Book, Trade
from rapid_markets.source.feed import ExchangeFeed


__all__ = [
    "watch_trades",
    "watch_books",
    "watch_symbol_trades",
    "watch_symbol_book",
    "Control",
    "NetworkError",
    "watch_data"
]


async def watch_symbol_book(
    exchange: ExchangeFeed, symbol: str, control: Control
) -> AsyncGenerator[Book, None, None]:
    while control.run():
        try:
            orderbook = await exchange.watch_order_book(symbol)

            bids, asks = orderbook['bids'], orderbook['asks']
            if bids and asks:
                yield Book(
                    timestamp=dt.datetime.now(dt.UTC),
                    exchange=exchange.name, symbol=symbol,
                    bids=bids, asks=asks
                )

        except Exception as e:
            warnings.warn(
                f"[{exchange.name}: {symbol}] orderbook - "
                f"{type(e).__name__}: {e}"
            )


async def watch_symbol_trades(
    exchange: ExchangeFeed, symbol: str, control: Control
) -> AsyncGenerator[Trade, None, None]:
    while control.run():
        try:
            trades = await exchange.watch_trades(symbol)

            for trade_data in trades:
                yield Trade(
                    timestamp=dt.datetime.now(dt.UTC),
                    exchange=exchange.name, symbol=symbol,
                    price=trade_data['price'], quantity=trade_data['amount'],
                    side=trade_data['side'].capitalize()
                )

        except Exception as e:
            warnings.warn(
                f"[{exchange.name}: {symbol}] orderbook - "
                f"{type(e).__name__}: {e}"
            )
            raise e


async def watch_books(
    exchanges_symbols: dict[ExchangeFeed, Iterable[str]], control: Control
) -> AsyncGenerator[Book, None, None]:
    tasks = []

    for exchange, symbols in exchanges_symbols.items():
        for symbol in symbols:
            tasks.append(watch_symbol_book(exchange=exchange, symbol=symbol, control=control))

    merged = stream.merge(*tasks)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item


async def watch_trades(
    exchanges_symbols: dict[ExchangeFeed, Iterable[str]], control: Control
) -> AsyncGenerator[Trade, None, None]:
    tasks = []

    for exchange, symbols in exchanges_symbols.items():
        for symbol in symbols:
            tasks.append(watch_symbol_trades(exchange=exchange, symbol=symbol, control=control))

    merged = stream.merge(*tasks)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item


async def watch_data(
    exchanges_symbols: dict[ExchangeFeed, Iterable[str]], control: Control
) -> AsyncGenerator[Book | Trade, None, None]:
    watches = [
        watch_trades(exchanges_symbols, control=control),
        watch_books(exchanges_symbols, control=control)
    ]

    merged = stream.merge(*watches)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item
