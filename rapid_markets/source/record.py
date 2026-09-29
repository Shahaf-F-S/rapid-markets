# record.py

import datetime as dt
from typing import AsyncGenerator, Iterable, Literal
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
    "watch"
]


async def watch_symbol_book(
    feed: ExchangeFeed, symbol: str, control: Control
) -> AsyncGenerator[Book, None, None]:
    while control.run():
        if not feed.active:
            continue

        try:
            orderbook = await feed.watch_order_book(symbol)

            bids, asks = orderbook['bids'], orderbook['asks']

            if bids and asks:
                yield Book(
                    timestamp=dt.datetime.now(dt.UTC),
                    exchange=feed.name, symbol=symbol,
                    bids=bids, asks=asks
                )

        except Exception as e:
            warnings.warn(
                f"[{feed.name}: {symbol}] orderbook - "
                f"{type(e).__name__}: {e}"
            )


async def watch_symbol_trades(
    feed: ExchangeFeed, symbol: str, control: Control
) -> AsyncGenerator[Trade, None, None]:
    while control.run():
        if not feed.active:
            continue

        try:
            trades = await feed.watch_trades(symbol)

            for trade_data in trades:
                yield Trade(
                    timestamp=dt.datetime.now(dt.UTC),
                    exchange=feed.name, symbol=symbol,
                    price=trade_data['price'], quantity=trade_data['amount'],
                    side=trade_data['side'].capitalize()
                )

        except Exception as e:
            warnings.warn(
                f"[{feed.name}: {symbol}] orderbook - "
                f"{type(e).__name__}: {e}"
            )
            raise e


async def watch_feeds(
    feeds: Iterable[ExchangeFeed],
    control: Control,
    gen: Literal[watch_symbol_book, watch_symbol_trades]
) -> AsyncGenerator[Book | Trade, None, None]:
    tasks = []

    for feed in feeds:
        for symbol in feed.subscribed:
            tasks.append(gen(feed=feed, symbol=symbol, control=control))

    merged = stream.merge(*tasks)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item


async def watch_books(
    feeds: Iterable[ExchangeFeed], control: Control
) -> AsyncGenerator[Book, None, None]:
    tasks = []

    for feed in feeds:
        for symbol in feed.subscribed:
            tasks.append(watch_symbol_book(feed=feed, symbol=symbol, control=control))

    merged = stream.merge(*tasks)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item


async def watch_trades(
    feeds: Iterable[ExchangeFeed], control: Control
) -> AsyncGenerator[Trade, None, None]:
    tasks = []

    for feed in feeds:
        for symbol in feed.subscribed:
            tasks.append(watch_symbol_trades(feed=feed, symbol=symbol, control=control))

    merged = stream.merge(*tasks)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item


async def watch(feeds: Iterable[ExchangeFeed], control: Control) -> AsyncGenerator[Book | Trade, None, None]:
    watches = [
        watch_trades(feeds, control=control),
        watch_books(feeds, control=control)
    ]

    merged = stream.merge(*watches)

    async with merged.stream() as streamer:
        async for item in streamer:
            yield item
