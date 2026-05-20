# market.py

import asyncio
import datetime as dt
from pathlib import Path
from typing import AsyncGenerator, Iterable

from aioitertools import zip as azip

from rapid_markets.base import labels
from rapid_markets.store.database import BaseDatabase, Data, TableLimits, TimePair
from rapid_markets.source import Book, Trade


__all__ = [
    'MarketDatabase'
]


class MarketDatabase(BaseDatabase):

    BOOKS_TABLE = 'books'
    TRADES_TABLE = 'trades'

    def __init__(self, path: str | Path, timeout: float = 60):
        super().__init__(path, timeout=timeout)
        self.books_table = self.BOOKS_TABLE
        self.trades_table = self.TRADES_TABLE

    def table(self, data: Data) -> str:
        return self.trades_table if (labels.PRICE in data) else self.books_table

    async def create(self):
        await self.create_exchange_symbol_timestamp(self.books_table, self.trades_table)

    async def insert_many(self, items: Iterable[Data], commit: bool = True):
        self._validate_connection()

        books = []
        trades = []

        for data in items:
            record = trades if (self.table(data) == self.trades_table) else books
            record.append(data)

        await asyncio.gather(
            super().insert_many(books, commit=commit),
            super().insert_many(trades, commit=commit)
        )

    async def simulate_market(self, limits: TableLimits) -> AsyncGenerator[Book | Trade, None, None]:
        trades_limits = limits.copy()
        trades_limits.table = self.trades_table
        trades_gen = self.select(trades_limits)

        books_limits = limits.copy()
        books_limits.table = self.books_table
        books_gen = self.select(books_limits)

        trade_data: Data | None = None
        book_data: Data | None = None
        trades_done = False
        books_done = False

        while True:
            if (trade_data is None) and not trades_done:
                try:
                    trade_data = await anext(trades_gen)

                except StopAsyncIteration:
                    trades_done = True

            if (book_data is None) and not books_done:
                try:
                    book_data = await anext(books_gen)

                except StopAsyncIteration:
                    books_done = True

            if (trade_data is None) and (book_data is None) and trades_done and books_done:
                break

            if (trade_data is not None) and (book_data is not None):
                if book_data[labels.TIMESTAMP] <= trade_data[labels.TIMESTAMP]:
                    yield Book.load(book_data)
                    book_data = None

                else:
                    yield Trade.load(trade_data)
                    trade_data = None

            elif trade_data is not None:
                yield Trade.load(trade_data)
                trade_data = None

            elif book_data is not None:
                yield Book.load(book_data)
                book_data = None

    async def select_targets_futures(
        self,
        limits: TableLimits,
        futures: Iterable[int | dt.timedelta],
        tables: dict[str, list[str]] | None = None
    ) -> AsyncGenerator[dict[int | dt.timedelta, TimePair], None, None]:
        limits.table = self.trades_table
        futures = list(futures)

        if tables is None:
            tables = {self.books_table: [labels.BID_PRICE, labels.ASK_PRICE]}

        async for data in azip(*(self.select_future_multi(limits, future, tables) for future in futures)):
            data: tuple[dict[str, TimePair], ...]
            results: dict[int | dt.timedelta, TimePair] = {}

            for future, pairs in zip(futures, data):
                results[future] = pairs[self.trades_table]

                for table in tables:
                    results[future].past.update(pairs[table].past)
                    results[future].future.update(pairs[table].future)

            yield results
