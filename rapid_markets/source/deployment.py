# deployment.py

import datetime as dt

from rapid_markets.base import labels
from rapid_markets.source.data import Book, Trade


__all__ = [
    'update_realtime_market'
]


def update_realtime_market(
    book: Book | dict[str, ...],
    trade: Trade | dict[str, ...],
    output: dict | None = None,
    now: bool = True,
    limit: int | None = None
) -> dict[str, ...]:
    if isinstance(book, Book):
        book = book.data(limit=limit)

    book: dict

    if labels.TIMESTAMP in book and labels.BOOKS_TIMESTAMP not in book:
        book[labels.BOOKS_TIMESTAMP] = book.pop(labels.TIMESTAMP)

    if isinstance(trade, Trade):
        trade = trade.data()

    trade: dict

    if labels.TIMESTAMP in trade and labels.BOOKS_TIMESTAMP not in trade:
        trade[labels.TRADES_TIMESTAMP] = trade.pop(labels.TIMESTAMP)

    output: dict
    output.update(book)
    output.update(trade)
    output[labels.TIMESTAMP] = (
        dt.datetime.now(dt.UTC) if now else max(
            book[labels.BOOKS_TIMESTAMP], trade[labels.TRADES_TIMESTAMP]
        )
    )

    return output
