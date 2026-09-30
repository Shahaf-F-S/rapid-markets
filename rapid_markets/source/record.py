# record.py

import asyncio
import datetime as dt
from collections.abc import Iterable
from typing import Self

from rapid_markets.base.control import Control
from rapid_markets.source.data import Book, Trade
from rapid_markets.source.feed import ExchangeFeed


__all__ = [
    "Watcher",
    "watch_market",
    "watch_books",
    "watch_trades",
    "Kind",
    "Control"
]


type Kind = type[Book] | type[Trade]
type Item = Book | Trade
type Stream = tuple[ExchangeFeed, str, Kind]


class Watcher:
    """
    Async iterator of Book / Trade objects for every (feed, symbol, kind)
    combination, in the order the exchanges deliver them.

    One fetch task runs per combination. When a fetch completes, its items
    are queued and the fetch is re-issued at once, so the output order is
    set by the servers alone.

    While the watcher is running, these take effect immediately:
    - feeds added or removed with `add` / `remove`;
    - kinds replaced by assigning `watcher.kinds = (Book,)`;
    - changes made on a feed itself (`add`, `remove`, `extend`, `reduce`,
      `activate`, `deactivate`, or setting `active`).

    The control governs the whole watcher: iteration ends when it stops or
    times out, and fetch errors go through it. An error it catches is
    retried after `feed.sleep`; any other error, or the one that exhausts
    `max_fails`, ends the iteration by being raised. For separate controls
    per feed, use a watcher per control.

        async with watch_market(feeds, control) as watcher:
            async for item in watcher:
                ...
    """

    def __init__(
        self,
        feeds: Iterable[ExchangeFeed],
        control: Control,
        kinds: Iterable[Kind] = (Book, Trade)
    ):
        self.feeds: set[ExchangeFeed] = set(feeds)
        self.control = control

        self._kinds: frozenset[Kind] = frozenset(kinds)
        self._tasks: dict[Stream, asyncio.Task] = {}
        self._queue: asyncio.Queue[Item | BaseException | None] = asyncio.Queue()
        self._running = False

    @property
    def kinds(self) -> frozenset[Kind]:
        return self._kinds

    @kinds.setter
    def kinds(self, kinds: Iterable[Kind]):
        self._kinds = frozenset(kinds)

        if self._running:
            for feed in self.feeds:
                self._sync(feed)

    def books_only(self) -> Self:
        self.kinds = (Book,)
        return self

    def trades_only(self) -> Self:
        self.kinds = (Trade,)
        return self

    def union(self) -> Self:
        self.kinds = (Book, Trade)
        return self

    def add(self, *feeds: ExchangeFeed) -> Self:
        for feed in feeds:
            self.feeds.add(feed)

            if self._running:
                feed.listen(self._sync)
                self._sync(feed)

        return self

    def remove(self, *feeds: ExchangeFeed) -> Self:
        for feed in feeds:
            self.feeds.discard(feed)

            if self._running:
                feed.unlisten(self._sync)
                self._sync(feed)

        return self

    def stop(self) -> None:
        self.control.stop()
        self._queue.put_nowait(None)  # wake a pending __anext__

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Item:
        if not self._running:
            self._running = True

            for feed in self.feeds:
                feed.listen(self._sync)
                self._sync(feed)

        item = await self._next()

        if isinstance(item, Book | Trade):
            return item

        await self.aclose()
        raise item or StopAsyncIteration

    async def aclose(self) -> None:
        self._running = False

        for feed in self.feeds:
            feed.unlisten(self._sync)

        tasks = tuple(self._tasks.values())
        self._tasks.clear()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        self._queue = asyncio.Queue()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def _next(self) -> Item | BaseException | None:
        """The next queued item or error; None once the control ends."""
        if not self.control.run():
            return None

        control = self.control
        remaining = None if control.timeout is None else (
            (control.start + control.timeout - dt.datetime.now()).total_seconds()
        )

        try:
            async with asyncio.timeout(remaining):
                return await self._queue.get()

        except TimeoutError:
            return None

    def _sync(self, feed: ExchangeFeed) -> None:
        wanted = {
            (feed, symbol, kind)
            for symbol in feed.subscribed
            for kind in self._kinds
        } if feed in self.feeds and feed.active else set()

        current = {stream for stream in self._tasks if stream[0] is feed}

        for stream in current - wanted:
            self._tasks.pop(stream).cancel()

        for stream in wanted - current:
            self._start(stream)

    def _start(self, stream: Stream) -> None:
        task = asyncio.create_task(self._fetch(*stream))
        task.add_done_callback(lambda t: self._on_done(stream, t))
        self._tasks[stream] = task

    def _on_done(self, stream: Stream, task: asyncio.Task) -> None:
        if self._tasks.get(stream) is not task:
            return

        if (error := task.exception()) is not None:
            self._queue.put_nowait(error)
            return

        self._start(stream)

        for item in task.result():
            self._queue.put_nowait(item)

    async def _fetch(self, feed: ExchangeFeed, symbol: str, kind: Kind) -> list[Item]:
        fetch = feed.books if kind is Book else feed.trades

        while True:
            with self.control:
                # noinspection argument-list
                return await fetch(symbol)

            # Reached only when the control caught the error: retry.
            # noinspection unreachable-code
            await asyncio.sleep(feed.sleep)


def watch_books(feeds: Iterable[ExchangeFeed], control: Control) -> Watcher:
    return Watcher(feeds, control, (Book,))


def watch_trades(feeds: Iterable[ExchangeFeed], control: Control) -> Watcher:
    return Watcher(feeds, control, (Trade,))


def watch_market(feeds: Iterable[ExchangeFeed], control: Control) -> Watcher:
    return Watcher(feeds, control, (Trade, Book))