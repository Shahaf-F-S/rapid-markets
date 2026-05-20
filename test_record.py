# test_record.py

import asyncio

from alive_progress import alive_bar

from rapid_markets.base import (
    AsyncGatherCallbacks, BaseAsyncCallbacks, Control, run_task
)
from rapid_markets.source import CCXTFeed, watch_data
from rapid_markets.store import MarketDatabase


async def run(
    exchanges_symbols: dict[CCXTFeed, set[str]],
    callbacks: BaseAsyncCallbacks,
    control: Control
):
    with control, alive_bar(title='Recording', monitor='{count}') as bar:
        async for data in watch_data(exchanges_symbols, control=control):
            await callbacks(data)
            bar()


async def main():
    controller = Control()
    symbols = {CCXTFeed('binance'): {'BTC/USDT'}}

    path = 'database/database.db'

    market_db = MarketDatabase(path)
    await market_db.connect()
    calls = AsyncGatherCallbacks().collect(market_db.insert)

    await run_task(task=run(symbols, callbacks=calls, control=controller))


if __name__ == '__main__':
    asyncio.run(main())