# test_record.py

import asyncio

from alive_progress import alive_bar

from rapid_markets.base import Control
from rapid_markets.source import Feed, watch_market
from rapid_markets.store import MarketDatabase


async def main():
    symbols = {Feed('binance').extend({'BTC/USDT', 'ETH/USDT'})}

    path = 'database/database.db'

    with (
        Control() as control,
        alive_bar(title='Recording', monitor='{count}') as bar
    ):
        async with (
            MarketDatabase(path) as db,
            watch_market(symbols, control) as watcher
        ):
            async for data in watcher:
                print(data)
                await db.insert(data.data())
                bar()


if __name__ == '__main__':
    asyncio.run(main())