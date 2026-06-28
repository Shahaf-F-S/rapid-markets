# test_record.py

import asyncio

from alive_progress import alive_bar

from rapid_markets.base import Control
from rapid_markets.source import CCXTFeed, watch_data
from rapid_markets.store import MarketDatabase


async def main():
    control = Control()
    symbols = {CCXTFeed('binance'): {'BTC/USDT', 'ETH/USDT'}}

    path = 'database/database.db'
    db = MarketDatabase(path)
    await db.connect()

    with control, alive_bar(title='Recording', monitor='{count}') as bar:
        async for data in watch_data(symbols, control=control):
            await db.insert(data.data())
            bar()

    await db.close()


if __name__ == '__main__':
    asyncio.run(main())