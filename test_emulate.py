# test_emulate.py

import asyncio

from alive_progress import alive_bar

from rapid_markets.store import MarketDatabase, TableLimits


async def main():
    path = 'database/database.db'
    limits = TableLimits(exchange='binance', symbol='BTC/USDT')

    db = MarketDatabase(path)
    await db.connect()

    with alive_bar(title='Emulating', monitor='{count}') as bar:
        async for data in db.simulate_market(limits):
            print(data)
            bar()

    await db.close()


if __name__ == '__main__':
    asyncio.run(main())