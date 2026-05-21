# test_emulate.py

import asyncio

from alive_progress import alive_bar

from rapid_markets.base import IterativeCallbacks, run_task
from rapid_markets.store import MarketDatabase, TableLimits


async def run(
    limits: TableLimits,
    db: MarketDatabase,
    callbacks: IterativeCallbacks
):
    with alive_bar(title='Emulating', monitor='{count}') as bar:
        async for data in db.simulate_market(limits):
            callbacks(data)
            bar()

    await db.close()


async def main():
    scheme = TableLimits(exchange='binance', symbol='BTC/USDT')

    path = 'database/database.db'

    market_db = MarketDatabase(path)
    await market_db.connect()

    calls = IterativeCallbacks().collect(print)

    await run_task(
        task=run(scheme, db=market_db, callbacks=calls),
        finish=[market_db.close()]
    )


if __name__ == '__main__':
    asyncio.run(main())