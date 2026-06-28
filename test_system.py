# test_system.py

import asyncio

from rapid_markets.store import MarketDatabase, TableLimits
from rapid_markets.strategy.evaluate import ModelEvaluator
from rapid_markets.strategy.pipeline import DataPipeline
from rapid_markets.strategy.model import LightGBMQuantileModel


async def main():
    path = 'database/database.db'
    source = TableLimits(exchange='binance', symbol='BTC/USDT')

    db = MarketDatabase(path)
    await db.connect()

    pipeline = DataPipeline(db=db, limits=source)
    split = await pipeline.run()

    await db.close()

    gb_model = LightGBMQuantileModel()
    gb_model.fit(split.train, split.val)

    report = ModelEvaluator(gb_model).evaluate(
        split.test, pipeline.cfg.label_cfg.quantiles
    )

    print(report.summary())


if __name__ == '__main__':
    asyncio.run(main())