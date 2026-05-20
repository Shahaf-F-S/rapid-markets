# predictions.py

from pathlib import Path
from typing import AsyncGenerator

from rapid_markets.store.database import BaseDatabase, TableLimits, Data


__all__ = [
    'PredictionsDatabase'
]


class PredictionsDatabase(BaseDatabase):

    Predictions_TABLE = 'predictions'

    def __init__(self, path: str | Path, timeout: float = 60):
        super().__init__(path, timeout=timeout)
        self.predictions_table = self.Predictions_TABLE

    def table(self, data: Data) -> str:
        return self.predictions_table

    async def create(self):
        await self.create_exchange_symbol_timestamp(self.predictions_table)

    async def predictions_limits(self, limits: TableLimits) -> TableLimits:
        limits.table = self.predictions_table
        return await self.limit(limits)

    async def select_predictions(self, limits: TableLimits) -> AsyncGenerator[Data, None, None]:
        limits.table = self.predictions_table
        async for row in self.select(limits):
            yield row
