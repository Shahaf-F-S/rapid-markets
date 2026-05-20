# model.py

import asyncio
import threading
import json
import datetime as dt
from pathlib import Path
from typing import AsyncGenerator, Callable
from itertools import chain
from dataclasses import dataclass, fields, field
from functools import partial

from rapid_markets.base import labels
from rapid_markets.store.database import BaseDatabase, TableLimits, Data


__all__ = [
    'ModelDatabase',
    'ModelData',
    'ModelDataset',
    'Exporter'
]


@dataclass
class ModelData:

    exchange: str
    symbol: str
    timestamp: dt.datetime
    features: dict[str, float]
    targets: dict[str, float]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features.keys())

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.targets.keys())

    @property
    def index_names(self) -> tuple[str, str, str]:
        return labels.EXCHANGE, labels.SYMBOL, labels.TIMESTAMP

    @property
    def feature_values(self) -> tuple[float, ...]:
        return tuple(self.features.values())

    @property
    def target_values(self) -> tuple[float, ...]:
        return tuple(self.targets.values())

    @property
    def index_values(self) -> tuple[str, str, dt.datetime]:
        return self.exchange, self.symbol, self.timestamp

    @classmethod
    def load(cls, features: Data, targets: Data) -> ModelData:
        exchange = features.pop(labels.EXCHANGE, targets.pop(labels.EXCHANGE))
        symbol = features.pop(labels.SYMBOL, targets.pop(labels.SYMBOL))
        timestamp = features.pop(labels.TIMESTAMP, targets.pop(labels.TIMESTAMP))

        return ModelData(
            exchange=exchange, symbol=symbol, timestamp=timestamp,
            features=features, targets=targets
        )

    def dump(self) -> Data:
        return dict(
            chain(
                zip(self.index_names, self.index_values),
                zip(self.feature_names, self.feature_values),
                zip(self.target_names, self.target_values)
            )
        )

    def dump_features(self) -> Data:
        return dict(
            chain(
                zip(self.index_names, self.index_values),
                zip(self.feature_names, self.feature_values)
            )
        )

    def dump_targets(self) -> Data:
        return dict(
            chain(
                zip(self.index_names, self.index_values),
                zip(self.target_names, self.target_values)
            )
        )


@dataclass(slots=True, repr=False)
class ModelDataset:

    size: float
    location: Path
    limits: TableLimits
    rows: int = 0
    files: list[Path] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def __repr__(self):
        active_fields = [
            (
                f"{f.name}={v!r}"
                if not isinstance(v := getattr(self, f.name), list) else
                f"{f.name}=[...{len(getattr(self, f.name))}...]"
            )
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        return f"{self.__class__.__name__}({', '.join(active_fields)})"


type DatasetAGet = AsyncGenerator[Path, None, None]


@dataclass(kw_only=True, slots=True, frozen=True)
class Exporter:

    dataset: ModelDataset
    _gen: Callable[..., DatasetAGet] = field(repr=False, compare=False)

    async def export(
        self,
        chunk_size: int = 50000,
        on_tick: Callable[[], ..., ...] = None
    ) -> DatasetAGet:
        async for file in self._gen(chunk_size=chunk_size, on_tick=on_tick):
            yield file


class ModelDatabase(BaseDatabase):

    FEATURES_TABLE = 'features'
    TARGETS_TABLE = 'targets'

    def __init__(self, path: str | Path, timeout: float = 60):
        super().__init__(path, timeout=timeout)
        self.features_table = self.FEATURES_TABLE
        self.targets_table = self.TARGETS_TABLE

    def table(self, data: Data | None = None) -> str:
        raise TypeError('ModelDatabase cannot be used for writing.')

    async def create(self):
        pass

    async def select_model_data(
        self, limits: TableLimits, x: bool = True, y: bool = True
    ) -> AsyncGenerator[ModelData, None, None]:
        features_limit = limits.copy()
        features_limit.table = self.features_table

        if not x:
            features_limit.columns = list(labels.INDEX_COLUMNS)

        features_gen = self.select(features_limit)

        targets_limit = limits.copy()
        targets_limit.table = self.targets_table

        if not y:
            targets_limit.columns = list(labels.INDEX_COLUMNS)

        targets_gen = self.select(targets_limit)

        try:
            features = await anext(features_gen)
            targets = await anext(targets_gen)

        except StopAsyncIteration:
            return

        while True:
            features_time: dt.datetime = features[labels.TIMESTAMP]
            targets_time: dt.datetime = targets[labels.TIMESTAMP]

            try:
                if features_time == targets_time:
                    yield ModelData.load(features=features, targets=targets)
                    features, targets = await asyncio.gather(anext(features_gen), anext(targets_gen))

                elif features_time < targets_time:
                    features = await anext(features_gen)

                elif targets_time < features_time:
                    targets = await anext(targets_gen)

            except StopAsyncIteration:
                break

    async def _split_limits(
        self, limits: TableLimits, size: float | int, fetch: bool = True
    ) -> tuple[TableLimits, TableLimits]:
        base = limits.copy()

        if fetch:
            base.start_index = None
            base.end_index = None
            base.table = self.features_table

            base = await self.limit(base)
            base.start_index = None
            base.end_index = None

        total = base.max_count or 0

        if isinstance(size, float):
            split_index = int(total * size + 1)

        else:
            split_index = int(min(size, total))

        first = await self.limit(
            TableLimits(
                table=base.table, exchange=limits.exchange, symbol=limits.symbol,
                max_count=split_index, start_time=base.start_time,
            )
        )
        first.columns = None

        second = await self.limit(
            TableLimits(
                table=base.table, exchange=limits.exchange, symbol=limits.symbol,
                start_time=base.start_time, end_time=base.end_time,
                start_index=first.max_count
            )
        )
        second.columns = None
        second.max_count = (second.end_index or 0) - (second.start_index or 0)
        second.start_index = 0
        second.end_index = second.max_count

        return first, second

    async def export_model_data_parquet(
        self,
        rows: AsyncGenerator[ModelData, ..., ...],
        output_dir: str | Path,
        limits: TableLimits | None = None,
        chunk_size: int = 50000,
        on_tick: Callable[[], ..., ...] = None
    ) -> AsyncGenerator[tuple[Path | None, Path | None], None, None]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if limits is not None:
            with open(output_path / 'limits.json', 'w') as f:
                json.dump(limits.dump(), f, indent=4, default=str)

        x_chunk: list[Data] = []
        y_chunk: list[Data] = []
        x_file_index = 0
        y_file_index = 0
        new_x = False
        new_y = False
        x_chunk_path = None
        y_chunk_path = None

        async for row in rows:
            row: ModelData

            if row.features:
                x_chunk.append(row.dump_features())

            if row.targets:
                y_chunk.append(row.dump_targets())

            if on_tick:
                on_tick()

            if len(x_chunk) >= chunk_size:
                x_chunk_path = output_path / f"parquet/{x_file_index:04d}/x.parquet"
                threading.Thread(target=self.save_parquet, args=(x_chunk, x_chunk_path)).start()
                x_chunk = []
                x_file_index += 1
                new_x = True

            if len(y_chunk) >= chunk_size:
                y_chunk_path = output_path / f"parquet/{y_file_index:04d}/y.parquet"
                threading.Thread(target=self.save_parquet, args=(y_chunk, y_chunk_path)).start()
                y_chunk = []
                y_file_index += 1
                new_y = True

            if new_x or new_y:
                yield x_chunk_path, y_chunk_path

        if len(x_chunk) >= chunk_size:
            x_chunk_path = output_path / f"parquet/{x_file_index:04d}/x.parquet"
            threading.Thread(target=self.save_parquet, args=(x_chunk, x_chunk_path)).start()
            new_x = True

        if len(y_chunk) >= chunk_size:
            y_chunk_path = output_path / f"parquet/{y_file_index:04d}/y.parquet"
            threading.Thread(target=self.save_parquet, args=(y_chunk, y_chunk_path)).start()
            new_y = True

        if new_x or new_y:
            yield x_chunk_path, y_chunk_path

    def _data_gen(
        self,
        gen: Callable[[TableLimits], AsyncGenerator[ModelData, None, None]],
        dataset: ModelDataset
    ) -> Callable[..., DatasetAGet]:
        async def wrapper(
            chunk_size: int = 50000,
            on_tick: Callable[[], ..., ...] = None
        ) -> DatasetAGet:
            dataset.rows = chunk_size

            async for chuck_file in self.export_model_data_parquet(
                gen(dataset.limits), output_dir=dataset.location,
                limits=dataset.limits, chunk_size=chunk_size, on_tick=on_tick
            ):
                dataset.files.append(chuck_file)
                yield chuck_file

        return wrapper

    async def exporters(
        self,
        destination: str | Path,
        train_size: float | int,
        val_size: float,
        limits: TableLimits,
        x: bool = True,
        y: bool = True
    ) -> tuple[Exporter, Exporter, Exporter]:
        features = []
        targets = []

        async def gen(l: TableLimits) -> AsyncGenerator[ModelData, None, None]:
            async for data in self.select_model_data(l, x=x, y=y):
                data: ModelData

                if not features:
                    features.extend(data.feature_names)
                    train_data.features.extend(features)
                    val_data.features.extend(features)
                    test_data.features.extend(features)

                if not targets:
                    targets.extend(data.target_names)
                    train_data.targets.extend(targets)
                    val_data.targets.extend(targets)
                    test_data.targets.extend(targets)

                yield data

        train_limits, untrain_limits = await self._split_limits(limits, size=train_size, fetch=True)
        val_limits, test_limits = await self._split_limits(untrain_limits, size=val_size, fetch=False)

        if isinstance(train_size, int):
            train_size = train_size / (
                (train_limits.max_count or 0) +
                (val_limits.max_count or 0) +
                (test_limits.max_count or 0)
            )

        train_data = ModelDataset(
            limits=train_limits, location=Path(destination) / 'train',
            size=round(train_size, 3)
        )
        val_data = ModelDataset(
            limits=val_limits, location=Path(destination) / 'val',
            size=round((1 - train_size) * val_size, 3)
        )
        test_data = ModelDataset(
            limits=test_limits, location=Path(destination) / 'test',
            size=round(1 - val_data.size - train_data.size)
        )

        _data_gen = partial(self._data_gen, gen=gen)

        return (
            Exporter(dataset=train_data, _gen=_data_gen(dataset=train_data)),
            Exporter(dataset=val_data, _gen=_data_gen(dataset=val_data)),
            Exporter(dataset=test_data, _gen=_data_gen(dataset=test_data))
        )