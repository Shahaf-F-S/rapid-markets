# database.py

import asyncio
import aiosqlite
import sqlite3
import datetime as dt
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from typing import AsyncGenerator, Iterable, Callable, Sequence, Self
import json
from itertools import batched

import pyarrow as pa
import pyarrow.parquet as pq

from rapid_markets.base import Control
import rapid_markets.base.labels as labels


__all__ = [
    'BaseDatabase',
    'Data',
    'TableLimits',
    'DataCompressor',
    'TimePair'
]


def datetime_to_stream(val: dt.datetime) -> str:
    return val.isoformat(sep=' ')


def stream_to_datetime(val: bytes) -> dt.datetime:
    return dt.datetime.fromisoformat(val.decode())


sqlite3.register_adapter(dt.datetime, datetime_to_stream)
sqlite3.register_adapter(list, json.dumps)
sqlite3.register_adapter(dict, json.dumps)
sqlite3.register_adapter(tuple, json.dumps)
sqlite3.register_adapter(set, json.dumps)
sqlite3.register_converter("JSON", lambda x: json.loads(x.decode()))
sqlite3.register_converter("DATETIME", stream_to_datetime)
sqlite3.register_converter("TIMESTAMP", stream_to_datetime)


@dataclass
class TableLimits:

    exchange: str
    symbol: str
    table: str | None = None
    max_count: int | None = None
    start_time: dt.datetime | None = None
    end_time: dt.datetime | None = None
    start_index: int | None = None
    end_index: int | None = None
    columns: list[str] | None = None
    filter: str | None = None

    def __repr__(self):
        active_fields = [
            (
                f"{f.name}={getattr(self, f.name)!r}"
                if f.name != 'columns' else
                f"{f.name}=[...{len(getattr(self, f.name))}...]"
            )
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        return f"{self.__class__.__name__}({', '.join(active_fields)})"

    def __and__(self, other) -> TableLimits:
        if not isinstance(other, TableLimits):
            return NotImplemented

        other: TableLimits

        if self.key != other.key:
            raise ValueError('keys must match.')

        if None in (self.max_count, other.max_count):
            max_count = self.max_count or other.max_count

        else:
            max_count = min(self.max_count or 0, other.max_count or 0)

        if None in (self.start_index, other.start_index):
            start_index = self.start_index or other.start_index

        else:
            start_index = max(self.start_index or 0, other.start_index or 0)

        if None in (self.end_index, other.end_index):
            end_index = self.end_index or other.end_index

        else:
            end_index = min(self.end_index or 0, other.end_index or 0)

        if None in (self.start_time, other.start_time):
            start_time = self.start_time or other.start_time

        else:
            start_time = max(
                self.start_time or dt.datetime.now(dt.UTC),
                other.start_time or dt.datetime.now(dt.UTC)
            )

        if None in (self.end_time, other.end_time):
            end_time = self.end_time or other.end_time

        else:
            end_time = min(
                self.end_time or dt.datetime.now(dt.UTC),
                other.end_time or dt.datetime.now(dt.UTC)
            )

        if None in (self.columns, other.columns):
            columns = self.columns or other.columns
            if columns is not None:
                columns = columns.copy()

        else:
            # noinspection PyTypeChecker
            his_cols = set(other.columns)
            # noinspection PyTypeChecker
            columns = [col for col in self.columns if col in his_cols]

        if None in (self.filter, other.filter):
            sql_filter = self.filter or other.filter

        else:
            sql_filter = f"({self.filter}) AND ({other.filter})"

        return TableLimits(
            exchange=self.exchange, symbol=self.symbol, table=self.table,
            max_count=max_count, start_time=start_time, end_time=end_time,
            start_index=start_index, end_index=end_index, columns=columns,
            filter=sql_filter
        )

    def __or__(self, other) -> TableLimits:
        if not isinstance(other, TableLimits):
            return NotImplemented

        other: TableLimits

        if self.key != other.key:
            raise ValueError('keys must match.')

        if None in (self.max_count, other.max_count):
            max_count = self.max_count or other.max_count

        else:
            max_count = max(self.max_count or 0, other.max_count or 0)

        if None in (self.start_index, other.start_index):
            start_index = self.start_index or other.start_index

        else:
            start_index = min(self.start_index or 0, other.start_index or 0)

        if None in (self.end_index, other.end_index):
            end_index = self.end_index or other.end_index

        else:
            end_index = max(self.end_index or 0, other.end_index or 0)

        if None in (self.start_time, other.start_time):
            start_time = self.start_time or other.start_time

        else:
            start_time = min(
                self.start_time or dt.datetime.now(dt.UTC),
                other.start_time or dt.datetime.now(dt.UTC)
            )

        if None in (self.end_time, other.end_time):
            end_time = self.end_time or other.end_time

        else:
            end_time = max(
                self.end_time or dt.datetime.now(dt.UTC),
                other.end_time or dt.datetime.now(dt.UTC)
            )

        if None in (self.columns, other.columns):
            columns = self.columns or other.columns
            if columns is not None:
                columns = columns.copy()

        else:
            # noinspection PyTypeChecker
            his_cols = set(other.columns)
            # noinspection PyTypeChecker
            columns = set(col for col in self.columns if col in his_cols)
            columns.update(his_cols - columns)
            columns = list(columns)

        if None in (self.filter, other.filter):
            sql_filter = self.filter or other.filter

        else:
            sql_filter = f"({self.filter}) AND ({other.filter})"

        return TableLimits(
            exchange=self.exchange, symbol=self.symbol, table=self.table,
            max_count=max_count, start_time=start_time, end_time=end_time,
            start_index=start_index, end_index=end_index, columns=columns,
            filter=sql_filter
        )

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.exchange, self.symbol, self.table

    def dump(self) -> dict[str, ...]:
        return asdict(self)

    def copy(self) -> TableLimits:
        return type(self)(**self.dump())

    @classmethod
    def load(cls, data: dict[str, ...]) -> TableLimits:
        data = data.copy()
        data['start_time'] = dt.datetime.fromisoformat(data['start_time']) if data['start_time'] else None
        data['end_time'] = dt.datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        return cls(**data)


# noinspection PyUnhashable
ADAPTERS = {
    dt.datetime: 'DATETIME',
    list: 'JSON',
    tuple: 'JSON',
    set: 'JSON',
    dict: 'JSON',
    float: 'REAL',
    int: 'NUMERIC',
    str: 'TEXT'
}


type Data = dict[str, ...]


@dataclass(slots=True, frozen=True)
class TimePair:

    past: Data
    future: Data
    difference: int | dt.timedelta

    def copy(self) -> TimePair:
        return TimePair(
            past=self.past.copy(),
            future=self.future.copy(),
            difference=self.difference
        )

    @property
    def exchange(self) -> str:
        return self.past[labels.EXCHANGE]

    @property
    def symbol(self) -> str:
        return self.past[labels.SYMBOL]


class BaseDatabase(ABC):

    def __init__(self, path: str | Path, timeout: float = 60):
        self.path = path
        self.connection: aiosqlite.Connection | None = None
        self.adapted: dict[str, set[str]] = {}
        self.timeout = timeout

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.close()

    def _validate_connection(self):
        if self.connection is None:
            raise RuntimeError('database is not connected.')

    async def connect(self):
        Path(self.path).parent.mkdir(exist_ok=True, parents=True)
        self.connection = await aiosqlite.connect(
            self.path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=self.timeout
        )
        self.connection.row_factory = aiosqlite.Row
        await self._setup_performance()
        await self.create()
        self.adapted.update({table: set(await self.columns(table)) for table in (await self.tables())})
        return self

    async def _setup_performance(self):
        self._validate_connection()

        # noinspection PyUnresolvedReferences,PyTypeChecker
        connection: aiosqlite.Connection = self.connection
        # 1. WAL mode allows readers and writers to not block each other
        await connection.execute("PRAGMA journal_mode=WAL;")
        # 2. NORMAL synchronous is safe for WAL and much faster than FULL.
        # It ensures integrity but doesn't wait for the disk to 'spin down'.
        await connection.execute("PRAGMA synchronous=NORMAL;")
        # 3. Increase cache size (e.g., 64MB) to keep more of the DB in RAM
        await connection.execute("PRAGMA cache_size=-64000;")
        # 4. Memory-mapped I/O: Faster reads by mapping the file to RAM
        await connection.execute("PRAGMA mmap_size=268435456;")  # 256MB
        # This allows SQLite to reuse deleted space more aggressively without locking the whole DB
        await connection.execute("PRAGMA auto_vacuum = INCREMENTAL;")

    async def create_exchange_symbol_timestamp(self, *tables: str):
        self._validate_connection()

        # noinspection PyUnresolvedReferences,PyTypeChecker
        connection: aiosqlite.Connection = self.connection

        for table in tables:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    {labels.EXCHANGE} TEXT,
                    {labels.SYMBOL} TEXT,
                    {labels.TIMESTAMP} TIMESTAMP
                )
                """
            )

            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_lookup ON {table} "
                f"({labels.EXCHANGE}, {labels.SYMBOL}, {labels.TIMESTAMP})"
            )
            await connection.commit()

    async def adapt(self, data: Data):
        # noinspection PyTypeChecker
        connection: aiosqlite.Connection = self.connection

        self._validate_connection()
        table = self.table(data)

        existing_cols = self.adapted[table]
        required_cols = set(data.keys())
        missing_cols = required_cols - existing_cols

        if not missing_cols:
            return

        # Start a manual transaction for schema changes
        await connection.execute("BEGIN TRANSACTION;")

        try:
            for col in missing_cols:
                self.adapted[table].add(col)
                kind = ADAPTERS.get(type(data[col]), 'BLOB')
                # Note: We still await here because aiosqlite is a wrapper,
                # but the BEGIN/COMMIT makes the DB side much faster.
                await connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {kind}')

            await connection.commit()

        except Exception as e:
            await connection.rollback()
            raise e

    @abstractmethod
    def table(self, data: Data) -> str:
        pass

    @abstractmethod
    async def create(self):
        pass

    @staticmethod
    def _insert_command(table: str, data: Data) -> str:
        cols = list(f'"{col}"' for col in data.keys())
        return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"

    # noinspection PyMethodMayBeStatic
    def to_row(self, item: Data) -> Data:
        return item

    async def insert(self, item: Data, commit: bool = True):
        self._validate_connection()

        # noinspection PyUnresolvedReferences,PyTypeChecker
        connection: aiosqlite.Connection = self.connection
        await self.adapt(item)
        await connection.execute(
            self._insert_command(self.table(item), item),
            tuple(item.values())
        )

        if commit:
            await connection.commit()

    async def insert_many(self, items: Sequence[Data], commit: bool = True):
        self._validate_connection()

        first_item = self.to_row(items[0])

        # noinspection PyUnresolvedReferences,PyTypeChecker
        connection: aiosqlite.Connection = self.connection
        await self.adapt(first_item)

        query = self._insert_command(self.table(first_item), first_item)
        values = (tuple(self.to_row(item).values()) for item in items)
        await connection.executemany(query, values)

        if commit:
            await connection.commit()

    async def _select(self, limits: TableLimits) -> AsyncGenerator[Data, None, None]:
        self._validate_connection()

        column_str = ', '.join(f'"{c}"' for c in limits.columns) if limits.columns else '*'
        offset = limits.start_index or 0
        limit_val = limits.max_count if limits.max_count is not None else -1

        query = f"""
            SELECT {column_str} FROM "{limits.table}"
            WHERE rowid IN (
                SELECT rowid FROM "{limits.table}"
                WHERE "{labels.EXCHANGE}" = ? AND "{labels.SYMBOL}" = ?
                {"AND " + labels.TIMESTAMP + " >= ?" if limits.start_time else ""}
                {"AND " + labels.TIMESTAMP + " <= ?" if limits.end_time else ""}
                LIMIT ? OFFSET ?
            )
        """

        params: list = [limits.exchange, limits.symbol]

        if limits.start_time:
            params.append(limits.start_time)

        if limits.end_time:
            params.append(limits.end_time)

        params.extend([limit_val, offset])

        # noinspection PyUnresolvedReferences
        async with self.connection.execute(query, params) as cursor:
            column_names = [d[0] for d in cursor.description]
            async for row in cursor:
                yield dict(zip(column_names, row))

    async def select(self, limits: TableLimits, max_columns: int = 5000) -> AsyncGenerator[Data, None, None]:
        if limits.table is None:
            raise ValueError('table must be defined')

        columns = list(limits.columns or (await self.columns(limits.table)))

        if len(columns) <= max_columns:
            async for data in self._select(limits):
                yield data

        else:
            selectors = []

            for batch_columns in batched(columns, max_columns, strict=False):
                batch_limits = limits.copy()
                batch_limits.columns = list(batch_columns)
                selectors.append(self._select(batch_limits))

            while True:
                try:
                    data = {}
                    for item in await asyncio.gather(*(anext(s) for s in selectors)):
                        data.update(item)

                    yield data

                except StopAsyncIteration:
                    break

    async def select_future_multi(
        self,
        limits: TableLimits,
        future: int | dt.timedelta,
        additional_tables: dict[str, Iterable[str] | None]
    ) -> AsyncGenerator[dict[str, TimePair], None, None]:
        self._validate_connection()

        if limits.table is None:
            raise ValueError('base table must be defined in the limits.')

        all_cols_sql = []
        base_cols = limits.columns or (await self.columns(limits.table))

        for c in base_cols:
            all_cols_sql.append(f'p."{c}" AS "base_p_{c}"')
            all_cols_sql.append(f'f."{c}" AS "base_f_{c}"')

        for table, table_columns in additional_tables.items():
            if table_columns is not None and not table_columns:
                continue

            t_cols = list(table_columns or ()) or await self.columns(table)
            for c in t_cols:
                all_cols_sql.append(f'p_{table}."{c}" AS "{table}_p_{c}"')
                all_cols_sql.append(f'f_{table}."{c}" AS "{table}_f_{c}"')

        if isinstance(future, int):
            base_future_logic = f"""
                SELECT rowid FROM {limits.table} 
                WHERE {labels.EXCHANGE} = p.{labels.EXCHANGE} AND {labels.SYMBOL} = p.{labels.SYMBOL}
                  AND {labels.TIMESTAMP} > p.{labels.TIMESTAMP}
                ORDER BY {labels.TIMESTAMP} LIMIT 1 OFFSET {future - 1}
            """
            modifier_param = []

        else:
            base_future_logic = f"""
            SELECT rowid FROM {limits.table} WHERE {labels.EXCHANGE} = p.{labels.EXCHANGE} AND 
            {labels.SYMBOL} = p.{labels.SYMBOL} AND {labels.TIMESTAMP} >= datetime(p.{labels.TIMESTAMP}, ?) 
            ORDER BY {labels.TIMESTAMP} LIMIT 1
            """
            modifier_param = [f"{future.total_seconds():+} seconds"]

        def get_asof_subquery(table_name, anchor_alias):
            return f"""
                SELECT rowid FROM {table_name}
                WHERE {labels.EXCHANGE} = p.{labels.EXCHANGE} AND {labels.SYMBOL} = p.{labels.SYMBOL}
                  AND {labels.TIMESTAMP} <= {anchor_alias}.{labels.TIMESTAMP}
                ORDER BY {labels.TIMESTAMP} DESC LIMIT 1
            """

        joins = [f"JOIN {limits.table} f ON f.rowid = ({base_future_logic})"]
        for table in additional_tables:
            joins.append(f"LEFT JOIN {table} p_{table} ON p_{table}.rowid = ({get_asof_subquery(table, 'p')})")
            joins.append(f"LEFT JOIN {table} f_{table} ON f_{table}.rowid = ({get_asof_subquery(table, 'f')})")

        where_clause = f"WHERE p.{labels.EXCHANGE} = ? AND p.{labels.SYMBOL} = ?"
        params = modifier_param + [limits.exchange, limits.symbol]

        if limits.start_time:
            where_clause += f" AND p.{labels.TIMESTAMP} >= ?"
            params.append(limits.start_time)

        query = f"""
            SELECT {', '.join(all_cols_sql)} FROM {limits.table} p {' '.join(joins)}
            {where_clause} ORDER BY p.{labels.TIMESTAMP} ASC
        """

        base_cols = set(base_cols)

        # noinspection PyUnresolvedReferences
        async with self.connection.execute(query, params) as cursor:
            async for row in cursor:
                row = dict(row)

                pairs = {}

                dict_row = dict(row)
                past_dict = {
                    c: dict_row[nc] for c in base_cols
                    if (nc := f"base_p_{c.split('base_p_', maxsplit=1)[-1]}") in dict_row
                }
                future_dict = {
                    c: dict_row[nc] for c in base_cols
                    if (nc := f"base_f_{c.split('base_f_', maxsplit=1)[-1]}") in dict_row
                }

                pairs[limits.table] = TimePair(past=past_dict, future=future_dict, difference=future)

                for table, table_columns in additional_tables.items():
                    a_past_dict = {
                        c: dict_row[nc] for c in (table_columns or ())
                        if (nc := f"{table}_p_{c.split(f'{table}_p_', maxsplit=1)[-1]}") in dict_row
                    }
                    a_future_dict = {
                        c: dict_row[nc] for c in (table_columns or ())
                        if (nc := f"{table}_f_{c.split(f'{table}_f_', maxsplit=1)[-1]}") in dict_row
                    }

                    pairs[table] = TimePair(past=a_past_dict, future=a_future_dict, difference=future)

                yield pairs

    @staticmethod
    def save_parquet(data: list[Data], file_path: str | Path):
        if not data:
            return

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyArgumentList
        pq.write_table(pa.Table.from_pylist(data), file_path, compression='snappy')

    async def export_parquet(
        self,
        rows: AsyncGenerator[Data, ..., ...],
        output_dir: str | Path,
        limits: TableLimits | None = None,
        chunk_size: int = 50000,
        on_tick: Callable[[], ..., ...] = None
    ) -> AsyncGenerator[Path, None, None]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if limits is not None:
            with open(output_path / 'limits.json', 'w') as f:
                json.dump(limits.dump(), f, indent=4, default=str)

        chunk_data: list[Data] = []
        file_index = 0

        async for row in rows:
            chunk_data.append(row)

            if on_tick:
                # noinspection calling-non-callable
                on_tick()

            if len(chunk_data) >= chunk_size:
                chunk_path = output_path / f"parquet/{file_index:04d}.parquet"
                threading.Thread(target=self.save_parquet, args=(chunk_data, chunk_path)).start()
                file_index += 1
                chunk_data = []

                yield chunk_path

        if chunk_data:
            chunk_path = output_path / f"parquet/{file_index:04d}.parquet"
            threading.Thread(target=self.save_parquet, args=(chunk_data, chunk_path)).start()

            yield chunk_path

    async def columns(self, table: str) -> list[str]:
        query = f"PRAGMA table_info({table})"

        # noinspection PyUnresolvedReferences
        async with self.connection.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [row["name"] for row in rows]

    # noinspection PyUnresolvedReferences
    async def limit(self, limits: TableLimits) -> TableLimits:
        if limits.table is None:
            raise ValueError('table must be defined')

        where_clauses = [f"{labels.EXCHANGE} = ?", f"{labels.SYMBOL} = ?"]
        params: list = [limits.exchange, limits.symbol]

        if limits.start_time:
            where_clauses.append(f"{labels.TIMESTAMP} >= ?")
            params.append(limits.start_time)

        if limits.end_time:
            where_clauses.append(f"{labels.TIMESTAMP} <= ?")
            params.append(limits.end_time)

        where_str = " WHERE " + " AND ".join(where_clauses)

        offset = limits.start_index or 0

        row_limit = -1
        if limits.end_index is not None:
            row_limit = max(0, limits.end_index - offset)

        if limits.max_count is not None:
            if row_limit == -1 or limits.max_count < row_limit:
                row_limit = limits.max_count

        query = f"""
            WITH SlicedData AS (
                SELECT {labels.TIMESTAMP} FROM "{limits.table}"
                {where_str}
                ORDER BY {labels.TIMESTAMP}
                LIMIT ? OFFSET ?
            )
            SELECT 
                MIN({labels.TIMESTAMP}), 
                MAX({labels.TIMESTAMP}), 
                COUNT(*) 
            FROM SlicedData
        """

        query_params = params + [row_limit, offset]

        async with self.connection.execute(query, query_params) as cursor:
            row = await cursor.fetchone()

            if not row or row[2] == 0:
                return TableLimits(
                    table=limits.table,
                    exchange=limits.exchange,
                    symbol=limits.symbol,
                    start_time=limits.start_time,
                    end_time=limits.end_time
                )

            start_ts, end_ts, total = row

        return TableLimits(
            table=limits.table,
            exchange=limits.exchange,
            symbol=limits.symbol,
            start_time=stream_to_datetime(start_ts.encode()),
            end_time=stream_to_datetime(end_ts.encode()),
            start_index=offset,
            end_index=offset + total,
            max_count=total,
            columns=await self.columns(limits.table)
        )

    async def limits(
        self,
        table: str,
        exchange: str,
        symbol: str,
        threshold: dt.timedelta
    ) -> list[TableLimits]:
        self._validate_connection()

        query = f"""
            WITH Gaps AS (
                SELECT 
                    {labels.TIMESTAMP},
                    LAG({labels.TIMESTAMP}) OVER (ORDER BY {labels.TIMESTAMP}) as prev_ts
                FROM {table}
                WHERE {labels.EXCHANGE} = ? AND {labels.SYMBOL} = ?
            ),
            Islands AS (
                SELECT 
                    {labels.TIMESTAMP},
                    CASE 
                        WHEN prev_ts IS NULL THEN 1
                        -- Check if current TS minus previous TS > gap_threshold
                        -- We use strftime/julianday for robust time math in SQLite
                        WHEN (julianday({labels.TIMESTAMP}) - julianday(prev_ts)) * 86400 > ? 
                        THEN 1 ELSE 0 
                    END as is_start
                FROM Gaps
            ),
            Groups AS (
                SELECT 
                    {labels.TIMESTAMP},
                    SUM(is_start) OVER (ORDER BY {labels.TIMESTAMP}) as group_id
                FROM Islands
            )
            SELECT 
                group_id,
                MIN({labels.TIMESTAMP}) as start_time,
                MAX({labels.TIMESTAMP}) as end_time,
                COUNT(*) as count
            FROM Groups
            GROUP BY group_id
            ORDER BY start_time
        """

        threshold_seconds = threshold.total_seconds()
        params = (exchange, symbol, threshold_seconds)
        sequences = []

        # noinspection PyUnresolvedReferences
        async with self.connection.execute(query, params) as cursor:
            async for row in cursor:
                sequences.append(
                    TableLimits(
                        exchange=exchange, symbol=symbol, table=table,
                        start_time=row["start_time"], end_time=row["end_time"],
                        max_count=row["count"]
                    )
                )

        return sequences

    async def tables(self) -> list[str]:
        self._validate_connection()

        # noinspection SqlResolve
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """

        # noinspection PyUnresolvedReferences
        async with self.connection.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [row["name"] for row in rows]

    async def close(self):
        if self.connection:
            await self.connection.close()


type DataKey = tuple[str, str, str]


@dataclass
class DataCompressor:

    db: BaseDatabase
    batch: list[Data] = field(default_factory=list)
    queue: asyncio.Queue[DataKey] = field(default_factory=asyncio.Queue)
    batch_size: int = 10000
    archive_batch_size: int = 50000
    archive_dir: Path | None = None
    retention_period = dt.timedelta(days=3)
    archive_step = dt.timedelta(days=2)
    counts: dict[DataKey, int] = field(default_factory=dict)

    async def compress_clear(
        self,
        table: str,
        exchange: str,
        symbol: str,
        on_tick: Callable[[], ..., ...] = None
    ):
        if self.archive_dir is None:
            return

        limits = await self.db.limit(
            TableLimits(exchange=exchange, symbol=symbol, table=table)
        )

        if not limits.start_time or not limits.end_time:
            return

        if (limits.end_time - limits.start_time) < self.retention_period:
            return

        archive_cutoff = limits.start_time + self.archive_step
        archive_limits = TableLimits(
            exchange=exchange, symbol=symbol, table=table,
            start_time=limits.start_time, end_time=archive_cutoff
        )

        async for _ in self.db.export_parquet(
            self.db.select(archive_limits),
            limits=archive_limits, output_dir=self.archive_dir,
            chunk_size=self.archive_batch_size, on_tick=on_tick
        ):
            pass

    async def collect(self, control: Control | None = None):
        while (control is None) or control.running:
            table, exchange, symbol = await self.queue.get()
            await self.compress_clear(table, exchange, symbol)

    async def insert(self, data: Data):
        table = self.db.table(data)
        key = (table, data[labels.EXCHANGE], data[labels.SYMBOL])

        self.counts[key] = self.counts.get(key, 0) + 1

        if self.counts[key] >= self.batch_size:
            await self.queue.put(key)