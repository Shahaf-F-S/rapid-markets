from __future__ import annotations

"""
pipeline.py
===========
Offline data pipeline: reads from MarketDatabase, extracts features,
generates Triple-Barrier labels, and produces train/val/test splits
ready for model training.

This is the bridge between the rapid_markets storage layer and the ML layer.
"""

from dataclasses import dataclass, field

from rapid_markets.store.market import MarketDatabase
from rapid_markets.store.database import TableLimits
from rapid_markets.source.data import Book

from rapid_markets.strategy.features import (
    FeatureConfig, FeatureStore, Snapshot
)
from rapid_markets.strategy.labels import (
    TripleBarrierConfig, TripleBarrierLabeler, LabelRecord, Action
)

__all__ = [
    "PipelineConfig",
    "DataPipeline",
    "DataSplit",
]


@dataclass
class PipelineConfig:
    """
    Controls how the pipeline extracts snapshots and labels from the DB.
    """
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    label_cfg: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)

    # How many future steps to collect for the price path
    # (used as input to the triple-barrier labeler)
    future_steps: int = 200

    # Step interval for down-sampling (1 = use every snapshot)
    # Higher values speed up label generation at cost of resolution
    sample_every: int = 1

    # Train/val/test split fractions (chronological — no shuffling)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implicit: 1 - train_frac - val_frac

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class DataSplit:

    train: list[LabelRecord]
    val: list[LabelRecord]
    test: list[LabelRecord]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_val(self) -> int:
        return len(self.val)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def action_counts(self, split: str = "train") -> dict[str, int]:
        recs = getattr(self, split)
        counts = {Action.name(a): 0 for a in [Action.BUY, Action.SELL, Action.HOLD]}
        for r in recs:
            counts[Action.name(r.action)] += 1
        return counts

    def summary(self) -> str:
        lines = [
            f"DataSplit: train={self.n_train} | val={self.n_val} | test={self.n_test}",
        ]
        for split in ["train", "val", "test"]:
            c = self.action_counts(split)
            lines.append(f"  {split}: {c}")
        return "\n".join(lines)


class DataPipeline:
    """
    Reads the MarketDatabase, produces Snapshots + future price paths,
    applies the Triple-Barrier labeler, and returns chronological splits.

    Usage::

        pipeline = DataPipeline(db, limits, cfg)
        split = await pipeline.run()
        # split.train, split.val, split.test are list[LabelRecord]
    """

    def __init__(
        self,
        db: MarketDatabase,
        limits: TableLimits,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.db = db
        self.limits = limits
        self.cfg = cfg or PipelineConfig()
        self._feature_store = FeatureStore(self.cfg.feature_cfg)
        self._labeler = TripleBarrierLabeler(self.cfg.label_cfg)

    # ------------------------------------------------------------------
    async def run(
        self,
        verbose: bool = True,
        max_records: int | None = None,
    ) -> DataSplit:
        """
        Full pipeline run.
        Returns a DataSplit with train/val/test LabelRecord lists.
        """
        snapshots_with_futures = await self._collect_snapshots(verbose, max_records)

        if verbose:
            print(f"[pipeline] Collected {len(snapshots_with_futures)} snapshot-path pairs")

        # Apply triple-barrier labeling
        records = self._labeler.label_batch(snapshots_with_futures)

        if verbose:
            n = len(records)
            buy = sum(1 for r in records if r.action == Action.BUY)
            sell = sum(1 for r in records if r.action == Action.SELL)
            hold = n - buy - sell
            print(f"[pipeline] Labels: BUY={buy} SELL={sell} HOLD={hold} total={n}")

        return self._split(records)

    # ------------------------------------------------------------------
    async def _collect_snapshots(
        self,
        verbose: bool,
        max_records: int | None,
    ) -> list[tuple[Snapshot, list[float]]]:
        """
        Stream the market replay, extract features into Snapshots, and
        collect a buffer of future mid prices for each entry point.

        Strategy: maintain a sliding buffer of (snapshot, future_price_buffer).
        When the buffer for a past snapshot has grown to `future_steps` prices,
        it is considered complete and added to the output.
        """
        cfg = self.cfg
        n_future = cfg.future_steps

        # Buffer: deque of [snapshot, list_of_future_mids]
        # We fill future mids as subsequent book events come in
        from collections import deque
        pending: deque[tuple[Snapshot, list[float]]] = deque()
        completed: list[tuple[Snapshot, list[float]]] = []

        step = 0
        n_events = 0

        async for event in self.db.simulate_market(self.limits):
            snap = self._feature_store.update(event)

            # Update future price buffers for all pending snapshots
            if isinstance(event, Book) and event.bids and event.asks:
                mid = (event.bids[0][0] + event.asks[0][0]) / 2.0
                for _, future_prices in pending:
                    if len(future_prices) < n_future:
                        future_prices.append(mid)

            # Move completed snapshots out of pending
            while pending and len(pending[0][1]) >= n_future:
                completed.append(pending.popleft())
                if max_records and len(completed) >= max_records:
                    # Drain remaining
                    if verbose:
                        print(f"[pipeline] Hit max_records={max_records}, stopping")
                    return completed

            # Emit new snapshot
            if snap is not None:
                step += 1
                if step % cfg.sample_every == 0:
                    pending.append((snap, []))

            n_events += 1
            if verbose and n_events % 50_000 == 0:
                print(
                    f"[pipeline] {n_events:,} events processed | "
                    f"pending={len(pending)} | completed={len(completed)}"
                )

        # Flush remaining (with whatever future data is available)
        while pending:
            snap, fp = pending.popleft()
            if len(fp) >= cfg.label_cfg.max_steps // 2:  # at least half-window
                completed.append((snap, fp))

        return completed

    # ------------------------------------------------------------------
    def _split(self, records: list[LabelRecord]) -> DataSplit:
        """Chronological train / val / test split (NO shuffling)."""
        n = len(records)
        i_train = int(n * self.cfg.train_frac)
        i_val = int(n * (self.cfg.train_frac + self.cfg.val_frac))

        return DataSplit(
            train=records[:i_train],
            val=records[i_train:i_val],
            test=records[i_val:],
        )
