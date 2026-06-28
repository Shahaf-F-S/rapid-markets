from __future__ import annotations

"""
evaluation.py
=============
Backtest engine and comprehensive evaluation metrics.

The BacktestEngine replays the market data stream through the full
FeatureStore → Model → TradeManager pipeline, recording all trade
events. MetricsReport then computes a rich set of performance metrics.

Design principle: the engine is completely stateless across calls —
you can run it multiple times with different models/configs and compare.
"""

import math
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rapid_markets.store.market import MarketDatabase
from rapid_markets.store.database import TableLimits

from rapid_markets.strategy.features import FeatureConfig, FeatureStore, Snapshot
from rapid_markets.strategy.model import BaseReturnModel
from rapid_markets.strategy.position import (
    Position, TradeEvent, ManagerConfig, TradeManager, ExitReason
)
from rapid_markets.strategy.labels import Action

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "MetricsReport",
    "ModelMetrics",
    "ModelEvaluator"
]

_EPS = 1e-10


@dataclass
class BacktestConfig:
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    manager_cfg: ManagerConfig = field(default_factory=ManagerConfig)
    verbose: bool = False


@dataclass
class ModelMetrics:
    """
    Evaluates the quality of the distributional predictions themselves,
    independent of trading outcomes.
    """
    # Pinball loss per quantile (lower = better calibration)
    pinball_by_quantile: dict[float, float]
    # Direction accuracy: did the model choose the right action?
    direction_accuracy: float
    # Coverage: fraction of true returns falling in [q05, q95]
    coverage_90: float
    # Interval sharpness: mean width of [q05, q95] (lower = sharper)
    interval_width_90: float
    n_samples: int

    def summary(self) -> str:
        avg_pinball = np.mean(list(self.pinball_by_quantile.values()))
        return (
            f"Model Metrics ({self.n_samples} samples)\n"
            f"  Direction accuracy:  {self.direction_accuracy:.3f}\n"
            f"  Avg pinball loss:    {avg_pinball:.6f}\n"
            f"  Coverage [5%,95%]:   {self.coverage_90:.3f}  (target: 0.90)\n"
            f"  Interval width 90%:  {self.interval_width_90:.5f}  (lower = sharper)\n"
            f"  Per-quantile pinball:\n"
            + "\n".join(
            f"    q{q:.2f}: {v:.6f}"
            for q, v in sorted(self.pinball_by_quantile.items())
        )
        )


@dataclass
class MetricsReport:
    """
    Full performance report from one backtest run.
    """
    # --- Trade statistics ---
    n_trades: int
    n_buy: int
    n_sell: int
    win_rate: float  # fraction of profitable trades
    avg_win_bps: float  # average P&L of winning trades (bps)
    avg_loss_bps: float  # average P&L of losing trades (bps)  — negative
    profit_factor: float  # |total wins| / |total losses|

    # --- Return statistics ---
    total_return_bps: float
    sharpe_ratio: float  # annualised
    sortino_ratio: float  # annualised (downside only)
    max_drawdown_bps: float  # maximum peak-to-trough (bps)
    calmar_ratio: float  # total_return / max_drawdown

    # --- Exit breakdown ---
    n_tp: int  # exited at take-profit
    n_sl: int  # exited at stop-loss
    n_flip: int  # exited on signal flip
    n_timeout: int  # exited on timeout
    n_forced: int

    # --- Duration ---
    avg_hold_steps: float
    median_hold_steps: float

    # --- Equity curve ---
    equity_curve: list[float]  # cumulative P&L (bps) per closed trade

    # --- Raw positions ---
    positions: list[Position]

    # --- Model prediction quality (optional) ---
    model_metrics: Optional[ModelMetrics] = None

    # --- Metadata ---
    exchange: str = ""
    symbol: str = ""
    start_time: Optional[dt.datetime] = None
    end_time: Optional[dt.datetime] = None
    n_snapshots: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"BACKTEST REPORT  [{self.exchange}: {self.symbol}]",
            f"  Period:   {self.start_time} → {self.end_time}",
            f"  Snapshots processed: {self.n_snapshots:,}",
            "-" * 60,
            "TRADES",
            f"  Total trades:    {self.n_trades}",
            f"  Buy / Sell:      {self.n_buy} / {self.n_sell}",
            f"  Win rate:        {self.win_rate:.2%}",
            f"  Avg win:         {self.avg_win_bps:.1f} bps",
            f"  Avg loss:        {self.avg_loss_bps:.1f} bps",
            f"  Profit factor:   {self.profit_factor:.2f}",
            "-" * 60,
            "RETURNS",
            f"  Total return:    {self.total_return_bps:.1f} bps",
            f"  Sharpe ratio:    {self.sharpe_ratio:.3f}",
            f"  Sortino ratio:   {self.sortino_ratio:.3f}",
            f"  Max drawdown:    {self.max_drawdown_bps:.1f} bps",
            f"  Calmar ratio:    {self.calmar_ratio:.3f}",
            "-" * 60,
            "EXIT BREAKDOWN",
            f"  Take-profit:     {self.n_tp}  ({self.n_tp / max(self.n_trades, 1):.1%})",
            f"  Stop-loss:       {self.n_sl}  ({self.n_sl / max(self.n_trades, 1):.1%})",
            f"  Signal flip:     {self.n_flip}",
            f"  Timeout:         {self.n_timeout}",
            f"  Forced:          {self.n_forced}",
            "-" * 60,
            "HOLDING PERIOD",
            f"  Avg steps:       {self.avg_hold_steps:.1f}",
            f"  Median steps:    {self.median_hold_steps:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_return_bps": self.total_return_bps,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_bps": self.max_drawdown_bps,
            "calmar_ratio": self.calmar_ratio,
            "n_tp": self.n_tp,
            "n_sl": self.n_sl,
            "avg_hold_steps": self.avg_hold_steps,
        }


class BacktestEngine:
    """
    Replays the market via db.simulate_market() and runs the
    FeatureStore → Model → TradeManager pipeline.

    Call .run() to get a MetricsReport.
    """

    def __init__(
        self,
        db: MarketDatabase,
        limits: TableLimits,
        model: BaseReturnModel,
        cfg: BacktestConfig | None = None,
    ) -> None:
        self.db = db
        self.limits = limits
        self.model = model
        self.cfg = cfg or BacktestConfig()

    # ------------------------------------------------------------------
    async def run(self) -> MetricsReport:
        cfg = self.cfg
        store = FeatureStore(cfg.feature_cfg)
        mgr_cfg = cfg.manager_cfg
        mgr = TradeManager(self.model, mgr_cfg)

        events: list[TradeEvent] = []
        n_snapshots = 0
        first_ts: Optional[dt.datetime] = None
        last_ts: Optional[dt.datetime] = None
        last_snap: Optional[Snapshot] = None
        exchange = self.limits.exchange
        symbol = self.limits.symbol

        async for market_event in self.db.simulate_market(self.limits):
            snap = store.update(market_event)
            if snap is None:
                continue

            n_snapshots += 1
            last_snap = snap
            if first_ts is None:
                first_ts = snap.timestamp
            last_ts = snap.timestamp

            ev_list = mgr.on_snapshot(snap)
            events.extend(ev_list)

            if cfg.verbose and n_snapshots % 10_000 == 0:
                n_open = sum(1 for e in events if e.kind == "open")
                n_close = sum(1 for e in events if e.kind == "close")
                print(
                    f"[backtest] {n_snapshots:,} snapshots | "
                    f"open={n_open} close={n_close}"
                )

        # Force-close any open position at end
        if last_snap is not None:
            events.extend(mgr.force_close_all(last_snap))

        return self._build_report(
            mgr.closed_positions, events, exchange, symbol,
            first_ts, last_ts, n_snapshots
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_report(
        positions: list[Position],
        events: list[TradeEvent],
        exchange: str,
        symbol: str,
        start_time: Optional[dt.datetime],
        end_time: Optional[dt.datetime],
        n_snapshots: int,
    ) -> MetricsReport:

        if not positions:
            # Return empty report
            return MetricsReport(
                n_trades=0, n_buy=0, n_sell=0,
                win_rate=0.0, avg_win_bps=0.0, avg_loss_bps=0.0, profit_factor=0.0,
                total_return_bps=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown_bps=0.0, calmar_ratio=0.0,
                n_tp=0, n_sl=0, n_flip=0, n_timeout=0, n_forced=0,
                avg_hold_steps=0.0, median_hold_steps=0.0,
                equity_curve=[],
                positions=[],
                exchange=exchange, symbol=symbol,
                start_time=start_time, end_time=end_time,
                n_snapshots=n_snapshots,
            )

        pnls_bps = [p.pnl_bps for p in positions if p.pnl_bps is not None]

        wins = [p for p in pnls_bps if p > 0]
        losses = [p for p in pnls_bps if p <= 0]

        win_rate = len(wins) / max(len(pnls_bps), 1)
        avg_win_bps = float(np.mean(wins)) if wins else 0.0
        avg_loss_bps = float(np.mean(losses)) if losses else 0.0
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / (total_losses + _EPS)

        total_return_bps = sum(pnls_bps)

        # Equity curve (cumulative)
        equity_curve = list(np.cumsum(pnls_bps))

        # Sharpe ratio (annualised)
        # Using per-trade returns; assumes each trade is one independent unit
        if len(pnls_bps) > 1:
            mu = np.mean(pnls_bps)
            sig = np.std(pnls_bps, ddof=1) + _EPS
            n_trades_py = max(len(pnls_bps), 1)
            # Approximate annualisation by trade frequency
            sharpe = (mu / sig) * math.sqrt(n_trades_py)
        else:
            sharpe = 0.0

        # Sortino ratio (downside only)
        if len(pnls_bps) > 1:
            downside = [p for p in pnls_bps if p < 0]
            downside_std = (np.std(downside, ddof=1) + _EPS) if len(downside) > 1 else _EPS
            sortino = (np.mean(pnls_bps) / downside_std) * math.sqrt(len(pnls_bps))
        else:
            sortino = 0.0

        # Max drawdown
        cum = np.cumsum(pnls_bps)
        running_max = np.maximum.accumulate(cum)
        drawdowns = cum - running_max
        max_dd_bps = float(np.min(drawdowns))

        calmar = total_return_bps / (abs(max_dd_bps) + _EPS)

        # Exit breakdown
        n_tp = sum(1 for p in positions if p.exit_reason == ExitReason.TP)
        n_sl = sum(1 for p in positions if p.exit_reason == ExitReason.SL)
        n_flip = sum(1 for p in positions if p.exit_reason == ExitReason.SIGNAL_FLIP)
        n_timeout = sum(1 for p in positions if p.exit_reason == ExitReason.TIMEOUT)
        n_forced = sum(1 for p in positions if p.exit_reason == ExitReason.FORCED)

        # Holding period
        hold_steps = [p.n_updates for p in positions]
        avg_hold = float(np.mean(hold_steps)) if hold_steps else 0.0
        med_hold = float(np.median(hold_steps)) if hold_steps else 0.0

        return MetricsReport(
            n_trades=len(positions),
            n_buy=sum(1 for p in positions if p.action == Action.BUY),
            n_sell=sum(1 for p in positions if p.action == Action.SELL),
            win_rate=win_rate,
            avg_win_bps=avg_win_bps,
            avg_loss_bps=avg_loss_bps,
            profit_factor=profit_factor,
            total_return_bps=total_return_bps,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_bps=max_dd_bps,
            calmar_ratio=calmar,
            n_tp=n_tp, n_sl=n_sl, n_flip=n_flip,
            n_timeout=n_timeout, n_forced=n_forced,
            avg_hold_steps=avg_hold,
            median_hold_steps=med_hold,
            equity_curve=equity_curve,
            positions=positions,
            exchange=exchange, symbol=symbol,
            start_time=start_time, end_time=end_time,
            n_snapshots=n_snapshots,
        )


# ------------------------------------------------------------------
# Model quality evaluator (separate from trade performance)
# ------------------------------------------------------------------

class ModelEvaluator:
    """
    Evaluates the distributional prediction quality of a model
    on a held-out test set, independently of trading outcomes.
    """

    def __init__(self, model: BaseReturnModel) -> None:
        self.model = model

    def evaluate(
        self,
        records: list[LabelRecord],  # noqa: F821
        quantiles: list[float] | None = None,
    ) -> ModelMetrics:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        pinball_sums: dict[float, float] = {q: 0.0 for q in quantiles}
        n_correct_dir = 0
        n_covered_90 = 0
        interval_widths: list[float] = []
        n = len(records)

        for rec in records:
            out = self.model.predict(rec.snapshot)
            action_pred, _ = out.best_action()
            true_action = rec.action

            if action_pred == true_action:
                n_correct_dir += 1

            # Pinball loss for the chosen action's head
            q_pred = out.quantiles.get(true_action, {})
            true_return = rec.best_return  # oracle best return

            for q in quantiles:
                pred_q = q_pred.get(q, 0.0)
                err = true_return - pred_q
                if err >= 0:
                    pinball_sums[q] += q * err
                else:
                    pinball_sums[q] += (q - 1.0) * err

            # Coverage: does the 90% interval contain the realised return?
            q05 = q_pred.get(0.05, -math.inf)
            q95 = q_pred.get(0.95, math.inf)
            if q05 <= true_return <= q95:
                n_covered_90 += 1
            interval_widths.append(q95 - q05)

        n_safe = max(n, 1)
        return ModelMetrics(
            pinball_by_quantile={q: s / n_safe for q, s in pinball_sums.items()},
            direction_accuracy=n_correct_dir / n_safe,
            coverage_90=n_covered_90 / n_safe,
            interval_width_90=float(np.mean(interval_widths)) if interval_widths else 0.0,
            n_samples=n,
        )
