from __future__ import annotations

"""
labels.py
=========
Triple-Barrier label generation for training the return-distribution model.

Given a sequence of Snapshots with their realised future price paths (from
db.select_targets_futures), this module:

  1. Sweeps a grid of (TP_mult, SL_mult) configurations.
  2. For each entry point, determines which barrier is hit first.
  3. Selects the optimal (TP*, SL*) that maximises realised P&L net of spread.
  4. Emits (Snapshot, LabelRecord) pairs ready for model training.

References
----------
Lopez de Prado (2018) – Advances in Financial Machine Learning, Ch. 3
Hwang et al. (2023) – Stop-loss adjusted labels for ML trading, Finance Research Letters
"""

import math
from dataclasses import dataclass, field

import numpy as np

from rapid_markets.strategy.features import Snapshot, FEATURE_NAMES

__all__ = [
    "TripleBarrierConfig",
    "LabelRecord",
    "TripleBarrierLabeler",
    "Action",
]


class Action:
    BUY = 0
    SELL = 1
    HOLD = 2
    NAMES = ["buy", "sell", "hold"]

    @staticmethod
    def name(a: int) -> str:
        return Action.NAMES[a]


@dataclass
class TripleBarrierConfig:
    """
    Parameters controlling the label generation sweep.

    TP and SL multipliers are expressed as multiples of the local
    realised volatility estimate (spread_adjusted_vol), so they adapt
    to current market conditions rather than using a fixed pip size.
    """

    # Grid of TP multipliers to sweep
    tp_mults: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0]
    )
    # Grid of SL multipliers to sweep
    sl_mults: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0]
    )
    # Maximum number of future steps (snapshots) before timeout → Hold
    max_steps: int = 200
    # Minimum price move to be considered a Buy/Sell (in bps)
    # Trades with expected return below this after costs → Hold
    min_return_bps: float = 1
    # Transaction cost model: one-way cost as fraction of mid
    # (half-spread at entry + half-spread at exit)
    cost_bps: float = 10

    # Quantiles to use as training targets for the distributional head
    quantiles: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )


@dataclass(slots=True)
class LabelRecord:
    """
    One training sample.

    Fields
    ------
    snapshot:
        The feature vector at entry time.
    action:
        Optimal action (Action.BUY / SELL / HOLD).
    best_tp_mult:
        The TP multiplier that produced the best outcome for the chosen action.
    best_sl_mult:
        The SL multiplier that produced the best outcome for the chosen action.
    best_return:
        Realised net return (after cost) under the optimal (TP, SL) config.
    return_quantiles:
        Dict mapping quantile level → realised return, computed across all
        (TP_mult × SL_mult) configurations — used as distributional targets.
    volatility:
        Local volatility estimate used to scale barriers (σ in price units).
    horizon_steps:
        Number of steps until exit under the optimal config.
    barrier_hit:
        "tp" | "sl" | "timeout"
    """

    snapshot: Snapshot
    action: int
    best_tp_mult: float
    best_sl_mult: float
    best_return: float
    return_quantiles: dict[float, float]  # quantile → realised return
    volatility: float
    horizon_steps: int
    barrier_hit: str

    def feature_array(self) -> np.ndarray:
        return self.snapshot.to_array(FEATURE_NAMES)

    def quantile_array(self, quantiles: list[float]) -> np.ndarray:
        return np.array(
            [self.return_quantiles.get(q, 0.0) for q in quantiles],
            dtype=np.float32
        )


class TripleBarrierLabeler:
    """
    Stateless utility.  Call .label(snapshot, future_prices) for each entry.

    Parameters
    ----------
    future_prices:
        A list of mid prices at future steps [p_1, p_2, ..., p_T].
        This is derived from db.select_targets_futures() in the pipeline.
    """

    def __init__(self, cfg: TripleBarrierConfig | None = None) -> None:
        self.cfg = cfg or TripleBarrierConfig()

    # ------------------------------------------------------------------
    def label(
        self,
        snapshot: Snapshot,
        future_prices: list[float],
    ) -> LabelRecord:
        cfg = self.cfg
        entry_price = snapshot.mid

        # Estimate local vol from feature vector (already computed)
        vol_feature = snapshot.features.get("vol_mid_short", 1e-5)
        # Convert vol (as fraction of mid) back to price units
        sigma = max(vol_feature * entry_price, entry_price * 1e-5)

        # Cost: one-way in price units
        cost = entry_price * cfg.cost_bps / 1e4

        # Limit future path to max_steps
        path = future_prices[: cfg.max_steps]
        n_path = len(path)

        # ------------------------------------------------------------------
        # For each action in {BUY, SELL} × (tp_mult, sl_mult):
        # simulate the trade and compute net return
        # ------------------------------------------------------------------
        # We accumulate all realised returns (across all configs) to build
        # the distributional quantile targets
        all_buy_returns: list[float] = []
        all_sell_returns: list[float] = []

        # Best outcome per action
        best: dict[int, dict] = {
            Action.BUY: {"ret": -math.inf, "tp": 0.0, "sl": 0.0, "steps": n_path, "hit": "timeout"},
            Action.SELL: {"ret": -math.inf, "tp": 0.0, "sl": 0.0, "steps": n_path, "hit": "timeout"},
        }

        for tp_m in cfg.tp_mults:
            for sl_m in cfg.sl_mults:
                tp_dist = tp_m * sigma
                sl_dist = sl_m * sigma

                # --- BUY ---
                buy_tp = entry_price + tp_dist
                buy_sl = entry_price - sl_dist
                b_ret, b_steps, b_hit = self._simulate(
                    path, entry_price, buy_tp, buy_sl, side=1, cost=cost
                )
                all_buy_returns.append(b_ret)
                if b_ret > best[Action.BUY]["ret"]:
                    best[Action.BUY] = {
                        "ret": b_ret, "tp": tp_m, "sl": sl_m,
                        "steps": b_steps, "hit": b_hit
                    }

                # --- SELL ---
                sell_tp = entry_price - tp_dist
                sell_sl = entry_price + sl_dist
                s_ret, s_steps, s_hit = self._simulate(
                    path, entry_price, sell_tp, sell_sl, side=-1, cost=cost
                )
                all_sell_returns.append(s_ret)
                if s_ret > best[Action.SELL]["ret"]:
                    best[Action.SELL] = {
                        "ret": s_ret, "tp": tp_m, "sl": sl_m,
                        "steps": s_steps, "hit": s_hit
                    }

        # Best return for holding (zero cost, zero return)
        hold_return = 0.0

        # ------------------------------------------------------------------
        # Optimal action
        # ------------------------------------------------------------------
        buy_ret = best[Action.BUY]["ret"]
        sell_ret = best[Action.SELL]["ret"]
        min_ret = cfg.min_return_bps / 1e4  # as fraction of price

        if buy_ret >= sell_ret and buy_ret > min_ret:
            action = Action.BUY
        elif sell_ret > buy_ret and sell_ret > min_ret:
            action = Action.SELL
        else:
            action = Action.HOLD

        b = best.get(action, {"tp": 0.0, "sl": 0.0, "ret": 0.0, "steps": 0, "hit": "timeout"})
        if action == Action.HOLD:
            b = {"tp": 0.0, "sl": 0.0, "ret": hold_return, "steps": 0, "hit": "timeout"}

        # ------------------------------------------------------------------
        # Distributional quantile targets
        # Combine buy and sell return distributions. For each action head we
        # want the distribution of returns *under that action's best configs*.
        # We emit a unified distribution (of all returns) as the target for
        # the quantile regression heads — the model learns to predict the
        # full outcome distribution for each action.
        # ------------------------------------------------------------------
        def _quantiles(returns: list[float], qs: list[float]) -> dict[float, float]:
            if not returns:
                return {q: 0.0 for q in qs}
            arr = np.array(returns, dtype=np.float32)
            return {q: float(np.quantile(arr, q)) for q in qs}

        # Build per-action quantile targets
        buy_q = _quantiles(all_buy_returns, cfg.quantiles)
        sell_q = _quantiles(all_sell_returns, cfg.quantiles)
        hold_q = {q: 0.0 for q in cfg.quantiles}  # hold is deterministic 0

        # For the LabelRecord, store the quantiles for the chosen action
        chosen_q = {Action.BUY: buy_q, Action.SELL: sell_q, Action.HOLD: hold_q}[action]

        return LabelRecord(
            snapshot=snapshot,
            action=action,
            best_tp_mult=b["tp"],
            best_sl_mult=b["sl"],
            best_return=b["ret"],
            return_quantiles=chosen_q,
            volatility=sigma,
            horizon_steps=b["steps"],
            barrier_hit=b["hit"],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _simulate(
        path: list[float],
        entry: float,
        tp_price: float,
        sl_price: float,
        side: int,  # +1 = long, -1 = short
        cost: float,
    ) -> tuple[float, int, str]:
        """
        Walk through the price path until TP, SL, or end.
        Returns (net_return_fraction, steps_taken, barrier_hit).
        """
        for i, p in enumerate(path):
            if side == 1:  # Long
                if p >= tp_price:
                    gross = (p - entry) / entry
                    return gross - 2 * cost / entry, i + 1, "tp"
                if p <= sl_price:
                    gross = (p - entry) / entry
                    return gross - 2 * cost / entry, i + 1, "sl"
            else:  # Short
                if p <= tp_price:
                    gross = (entry - p) / entry
                    return gross - 2 * cost / entry, i + 1, "tp"
                if p >= sl_price:
                    gross = (entry - p) / entry
                    return gross - 2 * cost / entry, i + 1, "sl"

        # Timeout: exit at last price
        if path:
            last = path[-1]
            gross = side * (last - entry) / entry
            return gross - 2 * cost / entry, len(path), "timeout"

        return -2 * cost / entry, 0, "timeout"

    # ------------------------------------------------------------------
    def label_batch(
        self,
        entries: list[tuple[Snapshot, list[float]]],
    ) -> list[LabelRecord]:
        """Label a batch of (snapshot, future_path) pairs."""
        return [self.label(snap, path) for snap, path in entries]
