from __future__ import annotations

"""
features.py
===========
Stateful, per-symbol feature extraction from the raw Book/Trade async stream.

Each incoming Book or Trade event updates a FeatureState, which maintains
rolling windows over recent order-book snapshots and trade events. When both
a book and at least one trade have been seen, a Snapshot (flat dict[str, float])
is emitted and can be fed into the model.

All features are computed from data available strictly at or before the
snapshot timestamp — zero lookahead.
"""

import math
import datetime as dt
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from rapid_markets.source.data import Book, Trade
import rapid_markets.base.labels as L

__all__ = [
    "FeatureConfig",
    "FeatureState",
    "FeatureStore",
    "Snapshot",
    "FEATURE_NAMES",
]


@dataclass
class FeatureConfig:
    """Controls the size of rolling windows and LOB depth used."""

    # How many past Book snapshots to keep for rolling stats
    book_window: int = 50
    # How many past Trades to keep for rolling stats
    trade_window: int = 100
    # Number of LOB levels to use for depth / OBI features
    lob_levels: int = 5
    # Minimum number of trades before emitting snapshots
    min_trades: int = 5
    # Short rolling window (fast signals)
    short_window: int = 10
    # Long rolling window (trend signals)
    long_window: int = 30


# ------------------------------------------------------------------
# Snapshot — the output unit consumed by the model
# ------------------------------------------------------------------

@dataclass(slots=True)
class Snapshot:
    """
    A flat, fully-numeric feature vector at a single point in time.
    All prices are normalised relative to the current mid price so the
    model is scale-invariant across assets and market regimes.
    """

    timestamp: dt.datetime
    exchange: str
    symbol: str
    mid: float  # raw mid price (for TP/SL calculation — NOT fed to model)
    spread: float  # raw spread in price units (for cost filtering)
    features: dict[str, float]  # model input

    def to_array(self, names: list[str]) -> np.ndarray:
        return np.array([self.features[n] for n in names], dtype=np.float32)


# ------------------------------------------------------------------
# All feature names (determines model input dimension)
# ------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # ----- Order-book structure -----
    "obi_1",  # order-book imbalance, top 1 level
    "obi_3",  # order-book imbalance, top 3 levels
    "obi_5",  # order-book imbalance, top 5 levels
    "depth_ratio",  # log(bid_depth_5 / ask_depth_5)
    "spread_bps",  # bid-ask spread in basis points
    "wmid_dev",  # (weighted_mid - mid) / mid  — Stoikov skew
    # ----- Short-horizon book momentum -----
    "wmid_ret_short",  # wmid return over short_window book updates
    "wmid_ret_long",  # wmid return over long_window book updates
    "obi_ma_short",  # moving average of obi_1 over short_window
    "obi_ma_long",  # moving average of obi_1 over long_window
    "spread_ma",  # moving average of spread_bps over short_window
    "spread_z",  # z-score of current spread vs recent spread_ma
    # ----- Volatility (realised) -----
    "vol_mid_short",  # std of wmid returns over short_window (realised vol)
    "vol_mid_long",  # std of wmid returns over long_window
    "vol_ratio",  # vol_mid_short / (vol_mid_long + eps)  — vol regime
    # ----- Trade flow -----
    "trade_flow_short",  # signed volume: Σ signed_qty over short_window trades
    "trade_flow_long",  # signed volume: Σ signed_qty over long_window trades
    "buy_ratio_short",  # buy_vol / total_vol over short_window trades
    "buy_ratio_long",  # buy_vol / total_vol over long_window trades
    "trade_rate",  # trades per second over the last trade_window trades
    "avg_trade_size",  # mean |quantity| over recent trades
    "large_trade_flag",  # 1 if last trade size > 2× avg, else 0
    # ----- VWAP deviation -----
    "vwap_dev",  # (mid - vwap) / mid  — mean-reversion signal
    # ----- Depth imbalance across levels -----
    "bid_depth_skew",  # concentration of bid depth at best vs 5 levels
    "ask_depth_skew",  # concentration of ask depth at best vs 5 levels
]

_N_FEATURES = len(FEATURE_NAMES)
_FEAT_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_EPS = 1e-10


# ------------------------------------------------------------------
# Per-symbol rolling state
# ------------------------------------------------------------------

class FeatureState:
    """
    Maintains rolling windows for one (exchange, symbol) pair.
    Call .update(event) with each Book or Trade in stream order.
    Returns a Snapshot when enough data exists, else None.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg

        # Book history — stores (timestamp, wmid, obi_1, spread_bps)
        self._books: Deque[tuple] = deque(maxlen=cfg.book_window)

        # Trade history — stores (timestamp, signed_qty, price, qty)
        self._trades: Deque[tuple] = deque(maxlen=cfg.trade_window)

        # Latest book snapshot (needed to build combined features)
        self._last_book: Optional[Book] = None

        # Latest trade
        self._last_trade: Optional[Trade] = None

    # ------------------------------------------------------------------
    def update(self, event: Book | Trade) -> Optional[Snapshot]:
        if isinstance(event, Book):
            return self._on_book(event)
        else:
            return self._on_trade(event)

    # ------------------------------------------------------------------
    def _on_book(self, book: Book) -> Optional[Snapshot]:
        if not book.bids or not book.asks:
            return None

        wmid = book.weighted_mid()
        obi1 = book.imbalance(1)
        spbps = book.spread_bps

        self._books.append((book.timestamp, wmid, obi1, spbps))
        self._last_book = book

        # Emit only when we also have enough trades
        if len(self._trades) < self.cfg.min_trades:
            return None
        if len(self._books) < self.cfg.short_window:
            return None

        return self._build_snapshot(book.timestamp, book.exchange, book.symbol)

    def _on_trade(self, trade: Trade) -> None:
        signed = trade.quantity if trade.side == L.BUY else -trade.quantity
        self._trades.append((trade.timestamp, signed, trade.price, trade.quantity))
        self._last_trade = trade
        return None  # snapshots emitted only on book updates

    # ------------------------------------------------------------------
    def _build_snapshot(
        self, ts: dt.datetime, exchange: str, symbol: str
    ) -> Snapshot:
        book = self._last_book
        cfg = self.cfg
        sw, lw = cfg.short_window, cfg.long_window

        # --- Book arrays ---
        book_ts = [b[0] for b in self._books]
        wmids = np.array([b[1] for b in self._books], dtype=np.float64)
        obis = np.array([b[2] for b in self._books], dtype=np.float64)
        spreads = np.array([b[3] for b in self._books], dtype=np.float64)

        cur_wmid = wmids[-1]
        cur_mid = book.mid
        cur_spbps = spreads[-1]

        # --- OBI features ---
        obi1 = book.imbalance(1)
        obi3 = book.imbalance(min(3, book.levels))
        obi5 = book.imbalance(min(cfg.lob_levels, book.levels))

        # --- Depth ratio ---
        bid_d, ask_d = book.depth(cfg.lob_levels)
        depth_ratio = math.log((bid_d + _EPS) / (ask_d + _EPS))

        # --- Spread features ---
        spread_ma = float(np.mean(spreads[-sw:]))
        spread_std = float(np.std(spreads[-sw:]) + _EPS)
        spread_z = (cur_spbps - spread_ma) / spread_std

        # --- Weighted-mid deviation from raw mid ---
        wmid_dev = (cur_wmid - cur_mid) / (cur_mid + _EPS)

        # --- Book momentum (log returns of wmid) ---
        def _wmid_ret(arr: np.ndarray, window: int) -> float:
            if len(arr) < window + 1:
                return 0.0
            past = arr[-window - 1]
            return math.log((arr[-1] + _EPS) / (past + _EPS))

        wmid_ret_s = _wmid_ret(wmids, sw)
        wmid_ret_l = _wmid_ret(wmids, min(lw, len(wmids) - 1))

        # --- OBI moving averages ---
        obi_ma_s = float(np.mean(obis[-sw:]))
        obi_ma_l = float(np.mean(obis[-min(lw, len(obis)):]))

        # --- Realised volatility (std of log returns) ---
        def _vol(arr: np.ndarray, window: int) -> float:
            if len(arr) < window + 1:
                return 0.0
            rets = np.diff(np.log(arr[-window - 1:] + _EPS))
            return float(np.std(rets) + _EPS)

        vol_s = _vol(wmids, sw)
        vol_l = _vol(wmids, min(lw, len(wmids) - 1))
        vol_ratio = vol_s / (vol_l + _EPS)

        # --- Trade features ---
        trades = list(self._trades)
        n_trades = len(trades)

        # Recent windows
        sw_t = min(sw, n_trades)
        lw_t = min(lw, n_trades)
        recent_s = trades[-sw_t:]
        recent_l = trades[-lw_t:]

        signed_s = sum(t[1] for t in recent_s)
        signed_l = sum(t[1] for t in recent_l)
        buy_vol_s = sum(t[3] for t in recent_s if t[1] > 0)
        total_s = sum(t[3] for t in recent_s) + _EPS
        buy_vol_l = sum(t[3] for t in recent_l if t[1] > 0)
        total_l = sum(t[3] for t in recent_l) + _EPS

        buy_ratio_s = buy_vol_s / total_s
        buy_ratio_l = buy_vol_l / total_l

        # Normalise signed volumes by total volume so they're scale-invariant
        trade_flow_s = signed_s / (total_s + _EPS)
        trade_flow_l = signed_l / (total_l + _EPS)

        # Trade rate (trades/sec over full window)
        if n_trades >= 2:
            dt_sec = max((trades[-1][0] - trades[0][0]).total_seconds(), _EPS)
            trade_rate = n_trades / dt_sec
        else:
            trade_rate = 0.0

        # Average and large-trade flag
        sizes = [t[3] for t in recent_l]
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        last_size = trades[-1][3] if trades else 0.0
        large_flag = 1.0 if last_size > 2.0 * avg_size + _EPS else 0.0

        # --- VWAP deviation ---
        prices = np.array([t[2] for t in recent_l], dtype=np.float64)
        qtys = np.array([t[3] for t in recent_l], dtype=np.float64)
        vwap = float(np.dot(prices, qtys) / (qtys.sum() + _EPS))
        vwap_dev = (cur_mid - vwap) / (cur_mid + _EPS)

        # --- Depth skew (best vs. 5-level) ---
        def _depth_skew(side: list, levels: int) -> float:
            if not side:
                return 0.0
            best_notional = side[0][0] * side[0][1]
            total_notional = sum(p * q for p, q in side[:levels]) + _EPS
            return best_notional / total_notional

        bid_skew = _depth_skew(book.bids, cfg.lob_levels)
        ask_skew = _depth_skew(book.asks, cfg.lob_levels)

        # --- Assemble feature dict ---
        feats: dict[str, float] = {
            "obi_1": obi1,
            "obi_3": obi3,
            "obi_5": obi5,
            "depth_ratio": depth_ratio,
            "spread_bps": cur_spbps,
            "wmid_dev": wmid_dev,
            "wmid_ret_short": wmid_ret_s,
            "wmid_ret_long": wmid_ret_l,
            "obi_ma_short": obi_ma_s,
            "obi_ma_long": obi_ma_l,
            "spread_ma": spread_ma,
            "spread_z": spread_z,
            "vol_mid_short": vol_s,
            "vol_mid_long": vol_l,
            "vol_ratio": vol_ratio,
            "trade_flow_short": trade_flow_s,
            "trade_flow_long": trade_flow_l,
            "buy_ratio_short": buy_ratio_s,
            "buy_ratio_long": buy_ratio_l,
            "trade_rate": trade_rate,
            "avg_trade_size": avg_size,
            "large_trade_flag": large_flag,
            "vwap_dev": vwap_dev,
            "bid_depth_skew": bid_skew,
            "ask_depth_skew": ask_skew,
        }

        assert set(feats) == set(FEATURE_NAMES), (
            f"Feature mismatch: {set(feats).symmetric_difference(FEATURE_NAMES)}"
        )

        return Snapshot(
            timestamp=ts,
            exchange=exchange,
            symbol=symbol,
            mid=cur_mid,
            spread=book.spread,
            features=feats,
        )


# ------------------------------------------------------------------
# Multi-symbol store
# ------------------------------------------------------------------

class FeatureStore:
    """
    Holds one FeatureState per (exchange, symbol) pair.
    Call .update(event) and receive Optional[Snapshot].
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()
        self._states: dict[tuple[str, str], FeatureState] = {}

    def _key(self, event: Book | Trade) -> tuple[str, str]:
        return (event.exchange, event.symbol)

    def update(self, event: Book | Trade) -> Optional[Snapshot]:
        key = self._key(event)
        if key not in self._states:
            self._states[key] = FeatureState(self.cfg)
        return self._states[key].update(event)

    def reset(self, exchange: str | None = None, symbol: str | None = None) -> None:
        if exchange is None and symbol is None:
            self._states.clear()
        else:
            to_del = [k for k in self._states if k[0] == exchange or k[1] == symbol]
            for k in to_del:
                del self._states[k]
