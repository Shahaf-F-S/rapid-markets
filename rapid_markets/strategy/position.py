from __future__ import annotations

"""
position.py
===========
Position dataclass and TradeManager.

The TradeManager is event-driven: it receives Snapshot + QuantileOutput
pairs and manages the full lifecycle:

  IDLE → (signal fires) → OPEN → (TP/SL/signal-flip/timeout) → CLOSED

It is source-agnostic — the same class runs in backtest and live.
In backtest, "orders" are filled at current mid; in live, a real broker
adapter would be injected.
"""

import uuid
import datetime as dt
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

from rapid_markets.strategy.features import Snapshot
from rapid_markets.strategy.model import QuantileOutput, BaseReturnModel
from rapid_markets.strategy.labels import Action

__all__ = [
    "PositionStatus",
    "ExitReason",
    "Position",
    "TradeEvent",
    "ManagerConfig",
    "TradeManager",
]


class PositionStatus(Enum):
    IDLE = auto()
    OPEN = auto()
    CLOSED = auto()


class ExitReason(Enum):
    TP = "take_profit"
    SL = "stop_loss"
    SIGNAL_FLIP = "signal_flip"
    TIMEOUT = "timeout"
    FORCED = "forced"


# ------------------------------------------------------------------
# Position
# ------------------------------------------------------------------

@dataclass
class Position:
    id: str
    exchange: str
    symbol: str
    action: int  # Action.BUY or Action.SELL
    entry_price: float
    entry_time: dt.datetime
    quantity: float
    tp: float  # take-profit price
    sl: float  # stop-loss price
    confidence: float  # model's EV score at entry

    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[dt.datetime] = None
    exit_reason: Optional[ExitReason] = None

    # Rolling re-evaluations
    n_updates: int = 0

    @property
    def pnl(self) -> Optional[float]:
        """Realised P&L as fraction of entry price (includes sign)."""
        if self.exit_price is None:
            return None
        if self.action == Action.BUY:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def pnl_bps(self) -> Optional[float]:
        p = self.pnl
        return p * 1e4 if p is not None else None

    @property
    def duration(self) -> Optional[dt.timedelta]:
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

    def is_tp_hit(self, price: float) -> bool:
        if self.action == Action.BUY:
            return price >= self.tp
        else:
            return price <= self.tp

    def is_sl_hit(self, price: float) -> bool:
        if self.action == Action.BUY:
            return price <= self.sl
        else:
            return price >= self.sl


# ------------------------------------------------------------------
# Trade event (emitted for logging / live execution)
# ------------------------------------------------------------------

@dataclass
class TradeEvent:
    kind: str  # "open" | "close" | "update"
    position: Position
    snapshot: Snapshot
    message: str = ""


# ------------------------------------------------------------------
# Manager config
# ------------------------------------------------------------------

@dataclass
class ManagerConfig:
    # Only open a trade when model score > this
    min_ev: float = 0.002  # 2 bps minimum expected return

    # Risk aversion for action selection
    risk_aversion: float = 0.3

    # Max simultaneous positions per symbol
    max_positions: int = 1

    # Spread guard: don't trade when spread > this many bps
    max_spread_bps: float = 5.0

    # Max number of book-update steps before forced exit
    max_hold_steps: int = 500

    # Cooldown: minimum steps between trades on the same symbol
    cooldown_steps: int = 10

    # Trade size: fraction of notional to use
    quantity: float = 1.0  # base units; adapt to your risk model

    # TP/SL derivation params (passed to QuantileOutput.optimal_tp_sl)
    tp_quantile: float = 0.90
    sl_quantile: float = 0.10
    tp_alpha: float = 1.0
    sl_beta: float = 1.0


class TradeManager:
    """
    Stateful manager for one (exchange, symbol) pair.

    Usage::

        mgr = TradeManager(model, cfg)
        for snapshot in stream:
            events = mgr.on_snapshot(snapshot)
            for ev in events:
                handle(ev)
    """

    def __init__(
        self,
        model: BaseReturnModel,
        cfg: ManagerConfig | None = None,
        on_event: Callable[[TradeEvent], None] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or ManagerConfig()
        self.on_event = on_event

        self._position: Optional[Position] = None
        self._steps_since_close: int = self.cfg.cooldown_steps + 1
        self._closed: list[Position] = []

    # ------------------------------------------------------------------
    @property
    def position(self) -> Optional[Position]:
        return self._position

    @property
    def closed_positions(self) -> list[Position]:
        return list(self._closed)

    # ------------------------------------------------------------------
    def on_snapshot(self, snapshot: Snapshot) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        self._steps_since_close += 1

        # Run model
        out = self.model.predict(snapshot)

        if self._position is not None:
            # --- Manage existing position ---
            events += self._manage_open(snapshot, out)
        else:
            # --- Consider opening a new position ---
            events += self._consider_entry(snapshot, out)

        for ev in events:
            if self.on_event:
                self.on_event(ev)

        return events

    # ------------------------------------------------------------------
    def _consider_entry(
        self, snap: Snapshot, out: QuantileOutput
    ) -> list[TradeEvent]:
        cfg = self.cfg

        # Cooldown guard
        if self._steps_since_close < cfg.cooldown_steps:
            return []

        # Spread guard
        if snap.spread / snap.mid * 1e4 > cfg.max_spread_bps:
            return []

        # Choose action
        action, score = out.best_action(risk_aversion=cfg.risk_aversion)

        if action == Action.HOLD:
            return []

        if out.ev[action] < cfg.min_ev:
            return []

        # Compute TP / SL
        tp, sl = out.optimal_tp_sl(
            action=action,
            entry_price=snap.mid,
            tp_quantile=cfg.tp_quantile,
            sl_quantile=cfg.sl_quantile,
            tp_alpha=cfg.tp_alpha,
            sl_beta=cfg.sl_beta,
        )

        pos = Position(
            id=str(uuid.uuid4())[:8],
            exchange=snap.exchange,
            symbol=snap.symbol,
            action=action,
            entry_price=snap.mid,
            entry_time=snap.timestamp,
            quantity=cfg.quantity,
            tp=tp,
            sl=sl,
            confidence=out.ev[action],
        )
        self._position = pos

        ev = TradeEvent(
            kind="open", position=pos, snapshot=snap,
            message=f"{Action.name(action).upper()} | EV={out.ev[action]:.4f} | "
                    f"TP={tp:.2f} SL={sl:.2f}"
            )
        return [ev]

    # ------------------------------------------------------------------
    def _manage_open(
        self, snap: Snapshot, out: QuantileOutput
    ) -> list[TradeEvent]:
        pos = self._position
        pos.n_updates += 1
        price = snap.mid
        events: list[TradeEvent] = []

        # Check TP
        if pos.is_tp_hit(price):
            return [self._close(pos, snap, price, ExitReason.TP)]

        # Check SL
        if pos.is_sl_hit(price):
            return [self._close(pos, snap, price, ExitReason.SL)]

        # Timeout
        if pos.n_updates >= self.cfg.max_hold_steps:
            return [self._close(pos, snap, price, ExitReason.TIMEOUT)]

        # Signal flip: model now strongly favours the opposite side
        opposite = Action.SELL if pos.action == Action.BUY else Action.BUY
        ev_opp = out.ev.get(opposite, 0.0)
        ev_curr = out.ev.get(pos.action, 0.0)
        if ev_opp > ev_curr + self.cfg.min_ev:
            return [self._close(pos, snap, price, ExitReason.SIGNAL_FLIP)]

        # Still holding — emit update event
        events.append(
            TradeEvent(
                kind="update", position=pos, snapshot=snap,
                message=f"Holding | steps={pos.n_updates} | mid={price:.2f} | "
                        f"EV_curr={ev_curr:.4f}"
            )
        )
        return events

    # ------------------------------------------------------------------
    def _close(
        self, pos: Position, snap: Snapshot, price: float, reason: ExitReason
    ) -> TradeEvent:
        pos.exit_price = price
        pos.exit_time = snap.timestamp
        pos.exit_reason = reason
        pos.status = PositionStatus.CLOSED
        self._position = None
        self._closed.append(pos)
        self._steps_since_close = 0

        return TradeEvent(
            kind="close", position=pos, snapshot=snap,
            message=f"EXIT {reason.value} | PnL={pos.pnl_bps:.2f}bps | "
                    f"duration={pos.n_updates} steps"
        )

    # ------------------------------------------------------------------
    def force_close_all(self, snap: Snapshot) -> list[TradeEvent]:
        """Force-close any open position (e.g. end of backtest session)."""
        if self._position is None:
            return []
        return [self._close(self._position, snap, snap.mid, ExitReason.FORCED)]
