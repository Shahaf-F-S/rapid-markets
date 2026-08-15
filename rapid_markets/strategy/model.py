from __future__ import annotations

"""
model.py
========
Return-distribution model: predicts full quantile distributions of returns
for each of the three actions {Buy, Sell, Hold}.

Architecture
------------
  Encoder  : MLP over flat feature vector (FEATURE_NAMES)
  Heads    : 3 × len(quantiles) outputs — one distribution per action

Training objective: Pinball (quantile) loss.

Alternative fast model: LightGBM quantile regression ensemble for rapid
baseline comparison with no GPU needed.

References
----------
Baruník, Hronec, Tobek (2025) – Forecasting stock return distributions with QNNs
Zhang et al. (2020) – DeepLOB (encoder design inspiration)
"""

import math
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rapid_markets.strategy.features import FEATURE_NAMES, Snapshot
from rapid_markets.strategy.labels import LabelRecord, Action, TripleBarrierConfig

__all__ = [
    "ModelConfig",
    "QuantileOutput",
    "BaseReturnModel",
    "QuantileNeuralNet",
    "LightGBMQuantileModel",
    "TrainingResult",
]

_EPS = 1e-8


# ------------------------------------------------------------------
# Output type
# ------------------------------------------------------------------

@dataclass(slots=True)
class QuantileOutput:
    """
    Predicted return quantiles per action, at one point in time.

    quantiles[Action.BUY]  = dict{τ → predicted_return}
    quantiles[Action.SELL] = dict{τ → predicted_return}
    quantiles[Action.HOLD] = dict{τ → predicted_return}  (usually ~0)
    """
    quantiles: dict[int, dict[float, float]]

    # Derived statistics computed on construction
    ev: dict[int, float] = field(default_factory=dict)  # E[R] ≈ q(0.50)
    std: dict[int, float] = field(default_factory=dict)  # spread of distribution
    tail: dict[int, float] = field(default_factory=dict)  # left-tail risk = q(0.05)

    def __post_init__(self) -> None:
        qs = self.quantiles
        for a in [Action.BUY, Action.SELL, Action.HOLD]:
            qd = qs.get(a, {})
            sorted_q = sorted(qd.keys())
            median = qd.get(0.50, 0.0)
            q05 = qd.get(0.05, median)
            q95 = qd.get(0.95, median)
            self.ev[a] = median
            self.std[a] = q95 - q05
            self.tail[a] = q05

    def best_action(self, risk_aversion: float = 0.5) -> tuple[int, float]:
        """
        Choose action maximising:  E[R] - risk_aversion × std(R)

        Returns (action, score).
        """
        scores = {
            a: self.ev[a] - risk_aversion * self.std[a]
            for a in [Action.BUY, Action.SELL, Action.HOLD]
        }
        best = max(scores, key=scores.get)
        return best, scores[best]

    def optimal_tp_sl(
        self,
        action: int,
        entry_price: float,
        tp_quantile: float = 0.90,
        sl_quantile: float = 0.10,
        tp_alpha: float = 1.0,
        sl_beta: float = 1.0,
    ) -> tuple[float, float]:
        """
        Derive TP and SL prices from the predicted return distribution.

        TP = entry × (1 + tp_alpha × q(tp_quantile))   for Buy
        SL = entry × (1 - sl_beta  × |q(sl_quantile)|) for Buy

        Parameters are inverted for Sell.
        Ensures TP and SL are on the correct sides of entry.
        """
        qd = self.quantiles.get(action, {})
        q_tp = qd.get(tp_quantile, 0.0)
        q_sl = qd.get(sl_quantile, 0.0)

        if action == Action.BUY:
            tp = entry_price * (1.0 + tp_alpha * max(q_tp, 0.0))
            sl = entry_price * (1.0 - sl_beta * abs(min(q_sl, 0.0)))
            # Clip so TP > entry > SL
            tp = max(tp, entry_price * 1.0001)
            sl = min(sl, entry_price * 0.9999)
        elif action == Action.SELL:
            tp = entry_price * (1.0 - tp_alpha * max(q_tp, 0.0))
            sl = entry_price * (1.0 + sl_beta * abs(min(q_sl, 0.0)))
            # Clip so TP < entry < SL
            tp = min(tp, entry_price * 0.9999)
            sl = max(sl, entry_price * 1.0001)
        else:
            tp, sl = entry_price, entry_price

        return tp, sl


@dataclass
class ModelConfig:
    # Quantiles to predict
    quantiles: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    # NN architecture
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 64])
    dropout: float = 0.2
    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10  # early stopping patience
    # Risk aversion for action selection (0 = pure EV, 1 = EV - σ)
    risk_aversion: float = 0.3
    # TP/SL derivation parameters
    tp_quantile: float = 0.90
    sl_quantile: float = 0.10
    tp_alpha: float = 1.0
    sl_beta: float = 1.0
    # Device
    device: str = "cpu"


class BaseReturnModel(ABC):
    """Common interface for all model implementations."""

    @abstractmethod
    def predict(self, snapshot: Snapshot) -> QuantileOutput:
        """Predict return distribution for one snapshot."""
        ...

    @abstractmethod
    def fit(
        self,
        records: list[LabelRecord],
        val_records: list[LabelRecord] | None = None,
    ) -> TrainingResult:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...

    def predict_action(
        self, snapshot: Snapshot, risk_aversion: float | None = None
    ) -> tuple[int, float, float, QuantileOutput]:
        """
        Convenience wrapper.
        Returns (action, tp_price, sl_price, full_output).
        """
        out = self.predict(snapshot)
        ra = risk_aversion if risk_aversion is not None else getattr(self, "cfg", ModelConfig()).risk_aversion
        action, score = out.best_action(risk_aversion=ra)
        tp, sl = out.optimal_tp_sl(
            action=action,
            entry_price=snapshot.mid,
            tp_quantile=getattr(self, "cfg", ModelConfig()).tp_quantile,
            sl_quantile=getattr(self, "cfg", ModelConfig()).sl_quantile,
            tp_alpha=getattr(self, "cfg", ModelConfig()).tp_alpha,
            sl_beta=getattr(self, "cfg", ModelConfig()).sl_beta,
        )
        return action, tp, sl, out


@dataclass
class TrainingResult:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    n_train: int
    n_val: int
    feature_names: list[str]

    def summary(self) -> str:
        return (
            f"Trained {self.n_train} samples | Val {self.n_val} | "
            f"Best epoch {self.best_epoch} | "
            f"Train loss {self.train_losses[self.best_epoch]:.6f} | "
            f"Val loss {self.val_losses[self.best_epoch]:.6f}"
        )


# ==================================================================
# 1. Neural Network Implementation
# ==================================================================

class _QuantileDataset(Dataset):
    """PyTorch Dataset wrapping LabelRecords."""

    def __init__(self, records: list[LabelRecord], quantiles: list[float]) -> None:
        self.records = records
        self.quantiles = quantiles
        n = len(FEATURE_NAMES)
        q = len(quantiles)

        # Pre-compute arrays
        self.X = np.zeros((len(records), n), dtype=np.float32)
        # Targets: [n_actions × n_quantiles]
        self.Y_buy = np.zeros((len(records), q), dtype=np.float32)
        self.Y_sell = np.zeros((len(records), q), dtype=np.float32)
        self.Y_hold = np.zeros((len(records), q), dtype=np.float32)
        self.actions = np.zeros(len(records), dtype=np.int64)

        for i, rec in enumerate(records):
            self.X[i] = rec.feature_array()
            self.actions[i] = rec.action
            for j, qval in enumerate(quantiles):
                # We need quantile targets for ALL actions, but labels only
                # contain quantiles for the chosen action. For unchosen
                # actions, we fall back to 0 — the model still trains on them
                # via the multi-head architecture.
                self.Y_buy[i, j] = rec.return_quantiles.get(qval, 0.0) \
                    if rec.action == Action.BUY else 0.0
                self.Y_sell[i, j] = rec.return_quantiles.get(qval, 0.0) \
                    if rec.action == Action.SELL else 0.0
                self.Y_hold[i, j] = 0.0  # hold is always ~0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y_buy[idx]),
            torch.from_numpy(self.Y_sell[idx]),
            torch.from_numpy(self.Y_hold[idx]),
            self.actions[idx],
        )


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Vectorised pinball (quantile) loss.
    pred, target: (batch, n_quantiles)
    quantiles:    (n_quantiles,)
    """
    err = target - pred
    loss = torch.where(
        err >= 0,
        quantiles.unsqueeze(0) * err,
        (quantiles.unsqueeze(0) - 1.0) * err,
    )
    return loss.mean()


def _monotone_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Penalise quantile crossings: q(τ₁) should < q(τ₂) when τ₁ < τ₂."""
    diff = pred[:, 1:] - pred[:, :-1]
    violations = F.relu(-diff)
    return violations.mean()


class _QNNModel(nn.Module):
    """
    MLP encoder → 3 quantile heads (Buy / Sell / Hold).
    """

    def __init__(self, n_features: int, n_quantiles: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_q = n_quantiles

        # Input batch normalisation
        self.bn_in = nn.BatchNorm1d(n_features)

        # Shared encoder
        layers = []
        in_dim = n_features
        for h in cfg.hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = in_dim

        # Three heads — one per action
        self.head_buy = nn.Linear(in_dim, n_quantiles)
        self.head_sell = nn.Linear(in_dim, n_quantiles)
        self.head_hold = nn.Linear(in_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.bn_in(x)
        z = self.encoder(x)
        return self.head_buy(z), self.head_sell(z), self.head_hold(z)


class QuantileNeuralNet(BaseReturnModel):
    """
    Quantile regression neural network.
    Predicts the full return distribution for Buy, Sell, and Hold.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self._net: _QNNModel | None = None
        self._q_tensor: torch.Tensor | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _build_net(self) -> _QNNModel:
        net = _QNNModel(
            n_features=len(FEATURE_NAMES),
            n_quantiles=len(self.cfg.quantiles),
            cfg=self.cfg,
        )
        return net.to(self.cfg.device)

    def _q_t(self) -> torch.Tensor:
        if self._q_tensor is None:
            self._q_tensor = torch.tensor(
                self.cfg.quantiles, dtype=torch.float32, device=self.cfg.device
            )
        return self._q_tensor

    # ------------------------------------------------------------------
    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feature_mean) / (self._feature_std + _EPS)

    def _compute_norm_stats(self, X: np.ndarray) -> None:
        self._feature_mean = X.mean(axis=0).astype(np.float32)
        self._feature_std = X.std(axis=0).astype(np.float32)
        # Prevent zero std
        self._feature_std = np.where(self._feature_std < _EPS, 1.0, self._feature_std)

    # ------------------------------------------------------------------
    def fit(
        self,
        records: list[LabelRecord],
        val_records: list[LabelRecord] | None = None,
    ) -> TrainingResult:
        cfg = self.cfg
        device = cfg.device

        # --- Normalise features ---
        X_all = np.stack([r.feature_array() for r in records])
        self._compute_norm_stats(X_all)

        # Patch records with normalised features (we operate on a copy)
        # We instead pass raw and normalise inside the net (BatchNorm)
        # so _compute_norm_stats is stored for inference normalisation

        # --- Datasets ---
        train_ds = _QuantileDataset(records, cfg.quantiles)
        val_ds = _QuantileDataset(val_records, cfg.quantiles) if val_records else None

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
            )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False
            ) if val_ds else None

        # --- Build model ---
        self._net = self._build_net()
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs
        )

        q_tensor = self._q_t()
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = math.inf
        best_epoch = 0
        best_state = None

        for epoch in range(cfg.max_epochs):
            # Train
            self._net.train()
            total_loss = 0.0
            n_batches = 0
            for X, Y_buy, Y_sell, Y_hold, actions in train_loader:
                X = X.to(device)
                Y_buy = Y_buy.to(device)
                Y_sell = Y_sell.to(device)
                Y_hold = Y_hold.to(device)
                actions = actions.to(device)

                pred_buy, pred_sell, pred_hold = self._net(X)

                # Buy head: only on Buy-labelled samples
                mask_buy = (actions == Action.BUY)
                mask_sell = (actions == Action.SELL)

                loss = torch.tensor(0.0, device=device, requires_grad=True)
                mono = torch.tensor(0.0, device=device)

                if mask_buy.any():
                    l_buy = _pinball_loss(pred_buy[mask_buy], Y_buy[mask_buy], q_tensor)
                    m_buy = _monotone_penalty(pred_buy[mask_buy])
                    loss = loss + l_buy + 0.1 * m_buy
                    mono = mono + m_buy

                if mask_sell.any():
                    l_sell = _pinball_loss(pred_sell[mask_sell], Y_sell[mask_sell], q_tensor)
                    m_sell = _monotone_penalty(pred_sell[mask_sell])
                    loss = loss + l_sell + 0.1 * m_sell

                # Hold head: train on all samples toward zero
                l_hold = _pinball_loss(pred_hold, Y_hold, q_tensor)
                loss = loss + 0.5 * l_hold

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_loss = total_loss / max(n_batches, 1)
            train_losses.append(epoch_loss)

            # Validation
            if val_loader:
                self._net.eval()
                val_total = 0.0
                val_n = 0
                with torch.no_grad():
                    for X, Y_buy, Y_sell, Y_hold, actions in val_loader:
                        X = X.to(device)
                        Y_buy = Y_buy.to(device)
                        Y_sell = Y_sell.to(device)
                        Y_hold = Y_hold.to(device)
                        actions = actions.to(device)
                        pb, ps, ph = self._net(X)

                        mask_b = (actions == Action.BUY)
                        mask_s = (actions == Action.SELL)
                        vl = torch.tensor(0.0, device=device)
                        if mask_b.any():
                            vl = vl + _pinball_loss(pb[mask_b], Y_buy[mask_b], q_tensor)
                        if mask_s.any():
                            vl = vl + _pinball_loss(ps[mask_s], Y_sell[mask_s], q_tensor)
                        vl = vl + 0.5 * _pinball_loss(ph, Y_hold, q_tensor)
                        val_total += vl.item()
                        val_n += 1
                val_loss = val_total / max(val_n, 1)
            else:
                val_loss = epoch_loss
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

            # Early stopping
            if epoch - best_epoch >= cfg.patience:
                break

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            n_train=len(records),
            n_val=len(val_records) if val_records else 0,
            feature_names=FEATURE_NAMES,
        )

    # ------------------------------------------------------------------
    def predict(self, snapshot: Snapshot) -> QuantileOutput:
        if self._net is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
        self._net.eval()

        x = torch.tensor(
            snapshot.to_array(FEATURE_NAMES), dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)

        with torch.no_grad():
            pb, ps, ph = self._net(x)

        qs = self.cfg.quantiles

        def _to_dict(t: torch.Tensor) -> dict[float, float]:
            vals = t.squeeze(0).cpu().numpy().tolist()
            return {q: v for q, v in zip(qs, vals)}

        return QuantileOutput(
            quantiles={
                Action.BUY: _to_dict(pb),
                Action.SELL: _to_dict(ps),
                Action.HOLD: _to_dict(ph),
            }
        )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict() if self._net else None,
                "cfg": self.cfg,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
                "feature_names": FEATURE_NAMES,
            }, path
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self._feature_mean = ckpt["feature_mean"]
        self._feature_std = ckpt["feature_std"]
        self._net = self._build_net()
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()


# ==================================================================
# 2. LightGBM Quantile Ensemble (fast baseline, no GPU needed)
# ==================================================================

class LightGBMQuantileModel(BaseReturnModel):
    """
    Fast gradient-boosting baseline using LightGBM quantile regression.
    Trains one model per (action, quantile) — fully parallel via joblib.
    No GPU required. Much faster to train than the NN for initial experiments.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self._models: dict[tuple[int, float], object] = {}
        self._lgb_params: dict = {
            "objective": "quantile",
            "metric": "quantile",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }

    def fit(
        self,
        records: list[LabelRecord],
        val_records: list[LabelRecord] | None = None,
    ) -> TrainingResult:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")

        cfg = self.cfg
        X_all = np.stack([r.feature_array() for r in records])

        # For each action, extract the subset of records
        buy_mask = np.array([r.action == Action.BUY for r in records])
        sell_mask = np.array([r.action == Action.SELL for r in records])

        train_losses: list[float] = []

        for action, mask in [(Action.BUY, buy_mask), (Action.SELL, sell_mask)]:
            if not mask.any():
                continue
            X_a = X_all[mask]
            for q in cfg.quantiles:
                Y_a = np.array(
                    [
                        r.return_quantiles.get(q, 0.0)
                        for r, m in zip(records, mask) if m
                    ], dtype=np.float32
                )

                params = {**self._lgb_params, "alpha": q}
                model = lgb.LGBMRegressor(**params)

                if val_records:
                    val_mask = np.array([r.action == action for r in val_records])
                    if val_mask.any():
                        X_v = np.stack([r.feature_array() for r in val_records])[val_mask]
                        Y_v = np.array(
                            [
                                r.return_quantiles.get(q, 0.0)
                                for r, m in zip(val_records, val_mask) if m
                            ], dtype=np.float32
                        )
                        model.fit(
                            X_a, Y_a,
                            eval_set=[(X_v, Y_v)],
                            callbacks=[lgb.early_stopping(20, verbose=False)]
                            )
                    else:
                        model.fit(X_a, Y_a)
                else:
                    model.fit(X_a, Y_a)

                self._models[(action, q)] = model
                train_losses.append(0.0)  # LightGBM handles loss internally

        # Hold head — trivially predict 0 (we still store a stub)
        for q in cfg.quantiles:
            self._models[(Action.HOLD, q)] = None  # signals: return 0.0

        n_val = len(val_records) if val_records else 0
        return TrainingResult(
            train_losses=train_losses,
            val_losses=[0.0] * len(train_losses),
            best_epoch=0,
            n_train=len(records),
            n_val=n_val,
            feature_names=FEATURE_NAMES,
        )

    def predict(self, snapshot: Snapshot) -> QuantileOutput:
        x = pd.DataFrame([snapshot.to_array(FEATURE_NAMES)], columns=FEATURE_NAMES)
        result: dict[int, dict[float, float]] = {}

        for action in [Action.BUY, Action.SELL, Action.HOLD]:
            qd: dict[float, float] = {}
            for q in self.cfg.quantiles:
                model = self._models.get((action, q))
                if model is None:
                    qd[q] = 0.0
                else:
                    qd[q] = float(model.predict(x)[0])
            # Enforce monotonicity by sorting values
            sorted_vals = sorted(qd.values())
            result[action] = {q: v for q, v in zip(sorted(qd.keys()), sorted_vals)}

        return QuantileOutput(quantiles=result)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models": self._models, "cfg": self.cfg}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._models = data["models"]
        self.cfg = data["cfg"]
