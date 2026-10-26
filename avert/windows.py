"""Causal sliding windows over per-asset feature sequences.

A window ending at (asset a, day t) contains days t-W+1 .. t of asset a only, so the label of
day t is predicted from past and present observations of the same device. The sequence start
is left-padded by repeating the first day. Splits are chronological (or asset-disjoint), never
random, so overlapping windows cannot straddle train and test labels.
"""
from __future__ import annotations

import numpy as np
import torch


class WindowStore:
    def __init__(self, X: np.ndarray, W: int, mean: np.ndarray, std: np.ndarray):
        A, D, F = X.shape
        Xn = (X - mean) / std
        pad = np.repeat(Xn[:, :1], W - 1, axis=1)
        self.Xp = np.concatenate([pad, Xn], axis=1).astype(np.float32)   # [A, D+W-1, F]
        self.W, self.A, self.D, self.F = W, A, D, F
        self._t = torch.from_numpy(self.Xp)
        self._view = self._t.unfold(1, W, 1)                              # [A, D, F, W] (no copy)
        c = np.concatenate([np.zeros((A, 1, F)), np.cumsum(self.Xp, 1)], 1)
        c2 = np.concatenate([np.zeros((A, 1, F)), np.cumsum(self.Xp.astype(np.float64) ** 2, 1)], 1)
        self._c, self._c2 = c, c2

    @classmethod
    def fit(cls, X: np.ndarray, W: int, train_mask: np.ndarray) -> "WindowStore":
        flat = X[train_mask]
        mean, std = flat.mean(0), flat.std(0)
        std[std < 1e-6] = 1.0
        return cls(X, W, mean, std)

    def windows(self, a: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """[B, F, W] channels-first batch for Conv1d; transpose for LSTMs."""
        return self._view[torch.as_tensor(a), torch.as_tensor(t)]

    def summary(self, a: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Window summary features for classical models: last, mean, std, last - mean."""
        W = self.W
        s1 = self._c[a, t + W] - self._c[a, t]
        s2 = self._c2[a, t + W] - self._c2[a, t]
        mean = s1 / W
        std = np.sqrt(np.maximum(s2 / W - mean**2, 0))
        last = self.Xp[a, t + W - 1]
        return np.hstack([last, mean, std, last - mean]).astype(np.float32)


def index_grid(asset_ids, day_lo: int, day_hi: int) -> tuple[np.ndarray, np.ndarray]:
    asset_ids = np.asarray(asset_ids)
    days = np.arange(day_lo, day_hi)
    a = np.repeat(asset_ids, len(days))
    t = np.tile(days, len(asset_ids))
    return a, t
