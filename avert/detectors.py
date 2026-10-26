"""Telemetry anomaly detectors.

Baselines: domain rules (window robust z-score), Isolation Forest, XGBoost, supervised LSTM.
Proposed:  TCN (supervised), LSTM Autoencoder (label-free training), hybrid TCN + LSTM-AE stacker.

Every detector returns a raw score (higher = more anomalous). Scores are mapped to [0, 1]
anomaly probabilities by Platt scaling fitted on the val_calib split (`calibrate`), which is
what the trust and authorisation engines consume.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .features import FEATURES
from .windows import WindowStore

FI = {f: i for i, f in enumerate(FEATURES)}


@dataclass
class Split:
    a: np.ndarray
    t: np.ndarray
    y: np.ndarray | None = None


@dataclass
class FitContext:
    store: WindowStore
    train: Split
    val: Split
    normal: Split          # label-free training set (commissioning period) for unsupervised models
    cfg: dict
    seed: int


class Detector:
    name = "base"
    supervised = False
    transform = "identity"

    def fit(self, ctx: FitContext):
        return self

    def score(self, store: WindowStore, a, t) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        return 0


# ---------------------------------------------------------------- classical baselines
class RulesDetector(Detector):
    """Max robust deviation of the latest report from its own window and from physical consistency."""
    name = "rules"
    window_channels = ("hours", "fuel_rate_ratio", "rpm_norm", "temp_rise", "idle_frac", "vib_norm")
    level_channels = ("meter_mismatch", "ambient_peer_dev", "hours_peer_dev")

    def score(self, store, a, t):
        F = store.F
        s = store.summary(a, t)
        last, std, dev = s[:, :F], s[:, 2 * F:3 * F], s[:, 3 * F:]
        wi = [FI[c] for c in self.window_channels]
        li = [FI[c] for c in self.level_channels]
        z_win = np.abs(dev[:, wi]) / (std[:, wi] + 0.25)
        z_lvl = np.abs(last[:, li])
        return np.maximum(z_win.max(1), z_lvl.max(1))


class IForestDetector(Detector):
    name = "iforest"

    def fit(self, ctx):
        X = ctx.store.summary(ctx.normal.a, ctx.normal.t)
        self.m = IsolationForest(n_estimators=ctx.cfg["iforest"]["n_estimators"], random_state=ctx.seed,
                                 n_jobs=1).fit(X)
        return self

    def score(self, store, a, t):
        return -self.m.score_samples(store.summary(a, t))


class XGBDetector(Detector):
    name = "xgboost"
    supervised = True
    transform = "logit"

    def fit(self, ctx):
        c = ctx.cfg["xgboost"]
        Xtr, ytr = ctx.store.summary(ctx.train.a, ctx.train.t), ctx.train.y
        Xva, yva = ctx.store.summary(ctx.val.a, ctx.val.t), ctx.val.y
        spw = float(min(20.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())))
        self.m = XGBClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                               learning_rate=c["learning_rate"], subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=30,
                               random_state=ctx.seed, n_jobs=ctx.cfg["deep"]["threads"], tree_method="hist")
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self

    def score(self, store, a, t):
        return self.m.predict_proba(store.summary(a, t))[:, 1]


# ---------------------------------------------------------------- deep models
class CausalBlock(nn.Module):
    """Residual block of two causal dilated convolutions.

    Implemented as Conv2d with a (1, k) kernel on a [B, C, 1, W] tensor: numerically the same as
    Conv1d but ~1.7x faster on CPU in our environment.
    """

    def __init__(self, cin, cout, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv2d(cin, cout, (1, k), dilation=(1, d))
        self.c2 = nn.Conv2d(cout, cout, (1, k), dilation=(1, d))
        self.drop = nn.Dropout(p)
        self.down = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + self.down(x))


class TCN(nn.Module):
    def __init__(self, F, channels=32, kernel=3, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        layers, cin = [], F
        for d in dilations:
            layers.append(CausalBlock(cin, channels, kernel, d, dropout))
            cin = channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):                    # x: [B, F, W]
        return self.head(self.net(x.unsqueeze(2))[:, :, 0, -1]).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, F, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                    # x: [B, F, W]
        _, (h, _) = self.lstm(x.transpose(1, 2))
        return self.head(h[-1]).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, F, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(F, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, F)

    def forward(self, x):                    # x: [B, F, W] -> reconstruction [B, W, F]
        seq = x.transpose(1, 2)
        _, (h, _) = self.enc(seq)
        z = h[-1].unsqueeze(1).repeat(1, seq.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


def _batches(n, bs, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class _Deep(Detector):
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def _setup(self, ctx):
        torch.manual_seed(ctx.seed)
        torch.set_num_threads(ctx.cfg["deep"]["threads"])
        self.rng = np.random.default_rng(ctx.seed)


class _SupervisedDeep(_Deep):
    supervised = True
    transform = "logit"

    def build(self, F, cfg) -> nn.Module:
        raise NotImplementedError

    def fit(self, ctx):
        self._setup(ctx)
        d = ctx.cfg["deep"]
        st, tr, va = ctx.store, ctx.train, ctx.val
        self.model = self.build(st.F, ctx.cfg)
        opt = torch.optim.Adam(self.model.parameters(), lr=d["lr"])
        pw = torch.tensor(min(20.0, (tr.y == 0).sum() / max(1, (tr.y == 1).sum())))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        ytr, yva = torch.tensor(tr.y, dtype=torch.float32), torch.tensor(va.y, dtype=torch.float32)
        best, best_state, bad = np.inf, None, 0
        for _ in range(d["epochs"]):
            self.model.train()
            for b in _batches(len(tr.a), d["batch_size"], self.rng):
                opt.zero_grad()
                loss = lossf(self.model(st.windows(tr.a[b], tr.t[b])), ytr[b])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = sum(lossf(self.model(st.windows(va.a[b], va.t[b])), yva[b]).item() * len(b)
                         for b in _batches(len(va.a), 4096)) / len(va.a)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= d["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def score(self, store, a, t):
        with torch.no_grad():
            out = [torch.sigmoid(self.model(store.windows(a[b], t[b]))).numpy()
                   for b in _batches(len(a), 4096)]
        return np.concatenate(out)


class TCNDetector(_SupervisedDeep):
    name = "tcn"

    def build(self, F, cfg):
        c = cfg["tcn"]
        return TCN(F, c["channels"], c["kernel"], tuple(c["dilations"]), c["dropout"])


class LSTMDetector(_SupervisedDeep):
    name = "lstm"

    def build(self, F, cfg):
        return LSTMClassifier(F, cfg["lstm"]["hidden"])


class LSTMAEDetector(_Deep):
    """Label-free training. Score: reconstruction error of the latest report (last window step).

    train_on: commissioning -> windows from the attack-free commissioning period only
              train_all     -> every training-period window (contaminated, fully unsupervised)
    """
    name = "lstm_ae"
    transform = "log"

    def fit(self, ctx):
        self._setup(ctx)
        d, c = ctx.cfg["deep"], ctx.cfg["lstm_ae"]
        st = ctx.store
        nm = ctx.normal if c.get("train_on", "commissioning") == "commissioning" else ctx.train
        self.model = LSTMAutoencoder(st.F, c["hidden"])
        opt = torch.optim.Adam(self.model.parameters(), lr=d["lr"])
        n_val = max(1, len(nm.a) // 10)
        perm = self.rng.permutation(len(nm.a))
        vi, ti = perm[:n_val], perm[n_val:]
        best, best_state, bad = np.inf, None, 0
        for _ in range(c["epochs"]):
            self.model.train()
            for b in _batches(len(ti), d["batch_size"], self.rng):
                x = st.windows(nm.a[ti[b]], nm.t[ti[b]])
                opt.zero_grad()
                loss = ((self.model(x) - x.transpose(1, 2)) ** 2).mean()
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                x = st.windows(nm.a[vi], nm.t[vi])
                vl = ((self.model(x) - x.transpose(1, 2)) ** 2).mean().item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= d["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def score(self, store, a, t):
        out = []
        with torch.no_grad():
            for b in _batches(len(a), 4096):
                x = store.windows(a[b], t[b])
                err = (self.model(x)[:, -1, :] - x[:, :, -1]) ** 2
                out.append(err.mean(1).numpy())
        return np.concatenate(out)


REGISTRY = {c.name: c for c in (RulesDetector, IForestDetector, XGBDetector, LSTMDetector,
                                TCNDetector, LSTMAEDetector)}


# ---------------------------------------------------------------- calibration and hybrid
def _tf(x, kind):
    x = np.asarray(x, np.float64)
    if kind == "logit":
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    if kind == "log":
        return np.log(np.maximum(x, 1e-9))
    return x


class Platt:
    """Maps a raw detector score to a calibrated anomaly probability (fitted on val_calib)."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, raw, y):
        self.m = LogisticRegression(C=1e3).fit(_tf(raw, self.kind)[:, None], y)
        return self

    def __call__(self, raw):
        return self.m.predict_proba(_tf(raw, self.kind)[:, None])[:, 1]


class HybridStacker:
    """Logistic stacking of two detectors' raw scores, fitted on val_calib.

    Default: TCN probability (logit) + LSTM-AE error (log) = the "hybrid" detector.
    kinds=("logit", "logit"): TCN + XGBoost = the "tcn_xgb" ensemble.
    """
    name = "hybrid"

    def __init__(self, kinds=("logit", "log")):
        self.kinds = kinds

    def fit(self, a_raw, b_raw, y):
        self.m = LogisticRegression(C=1e3).fit(self._x(a_raw, b_raw), y)
        return self

    def _x(self, a_raw, b_raw):
        return np.column_stack([_tf(a_raw, self.kinds[0]), _tf(b_raw, self.kinds[1])])

    def __call__(self, a_raw, b_raw):
        return self.m.predict_proba(self._x(a_raw, b_raw))[:, 1]


def timed_score(det: Detector, store, a, t, n=4096):
    """Raw scores plus mean inference latency per window (ms) measured on up to n windows."""
    k = min(n, len(a))
    t0 = time.perf_counter()
    det.score(store, a[:k], t[:k])
    ms = (time.perf_counter() - t0) * 1000 / max(1, k)
    return det.score(store, a, t), ms
