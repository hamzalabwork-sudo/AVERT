"""Device trust models, vectorised over assets. T in [0, 1], higher = more trusted.

All models receive the same per-report evidence: the calibrated anomaly score s in [0, 1].
The adaptive model additionally uses (each switchable for ablation):
  asymmetric  - negative evidence weighted by kappa > 1 (trust is lost faster than gained)
  exposure    - negative evidence scaled by the claim's size relative to expected usage
  hard        - hard security violations add a fixed penalty
  feedback    - outcomes of held-transaction verification are fed back as strong evidence
Trust is always read *before* the current report updates it, so a low-trust device cannot be
auto-approved just because its current anomaly score is low.
"""
from __future__ import annotations

import numpy as np


class TrustModel:
    name = "base"

    def __init__(self, n: int, cfg: dict):
        self.n, self.cfg = n, cfg
        a0, b0 = cfg["prior"]
        self.t0 = a0 / (a0 + b0)

    def value(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, s, hard, exp_w, ver_fraud, ver_legit, active):
        pass


class NoTrust(TrustModel):
    name = "none"

    def value(self):
        return np.ones(self.n)


class StaticTrust(TrustModel):
    name = "static"

    def value(self):
        return np.full(self.n, self.t0)


class EWMATrust(TrustModel):
    name = "ewma"

    def __init__(self, n, cfg):
        super().__init__(n, cfg)
        self.T = np.full(n, self.t0)
        self.lam = cfg["ewma"]["lam"]

    def value(self):
        return self.T.copy()

    def update(self, s, hard, exp_w, ver_fraud, ver_legit, active):
        self.T = np.where(active, self.lam * self.T + (1 - self.lam) * (1 - s), self.T)


class BetaTrust(TrustModel):
    """Beta reputation with forgetting (Josang & Ismail, 2002); symmetric evidence."""
    name = "beta"

    def __init__(self, n, cfg, gamma=None):
        super().__init__(n, cfg)
        self.a0, self.b0 = cfg["prior"]
        self.a = np.full(n, float(self.a0))
        self.b = np.full(n, float(self.b0))
        self.gamma = gamma if gamma is not None else cfg["beta"]["gamma"]

    def value(self):
        return self.a / (self.a + self.b)

    def _decay(self, active):
        g = self.gamma
        self.a = np.where(active, self.a0 + g * (self.a - self.a0), self.a)
        self.b = np.where(active, self.b0 + g * (self.b - self.b0), self.b)

    def update(self, s, hard, exp_w, ver_fraud, ver_legit, active):
        self._decay(active)
        self.a += active * (1 - s)
        self.b += active * s


class AdaptiveTrust(BetaTrust):
    name = "adaptive"

    def __init__(self, n, cfg, asymmetric=True, exposure=True, hard=True, feedback=True):
        super().__init__(n, cfg, gamma=cfg["adaptive"]["gamma"])
        c = cfg["adaptive"]
        self.kappa = c["kappa"] if asymmetric else 1.0
        self.use_exp, self.use_hard, self.use_fb = exposure, hard, feedback
        self.cap, self.hard_pen, self.vw = c["exposure_cap"], c["hard_penalty"], c["verify_weight"]

    def update(self, s, hard, exp_w, ver_fraud, ver_legit, active):
        self._decay(active)
        w = np.clip(exp_w, 0.5, self.cap) if self.use_exp else 1.0
        self.a += active * (1 - s)
        self.b += active * self.kappa * w * s
        if self.use_hard:
            self.b += self.hard_pen * hard
        if self.use_fb:
            self.b += self.vw * ver_fraud
            self.a += self.vw * ver_legit


def make_trust(name: str, n: int, cfg: dict, opts: dict | None = None) -> TrustModel:
    opts = opts or {}
    return {"none": NoTrust, "static": StaticTrust, "ewma": EWMATrust, "beta": BetaTrust,
            "adaptive": AdaptiveTrust}[name](n, cfg, **opts)
