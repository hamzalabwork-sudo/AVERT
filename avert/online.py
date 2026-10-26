"""Streaming (online) implementation of the deterministic trust-aware authorisation (DTA) pipeline.

A batch pipeline computes features over the whole timeline at once. An adaptive adversary interacts with
the defence day by day, so this module provides:

  OnlineFeatureEngine   updates the 19 features of avert.features from one day's reports; `step` evaluates
                        a (possibly hypothetical) day without changing state, `commit` applies it.
  OnlineDetector        scores the day's causal windows with the trained TCN + XGBoost stack.
  OnlineDefender        the DTA trust / risk / decision / verification logic, one day at a time.

The engine is initialised from the batch pipeline on days < t0 and is verified to reproduce the batch
features exactly on the remaining days (tests/test_online.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .authorisation import APPROVE, HOLD, REJECT
from .features import CUSUM_SLACK, DRIFT_DAYS, FEATURES, build_features
from .generator import DROPOUT_CHANNELS
from .trust import make_trust

CH = list(DROPOUT_CHANNELS)
REPORT_FIELDS = ["reported_hours", "meter_delta", *CH, "delay_h"]
FI = {f: i for i, f in enumerate(FEATURES)}


def day_reports(records: pd.DataFrame, day: int, prefix: str = "") -> dict:
    """Arrays (asset order) of one day's report fields. prefix='h_' reads the honest (pre-attack) copy."""
    d = records[records.day == day].sort_values("asset_id")
    out = {}
    for f in REPORT_FIELDS:
        col = f if (f == "delay_h" or not prefix) else prefix + f
        out[f] = d[col].to_numpy(dtype=float).copy()
    out["day"] = day
    return out


def _site_stat(values: np.ndarray, site: np.ndarray, sites: list[np.ndarray], fn) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for idx in sites:
        out[idx] = fn(values[idx])
    return out


def _nanmedian_or_nan(x):
    x = x[~np.isnan(x)]
    return np.median(x) if x.size else np.nan


@dataclass
class _Pending:
    filled: np.ndarray
    worked: np.ndarray
    dev: np.ndarray
    drift: np.ndarray
    cus_raw: np.ndarray
    frr: np.ndarray
    peer_med: np.ndarray
    baseline: np.ndarray
    day: int


class OnlineFeatureEngine:
    def __init__(self, obs: pd.DataFrame, assets: pd.DataFrame, commissioning_days: int, t0: int):
        A = len(assets)
        self.A, self.t0, self.comm = A, t0, commissioning_days
        meta = assets.sort_values("asset_id")
        self.site = meta.site.to_numpy()
        self.sites = [np.flatnonzero(self.site == s) for s in np.unique(self.site)]
        self.lph = meta.fuel_lph.to_numpy(float)
        self.idle_rpm = meta.idle_rpm.to_numpy(float)
        self.rated_rpm = meta.rated_rpm.to_numpy(float)
        self.vib = meta.vib.to_numpy(float)

        hist = obs[obs.day < t0].sort_values(["asset_id", "day"]).reset_index(drop=True)
        feat = build_features(hist, assets, commissioning_days)
        F = feat[list(FEATURES)].to_numpy(float).reshape(A, t0, len(FEATURES))
        self.batch_hist = F                                        # [A, t0, F] (for detector windows)

        # last filled channel values (ffill then leading bfill, as in build_features)
        filled = hist.groupby("asset_id")[CH].ffill().bfill()
        self.last_ch = filled.to_numpy(float).reshape(A, t0, len(CH))[:, -1, :]

        h = hist.reported_hours.to_numpy(float).reshape(A, t0)
        worked = h > 0.2
        dev = F[:, :, FI["usage_dev"]]
        # commissioning ratio r_i (identical to features.expected_usage)
        hw = np.where(worked, h, np.nan)
        peer = np.zeros((A, t0))
        for t in range(min(t0, commissioning_days)):
            peer[:, t] = np.nan_to_num(_site_stat(hw[:, t], self.site, self.sites, _nanmedian_or_nan), nan=0.0)
        comm = worked & (peer > 0.2) & (np.arange(t0)[None, :] < commissioning_days)
        ratio = np.where(comm, h / np.where(peer > 0.2, peer, np.nan), np.nan)
        self.r = np.array([np.median(x[~np.isnan(x)]) if (~np.isnan(x)).any() else 1.0 for x in ratio])
        frr = np.where(worked, F[:, :, FI["fuel_rate_ratio"]], np.nan)
        cm = np.arange(t0) < commissioning_days
        with np.errstate(all="ignore"):
            self.comm_frr = np.array([np.nanmean(frr[i, cm]) if (~np.isnan(frr[i, cm])).any() else np.nan
                                      for i in range(A)])

        # drift buffer (last 28 worked-day deviations), raw CUSUM state, carried feature values
        self.buf = np.full((A, DRIFT_DAYS), np.nan)
        self.cus_raw = np.zeros(A)
        for i in range(A):
            w = dev[i, worked[i]]
            if w.size:
                tail = w[-DRIFT_DAYS:]
                self.buf[i, -tail.size:] = tail
                s = 0.0
                for x in w:
                    s = max(0.0, s + x - CUSUM_SLACK)
                self.cus_raw[i] = s
        self.last_drift = F[:, -1, FI["usage_drift28"]].copy()
        self.frr_buf = frr[:, -DRIFT_DAYS:].copy()
        if self.frr_buf.shape[1] < DRIFT_DAYS:
            pad = np.full((A, DRIFT_DAYS - self.frr_buf.shape[1]), np.nan)
            self.frr_buf = np.concatenate([pad, self.frr_buf], 1)
        self.day = t0

    # ------------------------------------------------------------------
    def step(self, rep: dict) -> tuple[np.ndarray, _Pending]:
        """Features [A, F] for one day's reports, without changing the engine state."""
        A = self.A
        raw = np.column_stack([rep[c] for c in CH])
        missing = np.isnan(raw).sum(1)
        filled = np.where(np.isnan(raw), self.last_ch, raw)
        fuel, rpm, temp, vibr, amb, idle = (filled[:, k] for k in range(len(CH)))
        h = np.asarray(rep["reported_hours"], float)
        meter = np.asarray(rep["meter_delta"], float)
        worked = h > 0.2
        hc = np.maximum(h, 0.2)
        f = np.zeros((A, len(FEATURES)))
        f[:, FI["hours"]] = h
        f[:, FI["worked"]] = worked
        f[:, FI["meter_mismatch"]] = h - meter
        frr_all = np.clip(np.where(worked, fuel / (hc * self.lph), 0.0), 0, 3)
        f[:, FI["fuel_rate_ratio"]] = frr_all
        f[:, FI["fuel_norm"]] = fuel / self.lph
        f[:, FI["rpm_norm"]] = np.clip(np.where(worked, (rpm - self.idle_rpm) / (self.rated_rpm - self.idle_rpm), 0.0), -1, 2)
        f[:, FI["temp_rise"]] = temp - amb
        f[:, FI["idle_frac"]] = np.clip(np.where(worked, idle / hc, 0.0), 0, 1)
        f[:, FI["vib_norm"]] = vibr / self.vib
        f[:, FI["ambient_peer_dev"]] = amb - _site_stat(amb, self.site, self.sites, _nanmedian_or_nan)
        f[:, FI["hours_peer_dev"]] = h - _site_stat(h, self.site, self.sites, np.median)
        f[:, FI["log_delay"]] = np.log1p(np.maximum(np.asarray(rep["delay_h"], float), 0))
        f[:, FI["missing_count"]] = missing
        dow = rep["day"] % 7
        f[:, FI["dow_sin"]] = np.sin(2 * np.pi * dow / 7)
        f[:, FI["dow_cos"]] = np.cos(2 * np.pi * dow / 7)

        peer = np.nan_to_num(_site_stat(np.where(worked, h, np.nan), self.site, self.sites, _nanmedian_or_nan), nan=0.0)
        base = np.where(peer > 0.2, self.r * peer, 0.0)
        active = worked | (peer > 0.2)
        dev = np.clip(np.where(active, (h - base) / np.maximum(base, 1.0), 0.0), -1, 3)
        f[:, FI["usage_dev"]] = dev
        with np.errstate(all="ignore"):
            drift_w = np.nanmean(np.concatenate([self.buf[:, 1:], dev[:, None]], 1), axis=1)
            drift = np.where(worked, drift_w, self.last_drift)
            cus_raw = np.where(worked, np.maximum(0.0, self.cus_raw + dev - CUSUM_SLACK), self.cus_raw)
            frr_t = np.where(worked, frr_all, np.nan)
            win = np.concatenate([self.frr_buf[:, 1:], frr_t[:, None]], 1)
            roll = np.where(np.isnan(win).all(1), np.nan, np.nanmean(win, axis=1))
            fd = np.clip(roll / self.comm_frr - 1, -2, 2)
        f[:, FI["usage_drift28"]] = drift
        f[:, FI["usage_cusum"]] = np.clip(cus_raw, 0, 20)
        f[:, FI["fuel_drift28"]] = fd
        f = np.nan_to_num(f, nan=0.0)
        return f, _Pending(filled, worked, dev, drift, cus_raw, frr_t, peer, base, rep["day"])

    def commit(self, p: _Pending):
        self.last_ch = p.filled
        w = p.worked
        if w.any():
            self.buf[w] = np.concatenate([self.buf[w, 1:], p.dev[w, None]], 1)
        self.last_drift = p.drift
        self.cus_raw = p.cus_raw
        self.frr_buf = np.concatenate([self.frr_buf[:, 1:], p.frr[:, None]], 1)
        self.day = p.day + 1


class OnlineDetector:
    """TCN + XGBoost stacked detector on streaming causal windows."""

    def __init__(self, tcn, xgb, stacker, mean, std, W: int, hist_raw: np.ndarray, store=None, t0=None):
        self.tcn, self.xgb, self.stack = tcn, xgb, stacker
        self.mean, self.std, self.W = mean.astype(np.float32), std.astype(np.float32), W
        z = ((hist_raw.astype(np.float32) - self.mean) / self.std).astype(np.float32)
        if z.shape[1] < W - 1:
            z = np.concatenate([np.repeat(z[:, :1], W - 1 - z.shape[1], 1), z], 1)
        self.ring = z[:, -(W - 1):].copy()                         # last W-1 standardised rows
        # Window summaries reproduce WindowStore.summary bit-for-bit: the batch pipeline obtains window
        # sums from running cumulative sums (float32 for the mean, float64 for the squares).
        self.C = self.C2 = None
        if store is not None:
            A, F = self.ring.shape[0], self.ring.shape[2]
            c32 = np.concatenate([np.zeros((A, 1, F), np.float32), np.cumsum(store.Xp, 1)], 1)
            self.C = c32[:, t0:t0 + W].copy()                      # c[t .. t+W-1], float32
            self.C2 = store._c2[:, t0:t0 + W].copy()               # float64

    def standardise(self, feat):
        return ((feat.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def score(self, feat: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        z = self.standardise(feat if idx is None else feat[idx])
        ring = self.ring if idx is None else self.ring[idx]
        win = np.concatenate([ring, z[:, None, :]], 1)             # [B, W, F]
        with torch.no_grad():
            raw_tcn = torch.sigmoid(self.tcn.model(torch.from_numpy(np.ascontiguousarray(win.transpose(0, 2, 1))))).numpy()
        W = self.W
        if self.C is not None:
            C = self.C if idx is None else self.C[idx]
            C2 = self.C2 if idx is None else self.C2[idx]
            c_end = (C[:, -1] + z).astype(np.float32)
            c2_end = C2[:, -1] + z.astype(np.float64) ** 2
            s1 = c_end.astype(np.float64) - C[:, 0].astype(np.float64)
            s2 = c2_end - C2[:, 0]
            m = s1 / W
            sd = np.sqrt(np.maximum(s2 / W - m ** 2, 0))
        else:
            w64 = win.astype(np.float64)
            m = w64.mean(1)
            sd = np.sqrt(np.maximum((w64 ** 2).mean(1) - m ** 2, 0))
        last = z
        summ = np.hstack([last, m, sd, last - m]).astype(np.float32)
        raw_xgb = self.xgb.m.predict_proba(summ)[:, 1]
        return self.stack(raw_tcn, raw_xgb)

    def commit(self, feat: np.ndarray):
        z = self.standardise(feat)
        self.ring = np.concatenate([self.ring[:, 1:], z[:, None, :]], 1)
        if self.C is not None:
            c_end = (self.C[:, -1] + z).astype(np.float32)
            c2_end = self.C2[:, -1] + z.astype(np.float64) ** 2
            self.C = np.concatenate([self.C[:, 1:], c_end[:, None, :]], 1)
            self.C2 = np.concatenate([self.C2[:, 1:], c2_end[:, None, :]], 1)


class OnlineDefender:
    """DTA decision layer, one reporting day at a time (mirrors authorisation.simulate)."""

    def __init__(self, spec, cfg, theta, n, seed):
        self.spec, self.cfg, self.theta = spec, cfg, theta
        self.trust = make_trust(spec.trust, n, cfg["trust"], spec.trust_opts)
        self.rng = np.random.default_rng(seed)
        self.pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.n = n
        self.d = 0
        au = cfg["authorisation"]
        self.p_det, self.v_delay = au["verification"]["p_detect"], au["verification"]["delay_days"]
        self.tol = cfg["generator"]["contract"]["fraud_tolerance_hours"]

    def risk(self, T, s):
        return 1 - T * (1 - s)                                     # v2 design: noisy-OR without F

    def decide(self, T, s, hard, h):
        ta, tr = self.theta
        R = self.risk(T, s)
        dec = np.where(R < ta, APPROVE, np.where(R < tr, HOLD, REJECT))
        if self.spec.gate:
            dec = np.where(T < self.cfg["trust"]["low"], np.maximum(dec, HOLD), dec)
        if self.spec.hard:
            dec = np.where(hard, REJECT, dec)
        return np.where(h > 0, dec, APPROVE), R

    def step(self, h, rate, ref_h, s, hard, baseline):
        """One day. Returns dict with decisions and settlement amounts (true usage known to the verifier)."""
        n = self.n
        T = self.trust.value()
        is_claim = h > 0
        dec, R = self.decide(T, s, hard, h)
        over_h = h - ref_h
        fraud = over_h > self.tol                                  # what an independent verification establishes
        verified = self.rng.random(n) < self.p_det
        held = dec == HOLD
        caught = held & fraud & verified
        claimed = h * rate
        true_amt = ref_h * rate
        paid = np.where(dec == REJECT, 0.0, np.where(caught, true_amt, claimed))
        _ = self.rng.random(n)                                     # audit draw (rate 0), keeps RNG aligned with simulate
        if held.any():
            fb = self.pending.setdefault(self.d + self.v_delay, (np.zeros(n), np.zeros(n)))
            fb[0][held & verified & fraud] = 1.0
            fb[1][held & verified & ~fraud] = 1.0
        vf, vl = self.pending.pop(self.d, (np.zeros(n), np.zeros(n)))
        hexp = np.maximum(baseline, 1.0)
        exp_w = np.where(is_claim, h / hexp, 0.0)
        self.trust.update(s, hard.astype(float), exp_w, vf, vl, np.ones(n, bool))
        self.d += 1
        return dict(T=T, R=R, dec=dec, paid=paid, claimed=claimed, true_amt=true_amt, fraud=fraud & is_claim,
                    is_claim=is_claim)
