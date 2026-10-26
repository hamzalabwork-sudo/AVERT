"""Real-data case study: electricity theft by under-reporting on SGCC smart-meter data.

Data: daily consumption of 42,372 customers over 2014-01-01 .. 2016-10-31 released by State Grid
Corporation of China (Zheng et al., IEEE TII 2018). Only customers labelled normal (FLAG = 0) with complete
records are used as *true* consumption; theft is injected so that the stolen value is known exactly.

Mapping to the claim model of docs/theory.md. A customer reports x_t and is billed r x_t; the utility loses
v_t = r (y_t - x_t) when x_t under-reports the true consumption y_t. An inspection (meter audit) measures the
true consumption up to an honest reconstruction error e_t ~ N(0, sigma_m), and establishes theft when
y_t - x_t > tau. The defender code works in over-claim form; with the per-customer constant K_i the
"credit claim" h_t = K_i - x_t over-claims the inspected value K_i - (y_t + e_t) by exactly y_t + e_t - x_t,
so every quantity of the simulator study (payer loss, recoveries, tolerance, aggregate test) carries over.

Units: each customer's series is divided by its commissioning-period median, so one unit is a typical day;
r_i = median kWh/day x tariff converts units to currency.

Days: commissioning 0-179 (attack-free; baselines, honest inspection noise), detector training 180-599
(separate customer pool with injected oblivious theft), validation 600-779 (fleet, oblivious theft;
threshold calibration and conformal quantiles), test 780-1033 (fleet, attacker library).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .authorisation import APPROVE, HOLD, SystemSpec
from .online import OnlineDefender
from .verification import AVERTDefender

COMM, TRAIN_END, VAL_SPLIT, VAL_END = 180, 600, 690, 780
W = 28
TARIFF = 0.5          # currency per kWh
EPS = 1e-3


# ----------------------------------------------------------------------------- data
def load_sgcc(path: str | Path, max_missing=0.05, min_median=1.0):
    """Normal customers with at most max_missing missing days and a commissioning median above min_median kWh.
    Missing days are linearly interpolated; spikes above mean + 2 std are clipped (Zheng et al. 2018)."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    order = np.argsort(pd.to_datetime(cols, format="%Y/%m/%d").values)
    X = df[cols].to_numpy(float)[:, order]
    flag = df.FLAG.to_numpy()
    miss = np.isnan(X).mean(1)
    med = np.nanmedian(np.where(np.isnan(X[:, :COMM]), np.nan, X[:, :COMM]), 1)
    keep = (flag == 0) & (miss <= max_missing) & (np.nan_to_num(med) > min_median)
    Y = pd.DataFrame(X[keep].T).interpolate(limit_direction="both").to_numpy().T
    cap = Y.mean(1, keepdims=True) + 2 * Y.std(1, keepdims=True)
    return np.clip(np.minimum(Y, cap), 0.0, None)


# ----------------------------------------------------------------------------- features / detector
def peer_stats(Xr, t):
    """Fleet medians of the current value and of the trailing 7- and 28-day means (from reported data)."""
    return (np.median(Xr[:, t]), np.median(Xr[:, t - 7:t].mean(1)), np.median(Xr[:, t - W:t].mean(1)))


def features(Xr, t, idx, peer, x=None):
    """Window features of day t for customers idx; x overrides the day-t report (candidate scoring)."""
    hist = Xr[idx, t - W:t]
    x = Xr[idx, t] if x is None else x
    m7, m28, s28 = hist[:, -7:].mean(1), hist.mean(1), hist.std(1)
    p0, p7, p28 = peer
    return np.column_stack([
        x, x - m7, x / (m7 + EPS), x / (m28 + EPS), (x - m28) / (s28 + 0.05), m7, m28, s28,
        (hist < 0.05).mean(1), hist.min(1), x / (p0 + EPS), m7 / (p7 + EPS), m28 / (p28 + EPS),
        x - hist[:, -7], m7 / (m28 + EPS)]).astype(np.float32)


def oblivious_theft(rng, Y, days, share, ep_len=(20, 80), n_ep=(1, 3)):
    """Injects the six classic false-data theft functions (Jokar et al., IEEE TSG 2016) in random episodes
    on a share of customers. Returns the reported series (copy) and the per-day attack mask."""
    X = Y.copy()
    mask = np.zeros_like(Y, bool)
    lo, hi = days
    who = rng.choice(len(Y), int(round(share * len(Y))), replace=False)
    for i in who:
        for _ in range(rng.integers(n_ep[0], n_ep[1] + 1)):
            L = int(rng.integers(*ep_len))
            s = int(rng.integers(lo, max(lo + 1, hi - L)))
            e = min(hi, s + L)
            y = Y[i, s:e]
            kind = rng.integers(6)
            if kind == 0:
                x = rng.uniform(0.1, 0.8) * y
            elif kind == 1:
                x = y * (rng.random(e - s) > rng.uniform(0.3, 0.9))
            elif kind == 2:
                x = rng.uniform(0.1, 0.8, e - s) * y
            elif kind == 3:
                x = np.minimum(y, np.quantile(y, rng.uniform(0.2, 0.6)))
            elif kind == 4:
                x = rng.uniform(0.1, 0.8, e - s) * y.mean()
            else:
                x = np.full(e - s, y.mean() * rng.uniform(0.5, 0.9))
            X[i, s:e] = np.minimum(x, y)
            mask[i, s:e] |= (y - X[i, s:e]) > 0
    return X, mask


def train_detector(Ytr, rng, tol, seed):
    X, mask = oblivious_theft(rng, Ytr, (COMM, TRAIN_END), share=0.5)
    Fs, ys = [], []
    for t in range(COMM + W, TRAIN_END):
        peer = peer_stats(X, t)
        Fs.append(features(X, t, np.arange(len(X)), peer))
        ys.append(mask[:, t] & ((Ytr[:, t] - X[:, t]) > tol))
    F, y = np.concatenate(Fs), np.concatenate(ys)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    clf.fit(F, y)
    return clf


# ----------------------------------------------------------------------------- attackers (under-reporting)
class Attacker:
    name = "attacker"

    def __init__(self, comp):
        self.comp = comp

    def alphas(self, t, y, case):            # fraction of true consumption hidden by each compromised customer
        return np.zeros(len(self.comp))

    def observe(self, t, dec):
        pass


class Oblivious(Attacker):
    """K0: replays pre-generated classic theft episodes (not adapted to the defence)."""
    name = "K0 oblivious"

    def alphas(self, t, y, case):
        x = case.k0_report[self.comp, t]
        return np.where(y > EPS, 1 - x / np.maximum(y, EPS), 0.0)


class Constant(Attacker):
    def __init__(self, comp, a):
        super().__init__(comp)
        self.a = a

    def alphas(self, t, y, case):
        return np.full(len(self.comp), self.a)


class Salami(Attacker):
    """Hides a fixed amount just below the single-inspection tolerance every day."""

    def __init__(self, comp, units):
        super().__init__(comp)
        self.units = units

    def alphas(self, t, y, case):
        return np.minimum(1.0, self.units / np.maximum(y, EPS))


class OnOff(Attacker):
    def __init__(self, comp, a=0.3, on=5, off=15):
        super().__init__(comp)
        self.a, self.on, self.off = a, on, off

    def alphas(self, t, y, case):
        return np.full(len(self.comp), self.a if (t - VAL_END) % (self.on + self.off) < self.on else 0.0)


class AIMD(Attacker):
    """K1: increases the hidden share after every unchallenged bill, halves it after an inspection hold."""

    def __init__(self, comp, a0=0.05, inc=0.02, dec=0.5):
        super().__init__(comp)
        self.a = np.full(len(comp), a0)
        self.inc, self.dec = inc, dec

    def alphas(self, t, y, case):
        return self.a.copy()

    def observe(self, t, dec):
        held = dec[self.comp] != APPROVE
        self.a = np.clip(np.where(held, self.a * self.dec, self.a + self.inc), 0.0, 0.95)


GRID = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50, 0.65, 0.80, 0.95])


class WhiteBox(Attacker):
    """K3: knows the detector, trust state and thresholds; hides the largest share still auto-approved.
    K3* (risk_aware=True) also knows the audit policy, penalty and tolerance and maximises expected profit."""

    def __init__(self, comp, risk_aware=False):
        super().__init__(comp)
        self.risk_aware = risk_aware

    def alphas(self, t, y, case):
        D, comp = case.defender, self.comp
        T = D.trust.value()[comp]
        ta = D.theta[0]
        low = case.cfg["trust"]["low"]
        lam = D.p["penalty"] if hasattr(D, "p") else 0.0
        rate = case.rate[comp]
        best_a, best_g = np.zeros(len(comp)), np.full(len(comp), -np.inf)
        for a in GRID:
            x = (1 - a) * y
            s = case.score(t, comp, x)
            R = 1 - T * (1 - s)
            app = (R < ta) & (T >= low)
            v = (y - x) * rate
            if not self.risk_aware:
                g = np.where(app, v, -np.inf if a > 0 else 0.0)
            else:
                single = (y - x) > D.tol - 0.05
                rel = (case.K[comp] - x) / np.maximum(case.K[comp] - case.base[comp, t], 1.0)
                q = D.audit_prob(R, rel) if hasattr(D, "audit_prob") else 0.0
                g = np.where(app, v - single * q * D.p_det * (1 + lam) * v, v - single * D.p_det * (1 + lam) * v)
            better = g > best_g + 1e-12
            best_a = np.where(better, a, best_a)
            best_g = np.where(better, g, best_g)
        return best_a


def library(comp, tol):
    lib = [("K0 oblivious", Oblivious(comp))]
    lib += [(f"constant {a:.2f}", Constant(comp, a)) for a in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35)]
    lib += [(f"salami {0.75 * tol:.2f}", Salami(comp, 0.75 * tol)), ("on-off 0.30 5/15", OnOff(comp)),
            ("K1 feedback", AIMD(comp)), ("K3 white-box", WhiteBox(comp)), ("K3* risk-aware", WhiteBox(comp, True))]
    return lib


# ----------------------------------------------------------------------------- case
class SGCCCase:
    """One draw of a customer fleet with a trained detector and calibrated DTA thresholds."""

    def __init__(self, Y_all, cfg, seed, n_fleet=1000, n_train=3000, compromised=0.20, tol=0.20, sigma_m=0.05,
                 log=print):
        rng = np.random.default_rng(seed)
        self.seed = seed
        perm = rng.permutation(len(Y_all))
        med = np.median(Y_all[:, :COMM], 1, keepdims=True)
        U = Y_all / np.maximum(med, EPS)
        self.Y = U[perm[:n_fleet]]
        Ytr = U[perm[n_fleet:n_fleet + n_train]]
        self.A, self.D = self.Y.shape
        self.comp = np.sort(rng.choice(self.A, int(round(compromised * self.A)), replace=False))
        self.rate = med[perm[:n_fleet], 0] * TARIFF
        self.K = np.maximum(3.0, 1.5 * np.quantile(self.Y[:, :COMM], 0.99, axis=1))
        # inspection measurement of true consumption (honest reconstruction error)
        self.meas = np.clip(self.Y + rng.normal(0.0, sigma_m, self.Y.shape), 0.0, None)
        self.sigma_e = float(1.4826 * np.median(np.abs((self.meas - self.Y)[:, :COMM])))
        # commissioning-anchored expected consumption: fleet median scaled by the customer's own level
        pm = np.median(self.Y, 0)                       # fleet median by day (honest data before attacks)
        lvl = self.Y[:, :COMM].mean(1) / pm[:COMM].mean()
        self.base = lvl[:, None] * pm[None, :]
        self.cfg = copy.deepcopy(cfg)
        self.cfg["generator"]["contract"]["fraud_tolerance_hours"] = tol
        self.tol = tol
        self.clf = train_detector(Ytr, rng, tol, seed)
        # oblivious theft on compromised customers in validation and test (K0)
        k0, _ = oblivious_theft(rng, self.Y[self.comp], (VAL_SPLIT - W, self.D), share=1.0)
        self.k0_report = self.Y.copy()
        self.k0_report[self.comp] = k0
        self._val_s = {}                                # detector scores on the fixed validation reports
        self.spec = SystemSpec("DTA", detector="xgboost", risk="noisy_or", use_F=False, audit=False,
                               exposure="baseline")
        self.theta, self.theta_conformal = self.calibrate()
        log(f"  sgcc seed={seed}: theta={self.theta} conformal={self.theta_conformal} sigma_e={self.sigma_e:.3f}")

    # one-day plumbing ---------------------------------------------------------------
    def score(self, t, idx, x=None):
        f = features(self.Xr, t, idx, self.peer, x)
        return self.clf.predict_proba(f)[:, 1]

    def _day(self, t, x, D, cache=False):
        self.Xr[:, t] = x
        self.peer = peer_stats(self.Xr, t)
        if cache and t in self._val_s:
            s = self._val_s[t]
        else:
            s = self.score(t, np.arange(self.A))
            if cache:
                self._val_s[t] = s
        K = self.K
        h = K - np.clip(x, 0.0, K)
        ref = K - np.clip(self.meas[:, t], 0.0, K)
        return D.step(h, self.rate, ref, s, np.zeros(self.A, bool), K - self.base[:, t]), s

    def _run(self, D, lo, hi, attacker=None, report=None):
        keys = ("dec", "paid", "claimed", "true_amt", "fraud", "is_claim", "T", "R")
        extra = ("recovered", "clawed", "fined", "verified", "flagged", "q")
        log = {k: [] for k in (*keys, *extra, "alpha", "s")}
        self.defender = D
        for t in range(lo, hi):
            y = self.Y[:, t]
            x = (report if report is not None else self.Y)[:, t].copy()
            alpha = np.zeros(len(self.comp))
            if attacker is not None:
                self.Xr[:, t] = y                       # candidate scoring sees honest reports of others
                self.peer = peer_stats(self.Xr, t)
                alpha = np.clip(attacker.alphas(t, y[self.comp], self), 0.0, 1.0)
                x[self.comp] = (1 - alpha) * y[self.comp]
            out, s = self._day(t, x, D, cache=attacker is None and report is self.k0_report)
            if attacker is not None:
                attacker.observe(t, out["dec"])
            for k in keys:
                log[k].append(out[k])
            for k in extra:
                log[k].append(out.get(k, np.zeros(self.A, dtype=bool if k in ("verified", "flagged") else float)))
            log["s"].append(s)
            log["alpha"].append(alpha)
        log = {k: np.stack(v, 1) for k, v in log.items()}
        held = (log["dec"] == HOLD) & log["is_claim"]
        log["verified"] = log["verified"] | held
        return log

    # calibration --------------------------------------------------------------------
    def make_defender(self, params=None, theta=None):
        th = theta or self.theta
        if params is None:
            return OnlineDefender(self.spec, self.cfg, th, self.A, self.seed + 1)
        th = theta or (self.theta_conformal if params.get("conformal") else self.theta)
        return AVERTDefender(self.spec, self.cfg, th, self.A, self.seed + 1, params, self.sigma_e)

    def _warm(self, D):
        """Validation days on the K0 reports (same for every defender); leaves Xr ready for the test."""
        self.Xr = self.k0_report.copy()
        self.Xr[:, VAL_END:] = np.nan
        return self._run(D, COMM + W, VAL_END, report=self.k0_report)

    def calibrate(self):
        """Approve threshold minimising stolen value settled on val_policy days subject to <= 5 % holds of
        legitimate bills (never reject: a bill can always be inspected). Conformal: (1 - alpha_hold) quantile
        of the risk of legitimate bills."""
        av = self.cfg["avert"]
        max_hold = self.cfg["authorisation"]["objective"]["max_legit_hold"]
        best, rows = None, []
        for ta in self.cfg["authorisation"]["grid"]["approve"] + [0.7, 0.8, 0.9]:
            log = self._warm(OnlineDefender(self.spec, self.cfg, (ta, 1.01), self.A, self.seed + 1))
            sl = slice(VAL_SPLIT - COMM - W, None)
            legit = log["is_claim"][:, sl] & ~log["fraud"][:, sl]
            hold = (log["dec"][:, sl][legit] == HOLD).mean()
            loss = ((log["paid"] - log["true_amt"])[:, sl] * log["fraud"][:, sl]).sum()
            rows.append((ta, hold, loss))
            if hold <= max_hold and (best is None or loss < best[1]):
                best = (ta, loss)
        ta = best[0] if best else min(rows, key=lambda r: r[1])[0]
        log = self._warm(OnlineDefender(self.spec, self.cfg, (ta, 1.01), self.A, self.seed + 1))
        sl = slice(VAL_SPLIT - COMM - W, None)
        legit = log["is_claim"][:, sl] & ~log["fraud"][:, sl]
        r = np.sort(log["R"][:, sl][legit])
        k = min(len(r) - 1, int(np.ceil((1 - av["alpha_hold"]) * (len(r) + 1))) - 1)
        return (ta, 1.01), (float(r[k]), 1.01)

    # test ---------------------------------------------------------------------------
    def warm_up(self, defences):
        self._defenders = {}
        for name, params in defences:
            D = self.make_defender(params)
            self._warm(D)
            self._defenders[name] = (D, self.Xr.copy())

    def run_test(self, attacker, defender):
        D, Xr = self._defenders[defender]
        D = copy.deepcopy(D)
        self.Xr = Xr.copy()
        return self._run(D, VAL_END, self.D, attacker=copy.deepcopy(attacker))
