"""AVERT defender: randomised verification, clawback, quarantine and aggregate audits (docs/theory.md).

On top of the deterministic decision layer (approve / hold / reject from risk R = 1 - T(1 - s)):

  * held claims are verified before settlement (as in the deterministic baseline);
  * auto-approved claims are audited after settlement with probability
        q(R, rel) = clip(q_floor + kappa_q * R * rel, q_floor, q_max),   rel = claim hours / expected hours,
    which removes the "safe" auto-approval region a white-box attacker exploits (Proposition 1);
  * an over-claim established by a verification (above the per-claim tolerance tau, detected with
    probability p_d) is clawed back and fined lambda times its value (Theorem 1), and the device's trust is
    slashed so that it is quarantined (all claims held) until it rebuilds trust (Theorem 2);
  * every verified claim adds its measured discrepancy (reported - true hours) to a per-device buffer; a
    one-sided test on the buffer mean, z * max(sigma, s_m) / sqrt(m) (s_m: 1.4826 x the buffer's median
    absolute deviation; only discrepancies within the tolerance enter the buffer), detects systematic
    sub-tolerance ("salami")
    over-claiming (Proposition 3). On detection the estimated relative bias is clawed back (plus the fine)
    on everything paid to the device since the buffer was last reset, and trust is slashed.
"""
from __future__ import annotations

import numpy as np

from .authorisation import APPROVE, HOLD, REJECT
from .online import OnlineDefender


class AVERTDefender(OnlineDefender):
    def __init__(self, spec, cfg, theta, n, seed, params: dict, sigma_e: float):
        super().__init__(spec, cfg, theta, n, seed)
        self.p = dict(params)
        self.sigma_e = float(sigma_e)
        M = int(self.p.get("agg_window", 30))
        self.M = M
        self.disc = np.full((n, M), np.nan)        # measured discrepancies (hours) of verified claims
        self.hrs = np.full((n, M), np.nan)         # reported hours of those claims
        self.paid_since = np.zeros(n)              # paid amount since last aggregate reset
        self.flags = np.zeros(n, int)              # aggregate-test detections per device
        vw = cfg["trust"]["adaptive"]["verify_weight"]
        self.slash_units = self.p.get("slash", vw) / vw   # trust evidence units for an established over-claim
        self.recent = []                           # fleet-wide sub-tolerance discrepancies, one array per day
        self.sigma_fleet = self.sigma_e

    def audit_prob(self, R, rel):
        p = self.p
        return np.clip(p["q_floor"] + p["kappa_q"] * R * rel, p["q_floor"], p.get("q_max", 1.0))

    def step(self, h, rate, ref_h, s, hard, baseline):
        n, p = self.n, self.p
        T = self.trust.value()
        is_claim = h > 0
        dec, R = self.decide(T, s, hard, h)
        over_h = h - ref_h
        single = over_h > self.tol                                 # establishable by one verification
        claimed = h * rate
        true_amt = ref_h * rate
        hexp = np.maximum(baseline, 1.0)
        rel = np.where(is_claim, h / hexp, 0.0)

        detect = self.rng.random(n) < self.p_det
        held = (dec == HOLD) & is_claim
        approved = (dec == APPROVE) & is_claim
        q = np.where(approved, self.audit_prob(R, rel), 0.0)
        audited = self.rng.random(n) < q
        audited &= approved
        caught_hold = held & single & detect
        caught_audit = audited & single & detect
        paid = np.where(dec == REJECT, 0.0, np.where(caught_hold, true_amt, claimed))
        over_val = np.maximum(claimed - true_amt, 0.0)
        clawed = np.where(caught_audit, over_val, 0.0)              # over-payment returned
        fined = p["penalty"] * over_val * (caught_hold | caught_audit)

        # aggregate audit on verified claims (verification reveals the true hours)
        verified = held | audited
        flagged = np.zeros(n, bool)
        if p.get("aggregate", False):
            # only sub-tolerance discrepancies: larger ones are the single-claim mechanism's domain
            sub = verified & (np.abs(over_h) <= self.tol)
            # honest-noise scale re-estimated from the whole fleet's recent verifications (robust MAD;
            # honest devices are the majority), so a fleet-wide rise in metering noise is tracked
            self.recent = (self.recent + [over_h[sub]])[-int(p.get("sigma_days", 60)):]
            pool = np.concatenate(self.recent)
            if pool.size >= 30:
                self.sigma_fleet = 1.4826 * float(np.median(np.abs(pool - np.median(pool))))
            for i in np.flatnonzero(sub):
                self.disc[i] = np.roll(self.disc[i], -1)
                self.hrs[i] = np.roll(self.hrs[i], -1)
                self.disc[i, -1] = over_h[i]
                self.hrs[i, -1] = h[i]
                d = self.disc[i][~np.isnan(self.disc[i])]
                m = d.size
                # scale: the largest of the calibrated, fleet-recent and device robust spreads (the device
                # MAD is insensitive to a constant offset and to a minority of pre-attack entries)
                sd = max(self.sigma_e, self.sigma_fleet, 1.4826 * float(np.median(np.abs(d - np.median(d)))))
                if m >= p["agg_min"] and d.mean() > p["agg_z"] * sd / np.sqrt(m):
                    hh = self.hrs[i][~np.isnan(self.hrs[i])]
                    bias = max(d.mean() / max(hh.mean(), 1e-9), 0.0)
                    claw = bias * (self.paid_since[i] + paid[i])
                    clawed[i] += claw
                    fined[i] += claw * p["penalty"]
                    flagged[i] = True
                    self.flags[i] += 1
                    self.disc[i] = np.nan
                    self.hrs[i] = np.nan
                    self.paid_since[i] = -paid[i]
        self.paid_since += paid

        # trust feedback (delayed like verification outcomes); established over-claims slash trust
        fraud_ev = (caught_hold | caught_audit).astype(float) * self.slash_units + flagged * self.slash_units
        # positive evidence only from successful verifications that found no established over-claim
        # (as in the deterministic baseline; failed verifications give no evidence either way)
        clean = verified & detect & ~single & ~flagged
        if verified.any() or flagged.any():
            fb = self.pending.setdefault(self.d + self.v_delay, (np.zeros(n), np.zeros(n)))
            fb[0][:] += fraud_ev
            fb[1][clean] = 1.0
        vf, vl = self.pending.pop(self.d, (np.zeros(n), np.zeros(n)))
        exp_w = np.where(is_claim, rel, 0.0)
        self.trust.update(s, hard.astype(float), exp_w, vf, vl, np.ones(n, bool))
        self.d += 1
        return dict(T=T, R=R, dec=dec, paid=paid, claimed=claimed, true_amt=true_amt, fraud=single & is_claim,
                    is_claim=is_claim, clawed=clawed, fined=fined, recovered=clawed + fined, verified=verified,
                    flagged=flagged, q=q)
