"""Learning adversary: tabular Q-learning over the settlement process (white-box observations).

The policy is shared by all compromised devices; each device is an independent learner that writes to the
same table, so one episode yields |C| x (claims) transitions. A transition runs from one claim decision to
the device's next claim decision; the reward is the realized profit in between (paid - true amount -
clawbacks - fines, in units of one typical day's pay), so delayed consequences (verification delay,
aggregate-test clawbacks, quarantine) are learned through the discounted value function.

State (64 cells), observed before the report is committed:
  trust bucket            T < T_min (quarantined) | < 0.75 | < 0.90 | >= 0.90
  periods since the last  hold / rejection / recovery:  <= 3 | <= 10 | <= 30 | > 30
  risk of the honest report relative to the approve threshold: R/theta_a < 0.5 | < 0.8 | < 1 | >= 1
Actions: over-claim fraction in GRID (includes sub-tolerance fractions, so salami strategies are learnable).
"""
from __future__ import annotations

import numpy as np

from .adaptive import Attacker
from .authorisation import APPROVE

GRID = np.array([0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60])
N_STATES = 4 * 4 * 4


def new_table():
    return np.zeros((N_STATES, len(GRID)))


class QLearningAttacker(Attacker):
    name = "RL (Q-learning)"

    def __init__(self, comp, Q, eps=0.1, lr=0.1, gamma=0.95, train=True, seed=0):
        super().__init__(comp)
        self.Q, self.eps, self.lr, self.gamma, self.train = Q, eps, lr, gamma, train
        self.rng = np.random.default_rng(seed)
        n = len(comp)
        self.prev_s = np.full(n, -1)
        self.prev_a = np.zeros(n, int)
        self.acc = np.zeros(n)
        self.last_event = np.full(n, -10_000)
        self.scale = None

    def _state(self, day, T, R_rel, low):
        tb = np.where(T < low, 0, np.where(T < 0.75, 1, np.where(T < 0.90, 2, 3)))
        since = day - self.last_event
        sb = np.digitize(since, [4, 11, 31])
        rb = np.digitize(R_rel, [0.5, 0.8, 1.0])
        return (tb * 4 + sb) * 4 + rb

    def alphas(self, day, rep, ctx):
        comp = self.comp
        D = ctx.defender
        if self.scale is None:
            self.scale = np.maximum(ctx.rate[comp] * 8.0, 1e-9)
        T = D.trust.value()[comp]
        feats, _ = ctx.engine.step(rep)
        s = ctx.detector.score(feats, comp)
        R_rel = (1 - T * (1 - s)) / max(D.theta[0], 1e-9)
        state = self._state(day, T, R_rel, ctx.cfg["trust"]["low"])
        claim = rep["reported_hours"][comp] > 0.2
        a = np.zeros(len(comp), int)
        for j in np.flatnonzero(claim):
            st = state[j]
            if self.train and self.prev_s[j] >= 0:
                ps, pa = self.prev_s[j], self.prev_a[j]
                target = self.acc[j] + self.gamma * self.Q[st].max()
                self.Q[ps, pa] += self.lr * (target - self.Q[ps, pa])
            if self.train and self.rng.random() < self.eps:
                a[j] = self.rng.integers(len(GRID))
            else:
                q = self.Q[st]
                a[j] = int(np.flatnonzero(q == q.max())[0])     # ties -> smallest over-claim
            self.prev_s[j], self.prev_a[j], self.acc[j] = st, a[j], 0.0
        return np.where(claim, GRID[a], 0.0)

    def feedback(self, day, out):
        comp = self.comp
        r = (out["paid"] - out["true_amt"] - out.get("recovered", 0.0))[comp]
        r = np.where(out["is_claim"][comp], r, 0.0)
        self.acc += r / self.scale
        event = (out["dec"][comp] != APPROVE) & out["is_claim"][comp]
        if "recovered" in out:
            event |= out["recovered"][comp] > 0
        self.last_event = np.where(event, day, self.last_event)

    def end_episode(self):
        """Terminal update (no bootstrap) for the last decision of every device."""
        if not self.train:
            return
        for j in np.flatnonzero(self.prev_s >= 0):
            ps, pa = self.prev_s[j], self.prev_a[j]
            self.Q[ps, pa] += self.lr * (self.acc[j] - self.Q[ps, pa])
