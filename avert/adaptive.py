"""Adaptive adversaries against deterministic trust-aware authorisation (DTA), and the runner.

Attacker knowledge levels (docs/research_plan.md):
  K0  oblivious   fixed attack families of the simulator (the generated records)
  K1  feedback    observes its own settlement outcomes and adapts (AIMD controller)
  K3  white-box   knows features, detector, trust and thresholds; each day picks the largest over-claim
                  that is still auto-approved (cannot see the verification randomness)
  K3* risk-aware  white-box that also knows the audit policy, penalty and tolerance, and picks the
                  over-claim with the highest expected immediate profit (salami included)
plus strategy families chosen offline: constant over-claim fraction, salami offset just below the
single-claim tolerance, and on-off bursts. The worst case is the maximum over the whole library.

All adaptive attackers start from the compromised device's honest telemetry and fake correlated channels
consistently: reported hours, hour meter, fuel and idle time are scaled together; engine speed, temperature
and vibration (intensive quantities) are left honest. Over-claims are only placed on worked days.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .authorisation import APPROVE, HOLD, SimData, SystemSpec, calibrate, simulate
from .detectors import REGISTRY, FitContext, HybridStacker, Split
from .features import FEATURES, build_features, expected_usage, hard_violations, observable_view, to_tensor
from .generator import generate_world
from .online import REPORT_FIELDS, OnlineDefender, OnlineDetector, OnlineFeatureEngine
from .verification import AVERTDefender
from .windows import WindowStore, index_grid

MAX_REPORTED = 19.5


# ----------------------------------------------------------------------------- attackers
def apply_overclaim(rep: dict, idx: np.ndarray, alpha: np.ndarray) -> dict:
    """Consistently inflate the reports of assets idx by fraction alpha of their honest hours."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rep.items()}
    h = out["reported_hours"][idx]
    worked = h > 0.2
    new_h = np.where(worked, np.minimum(np.round(h * (1 + alpha), 1), MAX_REPORTED), h)
    scale = np.where(worked & (h > 0), new_h / np.maximum(h, 1e-9), 1.0)
    out["reported_hours"][idx] = new_h
    out["meter_delta"][idx] = out["meter_delta"][idx] + (new_h - h)
    out["fuel_l"][idx] = out["fuel_l"][idx] * scale
    out["idle_hours"][idx] = out["idle_hours"][idx] * scale
    return out


class Attacker:
    name = "base"

    def __init__(self, comp: np.ndarray):
        self.comp = comp

    def alphas(self, day, rep, ctx) -> np.ndarray:
        raise NotImplementedError

    def observe(self, day, dec):
        pass


class ConstantInflation(Attacker):
    def __init__(self, comp, alpha):
        super().__init__(comp)
        self.alpha = alpha
        self.name = f"const({alpha:.2f})"

    def alphas(self, day, rep, ctx):
        return np.full(len(self.comp), self.alpha)


class AIMDFeedback(Attacker):
    """K1: additive increase while claims are paid, multiplicative decrease after a hold or rejection."""

    def __init__(self, comp, a0=0.05, inc=0.02, dec=0.5, cap=1.0):
        super().__init__(comp)
        self.alpha = np.full(len(comp), a0)
        self.inc, self.decf, self.cap = inc, dec, cap
        self.name = "K1 feedback (AIMD)"

    def alphas(self, day, rep, ctx):
        return self.alpha.copy()

    def observe(self, day, dec):
        d = dec[self.comp]
        self.alpha = np.where(d == APPROVE, np.minimum(self.alpha + self.inc, self.cap), self.alpha * self.decf)


class WhiteBoxGreedy(Attacker):
    """K3: evaluates candidate over-claims through the real defence and takes the largest auto-approved one."""

    def __init__(self, comp, grid=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.32, 0.40,
                                    0.50, 0.65, 0.80, 1.0)):
        super().__init__(comp)
        self.grid = np.asarray(grid)
        self.name = "K3 white-box"

    def alphas(self, day, rep, ctx):
        T = ctx.defender.trust.value()[self.comp]
        best = np.zeros(len(self.comp))
        ta = ctx.defender.theta[0]
        low = ctx.cfg["trust"]["low"]
        for a in self.grid[1:]:
            cand = apply_overclaim(rep, self.comp, np.full(len(self.comp), a))
            feats, _ = ctx.engine.step(cand)
            s = ctx.detector.score(feats, self.comp)
            ok = (1 - T * (1 - s) < ta) & (T >= low)
            best = np.where(ok, a, best)
        return best


class OnOff(Attacker):
    """Bursts of over-claiming separated by honest periods (to rebuild trust)."""

    def __init__(self, comp, alpha=0.30, on=5, off=15):
        super().__init__(comp)
        self.alpha, self.on, self.off = alpha, on, off
        self.name = f"on-off({alpha:.2f}, {on}/{off})"

    def alphas(self, day, rep, ctx):
        k = (day - ctx.ve) % (self.on + self.off)
        return np.full(len(self.comp), self.alpha if k < self.on else 0.0)


class SalamiOffset(Attacker):
    """Adds a fixed number of hours just below the single-claim verification tolerance on every worked day."""

    def __init__(self, comp, hours=0.40):
        super().__init__(comp)
        self.hours = hours
        self.name = f"salami(+{hours:.2f} h)"

    def alphas(self, day, rep, ctx):
        h = rep["reported_hours"][self.comp]
        return np.where(h > 0.2, self.hours / np.maximum(h, 1e-9), 0.0)


class RiskAwareWhiteBox(Attacker):
    """K3*: knows the full defence including audit probabilities, penalty and tolerance; maximises the
    expected immediate profit of each claim (it does not anticipate aggregate audits or quarantine)."""

    def __init__(self, comp, grid=(0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25,
                                    0.32, 0.40, 0.50, 0.65, 0.80, 1.0)):
        super().__init__(comp)
        self.grid = np.asarray(grid)
        self.name = "K3* risk-aware white-box"

    def alphas(self, day, rep, ctx):
        D = ctx.defender
        comp = self.comp
        T = D.trust.value()[comp]
        ta, tr = D.theta
        low = ctx.cfg["trust"]["low"]
        lam = D.p["penalty"] if hasattr(D, "p") else 0.0
        h = rep["reported_hours"][comp]
        rate = ctx.rate[comp]
        best_a, best_g = np.zeros(len(comp)), np.full(len(comp), -np.inf)
        for a in self.grid:
            cand = apply_overclaim(rep, comp, np.full(len(comp), a))
            feats, pend = ctx.engine.step(cand)
            s = ctx.detector.score(feats, comp)
            R = 1 - T * (1 - s)
            new_h = cand["reported_hours"][comp]
            v = (new_h - h) * rate
            single = (new_h - h) > D.tol - 0.05             # margin for honest metering noise
            app = (R < ta) & (T >= low)
            rej = R >= tr
            rel = new_h / np.maximum(pend.baseline[comp], 1.0)
            q = D.audit_prob(R, rel) if hasattr(D, "audit_prob") else 0.0
            g = np.where(app, v - single * q * D.p_det * (1 + lam) * v,
                         np.where(rej, -h * rate, v - single * D.p_det * (1 + lam) * v))
            better = g > best_g + 1e-9
            best_a = np.where(better, a, best_a)
            best_g = np.where(better, g, best_g)
        return best_a


# ----------------------------------------------------------------------------- defender training
class World:
    """One simulated fleet with a trained DTA defender (TCN+XGBoost detector, adaptive trust, calibrated thresholds)."""

    def __init__(self, cfg, g, seed, log=print):
        t_start = time.time()
        self.cfg, self.g, self.seed = cfg, g, seed
        sp, W = cfg["splits"], cfg["windows"]["length"]
        A, D = g["n_assets"], g["n_days"]
        te, vs, ve = sp["train_end"], sp["val_split"], sp["val_end"]
        self.A, self.D, self.te, self.vs, self.ve, self.W = A, D, te, vs, ve, W
        rec, assets = generate_world(g, seed)
        self.records = rec.sort_values(["asset_id", "day"]).reset_index(drop=True)
        self.assets = assets.sort_values("asset_id").reset_index(drop=True)
        obs = observable_view(self.records)
        self.obs = obs
        feat = build_features(obs, self.assets, g["commissioning_days"])
        self.X = to_tensor(feat, A, D)
        self.hard = hard_violations(obs, g["contract"]).to_numpy().reshape(A, D)
        grid = lambda c: self.records[c].to_numpy().reshape(A, D)
        self.grid = grid
        y = grid("anomaly_label")
        self.comp = np.flatnonzero(self.assets.compromised.to_numpy())

        mask = np.zeros((A, D), bool)
        mask[:, :te] = True
        flat = self.X[mask]
        self.mean, self.std = flat.mean(0), flat.std(0)
        self.std[self.std < 1e-6] = 1.0
        store = WindowStore(self.X, W, self.mean, self.std)
        self.store = store

        def split(lo, hi, labels=True):
            a, t = index_grid(np.arange(A), lo, hi)
            return Split(a, t, y[a, t] if labels else None)

        tr, vc = split(0, te), split(te, vs)
        normal = split(10, g["commissioning_days"], labels=False)
        ctx = FitContext(store, tr, vc, normal, cfg["detectors"], seed)
        self.tcn = REGISTRY["tcn"]().fit(ctx)
        self.xgb = REGISTRY["xgboost"]().fit(ctx)
        a_all, t_all = index_grid(np.arange(A), te, D)
        raw_t = self.tcn.score(store, a_all, t_all).reshape(A, D - te)
        raw_x = self.xgb.score(store, a_all, t_all).reshape(A, D - te)
        self.stacker = HybridStacker(kinds=("logit", "logit")).fit(raw_t[vc.a, vc.t - te], raw_x[vc.a, vc.t - te], vc.y)
        self.prob = self.stacker(raw_t.ravel(), raw_x.ravel()).reshape(A, D - te)

        # DTA design and calibration of its thresholds on validation days
        d = cfg["baseline_design"]
        self.spec = SystemSpec("DTA", detector=cfg["detectors"]["primary"], risk=d["risk"], use_F=d["use_F"],
                               audit=d["audit"], exposure=d["exposure"])
        hrs = self.records.reported_hours.where(self.records.reported_hours > 0)
        peer_med = hrs.groupby([self.records.site, self.records.day]).transform("median").fillna(0.0).to_numpy().reshape(A, D)
        base = expected_usage(obs, g["commissioning_days"]).baseline.to_numpy().reshape(A, D)

        def simdata(lo, hi):
            sl = (slice(None), slice(lo, hi))
            return SimData(hours=grid("reported_hours")[sl], claimed=grid("claimed_amount")[sl],
                           rate=self.assets.rate.to_numpy(), nominal=self.assets.nominal_hours.to_numpy(),
                           peer_med=peer_med[sl], hard=self.hard[sl], scores={"tcn_xgb": self.prob[:, lo - te:hi - te]},
                           ref_hours=grid("reference_hours")[sl], overclaim=grid("overclaim_amount")[sl],
                           fraud=grid("fraud_label")[sl], compromised=self.assets.compromised.to_numpy(),
                           family=grid("attack_family")[sl], baseline=base[sl])

        self.theta, _, self.feasible = calibrate(self.spec, simdata(te, ve), cfg, vs - te, seed=seed)
        # conformal thresholds (Proposition 4): quantiles of the risk of legitimate validation claims
        self.val_data = simdata(te, ve)
        trace = simulate(self.spec, self.val_data, cfg, self.theta, seed=seed, eval_from=vs - te, trace=True)["_trace"]
        Rv = trace["R"][:, vs - te:]
        legit = (self.val_data.hours[:, vs - te:] > 0) & (self.val_data.fraud[:, vs - te:] == 0)
        r = np.sort(Rv[legit])
        nr = len(r)

        def cq(alpha):
            k = min(nr - 1, int(np.ceil((1 - alpha) * (nr + 1))) - 1)
            return float(r[k])

        av = cfg["avert"]
        self.theta_conformal = (cq(av["alpha_hold"]), max(cq(av["alpha_reject"]), cq(av["alpha_hold"]) + 1e-6))
        # honest metering-discrepancy scale from commissioning-period calibration audits (attack-free period)
        rh, ref = grid("reported_hours"), grid("reference_hours")
        cm = rh[:, :g["commissioning_days"]] > 0.2
        dd = (rh - ref)[:, :g["commissioning_days"]][cm]
        self.sigma_e = float(1.4826 * np.median(np.abs(dd - np.median(dd))))
        self.batch_test = simulate(self.spec, simdata(te, D), cfg, self.theta, seed=seed + 1, eval_from=ve - te)
        # report-field arrays for fast day access
        self.fields = {f: grid(f).astype(float) for f in REPORT_FIELDS}
        self.honest = {f: grid("h_" + f).astype(float) for f in REPORT_FIELDS if f != "delay_h"}
        self.honest["delay_h"] = (grid("ts_recv_h") - grid("h_ts_sent_h")).astype(float)
        self.ref_h = grid("reference_hours").astype(float)
        self.rate = self.assets.rate.to_numpy(float)
        self._checkpoint = None
        log(f"  world seed={seed}: defender trained in {time.time() - t_start:.0f}s, theta={self.theta}")

    def rep(self, day, honest_for=None):
        r = {f: self.fields[f][:, day].copy() for f in REPORT_FIELDS}
        r["day"] = day
        if honest_for is not None:
            for f in REPORT_FIELDS:
                r[f][honest_for] = self.honest[f][honest_for, day]
        return r

    # --------------------------------------------------------------------- online run
    def make_defender(self, name: str, params: dict | None = None):
        if params is None:
            return OnlineDefender(self.spec, self.cfg, self.theta, self.A, self.seed + 1)
        theta = self.theta_conformal if params.get("conformal") else self.theta
        return AVERTDefender(self.spec, self.cfg, theta, self.A, self.seed + 1, params, self.sigma_e)

    def warm_up(self, verify=True):
        """Run the streaming feature engine and detector over validation days on the original records.
        Checkpoint them at test start and keep the daily inputs so each defender can be warmed up cheaply."""
        self.engine = OnlineFeatureEngine(self.obs, self.assets, self.g["commissioning_days"], self.te)
        self.detector = OnlineDetector(self.tcn, self.xgb, self.stacker, self.mean, self.std, self.W,
                                       self.engine.batch_hist, store=self.store, t0=self.te)
        max_feat, max_s = 0.0, 0.0
        self._val_inputs = []
        for day in range(self.te, self.ve):
            rep = self.rep(day)
            feats, pend = self.engine.step(rep)
            s = self.detector.score(feats)
            if verify:
                max_feat = max(max_feat, float(np.abs(feats - self.X[:, day]).max()))
                max_s = max(max_s, float(np.abs(s - self.prob[:, day - self.te]).max()))
            self._val_inputs.append((rep["reported_hours"], self.ref_h[:, day], s, self.hard[:, day], pend.baseline))
            self.engine.commit(pend)
            self.detector.commit(feats)
        self._pipe = copy.deepcopy((self.engine, self.detector))
        self._defenders = {}
        self.add_defender("DTA", None)
        return max_feat, max_s

    def add_defender(self, name: str, params: dict | None):
        """Warm up a defender over the validation days (same inputs for every defender)."""
        d = self.make_defender(name, params)
        for h, ref, s, hard, base in self._val_inputs:
            d.step(h, self.rate, ref, s, hard, base)
        self._defenders[name] = d

    def run_test(self, attacker: Attacker | None, defender: str = "DTA"):
        """Test period. attacker=None replays the original (K0) records."""
        self.engine, self.detector = copy.deepcopy(self._pipe)
        self.defender = copy.deepcopy(self._defenders[defender])
        comp = self.comp
        keys = ("dec", "paid", "claimed", "true_amt", "fraud", "is_claim", "T")
        extra = ("recovered", "clawed", "fined", "verified", "flagged", "q")
        log = {k: [] for k in (*keys, *extra, "alpha", "s")}
        for day in range(self.ve, self.D):
            if attacker is None:
                rep = self.rep(day)
                hard = self.hard[:, day]
                alpha = np.zeros(len(comp))
            else:
                base_rep = self.rep(day, honest_for=comp)
                alpha = attacker.alphas(day, base_rep, self)
                rep = apply_overclaim(base_rep, comp, alpha)
                hard = self.hard[:, day].copy()
                hard[comp] = rep["reported_hours"][comp] > self.g["contract"]["max_daily_hours"]
            feats, pend = self.engine.step(rep)
            s = self.detector.score(feats)
            out = self.defender.step(rep["reported_hours"], self.rate, self.ref_h[:, day], s, hard, pend.baseline)
            self.engine.commit(pend)
            self.detector.commit(feats)
            if attacker is not None:
                attacker.observe(day, out["dec"])
            for k in keys:
                log[k].append(out[k])
            for k in extra:
                log[k].append(out.get(k, np.zeros(self.A, dtype=bool if k in ("verified", "flagged") else float)))
            log["s"].append(s)
            log["alpha"].append(alpha)
        log = {k: np.stack(v, 1) for k, v in log.items()}
        held = (log["dec"] == HOLD) & log["is_claim"]
        log["verified"] = log["verified"] | held                     # DTA verifies held claims too
        return log


def summarise(w: World, log: dict, review_cost: float = 25.0) -> dict:
    comp = w.comp
    hon = np.setdiff1d(np.arange(w.A), comp)
    cl = log["is_claim"]
    gain = (log["paid"] - log["true_amt"]) * cl
    over = (log["claimed"] - log["true_amt"]) * cl
    rec = log["recovered"]
    attempted = np.clip(over[comp], 0, None).sum()
    extracted = np.clip(gain[comp], 0, None).sum()
    net = gain[comp].sum() - rec[comp].sum()
    legit_h = cl[hon] & ~log["fraud"][hon]
    dec_h = log["dec"][hon]
    fr = log["fraud"][comp]
    n_ver = int((log["verified"] & cl).sum())
    return dict(
        attempted_overclaim=float(attempted),
        extracted_value=float(extracted),                  # paid above the true amount, before recovery
        recovered_value=float(rec[comp].sum()),            # clawbacks + fines collected
        clawed_value=float(log["clawed"][comp].sum()),
        fine_income=float(log["fined"][comp].sum()),
        payer_net_loss=float(extracted - log["clawed"][comp].sum()),  # over-payment not clawed back (fine income excluded)
        attacker_profit=float(net),                        # vs. honest pay: over-payment - lost legit pay - clawbacks - fines
        extraction_rate=float(extracted / attempted) if attempted > 0 else 0.0,
        share_overclaim_below_tolerance=float(np.clip(over[comp], 0, None)[~fr].sum() / attempted) if attempted > 0 else 0.0,
        comp_auto_approved=float((log["dec"][comp][cl[comp]] == APPROVE).mean()),
        comp_mean_trust_end=float(log["T"][comp, -1].mean()),
        verification_rate=float(n_ver / max(1, cl.sum())),
        honest_verification_rate=float((log["verified"] & cl)[hon].sum() / max(1, cl[hon].sum())),
        verification_cost=float(review_cost * n_ver),
        honest_legit_hold_rate=float((dec_h[legit_h] == HOLD).mean()),
        honest_devices_flagged=int(log["flagged"][hon].any(1).sum()),
        honest_value_clawed=float(rec[hon].sum()),
        comp_devices_flagged=int(log["flagged"][comp].any(1).sum()),
        mean_alpha=float(log["alpha"][:, cl[comp].any(0)].mean()) if log["alpha"].size else 0.0,
    )
