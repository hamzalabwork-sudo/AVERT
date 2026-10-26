"""Financial-risk-aware authorisation and settlement simulation.

Risk (noisy-OR):     R = 1 - T * (1 - s) * (1 - F)
  T  device trust before this report      (distrust 1 - T raises risk)
  s  calibrated anomaly score of the report
  F  financial exposure = f_max * clip((h - h_exp) / (tau * h_exp), 0, 1): excess of the claimed
     hours over expected usage, h_exp = max(rolling median of the asset's last paid claims,
     today's median reported hours of working peers at the same site).
     F is capped at f_max < 1, and calibrated reject thresholds satisfy
     theta_r > 1 - T_trusted * (1 - f_max), so a trusted device whose report carries no anomaly
     evidence can be escalated to verification by exposure, but never rejected by it alone.
Decision:  hard violation -> Reject;  R < theta_a -> Approve;  R < theta_r -> Hold;  else Reject.
Gate:      T < trust.low -> at least Hold (no automatic approval for low-trust devices).

Hold = independent verification (e.g. operator log / site inspection). With probability
p_detect it reveals the true usage after `delay_days`; a detected over-claim is settled at the
verified amount. This verification outcome is the only route by which reference hours reach
the system, and it is modelled explicitly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .trust import make_trust

APPROVE, HOLD, REJECT = 0, 1, 2


@dataclass
class SystemSpec:
    name: str
    kind: str = "risk"            # risk | approve_all | hard_only | amount | oracle
    detector: str | None = "tcn"  # None -> s = 0
    trust: str = "adaptive"
    risk: str = "noisy_or"        # noisy_or | expected_loss | weighted | detector | trust
    use_F: bool = True
    gate: bool = True
    hard: bool = True
    thresholds: str = "calibrated"  # calibrated | provisional
    trust_opts: dict = field(default_factory=dict)
    group: str = "main"
    exposure: str = "baseline"    # baseline (commissioning-anchored, v2) | rolling (v1: median of paid claims)
    audit: bool = False           # random post-settlement audits of approved claims feed trust (opt-in)


@dataclass
class SimData:
    """Arrays shaped [n_assets_subset, n_days_range]."""
    hours: np.ndarray
    claimed: np.ndarray
    rate: np.ndarray            # [n]
    nominal: np.ndarray         # [n] contract nominal daily hours
    peer_med: np.ndarray        # same-site median reported hours of working assets that day
    hard: np.ndarray
    scores: dict                # detector name -> [n, d] calibrated anomaly probability
    # evaluator-only (used for outcome accounting and modelled verification only)
    ref_hours: np.ndarray
    overclaim: np.ndarray
    fraud: np.ndarray
    compromised: np.ndarray     # [n]
    family: np.ndarray | None = None  # attack family per record (reporting only)
    baseline: np.ndarray | None = None  # commissioning-anchored, peer-adjusted expected hours


def risk_score(kind, T, s, F, weights, x=None, el_floor=0.5):
    if kind == "noisy_or":
        return 1 - T * (1 - s) * (1 - F)
    if kind == "expected_loss":
        # q = probability-like untrustworthiness of the claim; x = normalised excess over expected
        # usage. Exposure scales the priority of suspicious claims but never exceeds q.
        q = 1 - T * (1 - s)
        return q * (el_floor + (1 - el_floor) * x)
    if kind == "weighted":
        w = weights
        return w[0] * (1 - T) + w[1] * s + w[2] * F
    if kind == "detector":
        return s
    if kind == "trust":
        return 1 - T
    raise ValueError(kind)


def simulate(spec: SystemSpec, data: SimData, cfg: dict, theta=(0.3, 0.6), amount_cap=None,
             seed=0, eval_from=0, trace=False):
    au, tc = cfg["authorisation"], cfg["trust"]
    n, D = data.hours.shape
    rng = np.random.default_rng(seed)
    trust = make_trust(spec.trust, n, tc, spec.trust_opts)
    H = au["history_days"]
    hist = np.full((n, H), np.nan)
    hptr = np.zeros(n, int)
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    p_det, v_delay = au["verification"]["p_detect"], au["verification"]["delay_days"]
    s_all = data.scores[spec.detector] if spec.detector else np.zeros((n, D))
    ta, tr = theta
    rows = np.arange(n)

    dec_all = np.zeros((n, D), np.int8)
    loss_all = np.zeros((n, D))
    audit_all = np.zeros((n, D), bool)
    # audits only make sense where trust consumes verification outcomes (adaptive trust with feedback)
    uses_fb = spec.trust == "adaptive" and spec.trust_opts.get("feedback", True)
    audit_rate = au.get("audit_rate", 0.0) if (spec.audit and spec.kind == "risk" and uses_fb) else 0.0
    T_all = np.zeros((n, D)) if trace else None
    R_all = np.zeros((n, D)) if trace else None

    for d in range(D):
        h, claim = data.hours[:, d], data.claimed[:, d]
        is_claim = h > 0
        T = trust.value()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(hist, axis=1)
        if spec.exposure == "baseline" and data.baseline is not None:
            hexp = np.maximum(data.baseline[:, d], 1.0)
        else:
            hexp = np.maximum(np.maximum(np.where(np.isnan(med), data.nominal, med), data.peer_med[:, d]), 1.0)
        F = au["f_max"] * np.clip((h - hexp) / (au["tau"] * hexp), 0, 1) if spec.use_F else np.zeros(n)
        s = s_all[:, d]
        hard = data.hard[:, d].astype(bool)
        fraud = data.fraud[:, d].astype(bool)

        if spec.kind == "approve_all":
            dec = np.zeros(n, int)
        elif spec.kind == "hard_only":
            dec = np.where(hard, REJECT, APPROVE)
        elif spec.kind == "amount":
            dec = np.where(hard, REJECT, np.where(claim > amount_cap, HOLD, APPROVE))
        elif spec.kind == "oracle":
            dec = np.where(fraud, REJECT, APPROVE)
        else:
            x = np.clip((h - hexp) / (au["tau"] * hexp), 0, 1) if spec.use_F else np.zeros(n)
            R = risk_score(spec.risk, T, s, F, au["combiner_weights"], x, au.get("el_floor", 0.5))
            dec = np.where(R < ta, APPROVE, np.where(R < tr, HOLD, REJECT))
            if spec.gate:
                dec = np.where(T < tc["low"], np.maximum(dec, HOLD), dec)
            if spec.hard:
                dec = np.where(hard, REJECT, dec)
            if trace:
                R_all[:, d] = R
        dec = np.where(is_claim, dec, APPROVE)   # zero-usage reports carry no payment

        verified = rng.random(n) < p_det
        held = dec == HOLD
        caught = held & fraud & verified
        loss = np.where(fraud & ((dec == APPROVE) | (held & ~verified)), data.overclaim[:, d], 0.0)

        paid_h = np.where(dec == REJECT, np.nan, np.where(caught, data.ref_hours[:, d], h))
        upd = is_claim & ~np.isnan(paid_h) & (paid_h > 0)
        hist[rows[upd], hptr[upd] % H] = paid_h[upd]
        hptr[upd] += 1

        # random audit of approved claims (after settlement: no recovery, evidence for trust only)
        audited = (dec == APPROVE) & is_claim & (rng.random(n) < audit_rate)
        checked = held | audited
        if checked.any():
            fb = pending.setdefault(d + v_delay, (np.zeros(n), np.zeros(n)))
            fb[0][checked & verified & fraud] = 1.0
            fb[1][checked & verified & ~fraud] = 1.0
        vf, vl = pending.pop(d, (np.zeros(n), np.zeros(n)))

        exp_w = np.where(is_claim, h / hexp, 0.0)
        trust.update(s, hard.astype(float), exp_w, vf, vl, np.ones(n, bool))

        dec_all[:, d] = dec
        loss_all[:, d] = loss
        audit_all[:, d] = audited
        if trace:
            T_all[:, d] = T

    m = outcome_metrics(dec_all[:, eval_from:], loss_all[:, eval_from:], data, eval_from, cfg,
                        audits=audit_all[:, eval_from:])
    if trace:
        m["_trace"] = dict(T=T_all, R=R_all, dec=dec_all)
    return m


def outcome_metrics(dec, loss, data: SimData, eval_from, cfg, audits=None):
    sl = slice(eval_from, None)
    claim = data.hours[:, sl] > 0
    fraud = data.fraud[:, sl].astype(bool) & claim
    legit = claim & ~fraud
    over = data.overclaim[:, sl]
    claimed = data.claimed[:, sl]
    c = cfg["authorisation"]["costs"]
    n_claims, n_fraud, n_legit = claim.sum(), fraud.sum(), legit.sum()
    fraud_value = over[fraud].sum()
    fvs = loss[fraud].sum()
    holds = ((dec == HOLD) & claim).sum()
    l_hold = ((dec == HOLD) & legit).sum()
    l_rej = ((dec == REJECT) & legit).sum()
    l_rej_val = claimed[(dec == REJECT) & legit].sum()
    blocked = (fraud & (loss == 0)).sum()
    n_audits = int(audits.sum()) if audits is not None else 0
    per_family = {}
    if data.family is not None:
        fam = data.family[:, sl]
        for f in np.unique(fam[fraud]):
            m = fraud & (fam == f)
            per_family[f"fvs_pct[{f}]"] = float(loss[m].sum() / over[m].sum()) if over[m].sum() > 0 else 0.0
    return dict(
        **per_family,
        n_claims=int(n_claims), n_fraud=int(n_fraud), fraud_value=float(fraud_value),
        fvs=float(fvs), fvs_pct=float(fvs / fraud_value) if fraud_value > 0 else 0.0,
        loss_prevented_pct=float(1 - fvs / fraud_value) if fraud_value > 0 else 1.0,
        fraud_block_rate=float(blocked / n_fraud) if n_fraud else 1.0,
        legit_hold_rate=float(l_hold / n_legit) if n_legit else 0.0,
        legit_reject_rate=float(l_rej / n_legit) if n_legit else 0.0,
        legit_rejected_value=float(l_rej_val), holds_per_1000=float(1000 * holds / max(1, n_claims)),
        audits_per_1000=float(1000 * n_audits / max(1, n_claims)),
        total_cost=float(fvs + c["review"] * (holds + n_audits) + c["false_reject_factor"] * l_rej_val),
    )


def theta_grid(cfg, spec: SystemSpec | None = None):
    g = cfg["authorisation"]["grid"]
    r_min = 0.0
    if spec is not None and spec.risk == "noisy_or" and spec.use_F:
        r_min = 1 - cfg["trust"]["trusted"] * (1 - cfg["authorisation"]["f_max"])
    out = []
    for a in g["approve"]:
        for gap in g["reject_gap"]:
            r = round(a + gap, 4)
            if r_min < r <= 1.0:
                out.append((a, r))
    return out


def select(results, cfg):
    """Minimise fraudulent value settled s.t. legit hold/reject budgets; fallback: min total cost."""
    ob = cfg["authorisation"]["objective"]
    feas = [r for r in results if r[1]["legit_hold_rate"] <= ob["max_legit_hold"]
            and r[1]["legit_reject_rate"] <= ob["max_legit_reject"]]
    if feas:
        return min(feas, key=lambda r: (round(r[1]["fvs"], 2), r[1]["total_cost"])), True
    return min(results, key=lambda r: r[1]["total_cost"]), False


def calibrate(spec: SystemSpec, val: SimData, cfg, eval_from, seed=0):
    """Returns (theta, amount_cap, feasible) chosen on the validation window only."""
    if spec.kind == "amount":
        claims = val.claimed[:, eval_from:][val.hours[:, eval_from:] > 0]
        cap = float(np.quantile(claims, 1 - cfg["authorisation"]["objective"]["max_legit_hold"]))
        return (0.3, 0.6), cap, True
    if spec.kind != "risk":
        return (0.3, 0.6), None, True
    if spec.thresholds == "provisional":
        p = cfg["authorisation"]["provisional"]
        return (p["approve"], p["reject"]), None, True
    res = [(th, simulate(spec, val, cfg, th, seed=seed, eval_from=eval_from)) for th in theta_grid(cfg, spec)]
    (th, _), feasible = select(res, cfg)
    return th, None, feasible
