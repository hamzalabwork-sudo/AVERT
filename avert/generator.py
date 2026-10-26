"""AdvLease: synthetic usage-based equipment-leasing telemetry with financial ground truth.

Design (see docs/methodology.md, Section 2):

1. Honest telemetry is produced by a *usage model*, not from labels. Each asset has a true
   duty cycle driven by its site calendar, project phases, maintenance and a load factor.
   Correlated channels (fuel, RPM, engine temperature, vibration, idle time, hour meter) are
   derived from the true hours and load, so manipulation that is inconsistent across channels
   is physically detectable.
2. Benign anomalies (sensor noise, dropouts, stuck sensors, delayed/out-of-order messages,
   clock drift, legitimate overtime, maintenance stops) are what drive false positives.
3. Attacks are parameterised families applied to compromised assets after commissioning:
   inflation, replay, forged identity, gradual compromise and on-off. Each compromised asset
   has an attacker-knowledge level (naive / consistent / adaptive).
4. Ground truth (reference hours, labels, family) is emitted in evaluator-only columns that the
   detection pipeline never reads (enforced in avert.features).

Threat model: the claimant side (a compromised device or an insider controlling it) benefits
from over-claimed usage; the payer is protected. Under-reporting is out of scope.
"""
from __future__ import annotations

from itertools import cycle

import numpy as np
import pandas as pd

ATTACK_FAMILIES = ("inflation", "replay", "forged_identity", "gradual", "on_off")
KNOWLEDGE_LEVELS = ("naive", "consistent", "adaptive")
DROPOUT_CHANNELS = ("fuel_l", "avg_rpm", "engine_temp_c", "vibration_rms", "ambient_c", "idle_hours")

# Columns available only to the evaluator. The detector, trust and authorisation engines
# must never read them (except reference hours revealed through a modelled verification).
HONEST_COLUMNS = (
    "h_reported_hours", "h_meter_delta", "h_fuel_l", "h_avg_rpm", "h_engine_temp_c", "h_vibration_rms",
    "h_ambient_c", "h_idle_hours", "h_seq", "h_ts_sent_h", "h_signature_valid",
)
EVALUATOR_ONLY = (
    "reference_hours", "reference_amount", "overclaim_amount", "anomaly_label", "fraud_label",
    "attack_family", "knowledge", "benign_event", "compromised", *HONEST_COLUMNS,
)
REPORT_HOUR = 20.0  # devices report at 20:00 local time for the day


def _u(rng, lo_hi, size=None):
    return rng.uniform(lo_hi[0], lo_hi[1], size)


def _ri(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _ar1(rng, n, phi, sd):
    x = np.empty(n)
    x[0] = rng.normal(0, sd / np.sqrt(1 - phi**2))
    eps = rng.normal(0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _sites(g, rng):
    S, D = g["n_sites"], g["n_days"]
    days = np.arange(D)
    dow = days % 7  # day 0 is a Monday
    holidays = rng.choice(D, size=max(1, int(10 * D / 365)), replace=False)
    ambient = np.zeros((S, D))
    level = np.zeros((S, D))
    phase_levels = (1.0, 0.5, 0.0)          # active, reduced, shutdown
    leave_prob = (1 / 90, 1 / 30, 1 / 10)
    next_state = ((1, 2), (0, 2), (0, 1))
    next_p = ((0.7, 0.3), (0.7, 0.3), (0.6, 0.4))
    for s in range(S):
        ambient[s] = (rng.uniform(5, 20) + rng.uniform(8, 14) * np.sin(2 * np.pi * (days - 105) / 365)
                      + _ar1(rng, D, 0.7, 2.5))
        p_work = np.where(dow < 5, 0.97, np.where(dow == 5, 0.5, 0.05))
        work = rng.random(D) < p_work
        work[holidays] = False
        state, ph = 0, np.empty(D)
        for t in range(D):
            ph[t] = phase_levels[state]
            if rng.random() < leave_prob[state]:
                state = int(rng.choice(next_state[state], p=next_p[state]))
        level[s] = ph * work
    return ambient, level


def _episodes(rng, D, start, ep_days, gap_days, alpha_fn):
    active, alpha = np.zeros(D, bool), np.zeros(D)
    t = start + _ri(rng, gap_days) // 2
    while t < D:
        dur = _ri(rng, ep_days)
        active[t:t + dur] = True
        alpha[t:t + dur] = alpha_fn()
        t += dur + _ri(rng, gap_days)
    return active, alpha


def _schedule(fam, know, start, D, ac, rng):
    """Days on which the attack is active and its intensity alpha (fractional over-claim)."""
    adaptive = know == "adaptive"
    if fam == "gradual":
        p = ac["gradual"]
        rate = _u(rng, ac["adaptive"]["gradual_rate"]) if adaptive else _u(rng, p["rate_per_day"])
        cap = ac["adaptive"]["gradual_cap"] if adaptive else p["cap"]
        onset = int(rng.integers(start, max(start + 1, D - 90)))
        k = np.arange(D) - onset + 1
        return k > 0, np.clip(rate * k, 0, cap)
    if fam == "on_off":
        p = ac["on_off"]
        a_rng = ac["adaptive"]["alpha"] if adaptive else p["alpha"]
        return _episodes(rng, D, start, p["on_days"], p["off_days"], lambda: _u(rng, a_rng))
    p = ac[fam]
    a_rng = ac["adaptive"]["alpha"] if adaptive else p.get("alpha", (0.0, 0.0))
    return _episodes(rng, D, start, p["episode_days"], p["gap_days"], lambda: _u(rng, a_rng))


def generate_world(g: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one simulated world. Returns (records, assets)."""
    rng = np.random.default_rng(seed)
    A, S, D = g["n_assets"], g["n_sites"], g["n_days"]
    duty, ben, con, ac = g["duty"], g["benign"], g["contract"], g["attacks"]
    shift = ac.get("shift") or {}
    types = list(g["asset_types"])

    site_amb, site_level = _sites(g, rng)

    # ---------------- assets ----------------
    rows = []
    for i in range(A):
        typ = types[int(rng.integers(len(types)))]
        spec = g["asset_types"][typ]
        mu = _u(rng, duty["hours_mean"])
        rows.append(dict(
            asset_id=i, device_id=f"EQ-{i:04d}", site=i % S, asset_type=typ, nominal_hours=round(mu, 1),
            mu_hours=mu, sd_hours=_u(rng, duty["hours_sd"]), load_mean=_u(rng, duty["load_mean"]),
            drift_min_per_day=rng.normal(0, ben["clock_drift_sd_min"]), **spec,
        ))
    assets = pd.DataFrame(rows)

    n_comp = int(round(g["compromise_fraction"] * A))
    comp_ids = rng.choice(A, size=n_comp, replace=False)
    fams = list(ac["families"])
    rng.shuffle(fams)
    fam_cycle = cycle(fams)
    kn_p = np.array([ac["knowledge_mix"][k] for k in KNOWLEDGE_LEVELS], float)
    assets["compromised"] = False
    assets["family"] = "none"
    assets["knowledge"] = "none"
    for i in comp_ids:
        assets.loc[i, ["compromised", "family", "knowledge"]] = (
            True, next(fam_cycle), KNOWLEDGE_LEVELS[int(rng.choice(3, p=kn_p / kn_p.sum()))])

    days = np.arange(D)
    frames = []
    for i in range(A):
        a = assets.iloc[i]
        s = int(a.site)
        amb = site_amb[s]
        lvl = site_level[s]

        # ---------------- honest usage ----------------
        maint = np.zeros(D, bool)
        t = 0
        while t < D:
            if rng.random() < duty["maintenance_rate"]:
                dur = _ri(rng, duty["maintenance_days"])
                maint[t:t + dur] = True
                t += dur
            else:
                t += 1
        worked = (lvl > 0) & ~maint & (rng.random(D) >= duty["absence_prob"])
        true_h = np.where(worked, np.clip(rng.normal(a.mu_hours * lvl, a.sd_hours), 0.5, 14.0), 0.0)
        overtime = worked & (rng.random(D) < duty["overtime_prob"])
        true_h[overtime] = np.minimum(true_h[overtime] + rng.uniform(3, 6, overtime.sum()), 20.0)

        L = np.clip(a.load_mean + _ar1(rng, D, 0.6, 0.08), 0.1, 0.95)
        since_maint = np.zeros(D)
        for t in range(1, D):
            since_maint[t] = 0 if maint[t] else since_maint[t - 1] + 1
        wear = 1 + 0.0004 * since_maint

        hours = np.where(worked, np.maximum(true_h + rng.normal(0, ben["hours_noise_sd"], D), 0), 0.0)
        meter = np.where(worked, np.maximum(true_h + rng.normal(0, ben["hours_noise_sd"], D), 0), 0.0)
        fuel = true_h * a.fuel_lph * (0.35 + 0.65 * L) * rng.lognormal(0, 0.05, D)
        idle = true_h * np.clip(rng.normal(0.25 - 0.2 * (L - 0.5), 0.05, D), 0.02, 0.6)
        rpm = np.where(worked, a.idle_rpm + (a.rated_rpm - a.idle_rpm) * L + rng.normal(0, 30, D), 0.0)
        temp = np.where(worked, a.base_temp + 0.15 * (amb - 15) + 6 * L + rng.normal(0, 1.0, D),
                        amb + rng.normal(0, 1.0, D))
        vib = np.where(worked, a.vib * (0.6 + 0.8 * L) * wear * rng.lognormal(0, 0.08, D),
                       np.abs(rng.normal(0, 0.02, D)))
        amb_rep = amb + rng.normal(0, 0.5, D)

        benign = np.array(["none"] * D, dtype=object)
        benign[maint] = "maintenance"
        benign[overtime] = "overtime"

        # stuck sensor (device fault, before transmission)
        stuck_ch = {"engine_temp_c": temp, "vibration_rms": vib, "ambient_c": amb_rep}
        t = 0
        while t < D:
            if rng.random() < ben["stuck_sensor_rate"]:
                dur = _ri(rng, ben["stuck_sensor_days"])
                arr = stuck_ch[list(stuck_ch)[int(rng.integers(3))]]
                arr[t:t + dur] = arr[t]
                benign[t:t + dur] = "stuck_sensor"
                t += dur
            else:
                t += 1

        # timestamps, sequence numbers, signatures
        offset_h = a.drift_min_per_day * (days % 30) / 60.0
        ts_sent = days * 24.0 + REPORT_HOUR + offset_h
        delayed = rng.random(D) < ben["delay_prob"]
        delay = np.where(delayed, _u(rng, ben["delay_hours"], D), rng.exponential(0.2, D))
        ts_recv = ts_sent + delay
        seq = 1000 + days
        sig = np.ones(D, int)

        # snapshot honest message content (attackers replay genuine past messages)
        honest = dict(hours=hours.copy(), meter=meter.copy(), fuel=fuel.copy(), rpm=rpm.copy(),
                      temp=temp.copy(), vib=vib.copy(), amb=amb_rep.copy(), idle=idle.copy())
        seq_out, ts_sent_out = seq.copy(), ts_sent.copy()
        modified = np.zeros(D, bool)
        fam_col = np.array(["none"] * D, dtype=object)

        # ---------------- attacks ----------------
        if a.compromised:
            fam, know = a.family, a.knowledge
            start = g["commissioning_days"]
            if shift.get("holdout_family") == fam:
                start = max(start, shift["start_day"])
            active, alpha = _schedule(fam, know, start, D, ac, rng)
            if shift.get("alpha_multiplier") is not None:
                alpha = np.where(days >= shift["start_day"], alpha * shift["alpha_multiplier"], alpha)
            types_other = [x for x in types if x != a.asset_type]
            for t in np.flatnonzero(active):
                al = alpha[t]
                if fam in ("inflation", "on_off", "gradual"):
                    h = hours[t]
                    if h > 0:
                        new = h * (1 + al)
                        if know != "naive":
                            new = min(new, 19.5)
                        r = new / h
                        hours[t] = new
                        if know != "naive":
                            meter[t] *= r
                            fuel[t] *= r
                            idle[t] *= r
                        modified[t] = True
                    elif (fam != "gradual" and know != "adaptive"
                          and rng.random() < ac["inflation"]["phantom_prob"]):
                        new = a.mu_hours * rng.uniform(0.6, 1.0)
                        hours[t] = new
                        if know != "naive":
                            meter[t] = new
                            fuel[t] = new * a.fuel_lph * (0.35 + 0.65 * a.load_mean)
                            idle[t] = 0.25 * new
                            rpm[t] = a.idle_rpm + (a.rated_rpm - a.idle_rpm) * a.load_mean
                        modified[t] = True
                elif fam == "replay":
                    p = ac["replay"]
                    if rng.random() >= p["daily_prob"]:
                        continue
                    lo, hi = max(0, t - p["lag_days"][1]), t - p["lag_days"][0]
                    if hi <= lo:
                        continue
                    cand = np.arange(lo, hi)
                    cand = cand[honest["hours"][cand] > hours[t] + 1.0]
                    if cand.size == 0:
                        continue
                    src = int(rng.choice(cand))
                    hours[t], meter[t], fuel[t] = honest["hours"][src], honest["meter"][src], honest["fuel"][src]
                    rpm[t], temp[t], vib[t] = honest["rpm"][src], honest["temp"][src], honest["vib"][src]
                    amb_rep[t], idle[t] = honest["amb"][src], honest["idle"][src]
                    if know == "naive":          # verbatim stale message: old seq and timestamp
                        seq_out[t], ts_sent_out[t] = seq[src], ts_sent[src]
                    elif know == "adaptive":     # re-sequenced, weather-corrected, jittered
                        new_amb = amb[t] + rng.normal(0, 0.5)
                        temp[t] += 0.15 * (new_amb - amb_rep[t])
                        amb_rep[t] = new_amb
                        j = 1 + rng.normal(0, 0.02)
                        hours[t] *= j
                        meter[t] *= j
                        fuel[t] *= j
                        idle[t] *= j
                    modified[t] = True
                elif fam == "forged_identity":
                    claim = max(hours[t], a.mu_hours) * (1 + al)
                    if know != "naive":
                        claim = min(claim, 19.5)
                    if know == "adaptive":
                        prof, Lf, ns = a, 0.6, 0.3
                        amb_f = amb[t] + rng.normal(0, 0.3)
                    else:
                        prof = g["asset_types"][types_other[int(rng.integers(len(types_other)))]]
                        prof = pd.Series(prof)
                        Lf, ns = 0.6, 1.0
                        amb_f = amb[t] + rng.choice([-1, 1]) * rng.uniform(3, 8)
                    if know == "naive":
                        sig[t] = 0
                    hours[t], meter[t] = claim, claim
                    fuel[t] = claim * prof.fuel_lph * (0.35 + 0.65 * Lf) * rng.lognormal(0, 0.05 * ns)
                    idle[t] = claim * 0.25 * (1 + rng.normal(0, 0.05 * ns))
                    rpm[t] = prof.idle_rpm + (prof.rated_rpm - prof.idle_rpm) * Lf + rng.normal(0, 30 * ns)
                    temp[t] = prof.base_temp + 0.15 * (amb_f - 15) + 6 * Lf + rng.normal(0, ns)
                    vib[t] = prof.vib * (0.6 + 0.8 * Lf) * rng.lognormal(0, 0.08 * ns)
                    amb_rep[t] = amb_f
                    modified[t] = True
                if modified[t]:
                    fam_col[t] = fam

        hours = np.round(hours, 1)
        meter = np.round(meter, 1)

        # transmission dropouts (after attacks)
        chans = dict(fuel_l=fuel, avg_rpm=rpm, engine_temp_c=temp, vibration_rms=vib,
                     ambient_c=amb_rep, idle_hours=idle)
        any_drop = np.zeros(D, bool)
        drop_masks = {}
        for name in DROPOUT_CHANNELS:
            m = rng.random(D) < ben["dropout_prob"]
            chans[name] = np.where(m, np.nan, chans[name])
            drop_masks[name] = m
            any_drop |= m
        # Honest (pre-attack) message content, with the same transmission dropouts. Evaluator-only:
        # adaptive adversaries start from it. Draws no random numbers, so worlds are unchanged.
        h_src = dict(fuel_l="fuel", avg_rpm="rpm", engine_temp_c="temp", vibration_rms="vib",
                     ambient_c="amb", idle_hours="idle")
        honest_cols = {f"h_{n}": np.where(drop_masks[n], np.nan, honest[h_src[n]]) for n in DROPOUT_CHANNELS}
        honest_cols.update(h_reported_hours=np.round(honest["hours"], 1), h_meter_delta=np.round(honest["meter"], 1),
                           h_seq=seq, h_ts_sent_h=ts_sent, h_signature_valid=np.ones(D, int))
        benign[any_drop & (benign == "none")] = "dropout"
        benign[delayed & (benign != "stuck_sensor")] = "delay"

        ref_h = np.round(true_h, 2)
        over_h = hours - ref_h
        fraud = over_h > con["fraud_tolerance_hours"]
        frames.append(pd.DataFrame(dict(
            asset_id=i, device_id=a.device_id, site=s, day=days, seq=seq_out,
            ts_sent_h=ts_sent_out, ts_recv_h=ts_recv, delay_h=ts_recv - ts_sent_out,
            signature_valid=sig, contract_active=1,
            reported_hours=hours, meter_delta=meter, rate=a.rate, claimed_amount=np.round(hours * a.rate, 2),
            **chans,
            reference_hours=ref_h, reference_amount=np.round(ref_h * a.rate, 2),
            overclaim_amount=np.where(fraud, np.round(over_h * a.rate, 2), 0.0),
            anomaly_label=modified.astype(int), fraud_label=fraud.astype(int),
            attack_family=fam_col, knowledge=np.where(modified, a.knowledge, "none"),
            benign_event=benign, compromised=bool(a.compromised),
            **honest_cols,
        )))
    records = pd.concat(frames, ignore_index=True)
    return records, assets
