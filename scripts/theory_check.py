"""Empirical checks of the theory in docs/theory.md (development seeds 310-319 only).

  A  renewal model (Monte Carlo, no simulator): measured extraction vs the Theorem 2 bound
  B  simulator, deterrence sweep: payer loss per unit of attempted over-claim vs 1 - q p_d (1 + lambda)
     (Theorem 1), uniform audits without quarantine or aggregate test, plus a quarantine variant (Theorem 2)
  C  simulator, salami offsets: verified claims needed before the aggregate test fires vs Proposition 3,
     and false flags on honest devices

    python scripts/theory_check.py --part A
    python scripts/theory_check.py --part BC --seeds 310 --out results/theory_dev
    python scripts/theory_check.py --part BC --role confirmatory --seeds 400,401,402,403,404 --out results/theory
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.adaptive import ConstantInflation, RiskAwareWhiteBox, SalamiOffset, World, summarise  # noqa: E402
from avert.config import load_config  # noqa: E402


def theorem2_bound(q, p, lam, K, vbar=1.0):
    qp = q * p
    return vbar * max(0.0, (1 - qp) - lam * qp) / (1 + K * qp)


def part_a(out: Path, n_claims=200_000, seed=0):
    """Renewal model: the attacker over-claims vbar on every claim outside quarantine (rho = 1) or on a random
    fraction rho of them; quarantine lasts K claims, all verified (the attacker is honest there)."""
    rng = np.random.default_rng(seed)
    rows = []
    p = 0.95
    for q in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        for lam in (0.0, 1.0, 3.0):
            for K in (0, 10, 30):
                for rho in (1.0, 0.5, 0.2):
                    u_v = rng.random(n_claims)
                    u_f = rng.random(n_claims)
                    gain, quarantine = 0.0, 0
                    for t in range(n_claims):
                        if quarantine:
                            quarantine -= 1
                            continue
                        if u_f[t] >= rho:
                            continue
                        if u_v[t] < q * p:              # verified and established
                            gain -= lam
                            quarantine = K
                        else:
                            gain += 1.0
                    g = gain / n_claims
                    b = theorem2_bound(q, p, lam, K)
                    rows.append(dict(q=q, lam=lam, K=K, rho=rho, measured=g, bound=b, holds=g <= b + 4 / np.sqrt(n_claims)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "theorem2_montecarlo.csv", index=False)
    tight = df[df.rho == 1.0]
    print(f"[A] bound holds in {df.holds.mean():.0%} of {len(df)} settings; "
          f"max |measured - bound| for rho=1 (always over-claim, where the bound is attained when positive): "
          f"{np.abs(tight.measured.clip(lower=0) - tight.bound).max():.4f}")
    return df


def uniform(cfg, q, lam, slash=None):
    vw = cfg["trust"]["adaptive"]["verify_weight"]
    return dict(cfg["avert"], q_floor=q, kappa_q=0.0, q_max=1.0, penalty=lam, slash=vw if slash is None else slash,
                aggregate=False, conformal=False)


def quarantine_lengths(log, comp, low):
    """Claims spent below the low-trust gate in each completed quarantine of a compromised device."""
    out = []
    for i in comp:
        below = log["T"][i] < low
        cl = log["is_claim"][i]
        t = 1
        while t < below.size:
            if below[t] and not below[t - 1]:
                e = t
                while e < below.size and below[e]:
                    e += 1
                if e < below.size:
                    out.append(int(cl[t:e].sum()))
                t = e
            t += 1
    return np.asarray(out)


def fleet_bound(w: World, log, q, p, lam, K, tau):
    """Theorem 2 summed over compromised devices: n_i (r_i tau + a_i) + K a_i with v_bar_i the largest
    over-claim value device i submitted (profit bound with lam; payer-loss bound with lam = 0)."""
    cl = log["is_claim"][w.comp]
    over = np.clip((log["claimed"] - log["true_amt"])[w.comp], 0, None) * cl
    n = cl.sum(1)
    vbar = over.max(1)
    qp = q * p
    term = max(max(0.0, 1 - qp * (1 + lam)) / (1 + K * qp), max(0.0, 1 - p * (1 + lam)) / p)
    a = vbar * term
    return float((n * (w.rate[w.comp] * tau + a) + K * a).sum())


def part_b(w: World, cfg, seed, rows):
    p = cfg["authorisation"]["verification"]["p_detect"]
    for slash, label in ((None, "no quarantine"), (cfg["avert"]["slash"], "quarantine")):
        lams = (0.0, 1.0, 3.0) if slash is None else (0.0,)
        for lam in lams:
            for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
                name = f"B q={q} lam={lam} {label}"
                w.add_defender(name, uniform(cfg, q, lam, slash))
                for aname, att in (("constant 0.20", ConstantInflation(w.comp, 0.20)),
                                   ("K3* risk-aware", RiskAwareWhiteBox(w.comp))):
                    t0 = time.time()
                    log = w.run_test(copy.deepcopy(att), name)
                    r = summarise(w, log)
                    ql = quarantine_lengths(log, w.comp, cfg["trust"]["low"])
                    r["quarantine_n"] = int(ql.size)
                    r["quarantine_median_claims"] = float(np.median(ql)) if ql.size else np.nan
                    r["quarantine_min_claims"] = float(ql.min()) if ql.size else np.nan
                    ratio = r["payer_net_loss"] / max(r["attempted_overclaim"], 1e-9)
                    K = 0 if slash is None or not ql.size else int(ql.min())
                    tau = cfg["generator"]["contract"]["fraud_tolerance_hours"]
                    r["K_used"] = K
                    r["bound_payer_loss"] = fleet_bound(w, log, q, p, 0.0, K, tau)
                    r["bound_profit"] = fleet_bound(w, log, q, p, lam, K, tau)
                    rows.append(dict(part="B", seed=seed, variant=label, q=q, lam=lam, attacker=aname,
                                     theory_gain_ratio=1 - q * p * (1 + lam), measured_gain_ratio=ratio, **r))
                    print(f"  [B s{seed}] {label:13s} q={q:.1f} lam={lam:.0f} {aname:15s} payer_loss/attempted={ratio:+.3f}"
                          f" loss={r['payer_net_loss']:.0f}<=bound {r['bound_payer_loss']:.0f}; profit={r['attacker_profit']:.0f}<=bound {r['bound_profit']:.0f} K={K}"
                          f" theory<={1 - q * p * (1 + lam):+.3f} payer_loss={r['payer_net_loss']:9.0f} ({time.time() - t0:.0f}s)",
                          flush=True)
                del w._defenders[name]


def part_c(w: World, cfg, seed, rows):
    from scipy.stats import norm
    av = cfg["avert"]
    w.add_defender("C AVERT", dict(av))
    z = av["agg_z"]
    for hours in (0.10, 0.20, 0.30, 0.40):
        log = w.run_test(SalamiOffset(w.comp, hours), "C AVERT")
        r = summarise(w, log)
        ver = log["verified"] & log["is_claim"]
        n_before = []
        for i in w.comp:
            f = np.flatnonzero(log["flagged"][i])
            n_before.append(int(ver[i, :f[0] + 1].sum()) if f.size else np.inf)
        n_before = np.asarray(n_before, float)
        # Proposition 3 with power 1/2 (median): m = (z sigma / delta)^2; with power 0.95 add z_{0.95}
        m50 = (z * w.sigma_e / hours) ** 2
        m95 = ((z + norm.ppf(0.95)) * w.sigma_e / hours) ** 2
        det = np.isfinite(n_before)
        rows.append(dict(part="C", seed=seed, attacker=f"salami +{hours:.2f} h", sigma_e=w.sigma_e,
                         prop3_m_median=max(m50, av["agg_min"]), prop3_m_95=max(m95, av["agg_min"]),
                         detected_share=det.mean(), median_verified_before_flag=float(np.median(n_before)) if det.any() else np.inf,
                         p95_verified_before_flag=float(np.quantile(n_before[det], 0.95)) if det.any() else np.inf, **r))
        print(f"  [C s{seed}] salami +{hours:.2f} h: detected {det.mean():.0%} of compromised devices, median verified claims"
              f" before flag {rows[-1]['median_verified_before_flag']:.1f} (Prop 3: {rows[-1]['prop3_m_median']:.1f} median,"
              f" {rows[-1]['prop3_m_95']:.1f} for 95%); honest devices flagged {r['honest_devices_flagged']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ABC")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default="310")
    ap.add_argument("--level", type=float, default=0.20)
    ap.add_argument("--out", default="results/theory")
    ap.add_argument("--role", default="dev", choices=("dev", "confirmatory"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "A" in args.part:
        part_a(out)
    if not set("BC") & set(args.part):
        return
    cfg = load_config(args.config)
    seeds = list(map(int, args.seeds.split(",")))
    if args.role == "dev":
        assert all(310 <= s <= 319 for s in seeds) or "quick" in args.config, "development seeds are 310-319"
    else:
        assert all(400 <= s <= 404 for s in seeds), "confirmatory theory checks use seeds 400-404"
    rows = []
    for seed in seeds:
        g = copy.deepcopy(cfg["generator"])
        g["compromise_fraction"] = args.level
        w = World(cfg, g, seed)
        w.warm_up(verify=False)
        if "C" in args.part:
            part_c(w, cfg, seed, rows)
            pd.DataFrame(rows).to_csv(out / "theory_sim.csv", index=False)
        if "B" in args.part:
            part_b(w, cfg, seed, rows)
            pd.DataFrame(rows).to_csv(out / "theory_sim.csv", index=False)


if __name__ == "__main__":
    main()
