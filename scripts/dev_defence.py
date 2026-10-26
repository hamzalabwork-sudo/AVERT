"""Attacker library vs defences, worst-case payer loss.

Development (seeds 310-319):
    python scripts/dev_defence.py --seeds 310,311 --out results/dev
    python scripts/dev_defence.py --seeds 310 --set avert.q_floor=0.1 --set avert.penalty=2 --tag qf0.1_l2 --which main
Confirmatory (docs/confirmatory_plan.md; seeds 400-419 in distribution, 420-424 under each shift):
    python scripts/dev_defence.py --role confirmatory --seeds 400-419 --out results/confirmatory --tag main
    python scripts/dev_defence.py --role confirmatory --seeds 420-424 --shift S2-noisy-meters --which shift

Defences: DTA (deterministic), DTA + uniform audits at several budgets (no penalty, no aggregate test),
AVERT (configs avert block) and, with --which all, one-mechanism ablations of AVERT.
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.adaptive import (AIMDFeedback, ConstantInflation, OnOff, RiskAwareWhiteBox, SalamiOffset,  # noqa: E402
                            WhiteBoxGreedy, World, summarise)
from avert.config import load_config  # noqa: E402


def library(comp):
    lib = [("K0 oblivious", None)]
    lib += [(f"constant {a:.2f}", ConstantInflation(comp, a)) for a in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35)]
    lib += [("salami +0.40 h", SalamiOffset(comp, 0.40)), ("on-off 0.30 5/15", OnOff(comp)),
            ("K1 feedback", AIMDFeedback(comp)), ("K3 white-box", WhiteBoxGreedy(comp)),
            ("K3* risk-aware", RiskAwareWhiteBox(comp))]
    return lib


# Pre-registered benign distribution shifts, active from the first test day (day 584).
SHIFTS = {
    "S1-usage-surge": {"benign_shift": dict(start_day=584, hours_multiplier=1.25)},
    "S2-noisy-meters": {"benign_shift": dict(start_day=584, noise_multiplier=3.0)},
    "S3-more-compromised": {"compromise_fraction": 0.40},
    "S4-unreliable-links": {"benign_shift": dict(start_day=584, delay_prob=0.15, dropout_prob=0.05)},
}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def defences(cfg, which="all"):
    """Defence variants. Uniform audits isolate randomisation; ablations remove one AVERT mechanism each."""
    av = dict(cfg["avert"])
    vw = cfg["trust"]["adaptive"]["verify_weight"]
    base = dict(av)

    def uni(q, penalty=0.0, slash=vw):
        return dict(base, q_floor=q, kappa_q=0.0, penalty=penalty, slash=slash, aggregate=False, conformal=False)

    if which == "grid":                      # development tuning of audit weighting and quarantine strength
        return [("DTA", None), ("uniform audits q=0.100 + penalty/slash", uni(0.10, av["penalty"], av["slash"]))] + [
            (f"AVERT kappa={kq} slash={sl:.0f}", dict(base, kappa_q=kq, slash=sl)) for kq in (0.5, 1.0, 2.0) for sl in (50.0, 100.0)]
    if which == "shift":
        return [("DTA", None), ("uniform audits q=0.075 + penalty/slash", uni(0.075, av["penalty"], av["slash"])),
                ("AVERT", base)]
    out = [("DTA", None)]
    out += [(f"uniform audits q={q:.2f}", uni(q)) for q in (0.05, 0.10, 0.15)]
    out += [(f"uniform audits q={q:.3f} + penalty/slash", uni(q, av["penalty"], av["slash"])) for q in (0.05, 0.075, 0.10)]
    out += [("AVERT", base)]
    if which == "all":
        out += [("AVERT -aggregate", dict(base, aggregate=False)),
                ("AVERT -penalty", dict(base, penalty=0.0)),
                ("AVERT -slash", dict(base, slash=vw)),
                ("AVERT -value weighting", dict(base, kappa_q=0.0)),
                ("AVERT -conformal", dict(base, conformal=False))]
    return out


def apply_sets(cfg, sets):
    for kv in sets:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default="310,311")
    ap.add_argument("--level", type=float, default=0.20)
    ap.add_argument("--out", default="results/dev")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--which", default="all", choices=("all", "main", "shift", "grid"))
    ap.add_argument("--role", default="dev", choices=("dev", "confirmatory"))
    ap.add_argument("--shift", default=None, choices=sorted(SHIFTS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_sets(cfg, args.set)
    seeds = parse_seeds(args.seeds)
    if args.role == "dev":
        assert all(310 <= s <= 319 for s in seeds) or "quick" in args.config, "development seeds are 310-319"
    else:
        assert not args.set, "the confirmatory configuration is frozen"
        lo, hi = (420, 424) if args.shift else (400, 419)
        assert all(lo <= s <= hi for s in seeds), f"confirmatory seeds for this scenario are {lo}-{hi}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        g = copy.deepcopy(cfg["generator"])
        g["compromise_fraction"] = args.level
        if args.shift:
            g.update(copy.deepcopy(SHIFTS[args.shift]))
        w = World(cfg, g, seed)
        w.warm_up(verify=False)
        for dname, params in defences(cfg, args.which):
            if params is not None:
                w.add_defender(dname, params)
            for aname, att in library(w.comp):
                t0 = time.time()
                r = dict(tag=args.tag, shift=args.shift or "none", seed=seed, defence=dname, attacker=aname,
                         **summarise(w, w.run_test(copy.deepcopy(att), dname)))
                rows.append(r)
                print(f"  [s{seed}] {dname:42s} {aname:18s} payer_loss={r['payer_net_loss']:9.0f} profit={r['attacker_profit']:10.0f} extracted={r['extracted_value']:9.0f}"
                      f" recovered={r['recovered_value']:9.0f} verif={r['verification_rate']:.3f}"
                      f" hold={r['honest_legit_hold_rate']:.3f} honest_flags={r['honest_devices_flagged']} ({time.time() - t0:.0f}s)",
                      flush=True)
        pd.DataFrame(rows).to_csv(out / f"dev_{args.tag}.csv", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    metrics = ("payer_net_loss", "extracted_value", "attacker_profit", "fine_income")
    for v in metrics:
        piv = df.pivot_table(index="attacker", columns="defence", values=v, aggfunc="mean", sort=False)
        print(f"\nmean {v} by attacker:\n", piv.round(0).to_string())
    worst = [df.groupby(["seed", "defence"])[v].max().groupby("defence").mean().rename("worst_" + v) for v in metrics]
    ops = df.groupby("defence")[["honest_verification_rate", "verification_rate", "honest_legit_hold_rate",
                                 "honest_devices_flagged", "honest_value_clawed"]].mean()
    print("\nworst case over the library (mean over seeds):\n", pd.concat([*worst, ops], axis=1).round(3).to_string())


if __name__ == "__main__":
    main()
