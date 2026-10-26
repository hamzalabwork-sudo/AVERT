"""Pilot (RQ1): how much value do adaptive attackers extract from deterministic trust-aware authorisation?

    python scripts/pilot_adaptive.py --seeds 300,301,302 --level 0.20 --out results/pilot

Pilot seeds only (300-302, docs/research_plan.md). For each fleet: train the DTA defender, verify that the
streaming pipeline reproduces the batch pipeline, then let each attacker control the compromised devices
during the whole test period.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.adaptive import AIMDFeedback, ConstantInflation, WhiteBoxGreedy, World, summarise  # noqa: E402
from avert.config import load_config  # noqa: E402

CONST = (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default="300,301,302")
    ap.add_argument("--level", type=float, default=0.20)
    ap.add_argument("--out", default="results/pilot")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in map(int, args.seeds.split(",")):
        g = copy.deepcopy(cfg["generator"])
        g["compromise_fraction"] = args.level
        w = World(cfg, g, seed)
        mf, ms = w.warm_up(verify=True)
        k0 = w.run_test(None)
        # verification against the batch pipeline (fraudulent value settled on the test period)
        fraud = k0["fraud"]
        fvs_online = float(((k0["paid"] - k0["true_amt"]) * fraud).clip(min=0).sum())
        check = dict(seed=seed, max_feature_diff=mf, max_score_diff=ms, fvs_online=fvs_online,
                     fvs_batch=w.batch_test["fvs"])
        print("  verification:", check, flush=True)
        (out / f"verify_s{seed}.json").write_text(json.dumps(check, indent=2))

        runs = [("K0 oblivious (fixed attack families)", None)]
        runs += [(f"oracle constant alpha={a:.2f}", ConstantInflation(w.comp, a)) for a in CONST]
        runs += [("K1 feedback (AIMD)", AIMDFeedback(w.comp)), ("K3 white-box greedy", WhiteBoxGreedy(w.comp))]
        for name, att in runs:
            t0 = time.time()
            log = k0 if att is None else w.run_test(att)
            r = dict(seed=seed, attacker=name, n_compromised=len(w.comp), **summarise(w, log))
            rows.append(r)
            print(f"  [s{seed}] {name:34s} extracted={r['extracted_value']:10.0f}  attempted={r['attempted_overclaim']:10.0f}"
                  f"  rate={r['extraction_rate']:.2f}  auto-approved={r['comp_auto_approved']:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(out / "pilot.csv", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.groupby("attacker", sort=False)[["extracted_value", "attempted_overclaim", "extraction_rate",
                                              "comp_auto_approved", "share_overclaim_below_tolerance",
                                              "honest_legit_hold_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
