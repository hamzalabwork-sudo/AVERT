"""Real-data case study (RQ4): SGCC electricity theft by under-reporting (avert/sgcc.py).

    python scripts/sgcc_case.py --seeds 310,311 --out results/sgcc_dev           # development
    python scripts/sgcc_case.py --seeds 400-419 --out results/sgcc --confirmatory

The raw file data/sgcc_raw/data.csv comes from https://github.com/henryRDlab/ElectricityTheftDetection
(three split archives; see README "Data").
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.adaptive import summarise  # noqa: E402
from avert.config import load_config  # noqa: E402
from avert.sgcc import SGCCCase, library, load_sgcc  # noqa: E402
from scripts.dev_defence import apply_sets, defences  # noqa: E402


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/sgcc_raw/data.csv")
    ap.add_argument("--seeds", default="310")
    ap.add_argument("--out", default="results/sgcc_dev")
    ap.add_argument("--which", default="main", choices=("all", "main"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="sgcc")
    ap.add_argument("--confirmatory", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_sets(cfg, args.set)
    seeds = parse_seeds(args.seeds)
    if args.confirmatory:
        assert all(400 <= s <= 419 for s in seeds), "confirmatory seeds are 400-419"
    else:
        assert all(310 <= s <= 319 for s in seeds), "development seeds are 310-319"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Y = load_sgcc(args.data)
    print(f"SGCC: {Y.shape[0]} normal customers with complete records, {Y.shape[1]} days", flush=True)
    rows = []
    for seed in seeds:
        case = SGCCCase(Y, cfg, seed)
        defs = defences(case.cfg, args.which)
        case.warm_up(defs)
        for dname, _ in defs:
            for aname, att in library(case.comp, case.tol):
                t0 = time.time()
                r = dict(tag=args.tag, seed=seed, defence=dname, attacker=aname, **summarise(case, case.run_test(att, dname)))
                rows.append(r)
                print(f"  [s{seed}] {dname:42s} {aname:16s} payer_loss={r['payer_net_loss']:8.0f} extracted={r['extracted_value']:8.0f}"
                      f" fines={r['fine_income']:8.0f} verif={r['verification_rate']:.3f} honest_verif={r['honest_verification_rate']:.3f}"
                      f" hold={r['honest_legit_hold_rate']:.3f} honest_flags={r['honest_devices_flagged']} ({time.time() - t0:.0f}s)",
                      flush=True)
        pd.DataFrame(rows).to_csv(out / f"{args.tag}.csv", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for v in ("payer_net_loss", "extracted_value"):
        print(f"\nmean {v} by attacker:\n",
              df.pivot_table(index="attacker", columns="defence", values=v, aggfunc="mean", sort=False).round(0).to_string())
    worst = [df.groupby(["seed", "defence"])[v].max().groupby("defence").mean().rename("worst_" + v)
             for v in ("payer_net_loss", "extracted_value")]
    ops = df.groupby("defence")[["honest_verification_rate", "honest_legit_hold_rate", "honest_devices_flagged"]].mean()
    print("\nworst case over the library (mean over seeds):\n", pd.concat([*worst, ops], axis=1).round(3).to_string())


if __name__ == "__main__":
    main()
