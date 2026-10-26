"""Learning adversary (docs/confirmatory_addendum_rl.md).

Train (development seeds only), one Q-table per target defence:
    python scripts/rl_attacker.py train --defence AVERT --seeds 310,311 --episodes 300 --out results/rl
    python scripts/rl_attacker.py train --defence DTA   --seeds 310,311 --episodes 300 --out results/rl
Evaluate the frozen policies on fresh fleets (seeds 430-439):
    python scripts/rl_attacker.py eval --seeds 430 --out results/rl
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
from avert.adaptive import AIMDFeedback, RiskAwareWhiteBox, WhiteBoxGreedy, World, summarise  # noqa: E402
from avert.config import load_config  # noqa: E402
from avert.rl import QLearningAttacker, new_table  # noqa: E402
from scripts.dev_defence import parse_seeds  # noqa: E402
from scripts.theory_check import fleet_bound, quarantine_lengths  # noqa: E402


def build(cfg, seed, level=0.20):
    g = copy.deepcopy(cfg["generator"])
    g["compromise_fraction"] = level
    w = World(cfg, g, seed)
    w.warm_up(verify=False)
    w.add_defender("AVERT", dict(cfg["avert"]))
    return w


def train(args, cfg):
    seeds = parse_seeds(args.seeds)
    assert all(310 <= s <= 319 for s in seeds), "training uses development seeds 310-319 only"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worlds = [build(cfg, s) for s in seeds]
    Q = new_table()
    rows = []
    n = args.episodes
    for ep in range(n):
        w = worlds[ep % len(worlds)]
        eps = max(0.02, 0.30 * (1 - ep / (0.8 * n)))
        att = QLearningAttacker(w.comp, Q, eps=eps, lr=0.1, gamma=0.95, train=True, seed=ep)
        t0 = time.time()
        log = w.run_test(att, args.defence, rng_seed=100_000 + ep)
        att.end_episode()
        r = summarise(w, log)
        rows.append(dict(episode=ep, seed=w.seed, eps=eps, payer_net_loss=r["payer_net_loss"],
                         attacker_profit=r["attacker_profit"], extracted_value=r["extracted_value"]))
        print(f"  [{args.defence} ep {ep:3d} s{w.seed} eps={eps:.2f}] profit={r['attacker_profit']:10.0f} "
              f"payer_loss={r['payer_net_loss']:9.0f} ({time.time() - t0:.0f}s)", flush=True)
        if (ep + 1) % 25 == 0 or ep == n - 1:
            np.save(out / f"q_{args.defence}.npy", Q)
            pd.DataFrame(rows).to_csv(out / f"train_{args.defence}.csv", index=False)


def evaluate(args, cfg):
    seeds = parse_seeds(args.seeds)
    assert all(430 <= s <= 439 for s in seeds), "evaluation uses the fresh seeds 430-439 (addendum)"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Qs = {d: np.load(out / f"q_{d}.npy") for d in ("AVERT", "DTA")}
    p = cfg["authorisation"]["verification"]["p_detect"]
    tau = cfg["generator"]["contract"]["fraud_tolerance_hours"]
    rows = []
    for seed in seeds:
        w = build(cfg, seed)
        for dname in ("DTA", "AVERT"):
            lib = [("RL (Q-learning)", QLearningAttacker(w.comp, Qs[dname].copy(), train=False)),
                   ("K1 feedback", AIMDFeedback(w.comp)), ("K3 white-box", WhiteBoxGreedy(w.comp)),
                   ("K3* risk-aware", RiskAwareWhiteBox(w.comp)), ("K0 oblivious", None)]
            for aname, att in lib:
                t0 = time.time()
                log = w.run_test(copy.deepcopy(att), dname)
                r = summarise(w, log)
                ql = quarantine_lengths(log, w.comp, cfg["trust"]["low"])
                K = int(ql.min()) if ql.size else 0
                q = cfg["avert"]["q_floor"] if dname == "AVERT" else 0.0
                r.update(K_used=K, quarantine_median_claims=float(np.median(ql)) if ql.size else np.nan,
                         bound_payer_loss=fleet_bound(w, log, q, p, 0.0, K, tau))
                rows.append(dict(seed=seed, defence=dname, attacker=aname, **r))
                print(f"  [s{seed}] {dname:5s} {aname:16s} payer_loss={r['payer_net_loss']:9.0f} "
                      f"extracted={r['extracted_value']:9.0f} bound={r['bound_payer_loss']:9.0f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / f"eval_{'_'.join(map(str, seeds))}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "eval"))
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default="310,311")
    ap.add_argument("--defence", default="AVERT", choices=("AVERT", "DTA"))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="results/rl")
    args = ap.parse_args()
    cfg = load_config(args.config)
    (train if args.mode == "train" else evaluate)(args, cfg)


if __name__ == "__main__":
    main()
